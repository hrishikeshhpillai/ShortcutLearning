import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from poolforge.data import CIFAR_MEAN, CIFAR_STD, train_batch
from poolforge.methods.lff import LfF


class EditGenerator(nn.Module):
    def __init__(self, eps, num_classes=0, ops=0):
        super().__init__()
        self.eps, self.ops = eps, ops
        if ops:
            self.kernels = nn.Parameter(0.01 * torch.randn(ops, 3, 3, 5, 5))
            self.op_head = nn.Linear(32, ops + 1)
            nn.init.zeros_(self.op_head.weight)
            nn.init.zeros_(self.op_head.bias)
        self.enc = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU())
        self.cond = nn.Linear(num_classes, 32) if num_classes else None
        self.affine = nn.Linear(32, 12)
        self.residual = nn.Conv2d(32, 3, 1)
        for layer in (self.affine, self.residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, target=None):
        h = self.enc(x)
        if self.cond is not None:
            h = F.relu(h + self.cond(target)[:, :, None, None])
        p = self.affine(h.mean((2, 3)))
        a = torch.eye(3, device=x.device) + 0.5 * torch.tanh(p[:, :9]).view(-1, 3, 3)
        r = F.interpolate(self.residual(F.adaptive_avg_pool2d(h, 8)), size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.einsum("bij,bjhw->bihw", a, x) + torch.tanh(p[:, 9:])[:, :, None, None] + self.eps * torch.tanh(r)
        if self.ops:
            q = torch.tanh(self.op_head(h.mean((2, 3))))
            filtered = torch.stack([F.conv2d(x, k, padding=2) for k in self.kernels], 1)
            out = out + (q[:, :self.ops, None, None, None] * filtered).sum(1)
            out = out + F.softplus(q[:, self.ops:])[:, :, None, None] * torch.randn_like(x)
        return out


class PoolForge(LfF):
    def __init__(self, ctx):
        super().__init__(ctx)
        args = ctx.args
        self.pool_count = torch.zeros(ctx.n_train, device=ctx.device)
        self.pool_fraction, self.fool_log = float("nan"), []
        if args.gen in ("full", "color"):
            self.generator = EditGenerator(args.gen_eps, 0 if args.gen_untargeted else ctx.num_classes,
                                           args.gen_ops if args.gen == "full" else 0).to(ctx.device)
            self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=ctx.cfg["lr"])

    def refresh_pool(self, step):
        args, y = self.ctx.args, self.ctx.train["y"]
        loss = self.ema_b.parameter.masked_fill(self.ema_b.updated == 0, -1)
        top = torch.full((self.ctx.num_classes,), args.pool_top, device=y.device)
        if args.pool_adaptive and step >= args.pool_adaptive_from * self.ctx.cfg["num_steps"]:
            unsure = (loss > math.log(2)).float()
            for c in range(self.ctx.num_classes):
                top[c] = unsure[y == c].mean().clamp(args.pool_top, 0.2)
            self.pool_fraction = top.mean().item()
        rank = torch.zeros(len(y), device=y.device)
        for c in range(self.ctx.num_classes):
            members = (y == c).nonzero().squeeze(1)
            rank[members] = loss[members].argsort().argsort().float() / len(members)
        self.pool_count = args.pool_decay * self.pool_count + (rank >= 1 - top[y]).float()
        score = self.pool_count + rank
        self.pools = []
        for c in range(self.ctx.num_classes):
            members = (y == c).nonzero().squeeze(1)
            m = max(2, int(top[c] * len(members)))
            self.pools.append(members[score[members].argsort(descending=True)[:m]])
        self.conflict_mask = torch.zeros(len(y), dtype=torch.bool, device=y.device)
        self.conflict_mask[torch.cat(self.pools)] = True

    def pool_sample(self, n, cls=None):
        device = self.ctx.device
        if cls is None:
            cls = torch.randint(self.ctx.num_classes, (n,), device=device)
        idx = torch.stack([self.pools[c][torch.randint(len(self.pools[c]), (1,), device=device)] for c in cls.tolist()]).squeeze(1)
        return cls, idx

    def source(self, x, y, where):
        if where == "pool":
            ys, idx = self.pool_sample(self.ctx.args.gen_batch)
            return train_batch(self.ctx.benchmark, self.ctx.train, idx), ys
        sub = torch.randperm(len(x), device=x.device)[:self.ctx.args.gen_batch]
        return x[sub], y[sub]

    def mix_loss(self):
        n = self.ctx.args.mix_batch
        cls, a = self.pool_sample(n)
        b = self.pool_sample(n, cls)[1]
        xa = train_batch(self.ctx.benchmark, self.ctx.train, a)
        if self.ctx.args.mix == "repeat":
            return F.cross_entropy(self.model(xa), cls)
        xb = train_batch(self.ctx.benchmark, self.ctx.train, b)
        lam = torch.rand(n, 1, device=self.ctx.device).view(n, 1, 1, 1)
        return F.cross_entropy(self.model(lam * xa + (1 - lam) * xb), cls)

    def gen_loss(self, x, y):
        args, c = self.ctx.args, self.ctx.num_classes
        xs, ys = self.source(x, y, args.gen_src)
        mode = self.model_b.training
        self.model_b.eval()
        if args.gen_untargeted:
            xg = self.generator(xs)
            p_b = F.softmax(self.model_b(xg), dim=1).gather(1, ys[:, None]).squeeze(1)
            fool = -torch.log((1 - p_b).clamp(min=1e-6)).mean()
        else:
            k = (ys + torch.randint(1, c, ys.shape, device=ys.device)) % c
            xg = self.generator(xs, F.one_hot(k, c).float())
            logit_g = self.model_b(xg)
            fool = F.cross_entropy(logit_g, k)
            p_b = 1 - (logit_g.argmax(1) == k).float()
        self.model_b.train(mode)
        loss_g = fool + args.gen_reg * (xg - xs).pow(2).mean()
        grads = torch.autograd.grad(loss_g, list(self.generator.parameters()), retain_graph=True)
        self.fool_log.append((p_b < 0.5).float().mean().item())
        return F.cross_entropy(self.model(xg), ys), grads

    def pgd_eps(self):
        return self.ctx.args.pgd_eps * (1.0 if self.ctx.benchmark.startswith("cmnist") else 4.0)

    def pgd_loss(self, x, y):
        args = self.ctx.args
        xs, ys = self.source(x, y, args.pgd_src)
        eps = self.pgd_eps()
        modes = self.model_b.training, self.model.training
        self.model_b.eval()
        self.model.eval()
        delta = torch.zeros_like(xs, requires_grad=True)
        for _ in range(args.pgd_steps):
            obj = F.cross_entropy(self.model_b(xs + delta), ys) - F.cross_entropy(self.model(xs + delta), ys)
            g, = torch.autograd.grad(obj, delta)
            delta = (delta + eps / 4 * g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
        with torch.no_grad():
            self.fool_log.append((self.model_b(xs + delta).argmax(1) != ys).float().mean().item())
        self.model_b.train(modes[0])
        self.model.train(modes[1])
        return F.cross_entropy(self.model(xs + delta.detach()), ys)

    def step(self, x, y, index, step):
        args = self.ctx.args
        loss, logit_d = self.reweighted_loss(x, y, index, uniform=args.no_reweight)
        gen_grads = None
        if step >= args.start:
            if step % 1000 == 0 or not hasattr(self, "pools"):
                self.refresh_pool(step)
            if args.mix != "none":
                loss = loss + args.mix_weight * self.mix_loss()
            if args.gen in ("full", "color"):
                gen, gen_grads = self.gen_loss(x, y)
                loss = loss + args.gen_weight * gen
            if args.pgd_weight > 0:
                loss = loss + args.pgd_weight * self.pgd_loss(x, y)
        self.optimizer_b.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer_b.step()
        self.optimizer.step()
        if gen_grads is not None:
            for p, g in zip(self.generator.parameters(), gen_grads):
                p.grad = g
            self.optimizer_g.step()
        return logit_d

    def to_image(self, x):
        if not self.ctx.benchmark.startswith("cmnist"):
            x = x * torch.tensor(CIFAR_STD, device=x.device)[:, None, None] + torch.tensor(CIFAR_MEAN, device=x.device)[:, None, None]
        return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()

    @torch.no_grad()
    def dump_case(self, path, n):
        args, train, c = self.ctx.args, self.ctx.train, self.ctx.num_classes
        torch.manual_seed(0)
        ys, idx = self.pool_sample(n)
        partner = self.pool_sample(n, ys)[1]
        xs, xp = train_batch(self.ctx.benchmark, train, idx), train_batch(self.ctx.benchmark, train, partner)
        k = (ys + torch.randint(1, c, ys.shape, device=ys.device)) % c
        modes = self.model_b.training, self.model.training
        self.model_b.eval()
        self.model.eval()
        out = dict(y=ys, k=k, bias=train["bias"][idx], partner_bias=train["bias"][partner],
                   aligned_bias=train["aligned_bias"], pool_size=self.conflict_mask.sum().item())
        images = dict(x=xs, mix=0.5 * xs + 0.5 * xp)
        if args.gen in ("full", "color"):
            images["gen"] = self.generator(xs) if args.gen_untargeted else self.generator(xs, F.one_hot(k, c).float())
        if args.pgd_weight > 0:
            delta = torch.zeros_like(xs)
            for _ in range(args.pgd_steps):
                with torch.enable_grad():
                    d = delta.clone().requires_grad_(True)
                    obj = F.cross_entropy(self.model_b(xs + d), ys) - F.cross_entropy(self.model(xs + d), ys)
                    g, = torch.autograd.grad(obj, d)
                delta = (delta + self.pgd_eps() / 4 * g.sign()).clamp(-self.pgd_eps(), self.pgd_eps())
            images["pgd"] = xs + delta
        for name, v in images.items():
            out[name] = self.to_image(v)
            out[f"{name}_pred_b"] = self.model_b(v).argmax(1)
            out[f"{name}_pred_d"] = self.model(v).argmax(1)
        self.model_b.train(modes[0])
        self.model.train(modes[1])
        torch.save({key: v.cpu() if torch.is_tensor(v) else v for key, v in out.items()}, path)
