import torch
import torch.nn.functional as F

from models.baselines.biasensemble import bias_committee
from models.baselines.erm import Method
from models.baselines.lff_losses import EMA, generalized_ce, relative_difficulty

LDD_SETTINGS = {
    "cmnist_1pct": dict(batch_size=256, lr=0.01, num_steps=50000, valid_freq=500, curr_step=10000, lambda_swap=1,
                        lambda_dis_align=10, lambda_swap_align=10, lr_decay_step=10000, lr_gamma=0.5),
    "cifar10c_1pct": dict(batch_size=256, lr=0.001, num_steps=50000, valid_freq=500, curr_step=10000, lambda_swap=1,
                          lambda_dis_align=5, lambda_swap_align=5, lr_decay_step=10000, lr_gamma=0.5),
    "bffhq_0.5pct": dict(batch_size=64, lr=0.0001, num_steps=50000, valid_freq=500, curr_step=10000, lambda_swap=0.1,
                         lambda_dis_align=2, lambda_swap_align=2, lr_decay_step=10000, lr_gamma=0.1),
}


class LDD(Method):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.ema_b, self.ema_d = EMA(ctx.train["y"], ctx.num_classes), EMA(ctx.train["y"], ctx.num_classes)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=ctx.cfg["lr_decay_step"],
                                                         gamma=ctx.cfg["lr_gamma"])
        self.pseudo_aligned = bias_committee(ctx) if ctx.args.be_committee else None

    def hessian_weights(self):
        return relative_difficulty(self.ema_b, self.ema_d)

    def step(self, x, y, index, step):
        cfg, model = self.ctx.cfg, self.model
        z_i, z_b = model.encode(x)
        logit_d = model.head(torch.cat([z_i, z_b.detach()], 1))
        logit_align = model.head_b(torch.cat([z_i.detach(), z_b], 1))
        self.ema_d.update(F.cross_entropy(logit_d, y, reduction="none").detach(), index)
        self.ema_b.update(F.cross_entropy(logit_align, y, reduction="none").detach(), index)
        weight = relative_difficulty(self.ema_b, self.ema_d, index)
        keep = torch.ones_like(weight) if self.pseudo_aligned is None else self.pseudo_aligned[index].float()
        loss = (F.cross_entropy(logit_d, y, reduction="none") * weight).mean() \
            + cfg["lambda_dis_align"] * (generalized_ce(logit_align, y) * keep).sum() / keep.sum().clamp(min=1)
        if step > cfg["curr_step"]:
            perm = torch.randperm(len(y), device=x.device)
            z_swap = z_b[perm]
            mix_conflict = model.head(torch.cat([z_i, z_swap.detach()], 1))
            mix_align = model.head_b(torch.cat([z_i.detach(), z_swap], 1))
            loss = loss + cfg["lambda_swap"] * (
                (F.cross_entropy(mix_conflict, y, reduction="none") * weight).mean()
                + cfg["lambda_swap_align"] * (generalized_ce(mix_align, y[perm]) * keep[perm]).sum()
                / keep[perm].sum().clamp(min=1))
        self.update(loss)
        if step >= cfg["curr_step"]:
            self.scheduler.step()
        return logit_d
