import torch
import torch.nn.functional as F

from utils.newton import body_features, ce_gradients, ce_hessian, with_bias
from utils.pairs import DEFAULT_K, build_pairs


class FavoredUnlearning:
    def __init__(self, benchmark, train, num_classes, seed, k=None, damping=0.01, hessian_samples=20000):
        self.benchmark, self.train, self.num_classes = benchmark, train, num_classes
        self.damping, self.hessian_samples = damping, hessian_samples
        device = train["y"].device
        pairs = build_pairs(benchmark, k or DEFAULT_K[benchmark], seed=seed, device=device)
        self.y_pairs = torch.from_numpy(pairs["y"]).long().to(device)
        self.c_index = torch.from_numpy(pairs["c_index"]).long().to(device)
        self.cbar_index = torch.from_numpy(pairs["cbar_index"]).long().to(device)
        self.cbar_images = None if pairs["cbar_images"] is None else torch.from_numpy(pairs["cbar_images"]).to(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        labels = train["y"]
        for label in self.y_pairs.unique():
            rows = (self.y_pairs == label).nonzero().squeeze(1)
            pool = (labels == label).nonzero().squeeze(1)
            torch.randint(len(pool), (len(rows),), device=device, generator=self.generator)
        self.bias0 = None

    def state(self, model):
        was_training = model.training
        phi = with_bias(body_features(model, self.benchmark, self.train["x"]).float())
        phi_cbar = phi[self.cbar_index] if self.cbar_images is None else \
            with_bias(body_features(model, self.benchmark, self.cbar_images).float())
        model.train(was_training)
        theta = torch.cat([model.head.weight, model.head.bias[:, None]], 1).float()
        phi_c = phi[self.c_index]
        loss_c = F.cross_entropy(phi_c @ theta.T, self.y_pairs, reduction="none")
        loss_cbar = F.cross_entropy(phi_cbar @ theta.T, self.y_pairs, reduction="none")
        return phi, phi_c, phi_cbar, theta, torch.sign(loss_cbar - loss_c), loss_c, loss_cbar

    def solve(self, phi, theta, weights, vector):
        subset = torch.randperm(len(phi), device=phi.device, generator=self.generator)[:self.hessian_samples]
        h = ce_hessian(theta, phi[subset], weights[subset].float())
        h += self.damping * h.diagonal().mean().clamp(min=1e-6) * torch.eye(len(h), device=h.device)
        return torch.linalg.solve(h, vector)

    @torch.no_grad()
    def step(self, model, weights, r, pairing="favored", hessian="weighted", random=False, adaptive=False,
             safeguard=False):
        phi, phi_c, phi_cbar, theta, sign, loss_c, loss_cbar = self.state(model)
        bias = (loss_cbar - loss_c).abs().mean()
        if adaptive:
            if self.bias0 is None:
                self.bias0 = bias.clamp(min=1e-8)
            r = r * float((bias / self.bias0).clamp(0, 2))
        if pairing == "eq9":
            sign = torch.ones_like(sign)
        y = self.y_pairs
        delta = (sign[:, None] * (ce_gradients(theta, phi_c, y) - ce_gradients(theta, phi_cbar, y))).mean(0)
        direction = delta if hessian == "identity" else self.solve(phi, theta, weights, delta)
        if random:
            direction = torch.randn(direction.shape, device=direction.device, generator=self.generator)
        update = (r * theta.norm() * direction / direction.norm()).reshape(theta.shape)
        scale = 1.0
        if safeguard:
            subset = torch.randperm(len(phi), device=phi.device, generator=self.generator)[:self.hessian_samples]
            w = weights[subset].float()

            def measure(t):
                lc = F.cross_entropy(phi_c @ t.T, y, reduction="none")
                lcb = F.cross_entropy(phi_cbar @ t.T, y, reduction="none")
                fit = (F.cross_entropy(phi[subset] @ t.T, self.train["y"][subset], reduction="none") * w).sum() / w.sum()
                return (lcb - lc).abs().mean(), fit

            bias_now, fit_now = measure(theta)
            for _ in range(4):
                bias_new, fit_new = measure(theta + scale * update)
                if bias_new < bias_now and fit_new <= 1.05 * fit_now:
                    break
                scale /= 2
            else:
                scale = 0.0
        update = scale * update
        model.head.weight += update[:, :-1].to(model.head.weight.dtype)
        model.head.bias += update[:, -1].to(model.head.bias.dtype)
        return dict(bias=float(bias), r=r * scale)
