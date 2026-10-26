import torch
import torch.nn.functional as F


def selecmix_loss(model_b, model_d, x, y):
    with torch.no_grad():
        z = F.normalize(model_b.features(x), dim=1)
        sim = z @ z.T
        same = y[:, None] == y[None, :]
        self_mask = torch.eye(len(y), dtype=torch.bool, device=x.device)
        positive = sim.masked_fill(~same | self_mask, float("inf")).argmin(1)
        negative = sim.masked_fill(same, float("-inf")).argmax(1)
        partner = torch.where(torch.rand(len(y), device=x.device) < 0.5, positive, negative)
        lam = torch.rand(len(y), device=x.device)
        lam = torch.minimum(lam, 1 - lam).view(-1, *([1] * (x.dim() - 1)))
        x_mix = lam * x + (1 - lam) * x[partner]
    return F.cross_entropy(model_d(x_mix), y[partner])
