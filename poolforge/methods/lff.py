import torch
import torch.nn.functional as F

from poolforge.data import predict_logits
from poolforge.methods.base import Method
from poolforge.metrics import split_scores
from poolforge.networks import build_model


class EMA:
    def __init__(self, labels, num_classes, alpha=0.7):
        self.labels, self.num_classes, self.alpha = labels, num_classes, alpha
        self.parameter = torch.zeros(len(labels), device=labels.device)
        self.updated = torch.zeros(len(labels), device=labels.device)

    def update(self, loss, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * loss
        self.updated[index] = 1

    def normalized(self, index=None):
        class_max = torch.zeros(self.num_classes, device=self.labels.device).scatter_reduce(
            0, self.labels, self.parameter, "amax", include_self=False)
        if index is None:
            return self.parameter / class_max[self.labels]
        return self.parameter[index] / class_max[self.labels[index]]


def generalized_ce(logits, targets, q=0.7):
    p_target = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1).detach()
    return F.cross_entropy(logits, targets, reduction="none") * (p_target ** q) * q


def relative_difficulty(ema_b, ema_d, index=None):
    loss_b, loss_d = ema_b.normalized(index), ema_d.normalized(index)
    return loss_b / (loss_b + loss_d + 1e-8)


class LfF(Method):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.model_b = build_model(ctx.benchmark, ctx.num_classes).to(ctx.device)
        self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=ctx.cfg["lr"], weight_decay=ctx.cfg["weight_decay"])
        self.ema_b = EMA(ctx.train["y"], ctx.num_classes)
        self.ema_d = EMA(ctx.train["y"], ctx.num_classes)
        self.weight_log = []

    def biased_valid(self):
        valid = self.ctx.data["valid"]
        pred = predict_logits(self.model_b, self.ctx.benchmark, valid["x"]).argmax(1)
        return split_scores(valid, pred, self.ctx.aligned, self.ctx.num_classes, self.ctx.num_biases)[0]

    def reweighted_loss(self, x, y, index, uniform=False):
        logit_b, logit_d = self.model_b(x), self.model(x)
        self.ema_b.update(F.cross_entropy(logit_b, y, reduction="none").detach(), index)
        self.ema_d.update(F.cross_entropy(logit_d, y, reduction="none").detach(), index)
        weight = relative_difficulty(self.ema_b, self.ema_d, index)
        if uniform:
            weight = torch.ones_like(weight)
        with torch.no_grad():
            truth = self.ctx.train["bias"][index] != self.ctx.train["aligned_bias"][y]
            if truth.any() and (~truth).any():
                self.weight_log.append((weight[truth].mean() / weight[~truth].mean().clamp(min=1e-8)).item())
        loss = generalized_ce(logit_b, y).mean() + (F.cross_entropy(logit_d, y, reduction="none") * weight).mean()
        return loss, logit_d

    def step(self, x, y, index, step):
        loss, logit_d = self.reweighted_loss(x, y, index)
        self.optimizer_b.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer_b.step()
        self.optimizer.step()
        return logit_d
