import torch
import torch.nn.functional as F


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
