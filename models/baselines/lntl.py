import torch.nn.functional as F

from models.baselines.erm import Method
from models.networks import GradientReversal


class LNTL(Method):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.select_from = ctx.cfg["num_steps"] // 2

    def step(self, x, y, index, step):
        args = self.ctx.args
        z = self.model.features(x)
        logits = self.model.head(z)
        loss = F.cross_entropy(logits, y)
        if step >= self.select_from:
            head = self.model.bias_head
            p = F.softmax(F.linear(z, head.weight.detach(), head.bias.detach()), 1)
            loss = loss + args.lntl_lambda * (p * (p + 1e-8).log()).sum(1).mean()
            loss = loss + F.cross_entropy(head(GradientReversal.apply(z, args.lntl_grl)), self.ctx.train["bias"][index])
        self.update(loss)
        return logits
