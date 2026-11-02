import torch
import torch.nn.functional as F

from poolforge.networks import network_for


class Method:
    sample_prob = None

    def __init__(self, ctx):
        self.ctx = ctx
        self.model = network_for(ctx.args.method, ctx.benchmark, ctx.num_classes).to(ctx.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ctx.cfg["lr"], weight_decay=ctx.cfg["weight_decay"])

    def biased_valid(self):
        return float("nan")

    def update(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


class Vanilla(Method):
    def step(self, x, y, index, step):
        logits = self.model(x)
        self.update(F.cross_entropy(logits, y))
        return logits
