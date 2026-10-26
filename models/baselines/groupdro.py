import torch
import torch.nn.functional as F

from models.baselines.erm import Method


class GroupDRO(Method):
    def __init__(self, ctx):
        super().__init__(ctx)
        args, train, device = ctx.args, ctx.train, ctx.device
        self.num_groups = ctx.num_classes * ctx.num_biases
        self.group_of = train["y"] * ctx.num_biases + train["bias"]
        self.group_weight = torch.ones(self.num_groups, device=device) / self.num_groups
        size = torch.zeros(self.num_groups, device=device).index_add_(0, self.group_of, torch.ones(ctx.n_train, device=device))
        self.group_adjust = args.dro_adjust / size.clamp(min=1).sqrt()
        if args.dro_balanced:
            prob = (1 / size.clamp(min=1))[self.group_of]
            self.sample_prob = prob / prob.sum()

    def step(self, x, y, index, step):
        logits = self.model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        group = self.group_of[index]
        group_loss = torch.zeros(self.num_groups, device=x.device).index_add_(0, group, losses)
        group_count = torch.zeros(self.num_groups, device=x.device).index_add_(0, group, torch.ones_like(losses))
        group_loss = group_loss / group_count.clamp(min=1)
        present = group_count > 0
        adjusted = group_loss.detach() + self.group_adjust
        self.group_weight = self.group_weight * torch.exp(self.ctx.args.dro_eta * adjusted * present)
        self.group_weight = self.group_weight / self.group_weight.sum()
        self.update((self.group_weight * group_loss).sum())
        return logits
