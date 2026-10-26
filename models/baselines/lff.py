import torch
import torch.nn.functional as F

from models.baselines.biasensemble import bias_committee, pseudo_group_sampling
from models.baselines.erm import Method
from models.baselines.lff_losses import EMA, generalized_ce, relative_difficulty
from models.baselines.selecmix import selecmix_loss
from models.networks import build_model
from utils.data import predict_logits
from utils.metrics import split_scores


class LfF(Method):
    def __init__(self, ctx):
        super().__init__(ctx)
        args, cfg = ctx.args, ctx.cfg
        self.pseudo_aligned = bias_committee(ctx) if args.be_committee else None
        if args.balance_pseudo:
            self.sample_prob = pseudo_group_sampling(ctx, self.pseudo_aligned, args.balance_mix)
        self.model_b = build_model(ctx.benchmark, ctx.num_classes).to(ctx.device)
        self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.ema_b = EMA(ctx.train["y"], ctx.num_classes)
        self.ema_d = EMA(ctx.train["y"], ctx.num_classes)

    def hessian_weights(self):
        return relative_difficulty(self.ema_b, self.ema_d)

    def biased_valid(self):
        valid = self.ctx.data["valid"]
        pred = predict_logits(self.model_b, self.ctx.benchmark, valid["x"]).argmax(1)
        return split_scores(valid, pred, self.ctx.aligned, self.ctx.num_classes, self.ctx.num_biases)[0]

    def step(self, x, y, index, step):
        logit_b, logit_d = self.model_b(x), self.model(x)
        self.ema_b.update(F.cross_entropy(logit_b, y, reduction="none").detach(), index)
        self.ema_d.update(F.cross_entropy(logit_d, y, reduction="none").detach(), index)
        weight = relative_difficulty(self.ema_b, self.ema_d, index)
        if self.pseudo_aligned is None:
            loss_bias = generalized_ce(logit_b, y).mean()
        else:
            keep = self.pseudo_aligned[index].float()
            loss_bias = (F.cross_entropy(logit_b, y, reduction="none") * keep).sum() / keep.sum().clamp(min=1)
        loss = loss_bias + (F.cross_entropy(logit_d, y, reduction="none") * weight).mean()
        if self.ctx.args.selecmix > 0:
            loss = loss + self.ctx.args.selecmix * selecmix_loss(self.model_b, self.model, x, y)
        self.optimizer_b.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer_b.step()
        self.optimizer.step()
        return logit_d
