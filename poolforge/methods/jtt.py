import torch
import torch.nn.functional as F

from poolforge.data import predict_logits, train_batch
from poolforge.methods.base import Vanilla
from poolforge.networks import build_model


class JTT(Vanilla):
    def __init__(self, ctx):
        super().__init__(ctx)
        args, cfg, train = ctx.args, ctx.cfg, ctx.train
        steps_per_epoch = -(-ctx.n_train // cfg["batch_size"])
        order = []
        for _ in range(args.jtt_epochs * steps_per_epoch):
            if not order:
                order = list(torch.randperm(ctx.n_train, device=ctx.device).split(cfg["batch_size"]))
            index = order.pop(0)
            self.update(F.cross_entropy(self.model(train_batch(ctx.benchmark, train, index)), train["y"][index]))
        errors = predict_logits(self.model, ctx.benchmark, train["x"]).argmax(1) != train["y"]
        prob = torch.where(errors, args.jtt_up, 1.0)
        self.sample_prob = prob / prob.sum()
        print(f"JTT stage 1 ({args.jtt_epochs} epochs): error set {int(errors.sum())} samples, upsampled x{args.jtt_up:g}",
              flush=True)
        self.model = build_model(ctx.benchmark, ctx.num_classes).to(ctx.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
