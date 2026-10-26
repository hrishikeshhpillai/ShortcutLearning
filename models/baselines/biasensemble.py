import torch
import torch.nn.functional as F

from models.baselines.lff_losses import generalized_ce
from models.networks import build_model
from utils.data import predict_logits, train_batch


def bias_committee(ctx):
    args, cfg, train = ctx.args, ctx.cfg, ctx.train
    votes = torch.zeros(ctx.n_train, device=ctx.device)
    for _ in range(args.be_committee):
        member = build_model(ctx.benchmark, ctx.num_classes).to(ctx.device)
        opt = torch.optim.Adam(member.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        for _ in range(args.be_steps):
            index = torch.randint(ctx.n_train, (cfg["batch_size"],), device=ctx.device)
            loss = generalized_ce(member(train_batch(ctx.benchmark, train, index)), train["y"][index]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        p_y = F.softmax(predict_logits(member, ctx.benchmark, train["x"]), 1).gather(1, train["y"][:, None])
        votes += (p_y.squeeze(1) >= 0.99).float()
        del member, opt
    pseudo_aligned = votes >= -(-args.be_committee // 2)
    print(f"bias committee: {int((~pseudo_aligned).sum())} pseudo bias-conflicting samples", flush=True)
    return pseudo_aligned


def pseudo_group_sampling(ctx, pseudo_aligned, mix):
    group = ctx.train["y"] * 2 + (~pseudo_aligned).long()
    size = torch.zeros(2 * ctx.num_classes, device=ctx.device).index_add_(0, group, torch.ones(ctx.n_train, device=ctx.device))
    prob = (1 / size.clamp(min=1))[group]
    return mix * prob / prob.sum() + (1 - mix) / ctx.n_train
