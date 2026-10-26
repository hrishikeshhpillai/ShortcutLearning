import argparse
import csv
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from models.baselines.erm import ERM
from models.baselines.groupdro import GroupDRO
from models.baselines.jtt import JTT
from models.baselines.ldd import LDD, LDD_SETTINGS
from models.baselines.lff import LfF
from models.baselines.lntl import LNTL
from models.proposed.unlearning import FavoredUnlearning
from utils.data import BENCHMARKS, aligned_bias, family, load_data, predict_logits, train_batch
from utils.metrics import split_scores

SETTINGS = {
    "cmnist_1pct": dict(batch_size=256, lr=1e-3, weight_decay=0.0, num_steps=23500, valid_freq=235),
    "cifar10c_1pct": dict(batch_size=256, lr=1e-3, weight_decay=0.0, num_steps=39200, valid_freq=196),
    "bffhq_0.5pct": dict(batch_size=64, lr=1e-4, weight_decay=0.0, num_steps=50000, valid_freq=500),
    "celeba": dict(batch_size=256, lr=1e-4, weight_decay=1e-4, num_steps=127200, valid_freq=636),
}
SELECTION_SPLIT = {"bffhq_0.5pct": "test"}
METHODS = {"vanilla": ERM, "lff": LfF, "jtt": JTT, "groupdro": GroupDRO, "lntl": LNTL, "ldd": LDD}


def parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--method", default="lff", choices=list(METHODS) + ["posthoc"])
    p.add_argument("--warmup", type=float)
    p.add_argument("--num_steps", type=int)
    p.add_argument("--unlearn_step", type=float, default=0.0)
    p.add_argument("--unlearn_pairing", default="favored", choices=["favored", "eq9"])
    p.add_argument("--unlearn_hessian", default="weighted", choices=["weighted", "identity"])
    p.add_argument("--unlearn_random", action="store_true")
    p.add_argument("--unlearn_adaptive", action="store_true")
    p.add_argument("--unlearn_safeguard", action="store_true")
    p.add_argument("--pairs_k", type=int)
    p.add_argument("--jtt_epochs", type=int, default=1)
    p.add_argument("--jtt_up", type=float, default=20.0)
    p.add_argument("--dro_eta", type=float, default=0.01)
    p.add_argument("--dro_adjust", type=float, default=0.0)
    p.add_argument("--dro_balanced", action="store_true")
    p.add_argument("--lntl_lambda", type=float, default=0.01)
    p.add_argument("--lntl_grl", type=float, default=0.1)
    p.add_argument("--be_committee", type=int, default=0)
    p.add_argument("--be_steps", type=int, default=1000)
    p.add_argument("--balance_pseudo", action="store_true")
    p.add_argument("--balance_mix", type=float, default=1.0)
    p.add_argument("--selecmix", type=float, default=0.0)
    return p


def settings(method, benchmark):
    cfg = dict(SETTINGS[family(benchmark)])
    if method == "ldd":
        cfg.update(LDD_SETTINGS[family(benchmark)], weight_decay=0.0)
    return cfg


def train(args, benchmark, seed, out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    cfg = settings(args.method, benchmark)
    full_steps = cfg["num_steps"]
    if args.num_steps:
        cfg["num_steps"] = args.num_steps
    warmup = args.warmup if args.warmup is not None else (0.02 if benchmark == "celeba" else 0.1)
    info = BENCHMARKS[benchmark]
    selection_split = SELECTION_SPLIT.get(benchmark, "valid")
    os.makedirs(out, exist_ok=True)
    print(f"{benchmark} seed {seed}: {vars(args)} {cfg}", flush=True)

    data = load_data(benchmark, device)
    train_split = data["train"]
    aligned = aligned_bias(train_split["y"].cpu().numpy(), train_split["bias"].cpu().numpy(), info["num_classes"],
                           info["num_biases"])
    ctx = SimpleNamespace(args=args, cfg=cfg, benchmark=benchmark, data=data, train=train_split, device=device,
                          num_classes=info["num_classes"], num_biases=info["num_biases"], aligned=aligned,
                          n_train=len(train_split["y"]))
    method = METHODS[args.method](ctx)
    unlearner = FavoredUnlearning(benchmark, train_split, info["num_classes"], seed, k=args.pairs_k) \
        if args.unlearn_step > 0 else None

    history = open(os.path.join(out, "history.csv"), "w", newline="")
    log = csv.writer(history)
    log.writerow(["step", "valid_conflicting", "valid_aligned", "test_conflicting", "test_aligned",
                  "biased_valid_conflicting"])
    best = dict(score=-1.0, step=-1)
    batches, start, correct, seen = [], time.time(), 0, 0
    for step in range(cfg["num_steps"]):
        if method.sample_prob is not None:
            index = torch.multinomial(method.sample_prob, cfg["batch_size"], replacement=True)
        else:
            if not batches:
                batches = list(torch.randperm(ctx.n_train, device=device).split(cfg["batch_size"]))
            index = batches.pop(0)
        x, y = train_batch(benchmark, train_split, index), train_split["y"][index]
        logits = method.step(x, y, index, step)
        if step % 10 == 0:
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        if step % cfg["valid_freq"] == 0 or step == cfg["num_steps"] - 1:
            if unlearner is not None and step >= warmup * full_steps:
                unlearner.step(method.model, method.hessian_weights(), args.unlearn_step, args.unlearn_pairing,
                               args.unlearn_hessian, args.unlearn_random, args.unlearn_adaptive, args.unlearn_safeguard)
            scores = {}
            for split in ("valid", "test"):
                pred = predict_logits(method.model, benchmark, data[split]["x"]).argmax(1)
                scores[split] = split_scores(data[split], pred, aligned, info["num_classes"], info["num_biases"])
            biased = method.biased_valid()
            log.writerow([step, *scores["valid"], *scores["test"], biased])
            history.flush()
            score = scores[selection_split][0]
            if score > best["score"] and step >= method.select_from:
                weights = method.hessian_weights() if hasattr(method, "ema_b") else None
                best = dict(score=score, step=step,
                            model={k: v.detach().cpu().clone() for k, v in method.model.state_dict().items()},
                            weights=None if weights is None else weights.cpu().numpy())
            print(f"step {step:6d}  epoch {step * cfg['batch_size'] / ctx.n_train:6.1f}  train acc "
                  f"{100 * correct / max(1, seen):6.2f}  valid conf {100 * scores['valid'][0]:6.2f}  test conf "
                  f"{100 * scores['test'][0]:6.2f}  best {100 * best['score']:6.2f} @ {best['step']}  "
                  f"({time.time() - start:.0f}s)", flush=True)
            correct = seen = 0
    history.close()
    torch.save(dict(model=best["model"], step=best["step"], weights=best["weights"], aligned_bias=aligned,
                    benchmark=benchmark, seed=seed, method=args.method, selection_split=selection_split,
                    selection_score=best["score"]), os.path.join(out, "best.pt"))
    print(f"done: best {selection_split} conflicting acc {100 * best['score']:.2f} at step {best['step']}", flush=True)
