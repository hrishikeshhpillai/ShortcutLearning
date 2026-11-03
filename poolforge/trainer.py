import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from poolforge.data import BENCHMARKS, aligned_bias, family, load_data, predict_logits, train_batch
from poolforge.methods.base import Vanilla
from poolforge.methods.jtt import JTT
from poolforge.methods.lc import LogitCorrection
from poolforge.methods.ldd import LDD, LDD_SETTINGS
from poolforge.methods.lff import LfF
from poolforge.methods.poolforge import PoolForge
from poolforge.metrics import split_scores

SETTINGS = {
    "cmnist_1pct": dict(batch_size=256, lr=1e-3, weight_decay=0.0, num_steps=23500, valid_freq=235),
    "cifar10c_1pct": dict(batch_size=256, lr=1e-3, weight_decay=0.0, num_steps=39200, valid_freq=196),
    "bffhq_0.5pct": dict(batch_size=64, lr=1e-4, weight_decay=0.0, num_steps=12000, valid_freq=500),
}
SELECTION_SPLIT = {"bffhq_0.5pct": "test"}
METHODS = {"vanilla": Vanilla, "lff": LfF, "jtt": JTT, "lc": LogitCorrection, "ldd": LDD, "poolforge": PoolForge}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="poolforge", choices=list(METHODS))
    p.add_argument("--num_steps", type=int)
    p.add_argument("--jtt_epochs", type=int, default=1)
    p.add_argument("--jtt_up", type=float, default=20.0)
    p.add_argument("--lc_alpha", type=float, default=0.5)
    p.add_argument("--lc_eta", type=float, default=1.0)
    p.add_argument("--lc_mix", type=float, default=1.0)
    p.add_argument("--no_reweight", action="store_true")
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--pool_top", type=float, default=0.01)
    p.add_argument("--pool_adaptive", type=int, default=1)
    p.add_argument("--pool_adaptive_from", type=float, default=0.25)
    p.add_argument("--pool_decay", type=float, default=0.8)
    p.add_argument("--mix", default="mix", choices=["mix", "repeat", "none"])
    p.add_argument("--mix_batch", type=int, default=32)
    p.add_argument("--mix_weight", type=float, default=1.0)
    p.add_argument("--gen", default="full", choices=["full", "color", "none"])
    p.add_argument("--gen_untargeted", action="store_true")
    p.add_argument("--gen_src", default="pool", choices=["pool", "batch"])
    p.add_argument("--gen_batch", type=int, default=64)
    p.add_argument("--gen_ops", type=int, default=4)
    p.add_argument("--gen_eps", type=float, default=0.5)
    p.add_argument("--gen_reg", type=float, default=1.0)
    p.add_argument("--gen_weight", type=float, default=1.0)
    p.add_argument("--pgd_weight", type=float, default=1.0)
    p.add_argument("--pgd_src", default="pool", choices=["pool", "batch"])
    p.add_argument("--pgd_eps", type=float, default=8 / 255)
    p.add_argument("--pgd_steps", type=int, default=5)
    p.add_argument("--dump_at", type=float, nargs="*", default=[])
    p.add_argument("--dump_n", type=int, default=512)
    return p


def settings(method, benchmark):
    cfg = dict(SETTINGS[family(benchmark)])
    if method == "ldd":
        cfg.update(LDD_SETTINGS[family(benchmark)], weight_decay=0.0)
    return cfg


def train(args, benchmark, seed, out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    cfg = settings(args.method, benchmark)
    if args.num_steps:
        cfg["num_steps"] = args.num_steps
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
    train_split["aligned_bias"] = torch.as_tensor(aligned, device=device)
    method = METHODS[args.method](ctx)

    history = open(os.path.join(out, "history.csv"), "w", newline="")
    log = csv.writer(history)
    log.writerow(["step", "valid_conflicting", "valid_aligned", "test_conflicting", "test_aligned",
                  "biased_valid_conflicting", "pool_fraction", "pool_precision", "pool_recall", "weight_ratio",
                  "fool_rate"])
    conflicting = train_split["bias"] != train_split["aligned_bias"][train_split["y"]]
    dump_steps = {min(int(f * cfg["num_steps"]), cfg["num_steps"] - 1) for f in args.dump_at}
    best = dict(score=-1.0, step=-1)
    batches, start, correct, seen = [], time.time(), 0, 0

    def mean_and_clear(name):
        values = getattr(method, name, [])
        value = sum(values) / len(values) if values else float("nan")
        values.clear()
        return value

    for step in range(cfg["num_steps"]):
        if method.sample_prob is not None:
            index = torch.multinomial(method.sample_prob, cfg["batch_size"], replacement=True)
        else:
            if not batches:
                batches = list(torch.randperm(ctx.n_train, device=device).split(cfg["batch_size"]))
            index = batches.pop(0)
        x, y = train_batch(benchmark, train_split, index), train_split["y"][index]
        logits = method.step(x, y, index, step)
        if step in dump_steps:
            with torch.random.fork_rng(devices=[device]):
                method.dump_case(os.path.join(out, f"case_step{step}.pt"), args.dump_n)
        if step % 10 == 0:
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        if step % cfg["valid_freq"] == 0 or step == cfg["num_steps"] - 1:
            scores = {}
            for split in ("valid", "test"):
                pred = predict_logits(method.model, benchmark, data[split]["x"]).argmax(1)
                scores[split] = split_scores(data[split], pred, aligned, info["num_classes"], info["num_biases"])
            mask = getattr(method, "conflict_mask", None)
            pool = [float("nan")] * 3
            if mask is not None:
                hit = (mask & conflicting).sum().item()
                pool = [method.pool_fraction, hit / max(mask.sum().item(), 1), hit / max(conflicting.sum().item(), 1)]
            log.writerow([step, *scores["valid"], *scores["test"], method.biased_valid(), *pool,
                          mean_and_clear("weight_log"), mean_and_clear("fool_log")])
            history.flush()
            score = scores[selection_split][0]
            if score > best["score"]:
                best = dict(score=score, step=step, test=scores["test"],
                            model={k: v.detach().cpu().clone() for k, v in method.model.state_dict().items()})
            print(f"step {step:6d}  epoch {step * cfg['batch_size'] / ctx.n_train:6.1f}  train acc "
                  f"{100 * correct / max(1, seen):6.2f}  valid conf {100 * scores['valid'][0]:6.2f}  test conf "
                  f"{100 * scores['test'][0]:6.2f}  best {100 * best['score']:6.2f} @ {best['step']}  "
                  f"({time.time() - start:.0f}s)", flush=True)
            correct = seen = 0
    history.close()
    torch.save(dict(model=best["model"], step=best["step"], aligned_bias=aligned, benchmark=benchmark, seed=seed,
                    method=args.method, selection_split=selection_split), os.path.join(out, "best.pt"))
    result = dict(benchmark=benchmark, seed=seed, selection_split=selection_split, best_step=best["step"],
                  test_conflicting=100 * best["test"][0], test_aligned=100 * best["test"][1])
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(f"done: test conflicting acc {result['test_conflicting']:.2f} at step {best['step']} "
          f"(selected on {selection_split})", flush=True)
