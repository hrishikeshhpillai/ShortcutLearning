import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.registry import RUNS
from utils.data import BENCHMARKS, MAIN, aligned_bias, load_split, preprocess
from utils.evaluate import load_checkpoint
from utils.pairs import cmnist_palette, recolor
from utils.tables import SHORT_TITLES, Table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["vanilla", "jtt", "lntl", "ldd", "lff", "fmd", "groupdro", "ours"]


@torch.no_grad()
def features_and_logits(model, benchmark, x, batch_size=250):
    feats, logits = [], []
    for i in range(0, len(x), batch_size):
        z = model.features(preprocess(benchmark, x[i:i + batch_size], train=False)).float()
        feats.append(z)
        logits.append(model.head(z).float())
    return torch.cat(feats), torch.cat(logits)


def balanced_subset(y, bias, seed=0):
    groups = y * (int(bias.max()) + 1) + bias
    present = torch.unique(groups)
    n = min(int((groups == g).sum()) for g in present)
    gen = torch.Generator(device=y.device).manual_seed(seed)
    return torch.cat([torch.nonzero(groups == g).squeeze(1)[torch.randperm(int((groups == g).sum()), generator=gen,
                                                                           device=y.device)[:n]] for g in present])


def probe_accuracy(z, target, num_values, seed=0, steps=500):
    gen = torch.Generator(device=z.device).manual_seed(seed)
    folds, correct = torch.randperm(len(z), generator=gen, device=z.device).chunk(2), 0
    for k in range(2):
        tr, te = folds[1 - k], folds[k]
        mean, std = z[tr].mean(0), z[tr].std(0) + 1e-6
        w = torch.zeros(z.shape[1], num_values, device=z.device, requires_grad=True)
        b = torch.zeros(num_values, device=z.device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.01, weight_decay=1e-4)
        xtr = (z[tr] - mean) / std
        for _ in range(steps):
            loss = F.cross_entropy(xtr @ w + b, target[tr])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            correct += int((((z[te] - mean) / std) @ w + b).argmax(1).eq(target[te]).sum())
    return correct / len(z)


def group_accuracy(pred, y, bias, aligned):
    conflicting = bias != torch.from_numpy(aligned).to(y.device)[y]
    out = {}
    for name, mask in (("aligned", ~conflicting), ("conflicting", conflicting)):
        cells = [(pred[mask & (y == c) & (bias == b)] == c).float().mean().item()
                 for c in torch.unique(y) for b in torch.unique(bias) if (mask & (y == c) & (bias == b)).any()]
        out[name] = 100 * float(np.mean(cells))
    return out


def evidence(predictions, seeds=(0, 1, 2)):
    from utils.data import load_data

    device = torch.device("cuda")
    results = {}
    for benchmark in MAIN:
        data = load_data(benchmark, device)
        test, info = data["test"], BENCHMARKS[benchmark]
        train = load_split(benchmark, "train")
        aligned = aligned_bias(train["y"], train["bias"], info["num_classes"], info["num_biases"])
        subset = balanced_subset(test["y"], test["bias"])
        palette = cmnist_palette(train["images"], train["bias"], info["num_biases"]) if benchmark.startswith("cmnist") else None
        for name in METHODS:
            rows = []
            for seed in seeds:
                model, _ = load_checkpoint(name, benchmark, seed, device)
                if model is None:
                    continue
                z, logits = features_and_logits(model, benchmark, test["x"])
                acc = group_accuracy(logits.argmax(1), test["y"], test["bias"], aligned)
                row = dict(gap=acc["aligned"] - acc["conflicting"])
                if benchmark != "celeba":
                    row["probe"] = 100 * probe_accuracy(z[subset], test["bias"][subset], info["num_biases"], seed)
                if palette is not None:
                    images = test["x"].cpu().numpy()
                    preds = torch.stack([features_and_logits(model, benchmark, torch.from_numpy(
                        recolor(images, np.full(len(images), c), palette)).to(device))[1].argmax(1)
                        for c in range(info["num_biases"])])
                    row["invariant"] = 100 * (preds == preds[0]).all(0).float().mean().item()
                    row["cf_acc"] = 100 * (preds == test["y"][None]).float().mean().item()
                rows.append(row)
            if rows:
                results[f"{name}|{benchmark}"] = {k: [r[k] for r in rows] for k in rows[0]}
        del data
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(REPO, "outputs", "cache"), exist_ok=True)
    json.dump(results, open(os.path.join(REPO, "outputs", "cache", "evidence.json"), "w"))
    return results


def evidence_tables(results):
    fmt = lambda v: f"{np.mean(v):.1f} ± {np.std(v):.1f}" if len(v) > 1 else f"{np.mean(v):.1f}"
    t = Table("evidence_bias", ["Method"] + [f"{SHORT_TITLES[b]} gap" for b in MAIN] +
              [f"{SHORT_TITLES[b]} probe" for b in MAIN if b != "celeba"],
              "Bias gap (aligned minus conflicting test accuracy) and bias probe (accuracy of a linear classifier that "
              "recovers the bias attribute from the penultimate features on a group-balanced test subset; chance: 10% "
              "CMNIST and C-CIFAR10, 50% BFFHQ). Lower is less shortcut use.")
    for name in METHODS:
        cells = [RUNS[name]["label"]]
        for key, benches in (("gap", MAIN), ("probe", [b for b in MAIN if b != "celeba"])):
            for b in benches:
                r = results.get(f"{name}|{b}")
                cells.append(fmt(r[key]) if r else "–")
        t.add(cells, bold=[0] if name == "ours" else [])
    t.save()
    c = Table("evidence_recolor", ["Method", "Color-invariant", "Counterfactual acc"],
              "CMNIST test digits redrawn in each of the 10 colors: share of digits whose prediction is the same in all "
              "10 colors, and accuracy over all recolored digits (%).")
    for name in METHODS:
        r = results.get(f"{name}|cmnist_1pct")
        if r:
            c.add([RUNS[name]["label"], fmt(r["invariant"]), fmt(r["cf_acc"])], bold=[0] if name == "ours" else [])
    c.save()
