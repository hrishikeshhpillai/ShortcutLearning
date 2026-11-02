import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import run_dir
from poolforge.data import BENCHMARKS, load_data, load_split
from poolforge.metrics import cell_accuracies
from results import test_predictions

KINDS = [("x", "pool image"), ("gen", "generator"), ("pgd", "PGD"), ("mix", "mixup")]
TITLES = {"cmnist_1pct": "CMNIST", "cifar10c_1pct": "C-CIFAR10", "bffhq_0.5pct": "BFFHQ"}
CIFAR_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CORRUPTIONS = ["snow", "frost", "fog", "bright", "contrast", "spatter", "elastic", "jpeg", "pixel", "saturate"]


def class_name(benchmark, c):
    c = int(c)
    if benchmark.startswith("cifar10c"):
        return CIFAR_NAMES[c]
    if benchmark.startswith("bffhq"):
        return ["young", "old"][c]
    return str(c)


def to_input(benchmark, images):
    x = images.float() / 255
    if benchmark.startswith("cmnist"):
        return x
    size = 64 if benchmark.startswith("bffhq") else 32
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.25


def probe_net(benchmark, num_biases):
    if benchmark.startswith("cmnist"):
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 28 * 28, 256), nn.ReLU(), nn.Linear(256, num_biases))
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(128, num_biases))


def train_probe(benchmark, device, out_dir, steps=3000):
    path = os.path.join(out_dir, f"bias_probe_{benchmark}.pt")
    info = BENCHMARKS[benchmark]
    data = load_data(benchmark, device)
    net = probe_net(benchmark, info["num_biases"]).to(device)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
        return net.eval(), json.load(open(path + ".json"))["test_acc"]
    torch.manual_seed(0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train = data["train"]
    counts = torch.bincount(train["bias"], minlength=info["num_biases"]).float()
    prob = (1 / counts)[train["bias"]]
    for _ in range(steps):
        idx = torch.multinomial(prob, 128, replacement=True)
        x = to_input(benchmark, train["x"][idx].permute(0, 3, 1, 2))
        loss = F.cross_entropy(net(x), train["bias"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    test = data["test"]
    with torch.no_grad():
        pred = torch.cat([net(to_input(benchmark, test["x"][i:i + 500].permute(0, 3, 1, 2))).argmax(1)
                          for i in range(0, len(test["y"]), 500)])
    acc = (pred == test["bias"]).float().mean().item()
    torch.save(net.state_dict(), path)
    json.dump(dict(test_acc=acc), open(path + ".json", "w"))
    return net, acc


@torch.no_grad()
def statistics(case, probe, benchmark, device):
    y, k, bias, aligned = case["y"], case["k"], case["bias"], case["aligned_bias"]
    row = dict(pool_size=case.get("pool_size", case.get("conflicting")), pool_conflicting=(bias != aligned[y]).float().mean().item())
    for kind, _ in KINDS:
        if kind not in case:
            continue
        b_pred = probe(to_input(benchmark, case[kind].to(device))).argmax(1).cpu()
        row[f"{kind}_fd_correct"] = (case[f"{kind}_pred_d"] == y).float().mean().item()
        row[f"{kind}_fb_correct"] = (case[f"{kind}_pred_b"] == y).float().mean().item()
        row[f"{kind}_bias_kept"] = (b_pred == bias).float().mean().item()
        row[f"{kind}_bias_aligned"] = (b_pred == aligned[y]).float().mean().item()
        if kind == "gen":
            row["gen_fb_target"] = (case["gen_pred_b"] == k).float().mean().item()
            row["gen_bias_target"] = (b_pred == aligned[k]).float().mean().item()
        if kind != "x":
            row[f"{kind}_edit"] = (case[kind].float() - case["x"].float()).abs().mean().item()
    return row


def pick(case, n):
    order, seen = [], set()
    for i, c in enumerate(case["y"].tolist()):
        if c not in seen:
            order.append(i)
            seen.add(c)
    order += [i for i in range(len(case["y"])) if i not in order]
    return order[:n]


def show(ax, image):
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.set_xticks([])
    ax.set_yticks([])


def grid(case, benchmark, path, n=6):
    rows = pick(case, n)
    kinds = [kd for kd in KINDS if kd[0] in case]
    fig, axes = plt.subplots(len(rows), len(kinds), figsize=(1.9 * len(kinds), 1.95 * len(rows)))
    name = lambda c: class_name(benchmark, int(c))
    for r, i in enumerate(rows):
        for col, (kind, title) in enumerate(kinds):
            ax = axes[r, col]
            show(ax, case[kind][i])
            ok_d = case[f"{kind}_pred_d"][i] == case["y"][i]
            label = f"$f_b$:{name(case[f'{kind}_pred_b'][i])}  $f_d$:{name(case[f'{kind}_pred_d'][i])}"
            ax.set_xlabel(label, fontsize=8, color="black" if ok_d else "red")
            if r == 0:
                ax.set_title(title if kind != "gen" else "generator\n(to target $k$)", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"y={name(case['y'][i])}", fontsize=9)
            if kind == "gen":
                ax.text(0.03, 0.97, f"k={name(case['k'][i])}", transform=ax.transAxes, fontsize=8, va="top",
                        color="white", bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strip(case, benchmark, path, index):
    kinds = [kd for kd in KINDS if kd[0] in case]
    fig, axes = plt.subplots(1, len(kinds), figsize=(1.2 * len(kinds), 1.2))
    for ax, (kind, _) in zip(axes, kinds):
        show(ax, case[kind][index])
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def thumbnails(case, benchmark, out_dir, index):
    tag = "".join(ch for ch in benchmark.split("_")[0] if ch.isalpha())
    name = lambda c: class_name(benchmark, int(c))
    y = int(case["y"][index])
    macros = [f"\\def\\{tag}y{{{name(y)}}}", f"\\def\\{tag}k{{{name(case['k'][index])}}}"]
    for kind, _ in KINDS:
        if kind in case:
            plt.imsave(os.path.join(out_dir, f"thumb_{benchmark}_{kind}.png"), case[kind][index].permute(1, 2, 0).numpy())
            macros += [f"\\def\\{tag}{kind}b{{{name(case[f'{kind}_pred_b'][index])}}}",
                       f"\\def\\{tag}{kind}d{{{name(case[f'{kind}_pred_d'][index])}}}"]
    train = load_split(benchmark, "train")
    aligned = case["aligned_bias"].numpy()
    for kind, mask in (("aligned", train["bias"] == aligned[train["y"]]), ("conflicting", train["bias"] != aligned[train["y"]])):
        i = np.nonzero(mask & (train["y"] == y))[0][0]
        plt.imsave(os.path.join(out_dir, f"thumb_{benchmark}_train_{kind}.png"), train["images"][i])
        macros.append(f"\\def\\{tag}train{kind}{{{bias_name(benchmark, train['bias'][i])}}}")
    open(os.path.join(out_dir, f"labels_{benchmark}.tex"), "w").write("\n".join(macros) + "\n")


def bias_name(benchmark, b):
    b = int(b)
    if benchmark.startswith("cifar10c"):
        return CORRUPTIONS[b]
    if benchmark.startswith("bffhq"):
        return ["female", "male"][b]
    return f"color {b}"


def test_cases(benchmark, device, out_dir, per_group=6, seed=0):
    preds, cells = {}, {}
    for name in ("lff", "poolforge"):
        folder = run_dir(name, benchmark, seed)
        if not os.path.exists(os.path.join(folder, "best.pt")):
            print(f"no checkpoint in {folder}")
            return None
        logits, test, aligned = test_predictions(folder, device)
        preds[name] = logits.argmax(1).cpu()
        info = BENCHMARKS[benchmark]
        cells[name], count = cell_accuracies(test["y"].cpu().numpy(), test["bias"].cpu().numpy(),
                                             preds[name].numpy(), info["num_classes"], info["num_biases"])
    y, bias, images = test["y"].cpu(), test["bias"].cpu(), test["x"].cpu()
    conflicting = bias != torch.as_tensor(aligned)[y]
    groups = [(c, b) for c in range(cells["poolforge"].shape[0]) for b in range(cells["poolforge"].shape[1])
              if count[c, b] > 0 and aligned[c] != b]
    groups.sort(key=lambda g: cells["poolforge"][g])
    worst, best = groups[:3], groups[-3:][::-1]
    group_rows = [dict(kind=kind, y=class_name(benchmark, c), bias=bias_name(benchmark, b), n=int(count[c, b]),
                       poolforge=100 * float(cells["poolforge"][c, b]), lff=100 * float(cells["lff"][c, b]))
                  for kind, gs in (("best", best), ("worst", worst)) for c, b in gs]
    ok_p, ok_l = preds["poolforge"] == y, preds["lff"] == y
    outcome = {"both correct": ok_p & ok_l, "PoolForge correct, LfF wrong": ok_p & ~ok_l,
               "PoolForge wrong, LfF correct": ~ok_p & ok_l, "both wrong": ~ok_p & ~ok_l}
    counts = {k: int((v & conflicting).sum()) for k, v in outcome.items()}
    g = torch.Generator().manual_seed(0)
    fig, axes = plt.subplots(len(outcome), per_group, figsize=(1.7 * per_group + 1.2, 1.9 * len(outcome)))
    for r, (label, mask) in enumerate(outcome.items()):
        idx = torch.nonzero(mask & conflicting).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)[:per_group]]
        for col in range(per_group):
            ax = axes[r, col]
            ax.axis("off")
            if col >= len(idx):
                continue
            i = idx[col].item()
            ax.imshow(images[i].numpy())
            ax.set_title(f"y={class_name(benchmark, y[i])}, {bias_name(benchmark, bias[i])}\nLfF: {class_name(benchmark, preds['lff'][i])}  "
                         f"Ours: {class_name(benchmark, preds['poolforge'][i])}", fontsize=8)
        axes[r, 0].text(-0.12, 0.5, label.replace(", ", ",\n") + f"\n(n={counts[label]})", transform=axes[r, 0].transAxes, fontsize=8.5,
                        ha="right", va="center")
    fig.tight_layout(rect=(0.11, 0, 1, 1))
    fig.savefig(os.path.join(out_dir, f"test_outcomes_{benchmark}.png"), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(5.4, 4.0))
    for ax, row, (c, b) in zip(axes.flatten(), group_rows, best + worst):
        i = torch.nonzero((y == c) & (bias == b)).flatten()[0].item()
        ax.imshow(images[i].numpy())
        ax.axis("off")
        ax.set_title(f"{row['kind']}: y={row['y']}, {row['bias']}\nOurs {row['poolforge']:.0f}%  LfF {row['lff']:.0f}%",
                     fontsize=7.5, color="#1a7f37" if row["kind"] == "best" else "#b42318")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"test_groups_{benchmark}.png"), dpi=170)
    plt.close(fig)
    return dict(groups=group_rows, outcomes=counts, conflicting=int(conflicting.sum()))


def main():
    p = argparse.ArgumentParser(description="Figures and statistics from the images saved by the case_study runs.")
    p.add_argument("--runs", default=None, help="folder holding <benchmark>/seed0/case_step*.pt (default: runs/case_study)")
    p.add_argument("--out", default="results/case_study")
    p.add_argument("--benchmarks", nargs="+", default=list(TITLES))
    p.add_argument("--test_only", action="store_true", help="only the best/worst test-group analysis")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    table = {}
    for benchmark in args.benchmarks:
        table.setdefault(benchmark, {})["test"] = test_cases(benchmark, device, args.out)
        if args.test_only:
            continue
        folder = os.path.join(args.runs, benchmark, "seed0") if args.runs else run_dir("case_study", benchmark, 0)
        files = sorted(glob.glob(os.path.join(folder, "case_step*.pt")), key=lambda f: int(f.split("step")[-1][:-3]))
        if not files:
            print(f"no saved images for {benchmark} in {folder}")
            continue
        probe, probe_acc = train_probe(benchmark, device, args.out)
        table[benchmark].update(probe_test_acc=probe_acc, steps={})
        for f in files:
            step = int(f.split("step")[-1][:-3])
            case = torch.load(f, weights_only=False)
            table[benchmark]["steps"][step] = statistics(case, probe, benchmark, device)
            grid(case, benchmark, os.path.join(args.out, f"grid_{benchmark}_step{step}.png"))
        index = pick(case, 1)[0]
        strip(case, benchmark, os.path.join(args.out, f"strip_{benchmark}.png"), index)
        thumbnails(case, benchmark, args.out, index)
        print(benchmark, json.dumps(table[benchmark], indent=1))
    json.dump(table, open(os.path.join(args.out, "statistics.json"), "w"), indent=1)
    write_stats_table(table, os.path.join(args.out, "case_stats.tex"))


def write_stats_table(table, path):
    rows = [("Pool images that are bias-conflicting", "pool_conflicting", 100),
            ("Generator: $f_b$ predicts target $k$", "gen_fb_target", 100),
            ("Generator: $f_d$ predicts $y$", "gen_fd_correct", 100),
            ("Generator: probe finds original bias", "gen_bias_kept", 100),
            ("Generator: probe finds bias of $k$", "gen_bias_target", 100),
            ("Generator: mean $|\\Delta|$ (0--255)", "gen_edit", 1),
            ("PGD: $f_b$ wrong", "pgd_fb_correct", -100),
            ("PGD: $f_d$ predicts $y$", "pgd_fd_correct", 100),
            ("PGD: probe finds original bias", "pgd_bias_kept", 100),
            ("PGD: mean $|\\Delta|$ (0--255)", "pgd_edit", 1),
            ("Mixup: $f_d$ predicts $y$", "mix_fd_correct", 100),
            ("Mixup: $f_b$ predicts $y$", "mix_fb_correct", 100)]
    names = [b for b in TITLES if b in table and table[b].get("steps")]
    last = {b: table[b]["steps"][max(table[b]["steps"], key=int)] for b in names}
    body = []
    for label, key, scale in rows:
        cells = []
        for b in names:
            v = last[b].get(key)
            cells.append("--" if v is None else f"{(100 - 100 * v) if scale < 0 else scale * v:.1f}")
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    probe = " & ".join(f"{100 * table[b]['probe_test_acc']:.1f}" for b in names)
    open(path, "w").write("\\begin{tabular}{l" + "c" * len(names) + "}\n\\toprule\n & " +
                          " & ".join(TITLES[b] for b in names) + " \\\\\n\\midrule\n" + "\n".join(body) +
                          f"\n\\midrule\nBias probe, test accuracy & {probe} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


if __name__ == "__main__":
    main()
