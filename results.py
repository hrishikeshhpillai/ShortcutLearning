import argparse
import glob
import json
import os

import numpy as np
import torch

from experiments import EXPERIMENTS, MAIN, RATIOS, REPO, run_dir
from poolforge.data import BENCHMARKS, aligned_bias, load_data, load_split, predict_logits
from poolforge.metrics import group_metrics
from poolforge.networks import network_for

METRICS = ("conflicting", "aligned", "worst", "unbiased")
SHORT = {"cmnist_1pct": "CMNIST", "cifar10c_1pct": "C-CIFAR10", "bffhq_0.5pct": "BFFHQ"}
BASELINES = ("vanilla", "lff", "jtt", "lc", "ldd")
NAMES = {"vanilla": "Vanilla", "lff": "LfF", "jtt": "JTT", "lc": "LC", "ldd": "LDD", "poolforge": "PoolForge"}


def test_predictions(folder, device, cache={}):
    ckpt = torch.load(os.path.join(folder, "best.pt"), weights_only=False)
    benchmark = ckpt["benchmark"]
    if benchmark not in cache:
        cache.clear()
        cache[benchmark] = load_data(benchmark, device)
    data = cache[benchmark]
    model = network_for(ckpt["method"], benchmark, BENCHMARKS[benchmark]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model"])
    return predict_logits(model, benchmark, data["test"]["x"]), data["test"], ckpt["aligned_bias"]


def run_metrics(folder, device):
    path = os.path.join(folder, "metrics.json")
    if os.path.exists(path):
        return json.load(open(path))
    logits, test, aligned = test_predictions(folder, device)
    info = BENCHMARKS[json.load(open(os.path.join(folder, "result.json")))["benchmark"]]
    m = group_metrics(test["y"].cpu().numpy(), test["bias"].cpu().numpy(), logits.argmax(1).cpu().numpy(), aligned,
                      info["num_classes"], info["num_biases"])
    out = {k: 100 * float(m[k]) for k in METRICS}
    json.dump(out, open(path, "w"), indent=1)
    return out


def collect(device):
    table = {}
    for name in EXPERIMENTS:
        for folder in sorted(glob.glob(os.path.join(REPO, "runs", name, "*", "seed*"))):
            if not os.path.exists(os.path.join(folder, "best.pt")):
                continue
            benchmark, seed = folder.split(os.sep)[-2], int(folder.split("seed")[-1])
            table.setdefault(name, {}).setdefault(benchmark, {})[seed] = run_metrics(folder, device)
    return table


def mean_std(table, name, benchmark, metric="conflicting", seeds=None):
    runs = table.get(name, {}).get(benchmark, {})
    values = [runs[s][metric] for s in sorted(runs) if seeds is None or s in seeds]
    if not values:
        return None, None, 0
    return float(np.mean(values)), float(np.std(values)) if len(values) > 1 else None, len(values)


def cell(mean, std, bold=False):
    if mean is None:
        return "--"
    text = f"{mean:.1f}" + (f" $\\pm$ {std:.1f}" if std is not None else "")
    return f"\\textbf{{{text}}}" if bold else text


def ratio_name(benchmark):
    data, ratio = benchmark.split("_")
    return {"cmnist": "CMNIST", "cifar10c": "C-CIFAR10"}[data] + " " + ratio.replace("pct", "\\%")


def dataset_rows():
    rows = []
    for benchmark in MAIN + RATIOS:
        info = BENCHMARKS[benchmark]
        sizes, train = {}, load_split(benchmark, "train")
        aligned = aligned_bias(train["y"], train["bias"], info["num_classes"], info["num_biases"])
        for split in ("train", "valid", "test"):
            raw = load_split(benchmark, split)
            conflicting = raw["bias"] != aligned[raw["y"]]
            sizes[split] = (len(raw["y"]), int(conflicting.sum()))
        rows.append(dict(benchmark=benchmark, shape=list(raw["images"].shape[1:]), **{s: v for s, v in sizes.items()}))
    return rows


def write_tables(table, out):
    os.makedirs(out, exist_ok=True)
    lines = []
    for metric, label in [("conflicting", "bias-conflicting"), ("unbiased", "unbiased"), ("worst", "worst-group")]:
        best = {b: max((mean_std(table, n, b, metric)[0] or -1) for n in BASELINES + ("poolforge",)) for b in MAIN}
        rows = []
        for name in BASELINES + ("poolforge",):
            vals = [mean_std(table, name, b, metric) for b in MAIN]
            means = [v[0] for v in vals]
            avg = float(np.mean(means)) if all(m is not None for m in means) else None
            row = " & ".join([NAMES[name]] + [cell(m, s, m is not None and abs(m - best[b]) < 1e-9)
                                              for (m, s, _), b in zip(vals, MAIN)] + [cell(avg, None)])
            rows.append(("\\rowcolor{ours}" if name == "poolforge" else "") + row + " \\\\")
        open(os.path.join(out, f"main_{metric}.tex"), "w").write(
            "\\begin{tabular}{lcccc}\n\\toprule\nMethod & " + " & ".join(SHORT[b] for b in MAIN) +
            " & Avg. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        lines.append(f"\n### Test {label} accuracy (%, mean ± std over seeds)\n")
        lines.append("| Method | " + " | ".join(SHORT[b] for b in MAIN) + " | Avg. |\n|---|" + "---|" * (len(MAIN) + 1))
        for name in BASELINES + ("poolforge",):
            vals = [mean_std(table, name, b, metric) for b in MAIN]
            means = [v[0] for v in vals]
            avg = f"{np.mean(means):.1f}" if all(m is not None for m in means) else "–"
            lines.append(f"| {NAMES[name]} | " + " | ".join(
                "–" if m is None else f"{m:.1f}" + (f" ± {s:.1f}" if s is not None else "") for m, s, _ in vals) + f" | {avg} |")

    rows, md = [], ["\n### Bias ratios (seed 0, test bias-conflicting %)\n",
                    "| Method | " + " | ".join(RATIOS) + " | Avg. |\n|---|" + "---|" * (len(RATIOS) + 1)]
    for name in ("lff", "poolforge"):
        vals = [mean_std(table, name, b, seeds=(0,))[0] for b in RATIOS]
        avg = float(np.mean(vals)) if all(v is not None for v in vals) else None
        rows.append(("\\rowcolor{ours}" if name == "poolforge" else "") + " & ".join([NAMES[name]] + [cell(v, None) for v in vals] + [cell(avg, None)]) + " \\\\")
        md.append(f"| {NAMES[name]} | " + " | ".join("–" if v is None else f"{v:.1f}" for v in vals) + f" | {'–' if avg is None else f'{avg:.1f}'} |")
    head = " & ".join(ratio_name(b) for b in RATIOS)
    open(os.path.join(out, "ratio.tex"), "w").write(
        "\\begin{tabular}{l" + "c" * (len(RATIOS) + 1) + "}\n\\toprule\nMethod & " + head + " & Avg. \\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    lines += md

    ablations = ["poolforge"] + [n for n in EXPERIMENTS if EXPERIMENTS[n]["group"] == "ablations"]
    ref = [mean_std(table, "poolforge", b, seeds=(0,))[0] for b in MAIN]
    rows, md = [], ["\n### Ablations (seed 0, test bias-conflicting %)\n",
                    "| Variant | " + " | ".join(SHORT[b] for b in MAIN) + " | Avg. | Δ |\n|---|" + "---|" * (len(MAIN) + 2)]
    lff = [mean_std(table, "lff", b, seeds=(0,))[0] for b in MAIN]
    for name in ablations + ["lff"]:
        vals = [mean_std(table, name, b, seeds=(0,))[0] for b in MAIN]
        avg = float(np.mean(vals)) if all(v is not None for v in vals) else None
        ref_avg = float(np.mean(ref)) if all(v is not None for v in ref) else None
        delta = None if avg is None or ref_avg is None or name == "poolforge" else avg - ref_avg
        label = "PoolForge (full)" if name == "poolforge" else ("LfF" if name == "lff" else EXPERIMENTS[name]["label"].replace("PoolForge, ", ""))
        dtext = "" if delta is None else f"{delta:+.1f}"
        rows.append(("\\rowcolor{ours}" if name == "poolforge" else "") + " & ".join(
            [label] + [cell(v, None) for v in vals] + [cell(avg, None), dtext.replace("-", "$-$")]) + " \\\\")
        md.append(f"| {label} | " + " | ".join("–" if v is None else f"{v:.1f}" for v in vals) +
                  f" | {'–' if avg is None else f'{avg:.1f}'} | {dtext} |")
    open(os.path.join(out, "ablation.tex"), "w").write(
        "\\begin{tabular}{p{4.2cm}ccccc}\n\\toprule\nVariant & CMN. & C-CIF. & BFF. & Avg. & $\\Delta$ \\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    lines += md
    return "\n".join(lines) + "\n"


def write_datasets(out):
    rows = dataset_rows()
    json.dump(rows, open(os.path.join(out, "datasets.json"), "w"), indent=1)
    body = []
    for r in rows:
        name = r["benchmark"].replace("cmnist", "CMNIST").replace("cifar10c", "C-CIFAR10").replace("bffhq", "BFFHQ").replace("_", " ").replace("pct", "\\%")
        body.append(f"{name} & {r['train'][0]:,} & {r['train'][1]:,} & {r['valid'][0]:,} & {r['valid'][1]:,} & {r['test'][0]:,} & {r['test'][1]:,} \\\\")
    open(os.path.join(out, "datasets.tex"), "w").write(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\nBenchmark & Train & Train conf. & Valid & Valid conf. & Test & Test conf. \\\\\n\\midrule\n" +
        "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    p = argparse.ArgumentParser(description="Evaluate finished runs and write the result tables.")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    device = torch.device("cuda")
    table = collect(device)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    json.dump(table, open(os.path.join(args.out, "all_runs.json"), "w"), indent=1)
    md = write_tables(table, os.path.join(args.out, "tables"))
    write_datasets(os.path.join(args.out, "tables"))
    macros = []
    for name, tag in (("poolforge", "PF"), ("lff", "LFF")):
        means = [mean_std(table, name, b)[0] for b in MAIN]
        macros.append(f"\\def\\{tag}avg{{{np.mean(means):.1f}}}" if all(m is not None for m in means) else f"\\def\\{tag}avg{{--}}")
    open(os.path.join(args.out, "tables", "numbers.tex"), "w").write("\n".join(macros) + "\n")
    open(os.path.join(args.out, "RESULTS.md"), "w").write("# Results\n\nGenerated by `python results.py` from `runs/`.\n" + md)
    print(md)


if __name__ == "__main__":
    main()
