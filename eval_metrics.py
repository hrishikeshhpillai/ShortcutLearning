"""Shared evaluation for all baselines (LfF, LDD, DFR).

Each run writes logs/<baseline>/<benchmark>/seed<k>/test_predictions.npz with
    y             target label per test sample
    bias          bias label per test sample
    pred          predicted label per test sample
    aligned_bias  bias value aligned with each class y: argmax_b P(y | b) in the training set
and optionally baseline, benchmark, seed, best_step, selection_split, selection_conflicting_acc
(baseline/benchmark/seed are otherwise read from the path).

Reported metric: bias-conflicting accuracy = mean accuracy over the (label, bias) cells whose
bias differs from the class's aligned bias. Cells are averaged with equal weight, so large
conflicting cells do not dominate small ones.

Usage:
    python eval_metrics.py logs/*/*/seed*/test_predictions.npz --out results/results.csv --summary
"""

import argparse
import csv
import os
import re

import numpy as np

COLUMNS = [
    "baseline", "benchmark", "seed", "selection_split", "best_step",
    "selection_conflicting_acc", "test_conflicting_acc", "n_conflicting_test", "path",
]


def cell_accuracies(y, bias, pred, num_classes, num_biases):
    correct = np.zeros((num_classes, num_biases))
    count = np.zeros((num_classes, num_biases))
    np.add.at(correct, (y, bias), (pred == y).astype(float))
    np.add.at(count, (y, bias), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return correct / count, count


def conflicting_accuracy(y, bias, pred, aligned_bias):
    num_classes = max(y.max(), pred.max(), len(aligned_bias) - 1) + 1
    num_biases = max(bias.max(), aligned_bias.max()) + 1
    accs, count = cell_accuracies(y, bias, pred, num_classes, num_biases)

    conflicting = np.ones_like(accs, dtype=bool)
    conflicting[np.arange(len(aligned_bias)), aligned_bias] = False
    conflicting &= count > 0
    return accs[conflicting].mean(), accs, count, conflicting


def run_info(path, result):
    """baseline/benchmark/seed from the file, falling back to logs/<baseline>/<benchmark>/seed<k>/."""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    match = re.fullmatch(r"seed(\d+)", parts[-2]) if len(parts) >= 4 else None
    info = {
        "baseline": parts[-4] if match else "",
        "benchmark": parts[-3] if match else "",
        "seed": int(match.group(1)) if match else "",
    }
    for key in ("baseline", "benchmark", "seed", "selection_split", "best_step", "selection_conflicting_acc"):
        if key in result.files:
            info[key] = result[key].item()
    return info


def evaluate_file(path):
    result = np.load(path)
    acc, accs, count, conflicting = conflicting_accuracy(
        result["y"].astype(int),
        result["bias"].astype(int),
        result["pred"].astype(int),
        result["aligned_bias"].astype(int),
    )
    row = {key: "" for key in COLUMNS}
    row.update(run_info(path, result))
    row["test_conflicting_acc"] = 100 * acc
    row["n_conflicting_test"] = int(count[conflicting].sum())
    row["path"] = path
    if row["selection_conflicting_acc"] != "":
        row["selection_conflicting_acc"] = 100 * row["selection_conflicting_acc"]
    return row, accs


def print_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["baseline"], row["benchmark"]), []).append(row["test_conflicting_acc"])
    print(f"\n{'baseline':10s} {'benchmark':16s} {'seeds':>5s}  test conflicting acc")
    for (baseline, benchmark), accs in sorted(groups.items()):
        print(f"{baseline:10s} {benchmark:16s} {len(accs):5d}  {np.mean(accs):6.2f} ± {np.std(accs):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", nargs="+")
    parser.add_argument("--out", help="write one row per run to this csv (overwritten)")
    parser.add_argument("--summary", action="store_true", help="print mean ± std over seeds")
    parser.add_argument("--show_cells", action="store_true", help="print the per-cell accuracy matrix")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.prediction_files):
        row, accs = evaluate_file(path)
        rows.append(row)
        print(f"{row['baseline']}/{row['benchmark']}/seed{row['seed']}: "
              f"test conflicting acc = {row['test_conflicting_acc']:.2f} "
              f"({row['n_conflicting_test']} samples, selected on {row['selection_split'] or '?'} "
              f"at step {row['best_step']})")
        if args.show_cells:
            print("cell accuracy (rows: label, cols: bias):")
            print(np.round(100 * accs, 1))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()
                })
        print(f"wrote {len(rows)} rows to {args.out}")

    if args.summary:
        print_summary(rows)


if __name__ == "__main__":
    main()
