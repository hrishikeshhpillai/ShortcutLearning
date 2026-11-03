import argparse
import csv
import glob
import os

import torch

from experiments import EXPERIMENTS, REPO
from results import test_predictions


def relative(path):
    path = str(path).replace(os.sep, "/")
    return path.split("/data/", 1)[-1]


def write(folder, out, device):
    logits, test, aligned = test_predictions(folder, device)
    prob = logits.softmax(1)
    pred = prob.argmax(1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "file", "label", "bias", "bias_conflicting", "prediction", "confidence"])
        for i in range(len(pred)):
            y, b = int(test["y"][i]), int(test["bias"][i])
            w.writerow([i, relative(test["paths"][i]), y, b, int(b != aligned[y]), int(pred[i]), f"{prob[i, pred[i]]:.4f}"])
    return (pred.cpu() == test["y"].cpu()).float().mean().item()


def main():
    p = argparse.ArgumentParser(description="Write test-set predictions of every checkpoint in runs/ to predictions/.")
    p.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS))
    p.add_argument("--out", default=os.path.join(REPO, "predictions"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for name in args.exp or [n for n in EXPERIMENTS if EXPERIMENTS[n]["group"] != "case_study"]:
        for folder in sorted(glob.glob(os.path.join(REPO, "runs", name, "*", "seed*"))):
            if not os.path.exists(os.path.join(folder, "best.pt")):
                continue
            benchmark, seed = folder.split(os.sep)[-2:]
            out = os.path.join(args.out, name, benchmark, f"{seed}.csv")
            acc = write(folder, out, torch.device(args.device))
            print(f"{name} {benchmark} {seed}: {100 * acc:.2f}% overall test accuracy -> {os.path.relpath(out, REPO)}", flush=True)


if __name__ == "__main__":
    main()
