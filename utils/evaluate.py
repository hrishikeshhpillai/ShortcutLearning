import csv
import os

import numpy as np
import torch

from models.networks import build_model, network_for
from models.registry import RUNS, checkpoint_dir
from utils.data import BENCHMARKS, load_data, predict_logits
from utils.metrics import group_metrics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "outputs", "cache")
E40_STEPS = 12000


def load_checkpoint(name, benchmark, seed, device):
    path = os.path.join(checkpoint_dir(name, benchmark, seed), "best.pt")
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    info = BENCHMARKS[benchmark]
    method = RUNS[name]["method"]
    model = build_model(benchmark, info["num_classes"]) if method == "posthoc" else \
        network_for(method, benchmark, info["num_classes"], info["num_biases"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


class Predictions:
    def __init__(self, recompute=False):
        self.recompute, self.data, self.device = recompute, {}, torch.device("cuda")

    def split(self, benchmark):
        if benchmark not in self.data:
            self.data = {benchmark: load_data(benchmark, self.device)}
        return self.data[benchmark]

    def get(self, name, benchmark, seed):
        cache = os.path.join(CACHE, name, benchmark, f"seed{seed}.npz")
        ckpt_path = os.path.join(checkpoint_dir(name, benchmark, seed), "best.pt")
        if not os.path.exists(ckpt_path):
            return None
        if os.path.exists(cache) and not self.recompute and os.path.getmtime(cache) >= os.path.getmtime(ckpt_path):
            return dict(np.load(cache))
        model, ckpt = load_checkpoint(name, benchmark, seed, self.device)
        data = self.split(benchmark)
        out = dict(aligned_bias=ckpt["aligned_bias"], selection_split=ckpt.get("selection_split", "valid"))
        for s in ("valid", "test"):
            out[f"pred_{s}"] = predict_logits(model, benchmark, data[s]["x"]).argmax(1).cpu().numpy()
            out[f"y_{s}"], out[f"bias_{s}"] = data[s]["y"].cpu().numpy(), data[s]["bias"].cpu().numpy()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **out)
        return out


def metrics(pred, split="test"):
    if pred is None:
        return None
    info_b = int(max(pred[f"bias_{split}"].max(), pred["aligned_bias"].max())) + 1
    num_classes = len(pred["aligned_bias"])
    return group_metrics(pred[f"y_{split}"], pred[f"bias_{split}"], pred[f"pred_{split}"], pred["aligned_bias"],
                         num_classes, info_b)


def history(name, benchmark, seed):
    path = os.path.join(checkpoint_dir(name, benchmark, seed), "history.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def selection_score(predictions, name, benchmark, seed=0):
    if benchmark == "bffhq_0.5pct":
        h = history(name, benchmark, seed)
        if h is not None:
            return 100 * h["test_conflicting"][h["step"] <= E40_STEPS].max()
    p = predictions.get(name, benchmark, seed)
    if p is None:
        return None
    split = "test" if benchmark == "bffhq_0.5pct" else "valid"
    return 100 * metrics(p, split)["conflicting"]


def published_protocol(name, benchmark, seed):
    h = history(name, benchmark, seed)
    if h is None:
        return None
    if benchmark.startswith("bffhq"):
        return 100 * h["test_conflicting"].max()
    n = BENCHMARKS[benchmark]["num_biases"]
    return 100 * (((n - 1) * h["test_conflicting"] + h["test_aligned"]) / n).max()
