import csv
import os

import torch
import torch.nn.functional as F

from models.networks import build_model
from utils.data import BENCHMARKS, load_data
from utils.metrics import split_scores
from utils.newton import body_features, ce_gradients, ce_hessian, counterfactual_bias, with_bias
from utils.pairs import DEFAULT_K, build_pairs

STEPS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
DAMPINGS = (1e-3, 1e-2, 1e-1, 1.0)
SELECTION_SPLIT = {"bffhq_0.5pct": "test"}


def posthoc(base_checkpoint, benchmark, seed, out, pairing="favored"):
    device = torch.device("cuda")
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    info = BENCHMARKS[benchmark]
    data = load_data(benchmark, device)
    model = build_model(benchmark, info["num_classes"]).to(device)
    model.load_state_dict(ckpt["model"])
    aligned = ckpt["aligned_bias"]
    selection_split = SELECTION_SPLIT.get(benchmark, "valid")
    theta0 = torch.cat([model.head.weight, model.head.bias[:, None]], 1).detach().double()
    phi = {s: with_bias(body_features(model, benchmark, data[s]["x"])) for s in ("train", "valid", "test")}

    pairs = build_pairs(benchmark, DEFAULT_K[benchmark], seed, device)
    y_pairs = torch.from_numpy(pairs["y"]).long().to(device)
    phi_c = phi["train"][torch.from_numpy(pairs["c_index"]).to(device)]
    if pairs["cbar_images"] is None:
        phi_cbar = phi["train"][torch.from_numpy(pairs["cbar_index"]).to(device)]
    else:
        phi_cbar = with_bias(body_features(model, benchmark, torch.from_numpy(pairs["cbar_images"]).to(device)))
    sign = torch.ones(len(y_pairs), dtype=theta0.dtype, device=device)
    if pairing == "favored":
        loss_c = F.cross_entropy(phi_c @ theta0.T, y_pairs, reduction="none")
        loss_cbar = F.cross_entropy(phi_cbar @ theta0.T, y_pairs, reduction="none")
        sign = torch.sign(loss_cbar - loss_c)
    delta = (sign[:, None] * (ce_gradients(theta0, phi_c, y_pairs) - ce_gradients(theta0, phi_cbar, y_pairs))).mean(0)
    weights = torch.ones(len(phi["train"]), dtype=torch.float64, device=device)
    if ckpt.get("weights") is not None:
        weights = torch.as_tensor(ckpt["weights"]).double().to(device)
    h = ce_hessian(theta0, phi["train"], weights)
    eye = torch.eye(len(h), dtype=h.dtype, device=device)
    directions = {g: torch.linalg.solve(h + g * h.diagonal().mean() * eye, delta) for g in DAMPINGS}

    def scores(theta, split):
        pred = (phi[split] @ theta.T).argmax(1)
        return split_scores(data[split], pred, aligned, info["num_classes"], info["num_biases"])

    grid, best = [], None
    for damping, direction in directions.items():
        direction = direction / direction.norm()
        for r in STEPS:
            theta = theta0 + r * theta0.norm() * direction.reshape(theta0.shape)
            row = dict(damping=damping, step=r, selection_conflicting=scores(theta, selection_split)[0],
                       valid_aligned=scores(theta, "valid")[1], test_conflicting=scores(theta, "test")[0],
                       cf_bias=counterfactual_bias(theta, phi_c, phi_cbar))
            grid.append(row)
            key = (round(row["selection_conflicting"], 10), round(row["valid_aligned"], 10))
            if best is None or key > best[0]:
                best = (key, row, theta)
    _, row, theta = best
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "grid.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    state = dict(ckpt["model"])
    state["head.weight"] = theta[:, :-1].float().cpu()
    state["head.bias"] = theta[:, -1].float().cpu()
    torch.save(dict(model=state, step=ckpt["step"], weights=None, aligned_bias=aligned, benchmark=benchmark, seed=seed,
                    method="posthoc", selection_split=selection_split, selection_score=row["selection_conflicting"],
                    damping=row["damping"], r=row["step"]), os.path.join(out, "best.pt"))
    print(f"post-hoc step r={row['step']} damping={row['damping']}: {selection_split} conf "
          f"{100 * row['selection_conflicting']:.2f}", flush=True)
