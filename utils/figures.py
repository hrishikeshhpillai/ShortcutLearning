import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from utils.evaluate import history, metrics
from utils.tables import LADDER, SEEDS, values

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b4b3ad"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
COLORS = {"ours": BLUE, "lff": ORANGE, "vanilla": AQUA, "ours_random_direction": AQUA, "ours_eq9": YELLOW}
LABELS = {"ours": "Ours", "lff": "LfF", "vanilla": "Vanilla", "ours_random_direction": "Random direction",
          "ours_eq9": "FMD Eq. 9 direction"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def top_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: outputs/figures/{name}.pdf")


def ratio_figure(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.2))
    ratios = (0.5, 1, 2, 5)
    for ax, (family, title) in zip(axes, (("cmnist", "Colored MNIST"), ("cifar10c", "Corrupted CIFAR-10"))):
        for m in ("vanilla", "lff", "ours"):
            stats = [values(predictions, m, f"{family}_{r}pct", "conflicting") for r in ("0.5", "1", "2", "5")]
            x = [r for r, v in zip(ratios, stats) if len(v)]
            mean = np.array([v.mean() for v in stats if len(v)])
            std = np.array([v.std() for v in stats if len(v)])
            ax.fill_between(x, mean - std, mean + std, color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=COLORS[m], linewidth=2, marker="o", markersize=4, label=LABELS[m],
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ratios)
        ax.set_xticklabels(["0.5%", "1%", "2%", "5%"])
        ax.set_xlim(0.42, 6)
        ax.set_xlabel("Share of bias-conflicting training samples")
        ax.set_title(title)
        style(ax)
    axes[0].set_ylabel("Test bias-conflicting acc. (%)")
    top_legend(fig, axes[0])
    save(fig, "ratio")


def curves_figure():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.2))
    per_epoch = {"cmnist_1pct": 256 / 55000, "cifar10c_1pct": 256 / 44969}
    for ax, (benchmark, title) in zip(axes, (("cmnist_1pct", "Colored MNIST"), ("cifar10c_1pct", "Corrupted CIFAR-10"))):
        for m in ("lff", "ours_eq9", "ours_random_direction", "ours"):
            h = history(m, benchmark, 0)
            if h is None:
                continue
            epoch = h["step"] * per_epoch[benchmark]
            y = 100 * np.convolve(h["test_conflicting"], np.ones(5) / 5, mode="same")
            ax.plot(epoch[2:-2], y[2:-2], color=COLORS[m], linewidth=1.5 if m != "ours" else 2, label=LABELS[m])
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        style(ax)
    axes[0].set_ylabel("Test bias-conflicting acc. (%)")
    top_legend(fig, axes[0])
    save(fig, "curves")


def groups_figure(predictions):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35))
    for ax, m in zip(axes, ("vanilla", "lff", "ours")):
        p = predictions.get(m, "cmnist_1pct", 0)
        cells = 100 * metrics(p)["cells"]
        im = ax.imshow(cells, cmap=SEQUENTIAL, vmin=0, vmax=100)
        aligned = p["aligned_bias"]
        ax.scatter(aligned, np.arange(len(aligned)), s=6, color="white", edgecolors=INK, linewidths=0.4)
        ax.set_title(f"{LABELS[m]}: worst group {cells.min():.0f}%")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Color (bias)")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Digit (label)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Test accuracy (%)")
    cbar.outline.set_visible(False)
    save(fig, "groups")


def ladder_figure(predictions):
    from utils.evaluate import selection_score
    from utils.data import MAIN

    names, deltas, kept_flags = [], [], []
    kept = "lff"
    labels = {"ours_eq9": "In-training step, Eq. 9", "ours": "Favored pairing (ours)", "ours_no_hessian": "Remove Hessian",
              "ours+be": "BiasEnsemble", "ours+selecmix": "SelecMix", "ours+be+selecmix": "BiasEnsemble + SelecMix",
              "ldd+ours": "LDD feature swap", "ours+balancing": "Pseudo-group balancing",
              "groupdro+ours": "GroupDRO objective", "ours+adaptive": "Adaptive step",
              "ours+adaptive_r0.5": "Adaptive step (r = 0.5)", "ours+safeguard": "Safeguard"}
    for m, how in LADDER[1:]:
        d = [selection_score(predictions, m, b) - selection_score(predictions, kept, b) for b in MAIN
             if selection_score(predictions, m, b) is not None and selection_score(predictions, kept, b) is not None]
        is_kept = how != "remove" and np.mean(d) > 0 and sum(x > 0 for x in d) >= len(d) / 2
        names.append(labels[m])
        deltas.append(np.mean(d))
        kept_flags.append(is_kept)
        if is_kept:
            kept = m
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    y = np.arange(len(names))[::-1]
    ax.barh(y, deltas, height=0.62, color=[BLUE if k else NEUTRAL for k in kept_flags], edgecolor="white", linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for yi, d, k in zip(y, deltas, kept_flags):
        ax.annotate(f"{d:+.1f}" + (" kept" if k else ""), (d, yi), xytext=(3 if d >= 0 else -3, 0),
                    textcoords="offset points", ha="left" if d >= 0 else "right", va="center", fontsize=6.5, color=INK)
    ax.set_xlabel("Mean Δ selection score vs. kept method (points)")
    ax.set_xlim(min(deltas) - 6, max(deltas) + 6)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "ladder")


def evidence_figure(results):
    methods = ["vanilla", "jtt", "lntl", "ldd", "lff", "fmd", "groupdro", "ours"]
    names = {"vanilla": "Vanilla", "jtt": "JTT", "lntl": "LNTL", "ldd": "LDD", "lff": "LfF", "fmd": "FMD",
             "groupdro": "GroupDRO", "ours": "Ours"}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1))
    panels = ((axes[0], "invariant", "cmnist_1pct", "CMNIST: color-invariant predictions (%) ↑", None),
              (axes[1], "probe", "cifar10c_1pct", "C-CIFAR10: bias probe accuracy (%) ↓", 10))
    for ax, key, benchmark, title, chance in panels:
        rows = [(names[m], results[f"{m}|{benchmark}"][key]) for m in methods if f"{m}|{benchmark}" in results]
        y = np.arange(len(rows))[::-1]
        means = [np.mean(v) for _, v in rows]
        ax.barh(y, means, height=0.62, color=[BLUE if n == "Ours" else NEUTRAL for n, _ in rows], edgecolor="white")
        for yi, mval in zip(y, means):
            ax.annotate(f"{mval:.1f}", (mval, yi), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5)
        if chance:
            ax.axvline(chance, color=MUTED, linewidth=0.8, linestyle="--")
            ax.annotate("chance", (chance, y[-1] - 0.45), xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                        color=MUTED, va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in rows])
        ax.set_title(title)
        ax.set_xlim(0, max(means) * 1.15)
        style(ax)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "evidence")


def all_figures(predictions, results=None):
    ratio_figure(predictions)
    curves_figure()
    groups_figure(predictions)
    ladder_figure(predictions)
    if results:
        evidence_figure(results)
