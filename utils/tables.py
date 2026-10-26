import os
import re

import numpy as np

from models.registry import COMPONENT_LABELS, PAIRS, RUNS
from utils.data import MAIN, TITLES
from utils.evaluate import metrics, published_protocol, selection_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "tables")
SEEDS = (0, 1, 2)
SHORT_TITLES = {"cmnist_1pct": "CMNIST", "cifar10c_1pct": "C-CIFAR10", "celeba": "CelebA", "bffhq_0.5pct": "BFFHQ"}
MAIN_METHODS = [("vanilla", "none"), ("jtt", "none"), ("ldd", "none"), ("lff", "none"), ("fmd", "pairs"),
                ("lntl", "all"), ("groupdro", "all"), ("ours", "pairs")]
CITE = {"vanilla": "", "jtt": "~\\cite{liu2021jtt}", "ldd": "~\\cite{lee2021ldd}", "lff": "~\\cite{nam2020lff}",
        "fmd": "~\\cite{chen2023fmd}", "lntl": "~\\cite{kim2019lntl}", "groupdro": "~\\cite{sagawa2020groupdro}",
        "ours": ""}
ABLATION = ["lff", "ours_posthoc", "ours_eq9", "ours_no_hessian", "ours_random_direction", "ours_r0.05", "ours_r0.1",
            "ours_r0.5", "ours_k50", "ours"]
LADDER = [("lff", "base"), ("ours_eq9", "add"), ("ours", "replace"), ("ours_no_hessian", "remove"),
          ("ours+be", "add"), ("ours+selecmix", "add"), ("ours+be+selecmix", "add"), ("ldd+ours", "add"),
          ("ours+balancing", "add"), ("groupdro+ours", "add"), ("ours+adaptive", "add"), ("ours+adaptive_r0.5", "add"),
          ("ours+safeguard", "add")]
LADDER_LABELS = {"lff": "LfF (base)", "ours_eq9": "+ unlearning step in training, FMD's Eq. 9 direction",
                 "ours": "+ favored pairing instead of Eq. 9 (ours)", "ours_no_hessian": "$-$ Hessian (gradient direction)"}


class Table:
    def __init__(self, name, header, caption=""):
        self.name, self.header, self.caption, self.rows = name, header, caption, []

    def add(self, cells, bold=()):
        self.rows.append((cells, set(bold)))

    def markdown(self):
        plain = lambda c: re.sub(r"~\\cite\{[^}]*\}", "", c).replace("$-$", "−")
        lines = ["| " + " | ".join(plain(h) for h in self.header) + " |", "|" + "---|" * len(self.header)]
        for cells, bold in self.rows:
            lines.append("| " + " | ".join(f"**{plain(c)}**" if i in bold else plain(c) for i, c in enumerate(cells)) + " |")
        return "\n".join(lines) + ("\n\n" + self.caption if self.caption else "") + "\n"

    def latex(self):
        def tex(c, b):
            c = c.replace("±", "$\\pm$").replace("%", "\\%").replace("Δ", "$\\Delta$").replace("−", "$-$")
            return f"\\textbf{{{c}}}" if b else c
        lines = ["\\begin{tabular}{l" + "c" * (len(self.header) - 1) + "}", "\\toprule",
                 " & ".join(tex(h, False) for h in self.header) + " \\\\", "\\midrule"]
        for cells, bold in self.rows:
            lines.append(" & ".join(tex(c, i in bold) for i, c in enumerate(cells)) + " \\\\")
        return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT, self.name + ".md"), "w").write(self.markdown())
        open(os.path.join(OUT, self.name + ".tex"), "w").write(self.latex())
        print(self.markdown())


def values(predictions, name, benchmark, key, seeds=SEEDS):
    out = []
    for s in seeds:
        m = metrics(predictions.get(name, benchmark, s))
        if m is not None:
            out.append(100 * m[key])
    return np.array(out)


def fmt(v, n=False):
    if not len(v):
        return "–"
    text = f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 1 else f"{v.mean():.1f}"
    return text + (f" ({len(v)})" if n and len(v) != len(SEEDS) else "")


def main_tables(predictions):
    for key, name, title in (("conflicting", "main_conflicting", "Test bias-conflicting accuracy"),
                             ("worst", "main_worst_group", "Test worst-group accuracy"),
                             ("aligned", "main_aligned", "Test bias-aligned accuracy"),
                             ("unbiased", "main_unbiased", "Test unbiased accuracy")):
        t = Table(name, ["Method", "Bias labels"] + [SHORT_TITLES[b] for b in MAIN],
                  f"{title} (%), mean ± std over 3 seeds. Bold: best among methods that use no or only pair bias labels.")
        cols = {b: {m: values(predictions, m, b, key) for m, _ in MAIN_METHODS} for b in MAIN}
        best = {b: max((v.mean() for m, v in cols[b].items() if len(v) and dict(MAIN_METHODS)[m] != "all"), default=None)
                for b in MAIN}
        for m, labels in MAIN_METHODS:
            cells = [RUNS[m]["label"] + CITE[m], labels] + [fmt(cols[b][m]) for b in MAIN]
            bold = [2 + i for i, b in enumerate(MAIN) if len(cols[b][m]) and best[b] is not None
                    and abs(cols[b][m].mean() - best[b]) < 1e-9 and labels != "all"]
            t.add(cells, bold)
        t.save()


def ablation_table(predictions):
    t = Table("ablation", ["Variant"] + [SHORT_TITLES[b] for b in MAIN],
              "Test bias-conflicting accuracy (%) of ours with one part changed, mean ± std (number of seeds if not 3).")
    for m in ABLATION:
        t.add([RUNS[m]["label"] if m not in ("lff", "ours") else {"lff": "LfF (no unlearning step)", "ours": "Ours"}[m]]
              + [fmt(values(predictions, m, b, "conflicting"), n=True) for b in MAIN], bold=[] if m != "ours" else [0])
    t.save()


def ladder_tables(predictions):
    score = {(m, b): selection_score(predictions, m, b) for m, _ in LADDER for b in MAIN}
    t = Table("ladder", ["Step", *[SHORT_TITLES[b] for b in MAIN], "Mean Δ", "Decision"],
              "Selection score (%) of seed 0 (validation bias-conflicting accuracy; BFFHQ: test, within 40 epochs). "
              "A component is kept when it raises the score on average and on at least half of the benchmarks.")
    kept = None
    for m, how in LADDER:
        cells = [LADDER_LABELS.get(m, "+ " + RUNS[m]["label"].replace("Ours + ", "").replace(" + ours", ""))]
        deltas = []
        for b in MAIN:
            s = score[(m, b)]
            cells.append("–" if s is None else f"{s:.1f}")
            if s is not None and kept is not None and score[(kept, b)] is not None:
                deltas.append(s - score[(kept, b)])
        if how == "base":
            decision, mean = "base", ""
        else:
            mean = f"{np.mean(deltas):+.1f}"
            better = np.mean(deltas) > 0 and sum(d > 0 for d in deltas) >= len(deltas) / 2
            if how == "remove":
                decision = "needed" if not better else "not needed"
            elif better:
                decision = "kept"
            else:
                decision = "dropped"
        if m == "ldd+ours":
            cells[0] = "+ LDD feature swap"
        if m == "groupdro+ours":
            cells[0] = "+ GroupDRO objective"
        t.add(cells + [mean, decision], bold=[len(cells) + 1] if decision == "kept" else [])
        if how == "base" or decision == "kept":
            kept = m
    t.save()
    single = {"adaptive": "ours+adaptive", "safeguard": "ours+safeguard", "be": "ours+be", "selecmix": "ours+selecmix",
              "balancing": "ours+balancing"}
    p = Table("pairs", ["Pair added to ours", *[SHORT_TITLES[b] for b in MAIN], "Δ vs better member", "Δ vs ours"],
              "Selection score (%) of seed 0 for every pair of candidate components on top of ours.")
    for a, c in PAIRS:
        name = f"ours+{a}+{c}"
        cells, dm, dk = [f"{COMPONENT_LABELS[a]} + {COMPONENT_LABELS[c]}"], [], []
        for b in MAIN:
            s = selection_score(predictions, name, b)
            members = [selection_score(predictions, single[x], b) for x in (a, c)]
            ref = score[(kept, b)]
            cells.append("–" if s is None else f"{s:.1f}")
            if s is not None and all(v is not None for v in members):
                dm.append(s - max(members))
            if s is not None and ref is not None:
                dk.append(s - ref)
        p.add(cells + [f"{np.mean(dm):+.1f}" if dm else "–", f"{np.mean(dk):+.1f}" if dk else "–"])
    p.save()


def ratio_table(predictions):
    t = Table("ratio", ["Method", "Dataset", "0.5%", "1%", "2%", "5%"],
              "Test bias-conflicting accuracy (%) by the share of bias-conflicting training samples; worst-group "
              "accuracy in brackets. Mean over the seeds available.")
    for family, title in (("cmnist", "CMNIST"), ("cifar10c", "C-CIFAR10")):
        for m in ("vanilla", "lff", "ours"):
            cells = [RUNS[m]["label"], title]
            for r in ("0.5pct", "1pct", "2pct", "5pct"):
                c, w = values(predictions, m, f"{family}_{r}", "conflicting"), values(predictions, m, f"{family}_{r}", "worst")
                cells.append("–" if not len(c) else f"{fmt(c)} ({w.mean():.1f})")
            t.add(cells, bold=[] if m != "ours" else [0])
    t.save()


def published_table():
    published = {"LfF~\\cite{nam2020lff}": (61.89, 33.07, 62.20), "LDD~\\cite{lee2021ldd}": (81.73, 36.49, 63.87),
                 "SelecMix~\\cite{hwang2022selecmix} (best variant)": (83.55, 43.68, 71.60)}
    t = Table("published", ["Method", "Source", "CMNIST", "C-CIFAR10", "BFFHQ"],
              "Unbiased test accuracy (CMNIST, C-CIFAR10) or bias-conflicting test accuracy (BFFHQ), the metrics of "
              "the published tables. Our runs: best test checkpoint (the oracle selection SelecMix reports).")
    for name, v in published.items():
        t.add([name, "published"] + [f"{x:.2f}" for x in v])
    for m in ("lff", "ldd", "ours"):
        cells = [RUNS[m]["label"], "ours" if m == "ours" else "our run"]
        for b in ("cmnist_1pct", "cifar10c_1pct", "bffhq_0.5pct"):
            v = np.array([x for x in (published_protocol(m, b, s) for s in SEEDS) if x is not None])
            cells.append(fmt(v))
        t.add(cells)
    t.save()


def all_tables(predictions):
    main_tables(predictions)
    ablation_table(predictions)
    ladder_tables(predictions)
    ratio_table(predictions)
    published_table()
