import argparse
import json
import os

from utils.evaluate import Predictions

REPO = os.path.dirname(os.path.abspath(__file__))
TABLES = ["main", "ablation", "ladder", "ratio", "published", "evidence"]
FIGURES = ["ratio", "curves", "groups", "ladder", "evidence"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="*", choices=TABLES)
    p.add_argument("--figures", nargs="*", choices=FIGURES)
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    everything = args.tables is None and args.figures is None
    tables = TABLES if everything or args.tables == [] else args.tables or []
    figures = FIGURES if everything or args.figures == [] else args.figures or []

    from utils import figures as fig
    from utils import tables as tab
    from utils.evidence import evidence, evidence_tables

    predictions = Predictions(args.recompute)
    results = None
    if "evidence" in tables or "evidence" in figures:
        cache = os.path.join(REPO, "outputs", "cache", "evidence.json")
        results = json.load(open(cache)) if os.path.exists(cache) and not args.recompute else evidence(predictions)
    steps = {"main": lambda: tab.main_tables(predictions), "ablation": lambda: tab.ablation_table(predictions),
             "ladder": lambda: tab.ladder_tables(predictions), "ratio": lambda: tab.ratio_table(predictions),
             "published": tab.published_table, "evidence": lambda: evidence_tables(results)}
    for t in tables:
        steps[t]()
    plots = {"ratio": lambda: fig.ratio_figure(predictions), "curves": fig.curves_figure,
             "groups": lambda: fig.groups_figure(predictions), "ladder": lambda: fig.ladder_figure(predictions),
             "evidence": lambda: fig.evidence_figure(results)}
    for f in figures:
        plots[f]()


if __name__ == "__main__":
    main()
