import argparse
import glob
import os
import re
import time

from experiments import EXPERIMENTS, REPO, flags_for, jobs, run_dir
from poolforge.methods.ldd import LDD_SETTINGS
from poolforge.trainer import SETTINGS, family, parser

LINE = re.compile(r"step\s+(\d+).*best\s+([\d.]+).*\((\d+)s\)")


def total_steps(name, benchmark):
    args = parser().parse_args(flags_for(name, benchmark))
    if args.method == "ldd":
        return LDD_SETTINGS[family(benchmark)]["num_steps"]
    return args.num_steps or SETTINGS[family(benchmark)]["num_steps"]


def fmt(seconds):
    if seconds is None:
        return "-"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def state(name, benchmark, seed):
    folder = run_dir(name, benchmark, seed)
    if os.path.exists(os.path.join(folder, "result.json")):
        return "done", None
    log = os.path.join(folder, "train.log")
    if not os.path.exists(log):
        return "queued", None
    matches = LINE.findall(open(log).read())
    if time.time() - os.path.getmtime(log) > 1800:
        return "stalled", matches[-1] if matches else None
    return "running", matches[-1] if matches else None


def main():
    p = argparse.ArgumentParser(description="Progress of all PoolForge experiment runs.")
    p.add_argument("--all", action="store_true", help="also list finished runs")
    args = p.parse_args()
    rows, counts, eta_total = [], {}, 0.0
    for name, benchmark, seed in jobs(list(EXPERIMENTS)):
        status, last = state(name, benchmark, seed)
        counts[status] = counts.get(status, 0) + 1
        total = total_steps(name, benchmark)
        if status == "done" and not args.all:
            continue
        step, best, elapsed = (int(last[0]), float(last[1]), int(last[2])) if last else (0, None, 0)
        left = (total - step) * elapsed / step if step else None
        if status == "running" and left is not None:
            eta_total = max(eta_total, left)
        rows.append((status, name, benchmark, seed, step, total, best, elapsed, left))
    order = {"running": 0, "stalled": 1, "queued": 2, "done": 3}
    rows.sort(key=lambda r: (order[r[0]], -(r[4] / r[5])))
    head = f"{'status':8} {'experiment':26} {'benchmark':15} {'seed':>4} {'step':>13} {'%':>5} {'best':>6} {'elapsed':>8} {'left':>8}"
    print(head)
    print("-" * len(head))
    for status, name, benchmark, seed, step, total, best, elapsed, left in rows:
        best_text = "-" if best is None else f"{best:.1f}"
        print(f"{status:8} {name:26} {benchmark:15} {seed:>4} {step:>6}/{total:<6} {100 * step / total:>5.1f} "
              f"{best_text:>6} {fmt(elapsed):>8} {fmt(left):>8}")
    print("-" * len(head))
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: order[kv[0]])) +
          f"   longest running run ends in {fmt(eta_total)}")
    typical = {"cmnist": 37 * 60, "cifar10c": 105 * 60, "bffhq": 50 * 60}
    queued = [r for r in rows if r[0] == "queued"]
    work = sum(typical[r[2].split("_")[0]] * (50000 / 39200 if r[1] == "ldd" else 1) for r in queued)
    print(f"queued work: {fmt(work)} of single-run time; with 4 runs in parallel about {fmt(work / 4 + eta_total)} to finish all")
    print("'best' = best validation bias-conflicting accuracy so far (BFFHQ: test).")


if __name__ == "__main__":
    main()
