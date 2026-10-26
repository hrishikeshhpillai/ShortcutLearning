import argparse
import os
import subprocess
import sys
import threading
import time

from models.registry import RUNS, checkpoint_dir, flags_for, jobs

REPO = os.path.dirname(os.path.abspath(__file__))


def run_one(name, benchmark, seed):
    from models.baselines.fmd import posthoc
    from utils.trainer import parser, train

    out = checkpoint_dir(name, benchmark, seed)
    args = parser().parse_args(flags_for(name, benchmark))
    if args.method == "posthoc":
        base = os.path.join(checkpoint_dir(RUNS[name]["base"], benchmark, seed), "best.pt")
        posthoc(base, benchmark, seed, out, args.unlearn_pairing)
    else:
        train(args, benchmark, seed, out)


def launch(todo, gpus, per_gpu):
    pending, lock, failed = list(todo), threading.Lock(), []

    def worker(gpu):
        while True:
            with lock:
                if not pending:
                    return
                name, benchmark, seed = pending.pop(0)
            out = checkpoint_dir(name, benchmark, seed)
            os.makedirs(out, exist_ok=True)
            print(f"[{time.strftime('%H:%M:%S')}] gpu {gpu}: {name} {benchmark} seed {seed}", flush=True)
            with open(os.path.join(out, "train.log"), "w") as log:
                code = subprocess.call([sys.executable, __file__, "--run", name, "--benchmark", benchmark,
                                        "--seed", str(seed), "--single"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
            if code:
                failed.append((name, benchmark, seed))
                print(f"FAILED {name} {benchmark} seed {seed}: see {out}/train.log", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus for _ in range(per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["baselines", "proposed", "all"])
    p.add_argument("--run", nargs="+", choices=list(RUNS))
    p.add_argument("--benchmark", nargs="+")
    p.add_argument("--seed", nargs="+", type=int)
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_gpu", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--single", action="store_true")
    args = p.parse_args()
    if args.single:
        run_one(args.run[0], args.benchmark[0], args.seed[0])
        return
    names = args.run or [n for n, s in RUNS.items() if args.group in (s["group"], "all")]
    if not names:
        p.error("give --group or --run")
    if args.list:
        for j in jobs(names, args.benchmark, args.seed):
            print(*j, "trained" if os.path.exists(os.path.join(checkpoint_dir(*j), "best.pt")) else "missing")
        return
    todo = [j for j in jobs(names, args.benchmark, args.seed)
            if args.overwrite or not os.path.exists(os.path.join(checkpoint_dir(*j), "best.pt"))]
    first = [j for j in todo if RUNS[j[0]]["method"] != "posthoc"]
    second = [j for j in todo if RUNS[j[0]]["method"] == "posthoc"]
    for name, benchmark, seed in second:
        base = (RUNS[name]["base"], benchmark, seed)
        if not os.path.exists(os.path.join(checkpoint_dir(*base), "best.pt")) and base not in first:
            first.append(base)
    first.sort(key=lambda j: j[1] != "bffhq_0.5pct")
    failed = launch(first, args.gpus.split(","), args.per_gpu) + launch(second, args.gpus.split(","), args.per_gpu)
    print(f"done: {len(todo) - len(failed)} trained, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
