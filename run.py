import argparse
import os
import subprocess
import sys
import threading
import time

from experiments import EXPERIMENTS, flags_for, jobs, run_dir

REPO = os.path.dirname(os.path.abspath(__file__))
NEEDS_MIB = {"bffhq": 11000}


def free_mib(gpu):
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip())


def launch(todo, gpus, per_gpu):
    pending, lock, failed = list(todo), threading.Lock(), []
    gpu_locks = {g: threading.Lock() for g in gpus}
    need = lambda j: NEEDS_MIB.get(j[1].split("_")[0], 3000)

    def worker(gpu):
        while True:
            with gpu_locks[gpu]:
                with lock:
                    if not pending:
                        return
                    fits = [j for j in pending if free_mib(gpu) >= need(j)]
                    if fits:
                        pending.remove(fits[0])
                if not fits:
                    time.sleep(60)
                    continue
                name, benchmark, seed = fits[0]
                out = run_dir(name, benchmark, seed)
                os.makedirs(out, exist_ok=True)
                print(f"[{time.strftime('%H:%M:%S')}] gpu {gpu}: {name} {benchmark} seed {seed}", flush=True)
                log = open(os.path.join(out, "train.log"), "w")
                proc = subprocess.Popen([sys.executable, __file__, "--exp", name, "--benchmark", benchmark, "--seed",
                                         str(seed), "--single"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
                time.sleep(90)
            if proc.wait():
                failed.append((name, benchmark, seed))
                print(f"FAILED {name} {benchmark} seed {seed}: see {out}/train.log", flush=True)
            log.close()

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus for _ in range(per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def main():
    p = argparse.ArgumentParser(description="Train baselines, PoolForge and its ablations.")
    p.add_argument("--group", choices=["baselines", "ours", "ablations", "case_study", "all"])
    p.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS))
    p.add_argument("--benchmark", nargs="+")
    p.add_argument("--seed", nargs="+", type=int)
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_gpu", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--single", action="store_true")
    args = p.parse_args()
    if args.single:
        from poolforge.trainer import parser, train
        name, benchmark, seed = args.exp[0], args.benchmark[0], args.seed[0]
        train(parser().parse_args(flags_for(name, benchmark)), benchmark, seed, run_dir(name, benchmark, seed))
        return
    names = args.exp or [n for n, s in EXPERIMENTS.items()
                         if args.group == s["group"] or (args.group == "all" and s["group"] != "case_study")]
    if not names:
        p.error("give --group or --exp")
    done = lambda j: os.path.exists(os.path.join(run_dir(*j), "result.json"))
    if args.list:
        for j in jobs(names, args.benchmark, args.seed):
            print(*j, "done" if done(j) else "missing")
        return
    todo = [j for j in jobs(names, args.benchmark, args.seed) if args.overwrite or not done(j)]
    failed = launch(todo, args.gpus.split(","), args.per_gpu)
    print(f"done: {len(todo) - len(failed)} trained, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
