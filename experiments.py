import os

REPO = os.path.dirname(os.path.abspath(__file__))
MAIN = ("cmnist_1pct", "cifar10c_1pct", "bffhq_0.5pct")
RATIOS = ("cmnist_0.5pct", "cmnist_2pct", "cmnist_5pct", "cifar10c_0.5pct", "cifar10c_2pct", "cifar10c_5pct")
JTT = {"cmnist_1pct": ["--jtt_epochs", "5", "--jtt_up", "100"], "cifar10c_1pct": ["--jtt_epochs", "5", "--jtt_up", "50"],
       "bffhq_0.5pct": ["--jtt_epochs", "1", "--jtt_up", "50"]}


def exp(group, label, method, flags=(), plan=None, per_benchmark=None):
    return dict(group=group, label=label, method=method, flags=list(flags), plan=plan or {b: (0,) for b in MAIN},
                per_benchmark=per_benchmark or {})


THREE_SEEDS = {b: (0, 1, 2) for b in MAIN}
EXPERIMENTS = {
    "vanilla": exp("baselines", "Vanilla (ERM)", "vanilla"),
    "lff": exp("baselines", "LfF", "lff", plan={**THREE_SEEDS, **{b: (0,) for b in RATIOS}}),
    "jtt": exp("baselines", "JTT", "jtt", per_benchmark=JTT),
    "lc": exp("baselines", "LC (logit correction + group mixup)", "lc"),
    "ldd": exp("baselines", "DisEnt / LDD", "ldd"),
    "poolforge": exp("ours", "PoolForge (ours)", "poolforge", plan={**THREE_SEEDS, **{b: (0,) for b in RATIOS}}),
}
for _name, _flags, _label in [
        ("no_reweight", ["--no_reweight"], "without LfF reweighting"),
        ("repeat", ["--mix", "repeat"], "pool images repeated instead of mixed"),
        ("no_gen", ["--gen", "none"], "without pool generator"),
        ("gen_color", ["--gen", "color"], "color-only generator (no learned filters and noise)"),
        ("gen_untargeted", ["--gen_untargeted"], "untargeted generator"),
        ("gen_batch", ["--gen_src", "batch"], "generator on batch images"),
        ("no_pgd", ["--pgd_weight", "0"], "without pool PGD"),
        ("pgd_batch", ["--pgd_src", "batch"], "PGD on batch images (BiasAdv)")]:
    EXPERIMENTS[f"poolforge_{_name}"] = exp("ablations", f"PoolForge, {_label}", "poolforge", _flags)
EXPERIMENTS["case_study"] = exp("case_study", "PoolForge, saving intermediate images", "poolforge",
                                ["--dump_at", "0.25", "0.5", "1.0"], per_benchmark={"bffhq_0.5pct": ["--dump_n", "128"]})


def run_dir(name, benchmark, seed):
    return os.path.join(REPO, "runs", name, benchmark, f"seed{seed}")


def flags_for(name, benchmark):
    spec = EXPERIMENTS[name]
    return ["--method", spec["method"]] + spec["flags"] + spec["per_benchmark"].get(benchmark, [])


def jobs(names, benchmarks=None, seeds=None):
    out = []
    for name in names:
        for benchmark, plan_seeds in EXPERIMENTS[name]["plan"].items():
            if benchmarks and benchmark not in benchmarks:
                continue
            out += [(name, benchmark, s) for s in (seeds if seeds is not None else plan_seeds)]
    return out
