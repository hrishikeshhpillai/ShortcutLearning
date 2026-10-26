import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ("cmnist_1pct", "cifar10c_1pct", "celeba", "bffhq_0.5pct")
CMNIST_RATIOS = ("cmnist_0.5pct", "cmnist_2pct", "cmnist_5pct")
CIFAR_RATIOS = ("cifar10c_0.5pct", "cifar10c_2pct", "cifar10c_5pct")
S3, S0 = (0, 1, 2), (0,)
SPARSE = {"cmnist_1pct": S3, "cifar10c_1pct": S3, "celeba": S0, "bffhq_0.5pct": S0}
OURS = ["--unlearn_step", "0.2"]
SHORT = {"bffhq_0.5pct": ["--num_steps", "12000"], "celeba": ["--num_steps", "25440"]}
JTT = {"cmnist_1pct": ["--jtt_epochs", "5", "--jtt_up", "100"], "cifar10c_1pct": ["--jtt_epochs", "5", "--jtt_up", "50"],
       "celeba": ["--jtt_epochs", "1", "--jtt_up", "50"], "bffhq_0.5pct": ["--jtt_epochs", "1", "--jtt_up", "50"]}


def plan(seeds, benchmarks=MAIN, **more):
    out = {b: seeds for b in benchmarks}
    out.update(more)
    return out


def run(group, kind, label, method, flags=(), extra=None, seeds=None, base=None):
    return dict(group=group, kind=kind, label=label, method=method, flags=list(flags), extra=extra or {},
                plan=seeds, base=base)


COMPONENTS = {"adaptive": ["--unlearn_adaptive"], "safeguard": ["--unlearn_safeguard"],
              "be": ["--be_committee", "5"], "selecmix": ["--selecmix", "0.1"],
              "balancing": ["--be_committee", "5", "--balance_pseudo"]}
COMPONENT_LABELS = {"adaptive": "adaptive step", "safeguard": "safeguard", "be": "BiasEnsemble",
                    "selecmix": "SelecMix", "balancing": "pseudo-group balancing"}
PAIRS = [("adaptive", "safeguard"), ("adaptive", "be"), ("adaptive", "selecmix"), ("adaptive", "balancing"),
         ("safeguard", "be"), ("safeguard", "selecmix"), ("safeguard", "balancing"), ("be", "selecmix"),
         ("selecmix", "balancing")]

RUNS = {
    "vanilla": run("baselines", "baseline", "Vanilla (ERM)", "vanilla",
                   seeds=plan(S3, **{b: S3 for b in CMNIST_RATIOS}, **{b: S0 for b in CIFAR_RATIOS})),
    "jtt": run("baselines", "baseline", "JTT", "jtt", extra=JTT, seeds=plan(S3)),
    "lntl": run("baselines", "baseline", "LNTL", "lntl", seeds=plan(S3)),
    "groupdro": run("baselines", "baseline", "GroupDRO", "groupdro", ["--dro_adjust", "2", "--dro_balanced"], seeds=plan(S3)),
    "ldd": run("baselines", "baseline", "LDD", "ldd", seeds=plan(S3, ("cmnist_1pct", "cifar10c_1pct", "bffhq_0.5pct"))),
    "lff": run("baselines", "baseline", "LfF", "lff", seeds=plan(S3, MAIN + CMNIST_RATIOS + CIFAR_RATIOS)),
    "fmd": run("baselines", "baseline", "FMD", "posthoc", seeds=plan(S3), base="vanilla"),
    "ours+be": run("baselines", "alternative", "Ours + BiasEnsemble", "lff", OURS + COMPONENTS["be"], SHORT, plan(S0)),
    "ours+selecmix": run("baselines", "alternative", "Ours + SelecMix", "lff", OURS + COMPONENTS["selecmix"],
                         {"bffhq_0.5pct": SHORT["bffhq_0.5pct"]}, plan(S0)),
    "ours+balancing": run("baselines", "alternative", "Ours + pseudo-group balancing", "lff",
                          OURS + COMPONENTS["balancing"] + ["--warmup", "0.02"], seeds=plan(S0)),
    "ours+adaptive": run("baselines", "alternative", "Ours + adaptive step", "lff", OURS + COMPONENTS["adaptive"],
                         seeds=plan(S0)),
    "ours+adaptive_r0.5": run("baselines", "alternative", "Ours + adaptive step (r = 0.5)", "lff",
                              ["--unlearn_step", "0.5", "--unlearn_adaptive"], seeds=plan(S0)),
    "ours+safeguard": run("baselines", "alternative", "Ours + safeguard", "lff", OURS + COMPONENTS["safeguard"],
                          seeds=plan(S0)),
    "groupdro+ours": run("baselines", "alternative", "GroupDRO + ours", "groupdro",
                         ["--dro_adjust", "2", "--dro_balanced"] + OURS + ["--warmup", "0.01"], seeds=plan(S0)),
    "ldd+ours": run("baselines", "alternative", "LDD + ours", "ldd", OURS, seeds=plan(S0, ("cmnist_1pct",))),
    "ldd+be": run("baselines", "alternative", "LDD + BiasEnsemble", "ldd", COMPONENTS["be"],
                  seeds=plan(S0, ("cmnist_1pct",))),
    "ldd+be+ours": run("baselines", "alternative", "LDD + BiasEnsemble + ours", "ldd", OURS + COMPONENTS["be"],
                       seeds=plan(S0, ("cmnist_1pct",))),
    "ours": run("proposed", "full", "Ours", "lff", OURS, seeds=plan(S3, MAIN + CMNIST_RATIOS + CIFAR_RATIOS)),
    "ours_eq9": run("proposed", "ablation", "FMD's Eq. 9 direction instead of favored pairing", "lff",
                    OURS + ["--unlearn_pairing", "eq9"], seeds=SPARSE),
    "ours_no_hessian": run("proposed", "ablation", "No Hessian (gradient direction)", "lff",
                           OURS + ["--unlearn_hessian", "identity"], seeds=SPARSE),
    "ours_random_direction": run("proposed", "ablation", "Random direction of the same length", "lff",
                                 OURS + ["--unlearn_random"], seeds=plan(S3)),
    "ours_r0.05": run("proposed", "ablation", "Step r = 0.05", "lff", ["--unlearn_step", "0.05"],
                      seeds=plan(S3, ("cmnist_1pct", "cifar10c_1pct", "celeba"))),
    "ours_r0.1": run("proposed", "ablation", "Step r = 0.1", "lff", ["--unlearn_step", "0.1"],
                     seeds={"celeba": S0, "bffhq_0.5pct": S3}),
    "ours_r0.5": run("proposed", "ablation", "Step r = 0.5", "lff", ["--unlearn_step", "0.5"],
                     seeds=plan(S3, ("cmnist_1pct", "cifar10c_1pct"))),
    "ours_k50": run("proposed", "ablation", "k = 50 pairs", "lff", OURS + ["--pairs_k", "50"],
                    seeds=plan(S3, ("cmnist_1pct", "cifar10c_1pct"))),
    "ours_posthoc": run("proposed", "ablation", "Post-hoc step after LfF (naive combination)", "posthoc",
                        ["--unlearn_pairing", "favored"], seeds=plan(S3), base="lff"),
}
for _a, _b in PAIRS:
    RUNS[f"ours+{_a}+{_b}"] = run("baselines", "alternative", f"Ours + {COMPONENT_LABELS[_a]} + {COMPONENT_LABELS[_b]}",
                                  "lff", OURS + COMPONENTS[_a] + COMPONENTS[_b], {"bffhq_0.5pct": SHORT["bffhq_0.5pct"]},
                                  plan(S0))


def checkpoint_dir(name, benchmark, seed):
    return os.path.join(REPO, "models", RUNS[name]["group"], "checkpoints", name, benchmark, f"seed{seed}")


def flags_for(name, benchmark):
    spec = RUNS[name]
    return ["--method", spec["method"]] + spec["flags"] + spec["extra"].get(benchmark, [])


def jobs(names=None, benchmarks=None, seeds=None):
    out = []
    for name in names or RUNS:
        for benchmark, planned in RUNS[name]["plan"].items():
            if benchmarks and benchmark not in benchmarks:
                continue
            out += [(name, benchmark, s) for s in planned if seeds is None or s in seeds]
    return out
