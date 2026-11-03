# PoolForge

PoolForge is a debiasing method that uses no bias labels during training. It extends
Learning from Failure (LfF, NeurIPS 2020) with:

- a **memory pool** of the training images that the biased model keeps failing on;
- three sources of extra training images built from that pool:
  - same-class **pool mixup**;
  - a learned **pool generator** whose edits are judged by the biased model;
  - **pool PGD**.

This repository has the method, five baselines and all ablations in one code base. It also has the scripts that produce
every number and figure in the report.

**Report:** [report/report.pdf](report/report.pdf) covers method, datasets, baselines, setup, results, ablations,
case study and reproduction.

> **Status (3 Nov 2026):** intermediate release. Results below are final for every finished run. Still running: LDD on
> BFFHQ and five ablation runs (untargeted and batch generator on BFFHQ, no pool PGD on C-CIFAR10 and BFFHQ, PGD on
> batch images); they are marked "–" in the tables.

![PoolForge training step](report/figures/flowchart.png)

## Results

Test accuracy on bias-conflicting images (%), averaged over (class, bias) groups. All methods use seed 0. All models are trained from scratch with the same code, backbone
and selection rule. Full tables, including the bias ratios and ablations, are in [results/RESULTS.md](results/RESULTS.md).

| Method | CMNIST | C-CIFAR10 | BFFHQ | Avg. |
|---|---|---|---|---|
| Vanilla | 32.3 | 8.3 | 55.0 | 31.9 |
| LfF | 73.3 | 39.2 | 67.8 | 60.1 |
| JTT | 63.0 | 18.5 | 55.2 | 45.6 |
| LC | 46.3 | **41.4** | 65.2 | 51.0 |
| LDD | 75.6 | 18.5 | – | – |
| **PoolForge** | **84.3** | 40.2 | **70.0** | **64.8** |

BFFHQ has no bias-conflicting validation images, so every method selects its BFFHQ checkpoint on the test set.

## Setup

```bash
pip install -r requirements.txt   # Python 3.10, PyTorch 2.5.1, one CUDA GPU
python prepare_data.py            # downloads the DisEnt release (CMNIST, C-CIFAR10, BFFHQ) into data/
```

`prepare_data.py` downloads the dataset release of [DisEnt](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled)
from Google Drive, unzips it into `data/` and caches every split as `data/cache/*.npz`. You can also place the unzipped
`cmnist/`, `cifar10c/` and `bffhq/` folders in `data/` yourself.

## Reproduce everything

```bash
python run.py --group all --gpus 0,1 --per_gpu 2   # every training run in the report -> runs/
python run.py --exp case_study --gpus 0             # PoolForge runs that save intermediate images
python results.py                                    # evaluate checkpoints, write results/RESULTS.md and LaTeX tables
python case_study.py                                 # case-study figures and statistics -> results/case_study/
bash report/build.sh                                 # rebuild report/report.pdf
```

- `run.py` skips finished runs, so it can be stopped and restarted.
- `--list` shows what is done. `--group baselines|ours|ablations|case_study` or `--exp NAME [NAME ...]` select a subset,
  and `--benchmark` and `--seed` narrow it further.
- One run alone: `python run.py --exp poolforge --benchmark cmnist_1pct --seed 0 --gpus 0`.

Training is deterministic: `torch.use_deterministic_algorithms(True)`, deterministic cuDNN and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Rerunning a run on the same hardware and software gives identical logs.

Measured wall-clock time of the seed-0 runs, with up to nine runs sharing two GPUs (so these are upper bounds):

| Benchmark | LfF | PoolForge |
|---|---|---|
| CMNIST | 9 min | 47 min |
| C-CIFAR10 | 51 min | 7.0 h |
| BFFHQ | 51 min | 6.6 h |

## Layout

```
poolforge/
  data.py            dataset loading, caching, augmentation
  networks.py        MLP (CMNIST), ResNet-20 (C-CIFAR10), ResNet-18 (BFFHQ), DisEnt network
  metrics.py         group accuracies: bias-conflicting, unbiased, worst-group
  trainer.py         training loop, evaluation, checkpoint selection, all hyperparameters
  methods/
    base.py          Vanilla (ERM)
    lff.py           LfF
    jtt.py           JTT
    lc.py            Logit Correction
    ldd.py           DisEnt / LDD
    poolforge.py     PoolForge: memory pool, pool mixup, pool generator, pool PGD
experiments.py       every experiment of the report (baselines, PoolForge, ablations, case study)
run.py               multi-GPU launcher
results.py           evaluation and result tables
case_study.py        saved-image analysis, bias probe, best/worst test groups
prepare_data.py      dataset download
report/              LaTeX source and PDF of the report
results/             tables, per-run metrics and case-study figures produced by the scripts above
```

Each run folder, `runs/<experiment>/<benchmark>/seed<k>/`, contains:

- `history.csv`: all metrics at every evaluation step;
- `result.json`: the score at the selected step;
- `best.pt`: the selected checkpoint;
- `train.log`.

## Benchmarks

| Benchmark | Target | Bias | Bias ratios | Backbone |
|---|---|---|---|---|
| Colored MNIST | digit | digit color | 0.5, 1, 2, 5% | 3-layer MLP |
| Corrupted CIFAR-10 | object | corruption type | 0.5, 1, 2, 5% | ResNet-20 |
| BFFHQ | age | gender | 0.5% | ResNet-18 |

## Credits

PoolForge reuses or modifies parts of earlier methods:

- *Learning from Failure* (LfF, NeurIPS 2020): reweighting, unchanged.
- *mixup* (ICLR 2018): pool mixup.
- *Fighting Fire with Fire* (CDvG, ICML 2023): the idea behind the pool generator.
- *BiasAdv* (CVPR 2023): pool PGD.

The baselines are:

- *Just Train Twice* (JTT, ICML 2021);
- *Avoiding Spurious Correlations via Logit Correction* (LC, ICLR 2023);
- *Learning Debiased Representation via Disentangled Feature Augmentation* (DisEnt, NeurIPS 2021).

The datasets are the DisEnt release. Full references are in the report.
