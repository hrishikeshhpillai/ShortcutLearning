# PoolForge

- Report: [PDF](report/report.pdf) · [LaTeX source](report/report.tex)
- Recorded results: [results/RESULTS.md](results/RESULTS.md)

Run commands from this folder. Python 3.10 is required. Training requires a CUDA GPU. The data and trained checkpoints are not included.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
```

## Run one experiment

```bash
python run.py --exp poolforge --benchmark cmnist_1pct --seed 0 --gpus 0
python progress.py
```

The checkpoint and log appear in `runs/poolforge/cmnist_1pct/seed0/`. To choose another seed, change `--seed`.

## Regenerate the report's result tables

```bash
python run.py --group all --benchmark cmnist_1pct cifar10c_1pct bffhq_0.5pct --seed 0 --gpus 0
python run.py --exp lff poolforge --benchmark cmnist_0.5pct cmnist_2pct cmnist_5pct cifar10c_0.5pct cifar10c_2pct cifar10c_5pct --seed 0 --gpus 0
python results.py --device cpu --out reproduced_results
```

See `reproduced_results/RESULTS.md` and `reproduced_results/tables/`. Existing completed runs are skipped. Change `--gpus 0` to the GPU IDs available on your machine.

## Other outputs

```bash
python predict.py --device cpu
python run.py --group case_study --seed 0 --gpus 0
python case_study.py --device cpu
```

Predictions appear in `predictions/`; case study images appear in `results/case_study/`.

To rebuild the included PDF, install `latexmk` and a TeX distribution, then run:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```
