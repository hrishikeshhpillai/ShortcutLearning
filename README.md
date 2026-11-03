# PoolForge

Report: [report/report.pdf](report/report.pdf)

All commands are run from the repository root.

## Step 1. Install

```bash
pip install -r requirements.txt
```

Python 3.10 and PyTorch 2.5.1. Steps 3–5 run on a CPU; step 6 needs a CUDA GPU.

## Step 2. Download the data

```bash
python prepare_data.py
```

Expected: `data/cmnist/`, `data/cifar10c/`, `data/bffhq/` and `data/cache/*.npz`, plus one line per split, for example
`cmnist_1pct test 10000`.

## Step 3. Reproduce all tables from the shipped checkpoints (no training)

```bash
rm -rf results
python results.py --device cpu
```

Expected: `results/RESULTS.md` and `results/tables/*.tex`. Test bias-conflicting accuracy (%):

| Model | CMNIST | C-CIFAR10 | BFFHQ | Avg. |
|---|---|---|---|---|
| Vanilla | 32.3 | 8.3 | 52.6 | 31.1 |
| LfF | 73.3 | 39.2 | 64.2 | 58.9 |
| JTT | 55.3 | 14.7 | 50.2 | 40.1 |
| LC | 46.3 | 41.4 | 64.2 | 50.6 |
| LDD | 75.6 | 18.5 | 59.4 | 51.2 |
| PoolForge | 84.4 | 40.0 | 72.2 | 65.5 |
| LfF, seeds 0–2 | 70.6 ± 2.0 | 35.8 ± 3.8 | 62.3 ± 1.6 | 56.2 ± 2.1 |
| PoolForge, seeds 0–2 | 83.2 ± 1.0 | 40.7 ± 0.8 | 71.2 ± 1.1 | 65.0 ± 0.8 |

## Step 4. Write the test predictions

```bash
python predict.py --device cpu
```

Expected: 66 files `predictions/<model>/<benchmark>/seed<k>.csv` with columns
`index, file, label, bias, bias_conflicting, prediction, confidence`.

## Step 5. Rebuild the figures and the report

```bash
python case_study.py --device cpu
bash report/build.sh
```

Expected: `results/case_study/*.png` and `report/report.pdf`.

## Step 6. Retrain (optional)

`run.py` skips any run that already exists, so first move the shipped runs of the models you want to retrain:

```bash
mkdir -p runs_shipped
mv runs/vanilla runs/lff runs/jtt runs/lc runs/ldd runs_shipped/   # baselines
mv runs/poolforge runs_shipped/                                     # PoolForge
mv runs/poolforge_* runs_shipped/                                   # ablations
mv runs/case_study runs_shipped/                                    # case study
```

Then run the matching command. Put the GPU ids you have in `--gpus`.

| Command | Models retrained | Runs | Time per run (one GPU) |
|---|---|---|---|
| `python run.py --group baselines --gpus 0,1` | Vanilla, JTT, LC, LDD (seed 0); LfF (seeds 0–2, plus seed 0 at the 0.5/2/5% ratios) | 21 | 2 min – 1 h (LDD on BFFHQ: 3 h) |
| `python run.py --group ours --gpus 0,1` | PoolForge (seeds 0–2, plus seed 0 at the 0.5/2/5% ratios) | 15 | 30 min (CMNIST), 4 h (C-CIFAR10), 6 h (BFFHQ) |
| `python run.py --group ablations --gpus 0,1` | the 8 PoolForge ablations (seed 0) | 24 | as PoolForge |
| `python run.py --group case_study --gpus 0,1` | PoolForge seed 0, saving the images used in step 5 | 3 | as PoolForge |
| `python run.py --group all --gpus 0,1 --per_gpu 2` | baselines, PoolForge and ablations | 60 | |

To retrain a single run, move only its folder and name it:

```bash
mv runs/poolforge/cmnist_1pct/seed0 runs_shipped/
python run.py --exp poolforge --benchmark cmnist_1pct --seed 0 --gpus 0
```

Model names for `--exp`: `vanilla`, `lff`, `jtt`, `lc`, `ldd`, `poolforge`, `poolforge_no_reweight`,
`poolforge_repeat`, `poolforge_no_gen`, `poolforge_gen_color`, `poolforge_gen_untargeted`, `poolforge_gen_batch`,
`poolforge_no_pgd`, `poolforge_pgd_batch`, `case_study`.
Benchmarks for `--benchmark`: `cmnist_1pct`, `cifar10c_1pct`, `bffhq_0.5pct`, `cmnist_0.5pct`, `cmnist_2pct`,
`cmnist_5pct`, `cifar10c_0.5pct`, `cifar10c_2pct`, `cifar10c_5pct`.

Check progress:

```bash
python progress.py
```

When training is done, repeat steps 3–5 to get the tables, predictions and report of the new runs.

Expected: retrained baselines repeat the shipped results exactly. Retrained PoolForge runs differ slightly from the
shipped ones; on CMNIST seed 0, two retrains gave 84.34 and 84.38 against 84.41 shipped.

## Step 7. Build the submission zip

```bash
python make_submission.py
```

Expected: `submission.zip` with the code, `predictions/` and every checkpoint in `runs/`.
