# ShortcutLearning: LfF + favored Newton unlearning

## Reproduce the results from the checkpoints

1. `pip install -r requirements.txt`
2. `python utils/prepare_data.py`
3. Put the checkpoints in `models/baselines/checkpoints/` and `models/proposed/checkpoints/`.
4. `python eval.py`

Tables are written to `outputs/tables/` (`.md` and `.tex`), figures to `outputs/figures/` (`.pdf` and `.png`).

## Retrain everything

1. `python train.py --group proposed --gpus 0,1 --per_gpu 3 --overwrite`
2. `python train.py --group baselines --gpus 0,1 --per_gpu 3 --overwrite`
3. `python eval.py --recompute`

## Retrain one model

1. `python train.py --list --group all`
2. `python train.py --run ours --benchmark cmnist_1pct --seed 0 --overwrite`
3. `python eval.py --recompute`

## Only some outputs

- `python eval.py --tables main ablation ladder ratio published evidence`
- `python eval.py --figures ratio curves groups ladder evidence`
