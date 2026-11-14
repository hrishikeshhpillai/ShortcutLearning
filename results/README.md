# Shortcut-learning baselines: setup and results

This folder holds the results of re-validating three debiasing baselines on four shortcut-learning benchmarks under one shared protocol. `results.csv` has one row per run and is rebuilt from the saved predictions with `eval_metrics.py` (see [Reproducing](#reproducing)).

## The problem

A classifier trained on biased data can learn a **shortcut**: a feature that predicts the label in most of the training data but is not what the task is about. For example, in Colored MNIST almost every "3" is drawn in the same color, so a network can recognize the color instead of the digit.

- **Bias-aligned** samples follow the shortcut (a "3" in its usual color).
- **Bias-conflicting** samples break it (a "3" in another color). They are rare in training.

A model that relies on the shortcut does well on aligned samples and badly on conflicting ones. Debiasing methods try to make the model learn the real feature, which shows up as higher accuracy on **bias-conflicting** samples.

## Benchmarks

| Benchmark | Task (label) | Bias (shortcut) | Classes | Conflicting ratio in train | Source |
|---|---|---|---|---|---|
| Colored MNIST | digit | foreground color | 10 | 1% (491 / 55,000) | LDD release |
| Corrupted CIFAR-10 | object | corruption type (10 types, most severe level) | 10 | 1% (442 / 44,969) | LDD release |
| BFFHQ | age (young / old) | gender | 2 | 0.5% (96 / 19,200) | LDD release |
| CelebA | blond hair | gender (Male) | 2 | natural split | official CelebA |

- **One shared copy of each dataset** is used by every baseline: the release from the Learning-Debiased-Disentangled (LDD) authors for the first three, and the official CelebA partition. The bias level ("severity") is fixed; only the conflicting ratio defines the setting.
- **Splits.** Colored MNIST and Corrupted CIFAR-10 have a validation set that is biased like the training set (45 / 58 conflicting samples) and an unbiased test set (~90% conflicting). BFFHQ's validation set has **no** conflicting samples; its test set is balanced (500 conflicting). CelebA uses its official train / validation / test split (blond men: 1,387 / 182 / 180).
- **Aligned vs. conflicting cells.** The aligned bias value of class y is the one that predicts y most strongly in the training set (argmax over b of P(y | b)). On CelebA this makes blond-female and non-blond-male aligned, so the conflicting groups are **blond men** and **non-blond women**.

## Baselines

| Baseline | Venue | Needs bias labels? | Idea |
|---|---|---|---|
| **LfF**, Learning from Failure | NeurIPS 2020 | no | Train a deliberately biased model with generalized cross-entropy, which amplifies the easy shortcut. Train a second ("debiased") model with per-sample weights `l_biased / (l_biased + l_debiased)`, so samples the biased model fails on (mostly bias-conflicting) count more. |
| **LDD** (DisEnt), Learning Debiased Representation via Disentangled Feature Augmentation | NeurIPS 2021 | no | Like LfF, a biased and a debiased encoder, with the same kind of reweighting. Their features are concatenated (intrinsic + bias), and after a warm-up the bias features are **swapped** between samples, creating synthetic bias-conflicting feature combinations to train on. |
| **LNTL**, Learning Not to Learn | CVPR 2019 | **yes** (training bias labels) | Train a classifier plus a bias predictor attached to its features. The bias predictor is trained through **gradient reversal**, so the classifier's features are pushed to carry no bias information. An entropy term also pushes the bias predictor's outputs on those features toward uniform. Two stages: a plain baseline, then unlearning that starts from it. |

### Backbones and settings

Each baseline keeps its own published backbone and hyperparameters where they exist.

| | Colored MNIST | Corrupted CIFAR-10 | BFFHQ | CelebA |
|---|---|---|---|---|
| LfF | 3-hidden-layer MLP | ResNet-20 | ResNet-18 from scratch (LDD's BFFHQ setting) | ResNet-18, ImageNet-pretrained, frozen (LfF's code) |
| LDD | MLP | ResNet-18 from scratch | ResNet-18 from scratch | ResNet-18 from scratch, BFFHQ hyperparameters (not in the paper) |
| LNTL | LNTL's 4-conv network, per-pixel color heads (official code) | ResNet-18 + bias-class head | ResNet-18 + bias-class head | ResNet-18 + bias-class head |

LNTL's official code only covers Colored MNIST. For the other three benchmarks a bias-class head (predicting corruption type or gender) is used, as in the paper's non-MNIST experiments. Its training budgets for BFFHQ (50 + 50 epochs) and CelebA (10 + 10 epochs) are our choice.

## Evaluation protocol

1. **Metric: bias-conflicting accuracy** on the test set, the mean accuracy over the conflicting (label, bias) cells, each cell weighted equally. All baselines are scored by the same `eval_metrics.py` from saved predictions.
2. **Checkpoint selection: best validation conflicting accuracy**, then a single test evaluation.
- **Exception, BFFHQ:** its validation set has no conflicting samples, so, as in the original papers (LDD, BiasEnsemble, SelecMix), the checkpoint is selected on **test** conflicting accuracy. BFFHQ numbers are therefore optimistic ("oracle" selection).
- **Exception, LDD on CelebA:** on a 2-class task an untrained, near-random model scores ~50% on conflicting samples and can beat trained-but-biased checkpoints. For LDD CelebA only, the checkpoint is the best validation conflicting accuracy **among checkpoints whose validation aligned accuracy is within 5 points of the run's best** (`--selection_aligned_margin 0.05`). This did not help: LDD never trained properly on CelebA (see notes), so it is reported as failed.
3. **Seed 0 only** so far. Paper numbers below are means over 3–5 seeds.

## Results (seed 0, test bias-conflicting accuracy, %)

| | Colored MNIST 1% | Corrupted CIFAR-10 1% | BFFHQ 0.5% | CelebA |
|---|---|---|---|---|
| **LfF** | 65.03 | 38.07 | 64.60 | 79.93 |
| **LDD** | 72.55 | 26.07 | 66.00 | **failed** (no trained model; see notes) |
| **LNTL** | 66.90 | 17.71 | 56.60 | 69.95 |
| *LNTL's own stage-1 baseline (no unlearning)* | *67.47* | *16.40* | *50.20* | — |

Published numbers on the **same data** (LDD/DFA paper, Table 2: unbiased accuracy, best-on-test selection):

| | Colored MNIST 1% | Corrupted CIFAR-10 1% | BFFHQ 0.5% |
|---|---|---|---|
| LfF | 61.89 ± 4.97 | 33.07 ± 0.77 | 62.2 ± 1.0 |
| LDD | 81.73 ± 2.34 | 36.49 ± 1.79 | 63.87 ± 0.31 |

## Baselines 2
### BiasEnsemble

| Implementation | Colored MNIST 0.50% | Colored MNIST 1.00% | Colored MNIST 2.00% | Colored MNIST 5.00% | BFFHQ 0.50% | BFFHQ 1.00% | BFFHQ 2.00% | BFFHQ 5.00% | Dogs & Cats 1.00% | Dogs & Cats 5.00% | BAR 1.00% | BAR 5.00% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official (LfF + BE) | 69.7 | 81.17 | 85.2 | 90.04 | 67.36 | 75.08 | 80.32 | 85.48 | 81.52 | 88.6 | 73.36 | 83.87 |
| Ours (LfF + BE) | 71.72 | 81.16 | 85.64 | 90.3 | 69.6 | 75.8 | 80.8 | 87 | 81.2 | 88.8 | 74.87 | 85.06 |
| Official (DisEnt + BE) | 71.34 | 82.11 | 84.66 | 90.15 | 67.56 | 73.48 | 79.48 | 84.84 | 80.74 | 86.84 | 73.29 | 84.96 |
| Ours (DisEnt + BE) | 72.75 | 82.12 | 85.73 | 90.55 | 70.8 | 75.2 | 81.6 | 84.2 | 81.2 | 88.1 | 75.39 | 84.36 |

### SelecMix

| Implementation | Colored MNIST 1% | BBFHQ 0.50% | CIFAR10C 1% |
|---|---:|---:|---:|
| Official V+ | 83.55±0.42 | 71.60±1.91 | 41.87±0.14 |
| Ours V+ | 82.84 | 68.4 | 41.61 |
| Official L+ | 82.80±0.71 | 70.80±2.95 | 43.68±0.51 |
| Ours L+ | 84.44 | 72.2 | 44.74 |

### Echoes

| Implementation | CelebA Smiling – Avg group acc | CelebA Smiling – Worst group acc | CelebA Narrow Eyes – Avg group acc | CelebA Narrow Eyes – Worst group acc | CelebA Arched Eyebrows – Avg group acc | CelebA Arched Eyebrows – Worst group acc | BBFHQ (0.5%) – Avg | BBFHQ (0.5%) – Worst | BBFHQ (0.5%) – Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official | 69.5±2.7 | 58.7±4.6 | 62.1±0.3 | 48.3±3.1 | 58.0±1.6 | 35.5±2.9 | 71.7 | 48.8 | 38.2 |
| Ours | 69.6 | 59.2 | 58.6 | 42.4 | 58.7 | 47.2 | 68.6 | 55.2 | 24 |

### DCWP

| Implementation | CMNIST 0.01 | CIFAR10-C 0.01 | BFFHQ 0.005 |
|---|---:|---:|---:|
| Official | 95.98 | 41.56 | 60.35 |
| Ours | 95.7 | 42.3 | 59.7 |


## Notes and caveats

- **LfF** matches or slightly exceeds the published numbers on the same data. Its own paper reports higher numbers (74.01 / 41.37) on its own generated datasets, which are a different data copy.
- **LfF on CelebA over-corrects.** Its 79.93% comes from a trained checkpoint (step 6,996) that recognizes blond hair well in both genders (blond women 93.4%, blond men 93.9%) but calls too many people blond, especially men (non-blond men only 44.9%, non-blond women 66.0%). Because non-blond people are ~87% of the test set, its overall accuracy is only ~60% despite the high conflicting accuracy.
- **LDD** matches on BFFHQ but is about 10 points lower on Colored MNIST and Corrupted CIFAR-10. The LDD training code is unchanged from upstream (only seeding, paths, selection and prediction dumps were added). A diagnostic rerun on Corrupted CIFAR-10 that also logged test accuracy showed that even best-on-test selection reaches only 31.7% unbiased accuracy, so the selection rule explains ~1 point of the gap; the rest comes from training (seed, environment or setup).
- **LDD failed on CelebA** (ResNet-18 from scratch, BFFHQ hyperparameters; LDD's paper has no CelebA setting). Its debiased classifier collapses to predicting "blond" for almost everyone, and more often for men than women (an inverted "male → blond" rule), even on the training set (~6% training accuracy at step 19,000), while its biased helper network is fine (~95%). LDD's reweighting gives near-zero weight to the large, easy non-blond majority and very large weight to the rare blond men. Over all 100 checkpoints the best validation aligned accuracy was 50.0% (chance, at step 0), so no trained checkpoint exists to report: both selection rules end up at a near-random early checkpoint (step 4,000: 43.7–67.8% in every test cell, 55.73% conflicting). The result is excluded from `results.csv`; both runs are kept in `logs_diag/ldd/` for reference.
- **LNTL** with its published strength (λ = 0.01, gradient reversal ×0.1) barely removes the bias on Colored MNIST, Corrupted CIFAR-10 and BFFHQ: the bias heads keep predicting the bias almost perfectly, and the result is close to LNTL's own plain baseline. On CelebA the gender head is pushed near chance and the result is a healthy, trained model.
- **Backbones differ between baselines**, so part of each difference comes from architecture. For example, LNTL's plain convnet already scores 67.5% on Colored MNIST, above LfF with its MLP.
- **Selection on small validation sets is noisy**: Colored MNIST / Corrupted CIFAR-10 validation has only 45 / 58 conflicting samples, so the validation score of the chosen checkpoint is well above its test score.

## Reproducing

Data: `../data/{cmnist,cifar10c,bffhq,CelebA}`. Outputs: `../logs/<baseline>/<benchmark>/seed<k>/`
(`test_predictions.npz`, best checkpoint, TensorBoard logs, `train.log`). Environment: conda env `DLDS`.

```bash
# LfF (configs: cmnist_1pct, cifar10c_1pct, bffhq, celeba_protocol)
cd LfF && python train.py with local_user cmnist_1pct seed=0 device=0

# LDD (hyperparameters from scripts/run_*_ours.sh; example: Colored MNIST 1%)
cd Learning-Debiased-Disentangled && python train.py --dataset cmnist --percent 1pct --lr 0.01 \
    --curr_step 10000 --lambda_swap 1 --lambda_dis_align 10 --lambda_swap_align 10 --use_lr_decay \
    --lr_decay_step 10000 --lr_gamma 0.5 --train_ours --tensorboard --seed 0 \
    --benchmark cmnist_1pct --data_dir ../data --log_dir ../logs/ldd

# LNTL (two-stage port: learning-not-to-learn/protocol.py)
cd learning-not-to-learn && python protocol.py --dataset cmnist --percent 1pct --benchmark cmnist_1pct --seed 0

# rebuild the results table
python eval_metrics.py logs/*/*/seed*/test_predictions.npz --out results/results.csv --summary
```
