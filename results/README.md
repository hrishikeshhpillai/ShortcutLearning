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

---

# Our model: LfF + FMD

We combine **LfF** (Nam et al., NeurIPS 2020) with **FMD**, *Fast Model DeBias with Machine Unlearning* (Chen et al., NeurIPS 2023). LfF trains the debiased classifier; FMD then removes the bias that is left by unlearning it from the last layer, in one Newton step. FMD has no official code; ours is in `lff_fmd/`.

## Method

1. **LfF** as in Baseline1: a biased model trained with generalized cross-entropy, and a debiased model trained with per-sample weights `l_b / (l_b + l_d)`. Same backbones and hyperparameters as the table above (LfF's configs; LDD's BFFHQ setting).
2. **Counterfactual pairs** `(c, c̄)`: `c` is a bias-aligned training sample, `c̄` has the same label and a different bias value.
   - Colored MNIST: `c̄` is `c` with the digit redrawn in another bias color (the release colors digits as intensity × a fixed palette color, so this counterfactual is exact). k = 5,000 pairs, FMD's setting.
   - CelebA: FMD's recipe: same `Blond_Hair`, opposite `Male`, the partner with the most agreeing attributes among the other 38. k = 200, FMD's setting.
   - Corrupted CIFAR-10, BFFHQ: no way to change the bias of an image, so `c̄` is the most similar (ImageNet ResNet-18 features) same-label bias-conflicting training sample. k = 1,000.
   - Building the pairs needs bias labels for these k samples. LfF alone needs none; LNTL needs them for the whole training set.
3. **Unlearning** (FMD Eq. 9) on the last linear layer θ = [W b]: θ ← θ + H⁻¹ Σᵢ (∇L(cᵢ) − ∇L(c̄ᵢ)), i.e. remove the influence of `c` and add that of `c̄`. H is the exact Hessian of the debiased model's own training loss, the LfF-weighted cross-entropy, in closed form from the penultimate features (checked against autograd in `lff_fmd/test_fmd.py`).
4. **Our change to FMD: "favored" pairing.** Eq. 9 always unlearns the bias-aligned member, which assumes the model prefers the bias. LfF often **over-corrects**: it fits the conflicting samples better than the aligned ones (Colored MNIST: 64% aligned vs. 71% conflicting test accuracy). Eq. 9 then moves the model further away from balance and *raises* FMD's own bias measure. For each pair we therefore unlearn whichever member the model currently fits better and learn the other. On a biased model this is exactly Eq. 9. Eq. 9 itself is kept as an ablation.
5. **Step size.** FMD gives none, and H is singular (softmax is shift-invariant), so we add damping λ = γ · mean(diag H) and take a step of relative length r: θ ← θ + r‖θ‖ d/‖d‖. γ ∈ {0.001, 0.01, 0.1, 1}, r ∈ {0, 0.005, …, 2}. Both are chosen on validation conflicting accuracy (BFFHQ: test, as above), r = 0 included. Ties are broken on validation aligned accuracy: the validation sets have only 45–58 conflicting samples (Colored MNIST, Corrupted CIFAR-10), so many steps tie, and among them we take the least biased.

FMD costs seconds on top of LfF: one pass to extract features, then linear algebra on a matrix of at most 1,026 × 1,026.

## Setup differences from Baseline1

- **Seeds 0, 1, 2** (Baseline1 used seed 0 only). Tables below are mean ± std.
- **LfF is our own re-run**, from `lff_fmd/train.py`, which follows `LfF/train.py` with the GPU-side data pipeline. It scores higher than the Baseline1 LfF run on Colored MNIST (seed 0: 73.25 vs. 65.03; 3-seed mean 70.57) and similar elsewhere. Both runs are inside the published spread (61.89 ± 4.97, LDD paper), so we compare LfF + FMD against our own re-run, which is its starting point.
- **CelebA backbone**: LfF trains its linear head on a frozen ImageNet ResNet-18 kept in train mode, so BatchNorm adapts to CelebA. We re-estimate BN statistics on the CelebA training set once and cache the features (and their horizontal flips, LfF's only augmentation). CelebA images come from the `flwrlabs/celeba` mirror (official Drive files can no longer be downloaded directly); split sizes and blond-men counts match the official ones exactly.
- New metrics in `eval_metrics.py`: **aligned accuracy** and **worst-group accuracy** (lowest (label, bias) cell). "Unbiased accuracy" below is the mean over all cells.

All tables: `results/analysis_tables.md`; per-run rows: `results/results_lff_fmd.csv`.

## 1. Comparative

Test bias-conflicting accuracy (%), mean ± std over 3 seeds; Baseline1 rows are seed 0.

| | Colored MNIST 1% | Corrupted CIFAR-10 1% | BFFHQ 0.5% | CelebA |
|---|---|---|---|---|
| LDD (Baseline1) | **72.55** | 26.07 | 66.00 | failed |
| LNTL (Baseline1) | 66.90 | 17.71 | 56.60 | 69.95 |
| LfF (Baseline1) | 65.03 | **38.07** | 64.60 | 79.93 |
| Vanilla ERM | 34.95 ± 2.35 | 10.09 ± 1.43 | 57.40 ± 2.05 | 61.50 ± 4.16 |
| LfF (our re-run) | 70.57 ± 1.95 | 36.64 ± 1.46 | 65.67 ± 1.04 | **81.13 ± 0.43** |
| **LfF + FMD (ours)** | 71.24 ± 1.76 | 36.89 ± 1.42 | **66.00 ± 1.02** | 80.80 ± 1.30 |

| LfF → LfF + FMD | Colored MNIST | Corrupted CIFAR-10 | BFFHQ | CelebA |
|---|---|---|---|---|
| aligned acc | 63.63 → **70.89** | 59.77 → **63.13** | 87.13 → **88.73** | 81.70 → **83.21** |
| worst-group acc | 7.29 → **30.20** | 3.00 → 3.33 | 58.40 → **61.87** | 66.41 → **69.58** |
| unbiased acc (all cells) | 69.88 → **71.21** | 38.95 → **39.52** | 76.40 → **77.37** | 81.41 → **82.01** |
| FMD's counterfactual bias | lower in 3/3 seeds | lower in 3/3 | lower in 2/3 | lower in 3/3 |

- **On the headline metric FMD adds little.** Conflicting accuracy rises by 0.25–0.67 points on three benchmarks and falls by 0.33 on CelebA. All changes are well inside one standard deviation across seeds.
- **What FMD does do is undo LfF's over-correction.** Aligned, worst-group and unbiased accuracy improve on all four benchmarks, and FMD's counterfactual bias drops in 11 of 12 runs. The clearest case is Colored MNIST: LfF leaves a whole aligned cell almost unlearned (blue 6s, see §3), FMD recovers it, and worst-group accuracy goes from 7% to 30%.
- **Against the other baselines**, LfF + FMD ties LDD for the best BFFHQ number (66.00) and is above every Baseline1 run on CelebA (LDD failed there), though below our own LfF re-run. It is behind LDD on Colored MNIST, and behind the Baseline1 LfF run on Corrupted CIFAR-10 (38.07, within one standard deviation of our 36.89). Backbones differ between methods (see Baselines above).

## 2. Ablation

Test conflicting accuracy (%), mean ± std over 3 seeds; aligned accuracy in the second table.

| Variant | Colored MNIST | Corrupted CIFAR-10 | BFFHQ | CelebA |
|---|---|---|---|---|
| **LfF + FMD (full)** | 71.24 ± 1.76 | 36.89 ± 1.42 | 66.00 ± 1.02 | 80.80 ± 1.30 |
| remove FMD (= LfF) | 70.57 ± 1.95 | 36.64 ± 1.46 | 65.67 ± 1.04 | 81.13 ± 0.43 |
| remove LfF (FMD on vanilla) | 40.51 ± 3.53 | 13.25 ± 2.91 | 58.07 ± 1.89 | 64.22 ± 6.35 |
| remove both (vanilla) | 34.95 ± 2.35 | 10.09 ± 1.43 | 57.40 ± 2.05 | 61.50 ± 4.16 |
| FMD's original Eq. 9 direction (no favored pairing) | 70.64 ± 1.89 | 36.61 ± 1.42 | 66.47 ± 0.74 | 83.25 ± 0.84 |
| no Hessian (plain gradient step, H = I) | 70.59 ± 2.10 | 36.75 ± 1.58 | 65.93 ± 1.23 | 81.67 ± 0.57 |
| Hessian without the LfF weights | 71.06 ± 2.36 | 36.50 ± 1.62 | 65.93 ± 0.94 | 80.80 ± 0.53 |
| matched real pairs instead of recolored digits | 70.91 ± 1.89 | – | – | – |

| Aligned accuracy (%) | Colored MNIST | Corrupted CIFAR-10 | BFFHQ | CelebA |
|---|---|---|---|---|
| **LfF + FMD (full)** | 70.89 ± 3.51 | 63.13 ± 5.93 | 88.73 ± 4.49 | 83.21 ± 1.88 |
| remove FMD (= LfF) | 63.63 ± 2.82 | 59.77 ± 4.00 | 87.13 ± 3.90 | 81.70 ± 1.53 |
| FMD's original Eq. 9 direction | 62.27 ± 2.59 | 59.47 ± 3.50 | 86.27 ± 4.51 | **66.80 ± 1.10** |
| no Hessian (H = I) | 64.90 ± 1.16 | 60.40 ± 3.34 | 88.00 ± 4.65 | 85.71 ± 1.38 |
| Hessian without the LfF weights | 64.70 ± 3.68 | 62.37 ± 3.89 | 87.20 ± 3.96 | 82.50 ± 2.35 |

- **LfF carries the conflicting accuracy.** FMD on a vanilla model helps (+5.6, +3.2, +0.7, +2.7 points) but stays far below LfF: a single last-layer step cannot undo a shortcut that the features themselves encode (Colored MNIST vanilla aligned accuracy stays at 99.6%).
- **Favored pairing vs. Eq. 9.** Eq. 9 has the higher conflicting accuracy on CelebA (83.25) only because it pushes LfF's over-correction further: aligned accuracy drops from 81.7 to 66.8. The conflicting-accuracy protocol rewards this. On Colored MNIST and BFFHQ, Eq. 9 also lowers aligned accuracy, and on Colored MNIST it raises FMD's own bias measure (seed 1: 0.318 → 0.332). Favored pairing is the only variant that improves both groups everywhere.
- **The Hessian matters little at this scale.** Without H (plain gradient step) the results are similar to, and on CelebA aligned accuracy better than, the full model. With a single linear layer, the damping and step search absorb most of what H⁻¹ does. Dropping the LfF weights from H is also within noise.
- **Number of pairs.** k from 50 to 5,000 changes conflicting accuracy by at most 0.7 points (figure below). k = 50 is enough. So the pairs, and the bias labels they need, can be few.
- **Pair type.** On Colored MNIST, matched real pairs give the same conflicting accuracy as exact recolored counterfactuals (70.91 vs. 71.24), and slightly higher aligned accuracy (72.43 vs. 70.89).

![k sweep](figures/ablation_k_sweep.png)

## 3. Case study: where LfF + FMD does badly

![accuracy along the FMD step](figures/case_step_tradeoff.png)

*Test accuracy along the FMD step r (seed 0, the chosen damping); the dashed line is the selected step.*

- **The step size is a cliff.** Accuracy is flat for small r, improves slightly, then collapses (Colored MNIST, BFFHQ and CelebA past r ≈ 1). The selected step usually sits just before the cliff: on BFFHQ it is right at the edge (r = 1), selected on test, so its gain should not be taken at face value. With validation sets of 45–58 conflicting samples, choosing r is the weakest part of the method.

![per-cell change](figures/case_cell_change.png)

*Change in test accuracy per (label, bias) cell, LfF → LfF + FMD, mean over 3 seeds. Outlined squares are the aligned cells.*

- **Colored MNIST:** almost all of the gain is one aligned cell. LfF misclassifies most blue 6s (it has learned "blue → not 6"), and FMD repairs that cell by about +40 points. The cost is small losses in a few conflicting cells (blue digits that are not 6, e.g. 0 and 2 in blue).
- **Corrupted CIFAR-10:** gains in the label-7 row and several aligned cells; losses in conflicting cells such as label 6 with corruption 4 (−7 points). The net change in conflicting accuracy is +0.25, so FMD mostly moves errors between cells rather than removing them.
- **BFFHQ:** FMD helps young men (+3.3) and young women (+2.4) but **hurts old women** (67.3 → 64.7), a bias-conflicting cell and the one that matters most here:

  | BFFHQ test accuracy (%) | young, female | young, male | old, female | old, male |
  |---|---|---|---|---|
  | LfF | 86.7 | 64.0 | **67.3** | 87.6 |
  | LfF + FMD | 89.1 | 67.3 | 64.7 | 88.4 |

- **CelebA:** FMD lowers accuracy on **blond men** (90.2 → 88.5), the rarest group (1,387 training images), and raises it on non-blond men (66.4 → 69.6). Seeds 1 and 2 lose 0.9–1.0 conflicting points. LfF's own failure mode on CelebA, predicting "blond" too often (as in Baseline1), is only partly fixed: non-blond women stay at 73%.

  | CelebA test accuracy (%) | not blond, female | not blond, male | blond, female | blond, male |
  |---|---|---|---|---|
  | Vanilla ERM | 75.8 | 66.2 | 48.9 | 47.2 |
  | LfF | 72.1 | 66.4 | 97.0 | **90.2** |
  | LfF + FMD | 73.1 | 69.6 | 96.8 | 88.5 |

![failures](figures/case_failures.png)

*The most confident bias-conflicting test errors of LfF + FMD (seed 0).*

- **Confident errors remain.** Many errors are made with probability ≥ 0.97. On Colored MNIST and Corrupted CIFAR-10 they are mostly hard images: badly drawn digits, and heavily corrupted images (snow-like, pixelated) where the object is barely visible. A last-layer edit cannot fix those.
- **CelebA's worst errors look like label noise.** All eight are women with platinum, grey or light-blonde hair labeled *not blond*. Part of what remains on CelebA is the annotation, not the model.
- **BFFHQ's worst errors** are older women predicted *young*, often with light hair or make-up. These are the old-female cell that FMD made worse.

## Takeaways

- LfF + FMD is never worse than LfF by more than noise on the protocol's metric, and it consistently reduces LfF's over-correction (aligned, worst-group and unbiased accuracy up on all four benchmarks) for a few seconds of extra compute.
- FMD's Eq. 9 is not safe to apply to an already-debiased model: it assumes the model favors the bias, and with LfF it can increase bias. Choosing the direction per pair fixes this.
- The protocol metric (conflicting accuracy alone) rewards over-correction; aligned or worst-group accuracy should be reported next to it.
- Gains on conflicting accuracy are within seed noise; three seeds and small validation sets limit what can be claimed.

## Reproducing LfF + FMD

Environment: `env260801` (PyTorch 2.5.1). Data: `data/` as above, plus `data/CelebA/parquet/` (from `huggingface.co/datasets/flwrlabs/celeba`).

```bash
# everything: LfF + vanilla training (3 seeds, GPUs 0 and 1), all FMD variants, tables and figures; restart-safe
setsid nohup bash lff_fmd/legacy/run_all.sh >> logs/run_all.log 2>&1 &
bash lff_fmd/status.sh                                     # progress

# single pieces
python lff_fmd/train.py --benchmark cmnist_1pct --method lff --seed 0
python lff_fmd/fmd.py --benchmark cmnist_1pct --seed 0 --name lff_fmd
python lff_fmd/test_fmd.py                                 # closed-form Hessian vs. autograd
python lff_fmd/analysis.py                                 # results_lff_fmd.csv, analysis_tables.md, figures/
```
