| Variant | CMNIST | C-CIFAR10 | CelebA | BFFHQ |
|---|---|---|---|---|
| LfF (no unlearning step) | 70.6 ± 2.0 | 36.6 ± 1.5 | 81.1 ± 0.4 | 65.7 ± 1.0 |
| Post-hoc step after LfF (naive combination) | 71.2 ± 1.8 | 36.9 ± 1.4 | 80.8 ± 1.3 | 66.0 ± 1.0 |
| FMD's Eq. 9 direction instead of favored pairing | 63.8 ± 5.0 | 33.2 ± 1.7 | 81.8 (1) | 65.8 (1) |
| No Hessian (gradient direction) | 47.7 ± 2.2 | 29.5 ± 2.1 | 67.1 (1) | 65.8 (1) |
| Random direction of the same length | 70.1 ± 1.7 | 36.4 ± 1.6 | 80.6 ± 0.2 | 66.5 ± 0.7 |
| Step r = 0.05 | 73.4 ± 2.2 | 37.8 ± 0.5 | 80.9 ± 0.8 | – |
| Step r = 0.1 | – | – | 82.3 (1) | 66.8 ± 3.2 |
| Step r = 0.5 | 73.0 ± 6.0 | 28.1 ± 2.8 | – | – |
| k = 50 pairs | 69.2 ± 3.9 | 36.5 ± 0.8 | – | – |
| **Ours** | 79.9 ± 1.5 | 38.0 ± 0.4 | 79.6 ± 1.4 | 64.8 ± 1.0 |

Test bias-conflicting accuracy (%) of ours with one part changed, mean ± std (number of seeds if not 3).
