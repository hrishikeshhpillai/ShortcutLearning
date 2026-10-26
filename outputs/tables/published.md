| Method | Source | CMNIST | C-CIFAR10 | BFFHQ |
|---|---|---|---|---|
| LfF | published | 61.89 | 33.07 | 62.20 |
| LDD | published | 81.73 | 36.49 | 63.87 |
| SelecMix (best variant) | published | 83.55 | 43.68 | 71.60 |
| LfF | our run | 71.3 ± 2.2 | 41.7 ± 1.0 | 65.7 ± 1.0 |
| LDD | our run | 80.4 ± 0.5 | 34.4 ± 2.0 | 61.5 ± 2.1 |
| Ours | ours | 81.5 ± 0.3 | 41.6 ± 1.3 | 64.9 ± 0.9 |

Unbiased test accuracy (CMNIST, C-CIFAR10) or bias-conflicting test accuracy (BFFHQ), the metrics of the published tables. Our runs: best test checkpoint (the oracle selection SelecMix reports).
