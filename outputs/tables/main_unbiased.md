| Method | Bias labels | CMNIST | C-CIFAR10 | CelebA | BFFHQ |
|---|---|---|---|---|---|
| Vanilla (ERM) | none | 41.4 ± 2.1 | 17.9 ± 1.1 | 59.5 ± 11.1 | 78.4 ± 1.1 |
| JTT | none | 63.7 ± 3.9 | 23.9 ± 1.1 | 76.3 ± 1.8 | 77.3 ± 0.9 |
| LDD | none | 76.3 ± 1.9 | 33.0 ± 2.0 | – | 75.2 ± 1.6 |
| LfF | none | 69.9 ± 1.9 | 39.0 ± 1.6 | 81.4 ± 0.8 | 76.4 ± 2.5 |
| FMD | pairs | 46.4 ± 3.2 | 17.3 ± 0.4 | 58.2 ± 9.6 | **78.8 ± 1.1** |
| LNTL | all | 50.1 ± 1.3 | 19.6 ± 0.7 | 79.3 ± 0.6 | 80.0 ± 0.6 |
| GroupDRO | all | 64.0 ± 5.0 | 31.4 ± 1.2 | 85.4 ± 0.1 | 72.5 ± 7.8 |
| Ours | pairs | **79.7 ± 1.7** | **40.3 ± 0.4** | **82.1 ± 0.4** | 74.6 ± 1.9 |

Test unbiased accuracy (%), mean ± std over 3 seeds. Bold: best among methods that use no or only pair bias labels.
