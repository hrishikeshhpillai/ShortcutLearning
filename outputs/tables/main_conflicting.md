| Method | Bias labels | CMNIST | C-CIFAR10 | CelebA | BFFHQ |
|---|---|---|---|---|---|
| Vanilla (ERM) | none | 35.0 ± 2.4 | 10.1 ± 1.4 | 61.5 ± 4.2 | 57.4 ± 2.0 |
| JTT | none | 60.7 ± 3.9 | 16.2 ± 0.8 | 76.1 ± 0.8 | 55.9 ± 1.8 |
| LDD | none | 76.2 ± 1.7 | 29.1 ± 2.4 | – | 61.5 ± 2.1 |
| LfF | none | 70.6 ± 2.0 | 36.6 ± 1.5 | **81.1 ± 0.4** | **65.7 ± 1.0** |
| FMD | pairs | 40.5 ± 3.5 | 13.2 ± 2.9 | 64.2 ± 6.3 | 58.1 ± 1.9 |
| LNTL | all | 44.7 ± 1.4 | 10.7 ± 0.8 | 68.1 ± 0.8 | 60.6 ± 1.0 |
| GroupDRO | all | 64.4 ± 4.8 | 27.3 ± 0.4 | 84.3 ± 0.1 | 79.1 ± 1.2 |
| Ours | pairs | **79.9 ± 1.5** | **38.0 ± 0.4** | 79.6 ± 1.4 | 64.8 ± 1.0 |

Test bias-conflicting accuracy (%), mean ± std over 3 seeds. Bold: best among methods that use no or only pair bias labels.
