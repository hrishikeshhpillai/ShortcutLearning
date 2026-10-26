| Method | Bias labels | CMNIST | C-CIFAR10 | CelebA | BFFHQ |
|---|---|---|---|---|---|
| Vanilla (ERM) | none | **99.7 ± 0.1** | 88.6 ± 1.8 | 57.6 ± 23.1 | 99.4 ± 0.3 |
| JTT | none | 90.6 ± 3.7 | **92.5 ± 4.6** | 76.4 ± 3.4 | 98.7 ± 0.4 |
| LDD | none | 77.9 ± 4.3 | 68.3 ± 4.9 | – | 88.8 ± 1.7 |
| LfF | none | 63.6 ± 2.8 | 59.8 ± 4.0 | 81.7 ± 1.5 | 87.1 ± 3.9 |
| FMD | pairs | 99.6 ± 0.1 | 53.4 ± 26.2 | 52.2 ± 15.4 | **99.5 ± 0.3** |
| LNTL | all | 99.1 ± 0.0 | 99.4 ± 0.1 | 90.5 ± 0.6 | 99.3 ± 0.5 |
| GroupDRO | all | 59.7 ± 7.5 | 68.1 ± 9.4 | 86.5 ± 0.1 | 65.8 ± 16.5 |
| Ours | pairs | 78.0 ± 3.2 | 60.7 ± 2.0 | **84.6 ± 1.6** | 84.5 ± 3.2 |

Test bias-aligned accuracy (%), mean ± std over 3 seeds. Bold: best among methods that use no or only pair bias labels.
