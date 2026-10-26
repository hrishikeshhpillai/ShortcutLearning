| Method | Bias labels | CMNIST | C-CIFAR10 | CelebA | BFFHQ |
|---|---|---|---|---|---|
| Vanilla (ERM) | none | 0.0 ± 0.0 | 0.0 ± 0.0 | 16.6 ± 7.3 | 42.7 ± 7.9 |
| JTT | none | 3.9 ± 1.5 | 0.0 ± 0.0 | 68.3 ± 0.1 | 48.5 ± 5.2 |
| LDD | none | 24.7 ± 9.0 | 1.3 ± 1.2 | – | 48.8 ± 1.5 |
| LfF | none | 7.3 ± 3.7 | 3.0 ± 3.6 | 66.4 ± 4.3 | 58.4 ± 1.2 |
| FMD | pairs | 0.0 ± 0.0 | 0.0 ± 0.0 | 25.1 ± 20.4 | 43.5 ± 7.7 |
| LNTL | all | 0.3 ± 0.4 | 0.0 ± 0.0 | 44.3 ± 1.8 | 53.5 ± 3.8 |
| GroupDRO | all | 27.6 ± 4.5 | 1.0 ± 1.4 | 81.3 ± 0.5 | 55.9 ± 16.2 |
| Ours | pairs | **48.5 ± 7.0** | **7.3 ± 3.3** | **71.3 ± 3.5** | **59.5 ± 1.6** |

Test worst-group accuracy (%), mean ± std over 3 seeds. Bold: best among methods that use no or only pair bias labels.
