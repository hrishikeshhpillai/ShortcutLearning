| Method | CMNIST gap | C-CIFAR10 gap | CelebA gap | BFFHQ gap | CMNIST probe | C-CIFAR10 probe | BFFHQ probe |
|---|---|---|---|---|---|---|---|
| Vanilla (ERM) | 64.7 ± 2.4 | 78.5 ± 3.1 | -3.9 ± 24.7 | 42.0 ± 1.9 | 95.8 ± 0.6 | 79.8 ± 6.2 | 81.0 ± 0.5 |
| JTT | 29.8 ± 0.5 | 76.3 ± 4.0 | 0.3 ± 3.3 | 42.7 ± 1.8 | 56.2 ± 7.7 | 84.5 ± 2.3 | 81.1 ± 0.6 |
| LNTL | 54.4 ± 1.4 | 88.6 ± 0.9 | 22.4 ± 0.7 | 38.7 ± 1.1 | 90.4 ± 0.3 | 93.9 ± 0.8 | 81.3 ± 0.6 |
| LDD | 1.7 ± 3.2 | 39.2 ± 6.2 | – | 27.3 ± 2.1 | 99.3 ± 0.5 | 85.1 ± 0.8 | 82.0 ± 0.7 |
| LfF | -6.9 ± 2.9 | 23.1 ± 3.1 | 0.6 ± 1.7 | 21.5 ± 2.9 | 73.8 ± 7.0 | 73.7 ± 4.0 | 73.0 ± 0.9 |
| FMD | 59.1 ± 3.6 | 40.1 ± 29.1 | -12.1 ± 13.8 | 41.5 ± 1.7 | 95.8 ± 0.6 | 79.8 ± 6.2 | 81.0 ± 0.5 |
| GroupDRO | -4.7 ± 3.1 | 40.9 ± 9.3 | 2.2 ± 0.1 | -13.3 ± 17.4 | 30.2 ± 2.2 | 56.1 ± 2.9 | 70.3 ± 1.7 |
| **Ours** | -1.8 ± 1.7 | 22.6 ± 2.2 | 5.0 ± 2.9 | 19.6 ± 2.8 | 48.9 ± 5.5 | 57.8 ± 3.5 | 71.3 ± 0.2 |

Bias gap (aligned minus conflicting test accuracy) and bias probe (accuracy of a linear classifier that recovers the bias attribute from the penultimate features on a group-balanced test subset; chance: 10% CMNIST and C-CIFAR10, 50% BFFHQ). Lower is less shortcut use.
