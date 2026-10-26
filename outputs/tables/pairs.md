| Pair added to ours | CMNIST | C-CIFAR10 | CelebA | BFFHQ | Δ vs better member | Δ vs ours |
|---|---|---|---|---|---|---|
| adaptive step + safeguard | 86.0 | 52.5 | 80.1 | 66.2 | -2.1 | -1.2 |
| adaptive step + BiasEnsemble | 84.2 | 31.2 | 74.2 | 68.2 | -6.7 | -8.0 |
| adaptive step + SelecMix | 82.9 | 35.9 | 78.1 | 59.6 | -7.0 | -8.3 |
| adaptive step + pseudo-group balancing | 88.2 | 40.9 | 76.3 | 61.0 | -5.2 | -5.8 |
| safeguard + BiasEnsemble | 86.8 | 38.4 | 74.2 | 65.2 | -6.4 | -6.2 |
| safeguard + SelecMix | 85.5 | 35.1 | 78.0 | 63.8 | -6.2 | -6.8 |
| safeguard + pseudo-group balancing | 84.2 | 39.9 | 76.8 | 59.6 | -8.4 | -7.3 |
| BiasEnsemble + SelecMix | 86.0 | 38.4 | 74.2 | 66.4 | -1.1 | -6.1 |
| SelecMix + pseudo-group balancing | 84.2 | 46.4 | 78.7 | 63.8 | +0.8 | -4.1 |

Selection score (%) of seed 0 for every pair of candidate components on top of ours.
