import numpy as np


def cell_accuracies(y, bias, pred, num_classes, num_biases):
    correct = np.zeros((num_classes, num_biases))
    count = np.zeros((num_classes, num_biases))
    np.add.at(correct, (y, bias), (pred == y).astype(float))
    np.add.at(count, (y, bias), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return correct / count, count


def group_metrics(y, bias, pred, aligned_bias, num_classes, num_biases):
    accs, count = cell_accuracies(y, bias, pred, num_classes, num_biases)
    present = count > 0
    conflicting = np.ones_like(present)
    conflicting[np.arange(len(aligned_bias)), aligned_bias] = False
    conflicting &= present
    aligned = present & ~conflicting
    return dict(conflicting=accs[conflicting].mean() if conflicting.any() else float("nan"),
                aligned=accs[aligned].mean(), worst=accs[present].min(), unbiased=accs[present].mean(), cells=accs)


def split_scores(split, pred, aligned_bias, num_classes, num_biases):
    m = group_metrics(split["y"].cpu().numpy(), split["bias"].cpu().numpy(), pred.cpu().numpy(), aligned_bias,
                      num_classes, num_biases)
    return m["conflicting"], m["aligned"]
