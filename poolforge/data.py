import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

BENCHMARKS = {
    "cmnist_1pct": dict(folder="cmnist", percent="1pct", num_classes=10, num_biases=10),
    "cifar10c_1pct": dict(folder="cifar10c", percent="1pct", num_classes=10, num_biases=10),
    "bffhq_0.5pct": dict(folder="bffhq", percent="0.5pct", num_classes=2, num_biases=2),
}
for _p in ("0.5pct", "2pct", "5pct"):
    BENCHMARKS[f"cmnist_{_p}"] = dict(BENCHMARKS["cmnist_1pct"], percent=_p)
    BENCHMARKS[f"cifar10c_{_p}"] = dict(BENCHMARKS["cifar10c_1pct"], percent=_p)

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def family(benchmark):
    for base in ("cmnist", "cifar10c"):
        if benchmark.startswith(base + "_"):
            return base + "_1pct"
    return benchmark


def ldd_paths(benchmark, split):
    info = BENCHMARKS[benchmark]
    root = os.path.join(DATA_DIR, info["folder"], info["percent"])
    if split == "train":
        pattern = [os.path.join(root, part, "*", "*") for part in ("align", "conflict")]
    elif info["folder"] == "bffhq":
        pattern = [os.path.join(DATA_DIR, "bffhq", split, "*")]
    elif split == "valid":
        pattern = [os.path.join(root, "valid", "*", "*") if info["folder"] == "cifar10c"
                   else os.path.join(root, "valid", "*")]
    else:
        pattern = [os.path.join(DATA_DIR, info["folder"], "test", "*", "*")]
    paths = sorted(p for pat in pattern for p in glob.glob(pat) if os.path.isfile(p))
    if not paths:
        raise FileNotFoundError(f"no images for {benchmark}/{split} under {pattern}; run python prepare_data.py")
    return paths


def parse_label_bias(path):
    stem = os.path.splitext(os.path.basename(path))[0].split("_")
    return int(stem[-2]), int(stem[-1])


def load_split(benchmark, split):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{benchmark}_{split}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    paths = ldd_paths(benchmark, split)
    labels = np.array([parse_label_bias(p) for p in paths])
    images = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    out = dict(images=images, y=labels[:, 0], bias=labels[:, 1], paths=np.array(paths))
    tmp = f"{cache}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, cache)
    return out


def aligned_bias(y, bias, num_classes, num_biases):
    count = np.zeros((num_classes, num_biases))
    np.add.at(count, (y, bias), 1.0)
    p_y_given_b = count / np.maximum(count.sum(0, keepdims=True), 1)
    return p_y_given_b.argmax(1)


def load_data(benchmark, device):
    splits = {}
    for split in ("train", "valid", "test"):
        raw = load_split(benchmark, split)
        splits[split] = dict(x=torch.from_numpy(raw["images"]).to(device), y=torch.from_numpy(raw["y"]).long().to(device),
                             bias=torch.from_numpy(raw["bias"]).long().to(device), paths=raw["paths"])
    return splits


def random_crop_flip(x, pad):
    b, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (dy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (dx[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    x = x[torch.arange(b, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
          rows, cols]
    flip = torch.rand(b, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(-1), x)


def normalize(x, mean, std):
    mean = torch.tensor(mean, device=x.device)[None, :, None, None]
    std = torch.tensor(std, device=x.device)[None, :, None, None]
    return (x - mean) / std


def preprocess(benchmark, batch, train):
    x = batch.permute(0, 3, 1, 2).float() / 255
    if benchmark.startswith("cmnist"):
        return x
    if benchmark.startswith("bffhq"):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    if train:
        x = random_crop_flip(x, 4)
    return normalize(x, CIFAR_MEAN, CIFAR_STD)


def train_batch(benchmark, train, index):
    return preprocess(benchmark, train["x"][index], train=True)


@torch.no_grad()
def predict_logits(model, benchmark, x, batch_size=500):
    was_training = model.training
    model.eval()
    logits = torch.cat([model(preprocess(benchmark, x[i:i + batch_size], train=False)).float()
                        for i in range(0, len(x), batch_size)])
    model.train(was_training)
    return logits
