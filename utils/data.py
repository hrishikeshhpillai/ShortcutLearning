import glob
import io
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
    "celeba": dict(folder="CelebA", num_classes=2, num_biases=2),
}
for _p in ("0.5pct", "2pct", "5pct"):
    BENCHMARKS[f"cmnist_{_p}"] = dict(BENCHMARKS["cmnist_1pct"], percent=_p)
    BENCHMARKS[f"cifar10c_{_p}"] = dict(BENCHMARKS["cifar10c_1pct"], percent=_p)

MAIN = ("cmnist_1pct", "cifar10c_1pct", "celeba", "bffhq_0.5pct")
RATIOS = ("0.5pct", "1pct", "2pct", "5pct")
TITLES = {"cmnist_1pct": "Colored MNIST", "cifar10c_1pct": "Corrupted CIFAR-10", "celeba": "CelebA",
          "bffhq_0.5pct": "BFFHQ"}
CELEBA_TARGET, CELEBA_BIAS = "Blond_Hair", "Male"
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
        raise FileNotFoundError(f"no images for {benchmark}/{split} under {pattern}; run python utils/prepare_data.py")
    return paths


def parse_label_bias(path):
    stem = os.path.splitext(os.path.basename(path))[0].split("_")
    return int(stem[-2]), int(stem[-1])


def load_split(benchmark, split):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{benchmark}_{split}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    if benchmark == "celeba":
        out = load_celeba_split(split)
    else:
        paths = ldd_paths(benchmark, split)
        labels = np.array([parse_label_bias(p) for p in paths])
        images = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
        out = dict(images=images, y=labels[:, 0], bias=labels[:, 1], paths=np.array(paths))
    tmp = f"{cache}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, cache)
    return out


def celeba_parquet_files(split):
    return sorted(glob.glob(os.path.join(DATA_DIR, "CelebA", "parquet", f"{split}-*.parquet")))


def load_celeba_split(split):
    import pyarrow.parquet as pq

    columns = pq.read_schema(celeba_parquet_files(split)[0]).names
    attr_names = [c for c in columns if c not in ("image", "celeb_id")]
    attrs = []
    for f in celeba_parquet_files(split):
        table = pq.read_table(f, columns=attr_names)
        attrs.append(np.stack([np.asarray(table[c]).astype(int) for c in attr_names], axis=1))
    attrs = (np.concatenate(attrs) > 0).astype(np.int64)
    y = attrs[:, attr_names.index(CELEBA_TARGET)]
    bias = attrs[:, attr_names.index(CELEBA_BIAS)]
    return dict(y=y, bias=bias, attrs=attrs, attr_names=np.array(attr_names),
                paths=np.array([f"{split}:{i}" for i in range(len(y))]))


def celeba_images(split, indices=None):
    import pyarrow.parquet as pq

    wanted = None if indices is None else set(int(i) for i in indices)
    offset = 0
    for f in celeba_parquet_files(split):
        column = pq.read_table(f, columns=["image"])["image"].to_pylist()
        for i, item in enumerate(column):
            if wanted is None or offset + i in wanted:
                yield offset + i, Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        offset += len(column)


def aligned_bias(y, bias, num_classes, num_biases):
    count = np.zeros((num_classes, num_biases))
    np.add.at(count, (y, bias), 1.0)
    p_y_given_b = count / np.maximum(count.sum(0, keepdims=True), 1)
    return p_y_given_b.argmax(1)


@torch.no_grad()
def celeba_features(split, device, flip=False, batch_size=256):
    import torchvision
    from torchvision import transforms as T

    cache = os.path.join(CACHE_DIR, f"celeba_feat_{split}{'_flip' if flip else ''}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.to(device)
    bn_state = os.path.join(CACHE_DIR, "celeba_resnet18_bn.pt")
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def batches(split_name, flip_images):
        buffer = []
        for _, image in celeba_images(split_name):
            x = tf(image)
            buffer.append(x.flip(-1) if flip_images else x)
            if len(buffer) == batch_size:
                yield torch.stack(buffer).to(device)
                buffer = []
        if buffer:
            yield torch.stack(buffer).to(device)

    if os.path.exists(bn_state):
        net.load_state_dict(torch.load(bn_state))
    else:
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        net.train()
        for x in batches("train", False):
            net(x)
        torch.save(net.state_dict(), f"{bn_state}.{os.getpid()}.tmp")
        os.replace(f"{bn_state}.{os.getpid()}.tmp", bn_state)
    net.eval()
    feats = np.concatenate([net(x).float().cpu().numpy() for x in batches(split, flip)])
    tmp = f"{cache}.{os.getpid()}.tmp.npy"
    np.save(tmp, feats.astype(np.float32))
    os.replace(tmp, cache)
    return feats


def load_data(benchmark, device):
    splits = {}
    for split in ("train", "valid", "test"):
        raw = load_split(benchmark, split)
        x = torch.from_numpy(celeba_features(split, device)) if benchmark == "celeba" else torch.from_numpy(raw["images"])
        splits[split] = dict(x=x.to(device), y=torch.from_numpy(raw["y"]).long().to(device),
                             bias=torch.from_numpy(raw["bias"]).long().to(device), paths=raw["paths"])
    if benchmark == "celeba":
        splits["train"]["x_flip"] = torch.from_numpy(celeba_features("train", device, flip=True)).to(device)
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
    if benchmark == "celeba":
        return batch
    x = batch.permute(0, 3, 1, 2).float() / 255
    if benchmark.startswith("cmnist"):
        return x
    if benchmark.startswith("bffhq"):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    if train:
        x = random_crop_flip(x, 4)
    return normalize(x, CIFAR_MEAN, CIFAR_STD)


def train_batch(benchmark, train, index):
    if benchmark == "celeba":
        flip = torch.rand(len(index), device=index.device) < 0.5
        x = torch.where(flip[:, None], train["x_flip"][index], train["x"][index])
        return preprocess(benchmark, x, train=True)
    return preprocess(benchmark, train["x"][index], train=True)


@torch.no_grad()
def predict_logits(model, benchmark, x, batch_size=500):
    was_training = model.training
    model.eval()
    logits = torch.cat([model(preprocess(benchmark, x[i:i + batch_size], train=False)).float()
                        for i in range(0, len(x), batch_size)])
    model.train(was_training)
    return logits
