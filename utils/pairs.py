import os

import numpy as np
import torch

from utils.data import BENCHMARKS, CACHE_DIR, CELEBA_BIAS, CELEBA_TARGET, aligned_bias, family, load_split

DEFAULT_K = {"cmnist_1pct": 5000, "cifar10c_1pct": 1000, "bffhq_0.5pct": 1000, "celeba": 200}
for _b in BENCHMARKS:
    DEFAULT_K.setdefault(_b, DEFAULT_K[family(_b)])


def default_kind(benchmark):
    return "recolor" if benchmark.startswith("cmnist") else "match"


def cmnist_palette(images, bias, num_biases):
    palette = np.zeros((num_biases, 3))
    for b in range(num_biases):
        x = images[bias == b].reshape(-1, 3).astype(float)
        lit = x[x.max(1) > 200]
        palette[b] = (lit / lit.max(1, keepdims=True)).mean(0)
    return palette


def recolor(images, new_bias, palette):
    intensity = images.max(-1, keepdims=True).astype(float)
    return np.clip(np.round(intensity * palette[new_bias][:, None, None, :]), 0, 255).astype(np.uint8)


@torch.no_grad()
def imagenet_embedding(benchmark, images, device, batch_size=500):
    import torch.nn.functional as F
    import torchvision

    cache = os.path.join(CACHE_DIR, f"{benchmark}_train_imagenet_emb.npy")
    if os.path.exists(cache):
        return np.load(cache)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out.append(F.normalize(net((x - mean) / std), dim=1).cpu().numpy())
    emb = np.concatenate(out).astype(np.float32)
    tmp = f"{cache}.{os.getpid()}.tmp.npy"
    np.save(tmp, emb)
    os.replace(tmp, cache)
    return emb


def build_pairs(benchmark, k, seed=0, device=torch.device("cuda")):
    kind = default_kind(benchmark)
    info = BENCHMARKS[benchmark]
    train = load_split(benchmark, "train")
    y, bias = train["y"], train["bias"]
    aligned = aligned_bias(y, bias, info["num_classes"], info["num_biases"])
    is_aligned = bias == aligned[y]
    rng = np.random.RandomState(seed)
    candidates = np.flatnonzero(is_aligned)
    c_index = rng.choice(candidates, size=k, replace=k > len(candidates))

    if kind == "recolor":
        palette = cmnist_palette(train["images"], bias, info["num_biases"])
        shift = rng.randint(1, info["num_biases"], size=k)
        cbar_bias = (aligned[y[c_index]] + shift) % info["num_biases"]
        return dict(y=y[c_index], c_index=c_index, cbar_index=np.full(k, -1), c_bias=bias[c_index],
                    cbar_bias=cbar_bias, cbar_images=recolor(train["images"][c_index], cbar_bias, palette))

    if benchmark == "celeba":
        names = list(train["attr_names"])
        keep = [i for i, n in enumerate(names) if n not in (CELEBA_TARGET, CELEBA_BIAS)]
        emb = train["attrs"][:, keep].astype(np.float32)

        def similarity(a, b):
            return emb[a] @ emb[b].T + (1 - emb[a]) @ (1 - emb[b]).T
    else:
        emb = imagenet_embedding(benchmark, train["images"], device)

        def similarity(a, b):
            return emb[a] @ emb[b].T

    cbar_index = np.empty(k, dtype=int)
    for label in np.unique(y[c_index]):
        rows = np.flatnonzero(y[c_index] == label)
        partners = np.flatnonzero((y == label) & ~is_aligned)
        for chunk in np.array_split(rows, max(1, len(rows) // 500)):
            sim = similarity(c_index[chunk], partners)
            sim = sim + 1e-6 * rng.rand(*sim.shape)
            cbar_index[chunk] = partners[sim.argmax(1)]
    return dict(y=y[c_index], c_index=c_index, cbar_index=cbar_index, c_bias=bias[c_index],
                cbar_bias=bias[cbar_index], cbar_images=None)
