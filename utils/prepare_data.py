import glob
import os
import shutil
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
LDD_FOLDER = "https://drive.google.com/drive/folders/1JEOqxrhU_IhkdcRohdbuEtFETUxfNmNT"
CELEBA_REPO = "flwrlabs/celeba"


def ldd_release():
    download = os.path.join(DATA, "_download")
    os.makedirs(download, exist_ok=True)
    if not all(os.path.exists(os.path.join(download, f"{n}.zip")) for n in ("cmnist", "cifar10c", "bffhq")):
        import gdown
        gdown.download_folder(LDD_FOLDER, output=download, quiet=False)
    for name in ("cmnist", "cifar10c", "bffhq"):
        if os.path.isdir(os.path.join(DATA, name)):
            continue
        path = next(iter(glob.glob(os.path.join(download, "**", f"{name}.zip"), recursive=True)))
        print(f"unzipping {path}")
        with zipfile.ZipFile(path) as z:
            z.extractall(DATA)


def celeba():
    target = os.path.join(DATA, "CelebA", "parquet")
    if len(glob.glob(os.path.join(target, "*.parquet"))) >= 25:
        return
    from huggingface_hub import snapshot_download
    tmp = os.path.join(DATA, "_download", "celeba")
    snapshot_download(CELEBA_REPO, repo_type="dataset", allow_patterns="*.parquet", local_dir=tmp)
    os.makedirs(target, exist_ok=True)
    for f in glob.glob(os.path.join(tmp, "**", "*.parquet"), recursive=True):
        shutil.move(f, os.path.join(target, os.path.basename(f)))


def main():
    os.makedirs(DATA, exist_ok=True)
    ldd_release()
    celeba()
    sys.path.insert(0, REPO)
    from utils.data import BENCHMARKS, load_split
    for benchmark in BENCHMARKS:
        for split in ("train", "valid", "test"):
            print(benchmark, split, len(load_split(benchmark, split)["y"]), flush=True)


if __name__ == "__main__":
    main()
