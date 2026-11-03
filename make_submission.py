import argparse
import glob
import os
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))
RUN_FILES = ("best.pt", "result.json", "history.csv", "train.log")


def members():
    files = ["README.md", "requirements.txt"] + sorted(glob.glob(os.path.join(REPO, "*.py")))
    files += sorted(glob.glob(os.path.join(REPO, "poolforge", "**", "*.py"), recursive=True))
    files += sorted(glob.glob(os.path.join(REPO, "predictions", "**", "*.csv"), recursive=True))
    for folder in sorted(glob.glob(os.path.join(REPO, "runs", "*", "*", "seed*"))):
        if os.path.exists(os.path.join(folder, "best.pt")):
            files += [os.path.join(folder, f) for f in RUN_FILES]
    return [os.path.relpath(os.path.join(REPO, f), REPO) for f in files]


def main():
    p = argparse.ArgumentParser(description="Pack code, checkpoints and predictions into one zip file.")
    p.add_argument("--out", default="submission.zip")
    args = p.parse_args()
    files = members()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(REPO, f), f)
    print(f"wrote {args.out}: {len(files)} files, {os.path.getsize(args.out) / 2 ** 20:.0f} MB")


if __name__ == "__main__":
    main()
