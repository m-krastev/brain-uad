"""Build a training folder of all healthy IXI slices plus the tumour-free
slices of N randomly chosen BraTS training subjects (the "local" scans),
repeated R times so they are not drowned out. Files are symlinks.

  python scripts/make_mix.py --n 20 --repeat 10 --out data/reflect_mix20
"""
import argparse
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ixi, brats = ROOT / "data/reflect_ixi_t2", ROOT / "data/reflect_brats21_t2"
    out = Path(a.out)
    for s in ("train", "val"):
        (out / s).mkdir(parents=True, exist_ok=True)
        for f in (ixi / s).glob("*.png"):
            if not (out / s / f.name).exists():
                os.symlink(f.resolve(), out / s / f.name)
    subjects = sorted({f.name.split("-slice_")[0] for f in (brats / "train").glob("*-T2.png")})
    local = sorted(random.Random(a.seed).sample(subjects, a.n)) if a.n else []
    for s in local:
        for f in (brats / "train").glob(f"{s}-slice_*"):
            for r in range(a.repeat):
                dst = out / "train" / f.name.replace(s, f"{s}r{r}")
                if not dst.exists():
                    os.symlink(f.resolve(), dst)
    (out / "local_subjects.txt").write_text("\n".join(local) + "\n")
    print(a.n, "local subjects;", len(list((out / "train").glob("*-T2.png"))), "training slices")


if __name__ == "__main__":
    main()
