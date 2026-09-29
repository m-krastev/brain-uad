"""Export BraTS 2021 T2 slices in REFLECT's PNG layout (third_party/REFLECT).

Follows REFLECT's description (Beizaee et al., MICCAI 2025): subjects split
80/10/10, 20 central axial slices per scan, training and validation on the
tumour-free slices only, and testing on the single slice per test subject with
the largest tumour area. Their exact intensity scaling and slice choice are not
published; here each scan's brain 1st-99th percentiles are mapped to 0-255 and
the 20 slices are centred on the brain's axial extent. Scans stay in the
SRI24/BraTS space (240 x 240, padded to 256 by REFLECT's loader) instead of
being re-registered to MNI152.

Besides REFLECT's test set, test_all/ and val_all/ hold all 20 slices of every
test and validation subject (with and without tumour) for the stricter
evaluation used here, where the threshold is chosen on val_all.

  python scripts/export_reflect.py --out data/reflect_brats21_t2
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import DATA, to_atlas_layout  # noqa: E402

N_SLICES = 20


def load(case: Path):
    import nibabel as nib

    img = to_atlas_layout(nib.load(case / f"{case.name}_t2.nii.gz"))
    seg = to_atlas_layout(nib.load(case / f"{case.name}_seg.nii.gz")) > 0
    brain = img > 0
    lo, hi = np.percentile(img[brain], [1, 99])
    img = (np.clip((img - lo) / (hi - lo), 0, 1) * 255 * brain).astype(np.uint8)
    zs = np.nonzero(brain.any(axis=(0, 1)))[0]
    zc = (zs.min() + zs.max()) // 2
    z0 = zc - N_SLICES // 2
    return img, brain, seg, range(z0, z0 + N_SLICES)


def save(d: Path, case: str, z: int, img, brain, seg=None):
    stem = d / f"{case}-slice_{z}"
    # (x, y) array in LPS layout -> image rows = y, columns = x
    Image.fromarray(img[:, :, z].T).save(f"{stem}-T2.png")
    Image.fromarray((brain[:, :, z].T * 255).astype(np.uint8)).save(f"{stem}-brainmask.png")
    if seg is not None:
        Image.fromarray((seg[:, :, z].T * 255).astype(np.uint8)).save(f"{stem}-segmentation.png")


def export(job):
    case, split, out = job
    img, brain, seg, zs = load(case)
    area = {z: int(seg[:, :, z].sum()) for z in zs}
    if split in ("train", "val"):
        for z in zs:
            if area[z] == 0 and brain[:, :, z].any():
                save(out / split, case.name, z, img, brain)
        if split == "val":
            for z in zs:
                if brain[:, :, z].any():
                    save(out / "val_all", case.name, z, img, brain, seg)
    else:
        zmax = max(area, key=area.get)
        if area[zmax] > 0:
            save(out / "test", case.name, zmax, img, brain, seg)
        for z in zs:
            if brain[:, :, z].any():
                save(out / "test_all", case.name, z, img, brain, seg)
    return case.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA / "reflect_brats21_t2"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=14)
    a = p.parse_args()
    out = Path(a.out)
    cases = sorted(c for c in (DATA / "raw" / "BraTS2021").glob("BraTS2021_*") if c.is_dir())
    perm = list(np.random.default_rng(a.seed).permutation(len(cases)))
    n_tr, n_va = int(0.8 * len(cases)), int(0.1 * len(cases))
    split = {}
    for i, j in enumerate(perm):
        split[cases[j].name] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    for s in ("train", "val", "test", "test_all", "val_all"):
        (out / s).mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    with ProcessPoolExecutor(a.workers) as ex:
        for i, name in enumerate(ex.map(export, [(c, split[c.name], out) for c in cases]), 1):
            if i % 100 == 0:
                print(f"[{i}/{len(cases)}] {name}", flush=True)
    for s in ("train", "val", "test", "test_all", "val_all"):
        print(s, len(list((out / s).glob("*-T2.png"))), "slices")


if __name__ == "__main__":
    main()
