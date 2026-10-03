"""Export healthy IXI T2 slices in REFLECT's PNG layout, for training REFLECT on
scans from other hospitals and scanners than the BraTS test data.

Each HD-BET skull-stripped IXI scan is registered affinely to the SRI24 atlas
in BraTS space (as in preprocess.py, at 1 mm), scaled like the BraTS export
(brain 1st-99th percentiles to 0-255) and cut into the 20 central axial
slices. The IXI training and validation subjects of splits.json are used.

  python scripts/export_reflect_ixi.py --out data/reflect_ixi_t2

--thick K1,K2,... simulates thick-slice acquisitions: each subject draws a slice
thickness k from the list (seeded by its name), the registered volume is
averaged over slabs of k slices along a randomly chosen axis and linearly
interpolated back to 1 mm, as when a thick-slice scan is resampled.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_reflect import N_SLICES, save  # noqa: E402
from preprocess import ATLAS, DATA, ROOT  # noqa: E402


def thicken(img, case, ks):
    """Box-average slabs of k voxels along one axis, resample back linearly."""
    from zlib import crc32

    from scipy.ndimage import zoom

    rng = np.random.default_rng(crc32(case.encode()))
    k, ax = int(rng.choice(ks)), int(rng.integers(3))
    if k == 1:
        return img
    n = img.shape[ax] // k * k
    sl = np.moveaxis(img, ax, 0)[:n]
    slab = sl.reshape(n // k, k, *sl.shape[1:]).mean(1)
    up = zoom(slab, (img.shape[ax] / slab.shape[0], 1, 1), order=1, grid_mode=True, mode="nearest")
    return np.moveaxis(up[: img.shape[ax]], 0, ax)


def export(job):
    case, split, out, ks = job
    import ants

    fixed = ants.image_read(str(ATLAS))
    if fixed.dimension == 4:
        fixed = ants.from_numpy(fixed.numpy().squeeze(), origin=fixed.origin[:3],
                                spacing=fixed.spacing[:3], direction=fixed.direction[:3, :3])
    moving = ants.image_read(str(DATA / "ixi_bet" / f"{case}.nii.gz"))
    moving = moving.new_image_like(np.clip(moving.numpy(), 0, None))
    reg = ants.registration(fixed, moving, type_of_transform="Affine", aff_metric="mattes")
    img = np.clip(ants.apply_transforms(fixed, moving, reg["fwdtransforms"]).numpy(), 0, None)
    brain = img > 0
    if ks:
        # normalised averaging: background zeros must not darken the brain edge
        img = np.where(brain, thicken(img, case, ks) / np.maximum(thicken(brain.astype(np.float32), case, ks), 1e-3), 0)
    lo, hi = np.percentile(img[brain], [1, 99])
    img = (np.clip((img - lo) / (hi - lo), 0, 1) * 255 * brain).astype(np.uint8)
    zs = np.nonzero(brain.any(axis=(0, 1)))[0]
    z0 = (zs.min() + zs.max()) // 2 - N_SLICES // 2
    for z in range(z0, z0 + N_SLICES):
        if brain[:, :, z].any():
            save(out / split, case.removesuffix("-T2"), z, img, brain)  # "-T2" would clash with REFLECT's file matching
    return case


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA / "reflect_ixi_t2"))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--thick", default=None, help="comma-separated slice thicknesses to draw from, e.g. 1,2,3,4,5")
    a = p.parse_args()
    out = Path(a.out)
    splits = json.loads((ROOT / "splits.json").read_text())
    ks = [int(k) for k in a.thick.split(",")] if a.thick else None
    jobs = [(c, s, out, ks) for s in ("train", "val") for c in splits[f"ixi_{s}"]]
    for s in ("train", "val"):
        (out / s).mkdir(parents=True, exist_ok=True)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    with ProcessPoolExecutor(a.workers) as ex:
        for i, c in enumerate(ex.map(export, jobs), 1):
            if i % 50 == 0:
                print(f"[{i}/{len(jobs)}] {c}", flush=True)
    for s in ("train", "val"):
        print(s, len(list((out / s).glob("*-T2.png"))), "slices")


if __name__ == "__main__":
    main()
