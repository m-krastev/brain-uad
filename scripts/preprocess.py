"""Preprocess IXI (healthy) and MSD Task01 (brain tumour) T2 scans.

Follows the protocol of pDDPM (Behrendt et al., MIDL 2023):
  IXI: HD-BET skull stripping (run beforehand, scripts/run_hdbet.sh), affine
       registration to the SRI24 atlas in BraTS space.
  BraTS 2021: skull stripped, in SRI24/BraTS space; T2 and whole tumour.
  MSD: already skull stripped and in SRI24/BraTS space (RAS header, flipped
       to the atlas' LPS array layout); T2 is channel 3.
  Both: crop the black border to 192 x 192 x 160, N4 bias-field correction,
       downsample by two to 96 x 96 x 80, drop the 15 top and bottom axial
       slices (50 remain), scale by the 99.5th percentile of brain intensities
       and clip to [0, 1].

Each case is saved as data/proc/<set>/<case>.npz with img (float16, z y x),
brain (bool) and, for MSD, seg (bool, whole tumour).

Usage:
  python scripts/preprocess.py ixi  --workers 12
  python scripts/preprocess.py msd  --workers 12
  python scripts/preprocess.py check
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ATLAS = DATA / "atlas" / "brats_sri24_skullstripped.nii"
# Crop box in atlas array space (x, y, z): 240 x 240 x 155 -> 192 x 192 x 160.
# x/y are centred on the atlas brain; z is zero-padded (2 below, 3 above).
CROP_XY = 192
Z_PAD = (2, 3)
DROP = 15


def atlas_array():
    import nibabel as nib
    return np.asarray(nib.load(ATLAS).dataobj, dtype=np.float32).squeeze()


def crop_box():
    a = atlas_array()
    xs, ys = np.nonzero(a.max(axis=2) > 0)
    cx, cy = (xs.min() + xs.max() + 1) // 2, (ys.min() + ys.max() + 1) // 2
    h = CROP_XY // 2
    x0, y0 = int(np.clip(cx - h, 0, a.shape[0] - CROP_XY)), int(np.clip(cy - h, 0, a.shape[1] - CROP_XY))
    return x0, y0


def finish(img: np.ndarray, brain: np.ndarray, seg: np.ndarray | None, box):
    """img/brain/seg in atlas array space (x, y, z) -> saved arrays (z, y, x)."""
    import SimpleITK as sitk

    x0, y0 = box
    sl = (slice(x0, x0 + CROP_XY), slice(y0, y0 + CROP_XY))
    pad = ((0, 0), (0, 0), Z_PAD)
    img = np.pad(img[sl], pad)
    brain = np.pad(brain[sl], pad)
    seg = None if seg is None else np.pad(seg[sl], pad)

    # N4 on a 2x-shrunk copy, as usual, within the brain mask.
    im = sitk.GetImageFromArray(img.astype(np.float32))
    mk = sitk.GetImageFromArray(brain.astype(np.uint8))
    shrink = [2, 2, 2]
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.Execute(sitk.Shrink(im, shrink), sitk.Shrink(mk, shrink))
    logbias = n4.GetLogBiasFieldAsImage(im)
    img = sitk.GetArrayFromImage(im / sitk.Exp(logbias)) * brain

    def down(a):  # 2x2x2 mean pooling
        s = a.shape
        return a.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean(axis=(1, 3, 5))

    img, brain_f = down(img), down(brain.astype(np.float32))
    brain = brain_f > 0.5
    img = img * brain
    seg = None if seg is None else (down(seg.astype(np.float32)) > 0.5) & brain

    keep = slice(DROP, img.shape[2] - DROP)
    img, brain = img[..., keep], brain[..., keep]
    seg = None if seg is None else seg[..., keep]

    scale = np.percentile(img[brain], 99.5)
    img = np.clip(img / scale, 0, 1)
    t = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))  # noqa: E731
    out = {"img": t(img).astype(np.float16), "brain": t(brain)}
    if seg is not None:
        out["seg"] = t(seg)
    return out


def do_ixi(args):
    src, dst, box = args
    import ants

    case = src.name.replace(".nii.gz", "")
    out = dst / f"{case}.npz"
    if out.exists():
        return out.name, "exists"
    fixed = ants.image_read(str(ATLAS))
    if fixed.dimension == 4:
        fixed = ants.from_numpy(fixed.numpy().squeeze(), origin=fixed.origin[:3],
                                spacing=fixed.spacing[:3], direction=fixed.direction[:3, :3])
    moving = ants.image_read(str(src))  # HD-BET skull-stripped scan
    moving = moving.new_image_like(np.clip(moving.numpy(), 0, None))
    reg = ants.registration(fixed, moving, type_of_transform="Affine", aff_metric="mattes")
    warped = ants.apply_transforms(fixed, moving, reg["fwdtransforms"], interpolator="linear")
    img = np.clip(warped.numpy(), 0, None)
    brain = img > 0
    np.savez_compressed(out, **finish(img * brain, brain, None, box))
    return out.name, "ok"


def do_msd(args):
    src, dst, box = args
    import nibabel as nib

    out = dst / src.name.replace(".nii.gz", ".npz")
    if out.exists():
        return out.name, "exists"
    # MSD stores the BraTS arrays with an identity (RAS) affine; the atlas uses
    # LPS, so both in-plane axes are flipped to match the atlas array.
    img = np.asarray(nib.load(src).dataobj, dtype=np.float32)[::-1, ::-1, :, 3]
    lab = np.asarray(nib.load(src.parent.parent / "labelsTr" / src.name).dataobj)[::-1, ::-1]
    brain = img > 0
    np.savez_compressed(out, **finish(img, brain, lab > 0, box))
    return out.name, "ok"


def to_atlas_layout(im):
    """Array of a BraTS-space NIfTI reoriented to the atlas' LPS array layout."""
    import nibabel as nib
    from nibabel import orientations as o

    t = o.ornt_transform(o.io_orientation(im.affine), o.axcodes2ornt(("L", "P", "S")))
    return o.apply_orientation(np.asarray(im.dataobj, dtype=np.float32), t)


def do_brats21(args):
    src, dst, box = args  # src: BraTS2021_XXXXX case folder
    import nibabel as nib

    out = dst / f"{src.name}.npz"
    if out.exists():
        return out.name, "exists"
    img = to_atlas_layout(nib.load(src / f"{src.name}_t2.nii.gz"))
    lab = to_atlas_layout(nib.load(src / f"{src.name}_seg.nii.gz"))
    brain = img > 0
    np.savez_compressed(out, **finish(img, brain, lab > 0, box))
    return out.name, "ok"


def run(fn, jobs, workers):
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    with ProcessPoolExecutor(workers) as ex:
        for i, (name, status) in enumerate(ex.map(fn, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name} {status}", flush=True)


def check():
    """Alignment check: brain-mask Dice of each set's mean mask vs the atlas."""
    import nibabel as nib  # noqa: F401

    a = atlas_array()
    ref = finish(a, a > 0, None, crop_box())["brain"]
    for s in ["ixi", "msd", "brats21"]:
        files = sorted((DATA / "proc" / s).glob("*.npz"))
        if not files:
            continue
        dices = []
        for f in files:
            b = np.load(f)["brain"]
            dices.append(2 * (b & ref).sum() / (b.sum() + ref.sum()))
        d = np.array(dices)
        print(f"{s}: n={len(d)} brain-vs-atlas Dice mean {d.mean():.3f} min {d.min():.3f} "
              f"worst {[files[i].stem for i in np.argsort(d)[:5]]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("set", choices=["ixi", "msd", "brats21", "check"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.set == "check":
        return check()
    box = crop_box()
    dst = DATA / "proc" / a.set
    dst.mkdir(parents=True, exist_ok=True)
    if a.set == "ixi":
        srcs = sorted((DATA / "ixi_bet").glob("IXI*.nii.gz"))
        run(do_ixi, [(s, dst, box) for s in srcs], a.workers)
    elif a.set == "msd":
        srcs = sorted((DATA / "raw" / "Task01_BrainTumour" / "imagesTr").glob("BRATS_*.nii.gz"))
        run(do_msd, [(s, dst, box) for s in srcs], a.workers)
    else:
        srcs = sorted(p for p in (DATA / "raw" / "BraTS2021").glob("BraTS2021_*") if p.is_dir())
        run(do_brats21, [(s, dst, box) for s in srcs], a.workers)
    (DATA / "proc" / f"{a.set}_crop.json").write_text(json.dumps({"crop_xy0": box, "z_pad": Z_PAD}))


if __name__ == "__main__":
    main()
