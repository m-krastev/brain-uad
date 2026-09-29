"""Preprocessed volumes (data/proc/<set>/*.npz) and fixed splits."""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "proc"
SPLITS = ROOT / "splits.json"


def make_splits(seed: int = 0) -> dict:
    """IXI: 158 test, 44 validation, rest training (as pDDPM's sizes, one fold).
    MSD and BraTS 2021: 100 unhealthy validation cases for threshold selection,
    rest test (pDDPM: 100 / 1151 on BraTS 2021)."""
    rng = np.random.default_rng(seed)
    ixi = sorted(p.stem for p in (PROC / "ixi").glob("*.npz"))
    msd = sorted(p.stem for p in (PROC / "msd").glob("*.npz"))
    ixi = list(rng.permutation(ixi))
    msd = list(rng.permutation(msd))
    brats = sorted(p.stem for p in (PROC / "brats21").glob("*.npz"))
    brats = list(np.random.default_rng(seed).permutation(brats))
    splits = {
        "ixi_test": sorted(ixi[:158]), "ixi_val": sorted(ixi[158:202]), "ixi_train": sorted(ixi[202:]),
        "msd_val": sorted(msd[:100]), "msd_test": sorted(msd[100:]),
        "brats_val": sorted(brats[:100]), "brats_test": sorted(brats[100:]),
    }
    SPLITS.write_text(json.dumps(splits, indent=1))
    return splits


def load_split(name: str):
    splits = json.loads(SPLITS.read_text())
    s = {"ixi": "ixi", "msd": "msd", "brats": "brats21"}[name.split("_")[0]]
    def read(c):
        with np.load(PROC / s / f"{c}.npz") as z:
            return {k: z[k] for k in z.files}

    return splits[name], [read(c) for c in splits[name]]


def rescale(img: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Map the brain's 1st-99th intensity percentiles to [0, 1] and clip, as the
    pDDPM reference code does. The stored volumes are clipped at the 99.5th
    percentile, above the 99th, so this is exact."""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x[brain], [1, 99])
    return (np.clip((x - lo) / (hi - lo), 0, 1) * brain).astype(np.float16)


class Volumes:
    """All volumes of a split stacked on the GPU: img (N, Z, H, W)."""

    def __init__(self, name: str, device):
        self.cases, arrs = load_split(name)
        self.img = torch.from_numpy(np.stack([rescale(a["img"], a["brain"]) for a in arrs])).to(device)
        self.brain = torch.from_numpy(np.stack([a["brain"] for a in arrs])).to(device)
        self.seg = (torch.from_numpy(np.stack([a["seg"] for a in arrs])).to(device)
                    if "seg" in arrs[0] else None)

    def __len__(self):
        return len(self.cases)

    def stack(self, vol: torch.Tensor, z: torch.Tensor, k: int) -> torch.Tensor:
        """Slices z-k..z+k of volumes vol as channels (B, 2k+1, H, W); zero outside."""
        img = self.img
        Z = img.shape[1]
        offs = torch.arange(-k, k + 1, device=img.device)
        zz = z[:, None] + offs[None]
        valid = (zz >= 0) & (zz < Z)
        out = img[vol[:, None], zz.clamp(0, Z - 1)].float()
        return out * valid[..., None, None]

    def random_batch(self, b: int, k: int) -> torch.Tensor:
        vol = torch.randint(len(self), (b,), device=self.img.device)
        z = torch.randint(self.img.shape[1], (b,), device=self.img.device)
        return self.stack(vol, z, k)
