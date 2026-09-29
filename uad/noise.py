"""Structured (simplex) noise for anomaly-detection diffusion models.

AnoDDPM (Wyatt et al., 2022) and pDDPM replace Gaussian noise with octaves of
simplex noise, which corrupts image structure at the scale of lesions. Here the
noise is drawn from a precomputed bank of 2D fields, built once with
opensimplex and augmented at sampling time with random flips, 90-degree
rotations, sign changes and random crops. As in the pDDPM reference code, the
fields are the raw octave sums (standard deviation about 0.55, not 1) and one
field is shared by all images of a batch.
"""
from pathlib import Path

import numpy as np
import torch


def simplex_octaves(size: int, rng: np.random.Generator, octaves: int = 6,
                    persistence: float = 0.8, frequency: float = 1 / 64) -> np.ndarray:
    """One size x size field: sum of octaves with doubling frequency."""
    import opensimplex

    opensimplex.seed(int(rng.integers(2**31)))
    coords = np.arange(size, dtype=np.float64)
    z = np.array([rng.uniform(0, 1000)])
    field = np.zeros((size, size))
    amp, freq = 1.0, frequency
    for _ in range(octaves):
        field += amp * opensimplex.noise3array(coords * freq, coords * freq, z)[0]
        amp *= persistence
        freq *= 2
    return field


def build_bank(path: Path, n: int, size: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    bank = np.stack([simplex_octaves(size, rng) for _ in range(n)]).astype(np.float16)
    np.save(path, bank)


class NoiseBank:
    def __init__(self, path: Path, device):
        self.bank = torch.from_numpy(np.load(path)).to(device)

    def sample(self, shape) -> torch.Tensor:
        """Noise of shape (B, C, H, W); H and W must not exceed the bank size."""
        b, c, h, w = shape
        idx = torch.randint(len(self.bank), (c,), device=self.bank.device)
        x = self.bank[idx].float()
        k = int(torch.randint(4, ()))
        x = torch.rot90(x, k, dims=(1, 2))
        if torch.rand(()) < 0.5:
            x = x.flip(2)
        dy = int(torch.randint(x.shape[1] - h + 1, ()))
        dx = int(torch.randint(x.shape[2] - w + 1, ()))
        x = x[:, dy:dy + h, dx:dx + w]
        sign = -1.0 if torch.rand(()) < 0.5 else 1.0
        return (x * sign)[None].expand(b, c, h, w)
