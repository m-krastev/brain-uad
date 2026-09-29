"""Precompute the simplex noise bank (data/simplex_bank.npy), in parallel."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uad.noise import simplex_octaves  # noqa: E402

N, SIZE, WORKERS = 8000, 160, 12


def chunk(seed):
    rng = np.random.default_rng(seed)
    return np.stack([simplex_octaves(SIZE, rng) for _ in range(N // WORKERS + 1)]).astype(np.float16)


if __name__ == "__main__":
    with ProcessPoolExecutor(WORKERS) as ex:
        bank = np.concatenate(list(ex.map(chunk, range(WORKERS))))[:N]
    out = Path(__file__).resolve().parents[1] / "data" / "simplex_bank.npy"
    np.save(out, bank)
    print(out, bank.shape, float(bank.std()))
