"""Qualitative figure: input, reconstruction, anomaly map and reference for
test cases, one row per case and one reconstruction/map pair per method.

  python scripts/figure.py ae ddpm pddpm_k0 pddpm_k2 --out results/figure.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uad.data import ROOT, Volumes  # noqa: E402
from uad.methods import Method  # noqa: E402
from uad.noise import NoiseBank  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--cases", type=int, default=4)
    p.add_argument("--out", default=str(ROOT / "results" / "figure.png"))
    a = p.parse_args()
    dev = torch.device("cuda")
    test = Volumes("msd_test", dev)
    rng = np.random.default_rng(1)
    picks = rng.choice(len(test), a.cases, replace=False)
    zs = [int(test.seg[v].sum((1, 2)).argmax()) for v in picks]
    bank = NoiseBank(ROOT / "data" / "simplex_bank.npy", dev)

    cols = 2 + 2 * len(a.runs)
    fig, ax = plt.subplots(a.cases, cols, figsize=(1.8 * cols, 1.9 * a.cases))
    for j, run in enumerate(a.runs):
        args = json.loads((ROOT / "results" / run / "args.json").read_text())
        m = Method(args["method"], args["k"], args["patch"], noise_bank=bank).to(dev).eval()
        m.net.load_state_dict(torch.load(ROOT / "results" / run / "best.pt", map_location=dev))
        k = args["k"]
        for i, (v, z) in enumerate(zip(picks, zs)):
            zz = torch.tensor([z], device=dev)
            x = test.stack(torch.tensor([int(v)], device=dev), zz, k)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                rec = m.reconstruct(x, 500).float()
            xc = x[0, k].cpu().numpy()
            r = rec[0, 0].cpu().numpy()
            brain = test.brain[v, z].cpu().numpy()
            amap = ndimage.median_filter(np.abs(r - xc) * brain, size=5)
            if j == 0:
                ax[i, 0].imshow(xc, cmap="gray", vmin=0, vmax=1)
                ax[i, 1].imshow(test.seg[v, z].cpu().numpy(), cmap="gray")
            ax[i, 2 + 2 * j].imshow(r, cmap="gray", vmin=0, vmax=1)
            ax[i, 3 + 2 * j].imshow(amap, cmap="inferno", vmin=0, vmax=0.5)
    titles = ["input", "reference"] + sum([[f"{r}\nrecon.", f"{r}\nerror"] for r in a.runs], [])
    for c, t in enumerate(titles):
        ax[0, c].set_title(t, fontsize=8)
    for x in ax.ravel():
        x.axis("off")
    plt.tight_layout()
    plt.savefig(a.out, dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()
