"""Show the tumour-free test slices with the most false-positive pixels under
the IXI-trained model: input, healthy reconstruction and anomaly map from
both models, thresholded at each model's strict threshold.

  third_party/REFLECT/.venv/bin/python scripts/fp_figure.py --ixi CKPT --brats CKPT \
      --data data/reflect_brats21_t2 --analysis results/fp_analysis --out results/fp_analysis/fp_slices.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from eval_reflect import load_model, slices

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def run(model, vae, imgs, steps, dev):
    from scipy.ndimage import gaussian_filter
    from skimage.transform import resize

    x = ((torch.from_numpy(np.stack(imgs)).float()[:, None] / 255 - 0.5) / 0.5).to(dev)
    enc = vae.encode(x).mean.mul_(0.18215)
    lat = enc.clone()
    for s in range(steps):
        lat = lat + model(lat, torch.full((len(x), 1), s / steps, device=dev)) / steps
    rec = vae.decode(lat / 0.18215)
    maps = []
    for xi, ii, ei, li in zip(x, rec, enc, lat):
        di = gaussian_filter(np.clip((ii - xi).abs().float().mean(0).cpu().numpy(), 0, 0.4) * 2.5, 3)
        dl = resize(gaussian_filter(np.clip((li - ei).abs().float().mean(0).cpu().numpy(), 0, 0.4) * 2.5, 1), (256, 256))
        maps.append(0.5 * di + 0.5 * dl)
    return ((rec[:, 0].float().cpu().numpy() + 1) / 2).clip(0, 1), np.stack(maps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ixi", required=True)
    p.add_argument("--brats", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = torch.device(a.device)
    items = slices(Path(a.data) / "test_all", "T2")
    an = Path(a.analysis)
    summ = json.loads((an / "summary.json").read_text())
    rows = list(csv.DictReader(open(an / "slices_ixi.csv")))
    idx = sorted((i for i, r in enumerate(rows) if r["tumour"] == "False"), key=lambda i: -int(rows[i]["fp_px"]))
    pick, seen = [], set()
    for i in idx:  # one slice per subject
        if items[i][0] not in seen:
            pick.append(i)
            seen.add(items[i][0])
        if len(pick) == a.n:
            break
    imgs = [items[i][1] for i in pick]
    brains = np.stack([items[i][2] for i in pick])
    out = {}
    for name, ck in (("ixi", a.ixi), ("brats", a.brats)):
        model, vae, _ = load_model(Path(ck), dev)
        rec, m = run(model, vae, imgs, 5, dev)
        out[name] = (rec, m * brains, summ[name]["threshold"])
        del model, vae
        torch.cuda.empty_cache()
    cols = ["input", "IXI recon.", "IXI map", "BraTS recon.", "BraTS map"]
    fig, ax = plt.subplots(len(pick), 5, figsize=(10, 2 * len(pick)))
    for r, i in enumerate(pick):
        ax[r, 0].imshow(imgs[r], cmap="gray", vmin=0, vmax=255)
        ax[r, 0].set_ylabel(items[i][0].replace("BraTS2021_", ""), fontsize=8)
        for c, name in ((1, "ixi"), (3, "brats")):
            rec, m, th = out[name]
            ax[r, c].imshow(rec[r], cmap="gray", vmin=0, vmax=1)
            ax[r, c + 1].imshow(imgs[r], cmap="gray", vmin=0, vmax=255)
            ax[r, c + 1].imshow(np.ma.masked_less_equal(m[r], th), cmap="autumn", alpha=0.8)
        for c in range(5):
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)


if __name__ == "__main__":
    main()
