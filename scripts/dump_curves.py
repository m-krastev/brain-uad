"""Save everything needed to re-threshold a model offline. Run with REFLECT's env:

  third_party/REFLECT/.venv/bin/python scripts/dump_curves.py \
      --model ixi=CKPT --data data/reflect_brats21_t2 --out results/curves

For each model, the 5-step anomaly maps of the validation and test subjects'
20 central slices (restricted to the brain) are reduced to per-slice
histograms on a 401-step threshold grid: for every slice and threshold, the
number of tumour and non-tumour brain pixels whose score exceeds it. Any
threshold rule, Dice aggregate, false-alarm rate or pixel-level precision-recall
curve can then be computed on the CPU, and subjects can be resampled for
bootstrap intervals (scripts/curves.py). The test maps and the healthy
reconstructions are also stored, quantised to 8 bits, for figures.

  OUT/NAME.npz   val_/test_ subject, tumour (bool per slice), pos/neg
                 (slices x 401 counts of tumour / non-tumour pixels > th),
                 n_pos, n_neg, n_seg, auprc (exact pixel-level, test)
  OUT/NAME_maps.npz   test anomaly maps and reconstructions (uint8)
  OUT/test_inputs.npz test images, brain masks and tumour masks
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.metrics import average_precision_score

from eval_reflect import load_model, slices

THS = np.linspace(0, 1, 401)


@torch.no_grad()
def maps_and_recons(model, vae, items, steps, dev, bs=32):
    """REFLECT's anomaly map (as in eval_reflect.anomaly_maps) plus the decoded
    reconstruction, i.e. the model's healthy version of the slice."""
    maps, recs = [], []
    for i in range(0, len(items), bs):
        x = torch.from_numpy(np.stack([it[1] for it in items[i:i + bs]])).float()[:, None] / 255
        x = ((x - 0.5) / 0.5).to(dev)
        enc = vae.encode(x).mean.mul_(0.18215)
        lat = enc.clone()
        dt = 1 / steps
        for s in range(steps):
            t = torch.full((x.shape[0], 1), s * dt, device=dev)
            lat = lat + model(lat, t) * dt
        img = vae.decode(lat / 0.18215)
        for xi, ii, ei, li in zip(x, img, enc, lat):
            di = (ii - xi).abs().float().mean(0).cpu().numpy()
            di = gaussian_filter(np.clip(di, 0, 0.4) * 2.5, sigma=3)
            dl = (li - ei).abs().float().mean(0).cpu().numpy()
            dl = resize(gaussian_filter(np.clip(dl, 0, 0.4) * 2.5, sigma=1), (256, 256))
            maps.append(0.5 * di + 0.5 * dl)
            recs.append(((ii.float().mean(0).cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
    return np.stack(maps), np.stack(recs)


def counts(maps, brain, seg):
    """Per slice and threshold: tumour and non-tumour brain pixels scoring > th."""
    pos = np.zeros((len(maps), len(THS)), np.int32)
    neg = np.zeros_like(pos)
    for k, (m, b, g) in enumerate(zip(maps, brain, seg)):
        for arr, mask in ((pos, b & g), (neg, b & ~g)):
            v = np.sort(m[mask])
            arr[k] = len(v) - np.searchsorted(v, THS, side="right")
    return pos, neg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="name=CKPT")
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--no-maps", action="store_true", help="skip saving test maps and reconstructions")
    a = p.parse_args()
    dev = torch.device("cuda")
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {"val": slices(data / "val_all", "T2"), "test": slices(data / "test_all", "T2")}
    if not (out / "test_inputs.npz").exists():
        t = sets["test"]
        np.savez_compressed(out / "test_inputs.npz", subject=np.array([i[0] for i in t]),
                            image=np.stack([i[1] for i in t]), brain=np.stack([i[2] for i in t]),
                            seg=np.stack([i[3] for i in t]))
    for spec in a.model:
        name, ck = spec.split("=", 1)
        model, vae, _ = load_model(Path(ck), dev)
        res = {"ckpt": ck, "thresholds": THS}
        for split, items in sets.items():
            brain = np.stack([i[2] for i in items])
            seg = np.stack([i[3] for i in items])
            m, r = maps_and_recons(model, vae, items, a.steps, dev)
            m = m * brain
            pos, neg = counts(m, brain, seg)
            res.update({f"{split}_subject": np.array([i[0] for i in items]),
                        f"{split}_tumour": seg.any(axis=(1, 2)),
                        f"{split}_pos": pos, f"{split}_neg": neg,
                        f"{split}_n_pos": (brain & seg).sum(axis=(1, 2)),
                        f"{split}_n_neg": (brain & ~seg).sum(axis=(1, 2)),
                        # tumour pixels including any outside the brain mask,
                        # which count as misses as in eval_reflect / operating_points
                        f"{split}_n_seg": seg.sum(axis=(1, 2))})
            if split == "test":
                res["test_auprc"] = average_precision_score(seg[brain], m[brain])
                if not a.no_maps:
                    np.savez_compressed(out / f"{name}_maps.npz",
                                        subject=res["test_subject"],
                                        maps=(m.clip(0, 1) * 255).round().astype(np.uint8),
                                        recons=r)
        del model, vae
        torch.cuda.empty_cache()
        np.savez_compressed(out / f"{name}.npz", **res)
        print(name, "test AUPRC", round(float(res["test_auprc"]), 4), flush=True)


if __name__ == "__main__":
    main()
