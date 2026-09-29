"""Score a trained REFLECT model under two protocols. Run with REFLECT's env:

  third_party/REFLECT/.venv/bin/python scripts/eval_reflect.py \
      --ckpt third_party/REFLECT/REFLECT_.../005-UNet_M-T2/checkpoints/last.pt \
      --data data/reflect_brats21_t2 --steps 1 --out results/reflect_brats21

The anomaly map is REFLECT's own (evaluate_REFLECT.py): half the clipped,
smoothed image difference plus half the clipped, smoothed latent difference.

  reflect  REFLECT's protocol: one slice per test subject (largest tumour),
           threshold swept on the test slices themselves. Reports the mean
           per-slice Dice at the best common threshold ("max Dice") and the
           pooled Dice at the best threshold ("global max Dice").
  strict   all 20 central slices of every test subject, with and without
           tumour; the threshold maximising pooled Dice on the validation
           subjects' slices is applied to the test slices. Reports pooled Dice,
           mean Dice over tumour slices, pooled AUPRC, and the fraction of
           tumour-free slices with any false-positive pixel.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
REFLECT = ROOT / "third_party" / "REFLECT"
sys.path.insert(0, str(REFLECT))


def load_model(ckpt: Path, dev):
    import yaml
    from huggingface_hub import hf_hub_download
    from medical_models import UNET_models

    cfg = yaml.safe_load((ckpt.parents[2] / "args.yml").read_text())
    vae_file = {"kl_f8": "VAE-Medical-klf8.pt", "kl_f4": "VAE-Medical-klf4.pt"}[cfg["vae"]]
    vae = torch.load(hf_hub_download("farzadbz/Medical-VAE", vae_file), weights_only=False,
                     map_location="cpu").to(dev).eval()
    ch = 4 if cfg["vae"] == "kl_f8" else 3
    model = UNET_models[cfg["model"]](in_channels=ch, out_channels=ch)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location="cpu")["model"])
    return model.to(dev).eval(), vae, cfg


def slices(d: Path, modality: str):
    from PIL import Image, ImageOps

    out = []
    for f in sorted(d.glob(f"*-{modality}.png")):
        def rd(suffix):
            im = Image.open(str(f).replace(f"-{modality}", suffix))
            return np.array(ImageOps.pad(im, (256, 256), color="#000").convert("L"))
        out.append((f.name.split("-slice_")[0], rd(f"-{modality}"), rd("-brainmask") > 0,
                    rd("-segmentation") > 0))
    return out


@torch.no_grad()
def anomaly_maps(model, vae, items, steps, dev, bs=32):
    maps = []
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
    return np.stack(maps)


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2 * (p & g).sum() / s


def sweep(maps, segs, ths, per_slice):
    best = (-1, None)
    for th in ths:
        if per_slice:
            v = float(np.mean([dice(m > th, g) for m, g in zip(maps, segs)]))
        else:
            v = dice(maps > th, segs)
        best = max(best, (v, th))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    model, vae, cfg = load_model(Path(a.ckpt), dev)
    data, mod = Path(a.data), cfg["modality"]
    ths = np.linspace(0, 1, 201)
    res = {"ckpt": a.ckpt, "data": a.data, "steps": a.steps}

    test = slices(data / "test", mod)
    tm = anomaly_maps(model, vae, test, a.steps, dev)
    ts = np.stack([it[3] for it in test])
    res["reflect_max_dice"], _ = sweep(tm, ts, ths, per_slice=True)
    res["reflect_global_max_dice"], _ = sweep(tm, ts, ths, per_slice=False)
    res["reflect_n"] = len(test)

    val = slices(data / "val_all", mod)
    vm = anomaly_maps(model, vae, val, a.steps, dev)
    vs = np.stack([it[3] for it in val])
    vb = np.stack([it[2] for it in val])
    res["val_pooled_dice"], th = sweep(vm * vb, vs, ths, per_slice=False)
    res["threshold"] = float(th)

    tall = slices(data / "test_all", mod)
    am = anomaly_maps(model, vae, tall, a.steps, dev)
    ab = np.stack([it[2] for it in tall])
    am = am * ab  # score only inside the brain
    gs = np.stack([it[3] for it in tall])
    pred = am > th
    tumour = gs.any(axis=(1, 2))
    res.update(
        strict_pooled_dice=dice(pred, gs),
        strict_mean_dice_tumour_slices=float(np.mean([dice(p_, g) for p_, g in zip(pred[tumour], gs[tumour])])),
        strict_auprc_pooled=float(average_precision_score(gs[ab], am[ab])),
        strict_fp_rate_normal_slices=float(pred[~tumour].any(axis=(1, 2)).mean()),
        strict_n_slices=int(len(tall)), strict_n_tumour_slices=int(tumour.sum()),
        oracle_pooled_dice_test_all=sweep(am, gs, ths, per_slice=False)[0],
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_steps{a.steps}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
