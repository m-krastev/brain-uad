"""Compare models at matched false-alarm rates rather than at each model's
Dice-optimal threshold. Run with REFLECT's env:

  third_party/REFLECT/.venv/bin/python scripts/operating_points.py \
      --model brats=CKPT --model ixi=CKPT --data data/reflect_brats21_t2 \
      --out results/operating_points.json

For each model, anomaly maps (5 steps, restricted to the brain) are computed
on the validation and test subjects' 20 central slices. Thresholds are fixed
on the validation subjects only: the Dice-optimal one, and the lowest ones
whose fraction of tumour-free validation slices with any positive pixel stays
at or below 5% and 10%. Each threshold is then applied to the test slices.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from eval_reflect import anomaly_maps, dice, load_model, slices, sweep


def stats(maps, segs, th):
    pred = maps > th
    t = segs.any(axis=(1, 2))
    return dict(threshold=float(th), pooled_dice=float(dice(pred, segs)),
                mean_dice_tumour_slices=float(np.mean([dice(p, g) for p, g in zip(pred[t], segs[t])])),
                fp_rate_normal_slices=float(pred[~t].any(axis=(1, 2)).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="name=CKPT")
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    data = Path(a.data)
    val, test = slices(data / "val_all", "T2"), slices(data / "test_all", "T2")
    vb, tb = np.stack([i[2] for i in val]), np.stack([i[2] for i in test])
    vs, ts = np.stack([i[3] for i in val]), np.stack([i[3] for i in test])
    vnormal = ~vs.any(axis=(1, 2))
    ths = np.linspace(0, 1, 401)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for spec in a.model:
        name, ck = spec.split("=", 1)
        model, vae, _ = load_model(Path(ck), dev)
        vm = anomaly_maps(model, vae, val, a.steps, dev) * vb
        tm = anomaly_maps(model, vae, test, a.steps, dev) * tb
        del model, vae
        torch.cuda.empty_cache()
        vmax = vm[vnormal].reshape(vnormal.sum(), -1).max(1)  # a normal slice fires iff its max exceeds th
        r = {"n_val_normal_slices": int(vnormal.sum()),
             "auprc_pooled": float(average_precision_score(ts[tb], tm[tb])),
             "dice_optimal": stats(tm, ts, sweep(vm, vs, ths, per_slice=False)[1])}
        for cap in (0.05, 0.10):
            th = next(t for t in ths if (vmax > t).mean() <= cap)
            r[f"val_fp_{int(cap * 100)}pct"] = stats(tm, ts, th)
        res[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
