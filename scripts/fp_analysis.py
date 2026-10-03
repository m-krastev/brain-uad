"""Where do a REFLECT model's false alarms come from? Run with REFLECT's env:

  third_party/REFLECT/.venv/bin/python scripts/fp_analysis.py \
      --model ixi=CKPT:results/reflect_ixi/eval_steps5.json \
      --model brats=CKPT:results/reflect_brats21/eval_steps5.json \
      --data data/reflect_brats21_t2 --steps 5 --out results/fp_analysis

For every model, the strict-protocol threshold from its evaluation JSON is
applied to all test_all slices. Per slice it records the false-positive pixels
(prediction outside the tumour mask), how many lie within 8 px of the brain
border, and their mean intensity. Per test subject it measures the through-plane
sharpness of the original BraTS T2 volume: the mean absolute intensity
difference between neighbouring voxels along each axis inside the brain. A
thick-slice acquisition resampled to 1 mm has a much smaller difference along
its slice axis than within the slice, so min/max over the three axes is a
resolution proxy (about 1 for isotropic scans, lower for thick slices).
"""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from eval_reflect import ROOT, anomaly_maps, load_model, slices


def sharpness(case: str):
    v = nib.load(ROOT / "data/raw/BraTS2021" / case / f"{case}_t2.nii.gz").get_fdata(dtype=np.float32)
    brain = v > 0
    lo, hi = np.percentile(v[brain], [1, 99])
    v = np.clip((v - lo) / (hi - lo), 0, 1)
    d = []
    for ax in range(3):
        both = brain & np.roll(brain, 1, axis=ax)
        d.append(float(np.abs(v - np.roll(v, 1, axis=ax))[both].mean()))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="name=CKPT:EVAL_JSON")
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    items = slices(Path(a.data) / "test_all", "T2")
    subjects = sorted({it[0] for it in items})
    sharp_file = out / "sharpness.json"
    if sharp_file.exists():
        sharp = json.loads(sharp_file.read_text())
    else:
        sharp = {s: sharpness(s) for s in subjects}
        sharp_file.write_text(json.dumps(sharp))
    ratio = {s: min(d) / max(d) for s, d in sharp.items()}
    axis = {s: int(np.argmin(d)) for s, d in sharp.items()}

    summary = {"n_subjects": len(subjects),
               "slice_axis_counts": {str(k): sum(v == k for v in axis.values()) for k in range(3)},
               "ratio_quartiles": np.percentile(list(ratio.values()), [25, 50, 75]).tolist()}
    border = [distance_transform_edt(it[2]) <= 8 for it in items]
    for spec in a.model:
        name, rest = spec.split("=", 1)
        ckpt, ev = rest.rsplit(":", 1)
        th = json.loads(Path(ev).read_text())["threshold"]
        model, vae, _ = load_model(Path(ckpt), dev)
        maps = anomaly_maps(model, vae, items, a.steps, dev) * np.stack([it[2] for it in items])
        del model, vae
        torch.cuda.empty_cache()
        rows = []
        for it, m, b in zip(items, maps, border):
            fp = (m > th) & ~it[3]
            rows.append(dict(subject=it[0], tumour=bool(it[3].any()), fp_px=int(fp.sum()),
                             fp_border_px=int((fp & b).sum()),
                             fp_mean_intensity=float(it[1][fp].mean() / 255) if fp.any() else None,
                             brain_mean_intensity=float(it[1][it[2]].mean() / 255)))
        with open(out / f"slices_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        normal = [r for r in rows if not r["tumour"]]
        per_subj = {}
        for r in normal:
            per_subj.setdefault(r["subject"], []).append(r["fp_px"] > 0)
        subs = sorted(per_subj)
        rate = np.array([np.mean(per_subj[s]) for s in subs])
        rr = np.array([ratio[s] for s in subs])
        rho, pval = spearmanr(rr, rate)
        q = np.percentile(rr, [25, 50, 75])
        bins = np.digitize(rr, q)
        fp_all = sum(r["fp_px"] for r in rows)
        fpi = [r["fp_mean_intensity"] for r in normal if r["fp_mean_intensity"] is not None]
        summary[name] = dict(
            threshold=th,
            fp_slice_rate_normal=float(np.mean([r["fp_px"] > 0 for r in normal])),
            n_normal_slices=len(normal), n_subjects_with_normal_slices=len(subs),
            fp_border_fraction=sum(r["fp_border_px"] for r in rows) / max(fp_all, 1),
            fp_mean_intensity_normal=float(np.mean(fpi)) if fpi else None,
            brain_mean_intensity_normal=float(np.mean([r["brain_mean_intensity"] for r in normal])),
            spearman_sharpness_vs_subject_fp_rate=[float(rho), float(pval)],
            fp_rate_by_sharpness_quartile=[float(rate[bins == k].mean()) for k in range(4)],
            fp_rate_by_slice_axis={str(k): float(rate[[axis[s] == k for s in subs]].mean())
                                   for k in range(3) if any(axis[s] == k for s in subs)},
        )
        print(name, json.dumps(summary[name], indent=1), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
