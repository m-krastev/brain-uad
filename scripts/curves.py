"""Offline analysis of the per-slice counts written by dump_curves.py: test
metrics at thresholds fixed on the validation subjects, subject-level bootstrap
confidence intervals, paired differences between models, and the Dice /
false-alarm trade-off over all thresholds. CPU only:

  uv run --no-project --with numpy,matplotlib python scripts/curves.py \
      --curves results/curves --out reported/curves

Definitions (as in eval_reflect.py and operating_points.py):
  pooled Dice       2 TP / (predicted + tumour pixels) over all test slices
  mean Dice         mean per-slice Dice over test slices that contain tumour
  false alarms      fraction of tumour-free test slices with any positive pixel
  AUPRC             pixel-level average precision inside the brain (the
                    bootstrap uses the 401-threshold step approximation; the
                    point estimate is exact)
Thresholds come from the validation subjects only, except for "at_test_fp",
which picks the lowest threshold keeping at most 5/10/20% of tumour-free *test*
slices firing (re-chosen in every bootstrap resample): a comparison at equal
false-alarm rates that does not depend on how well validation thresholds
transfer. Otherwise: the one maximising pooled
validation Dice ("dice_opt"), and the lowest ones keeping at most 5% / 10% of
tumour-free validation slices firing ("val_fp5", "val_fp10"). Confidence
intervals resample test subjects (all 20 slices of a subject together), with
the validation thresholds held fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

B = 2000
RULES = ("dice_opt", "val_fp5", "val_fp10")


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def thresholds(d):
    pos, neg, nseg = d["val_pos"], d["val_neg"], d["val_n_seg"]
    pooled = 2 * pos.sum(0) / (pos.sum(0) + neg.sum(0) + nseg.sum())
    normal = ~d["val_tumour"]
    fires = (neg[normal] > 0).mean(0)  # per threshold
    th = {"dice_opt": int(np.argmax(pooled))}
    for cap, k in ((0.05, "val_fp5"), (0.10, "val_fp10")):
        th[k] = int(np.argmax(fires <= cap))  # first (lowest) threshold meeting the cap
    return th


def per_subject(d):
    """Sum per-slice quantities into per-subject arrays (S x 401) so a bootstrap
    resample is a weighted sum."""
    subj = d["test_subject"]
    names, inv = np.unique(subj, return_inverse=True)
    S = len(names)
    pos, neg = d["test_pos"].astype(np.float64), d["test_neg"].astype(np.float64)
    tum = d["test_tumour"]
    nseg = d["test_n_seg"].astype(np.float64)
    slice_dice = 2 * pos / np.maximum(pos + neg + nseg[:, None], 1)
    agg = lambda x: np.stack([np.bincount(inv, weights=x[:, j], minlength=S) for j in range(x.shape[1])], 1)
    out = {
        "names": names,
        "pos": agg(pos), "neg": agg(neg),
        "pred_px": agg(pos + neg),
        "n_seg": np.bincount(inv, weights=nseg, minlength=S),
        "n_pos": np.bincount(inv, weights=d["test_n_pos"].astype(np.float64), minlength=S),
        "dice_sum": agg(np.where(tum[:, None], slice_dice, 0)),
        "n_tum": np.bincount(inv, weights=tum.astype(float), minlength=S),
        "fire": agg(((neg > 0) & ~tum[:, None]).astype(float)),
        "n_norm": np.bincount(inv, weights=(~tum).astype(float), minlength=S),
    }
    return out


def metrics(s, w, j=None):
    """Metrics for subject weights w (B x S or S); at threshold index j, or all."""
    sl = slice(None) if j is None else j
    tp = w @ s["pos"][:, sl]
    pred = w @ s["pred_px"][:, sl]
    nseg = w @ s["n_seg"]
    ntum, nnorm = w @ s["n_tum"], w @ s["n_norm"]
    if j is None:
        nseg, ntum, nnorm = nseg[..., None], ntum[..., None], nnorm[..., None]
    return {"pooled_dice": 2 * tp / (pred + nseg),
            "mean_dice": (w @ s["dice_sum"][:, sl]) / ntum,
            "fp_rate": (w @ s["fire"][:, sl]) / nnorm}


def auprc(s, w):
    tp, fp = w @ s["pos"], w @ s["neg"]
    npos = (w @ s["n_pos"])[..., None]
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    rec = tp / npos
    # thresholds ascend, so recall descends: AP = sum over steps of dR * P
    drec = rec[..., :-1] - rec[..., 1:]
    return (drec * prec[..., :-1]).sum(-1) + rec[..., -1] * prec[..., -1]


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for f in sorted(Path(a.curves).glob("*.npz")):
        if f.stem.endswith("_maps") or f.stem == "test_inputs":
            continue
        models[f.stem] = load(f)
    names = list(models)
    subj = [per_subject(models[n]) for n in names]
    S = len(subj[0]["names"])
    for s in subj:
        assert (s["names"] == subj[0]["names"]).all()
    rng = np.random.default_rng(a.seed)
    W = rng.multinomial(S, np.full(S, 1 / S), size=B).astype(np.float64)
    one = np.ones(S)

    res, boot = {}, {}
    for n, s in zip(names, subj):
        d = models[n]
        th = thresholds(d)
        r = {"n_test_subjects": S, "auprc": float(d["test_auprc"]),
             "auprc_ci": ci(auprc(s, W))}
        boot[n] = {"auprc": auprc(s, W)}
        for rule in RULES:
            j = th[rule]
            pt = metrics(s, one, j)
            bs = metrics(s, W, j)
            r[rule] = {"threshold": float(d["thresholds"][j])}
            for k in pt:
                r[rule][k] = float(pt[k])
                r[rule][k + "_ci"] = ci(bs[k])
                boot[n][f"{rule}/{k}"] = bs[k]
        # Dice at a fixed test false-alarm rate (threshold chosen on the test
        # set itself: an oracle that removes the validation-to-test transfer)
        full = metrics(s, one)
        fullb = metrics(s, W)
        r["at_test_fp"] = {}
        for cap in (0.05, 0.10, 0.20):
            key = f"{int(cap * 100)}pct"
            j = int(np.argmax(full["fp_rate"] <= cap))
            jb = np.argmax(fullb["fp_rate"] <= cap, axis=1)  # per resample
            r["at_test_fp"][key] = {"threshold": float(d["thresholds"][j])}
            for k in ("pooled_dice", "mean_dice"):
                vb = fullb[k][np.arange(B), jb]
                r["at_test_fp"][key][k] = float(full[k][j])
                r["at_test_fp"][key][k + "_ci"] = ci(vb)
                boot[n][f"test_fp{int(cap * 100)}/{k}"] = vb
            r["at_test_fp"][key]["fp_rate"] = float(full["fp_rate"][j])
        res[n] = r
        np.savez_compressed(out / f"curve_{n}.npz", thresholds=d["thresholds"], **full)

    # paired differences (same resampled subjects for both models)
    pairs = [(x, y) for y in ("brats", "ixi") if y in names for x in names if x != y]
    diffs = {}
    for x, y in pairs:
        diffs[f"{x} - {y}"] = {}
        for k in boot[x]:
            dlt = boot[x][k] - boot[y][k]
            diffs[f"{x} - {y}"][k] = {"ci": ci(dlt), "p_le_0": float((dlt <= 0).mean())}
    (out / "summary.json").write_text(json.dumps({"models": res, "paired_differences": diffs}, indent=1))

    # markdown table
    lines = ["| Model | AUPRC | Dice-optimal: pooled Dice / FA | ≤5% val FA: pooled Dice / mean Dice / test FA "
             "| 5% test FA: pooled / mean Dice |",
             "|---|---|---|---|---|"]
    f = lambda v, c, s=100: f"{v * s:.1f} [{c[0] * s:.1f}, {c[1] * s:.1f}]"
    for n in names:
        r = res[n]
        do, v5, t5 = r["dice_opt"], r["val_fp5"], r["at_test_fp"]["5pct"]
        lines.append(f"| {n} | {r['auprc']:.3f} [{r['auprc_ci'][0]:.3f}, {r['auprc_ci'][1]:.3f}] | "
                     f"{f(do['pooled_dice'], do['pooled_dice_ci'])} / {f(do['fp_rate'], do['fp_rate_ci'])} | "
                     f"{f(v5['pooled_dice'], v5['pooled_dice_ci'])} / {f(v5['mean_dice'], v5['mean_dice_ci'])} / "
                     f"{f(v5['fp_rate'], v5['fp_rate_ci'])} | "
                     f"{f(t5['pooled_dice'], t5['pooled_dice_ci'])} / {f(t5['mean_dice'], t5['mean_dice_ci'])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
