"""Figures for the README and report, from results/curves (dump_curves.py),
reported/curves (curves.py) and the reported/*.json evaluations. CPU only:

  uv run --no-project --with numpy,matplotlib,scipy python scripts/paper_figures.py \
      --curves results/curves --summary reported/curves --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"brats": "BraTS healthy slices (in-domain)", "ixi": "IXI (other hospitals)",
         "ixi_thick": "IXI + thick-slice augmentation", "ft_mix0": "IXI, +10 epochs IXI only",
         "ft_mix5": "IXI, fine-tuned with 5 local subjects", "ft_mix20": "IXI, fine-tuned with 20 local subjects",
         "ft_mix50": "IXI, fine-tuned with 50 local subjects"}
COLOR = {"brats": "#222222", "ixi": "#d55e00", "ixi_thick": "#cc79a7", "ft_mix0": "#999999",
         "ft_mix5": "#9ecae1", "ft_mix20": "#4292c6", "ft_mix50": "#08519c"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def tradeoff(summary, curves_dir, out, names):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for n in names:
        c = np.load(curves_dir / f"curve_{n}.npz")
        r = summary["models"][n]
        for ax, k in zip(axes, ("pooled_dice", "mean_dice")):
            ax.plot(c["fp_rate"] * 100, c[k] * 100, color=COLOR[n], lw=1.6 if n in ("brats", "ixi") else 1.2,
                    label=LABEL[n])
            ax.plot(r["val_fp5"]["fp_rate"] * 100, r["val_fp5"][k] * 100, "o", color=COLOR[n], ms=4.5)
            ax.plot(r["dice_opt"]["fp_rate"] * 100, r["dice_opt"][k] * 100, "x", color=COLOR[n], ms=5)
    for ax, t in zip(axes, ("Pooled Dice (%)", "Mean Dice over tumour slices (%)")):
        ax.set_xlim(0, 50)
        ax.set_xlabel("Tumour-free slices with a false alarm (%)")
        ax.set_ylabel(t)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylim(40, 85)
    axes[1].set_ylim(25, 70)
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="", marker="o", color="k", ms=4.5), plt.Line2D([], [], ls="", marker="x", color="k", ms=5)]
    l += ["threshold for ≤5% validation false alarms", "Dice-optimal validation threshold"]
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
    fig.suptitle("Dice against false alarms over all thresholds (BraTS 2021 test subjects, T2)", fontsize=9.5)
    save(fig, out, "tradeoff")


def local_finetuning(summary, out):
    """Paired bootstrap differences from the IXI model of the same training
    seed: the shared subject variance cancels, so these intervals are much
    tighter than per-model ones. One line per training seed."""
    m, dif = summary["models"], summary["paired_differences"]
    ns = [0, 5, 20, 50]
    seeds = [(10, "ixi", "ft_mix{}"), (11, "ixi_s11", "ft_s11_mix{}")]
    seeds = [x for x in seeds if x[1] in m]
    panels = (("dice_opt/fp_rate", "dice_opt", "fp_rate", "False alarms at the Dice-optimal\nvalidation threshold, change from IXI (pp)"),
              ("test_fp10/pooled_dice", ("at_test_fp", "10pct"), "pooled_dice",
               "Pooled Dice at 10% test false alarms,\nchange from IXI (pp)"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    shades = {10: COLOR["ft_mix50"], 11: COLOR["ft_mix20"]}
    for ax, (dk, rule, k, title) in zip(axes, panels):
        get = (lambda x: m[x][rule][k]) if isinstance(rule, str) else (lambda x: m[x][rule[0]][rule[1]][k])
        for i, (sd, base, pat) in enumerate(seeds):
            xs = [j for j, n in enumerate(ns) if pat.format(n) in m]
            keys = [pat.format(ns[j]) for j in xs]
            y = np.array([get(x) - get(base) for x in keys]) * 100
            lo = np.array([dif[f"{x} - {base}"][dk]["ci"][0] for x in keys]) * 100
            hi = np.array([dif[f"{x} - {base}"][dk]["ci"][1] for x in keys]) * 100
            off = (i - (len(seeds) - 1) / 2) * 0.12
            ax.errorbar(np.array(xs) + off, y, yerr=[y - lo, hi - y], fmt="o-", color=shades[sd], capsize=3, lw=1.4,
                        label=f"IXI model (training seed {sd}) fine-tuned 10 more epochs")
        ax.axhline(0, color=COLOR["ixi"], ls=":", lw=1, label="IXI model (same seed)")
        ax.set_xticks(range(4), [str(n) for n in ns])
        ax.set_xlabel("Local (BraTS) subjects in the fine-tuning data")
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.17))
    fig.text(0.5, -0.22, "Points: test-set difference; bars: 95% paired bootstrap interval over test subjects. "
             "0 local subjects = the same extra training on IXI only.", ha="center", fontsize=7)
    save(fig, out, "local_finetuning")


def protocol(out):
    rep = ROOT / "reported"
    b = json.loads((rep / "reflect_brats21_steps5.json").read_text())
    i = json.loads((rep / "reflect_ixi_to_brats21_steps5.json").read_text())
    groups = ["Paper\n(1 slice/subject,\ntest threshold)", "Ours, same\nprotocol", "All 20 slices,\nvalidation threshold,\npooled",
              "All 20 slices,\nvalidation threshold,\nmean per slice"]
    vb = [79.6, b["reflect_global_max_dice"] * 100, b["strict_pooled_dice"] * 100, b["strict_mean_dice_tumour_slices"] * 100]
    vi = [np.nan, i["reflect_global_max_dice"] * 100, i["strict_pooled_dice"] * 100, i["strict_mean_dice_tumour_slices"] * 100]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.0, 2.9), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(4)
    ax.bar(x - 0.18, vb, 0.36, color=COLOR["brats"], label="trained on BraTS healthy slices")
    ax.bar(x + 0.18, vi, 0.36, color=COLOR["ixi"], label="trained on IXI")
    for xx, v in zip(x - 0.18, vb):
        ax.text(xx, v + 1, f"{v:.1f}", ha="center", fontsize=7)
    for xx, v in zip(x + 0.18, vi):
        if not np.isnan(v):
            ax.text(xx, v + 1, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, groups, fontsize=7)
    ax.set_ylabel("Dice (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Same models, different evaluation protocols", fontsize=9, pad=14)
    fa = [b["strict_fp_rate_normal_slices"] * 100, i["strict_fp_rate_normal_slices"] * 100]
    ax2.bar([0, 1], fa, color=[COLOR["brats"], COLOR["ixi"]], width=0.6)
    for xx, v in enumerate(fa):
        ax2.text(xx, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
    ax2.set_xticks([0, 1], ["BraTS", "IXI"])
    ax2.set_ylabel("Tumour-free slices\nwith false alarms (%)")
    ax2.set_title("Not measured by\nthe 1-slice protocol", fontsize=8.5)
    save(fig, out, "protocol")


def outline(ax, mask, color):
    edge = mask & ~binary_erosion(mask, iterations=1)
    yy, xx = np.nonzero(edge)
    ax.scatter(xx, yy, s=0.15, c=color, marker="s", linewidths=0)


def qualitative(curves_dir, summary, out, models=("brats", "ixi", "ft_mix50")):
    inp = np.load(curves_dir / "test_inputs.npz")
    maps = {m: np.load(curves_dir / f"{m}_maps.npz") for m in models if (curves_dir / f"{m}_maps.npz").exists()}
    if "brats" not in maps or "ixi" not in maps:
        return
    img, brain, seg, subj = inp["image"], inp["brain"], inp["seg"], inp["subject"]
    th = {m: summary["models"][m]["val_fp5"]["threshold"] for m in maps}
    area = seg.sum((1, 2))
    rng = np.random.default_rng(1)
    # rows: tumours of three sizes, then the tumour-free slices where the IXI
    # model's thresholded map is largest (one per subject)
    picks = []
    for q in (0.9, 0.6, 0.3):
        tum = np.where(area > 0)[0]
        target = np.quantile(area[tum], q)
        cand = tum[np.argsort(np.abs(area[tum] - target))[:20]]
        picks.append(int(rng.choice(cand)))
    normal = np.where(area == 0)[0]
    fp_ixi = ((maps["ixi"]["maps"][normal] / 255.0) > th["ixi"]).sum((1, 2))
    seen = set()
    for k in normal[np.argsort(-fp_ixi)]:
        if subj[k] not in seen:
            seen.add(subj[k])
            picks.append(int(k))
        if len(picks) == 5:
            break
    cols = [("T2 input\n(tumour outlined)", None)]
    for m in maps:
        short = {"brats": "BraTS-trained", "ixi": "IXI-trained", "ft_mix50": "IXI + 50 local"}[m]
        cols += [(f"{short}:\nhealthy reconstruction", ("rec", m)), (f"{short}:\nanomaly map", ("map", m))]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(1.25 * len(cols), 1.3 * len(picks)))
    for r, k in enumerate(picks):
        crop = np.s_[8:248, 8:248]
        for c, (title, what) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if what is None:
                ax.imshow(img[k][crop], cmap="gray", vmin=0, vmax=255)
                if seg[k].any():
                    outline(ax, seg[k][crop], "#00e5ff")
            elif what[0] == "rec":
                ax.imshow(maps[what[1]]["recons"][k][crop], cmap="gray", vmin=0, vmax=255)
            else:
                mp = maps[what[1]]["maps"][k][crop] / 255.0
                ax.imshow(img[k][crop], cmap="gray", vmin=0, vmax=255)
                ax.imshow(np.ma.masked_where(mp <= th[what[1]], mp), cmap="autumn", vmin=th[what[1]], vmax=1, alpha=0.85)
                if seg[k].any():
                    outline(ax, seg[k][crop], "#00e5ff")
            if r == 0:
                ax.set_title(title, fontsize=6.5)
        axes[r, 0].set_ylabel("tumour" if seg[k].any() else "no tumour", fontsize=7)
    fig.suptitle("BraTS 2021 test slices. Anomaly maps thresholded where ≤5% of tumour-free validation slices fire.",
                 fontsize=7.5, y=0.95)
    plt.subplots_adjust(wspace=0.03, hspace=0.05)
    save(fig, out, "qualitative")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((Path(a.summary) / "summary.json").read_text())
    names = [n for n in ("brats", "ixi", "ixi_thick", "ft_mix20", "ft_mix50") if n in summary["models"]]
    protocol(out)
    tradeoff(summary, Path(a.summary), out, names)
    local_finetuning(summary, out)
    qualitative(Path(a.curves), summary, out)


if __name__ == "__main__":
    main()
