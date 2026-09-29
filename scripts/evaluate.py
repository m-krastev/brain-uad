"""Evaluate a trained method on MSD brain tumours (T2) and healthy IXI test scans.

Post-processing and scoring follow pDDPM: the anomaly map |x - x_rec| is
smoothed with a median filter (kernel 5), the brain mask is eroded three times,
a threshold is chosen greedily for the best mean Dice on the 100 unhealthy
validation scans, and connected components under 7 voxels are removed. Test
metrics are the mean per-volume Dice, the AUPRC (per volume and pooled over all
test voxels) and the healthy l1 error.

Test-time ensembling (additions): --t-test and --shifts accept several values;
their anomaly maps are averaged.

  python scripts/evaluate.py pddpm_k0 --t-test 500 --shifts 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uad.data import ROOT, Volumes  # noqa: E402
from uad.methods import Method  # noqa: E402
from uad.noise import NoiseBank  # noqa: E402


@torch.no_grad()
def anomaly_maps(model, vols, k, ts, shifts, seeds):
    N, Z = vols.img.shape[:2]
    out = np.zeros(tuple(vols.img.shape), np.float32)
    recs = []
    for v in range(N):
        z = torch.arange(Z, device=vols.img.device)
        x = vols.stack(torch.full_like(z, v), z, k)
        acc = torch.zeros_like(x[:, :1])
        n = 0
        if model is None:
            out[v] = vols.img[v].float().cpu().numpy() * vols.brain[v].cpu().numpy()
            continue
        for t in ts:
            for s in shifts:
                for sd in seeds:
                    with torch.autocast("cuda", torch.bfloat16):
                        rec = model.reconstruct(x, t, (s, s), seed=sd).float()
                    acc += (rec - x[:, k:k + 1]).abs()
                    n += 1
        out[v] = (acc / n)[:, 0].cpu().numpy() * vols.brain[v].cpu().numpy()
    return out


def postprocess(maps, brains):
    out = np.empty_like(maps)
    for i, (m, b) in enumerate(zip(maps, brains)):
        eb = ndimage.binary_erosion(b, iterations=3)
        out[i] = ndimage.median_filter(m, size=5) * eb
    return out


def remove_small(binary, min_size=7):
    lab, n = ndimage.label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


def mean_dice(maps, segs, thr):
    return float(np.mean([dice(remove_small(m > thr), g) for m, g in zip(maps, segs)]))


def choose_threshold(maps, segs, n=60):
    vals = maps[maps > 0]
    qs = np.quantile(vals, np.linspace(0.80, 0.9995, n))
    scores = [mean_dice(maps, segs, q) for q in qs]
    i = int(np.argmax(scores))
    # refine between neighbours
    lo, hi = qs[max(i - 1, 0)], qs[min(i + 1, n - 1)]
    fine = np.linspace(lo, hi, 15)
    fs = [mean_dice(maps, segs, q) for q in fine]
    j = int(np.argmax(fs))
    return float(fine[j]), float(fs[j])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--t-test", type=int, nargs="+", default=[500])
    p.add_argument("--shifts", type=int, nargs="+", default=[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--patch", type=int, default=None, help="override the training patch size")
    p.add_argument("--tag", default="")
    p.add_argument("--test-set", default="msd", choices=["msd", "brats"])
    a = p.parse_args()

    run = ROOT / "results" / a.name
    dev = torch.device("cuda")
    if a.name == "thresh":  # intensity itself as the anomaly score (Meissen et al., 2022)
        run.mkdir(parents=True, exist_ok=True)
        args = {"method": "thresh", "k": 0, "patch": 0}
        model = None
    else:
        args = json.loads((run / "args.json").read_text())
        bank = None if args["method"] == "ae" else NoiseBank(ROOT / "data" / "simplex_bank.npy", dev)
        model = Method(args["method"], args["k"], a.patch or args["patch"], noise_bank=bank).to(dev).eval()
        model.net.load_state_dict(torch.load(run / a.ckpt, map_location=dev))
    k = args["k"]
    cfg = dict(ts=a.t_test, shifts=a.shifts, seeds=a.seeds)

    res = {"name": a.name, "ckpt": a.ckpt, **cfg, "patch": a.patch or args["patch"]}
    ixi = Volumes("ixi_test", dev)
    raw = anomaly_maps(model, ixi, k, **cfg)
    res["ixi_l1"] = float(raw.sum() / ixi.brain.sum().item())
    del ixi

    val = Volumes(f"{a.test_set}_val", dev)
    vm = postprocess(anomaly_maps(model, val, k, **cfg), val.brain.cpu().numpy())
    thr, vdice = choose_threshold(vm, val.seg.cpu().numpy())
    res.update(threshold=thr, val_dice=vdice)
    del val, vm

    test = Volumes(f"{a.test_set}_test", dev)
    tm = postprocess(anomaly_maps(model, test, k, **cfg), test.brain.cpu().numpy())
    segs, brains = test.seg.cpu().numpy(), test.brain.cpu().numpy()
    dices = [dice(remove_small(m > thr), g) for m, g in zip(tm, segs)]
    aps = [average_precision_score(g[b], m[b]) for m, g, b in zip(tm, segs, brains) if g.any()]
    res.update(
        test_dice=float(np.mean(dices)), test_dice_std=float(np.std(dices)),
        test_auprc_vol=float(np.mean(aps)),
        test_auprc_pooled=float(average_precision_score(segs[brains], tm[brains])),
        n_test=len(dices),
    )
    res["test_set"] = a.test_set
    tag = a.tag or f"{a.test_set}_t{'-'.join(map(str, a.t_test))}_s{'-'.join(map(str, a.shifts))}_n{len(a.seeds)}"
    (run / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    np.savetxt(run / f"dice_{tag}.txt", dices, fmt="%.4f")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
