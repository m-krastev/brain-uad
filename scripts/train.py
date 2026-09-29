"""Train a method on healthy IXI slices.

Slices are drawn uniformly with replacement. Training length is given in
iterations; pDDPM's 1600 epochs of one slice per training volume at batch size
32 correspond to about 18k iterations. The checkpoint with the lowest
validation reconstruction error on healthy IXI scans is kept, as in pDDPM.

  python scripts/train.py --method pddpm --k 0 --iters 20000 --name pddpm_k0
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uad.data import ROOT, Volumes  # noqa: E402
from uad.methods import Method  # noqa: E402
from uad.noise import NoiseBank  # noqa: E402


@torch.no_grad()
def val_error(model, vols, k, t_test, bs=25):
    """Mean l1 reconstruction error inside the brain on healthy validation scans."""
    err, n = 0.0, 0
    N, Z = vols.img.shape[:2]
    for v in range(N):
        for z0 in range(0, Z, bs):
            z = torch.arange(z0, min(z0 + bs, Z), device=vols.img.device)
            x = vols.stack(torch.full_like(z, v), z, k)
            with torch.autocast("cuda", torch.bfloat16):
                rec = model.reconstruct(x, t_test).float()
            b = vols.brain[v, z].unsqueeze(1)
            err += ((rec - x[:, k:k + 1]).abs() * b).sum().item()
            n += b.sum().item()
    return err / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["ae", "ddpm", "pddpm"], required=True)
    p.add_argument("--k", type=int, default=0, help="neighbouring slices each side (2.5D)")
    p.add_argument("--patch", type=int, default=48)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--t-test", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpointing", action="store_true", help="gradient checkpointing (saves memory)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--name", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    train, val = Volumes("ixi_train", dev), Volumes("ixi_val", dev)
    bank = None if a.method == "ae" else NoiseBank(ROOT / "data" / "simplex_bank.npy", dev)
    model = Method(a.method, a.k, a.patch, noise_bank=bank).to(dev)
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = a.checkpointing
    ema = copy.deepcopy(model).eval()
    if a.compile:
        model.net = torch.compile(model.net)
    opt = torch.optim.Adam(model.net.parameters(), lr=a.lr)
    print(f"{a.name}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{len(train)} training volumes", flush=True)

    best = float("inf")
    log = open(out / "log.jsonl", "a")
    for block in range(1, a.iters // a.val_every + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(a.val_every):
            x = train.random_batch(a.bs, a.k)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model.loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.net.parameters(), model.net.parameters()):
                    pe.lerp_(pm, 1 - 0.999)
                for be, bm in zip(ema.net.buffers(), model.net.buffers()):
                    be.copy_(bm)
            tot += loss.item()
        rec = {"iter": block * a.val_every, "loss": tot / a.val_every,
               "sec": round(time.time() - t0, 1)}
        rec["val_l1"] = val_error(ema, val, a.k, a.t_test)
        if rec["val_l1"] < best:
            best = rec["val_l1"]
            torch.save(ema.net.state_dict(), out / "best.pt")
        torch.save(ema.net.state_dict(), out / "last.pt")
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
