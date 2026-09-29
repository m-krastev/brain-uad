"""Reconstruction-based anomaly detection methods.

Every method maps a stack of 2k+1 axial slices (B, C, H, W) to a
reconstruction of the same stack, trained with an l1 loss on healthy data.
The anomaly score is the absolute reconstruction error of the centre slice.

  ae     convolutional autoencoder with a dense bottleneck (Baur et al., 2021)
  ddpm   AnoDDPM-style: the whole image is noised with simplex noise at step
         t, and the network predicts x0 directly (Wyatt et al., 2022)
  pddpm  patched DDPM: only one patch is noised, the rest of the image is
         clean context; the loss covers the noised patch (Behrendt et al., 2023)

With k > 0 (2.5D), the patch is noised through all 2k+1 slices, so the
neighbouring slices add through-plane context outside the patch but never
show the clean content of the noised region.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


class AE(nn.Module):
    def __init__(self, ch: int = 1, size: int = 96, latent: int = 512, width=(32, 64, 128, 256)):
        super().__init__()
        enc, c = [], ch
        for w in width:
            enc += [nn.Conv2d(c, w, 4, 2, 1), nn.BatchNorm2d(w), nn.LeakyReLU(0.2)]
            c = w
        self.enc = nn.Sequential(*enc)
        self.s = size // 2 ** len(width)
        self.fc = nn.Linear(c * self.s * self.s, latent)
        self.fd = nn.Linear(latent, c * self.s * self.s)
        dec = []
        for w in reversed((ch,) + tuple(width[:-1])):
            dec += [nn.ConvTranspose2d(c, w, 4, 2, 1)]
            if w != ch:
                dec += [nn.BatchNorm2d(w), nn.LeakyReLU(0.2)]
            c = w
        self.dec = nn.Sequential(*dec)
        self.c = width[-1]

    def forward(self, x):
        z = self.fc(self.enc(x).flatten(1))
        h = self.fd(z).view(-1, self.c, self.s, self.s)
        return torch.sigmoid(self.dec(h))


class Schedule:
    def __init__(self, T: int = 1000, device="cuda"):
        self.T = T
        beta = torch.linspace(1e-4, 2e-2, T, device=device)
        self.abar = torch.cumprod(1 - beta, 0)

    def noise(self, x0, t, eps):
        a = self.abar[t - 1].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps


def grid_starts(size: int, p: int, shift: int = 0):
    """Patch starts covering [0, size) with patches of size p.

    shift = 0 gives pDDPM's evenly spaced grid. shift > 0 gives a grid offset
    by shift, completed with patches at both borders so that every pixel is
    still covered (overlaps are averaged).
    """
    if shift == 0:
        n = math.ceil(size / p)
        return torch.linspace(0, size - p, n).round().long().tolist() if n > 1 else [0]
    inner = list(range(shift, size - p, p))
    return sorted({0, size - p, *inner})


def patch_masks(H, W, p, shift=(0, 0), device="cuda"):
    """All grid patch masks (K, 1, H, W) for a patch size p and grid shift."""
    ms = []
    for y in grid_starts(H, p, shift[0]):
        for x in grid_starts(W, p, shift[1]):
            m = torch.zeros(1, H, W, device=device)
            m[:, y:y + p, x:x + p] = 1
            ms.append(m)
    return torch.stack(ms)


class Method(nn.Module):
    def __init__(self, kind: str, k: int = 0, patch: int = 48, size: int = 96, noise_bank=None):
        super().__init__()
        self.kind, self.k, self.patch, self.size = kind, k, patch, size
        ch = 2 * k + 1
        self.net = AE(ch, size) if kind == "ae" else UNet(ch, ch)
        self.bank = noise_bank
        self.sched = None

    def _sched(self, device):
        if self.sched is None:
            self.sched = Schedule(device=device)
        return self.sched

    def loss(self, x0):
        if self.kind == "ae":
            return F.l1_loss(self.net(x0), x0)
        s = self._sched(x0.device)
        b = x0.shape[0]
        t = torch.randint(1, s.T + 1, (b,), device=x0.device)
        eps = self.bank.sample(x0.shape)
        xt = s.noise(x0, t, eps)
        if self.kind == "ddpm":
            return F.l1_loss(self.net(xt, t), x0)
        masks = patch_masks(self.size, self.size, self.patch, device=x0.device)
        m = masks[torch.randint(len(masks), (b,), device=x0.device)]
        xin = xt * m + x0 * (1 - m)
        rec = self.net(xin, t)
        return ((rec - x0).abs() * m).sum() / (m.sum() * x0.shape[1])

    @torch.no_grad()
    def reconstruct(self, x0, t_test: int = 500, shift=(0, 0), seed: int = 0):
        """Reconstruction of the centre slice (B, 1, H, W)."""
        c = self.k
        if self.kind == "ae":
            return self.net(x0)[:, c:c + 1]
        s = self._sched(x0.device)
        b = x0.shape[0]
        t = torch.full((b,), t_test, device=x0.device, dtype=torch.long)
        g = torch.random.fork_rng(devices=[x0.device])
        with g:
            torch.manual_seed(seed)
            eps = self.bank.sample(x0.shape)
        xt = s.noise(x0, t, eps)
        if self.kind == "ddpm":
            return self.net(xt, t)[:, c:c + 1]
        masks = patch_masks(self.size, self.size, self.patch, shift, device=x0.device)
        acc = torch.zeros_like(x0[:, :1])
        cnt = torch.zeros_like(x0[:, :1])
        for m in masks:
            m = m[None]
            rec = self.net(xt * m + x0 * (1 - m), t)[:, c:c + 1]
            acc += rec * m
            cnt += m
        return acc / cnt
