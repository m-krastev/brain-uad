"""Denoising U-Net in the style of Dhariwal & Nichol (2021), as used by pDDPM.

Residual blocks with GroupNorm and SiLU, sinusoidal time embedding injected
with scale-shift norm, one stack of residual blocks per resolution and skip
connections between the encoder and decoder.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, temb: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(temb, 2 * cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.use_checkpoint = False

    def forward(self, x, emb):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    """channels: width per resolution; blocks: residual blocks per resolution.

    in_ch counts every input channel (the noised image plus any clean context
    channels); out_ch is the number of predicted channels (the image).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, channels=(128, 256, 256), blocks: int = 3):
        super().__init__()
        temb = channels[0] * 4
        self.temb_dim = channels[0]
        self.time = nn.Sequential(nn.Linear(channels[0], temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        self.down = nn.ModuleList()
        skips = [channels[0]]
        c = channels[0]
        for i, ch in enumerate(channels):
            for _ in range(blocks):
                self.down.append(ResBlock(c, ch, temb))
                c = ch
                skips.append(c)
            if i < len(channels) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)

        self.mid = nn.ModuleList([ResBlock(c, c, temb), ResBlock(c, c, temb)])

        self.up = nn.ModuleList()
        for i, ch in reversed(list(enumerate(channels))):
            for j in range(blocks + 1):
                self.up.append(ResBlock(c + skips.pop(), ch, temb))
                c = ch
                if i > 0 and j == blocks:
                    self.up.append(nn.ConvTranspose2d(c, c, 4, stride=2, padding=1))

        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))

    def forward(self, x, t):
        emb = self.time(timestep_embedding(t, self.temb_dim))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        for m in self.mid:
            h = m(h, emb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], dim=1), emb)
            else:
                h = m(h)
        return self.out(h)
