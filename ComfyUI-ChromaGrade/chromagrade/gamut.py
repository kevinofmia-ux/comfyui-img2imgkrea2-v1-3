"""Gamut handling: get out of Oklab and into displayable sRGB without clipping.

Naive `clamp(rgb, 0, 1)` is responsible for most of what people call "the AI
colour look": clipping a single channel shifts hue *and* lightness, so a
saturated sunset turns into flat orange paste and skin under a warm grade goes
waxy. Two mechanisms replace it here.

1. A smooth toe and shoulder on Oklab lightness, so a grade that pushes past
   black or white rolls off asymptotically instead of piling up on the
   boundary. This is what protects against crushed blacks and blown highlights.
2. Chroma reduction toward the neutral axis at constant lightness and constant
   hue, found by bisection. When a colour genuinely cannot be shown, the thing
   that gives is saturation -- the perceptual attribute the eye is least
   sensitive to in isolation -- while hue and lightness, which it is very
   sensitive to, are preserved exactly.

This is the "preserve lightness" family of gamut mapping described by Bjorn
Ottosson in https://bottosson.github.io/posts/gamutclipping/ .
"""

from __future__ import annotations

import torch

from .colorspace import oklab_to_linear

__all__ = ["compress_lightness", "gamut_map_to_linear"]

# Above this fraction of out-of-gamut pixels it is cheaper to bisect the whole
# tensor than to gather, bisect a subset and scatter back.
_SUBSET_FRACTION = 0.5


def compress_lightness(lightness: torch.Tensor, head: float, toe: float) -> torch.Tensor:
    """Roll ``L`` smoothly into ``[0, 1]`` using shoulders sized to the overshoot.

    ``head`` and ``toe`` are the widths of the roll-off regions. Inside
    ``[toe, 1 - head]`` this is exactly the identity, and both branches match
    the identity in value *and* slope at the knee, so there is no crease where
    compression begins.

    Sizing the shoulders to the actual overshoot is the point. A fixed shoulder
    would dim pure white on every image, including ones that never went out of
    range; an adaptive one is inert when the grade stays in bounds and engages
    exactly as hard as it has to when the grade pushes past. There is no way to
    have a smooth, bounded, monotone roll-off that *also* fixes white at 1.0 --
    any C1 function that equals the identity at 1 with unit slope must exceed 1
    just above it -- so the compromise is to make the loss proportional to the
    problem: with a 2% overshoot, white lands at L = 0.995.
    """
    out = lightness
    if head > 1e-6:
        s = 1.0 - head
        out = torch.where(out > s, s + head * torch.tanh((out - s) / head), out)
    if toe > 1e-6:
        out = torch.where(out < toe, toe + toe * torch.tanh((out - toe) / toe), out)
    return out


def _in_gamut(lin: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    return ((lin >= -tol) & (lin <= 1.0 + tol)).all(dim=-1, keepdim=True)


def _bisect(lab: torch.Tensor, iterations: int) -> torch.Tensor:
    """Chroma bisection on ``[N, 3]`` Oklab, all of which is out of gamut."""
    lo = torch.zeros_like(lab[..., 0:1])
    hi = torch.ones_like(lo)
    # t = 0 is achromatic at a lightness already inside [0, 1], hence always
    # displayable; seed the answer with it so every point has a valid fallback.
    best = oklab_to_linear(torch.cat([lab[..., 0:1], torch.zeros_like(lab[..., 1:3])], dim=-1)).clamp(0.0, 1.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        trial = torch.cat([lab[..., 0:1], lab[..., 1:3] * mid], dim=-1)
        trial_lin = oklab_to_linear(trial)
        ok = _in_gamut(trial_lin)
        best = torch.where(ok, trial_lin, best)
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)
    return best.clamp(0.0, 1.0)


def gamut_map_to_linear(oklab: torch.Tensor, iterations: int = 16, max_shoulder: float = 0.15) -> torch.Tensor:
    """Oklab -> in-gamut linear sRGB.

    ``iterations`` bisection steps resolve the chroma scale to ``2**-16``, far
    below a quantisation step at any bit depth anyone will use. Colours already
    inside the gamut are returned untouched -- an ordinary in-gamut image is
    never silently desaturated by a fraction of a percent -- and when only part
    of the frame is out of gamut the bisection runs on that subset alone, which
    is what keeps this affordable at 4K.

    The lightness shoulders are sized from the data: an image whose lightness
    never leaves ``[0, 1]`` gets a pure clamp and a bit-exact white point.
    """
    lightness = oklab[..., 0:1]
    head = min(max_shoulder, max(0.0, float(lightness.max()) - 1.0))
    toe = min(max_shoulder, max(0.0, -float(lightness.min())))
    lab = torch.cat(
        [compress_lightness(lightness, head, toe).clamp(0.0, 1.0), oklab[..., 1:3]],
        dim=-1,
    )

    lin = oklab_to_linear(lab)
    outside = ~_in_gamut(lin)
    n_out = int(outside.sum())
    if n_out == 0:
        return lin.clamp(0.0, 1.0)

    total = outside.numel()
    if n_out > _SUBSET_FRACTION * total:
        fixed = _bisect(lab, iterations)
        return torch.where(outside, fixed, lin.clamp(0.0, 1.0))

    shape = lab.shape
    flat_lab = lab.reshape(-1, 3)
    flat_lin = lin.reshape(-1, 3).clamp(0.0, 1.0)
    idx = outside.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    flat_lin[idx] = _bisect(flat_lab[idx], iterations)
    return flat_lin.reshape(shape)
