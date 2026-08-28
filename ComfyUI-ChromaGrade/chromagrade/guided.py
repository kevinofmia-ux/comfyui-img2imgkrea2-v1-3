"""Edge-aware base/detail decomposition (guided filter) and detail restoration.

A colour grade that expands contrast necessarily expands whatever is riding on
top of the tones it stretches -- sensor noise, film grain, diffusion-model
texture. That is the artefact Pitie's *regrain* stage exists to undo. This
module does the same job with a guided filter, which is O(1) per pixel, needs
no iteration and has an explicit edge-awareness parameter.

The important detail: both the input and the graded output are decomposed using
the *same guide* (the target's own luminance). Sharing the guide means both
base layers have identical edge structure, so exchanging their detail layers
cannot produce a halo -- there is no edge in one that is not in the other.

Reference: K. He, J. Sun, X. Tang, "Guided Image Filtering", ECCV 2010 / TPAMI
2013. The O(1) box-filter formulation is the one described there.
"""

from __future__ import annotations

import torch

__all__ = ["box_filter", "guided_filter", "restore_detail", "detail_radius"]


def _box_1d(x: torch.Tensor, radius: int, dim: int) -> torch.Tensor:
    """Normalised moving average along ``dim``, with correct edge counts."""
    n = x.shape[dim]
    if radius <= 0 or n < 2:
        return x
    cum = torch.cumsum(x, dim=dim)
    zero = torch.zeros_like(x.narrow(dim, 0, 1))
    cum = torch.cat([zero, cum], dim=dim)

    ar = torch.arange(n, device=x.device)
    idx_hi = (ar + radius + 1).clamp(0, n)
    idx_lo = (ar - radius).clamp(0, n)
    hi = cum.index_select(dim, idx_hi)
    lo = cum.index_select(dim, idx_lo)

    count = (idx_hi - idx_lo).to(x.dtype)
    shape = [1] * x.ndim
    shape[dim] = n
    return (hi - lo) / count.reshape(shape).clamp_min(1.0)


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable normalised box filter over ``[..., H, W, C]``."""
    return _box_1d(_box_1d(x, radius, -3), radius, -2)


def detail_radius(height: int, width: int) -> int:
    """Filter radius scaled to the image, clamped to a useful band.

    Tied to the short edge so that "fine detail" means the same thing at 512px
    and at 4K, and capped so that a very large frame does not turn the filter
    into an expensive global blur.
    """
    return int(min(24, max(2, round(min(height, width) / 64.0))))


def guided_filter(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float = 1e-4) -> torch.Tensor:
    """Single-channel guided filter. ``guide`` and ``src`` are ``[..., H, W, 1]``."""
    mean_i = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    corr_i = box_filter(guide * guide, radius)
    corr_ip = box_filter(guide * src, radius)

    var_i = (corr_i - mean_i * mean_i).clamp_min(0.0)
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box_filter(a, radius) * guide + box_filter(b, radius)


def restore_detail(
    lightness_in: torch.Tensor,
    lightness_out: torch.Tensor,
    amount: float,
    radius: int | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Blend the graded detail layer back toward the target's original one.

    ``amount = 0`` leaves the grade exactly as the LUT produced it. ``amount =
    1`` keeps the graded *base* -- all the colour and tonal character of the
    reference -- while restoring the target's original detail amplitude
    exactly, so neither noise nor texture is scaled by the tone curve's local
    slope. Intermediate values trade the two off.

    Both tensors are ``[B, H, W, 1]`` Oklab lightness.
    """
    if amount <= 0.0:
        return lightness_out
    h, w = int(lightness_in.shape[-3]), int(lightness_in.shape[-2])
    r = detail_radius(h, w) if radius is None else int(radius)
    if r <= 0:
        return lightness_out

    base_in = guided_filter(lightness_in, lightness_in, r, eps)
    base_out = guided_filter(lightness_in, lightness_out, r, eps)
    detail_in = lightness_in - base_in
    detail_out = lightness_out - base_out

    w_t = float(min(max(amount, 0.0), 1.0))
    return base_out + detail_out + (detail_in - detail_out) * w_t
