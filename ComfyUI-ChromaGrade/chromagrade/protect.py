"""Skin-tone and neutral-axis protection, expressed as colour-domain rules.

Both guards here operate on *colours*, not on pixels: they take the pair
``(input Oklab, transformed Oklab)`` and return a corrected transformed value.
That is deliberate -- it means they can be evaluated on a LUT lattice along
with everything else, so protection costs nothing at full resolution and cannot
introduce spatial artefacts (no mask, no edges, no halos, nothing to feather).

Skin protection
    The failure mode that ruins a reference grade on people is hue rotation:
    the reference's palette pulls skin toward green or magenta and the face
    reads as ill. Chroma and lightness changes on skin are usually *wanted* --
    that is the grade. So the rule is: inside the skin locus, preserve the
    input hue angle and cap chroma amplification; leave lightness entirely to
    the grade.

Neutral guard
    A near-neutral surface should be allowed to take on the reference's cast
    (that is what white balance in a grade means) but must not be allowed to
    become a saturated colour, which is what an aggressive distribution match
    will otherwise do to a flat sky or a grey wall. The cap is affine in input
    chroma, so it binds near the neutral axis and is inert everywhere else.
"""

from __future__ import annotations

import math

import torch

from .colorspace import srgb_to_oklab

__all__ = ["skin_hue_window", "skin_membership", "apply_skin_protection", "apply_neutral_guard"]

# A hand-specified ladder of representative skin sRGB values spanning very
# light to deep, used only to *derive* the hue window below -- no external
# dataset or third-party palette is embedded. Exact values do not matter much;
# the window is widened by a margin afterwards and the membership falls off
# smoothly, so this fixes a locus rather than a hard classification.
_SKIN_LADDER_SRGB8 = (
    (0xF6, 0xE0, 0xD2),
    (0xF0, 0xD3, 0xB8),
    (0xEE, 0xC6, 0xA5),
    (0xE0, 0xB0, 0x8C),
    (0xD2, 0x9E, 0x77),
    (0xC0, 0x88, 0x62),
    (0xA5, 0x6C, 0x4A),
    (0x8A, 0x57, 0x3B),
    (0x6B, 0x42, 0x2C),
    (0x4A, 0x2C, 0x1E),
    (0x33, 0x1E, 0x15),
)

_HUE_MARGIN = 0.22  # radians of hard window padding beyond the observed ladder
_HUE_FALLOFF = 0.30  # radians of Gaussian roll-off outside the padded window

_CHROMA_IN = (0.010, 0.028)  # below this, it is a neutral, not skin
_CHROMA_OUT = (0.115, 0.190)  # above this, it is a saturated object, not skin
_LIGHT_IN = (0.14, 0.26)
_LIGHT_OUT = (0.92, 0.99)

_WINDOW_CACHE: tuple[float, float] | None = None


def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = ((x - lo) / max(hi - lo, 1e-9)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def skin_hue_window() -> tuple[float, float]:
    """``(lo, hi)`` Oklab hue angles in radians bounding the skin locus."""
    global _WINDOW_CACHE
    if _WINDOW_CACHE is None:
        rgb = torch.tensor(_SKIN_LADDER_SRGB8, dtype=torch.float32) / 255.0
        lab = srgb_to_oklab(rgb)
        hues = torch.atan2(lab[:, 2], lab[:, 1])
        _WINDOW_CACHE = (
            float(hues.min()) - _HUE_MARGIN,
            float(hues.max()) + _HUE_MARGIN,
        )
    return _WINDOW_CACHE


def _angular_distance_to_window(h: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Shortest angular distance from ``h`` to the arc ``[lo, hi]``; 0 inside."""
    two_pi = 2.0 * math.pi
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    delta = h - centre
    delta = delta - two_pi * torch.round(delta / two_pi)
    return (delta.abs() - half).clamp_min(0.0)


def skin_membership(oklab: torch.Tensor) -> torch.Tensor:
    """Soft membership in the skin locus, ``[..., 1]`` in ``[0, 1]``."""
    lo, hi = skin_hue_window()
    lightness = oklab[..., 0:1]
    a = oklab[..., 1:2]
    b = oklab[..., 2:3]
    c = (a * a + b * b).clamp_min(0.0).sqrt()
    h = torch.atan2(b, a)

    dist = _angular_distance_to_window(h, lo, hi)
    w_hue = torch.exp(-0.5 * (dist / _HUE_FALLOFF) ** 2)
    w_chroma = _smoothstep(c, *_CHROMA_IN) * (1.0 - _smoothstep(c, *_CHROMA_OUT))
    w_light = _smoothstep(lightness, *_LIGHT_IN) * (1.0 - _smoothstep(lightness, *_LIGHT_OUT))
    return (w_hue * w_chroma * w_light).clamp(0.0, 1.0)


def apply_skin_protection(
    lab_in: torch.Tensor,
    lab_out: torch.Tensor,
    amount: float,
    max_chroma_gain: float = 1.35,
) -> torch.Tensor:
    """Pull the graded hue back toward the original inside the skin locus.

    Lightness is never touched -- the grade owns it. The hue angle is
    interpolated *circularly* so a rotation across the +/-pi seam does not
    swing the long way round, and chroma is only limited when the grade tried
    to increase it.
    """
    if amount <= 0.0:
        return lab_out

    weight = skin_membership(lab_in) * float(amount)
    if not bool((weight > 1e-4).any()):
        return lab_out

    a_in, b_in = lab_in[..., 1:2], lab_in[..., 2:3]
    a_out, b_out = lab_out[..., 1:2], lab_out[..., 2:3]
    c_in = (a_in * a_in + b_in * b_in).clamp_min(0.0).sqrt()
    c_out = (a_out * a_out + b_out * b_out).clamp_min(0.0).sqrt()
    h_in = torch.atan2(b_in, a_in)
    h_out = torch.atan2(b_out, a_out)

    two_pi = 2.0 * math.pi
    delta = h_in - h_out
    delta = delta - two_pi * torch.round(delta / two_pi)
    h_final = h_out + delta * weight

    capped = torch.minimum(c_out, c_in * max_chroma_gain + 1e-4)
    c_final = torch.lerp(c_out, capped, weight)

    return torch.cat([lab_out[..., 0:1], c_final * torch.cos(h_final), c_final * torch.sin(h_final)], dim=-1)


def apply_neutral_guard(
    lab_in: torch.Tensor,
    lab_out: torch.Tensor,
    slope: float = 3.0,
    offset: float = 0.055,
) -> torch.Tensor:
    """Cap how saturated a near-neutral input colour is allowed to become.

    ``cap = slope * C_in + offset``. At ``C_in = 0`` a pure grey may pick up a
    cast of up to ``offset`` in Oklab chroma -- clearly visible, roughly what a
    warm or cool grade actually does to a grey card -- but it can never turn
    into a colour. By ``C_in ~ 0.09`` the cap is above the sRGB gamut boundary
    and the rule stops doing anything at all.
    """
    a_out, b_out = lab_out[..., 1:2], lab_out[..., 2:3]
    c_out = (a_out * a_out + b_out * b_out).clamp_min(0.0).sqrt()
    a_in, b_in = lab_in[..., 1:2], lab_in[..., 2:3]
    c_in = (a_in * a_in + b_in * b_in).clamp_min(0.0).sqrt()

    cap = slope * c_in + offset
    scale = torch.where(c_out > cap, cap / c_out.clamp_min(1e-9), torch.ones_like(c_out))
    return torch.cat([lab_out[..., 0:1], a_out * scale, b_out * scale], dim=-1)
