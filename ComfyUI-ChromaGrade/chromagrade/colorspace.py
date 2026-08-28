"""Colour-space primitives.

Everything downstream of this module assumes three domains and never mixes them
up:

``srgb``
    Gamma-encoded sRGB in ``[0, 1]``. This is what ComfyUI IMAGE tensors carry
    and what a ``.cube`` LUT is indexed by. Statistics are *never* computed
    here -- distances in gamma-encoded RGB are perceptually meaningless and
    ratio operations (exposure, white balance) are outright wrong.

``linear``
    Linear-light sRGB primaries. Exposure and white balance are physical
    scalings of light, so they happen here and only here.

``oklab``
    Ottosson's Oklab. Nearly uniform, well-behaved hue lines, and cheap. All
    distribution matching (tone curve, optimal transport, palette statistics)
    happens here so that "equal numeric change" means "roughly equal perceived
    change" and so that the luminance/chrominance split is meaningful.

Chromatic adaptation reuses Oklab's own cone-response matrix ``M1`` rather than
bolting on a separate Bradford/CAT16 pipeline: ``M1`` already maps linear sRGB
to an LMS-like cone space, which is exactly the domain a von Kries gain is
defined in. One matrix family, one place to be wrong.

Reference: Bjorn Ottosson, "A perceptual color space for image processing"
(2020), https://bottosson.github.io/posts/oklab/ -- published as public domain
/ MIT-style permissive by the author.
"""

from __future__ import annotations

import torch

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "linear_to_lms",
    "lms_to_linear",
    "linear_to_oklab",
    "oklab_to_linear",
    "srgb_to_oklab",
    "oklab_to_srgb",
    "luminance",
    "chroma",
    "hue",
]

# Linear sRGB -> LMS-like cone response (Oklab stage 1).
_M1 = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)

# Non-linear LMS' -> Oklab (Oklab stage 2).
_M2 = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)

# Rec.709 luminance weights, for linear-light luminance.
_LUMA_709 = (0.2126, 0.7152, 0.0722)

# Inverses are computed rather than transcribed: one fewer place for a typo,
# and the round trip is then exact to float64 rather than to however many
# digits somebody's blog post printed.
_CACHE: dict[tuple[torch.device, torch.dtype], dict[str, torch.Tensor]] = {}


def _matrices(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    key = (device, dtype)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    m1 = torch.tensor(_M1, dtype=torch.float64, device=device)
    m2 = torch.tensor(_M2, dtype=torch.float64, device=device)
    mats = {
        "m1": m1.to(dtype),
        "m1_inv": torch.linalg.inv(m1).to(dtype),
        "m2": m2.to(dtype),
        "m2_inv": torch.linalg.inv(m2).to(dtype),
        "luma": torch.tensor(_LUMA_709, dtype=torch.float64, device=device).to(dtype),
    }
    _CACHE[key] = mats
    return mats


def _matmul3(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 matrix to the last axis of ``x`` (``[..., 3]``)."""
    return torch.einsum("...c,dc->...d", x, m)


def _signed_pow(x: torch.Tensor, p: float) -> torch.Tensor:
    """``sign(x) * |x| ** p``.

    Intermediate stages of a grade routinely produce slightly negative linear
    values (an out-of-gamut colour on its way to the gamut mapper). Folding the
    sign out keeps every transfer function odd-symmetric, which means it stays
    invertible and monotone across zero instead of producing NaN.
    """
    return torch.copysign(x.abs().clamp_min(0.0).pow(p), x)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Exact piecewise sRGB EOTF, extended odd-symmetrically below zero."""
    a = x.abs()
    lin = torch.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055).pow(2.4))
    return torch.copysign(lin, x)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Exact piecewise sRGB OETF, extended odd-symmetrically below zero."""
    a = x.abs()
    enc = torch.where(a <= 0.0031308, a * 12.92, 1.055 * a.clamp_min(0.0).pow(1.0 / 2.4) - 0.055)
    return torch.copysign(enc, x)


def linear_to_lms(x: torch.Tensor) -> torch.Tensor:
    """Linear sRGB -> cone-response LMS (the domain von Kries gains live in)."""
    return _matmul3(x, _matrices(x.device, x.dtype)["m1"])


def lms_to_linear(x: torch.Tensor) -> torch.Tensor:
    return _matmul3(x, _matrices(x.device, x.dtype)["m1_inv"])


def linear_to_oklab(x: torch.Tensor) -> torch.Tensor:
    mats = _matrices(x.device, x.dtype)
    lms = _matmul3(x, mats["m1"])
    return _matmul3(_signed_pow(lms, 1.0 / 3.0), mats["m2"])


def oklab_to_linear(x: torch.Tensor) -> torch.Tensor:
    mats = _matrices(x.device, x.dtype)
    lms_ = _matmul3(x, mats["m2_inv"])
    return _matmul3(_signed_pow(lms_, 3.0), mats["m1_inv"])


def srgb_to_oklab(x: torch.Tensor) -> torch.Tensor:
    return linear_to_oklab(srgb_to_linear(x))


def oklab_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return linear_to_srgb(oklab_to_linear(x))


def luminance(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 relative luminance from *linear* RGB, shape ``[..., 1]``."""
    mats = _matrices(linear_rgb.device, linear_rgb.dtype)
    return (linear_rgb * mats["luma"]).sum(dim=-1, keepdim=True)


def chroma(oklab: torch.Tensor) -> torch.Tensor:
    """Oklab chroma ``sqrt(a^2 + b^2)``, shape ``[..., 1]``."""
    return oklab[..., 1:3].pow(2).sum(dim=-1, keepdim=True).clamp_min(0.0).sqrt()


def hue(oklab: torch.Tensor) -> torch.Tensor:
    """Oklab hue angle in radians, shape ``[..., 1]``."""
    return torch.atan2(oklab[..., 2:3], oklab[..., 1:2])
