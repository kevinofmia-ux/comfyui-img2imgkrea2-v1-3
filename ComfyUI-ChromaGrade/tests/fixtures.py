"""Synthetic image fixtures.

Everything here is generated from a fixed seed, so the tests and the contact
sheet are reproducible and the repository carries no third-party image assets
whose licence would have to be tracked. The scenes are deliberately varied
along the axes the node has to survive: dynamic range, saturation, the presence
of skin, the presence of fine texture, and degeneracy.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "SCENES",
    "REFERENCES",
    "make",
    "gradient_scene",
    "portrait",
    "neon",
    "fog",
    "greyscale",
    "constant",
    "high_key",
    "low_key",
    "warm_sunset_reference",
    "cool_teal_reference",
    "bleach_reference",
    "sepia_reference",
]


def _grid(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(0.0, 1.0, h).view(h, 1).expand(h, w)
    x = torch.linspace(0.0, 1.0, w).view(1, w).expand(h, w)
    return y, x


def _texture(h: int, w: int, seed: int, scale: float = 0.035) -> torch.Tensor:
    """Deterministic fine texture, so "did detail survive" is measurable."""
    g = torch.Generator().manual_seed(seed)
    fine = torch.rand(h, w, 1, generator=g) - 0.5
    y, x = _grid(h, w)
    ripple = torch.sin(x * 47.0) * torch.cos(y * 61.0)
    return (fine[..., 0] * 1.2 + ripple * 0.4).unsqueeze(-1) * scale


def _finish(rgb: torch.Tensor) -> torch.Tensor:
    return rgb.clamp(0.0, 1.0).unsqueeze(0)


def gradient_scene(h: int = 256, w: int = 384, seed: int = 11) -> torch.Tensor:
    """Sky over ground: a wide tonal range with a clear horizon edge."""
    y, x = _grid(h, w)
    sky = torch.stack([0.35 + 0.25 * (1 - y), 0.55 + 0.30 * (1 - y), 0.80 + 0.18 * (1 - y)], dim=-1)
    ground = torch.stack([0.28 + 0.20 * y, 0.24 + 0.16 * y, 0.14 + 0.08 * y], dim=-1)
    horizon = (y > 0.55).unsqueeze(-1).to(sky.dtype)
    rgb = sky * (1 - horizon) + ground * horizon
    rgb = rgb + _texture(h, w, seed)
    # A few saturated accents so palette transfer has something to chew on.
    blob = ((x - 0.25) ** 2 + (y - 0.30) ** 2 < 0.010).unsqueeze(-1).to(rgb.dtype)
    rgb = rgb * (1 - blob) + torch.tensor([0.92, 0.30, 0.12]) * blob
    return _finish(rgb)


def portrait(h: int = 256, w: int = 256, seed: int = 23) -> torch.Tensor:
    """A skin-toned oval on a neutral background, with hair and a highlight."""
    y, x = _grid(h, w)
    rgb = torch.stack([0.30 + 0.10 * y, 0.31 + 0.10 * y, 0.34 + 0.10 * y], dim=-1)

    face = (((x - 0.5) / 0.26) ** 2 + ((y - 0.52) / 0.34) ** 2 < 1.0).unsqueeze(-1).to(rgb.dtype)
    shade = (0.82 + 0.18 * (1.0 - ((x - 0.42) ** 2 + (y - 0.40) ** 2).sqrt())).unsqueeze(-1)
    skin = torch.tensor([0.86, 0.68, 0.57]) * shade
    rgb = rgb * (1 - face) + skin * face

    hair = (((x - 0.5) / 0.32) ** 2 + ((y - 0.36) / 0.30) ** 2 < 1.0).unsqueeze(-1).to(rgb.dtype)
    hair = hair * (1 - face)
    rgb = rgb * (1 - hair) + torch.tensor([0.16, 0.11, 0.09]) * hair

    spec = (((x - 0.60) / 0.05) ** 2 + ((y - 0.36) / 0.05) ** 2 < 1.0).unsqueeze(-1).to(rgb.dtype)
    rgb = rgb + spec * 0.22

    return _finish(rgb + _texture(h, w, seed, scale=0.022))


def neon(h: int = 224, w: int = 320, seed: int = 37) -> torch.Tensor:
    """Highly saturated primaries near the gamut boundary."""
    y, x = _grid(h, w)
    r = (torch.sin(x * 9.0) * 0.5 + 0.5) ** 2
    g = (torch.cos(y * 7.0) * 0.5 + 0.5) ** 2
    b = (torch.sin((x + y) * 6.0) * 0.5 + 0.5) ** 2
    rgb = torch.stack([r, g * 0.6, b], dim=-1) * 0.95 + 0.02
    return _finish(rgb + _texture(h, w, seed, scale=0.015))


def fog(h: int = 200, w: int = 300, seed: int = 41) -> torch.Tensor:
    """Low dynamic range, low saturation: the classic 'nothing to grade' case."""
    y, _ = _grid(h, w)
    base = 0.52 + 0.06 * y
    rgb = torch.stack([base * 1.02, base, base * 0.98], dim=-1)
    return _finish(rgb + _texture(h, w, seed, scale=0.012))


def greyscale(h: int = 192, w: int = 256, seed: int = 53) -> torch.Tensor:
    y, x = _grid(h, w)
    v = (0.15 + 0.7 * (0.5 * y + 0.5 * x)).unsqueeze(-1)
    rgb = v.expand(h, w, 3).clone()
    return _finish(rgb + _texture(h, w, seed, scale=0.02))


def high_key(h: int = 160, w: int = 160, seed: int = 59) -> torch.Tensor:
    """Everything crowded against white."""
    y, x = _grid(h, w)
    rgb = torch.stack([0.93 + 0.06 * x, 0.94 + 0.05 * y, 0.95 + 0.04 * x], dim=-1)
    return _finish(rgb + _texture(h, w, seed, scale=0.008))


def low_key(h: int = 160, w: int = 160, seed: int = 61) -> torch.Tensor:
    """Everything crowded against black."""
    y, x = _grid(h, w)
    rgb = torch.stack([0.02 + 0.06 * x, 0.015 + 0.05 * y, 0.03 + 0.07 * x], dim=-1)
    return _finish(rgb + _texture(h, w, seed, scale=0.006))


def constant(value: float = 0.0, h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.full((1, h, w, 3), float(value))


def warm_sunset_reference(h: int = 180, w: int = 240) -> torch.Tensor:
    y, x = _grid(h, w)
    rgb = torch.stack(
        [0.20 + 0.78 * (1 - y), 0.06 + 0.52 * (1 - y) ** 1.6, 0.05 + 0.22 * (1 - y) ** 2.4],
        dim=-1,
    )
    rgb = rgb + torch.stack([0.02 * x, 0.01 * x, 0.06 * y], dim=-1)
    return _finish(rgb)


def cool_teal_reference(h: int = 180, w: int = 240) -> torch.Tensor:
    """Crushed, cold, low-saturation: a 'teal and orange' shadow character."""
    y, x = _grid(h, w)
    lum = 0.10 + 0.55 * y**1.4
    rgb = torch.stack([lum * 0.78 + 0.02 * x, lum * 1.02, lum * 1.18 + 0.05], dim=-1)
    warm = (y > 0.86).unsqueeze(-1).to(rgb.dtype)
    rgb = rgb * (1 - warm) + torch.stack([lum * 1.25, lum * 1.02, lum * 0.72], dim=-1) * warm
    return _finish(rgb)


def bleach_reference(h: int = 180, w: int = 240) -> torch.Tensor:
    """High contrast, near-neutral, lifted blacks."""
    y, x = _grid(h, w)
    v = (0.5 + 0.5 * torch.tanh((y - 0.5) * 4.0)) * 0.86 + 0.10
    rgb = torch.stack([v * 1.01, v, v * 0.99 + 0.01 * x], dim=-1)
    return _finish(rgb)


def sepia_reference(h: int = 180, w: int = 240) -> torch.Tensor:
    y, _ = _grid(h, w)
    v = 0.12 + 0.72 * y
    rgb = torch.stack([v * 1.10, v * 0.92, v * 0.66], dim=-1)
    return _finish(rgb)


SCENES = {
    "gradient": gradient_scene,
    "portrait": portrait,
    "neon": neon,
    "fog": fog,
    "greyscale": greyscale,
    "high_key": high_key,
    "low_key": low_key,
}

REFERENCES = {
    "sunset": warm_sunset_reference,
    "teal": cool_teal_reference,
    "bleach": bleach_reference,
    "sepia": sepia_reference,
}


def make(name: str) -> torch.Tensor:
    if name in SCENES:
        return SCENES[name]()
    if name in REFERENCES:
        return REFERENCES[name]()
    raise KeyError(f"unknown fixture {name!r}; known: {sorted(SCENES) + sorted(REFERENCES)}")


def checkerboard(h: int = 64, w: int = 64, size: int = 8) -> torch.Tensor:
    """Maximum-frequency structure, for detail-preservation checks."""
    y, x = _grid(h, w)
    cell = ((y * h / size).floor() + (x * w / size).floor()) % 2
    return _finish(cell.unsqueeze(-1).expand(h, w, 3) * 0.8 + 0.1)


def radial_ramp(h: int = 256, w: int = 256) -> torch.Tensor:
    """A smooth wide-range ramp: the fixture banding would show up in."""
    y, x = _grid(h, w)
    r = (((x - 0.5) ** 2 + (y - 0.5) ** 2).sqrt() / math.sqrt(0.5)).clamp(0, 1)
    return _finish((1.0 - r).unsqueeze(-1).expand(h, w, 3).clone())
