"""Colour-space round trips, gamut mapping and protection rules."""

from __future__ import annotations

import torch

from chromagrade.colorspace import (
    linear_to_lms,
    linear_to_oklab,
    linear_to_srgb,
    lms_to_linear,
    luminance,
    oklab_to_linear,
    oklab_to_srgb,
    srgb_to_linear,
    srgb_to_oklab,
)
from chromagrade.gamut import compress_lightness, gamut_map_to_linear
from chromagrade.protect import apply_neutral_guard, apply_skin_protection, skin_hue_window, skin_membership


def _sample_colours(n: int = 4096) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.rand(n, 3, generator=g)


def test_srgb_linear_roundtrip():
    x = _sample_colours()
    back = linear_to_srgb(srgb_to_linear(x))
    assert float((back - x).abs().max()) < 1e-5


def test_srgb_transfer_is_odd_symmetric():
    x = torch.linspace(-1.0, 1.0, 101)
    assert float((srgb_to_linear(-x) + srgb_to_linear(x)).abs().max()) < 1e-6
    assert torch.isfinite(srgb_to_linear(x)).all()
    assert torch.isfinite(linear_to_srgb(x)).all()


def test_oklab_roundtrip():
    x = _sample_colours()
    back = oklab_to_srgb(srgb_to_oklab(x))
    assert float((back - x).abs().max()) < 2e-5


def test_oklab_roundtrip_out_of_gamut():
    """Intermediate stages produce negative and >1 linear values; they must survive."""
    g = torch.Generator().manual_seed(3)
    lin = torch.rand(2048, 3, generator=g) * 2.4 - 0.7
    back = oklab_to_linear(linear_to_oklab(lin))
    assert torch.isfinite(back).all()
    assert float((back - lin).abs().max()) < 1e-4


def test_lms_roundtrip():
    x = _sample_colours()
    assert float((lms_to_linear(linear_to_lms(x)) - x).abs().max()) < 1e-5


def test_known_oklab_anchors():
    """White, black and mid grey must land where Ottosson's definition says."""
    white = linear_to_oklab(torch.tensor([[1.0, 1.0, 1.0]]))
    assert abs(float(white[0, 0]) - 1.0) < 1e-4
    assert float(white[0, 1:].abs().max()) < 1e-3

    black = linear_to_oklab(torch.zeros(1, 3))
    assert float(black.abs().max()) < 1e-6

    grey = linear_to_oklab(torch.full((1, 3), 0.5))
    assert float(grey[0, 1:].abs().max()) < 1e-3
    assert 0.0 < float(grey[0, 0]) < 1.0


def test_luminance_weights():
    lum = luminance(torch.eye(3))
    assert abs(float(lum[0]) - 0.2126) < 1e-6
    assert abs(float(lum[1]) - 0.7152) < 1e-6
    assert abs(float(lum[2]) - 0.0722) < 1e-6


def test_compress_lightness_is_identity_without_overshoot():
    lightness = torch.linspace(0.0, 1.0, 256)
    assert float((compress_lightness(lightness, 0.0, 0.0) - lightness).abs().max()) == 0.0


def test_compress_lightness_is_monotone_and_bounded():
    lightness = torch.linspace(-0.6, 1.9, 1024)
    out = compress_lightness(lightness, head=0.08, toe=0.05)
    assert float(out.max()) <= 1.0 + 1e-6
    assert float(out.min()) >= -1e-6 or float(out.min()) >= -0.0
    diffs = out[1:] - out[:-1]
    assert float(diffs.min()) >= -1e-6


def test_gamut_map_preserves_in_gamut_colours():
    x = _sample_colours()
    lab = srgb_to_oklab(x)
    lin = gamut_map_to_linear(lab)
    assert float((lin - srgb_to_linear(x)).abs().max()) < 1e-4


def test_gamut_map_pulls_impossible_colours_in():
    """A hugely over-saturated Oklab colour must come back displayable, with hue kept."""
    lab = srgb_to_oklab(torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.3, 0.95]]))
    boosted = torch.cat([lab[:, 0:1], lab[:, 1:3] * 4.0], dim=-1)
    lin = gamut_map_to_linear(boosted)
    assert torch.isfinite(lin).all()
    assert float(lin.min()) >= -1e-6 and float(lin.max()) <= 1.0 + 1e-6

    mapped = linear_to_oklab(lin)
    hue_in = torch.atan2(boosted[:, 2], boosted[:, 1])
    hue_out = torch.atan2(mapped[:, 2], mapped[:, 1])
    delta = (hue_in - hue_out).abs()
    assert float(delta.max()) < 0.05, "gamut mapping must not rotate hue"
    assert float((mapped[:, 0] - boosted[:, 0]).abs().max()) < 0.02, "lightness must survive"


def test_gamut_map_white_point_is_exact():
    lab = srgb_to_oklab(torch.ones(1, 3))
    out = linear_to_srgb(gamut_map_to_linear(lab))
    assert float((out - 1.0).abs().max()) < 1e-4


def test_skin_window_is_sane():
    lo, hi = skin_hue_window()
    assert -1.0 < lo < hi < 2.0
    assert hi - lo < 2.0


def test_skin_membership_selects_skin_and_rejects_others():
    skin = srgb_to_oklab(torch.tensor([[0.86, 0.68, 0.57], [0.52, 0.35, 0.26], [0.30, 0.19, 0.14]]))
    other = srgb_to_oklab(torch.tensor([[0.1, 0.4, 0.9], [0.1, 0.8, 0.2], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]))
    assert float(skin_membership(skin).min()) > 0.35
    assert float(skin_membership(other).max()) < 0.1


def test_skin_protection_restores_hue():
    lab_in = srgb_to_oklab(torch.tensor([[0.86, 0.68, 0.57]]))
    # A grade that rotates skin hue by ~40 degrees.
    theta = 0.7
    rot = torch.tensor([[float(torch.cos(torch.tensor(theta))), -float(torch.sin(torch.tensor(theta)))],
                        [float(torch.sin(torch.tensor(theta))), float(torch.cos(torch.tensor(theta)))]])
    lab_out = torch.cat([lab_in[:, 0:1], lab_in[:, 1:3] @ rot.T], dim=-1)

    protected = apply_skin_protection(lab_in, lab_out, amount=1.0)
    h_in = torch.atan2(lab_in[:, 2], lab_in[:, 1])
    h_prot = torch.atan2(protected[:, 2], protected[:, 1])
    assert float((h_in - h_prot).abs().max()) < 0.05
    assert float((protected[:, 0] - lab_out[:, 0]).abs().max()) < 1e-6, "lightness must be untouched"


def test_skin_protection_is_a_noop_at_zero():
    lab_in = srgb_to_oklab(torch.tensor([[0.86, 0.68, 0.57]]))
    lab_out = lab_in * 1.2
    assert torch.equal(apply_skin_protection(lab_in, lab_out, 0.0), lab_out)


def test_neutral_guard_caps_grey_but_not_colour():
    grey_in = srgb_to_oklab(torch.tensor([[0.5, 0.5, 0.5]]))
    wild_out = torch.cat([grey_in[:, 0:1], torch.tensor([[0.25, -0.20]])], dim=-1)
    capped = apply_neutral_guard(grey_in, wild_out)
    assert float(capped[:, 1:3].norm()) <= 0.056

    colour_in = srgb_to_oklab(torch.tensor([[0.9, 0.2, 0.15]]))
    boosted = torch.cat([colour_in[:, 0:1], colour_in[:, 1:3] * 1.3], dim=-1)
    assert float((apply_neutral_guard(colour_in, boosted) - boosted).abs().max()) < 1e-6
