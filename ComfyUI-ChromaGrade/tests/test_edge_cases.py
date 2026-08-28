"""Degenerate and hostile inputs. Nothing here may raise, NaN or posterise."""

from __future__ import annotations

import torch

from chromagrade import MatchParams, color_match
from tests import fixtures
from tests.metrics import banding_score

REFERENCE = fixtures.warm_sunset_reference()


def _assert_sane(out: torch.Tensor, shape):
    assert out.shape == shape
    assert torch.isfinite(out).all(), "NaN or Inf reached the output"
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_constant_black_target_stays_black():
    """Exposure matching is a multiplicative gain, and zero times anything is zero.

    A frame with literally no signal has nothing to grade, and no amount of
    gain will invent any. It comes back black rather than being lifted to the
    reference's average -- predictable, and the honest answer.
    """
    black = fixtures.constant(0.0)
    out = color_match(black, REFERENCE, MatchParams())
    _assert_sane(out, black.shape)
    assert float(out.max()) < 0.06


def test_constant_white_target_takes_the_reference_level():
    """A flat frame still has a well-defined level, so level matching applies.

    The tone curve's *shape* is gated off (there is no distribution to
    transport) but exposure and cast are not, so a white card graded against a
    dark reference comes back as a darker card -- flat, in range, and sitting
    where the reference sits.
    """
    from chromagrade.colorspace import srgb_to_oklab

    white = fixtures.constant(1.0)
    out = color_match(white, REFERENCE, MatchParams())
    _assert_sane(out, white.shape)

    spread = out.reshape(-1, 3).amax(dim=0) - out.reshape(-1, 3).amin(dim=0)
    assert float(spread.max()) < 1e-4, "a flat input must stay flat"

    ref_l = srgb_to_oklab(REFERENCE.reshape(-1, 3))[:, 0].mean()
    out_l = srgb_to_oklab(out.reshape(-1, 3))[:, 0].mean()
    assert abs(float(out_l - ref_l)) < 0.30, "the level should land near the reference's"


def test_constant_white_target_is_untouched_when_tonality_is_off():
    white = fixtures.constant(1.0)
    out = color_match(white, REFERENCE, MatchParams(tonal_transfer=0.0))
    _assert_sane(out, white.shape)
    assert float(out.min()) > 0.90


def test_constant_mid_target_stays_flat():
    mid = fixtures.constant(0.5)
    out = color_match(mid, REFERENCE, MatchParams())
    _assert_sane(out, mid.shape)
    per_channel_spread = out.reshape(-1, 3).amax(dim=0) - out.reshape(-1, 3).amin(dim=0)
    assert float(per_channel_spread.max()) < 1e-4, "a flat input must stay flat"


def test_constant_reference_drives_the_target_to_it():
    out = color_match(fixtures.gradient_scene(), fixtures.constant(0.0), MatchParams())
    _assert_sane(out, fixtures.gradient_scene().shape)
    assert float(out.max()) < 0.05


def test_greyscale_target_gets_a_cast_not_a_mess():
    from chromagrade.colorspace import chroma, srgb_to_oklab

    grey = fixtures.greyscale()
    out = color_match(grey, REFERENCE, MatchParams())
    _assert_sane(out, grey.shape)
    c = chroma(srgb_to_oklab(out))
    assert float(c.max()) < 0.10, "a neutral plate must not become saturated"
    assert float(c.mean()) > 0.002, "it should still pick up the reference's cast"


def test_low_contrast_target_is_gradeable():
    fog = fixtures.fog()
    out = color_match(fog, REFERENCE, MatchParams())
    _assert_sane(out, fog.shape)
    assert float((out - fog).abs().max()) > 0.02, "low contrast is not the same as degenerate"


def test_high_key_and_low_key_targets():
    for scene in (fixtures.high_key(), fixtures.low_key()):
        for ref in (REFERENCE, fixtures.cool_teal_reference(), fixtures.bleach_reference()):
            out = color_match(scene, ref, MatchParams())
            _assert_sane(out, scene.shape)


def test_saturated_target_does_not_clip_to_paste():
    neon = fixtures.neon()
    out = color_match(neon, fixtures.bleach_reference(), MatchParams())
    _assert_sane(out, neon.shape)
    # If gamut handling had degenerated into a clamp, a large fraction of pixels
    # would sit exactly on a channel boundary.
    on_rail = ((out <= 1e-6) | (out >= 1.0 - 1e-6)).any(dim=-1).float().mean()
    assert float(on_rail) < 0.05


def test_smooth_ramp_does_not_band():
    ramp = fixtures.radial_ramp()
    out = color_match(ramp, fixtures.cool_teal_reference(), MatchParams())
    _assert_sane(out, ramp.shape)
    assert banding_score(out) < 12.0 * banding_score(ramp) + 2e-3


def test_one_pixel_images():
    tiny = torch.tensor([[[[0.3, 0.6, 0.2]]]])
    out = color_match(tiny, REFERENCE, MatchParams())
    _assert_sane(out, tiny.shape)
    out2 = color_match(fixtures.gradient_scene(), tiny, MatchParams())
    _assert_sane(out2, fixtures.gradient_scene().shape)


def test_extreme_aspect_ratios():
    for shape in ((1, 1, 512, 3), (1, 512, 1, 3), (1, 3, 900, 3)):
        img = torch.rand(shape, generator=torch.Generator().manual_seed(5))
        out = color_match(img, REFERENCE, MatchParams())
        _assert_sane(out, img.shape)


def test_out_of_range_input_is_clamped_not_propagated():
    dirty = fixtures.gradient_scene() * 1.6 - 0.3
    out = color_match(dirty, REFERENCE, MatchParams())
    _assert_sane(out, dirty.shape)


def test_nan_free_on_pathological_reference():
    ref = torch.zeros(1, 8, 8, 3)
    ref[0, 0, 0] = 1.0  # one lone white pixel in pure black
    out = color_match(fixtures.gradient_scene(), ref, MatchParams())
    _assert_sane(out, fixtures.gradient_scene().shape)


def test_all_control_extremes_are_safe():
    target, ref = fixtures.portrait(), fixtures.cool_teal_reference()
    for mode in ("quality", "fast"):
        for value in (0.0, 1.0):
            params = MatchParams(
                mode=mode,
                strength=value,
                white_balance=value,
                tonal_transfer=value,
                palette_transfer=value,
                skin_protection=value,
                detail_preservation=value,
                saturation=value * 2.0,
            )
            _assert_sane(color_match(target, ref, params), target.shape)


def test_parameters_are_clamped_not_trusted():
    params = MatchParams(
        mode="nonsense",
        strength=7.0,
        white_balance=-3.0,
        tonal_transfer=float("nan"),
        palette_transfer=1e9,
        skin_protection=-1.0,
        detail_preservation=2.0,
        saturation=50.0,
    ).normalised()
    assert params.mode == "quality"
    assert params.strength == 1.0
    assert params.white_balance == 0.0
    assert params.palette_transfer == 1.0
    assert params.skin_protection == 0.0
    assert params.detail_preservation == 1.0
    assert params.saturation == 2.0
    # NaN propagates silently through Python's min/max, so it must be caught
    # explicitly and replaced by the default rather than clamped.
    assert params.tonal_transfer == 1.0

    out = color_match(fixtures.fog(), REFERENCE, params)
    _assert_sane(out, fixtures.fog().shape)
    assert float(out.mean()) > 0.05, "a NaN control must not collapse the frame to black"


def test_non_numeric_parameters_fall_back_to_defaults():
    params = MatchParams(strength="loud", saturation=None, detail_preservation=object()).normalised()
    assert params.strength == 1.0
    assert params.saturation == 1.0
    assert params.detail_preservation == 0.5


def test_bad_inputs_produce_readable_errors():
    good = fixtures.gradient_scene()
    cases = [
        (torch.empty(0, 4, 4, 3), good, "empty"),
        (good, torch.empty(1, 0, 4, 3), "empty"),
        (torch.rand(1, 4, 4, 2), good, "channels"),
        (torch.rand(4, 4), good, "IMAGE"),
    ]
    for target, reference, needle in cases:
        try:
            color_match(target, reference, MatchParams())
        except (ValueError, TypeError) as exc:
            assert needle in str(exc), f"unhelpful message for {needle!r}: {exc}"
            continue
        raise AssertionError(f"expected a clear error for the {needle!r} case")

    try:
        color_match("not a tensor", good, MatchParams())
    except TypeError as exc:
        assert "torch.Tensor" in str(exc)
    else:
        raise AssertionError("expected a TypeError for a non-tensor input")
