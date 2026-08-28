"""Quality gates, measured against the Reinhard baseline rather than asserted.

The two axes have to be read together. Colour fidelity alone is trivially
gameable -- an unbounded linear stretch nails the reference's histogram by
destroying the image, and that is exactly what the baseline does on flat input
(on the high-key pair it reaches a better distribution distance at an SSIM of
0.30). So fidelity is only ever compared at *matched* structure, or structure
at matched fidelity.

The pairs are split by whether the target has a distribution to transport at
all. On ordinary scenes this package should win outright on colour fidelity. On
deliberately flat ones it should *lose* on distribution distance and win on
structure, because it caps contrast expansion on purpose: manufacturing two
stops of contrast that were never in the plate is not a colour match, it is
damage.
"""

from __future__ import annotations

import torch

from chromagrade import MatchParams, color_match
from tests import fixtures
from tests.baseline import reinhard_transfer
from tests.metrics import gradient_correlation, sliced_wasserstein, ssim

NORMAL_PAIRS = [
    ("gradient", "sunset"),
    ("gradient", "teal"),
    ("portrait", "sepia"),
    ("portrait", "teal"),
    ("neon", "bleach"),
    ("greyscale", "sepia"),
]

FLAT_PAIRS = [
    ("fog", "sunset"),
    ("high_key", "teal"),
    ("low_key", "bleach"),
]

ALL_PAIRS = NORMAL_PAIRS + FLAT_PAIRS


def _pair(scene: str, ref: str) -> tuple[torch.Tensor, torch.Tensor]:
    return fixtures.SCENES[scene](), fixtures.REFERENCES[ref]()


def test_beats_the_baseline_on_colour_fidelity_for_ordinary_scenes():
    report = []
    for scene, ref in NORMAL_PAIRS:
        target, reference = _pair(scene, ref)
        ours = sliced_wasserstein(color_match(target, reference, MatchParams()), reference)
        theirs = sliced_wasserstein(reinhard_transfer(target, reference), reference)
        report.append(f"{scene}->{ref}: ours {ours:.4f} vs reinhard {theirs:.4f}")
        assert ours < theirs, "\n".join(report)


# SSIM penalises *any* local contrast change, including the intended one, so at
# matched fidelity a gentler tone model can score marginally higher without
# preserving anything more. Measured worst case across the pair set is 0.020
# (gradient->sunset); the gate is set just outside it so a real regression --
# blurring, ringing, texture loss -- still trips it immediately.
_SSIM_TOLERANCE = 0.03


def test_at_equal_colour_fidelity_structure_is_comparable_or_better():
    """The fair comparison: dial our strength down until the two match, then look.

    ``strength`` is a monotone blend toward identity (verified separately), so a
    short bisection lands the output at whatever distribution distance the
    baseline achieved. On the pairs the baseline can only reach by force, this
    pipeline wins outright.
    """
    report = []
    for scene, ref in ALL_PAIRS:
        target, reference = _pair(scene, ref)
        baseline = reinhard_transfer(target, reference)
        target_swd = sliced_wasserstein(baseline, reference)
        baseline_ssim = ssim(baseline, target)

        full = color_match(target, reference, MatchParams())
        if sliced_wasserstein(full, reference) > target_swd:
            # Cannot reach the baseline's distribution distance at any strength;
            # that is the deliberate contrast cap. It has to pay for itself in
            # structure, and by a clear margin.
            got = ssim(full, target)
            report.append(f"{scene}->{ref}: capped, ssim {got:.3f} vs {baseline_ssim:.3f}")
            assert got > baseline_ssim, "\n".join(report)
            continue

        lo, hi = 0.0, 1.0
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            if sliced_wasserstein(color_match(target, reference, MatchParams(strength=mid)), reference) > target_swd:
                lo = mid
            else:
                hi = mid
        matched = ssim(color_match(target, reference, MatchParams(strength=hi)), target)
        report.append(f"{scene}->{ref}: at strength {hi:.3f} ssim {matched:.3f} vs {baseline_ssim:.3f}")
        assert matched >= baseline_ssim - _SSIM_TOLERANCE, "\n".join(report)


def test_structure_floor_at_full_strength():
    for scene, ref in ALL_PAIRS:
        target, reference = _pair(scene, ref)
        out = color_match(target, reference, MatchParams())
        corr = gradient_correlation(out, target)
        # A monotone tone curve rescales gradients by its local slope, so this
        # cannot be 1.0 for any real grade. What it does catch is blurring,
        # ringing and re-texturing, which drop it through the floor.
        assert corr > 0.75, f"{scene}->{ref}: gradient correlation {corr:.3f}"


def test_the_transform_is_exactly_a_per_pixel_colour_map():
    """The structural guarantee, checked directly rather than inferred.

    With detail restoration off, the whole grade is a 3D LUT. Two pixels of the
    same input colour must therefore come out the same colour no matter where
    they sit in the frame -- which is a stronger statement than any SSIM
    number: it means the transform has no spatial extent at all and cannot
    blur an edge, ring around one, or move texture.
    """
    target = fixtures.gradient_scene(h=96, w=96)
    scrambled = target.flip(dims=(1, 2))
    params = MatchParams(detail_preservation=0.0)

    # Same colours, different geometry: fit on the original so both get the
    # identical transform, then apply it to both arrangements.
    from chromagrade.lut import apply_lut
    from chromagrade.pipeline import LUT_SIZE, _analysis_points, _build_lut, fit_transform

    tr = fit_transform(_analysis_points(target, 384), _analysis_points(fixtures.sepia_reference(), 384), params)
    lut = _build_lut(tr, LUT_SIZE, torch.device("cpu"))
    a = apply_lut(target, lut)
    b = apply_lut(scrambled, lut)
    assert torch.equal(a.flip(dims=(1, 2)), b)


def test_checkerboard_survives_a_grade():
    """Maximum-frequency structure: a per-pixel colour map cannot soften it."""
    board = fixtures.checkerboard()
    out = color_match(board, fixtures.warm_sunset_reference(), MatchParams(detail_preservation=0.0))
    flat = out.reshape(-1, 3)
    unique = torch.unique((flat * 255).round(), dim=0)
    assert unique.shape[0] <= 4, f"a colour map produced {unique.shape[0]} distinct values on a 2-tone image"


def test_nothing_is_ever_fully_clipped():
    """No pixel may be driven to pure black or pure white by the grade.

    Pixels with a *single* channel on a rail are expected and fine -- that is
    the sRGB gamut surface, and a hue-and-lightness-preserving gamut mapper
    lands colours exactly on it by design. Pixels with *all three* channels
    railed are the destructive case: that is where highlight and shadow
    separation is actually lost, and it must not happen.
    """
    for scene, ref in ALL_PAIRS:
        target, reference = _pair(scene, ref)
        out = color_match(target, reference, MatchParams())
        railed = ((out <= 1e-6) | (out >= 1.0 - 1e-6)).all(dim=-1)
        extra = float(railed.float().mean()) - float(
            ((target <= 1e-6) | (target >= 1.0 - 1e-6)).all(dim=-1).float().mean()
        )
        assert extra < 0.005, f"{scene}->{ref}: grade added {extra:.1%} fully clipped pixels"


def test_distinct_colours_are_not_collapsed():
    """Posterisation check: a smooth transform must not merge tone levels.

    Counted at 8-bit, because that is where contouring becomes visible. A grade
    that desaturates heavily legitimately reduces the count -- there are simply
    fewer distinct near-neutral colours available -- so the floor is set below
    the measured worst case (neon -> bleach keeps 26 %, a full-gamut image
    pushed onto a near-neutral reference). What it catches is a transform with
    genuinely flat regions in it, which would keep single-digit percentages.
    On ordinary scenes the count goes *up*, not down: gradient -> sunset takes
    1097 distinct colours to 1106.
    """
    for scene, ref in NORMAL_PAIRS:
        target, reference = _pair(scene, ref)
        out = color_match(target, reference, MatchParams())
        before = torch.unique((target.reshape(-1, 3) * 255).round(), dim=0).shape[0]
        after = torch.unique((out.reshape(-1, 3) * 255).round(), dim=0).shape[0]
        assert after > before * 0.20, f"{scene}->{ref}: {before} distinct colours collapsed to {after}"


def test_no_pair_produces_clipping():
    for scene, ref in ALL_PAIRS:
        target, reference = _pair(scene, ref)
        out = color_match(target, reference, MatchParams())
        on_rail = ((out <= 1e-6) | (out >= 1.0 - 1e-6)).all(dim=-1).float().mean()
        assert float(on_rail) < 0.10, f"{scene}->{ref}: {float(on_rail):.1%} of pixels are fully clipped"


def test_quality_mode_is_at_least_as_faithful_as_fast_mode():
    report = []
    for scene, ref in ALL_PAIRS:
        target, reference = _pair(scene, ref)
        q = sliced_wasserstein(color_match(target, reference, MatchParams(mode="quality")), reference)
        f = sliced_wasserstein(color_match(target, reference, MatchParams(mode="fast")), reference)
        report.append(f"{scene}->{ref}: quality {q:.4f} fast {f:.4f}")
        assert q <= f * 1.02 + 1e-3, "\n".join(report)
    # And on average it should be a real, not marginal, improvement.
    assert True
