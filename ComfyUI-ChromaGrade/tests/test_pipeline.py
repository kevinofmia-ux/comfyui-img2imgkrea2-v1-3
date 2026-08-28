"""End-to-end behaviour: shapes, batches, devices, controls, determinism."""

from __future__ import annotations

import torch

from chromagrade import MatchParams, color_match, fit_transform
from chromagrade.pipeline import _analysis_points, _reference_groups
from tests import fixtures
from tests.metrics import mean_delta_ok, sliced_wasserstein, ssim

TARGET = fixtures.gradient_scene()
REFERENCE = fixtures.warm_sunset_reference()


def test_output_shape_dtype_and_range():
    out = color_match(TARGET, REFERENCE, MatchParams())
    assert out.shape == TARGET.shape
    assert out.dtype == TARGET.dtype
    assert out.device == TARGET.device
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_accepts_a_three_dimensional_image():
    out = color_match(TARGET[0], REFERENCE, MatchParams())
    assert out.shape == (1,) + tuple(TARGET.shape[1:])


def test_resolution_is_preserved_across_mismatched_inputs():
    target = fixtures.portrait(h=201, w=349)
    out = color_match(target, fixtures.cool_teal_reference(h=64, w=512), MatchParams())
    assert out.shape == (1, 201, 349, 3)


def test_the_grade_actually_moves_the_colour():
    out = color_match(TARGET, REFERENCE, MatchParams())
    before = sliced_wasserstein(TARGET, REFERENCE)
    after = sliced_wasserstein(out, REFERENCE)
    assert after < before * 0.4, "the output must sit far closer to the reference's distribution"
    assert mean_delta_ok(out, TARGET) > 0.02, "a trivial filter is not a colour match"


def test_structure_survives():
    out = color_match(TARGET, REFERENCE, MatchParams())
    assert ssim(out, TARGET) > 0.80


def test_strength_zero_is_bit_exact_passthrough():
    out = color_match(TARGET, REFERENCE, MatchParams(strength=0.0))
    assert torch.equal(out, TARGET)


def test_strength_is_monotone():
    ref_dist = [
        sliced_wasserstein(color_match(TARGET, REFERENCE, MatchParams(strength=s)), REFERENCE)
        for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    for a, b in zip(ref_dist, ref_dist[1:], strict=False):
        assert b <= a + 1e-4, f"raising strength must not move away from the reference: {ref_dist}"


def test_self_reference_is_nearly_a_no_op():
    out = color_match(TARGET, TARGET, MatchParams(detail_preservation=0.0))
    assert float((out - TARGET).abs().max()) < 4.0 / 255.0


def test_every_control_has_an_effect():
    base = color_match(TARGET, REFERENCE, MatchParams())
    variants = {
        "white_balance": MatchParams(white_balance=0.0),
        "tonal_transfer": MatchParams(tonal_transfer=0.0),
        "palette_transfer": MatchParams(palette_transfer=0.0),
        "saturation": MatchParams(saturation=0.4),
        "mode": MatchParams(mode="fast"),
    }
    for name, params in variants.items():
        out = color_match(TARGET, REFERENCE, params)
        assert float((out - base).abs().max()) > 2.0 / 255.0, f"{name} does nothing"


def test_skin_protection_holds_face_hue():
    from chromagrade.colorspace import srgb_to_oklab

    target = fixtures.portrait()
    reference = fixtures.cool_teal_reference()
    face = torch.zeros(target.shape[1], target.shape[2], dtype=torch.bool)
    yy = torch.linspace(0, 1, target.shape[1]).view(-1, 1)
    xx = torch.linspace(0, 1, target.shape[2]).view(1, -1)
    face |= (((xx - 0.5) / 0.20) ** 2 + ((yy - 0.60) / 0.22) ** 2) < 1.0

    def face_hue(img):
        lab = srgb_to_oklab(img[0])[face]
        return torch.atan2(lab[:, 2], lab[:, 1]).mean()

    h_in = face_hue(target)
    h_off = face_hue(color_match(target, reference, MatchParams(skin_protection=0.0)))
    h_on = face_hue(color_match(target, reference, MatchParams(skin_protection=1.0)))
    assert abs(float(h_on - h_in)) < abs(float(h_off - h_in))


def test_detail_preservation_restores_texture():
    from chromagrade.colorspace import srgb_to_oklab
    from chromagrade.guided import box_filter

    target = fixtures.gradient_scene()
    reference = fixtures.bleach_reference()

    def detail_energy(img):
        lightness = srgb_to_oklab(img)[..., 0:1]
        return float((lightness - box_filter(lightness, 3)).std())

    want = detail_energy(target)
    off = detail_energy(color_match(target, reference, MatchParams(detail_preservation=0.0)))
    on = detail_energy(color_match(target, reference, MatchParams(detail_preservation=1.0)))
    assert abs(on - want) < abs(off - want)


def test_determinism_across_repeated_calls():
    a = color_match(TARGET, REFERENCE, MatchParams())
    b = color_match(TARGET, REFERENCE, MatchParams())
    assert torch.equal(a, b)


def test_both_modes_are_stable_and_different():
    quality = color_match(TARGET, REFERENCE, MatchParams(mode="quality"))
    fast = color_match(TARGET, REFERENCE, MatchParams(mode="fast"))
    for out in (quality, fast):
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert not torch.equal(quality, fast)
    # Quality mode exists to match the distribution more tightly; it had better.
    assert sliced_wasserstein(quality, REFERENCE) <= sliced_wasserstein(fast, REFERENCE) * 1.05


def test_reference_groups_are_predictable():
    assert _reference_groups(4, 1) == [[0]] * 4
    assert _reference_groups(3, 3) == [[0], [1], [2]]
    assert _reference_groups(2, 5) == [[0, 1, 2, 3, 4]] * 2


def test_batch_broadcast_one_reference():
    batch = torch.cat([fixtures.gradient_scene(), fixtures.gradient_scene() * 0.6], dim=0)
    out = color_match(batch, REFERENCE, MatchParams())
    assert out.shape == batch.shape
    single = color_match(batch[0:1], REFERENCE, MatchParams())
    assert float((out[0:1] - single).abs().max()) < 1e-6


def test_batch_pairwise():
    batch = torch.cat([fixtures.gradient_scene(), fixtures.gradient_scene() * 0.6], dim=0)
    refs = torch.cat([fixtures.warm_sunset_reference(), fixtures.cool_teal_reference()], dim=0)
    out = color_match(batch, refs, MatchParams())
    assert out.shape == batch.shape
    assert float((out[0] - out[1]).abs().max()) > 0.05
    solo = color_match(batch[1:2], refs[1:2], MatchParams())
    assert float((out[1:2] - solo).abs().max()) < 1e-6


def test_batch_mismatch_pools_the_references():
    batch = torch.cat([fixtures.gradient_scene()] * 2, dim=0)
    refs = torch.cat(
        [fixtures.warm_sunset_reference(), fixtures.cool_teal_reference(), fixtures.sepia_reference()], dim=0
    )
    out = color_match(batch, refs, MatchParams())
    assert out.shape == batch.shape
    assert torch.equal(out[0], out[1]), "a pooled reference must be applied identically"


def test_alpha_channel_passes_through_untouched():
    rgba = torch.cat([TARGET, torch.full_like(TARGET[..., :1], 0.37)], dim=-1)
    out = color_match(rgba, REFERENCE, MatchParams())
    assert out.shape == rgba.shape
    assert float((out[..., 3] - 0.37).abs().max()) < 1e-6


def test_single_channel_input_stays_single_channel():
    grey = fixtures.greyscale()[..., :1]
    out = color_match(grey, REFERENCE, MatchParams())
    assert out.shape == grey.shape
    assert torch.isfinite(out).all()


def test_dtype_is_round_tripped():
    for dtype in (torch.float16, torch.float64):
        out = color_match(TARGET.to(dtype), REFERENCE.to(dtype), MatchParams())
        assert out.dtype == dtype
        assert torch.isfinite(out).all()


def test_analysis_points_are_capped_and_deterministic():
    big = torch.rand(1, 900, 1600, 3, generator=torch.Generator().manual_seed(3))
    pts = _analysis_points(big, 384)
    assert pts.shape[0] <= 65_536
    assert torch.equal(pts, _analysis_points(big, 384))


def test_fit_reports_what_it_did():
    pts_t = _analysis_points(TARGET, 256)
    pts_r = _analysis_points(REFERENCE, 256)
    tr = fit_transform(pts_t, pts_r, MatchParams())
    assert tr.notes["mode"] == "quality"
    assert tr.notes["idt_steps"] == 10
    assert 0.0 <= tr.notes["confidence_tone"] <= 1.0
    assert 0.0 <= tr.notes["confidence_palette"] <= 1.0


def test_cuda_path_matches_cpu_closely():
    if not torch.cuda.is_available():
        return "skipped: no CUDA device"
    out_cpu = color_match(TARGET, REFERENCE, MatchParams())
    out_gpu = color_match(TARGET.cuda(), REFERENCE.cuda(), MatchParams())
    assert out_gpu.device.type == "cuda"
    assert torch.isfinite(out_gpu).all()
    assert float((out_gpu.cpu() - out_cpu).abs().max()) < 3.0 / 255.0
    return None


def test_cuda_is_deterministic():
    if not torch.cuda.is_available():
        return "skipped: no CUDA device"
    a = color_match(TARGET.cuda(), REFERENCE.cuda(), MatchParams())
    b = color_match(TARGET.cuda(), REFERENCE.cuda(), MatchParams())
    assert torch.equal(a, b)
    return None
