"""LUT lattice construction, smoothing and tetrahedral interpolation."""

from __future__ import annotations

import torch

from chromagrade.guided import box_filter, guided_filter, restore_detail
from chromagrade.lut import apply_lut, blend_toward_identity, identity_lattice, lattice_coordinates, smooth_lut


def test_identity_lut_is_a_noop():
    lut = identity_lattice(33, torch.device("cpu"))
    g = torch.Generator().manual_seed(3)
    img = torch.rand(1, 37, 41, 3, generator=g)
    out = apply_lut(img, lut)
    assert float((out - img).abs().max()) < 1e-6


def test_lut_reproduces_lattice_nodes_exactly():
    size = 17
    coords = lattice_coordinates(size, torch.device("cpu"))
    g = torch.Generator().manual_seed(5)
    lut = torch.rand(size, size, size, 3, generator=g)
    out = apply_lut(coords.reshape(1, 1, -1, 3), lut).reshape(-1, 3)
    assert float((out - lut.reshape(-1, 3)).abs().max()) < 1e-5


def test_tetrahedral_preserves_the_neutral_axis():
    """The reason tetrahedral is used instead of trilinear: greys stay grey."""
    size = 33
    coords = lattice_coordinates(size, torch.device("cpu"))
    # A LUT that is a pure per-channel curve keeps neutrals neutral by
    # construction; interpolation must not break that between lattice nodes.
    lut = (coords**1.8).reshape(size, size, size, 3)
    grey = torch.linspace(0.0, 1.0, 512).view(1, 1, -1, 1).expand(1, 1, 512, 3).contiguous()
    out = apply_lut(grey, lut)
    spread = out.amax(dim=-1) - out.amin(dim=-1)
    assert float(spread.max()) < 1e-5


def test_tetrahedral_matches_the_true_function_closely():
    size = 33
    coords = lattice_coordinates(size, torch.device("cpu"))

    def f(x):
        return (x**1.6) * 0.8 + 0.1 * x.flip(-1) + 0.05

    lut = f(coords).reshape(size, size, size, 3).clamp(0, 1)
    g = torch.Generator().manual_seed(7)
    probe = torch.rand(1, 64, 64, 3, generator=g)
    err = (apply_lut(probe, lut) - f(probe).clamp(0, 1)).abs().max()
    assert float(err) < 2.0 / 255.0, "33-cube interpolation error must stay sub-code-value"


def test_apply_lut_is_chunk_invariant():
    size = 17
    g = torch.Generator().manual_seed(11)
    lut = torch.rand(size, size, size, 3, generator=g)
    img = torch.rand(1, 128, 128, 3, generator=g)
    assert torch.equal(apply_lut(img, lut, chunk_pixels=1 << 20), apply_lut(img, lut, chunk_pixels=1000))


def test_apply_lut_clamps_out_of_range_input():
    lut = identity_lattice(9, torch.device("cpu"))
    img = torch.tensor([[[[-0.5, 0.5, 1.7]]]])
    out = apply_lut(img, lut)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_apply_lut_rejects_a_malformed_table():
    for bad in (torch.rand(4, 4, 5, 3), torch.rand(1, 4, 4, 4), torch.rand(1, 1, 1, 3)):
        try:
            apply_lut(torch.rand(1, 2, 2, 3), bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for LUT shape {tuple(bad.shape)}")


def test_smoothing_leaves_an_affine_lut_alone():
    """Linear-extrapolating boundaries mean black and white do not creep."""
    lut = identity_lattice(33, torch.device("cpu"))
    assert float((smooth_lut(lut, 1.0, passes=2) - lut).abs().max()) < 1e-5


def test_smoothing_actually_smooths():
    size = 17
    lut = identity_lattice(size, torch.device("cpu")).clone()
    lut[8, 8, 8] = torch.tensor([1.0, 0.0, 1.0])  # a spike
    before = float((lut[8, 8, 8] - identity_lattice(size, torch.device("cpu"))[8, 8, 8]).abs().max())
    after = float((smooth_lut(lut, 1.0)[8, 8, 8] - identity_lattice(size, torch.device("cpu"))[8, 8, 8]).abs().max())
    assert after < before


def test_blend_toward_identity_endpoints():
    g = torch.Generator().manual_seed(13)
    lut = torch.rand(17, 17, 17, 3, generator=g)
    ident = identity_lattice(17, torch.device("cpu"))
    assert torch.equal(blend_toward_identity(lut, 1.0), lut)
    assert float((blend_toward_identity(lut, 0.0) - ident).abs().max()) < 1e-6
    half = blend_toward_identity(lut, 0.5)
    assert float((half - 0.5 * (lut + ident)).abs().max()) < 1e-6


def test_box_filter_averages_correctly():
    x = torch.ones(1, 8, 8, 1)
    assert float((box_filter(x, 2) - 1.0).abs().max()) < 1e-6
    ramp = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1).expand(1, 8, 8, 1).contiguous()
    out = box_filter(ramp, 1)
    # Interior of a linear ramp is unchanged by a symmetric average.
    assert float((out[0, 4, 2:6, 0] - ramp[0, 4, 2:6, 0]).abs().max()) < 1e-5


def test_guided_filter_preserves_edges():
    step = torch.cat([torch.zeros(1, 32, 16, 1), torch.ones(1, 32, 16, 1)], dim=2)
    g = torch.Generator().manual_seed(17)
    noisy = step + torch.randn(1, 32, 32, 1, generator=g) * 0.05
    out = guided_filter(step, noisy, radius=4, eps=1e-4)
    assert float(out[0, 16, 4, 0]) < 0.15
    assert float(out[0, 16, 27, 0]) > 0.85
    # It should have removed most of the noise inside each flat region.
    assert float(out[0, :, :14, 0].std()) < float(noisy[0, :, :14, 0].std()) * 0.5


def test_restore_detail_endpoints():
    g = torch.Generator().manual_seed(19)
    base = torch.rand(1, 48, 48, 1, generator=g) * 0.2 + 0.4
    graded = base * 1.4
    assert torch.equal(restore_detail(base, graded, 0.0), graded)
    full = restore_detail(base, graded, 1.0)
    assert torch.isfinite(full).all()
    # At amount = 1 the detail amplitude should be back to the original's.
    def detail_energy(x):
        return float((x - box_filter(x, 3)).std())

    assert abs(detail_energy(full) - detail_energy(base)) < abs(detail_energy(graded) - detail_energy(base))
