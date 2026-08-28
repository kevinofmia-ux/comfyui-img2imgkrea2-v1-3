"""Monotone maps, quantile fitting, deterministic rotations and IDT."""

from __future__ import annotations

import torch

from chromagrade.stats import mkl_transform, quantiles, sym_inv_sqrt, sym_sqrt, weighted_mean_cov
from chromagrade.transport import IDTChain, MonotoneMap, build_quantile_map, fibonacci_rotations


def test_quantiles_match_a_known_distribution():
    x = torch.linspace(0.0, 1.0, 100_000)
    q = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    got = quantiles(x, q)
    assert float((got - q).abs().max()) < 2e-3


def test_quantiles_handle_constant_and_empty():
    assert float((quantiles(torch.full((1000,), 0.42), torch.tensor([0.0, 0.5, 1.0])) - 0.42).abs().max()) < 1e-6
    assert torch.isfinite(quantiles(torch.empty(0), torch.tensor([0.5]))).all()


def test_quantiles_are_deterministic():
    g = torch.Generator().manual_seed(5)
    x = torch.rand(50_000, generator=g)
    q = torch.linspace(0.01, 0.99, 33)
    assert torch.equal(quantiles(x, q), quantiles(x, q))


def test_monotone_map_interpolates_knots():
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    y = torch.tensor([0.0, 0.10, 0.40, 0.80, 1.0])
    m = MonotoneMap(x, y)
    assert float((m(x) - y).abs().max()) < 1e-5


def test_monotone_map_is_monotone_and_extrapolates_linearly():
    x = torch.tensor([0.1, 0.3, 0.55, 0.8, 0.95])
    y = torch.tensor([0.05, 0.35, 0.50, 0.90, 0.98])
    m = MonotoneMap(x, y)
    t = torch.linspace(-0.5, 1.6, 4096)
    v = m(t)
    assert torch.isfinite(v).all()
    assert float((v[1:] - v[:-1]).min()) >= -1e-6, "map must never invert local contrast"


def test_quantile_map_of_identical_distributions_is_identity():
    g = torch.Generator().manual_seed(9)
    x = torch.rand(60_000, generator=g) ** 1.7
    m = build_quantile_map(x, x, n_knots=41, smooth=2)
    probe = torch.linspace(0.02, 0.98, 512)
    assert float((m(probe) - probe).abs().max()) < 5e-3


def test_quantile_map_moves_one_distribution_onto_another():
    g = torch.Generator().manual_seed(13)
    src = torch.rand(80_000, generator=g) * 0.3 + 0.1
    dst = torch.rand(80_000, generator=g) * 0.8 + 0.15
    m = build_quantile_map(src, dst, n_knots=41)
    moved = m(src)
    q = torch.tensor([0.1, 0.5, 0.9])
    assert float((quantiles(moved, q) - quantiles(dst, q)).abs().max()) < 0.03


def test_quantile_map_respects_the_slope_clamp():
    g = torch.Generator().manual_seed(17)
    src = torch.rand(40_000, generator=g) * 0.01 + 0.5  # almost no spread
    dst = torch.rand(40_000, generator=g)  # full spread
    m = build_quantile_map(src, dst, n_knots=41, slope_range=(0.25, 4.0))
    t = torch.linspace(0.495, 0.515, 2048)
    slope = (m(t)[1:] - m(t)[:-1]) / (t[1] - t[0])
    assert float(slope.max()) < 4.6, "an unbounded stretch is how contrast blows out"


def test_quantile_map_identity_blend():
    g = torch.Generator().manual_seed(19)
    src = torch.rand(20_000, generator=g) * 0.4
    dst = torch.rand(20_000, generator=g) * 0.4 + 0.6
    m = build_quantile_map(src, dst, identity_blend=1.0)
    probe = torch.linspace(0.05, 0.35, 256)
    assert float((m(probe) - probe).abs().max()) < 1e-3


def test_fibonacci_rotations_are_orthonormal_and_deterministic():
    r = fibonacci_rotations(24, torch.device("cpu"))
    eye = torch.eye(3).expand(24, 3, 3)
    assert float((r @ r.transpose(1, 2) - eye).abs().max()) < 1e-5
    assert float(torch.linalg.det(r).abs().min()) > 0.999
    assert torch.equal(r, fibonacci_rotations(24, torch.device("cpu")))


def test_fibonacci_rotations_cover_the_sphere():
    axes = fibonacci_rotations(64, torch.device("cpu"))[:, 0, :]
    # No pair of directions should be nearly parallel: that is the whole point
    # of replacing random rotations with a spread construction.
    cos = (axes @ axes.T).abs() - torch.eye(64) * 2
    assert float(cos.max()) < 0.995


def test_sym_sqrt_and_inv_sqrt():
    g = torch.Generator().manual_seed(23)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    m = a @ a.T + torch.eye(3, dtype=torch.float64) * 0.1
    s = sym_sqrt(m)
    assert float((s @ s - m).abs().max()) < 1e-8
    inv = sym_inv_sqrt(m)
    assert float((inv @ inv @ m - torch.eye(3, dtype=torch.float64)).abs().max()) < 1e-7


def test_sym_inv_sqrt_survives_a_singular_matrix():
    m = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    out = sym_inv_sqrt(m)
    assert torch.isfinite(out).all()


def test_mkl_maps_gaussian_onto_gaussian():
    g = torch.Generator().manual_seed(29)
    src = torch.randn(50_000, 3, generator=g) @ torch.tensor([[0.3, 0.1, 0.0], [0.0, 0.2, 0.05], [0.0, 0.0, 0.15]])
    dst = torch.randn(50_000, 3, generator=g) @ torch.tensor([[0.5, -0.2, 0.1], [0.0, 0.4, 0.0], [0.0, 0.0, 0.3]]) + 0.4

    ms, cs, _ = weighted_mean_cov(src)
    md, cd, _ = weighted_mean_cov(dst)
    a, b = mkl_transform(ms, cs, md, cd)
    moved = src.to(torch.float64) @ a.T + b

    mm, cm, _ = weighted_mean_cov(moved)
    assert float((mm - md).abs().max()) < 1e-6
    assert float((cm - cd).abs().max()) < 1e-4


def test_mkl_is_identity_for_identical_clouds():
    g = torch.Generator().manual_seed(31)
    pts = torch.randn(20_000, 3, generator=g) * 0.2
    m, c, _ = weighted_mean_cov(pts)
    a, b = mkl_transform(m, c, m, c)
    assert float((a - torch.eye(3, dtype=a.dtype)).abs().max()) < 1e-6
    assert float(b.abs().max()) < 1e-6


def test_mkl_caps_the_gain_on_a_degenerate_source():
    g = torch.Generator().manual_seed(37)
    src = torch.randn(20_000, 3, generator=g) * torch.tensor([0.2, 1e-5, 1e-5])
    dst = torch.randn(20_000, 3, generator=g) * 0.3
    ms, cs, _ = weighted_mean_cov(src)
    md, cd, _ = weighted_mean_cov(dst)
    a, _ = mkl_transform(ms, cs, md, cd, max_gain=6.0)
    assert float(torch.linalg.svdvals(a).max()) <= 6.01


def test_idt_chain_tightens_the_match_and_replays_exactly():
    g = torch.Generator().manual_seed(41)
    src = torch.rand(40_000, 3, generator=g) * torch.tensor([0.4, 0.2, 0.2])
    dst = torch.rand(40_000, 3, generator=g) ** 2 * torch.tensor([0.9, 0.4, 0.5]) + 0.05

    chain = IDTChain.fit(src, dst, iterations=10)
    assert len(chain) == 10
    moved = chain(src)

    q = torch.linspace(0.05, 0.95, 19)
    before = sum(float((quantiles(src[:, c], q) - quantiles(dst[:, c], q)).abs().mean()) for c in range(3))
    after = sum(float((quantiles(moved[:, c], q) - quantiles(dst[:, c], q)).abs().mean()) for c in range(3))
    assert after < before * 0.25

    # Replayable as a pure function of colour -- this is what lets it be baked
    # into a LUT rather than applied as a per-pixel displacement.
    assert torch.equal(chain(src), moved)
    assert torch.isfinite(chain(torch.rand(1000, 3) * 3.0 - 1.0)).all()


def test_idt_declines_to_fit_tiny_clouds():
    assert len(IDTChain.fit(torch.rand(4, 3), torch.rand(4, 3))) == 0
