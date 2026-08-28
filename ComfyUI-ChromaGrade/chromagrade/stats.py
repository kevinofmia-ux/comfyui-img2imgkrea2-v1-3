"""Robust statistics and the closed-form Monge-Kantorovich linear transport.

Two things matter here and both are about degenerate inputs. A user will hand
this node a solid black frame, a greyscale plate, or a flat sky, and a
covariance estimated from those is singular. Every routine below is written so
that the singular case produces a *sane, stable* transform (typically identity
or a pure scale) rather than an exception or an inf.

All matrix work happens in float64. A 3x3 eigendecomposition costs nothing and
the difference between float32 and float64 on an ill-conditioned covariance is
the difference between a clean grade and a magenta screen.
"""

from __future__ import annotations

import torch

__all__ = [
    "quantiles",
    "trimmed_log_mean",
    "weighted_mean",
    "weighted_mean_cov",
    "sym_sqrt",
    "sym_inv_sqrt",
    "mkl_transform",
]

# Below this the covariance is treated as rank-deficient in that direction and
# the transport falls back to "leave this axis alone".
_EPS = 1e-12

# Bins for the histogram quantile estimator. Resolution is (max - min) / bins,
# i.e. ~5e-4 of the data range -- roughly an eighth of an 8-bit code value, and
# far finer than the smoothing applied to the resulting curves.
_QUANTILE_BINS = 2048


def quantiles(x: torch.Tensor, q: torch.Tensor, bins: int = _QUANTILE_BINS) -> torch.Tensor:
    """Quantiles of a 1-D tensor, estimated from a histogram CDF.

    Deliberately *not* a sort. Distribution transfer needs a few dozen
    quantiles from a few tens of thousands of samples, thirty-odd times per
    grade; ``torch.sort`` on CPU costs about 0.23 us per element, which turns
    that into seconds. A histogram plus a cumulative sum is O(N) with a tiny
    constant -- two orders of magnitude quicker here -- and the resulting
    quantiles are accurate to one bin width, which is finer than the curves
    built from them are ever smoothed to.

    It is also exactly reproducible: bin counts are integers, so the result
    does not depend on reduction order, thread count or device.
    """
    x = x.reshape(-1)
    if x.numel() == 0:
        return torch.zeros_like(q)

    xf = x.to(torch.float32)
    lo = float(xf.min())
    hi = float(xf.max())
    if not (hi > lo):
        return torch.full_like(q, lo)

    hist = torch.histc(xf, bins=bins, min=lo, max=hi)
    cdf = torch.cumsum(hist, dim=0)
    total = cdf[-1].clamp_min(1.0)
    cdf = torch.cat([torch.zeros(1, dtype=cdf.dtype, device=cdf.device), cdf / total])

    edges = torch.linspace(lo, hi, bins + 1, dtype=cdf.dtype, device=cdf.device)
    qq = q.reshape(-1).clamp(0.0, 1.0).to(cdf.dtype).contiguous()
    idx = torch.searchsorted(cdf.contiguous(), qq).clamp(1, bins)

    c0, c1 = cdf[idx - 1], cdf[idx]
    e0, e1 = edges[idx - 1], edges[idx]
    frac = ((qq - c0) / (c1 - c0).clamp_min(_EPS)).clamp(0.0, 1.0)
    return (e0 + frac * (e1 - e0)).reshape(q.shape).to(q.dtype)


def trimmed_log_mean(y: torch.Tensor, lo: float = 0.02, hi: float = 0.98, floor: float = 1e-4) -> torch.Tensor:
    """Mean of ``log(y)`` over the central quantile range.

    This is the exposure statistic. The log makes it a *geometric* mean, which
    is the right average for a quantity that is going to be corrected by a
    multiplicative gain; the trim stops a specular highlight or a letterbox bar
    from dictating the whole frame's exposure.
    """
    y = y.reshape(-1)
    if y.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=y.device)
    bounds = quantiles(y, torch.tensor([lo, hi], device=y.device))
    low, high = bounds[0], bounds[1]
    if not torch.isfinite(low) or not torch.isfinite(high) or high <= low:
        sel = y
    else:
        mask = (y >= low) & (y <= high)
        sel = y[mask] if bool(mask.any()) else y
    return sel.clamp_min(floor).log().mean()


def weighted_mean(pts: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean of ``[N, D]`` points plus Kish's effective sample size.

    Split out from :func:`weighted_mean_cov` because the band-residual stage
    needs means only, and building three-by-three covariances it then throws
    away was a measurable share of the fit.
    """
    w = weights.reshape(-1).clamp_min(0.0).to(pts.dtype)
    total = w.sum()
    if float(total) <= _EPS:
        return pts.new_zeros(pts.shape[-1]), pts.new_zeros(())
    n_eff = total.pow(2) / w.pow(2).sum().clamp_min(_EPS)
    mean = (pts * (w / total)[:, None]).sum(dim=0)
    return mean, n_eff


def weighted_mean_cov(
    pts: torch.Tensor,
    weights: torch.Tensor | None = None,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted mean, covariance and effective sample count of ``[N, D]`` points.

    ``ridge`` is added to the diagonal as a fraction of the mean variance, not
    as an absolute value, so it scales correctly whether the points are Oklab
    (range ~1) or something else entirely.
    """
    pts = pts.to(torch.float64)
    n, d = pts.shape
    if n == 0:
        return (
            torch.zeros(d, dtype=torch.float64, device=pts.device),
            torch.eye(d, dtype=torch.float64, device=pts.device) * _EPS,
            torch.zeros((), dtype=torch.float64, device=pts.device),
        )
    if weights is None:
        w = torch.full((n,), 1.0 / n, dtype=torch.float64, device=pts.device)
        n_eff = torch.tensor(float(n), dtype=torch.float64, device=pts.device)
    else:
        w = weights.to(torch.float64).reshape(-1).clamp_min(0.0)
        total = w.sum()
        if float(total) <= _EPS:
            w = torch.full((n,), 1.0 / n, dtype=torch.float64, device=pts.device)
            n_eff = torch.tensor(float(n), dtype=torch.float64, device=pts.device)
        else:
            n_eff = total.pow(2) / w.pow(2).sum().clamp_min(_EPS)
            w = w / total
    mean = (pts * w[:, None]).sum(dim=0)
    centred = pts - mean
    cov = torch.einsum("n,ni,nj->ij", w, centred, centred)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    scale = torch.diagonal(cov).mean().clamp_min(_EPS)
    cov = cov + torch.eye(d, dtype=torch.float64, device=pts.device) * (ridge * scale + _EPS)
    return mean, cov, n_eff


def _eigh_psd(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = 0.5 * (m + m.transpose(-1, -2))
    try:
        evals, evecs = torch.linalg.eigh(m)
    except Exception:  # pragma: no cover - LAPACK failure on pathological input
        d = m.shape[-1]
        eye = torch.eye(d, dtype=m.dtype, device=m.device)
        return torch.ones(d, dtype=m.dtype, device=m.device) * _EPS, eye
    return evals.clamp_min(0.0), evecs


def sym_sqrt(m: torch.Tensor) -> torch.Tensor:
    """Principal square root of a symmetric PSD matrix."""
    evals, evecs = _eigh_psd(m)
    return evecs @ torch.diag(evals.clamp_min(0.0).sqrt()) @ evecs.transpose(-1, -2)


def sym_inv_sqrt(m: torch.Tensor, floor_ratio: float = 1e-8) -> torch.Tensor:
    """Inverse square root of a symmetric PSD matrix, with eigenvalue flooring.

    Eigenvalues are floored relative to the largest one. A greyscale plate has
    two near-zero eigenvalues in Oklab; without the floor their inverse square
    roots explode and the transport hurls the chroma axes to infinity. With it,
    those directions simply become very stiff, which is the behaviour you want:
    "there is no information along this axis, so do not move much along it".
    """
    evals, evecs = _eigh_psd(m)
    floor = evals.max().clamp_min(_EPS) * floor_ratio
    inv_sqrt = evals.clamp_min(floor).rsqrt()
    return evecs @ torch.diag(inv_sqrt) @ evecs.transpose(-1, -2)


def mkl_transform(
    mean_src: torch.Tensor,
    cov_src: torch.Tensor,
    mean_dst: torch.Tensor,
    cov_dst: torch.Tensor,
    max_gain: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form linear Monge-Kantorovich map between two Gaussians.

    Returns ``(A, b)`` such that ``x -> (x - mean_src) @ A.T + mean_dst`` is the
    optimal-transport map under a quadratic cost when both clouds are treated
    as Gaussian, i.e.

        A = S^-1/2 (S^1/2 T S^1/2)^1/2 S^-1/2

    with ``S = cov_src`` and ``T = cov_dst``. This is Pitie & Kokaram, "The
    linear Monge-Kantorovitch linear colour mapping for example-based colour
    transfer", CVMP 2007.

    ``max_gain`` caps the spectral radius of ``A``. An unbounded ``A`` is the
    single most common source of blown-out example-based grades: when the
    target has almost no variance along some direction and the reference has a
    lot, the "optimal" map is a huge stretch that turns sensor noise into
    colour blotches. Capping the singular values keeps the map optimal in the
    directions that carry information and merely firm in the ones that do not.
    """
    s_sqrt = sym_sqrt(cov_src)
    s_inv_sqrt = sym_inv_sqrt(cov_src)
    middle = sym_sqrt(s_sqrt @ cov_dst @ s_sqrt)
    a = s_inv_sqrt @ middle @ s_inv_sqrt
    a = 0.5 * (a + a.transpose(-1, -2))

    evals, evecs = _eigh_psd(a)
    evals = evals.clamp(1.0 / max_gain, max_gain)
    a = evecs @ torch.diag(evals) @ evecs.transpose(-1, -2)

    if not bool(torch.isfinite(a).all()):
        a = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    b = mean_dst - mean_src @ a.transpose(-1, -2)
    if not bool(torch.isfinite(b).all()):
        b = torch.zeros_like(b)
    return a, b
