"""Monotone 1-D transport maps and deterministic N-D distribution transfer.

The two building blocks here are:

``MonotoneMap``
    A slope-limited, monotone-cubic (PCHIP) interpolant fitted through matched
    quantile pairs. This is the workhorse: it is what turns "the reference's
    shadows sit at 0.18 and the target's at 0.05" into an actual curve. Being
    monotone means it can never invert local contrast; being slope-limited
    means it can never crush a decade of shadow detail into one code value;
    being cubic means it has no first-derivative discontinuities, which is what
    stops quantile matching from banding.

``IDTChain``
    Pitie, Kokaram & Dahyot's Iterative Distribution Transfer -- repeatedly
    rotate the joint colour cloud, match its three marginals independently, and
    rotate back. Given enough rotations this converges to a full N-D
    distribution match, which is what captures palette *relationships* rather
    than just per-channel statistics.

    The published algorithm draws random rotations. That would make this node
    non-deterministic, so the rotations here come from a fixed Fibonacci-sphere
    construction instead: evenly spread by design rather than by expectation,
    identical on every run, every machine and every device.

References:
  * F. Pitie, A. Kokaram, R. Dahyot, "Automated colour grading using colour
    distribution transfer", CVIU 107(1):123-137, 2007.
  * F. Pitie, A. Kokaram, "The linear Monge-Kantorovitch linear colour mapping
    for example-based colour transfer", CVMP 2007.
"""

from __future__ import annotations

import math

import torch

from .stats import quantiles

__all__ = ["MonotoneMap", "build_quantile_map", "fibonacci_rotations", "IDTChain"]


class MonotoneMap:
    """A strictly increasing scalar map defined by knots and PCHIP tangents.

    Outside the knot range the map continues linearly with the end tangents, so
    it stays defined -- and monotone -- for colours that never appeared in the
    analysis sample. That matters because the map is later evaluated on a full
    LUT lattice, most of which is nowhere near the image's actual gamut.
    """

    __slots__ = ("x", "y", "d", "_coef", "_ends", "_cache")

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        x = x.reshape(-1).to(torch.float32)
        y = y.reshape(-1).to(torch.float32)
        if x.numel() < 2:
            raise ValueError("MonotoneMap needs at least two knots")
        self.x = x
        self.y = y
        self.d = _pchip_tangents(x, y)
        # Per-segment cubic coefficients in the local variable u = t - x[i].
        # Precomputing them turns evaluation into three gathers and a Horner
        # step; the textbook Hermite-basis form needs eight gathers and about
        # fifteen elementwise ops, and this map is evaluated tens of millions of
        # times per grade.
        h = (x[1:] - x[:-1]).clamp_min(1e-12)
        delta = (y[1:] - y[:-1]) / h
        d0, d1 = self.d[:-1], self.d[1:]
        c2 = (3.0 * delta - 2.0 * d0 - d1) / h
        c3 = (d0 + d1 - 2.0 * delta) / (h * h)
        self._coef = torch.stack([y[:-1], d0, c2, c3], dim=0)
        # (y_first, slope_first, y_last, slope_last) for the linear tails.
        self._ends = torch.stack([y[0], self.d[0], y[-1], self.d[-1]])
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def device(self) -> torch.device:
        return self.x.device

    def _on(self, device: torch.device, dtype: torch.dtype):
        key = (device, dtype)
        got = self._cache.get(key)
        if got is None:
            got = (
                self.x.to(device=device, dtype=dtype),
                self._coef.to(device=device, dtype=dtype),
                self._ends.to(device=device, dtype=dtype),
            )
            self._cache[key] = got
        return got

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        x, coef, ends = self._on(t.device, t.dtype)
        n = x.numel()

        shape = t.shape
        flat = t.reshape(-1).contiguous()
        idx = (torch.searchsorted(x, flat, right=True) - 1).clamp(0, n - 2)

        u = flat - x[idx]
        c0, c1, c2, c3 = coef[0][idx], coef[1][idx], coef[2][idx], coef[3][idx]
        val = c0 + u * (c1 + u * (c2 + u * c3))

        # Past the end knots the cubic would be free to turn around, so both
        # tails are replaced by straight lines with the end tangents. That keeps
        # the map monotone -- and therefore contrast-preserving -- for the large
        # part of the LUT lattice that lies outside the image's own gamut.
        val = torch.where(flat < x[0], ends[0] + (flat - x[0]) * ends[1], val)
        val = torch.where(flat > x[-1], ends[2] + (flat - x[-1]) * ends[3], val)
        return val.reshape(shape)


def _pchip_tangents(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fritsch-Carlson tangents: the standard monotone-cubic construction."""
    h = (x[1:] - x[:-1]).clamp_min(1e-12)
    delta = (y[1:] - y[:-1]) / h
    n = x.numel()
    d = torch.zeros_like(x)
    if n == 2:
        return delta.repeat(2)

    d0, d1 = delta[:-1], delta[1:]
    h0, h1 = h[:-1], h[1:]
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    same_sign = (d0 * d1) > 0
    denom = torch.where(same_sign, w1 / d0.masked_fill(~same_sign, 1.0) + w2 / d1.masked_fill(~same_sign, 1.0), torch.ones_like(d0))
    interior = torch.where(same_sign, (w1 + w2) / denom, torch.zeros_like(d0))
    d[1:-1] = interior

    # One-sided ends, clamped so an end tangent cannot overshoot its segment by
    # more than 3x (the standard non-overshoot condition).
    d[0] = _end_tangent(h[0], h[1] if n > 2 else h[0], delta[0], delta[1] if n > 2 else delta[0])
    d[-1] = _end_tangent(h[-1], h[-2] if n > 2 else h[-1], delta[-1], delta[-2] if n > 2 else delta[-1])
    return d


def _end_tangent(h0: torch.Tensor, h1: torch.Tensor, d0: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
    """One-sided end tangent, kept inside the non-overshoot band ``[0, 3*d0]``.

    Clamping into the band (rather than only zeroing sign flips) also keeps the
    linear extrapolation past the last knot pointing the same way the data
    does, which is what the LUT lattice relies on for colours outside the
    image's own gamut.
    """
    t = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1).clamp_min(1e-12)
    lo = torch.minimum(torch.zeros_like(d0), 3.0 * d0)
    hi = torch.maximum(torch.zeros_like(d0), 3.0 * d0)
    return t.clamp(lo, hi)


def build_quantile_map(
    src: torch.Tensor,
    dst: torch.Tensor,
    n_knots: int = 33,
    q_lo: float = 0.002,
    q_hi: float = 0.998,
    slope_range: tuple[float, float] = (0.2, 5.0),
    smooth: int = 1,
    identity_blend: float = 0.0,
) -> MonotoneMap:
    """Fit a monotone map carrying ``src``'s distribution onto ``dst``'s.

    The knots are matched quantile pairs. Three guards then turn a raw quantile
    match -- which is fragile -- into something safe to ship:

    ``q_lo``/``q_hi``
        Ignore the extreme tails. The 0.0 and 1.0 quantiles are single pixels
        and are routinely a dead pixel or a specular clip; anchoring a curve to
        them warps the whole thing.
    ``slope_range``
        Clamp per-segment slope, then re-integrate. A slope of 0.05 is
        posterisation and a slope of 30 is a blown highlight; neither is a
        grade. The curve is re-anchored on the median afterwards so clamping
        changes the *shape* without shifting the overall level.
    ``smooth``
        A couple of passes of 3-tap smoothing on the knot values, to take out
        the sampling jitter that quantile estimates always carry.
    """
    device = src.device
    q = torch.linspace(q_lo, q_hi, n_knots, device=device)
    xs = quantiles(src, q).to(torch.float32)
    ys = quantiles(dst, q).to(torch.float32)

    # Strictly increasing x. A degenerate source (constant image) collapses all
    # knots onto one value; spreading them by a fixed epsilon keeps the map
    # well-defined and it degenerates gracefully to a near-constant output.
    xs = torch.cummax(xs, dim=0).values
    span = float(xs[-1] - xs[0])
    min_step = max(span, 1e-3) * 1e-4
    steps = torch.arange(n_knots, device=device, dtype=xs.dtype) * min_step
    xs = xs + steps

    # Smooth the *correction* (ys - xs), never the knot values themselves. The
    # quantile positions carry the shape of the data and smoothing them bends
    # the curve away from identity even when source and destination are the
    # same distribution; the correction is the part that carries the sampling
    # noise. With this, matching an image against itself is an exact no-op
    # instead of a ~4/255 lightness wobble.
    delta = ys - xs
    for _ in range(max(0, smooth)):
        delta = _smooth3(delta)
    ys = torch.cummax(xs + delta, dim=0).values

    h = (xs[1:] - xs[:-1]).clamp_min(1e-12)
    slopes = ((ys[1:] - ys[:-1]) / h).clamp(slope_range[0], slope_range[1])
    rebuilt = torch.cat([torch.zeros(1, device=device, dtype=ys.dtype), torch.cumsum(slopes * h, dim=0)])

    # Re-anchor on the median pair so the slope clamp cannot introduce a level
    # shift of its own.
    mid = n_knots // 2
    rebuilt = rebuilt - rebuilt[mid] + ys[mid]

    if identity_blend > 0.0:
        rebuilt = torch.lerp(rebuilt, xs, torch.tensor(float(identity_blend), device=device, dtype=ys.dtype))

    return MonotoneMap(xs, rebuilt)


def _smooth3(v: torch.Tensor) -> torch.Tensor:
    padded = torch.cat([v[:1], v, v[-1:]])
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def fibonacci_rotations(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[n, 3, 3]`` orthonormal bases whose first axes tile the sphere evenly.

    Deterministic by construction -- no RNG, no seed, no device-dependent
    stream. The first axis of rotation ``k`` is the ``k``-th point of a
    Fibonacci spiral on the hemisphere; the remaining two are completed by
    Gram-Schmidt against a fixed helper vector, with a fallback helper for the
    degenerate case where the primary axis is parallel to it.
    """
    n = max(1, int(n))
    idx = torch.arange(n, device=device, dtype=torch.float64)
    # Hemisphere is enough: a direction and its negation give the same marginal.
    z = (idx + 0.5) / n
    r = (1.0 - z * z).clamp_min(0.0).sqrt()
    golden = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden * idx
    axis = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    helper = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device).expand_as(axis).clone()
    degenerate = axis[:, 2].abs() > 0.9
    helper[degenerate] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)

    u = helper - axis * (helper * axis).sum(dim=-1, keepdim=True)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = torch.cross(axis, u, dim=-1)
    return torch.stack([axis, u, v], dim=1).to(dtype)


class IDTChain:
    """A replayable Iterative Distribution Transfer map.

    The chain stores ``(rotation, three 1-D maps)`` per iteration, so it is a
    genuine function of colour rather than a per-pixel displacement field. That
    is what lets the whole thing be evaluated later on a LUT lattice: the
    analysis runs on a downsampled cloud, but the *transform* it produces can be
    applied to any colour at all.
    """

    def __init__(self) -> None:
        self.steps: list[tuple[torch.Tensor, list[MonotoneMap]]] = []

    @classmethod
    def fit(
        cls,
        src: torch.Tensor,
        dst: torch.Tensor,
        iterations: int = 10,
        n_knots: int = 33,
        slope_range: tuple[float, float] = (0.25, 4.0),
        relaxation: float = 1.0,
    ) -> "IDTChain":
        chain = cls()
        if src.shape[0] < 16 or dst.shape[0] < 16:
            return chain
        rotations = fibonacci_rotations(iterations, src.device, src.dtype)
        current = src
        for k in range(iterations):
            rot = rotations[k]
            proj_s = current @ rot.transpose(0, 1)
            proj_d = dst @ rot.transpose(0, 1)
            maps = [
                build_quantile_map(
                    proj_s[:, c],
                    proj_d[:, c],
                    n_knots=n_knots,
                    slope_range=slope_range,
                    identity_blend=1.0 - relaxation,
                )
                for c in range(3)
            ]
            moved = torch.stack([maps[c](proj_s[:, c]) for c in range(3)], dim=-1)
            current = moved @ rot
            chain.steps.append((rot, maps))
        return chain

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        out = pts
        for rot, maps in self.steps:
            rot = rot.to(device=out.device, dtype=out.dtype)
            proj = out @ rot.transpose(0, 1)
            moved = torch.stack([maps[c](proj[:, c]) for c in range(3)], dim=-1)
            out = moved @ rot
        return out

    def __len__(self) -> int:
        return len(self.steps)
