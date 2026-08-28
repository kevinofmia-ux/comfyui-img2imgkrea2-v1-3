"""3D LUT construction, regularisation and tetrahedral application.

This module is the reason the node is safe at full resolution.

Every analytical stage of the grade is a *function of colour alone*. So instead
of evaluating that function per pixel, it is evaluated once on a lattice over
the sRGB cube and the image is then pushed through the resulting 3D LUT. Three
things fall out of that:

* **Structure is untouchable.** A 3D LUT is a pure per-pixel colour map. It has
  no spatial extent, so it cannot produce halos, cannot soften an edge and
  cannot lose texture. Whatever geometry, grain and micro-detail went in comes
  out, remapped.
* **Cost is bounded.** The expensive analysis runs on ~35k lattice nodes
  regardless of whether the image is 512px or 8K. Full resolution only pays for
  an interpolation.
* **Banding can be regularised away.** A lattice can be smoothed; a per-pixel
  closed-form expression cannot. One light pass with a linear-extrapolating
  boundary removes the quantile staircase that distribution matching leaves
  behind, without moving the black or white point.

Interpolation is tetrahedral rather than trilinear. Tetrahedral reproduces the
LUT exactly along the neutral diagonal, which is precisely where trilinear's
error peaks -- and a colour cast on greys is the single most visible defect a
LUT can have.
"""

from __future__ import annotations

import torch

__all__ = ["identity_lattice", "lattice_coordinates", "smooth_lut", "blend_toward_identity", "apply_lut"]

# Pixels per chunk when applying. Eight gathers plus working space at fp32 puts
# this around 150 MB of transient allocation, which is comfortable on any card
# that is already running a diffusion model.
_CHUNK_PIXELS = 1 << 20


def lattice_coordinates(size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[size**3, 3]`` sRGB-encoded lattice input coordinates, R-major."""
    axis = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    r, g, b = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([r, g, b], dim=-1).reshape(-1, 3)


def identity_lattice(size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[size, size, size, 3]`` identity LUT."""
    return lattice_coordinates(size, device, dtype).reshape(size, size, size, 3)


def _smooth_axis(t: torch.Tensor, axis: int) -> torch.Tensor:
    """One [1/8, 3/4, 1/8] pass with linearly extrapolated boundaries.

    Linear extrapolation (rather than replication) makes the filter exact on
    affine data, so the endpoints -- black and white -- do not creep inward.
    Replication would pull pure white down by a fraction of a percent per axis,
    which reads as a washed-out white point.
    """
    n = t.shape[axis]
    if n < 3:
        return t
    first = t.narrow(axis, 0, 1)
    second = t.narrow(axis, 1, 1)
    last = t.narrow(axis, n - 1, 1)
    penult = t.narrow(axis, n - 2, 1)
    padded = torch.cat([2.0 * first - second, t, 2.0 * last - penult], dim=axis)
    a = padded.narrow(axis, 0, n)
    b = padded.narrow(axis, 1, n)
    c = padded.narrow(axis, 2, n)
    return 0.125 * a + 0.75 * b + 0.125 * c


def smooth_lut(lut: torch.Tensor, weight: float = 1.0, passes: int = 1) -> torch.Tensor:
    """Separable 3-tap smoothing of the lattice, blended by ``weight``."""
    if weight <= 0.0 or passes <= 0:
        return lut
    out = lut
    for _ in range(passes):
        s = out
        for axis in (0, 1, 2):
            s = _smooth_axis(s, axis)
        out = torch.lerp(out, s, min(1.0, float(weight)))
    return out


def blend_toward_identity(lut: torch.Tensor, strength: float) -> torch.Tensor:
    """Interpolate the LUT with identity. ``strength=0`` is a perfect no-op."""
    strength = float(min(max(strength, 0.0), 1.0))
    if strength >= 1.0:
        return lut
    ident = identity_lattice(lut.shape[0], lut.device, lut.dtype)
    if strength <= 0.0:
        return ident
    return torch.lerp(ident, lut, strength)


def _tetrahedral(x: torch.Tensor, flat_lut: torch.Tensor, size: int) -> torch.Tensor:
    f = x.clamp(0.0, 1.0) * (size - 1)
    i0 = f.floor().clamp(0, size - 2).to(torch.long)
    d = (f - i0.to(f.dtype)).clamp(0.0, 1.0)

    ir, ig, ib = i0[:, 0], i0[:, 1], i0[:, 2]
    base = (ir * size + ig) * size + ib
    s2 = size * size

    def node(dr: int, dg: int, db: int) -> torch.Tensor:
        return flat_lut[base + dr * s2 + dg * size + db]

    v000 = node(0, 0, 0)
    v001 = node(0, 0, 1)
    v010 = node(0, 1, 0)
    v011 = node(0, 1, 1)
    v100 = node(1, 0, 0)
    v101 = node(1, 0, 1)
    v110 = node(1, 1, 0)
    v111 = node(1, 1, 1)

    dr = d[:, 0:1]
    dg = d[:, 1:2]
    db = d[:, 2:3]

    r_gt_g = dr > dg
    g_gt_b = dg > db
    r_gt_b = dr > db
    b_gt_g = db > dg
    b_gt_r = db > dr

    # The six tetrahedra of the unit cube, written as "which edge does each of
    # the three barycentric steps walk along". Selecting the three coefficient
    # vectors first (rather than evaluating six full expressions) keeps this to
    # three temporaries.
    c1 = r_gt_g & g_gt_b  # dr > dg > db
    c2 = r_gt_g & ~g_gt_b & r_gt_b  # dr > db >= dg
    c3 = r_gt_g & ~g_gt_b & ~r_gt_b  # db >= dr > dg
    c4 = ~r_gt_g & b_gt_g  # db > dg >= dr
    c5 = ~r_gt_g & ~b_gt_g & b_gt_r  # dg >= db > dr

    def pick(o1, o2, o3, o4, o5, o6):
        out = o6
        out = torch.where(c5, o5, out)
        out = torch.where(c4, o4, out)
        out = torch.where(c3, o3, out)
        out = torch.where(c2, o2, out)
        out = torch.where(c1, o1, out)
        return out

    cr = pick(v100 - v000, v100 - v000, v101 - v001, v111 - v011, v111 - v011, v110 - v010)
    cg = pick(v110 - v100, v111 - v101, v111 - v101, v011 - v001, v010 - v000, v010 - v000)
    cb = pick(v111 - v110, v101 - v100, v001 - v000, v001 - v000, v011 - v010, v111 - v110)

    return v000 + cr * dr + cg * dg + cb * db


def apply_lut(image: torch.Tensor, lut: torch.Tensor, chunk_pixels: int = _CHUNK_PIXELS) -> torch.Tensor:
    """Apply a ``[N, N, N, 3]`` LUT to an ``[..., 3]`` image in ``[0, 1]``."""
    if lut.ndim != 4 or lut.shape[-1] != 3 or lut.shape[0] != lut.shape[1] or lut.shape[0] != lut.shape[2]:
        raise ValueError(f"expected a cubic [N, N, N, 3] LUT, got {tuple(lut.shape)}")
    size = int(lut.shape[0])
    if size < 2:
        raise ValueError("LUT size must be at least 2")

    shape = image.shape
    x = image.reshape(-1, 3)
    lut = lut.to(device=x.device, dtype=x.dtype)
    flat_lut = lut.reshape(-1, 3).contiguous()

    total = x.shape[0]
    if total <= chunk_pixels:
        return _tetrahedral(x, flat_lut, size).reshape(shape)

    out = torch.empty_like(x)
    for start in range(0, total, chunk_pixels):
        stop = min(start + chunk_pixels, total)
        out[start:stop] = _tetrahedral(x[start:stop], flat_lut, size)
    return out.reshape(shape)
