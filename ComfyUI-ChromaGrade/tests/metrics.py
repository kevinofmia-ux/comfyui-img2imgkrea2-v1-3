"""Measurement helpers shared by the tests and the evaluation harness.

Two families, because a colour-transfer method has to be judged on two axes at
once and optimising either alone produces something useless:

* **Colour fidelity** -- how close the output's colour *distribution* is to the
  reference's. Sliced Wasserstein in Oklab, which is the natural metric for a
  method that is itself optimal transport, and unlike a per-channel histogram
  distance it sees palette geometry (a red/cyan image and a magenta/green one
  can have identical marginals).
* **Structure preservation** -- how much of the target's geometry and texture
  survived. SSIM on luminance plus a gradient-correlation term, both computed
  against the *target*, since a grade is supposed to leave structure alone.
"""

from __future__ import annotations

import math

import torch

from chromagrade.colorspace import srgb_to_oklab
from chromagrade.transport import fibonacci_rotations

__all__ = ["sliced_wasserstein", "ssim", "gradient_correlation", "mean_delta_ok", "banding_score"]


def _flatten_oklab(image: torch.Tensor) -> torch.Tensor:
    return srgb_to_oklab(image.reshape(-1, image.shape[-1])[:, :3])


def sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, n_slices: int = 96, n_quantiles: int = 128) -> float:
    """Sliced 1-Wasserstein distance between two images' Oklab colour clouds.

    Directions come from the same deterministic Fibonacci construction the
    transport uses, so the number is exactly reproducible. Lower is a closer
    match to the reference's grade.
    """
    pa, pb = _flatten_oklab(a), _flatten_oklab(b)
    dirs = fibonacci_rotations(n_slices, pa.device, pa.dtype)[:, 0, :]  # [n, 3]
    q = torch.linspace(0.001, 0.999, n_quantiles, device=pa.device)

    proj_a = pa @ dirs.transpose(0, 1)  # [Na, n]
    proj_b = pb @ dirs.transpose(0, 1)
    qa = torch.quantile(proj_a, q, dim=0)
    qb = torch.quantile(proj_b, q, dim=0)
    return float((qa - qb).abs().mean())


def _gaussian_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    return k / k.sum()


def _blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of ``[B, 1, H, W]`` with reflect padding."""
    r = (kernel.numel() - 1) // 2
    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)
    x = torch.nn.functional.pad(x, (r, r, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, kx)
    x = torch.nn.functional.pad(x, (0, 0, r, r), mode="reflect")
    return torch.nn.functional.conv2d(x, ky)


def _luma(image: torch.Tensor) -> torch.Tensor:
    lab = srgb_to_oklab(image[..., :3])
    return lab[..., 0:1].permute(0, 3, 1, 2)


def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> float:
    """Standard SSIM on Oklab lightness. 1.0 means structurally identical."""
    xa, xb = _luma(a), _luma(b)
    kernel = _gaussian_kernel(window, sigma, xa.device, xa.dtype)
    c1, c2 = 0.01**2, 0.03**2

    mu_a, mu_b = _blur(xa, kernel), _blur(xb, kernel)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = _blur(xa * xa, kernel) - mu_aa
    var_b = _blur(xb * xb, kernel) - mu_bb
    cov = _blur(xa * xb, kernel) - mu_ab

    num = (2 * mu_ab + c1) * (2 * cov + c2)
    den = (mu_aa + mu_bb + c1) * (var_a + var_b + c2)
    return float((num / den.clamp_min(1e-12)).mean())


def gradient_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of the two images' luminance gradient fields.

    SSIM can be fooled by a global tone change; the gradient field cannot. If a
    method blurred, ringed or re-textured the image, this is what drops.
    """
    xa, xb = _luma(a), _luma(b)
    ga = torch.cat([xa[:, :, 1:, :] - xa[:, :, :-1, :], torch.zeros_like(xa[:, :, :1, :])], dim=2)
    gb = torch.cat([xb[:, :, 1:, :] - xb[:, :, :-1, :], torch.zeros_like(xb[:, :, :1, :])], dim=2)
    ga2 = torch.cat([xa[:, :, :, 1:] - xa[:, :, :, :-1], torch.zeros_like(xa[:, :, :, :1])], dim=3)
    gb2 = torch.cat([xb[:, :, :, 1:] - xb[:, :, :, :-1], torch.zeros_like(xb[:, :, :, :1])], dim=3)

    va = torch.cat([ga.reshape(-1), ga2.reshape(-1)])
    vb = torch.cat([gb.reshape(-1), gb2.reshape(-1)])
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = (va.norm() * vb.norm()).clamp_min(1e-12)
    return float((va * vb).sum() / denom)


def mean_delta_ok(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Euclidean distance in Oklab -- a rough perceptual difference."""
    pa, pb = _flatten_oklab(a), _flatten_oklab(b)
    return float((pa - pb).norm(dim=-1).mean())


def banding_score(image: torch.Tensor) -> float:
    """Detects contouring on a smooth ramp.

    Along a monotone gradient the second derivative of lightness should be
    near-zero everywhere. Banding is exactly a train of second-derivative
    spikes, so the 99.9th percentile of |d2L/dx2| separates a clean ramp from a
    posterised one by orders of magnitude. Lower is better.
    """
    x = _luma(image)
    d2 = x[:, :, :, 2:] - 2 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
    v = d2.abs().reshape(-1)
    k = max(1, int(math.ceil(v.numel() * 0.001)))
    return float(torch.topk(v, k).values.min())
