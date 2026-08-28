"""Reinhard et al. colour transfer -- the comparison baseline.

E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley, "Color Transfer between
Images", IEEE Computer Graphics and Applications 21(5), 2001.

The canonical simple method: decorrelate into Ruderman's l-alpha-beta space,
match per-channel mean and standard deviation, convert back. It is the thing
"just do histogram matching" usually turns into, and it is the right yardstick
-- if the pipeline in this package cannot beat it on both colour fidelity *and*
structure preservation, the extra machinery is not paying for itself.

Included here rather than in the package because it exists only to be measured
against.
"""

from __future__ import annotations

import math

import torch

__all__ = ["reinhard_transfer"]

_RGB_TO_LMS = torch.tensor(
    [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ]
)

_LOGLMS_TO_LAB = torch.tensor(
    [
        [1.0 / math.sqrt(3.0), 0.0, 0.0],
        [0.0, 1.0 / math.sqrt(6.0), 0.0],
        [0.0, 0.0, 1.0 / math.sqrt(2.0)],
    ]
) @ torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -2.0],
        [1.0, -1.0, 0.0],
    ]
)


def _mm(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...c,dc->...d", x, m.to(device=x.device, dtype=x.dtype))


def _to_lab(rgb: torch.Tensor) -> torch.Tensor:
    lms = _mm(rgb.clamp_min(1.0 / 255.0), _RGB_TO_LMS).clamp_min(1e-6)
    return _mm(torch.log10(lms), _LOGLMS_TO_LAB)


def _from_lab(lab: torch.Tensor) -> torch.Tensor:
    inv_lab = torch.linalg.inv(_LOGLMS_TO_LAB.to(torch.float64)).to(lab.dtype)
    inv_lms = torch.linalg.inv(_RGB_TO_LMS.to(torch.float64)).to(lab.dtype)
    log_lms = _mm(lab, inv_lab)
    return _mm(torch.pow(10.0, log_lms.clamp(-8.0, 8.0)), inv_lms)


def reinhard_transfer(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """``[B, H, W, 3]`` sRGB in, graded ``[B, H, W, 3]`` sRGB out."""
    out = torch.empty_like(target)
    ref_lab = _to_lab(reference[..., :3].reshape(-1, 3))
    ref_mean = ref_lab.mean(dim=0)
    ref_std = ref_lab.std(dim=0).clamp_min(1e-6)
    for i in range(target.shape[0]):
        lab = _to_lab(target[i, ..., :3])
        mean = lab.reshape(-1, 3).mean(dim=0)
        std = lab.reshape(-1, 3).std(dim=0).clamp_min(1e-6)
        shifted = (lab - mean) * (ref_std / std) + ref_mean
        out[i] = _from_lab(shifted).clamp(0.0, 1.0)
    return out
