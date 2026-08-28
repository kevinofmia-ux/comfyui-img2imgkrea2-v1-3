"""Measure ChromaGrade against the Reinhard baseline and print a markdown table.

    python tools/evaluate.py

Columns:

``SWD``
    Sliced 1-Wasserstein distance in Oklab between the *output* and the
    *reference* colour clouds. Lower means a closer match to the grade.
``SSIM``
    Structural similarity between the output and the *target* lightness. Higher
    means less of the original image was disturbed.
``rails``
    Fraction of pixels with at least one channel pinned to 0 or 1 -- the
    clipping that destroys highlight and shadow separation.

The two quality columns move against each other by construction, so read them
as a pair. ``tests/test_quality.py`` turns the same measurements into gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromagrade import MatchParams, color_match  # noqa: E402
from tests import fixtures  # noqa: E402
from tests.baseline import reinhard_transfer  # noqa: E402
from tests.metrics import gradient_correlation, sliced_wasserstein, ssim  # noqa: E402

PAIRS = [
    ("gradient", "sunset"),
    ("gradient", "teal"),
    ("portrait", "sepia"),
    ("portrait", "teal"),
    ("neon", "bleach"),
    ("greyscale", "sepia"),
    ("fog", "sunset"),
    ("high_key", "teal"),
    ("low_key", "bleach"),
]


def _rails(x: torch.Tensor) -> float:
    return float(((x <= 1e-6) | (x >= 1.0 - 1e-6)).any(dim=-1).float().mean())


def main() -> int:
    header = (
        "| pair | SWD ours | SWD fast | SWD Reinhard | SSIM ours | SSIM Reinhard "
        "| grad corr ours | rails ours | rails Reinhard |"
    )
    print(header)
    print("|" + "---|" * 8)

    totals = {"ours": 0.0, "fast": 0.0, "reinhard": 0.0}
    for scene, ref in PAIRS:
        target = fixtures.SCENES[scene]()
        reference = fixtures.REFERENCES[ref]()

        ours = color_match(target, reference, MatchParams())
        fast = color_match(target, reference, MatchParams(mode="fast"))
        base = reinhard_transfer(target, reference)

        s_ours = sliced_wasserstein(ours, reference)
        s_fast = sliced_wasserstein(fast, reference)
        s_base = sliced_wasserstein(base, reference)
        totals["ours"] += s_ours
        totals["fast"] += s_fast
        totals["reinhard"] += s_base

        print(
            f"| {scene} -> {ref} | {s_ours:.4f} | {s_fast:.4f} | {s_base:.4f} "
            f"| {ssim(ours, target):.3f} | {ssim(base, target):.3f} "
            f"| {gradient_correlation(ours, target):.3f} "
            f"| {_rails(ours):.1%} | {_rails(base):.1%} |"
        )

    n = len(PAIRS)
    print(
        f"| **mean** | **{totals['ours'] / n:.4f}** | **{totals['fast'] / n:.4f}** "
        f"| **{totals['reinhard'] / n:.4f}** | | | | | |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
