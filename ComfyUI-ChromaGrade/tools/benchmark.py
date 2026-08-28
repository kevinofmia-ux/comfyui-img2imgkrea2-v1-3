"""Runtime and peak-memory benchmark at practical image sizes.

    python tools/benchmark.py
    python tools/benchmark.py --sizes 1024 2048 --device cuda

Reports the fit/LUT/apply split separately from the total, because they scale
differently: fitting is bounded by the analysis budget and is effectively
constant, while application and detail restoration scale with pixel count.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromagrade import MatchParams, color_match  # noqa: E402
from chromagrade.lut import apply_lut  # noqa: E402
from chromagrade.pipeline import LUT_SIZE, _analysis_points, _build_lut, fit_transform  # noqa: E402
from tests import fixtures  # noqa: E402


def _scene(size: int, device: torch.device) -> torch.Tensor:
    base = fixtures.gradient_scene(h=256, w=256).permute(0, 3, 1, 2)
    big = torch.nn.functional.interpolate(base, size=(size, size), mode="bilinear", align_corners=False)
    g = torch.Generator().manual_seed(101)
    noise = torch.rand(1, 3, size, size, generator=g) * 0.04
    return (big + noise).clamp(0, 1).permute(0, 2, 3, 1).contiguous().to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device: torch.device, repeats: int = 3) -> float:
    fn()
    _sync(device)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def run(sizes: list[int], devices: list[str], modes: list[str]) -> None:
    reference = fixtures.warm_sunset_reference(h=512, w=512)
    print(
        "note: the stage columns run on the named device, but `total` calls the node's own\n"
        "      entry point, which moves CPU inputs above 512x512 to the accelerator on purpose.\n"
        "      So a cpu `total` below the cpu stage sum is that policy working, not an error.\n"
    )
    print(f"{'device':>7} {'mode':>8} {'size':>10} {'fit':>9} {'lut':>9} {'apply':>9} {'total':>10} {'peak MB':>9}")
    for dev_name in devices:
        device = torch.device(dev_name)
        ref = reference.to(device)
        for mode in modes:
            params = MatchParams(mode=mode)
            for size in sizes:
                image = _scene(size, device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                pts_t = _analysis_points(image, 384)
                pts_r = _analysis_points(ref, 384)
                fit_ms = _time(partial(fit_transform, pts_t, pts_r, params), device)
                transform = fit_transform(pts_t, pts_r, params)
                lut_ms = _time(partial(_build_lut, transform, LUT_SIZE, device), device)
                lut = _build_lut(transform, LUT_SIZE, device)
                apply_ms = _time(partial(apply_lut, image, lut), device)
                total_ms = _time(partial(color_match, image, ref, params), device)

                peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
                print(
                    f"{dev_name:>7} {mode:>8} {size:>5}x{size:<4} "
                    f"{fit_ms:8.1f}m {lut_ms:8.1f}m {apply_ms:8.1f}m {total_ms:9.1f}m {peak:9.0f}"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--device", nargs="+", default=None, help="cpu, cuda, or both")
    ap.add_argument("--modes", nargs="+", default=["quality", "fast"])
    args = ap.parse_args()

    devices = args.device or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    run(args.sizes, devices, args.modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
