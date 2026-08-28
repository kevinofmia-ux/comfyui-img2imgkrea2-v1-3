"""Render a visual contact sheet across every scene/reference combination.

    python tools/contact_sheet.py --out docs/contact_sheet.png

Every tile is generated from the synthetic fixtures in ``tests/fixtures.py``,
so the sheet is reproducible and carries no third-party image assets. The
leftmost column is the untouched target and the top row is the reference, which
makes it easy to see at a glance whether a grade picked up the reference's
*treatment* or started importing its content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromagrade import MatchParams, color_match  # noqa: E402
from tests import fixtures  # noqa: E402

TILE = 176
PAD = 8
LABEL = 18


def _resize(img: torch.Tensor, size: int) -> torch.Tensor:
    chw = img[..., :3].permute(0, 3, 1, 2)
    out = torch.nn.functional.interpolate(chw, size=(size, size), mode="area" if chw.shape[-1] > size else "bilinear")
    return out.permute(0, 2, 3, 1)[0].clamp(0, 1)


def build(mode: str, strength: float) -> torch.Tensor:
    scenes = list(fixtures.SCENES)
    refs = list(fixtures.REFERENCES)
    cols = 1 + len(refs)
    rows = 1 + len(scenes)

    w = cols * TILE + (cols + 1) * PAD
    h = rows * (TILE + LABEL) + (rows + 1) * PAD
    sheet = torch.full((h, w, 3), 0.10)

    def place(r: int, c: int, tile: torch.Tensor) -> None:
        y = PAD + r * (TILE + LABEL + PAD)
        x = PAD + c * (TILE + PAD)
        sheet[y : y + TILE, x : x + TILE] = tile

    for j, ref in enumerate(refs):
        place(0, 1 + j, _resize(fixtures.REFERENCES[ref](), TILE))

    params = MatchParams(mode=mode, strength=strength)
    for i, scene in enumerate(scenes):
        target = fixtures.SCENES[scene]()
        place(1 + i, 0, _resize(target, TILE))
        for j, ref in enumerate(refs):
            graded = color_match(target, fixtures.REFERENCES[ref](), params)
            place(1 + i, 1 + j, _resize(graded, TILE))
    return sheet, scenes, refs


def write_png(sheet: torch.Tensor, scenes, refs, path: Path, mode: str, strength: float) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - Pillow ships with ComfyUI
        raise SystemExit("Pillow is required to write the contact sheet: pip install pillow") from None

    arr = (sheet.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    def label(r: int, c: int, text: str) -> None:
        y = PAD + r * (TILE + LABEL + PAD) + TILE + 3
        x = PAD + c * (TILE + PAD) + 2
        draw.text((x, y), text, fill=(220, 220, 220))

    label(0, 0, f"ChromaGrade  mode={mode}  strength={strength:g}")
    for j, ref in enumerate(refs):
        label(0, 1 + j, f"ref: {ref}")
    for i, scene in enumerate(scenes):
        label(1 + i, 0, f"target: {scene}")
        for j, ref in enumerate(refs):
            label(1 + i, 1 + j, f"{scene} -> {ref}")

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"wrote {path} ({img.width}x{img.height})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "contact_sheet.png")
    ap.add_argument("--mode", default="quality", choices=["quality", "fast"])
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()

    sheet, scenes, refs = build(args.mode, args.strength)
    write_png(sheet, scenes, refs, args.out, args.mode, args.strength)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
