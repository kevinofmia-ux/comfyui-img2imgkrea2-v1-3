"""ComfyUI node registration for ChromaGrade."""

from __future__ import annotations

import logging

import torch

from .pipeline import MODES, MatchParams, color_match

logger = logging.getLogger("ChromaGrade")

__all__ = ["ChromaGradeColorMatch", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_CATEGORY = "image/color"


class ChromaGradeColorMatch:
    """Reference-based colour matching.

    Two IMAGE inputs, one IMAGE output. Everything else is a dial, and every
    dial is wired to something real.
    """

    DESCRIPTION = (
        "Transfers the colour treatment of COLOR_REFERENCE onto TARGET_IMAGE -- white balance, "
        "exposure, contrast, palette and shadow/highlight coloration -- while leaving the target's "
        "content, geometry and detail alone. The grade is fitted at low resolution, baked into a "
        "smooth 3D LUT and applied at full resolution, so it cannot soften edges or create halos."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TARGET_IMAGE": (
                    "IMAGE",
                    {"tooltip": "The image to be graded. Its content, geometry and resolution are preserved."},
                ),
                "COLOR_REFERENCE": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "The image supplying the look. Only its colour statistics are used -- none of its "
                            "objects, texture or composition can appear in the result."
                        )
                    },
                ),
                "mode": (
                    list(MODES),
                    {
                        "default": "quality",
                        "tooltip": (
                            "quality: full distribution transfer plus per-lightness-band coloration. "
                            "fast: closed-form Monge-Kantorovich transfer only -- roughly 3x quicker to fit, "
                            "slightly less faithful on complex palettes, equally artefact-free."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blends the finished grade with the untouched target. 0.0 is an exact pass-through.",
                    },
                ),
                "white_balance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "How far to carry the target's estimated illuminant onto the reference's. "
                            "Lower this when the target's white balance is already correct and you only want "
                            "the reference's palette."
                        ),
                    },
                ),
                "tonal_transfer": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Exposure, contrast and black/white point matching. 0.0 keeps the target's own "
                            "tonality and transfers colour only."
                        ),
                    },
                ),
                "palette_transfer": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Chroma distribution matching: palette relationships, saturation character and "
                            "shadow/highlight coloration. 0.0 keeps white balance and tonality only."
                        ),
                    },
                ),
                "skin_protection": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Holds hue steady inside the skin locus and caps chroma gain there, so a reference "
                            "with an unrelated palette cannot push faces green or magenta. Lightness is never "
                            "affected. Raise it for portraits, drop it for landscapes."
                        ),
                    },
                ),
                "detail_preservation": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Restores the target's original fine-detail amplitude after grading, using an "
                            "edge-aware guided filter. Counteracts the grain and noise amplification that any "
                            "contrast expansion causes. 1.0 keeps detail exactly as it arrived."
                        ),
                    },
                ),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Final chroma trim in Oklab, applied before gamut mapping. 1.0 leaves the match as fitted.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = _CATEGORY

    def match(
        self,
        TARGET_IMAGE: torch.Tensor,  # noqa: N803 - the product-facing socket name
        COLOR_REFERENCE: torch.Tensor,  # noqa: N803
        mode: str = "quality",
        strength: float = 1.0,
        white_balance: float = 1.0,
        tonal_transfer: float = 1.0,
        palette_transfer: float = 1.0,
        skin_protection: float = 0.5,
        detail_preservation: float = 0.5,
        saturation: float = 1.0,
    ):
        params = MatchParams(
            mode=mode,
            strength=strength,
            white_balance=white_balance,
            tonal_transfer=tonal_transfer,
            palette_transfer=palette_transfer,
            skin_protection=skin_protection,
            detail_preservation=detail_preservation,
            saturation=saturation,
        )
        try:
            with torch.no_grad():
                result = color_match(TARGET_IMAGE, COLOR_REFERENCE, params)
        except (ValueError, TypeError) as exc:
            # These are the ones with a human-readable cause already attached.
            raise RuntimeError(f"ChromaGrade: {exc}") from exc
        except Exception as exc:  # pragma: no cover - genuinely unexpected
            logger.exception("ChromaGrade: colour match failed")
            raise RuntimeError(
                "ChromaGrade: colour match failed unexpectedly "
                f"({type(exc).__name__}: {exc}). Target shape "
                f"{tuple(getattr(TARGET_IMAGE, 'shape', ()))}, reference shape "
                f"{tuple(getattr(COLOR_REFERENCE, 'shape', ()))}."
            ) from exc
        return (result,)


NODE_CLASS_MAPPINGS = {"ChromaGradeColorMatch": ChromaGradeColorMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"ChromaGradeColorMatch": "AI Color Match (ChromaGrade)"}
