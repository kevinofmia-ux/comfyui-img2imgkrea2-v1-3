"""ChromaGrade -- reference-based colour matching for ComfyUI.

The public surface is deliberately small:

    from chromagrade import MatchParams, color_match
    graded = color_match(target_bhwc, reference_bhwc, MatchParams())

Everything else in the package is a stage of that one call and is documented in
``docs/METHOD.md``.
"""

from __future__ import annotations

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, ChromaGradeColorMatch
from .pipeline import LUT_SIZE, MODES, GradeTransform, MatchParams, color_match, fit_transform

__version__ = "1.0.0"

__all__ = [
    "ChromaGradeColorMatch",
    "GradeTransform",
    "LUT_SIZE",
    "MODES",
    "MatchParams",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "color_match",
    "fit_transform",
    "__version__",
]
