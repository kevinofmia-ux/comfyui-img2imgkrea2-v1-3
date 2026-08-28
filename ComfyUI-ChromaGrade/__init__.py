"""ComfyUI-ChromaGrade
====================

Reference-based AI colour matching. Feed it the image you want graded and an
image whose look you want, and it transfers the *treatment* -- white balance,
exposure, contrast, palette, shadow and highlight coloration -- without
importing any of the reference's content.

The grade is fitted from downsampled statistics, baked into a smooth
regularised 3D LUT and applied at full resolution with tetrahedral
interpolation. Because a 3D LUT is a pure per-pixel colour map, the target's
geometry, texture and fine detail survive by construction: there is nothing in
the transform that can blur an edge or ring around one.

No model weights, no downloads, no network access, fully deterministic. See
README.md for usage and docs/METHOD.md for the method and its citations.

Registration is guarded: if anything in here fails, ComfyUI gets one clear log
line instead of a broken custom-node scan.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ChromaGrade")

__version__ = "1.0.0"

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

try:
    from .chromagrade import (
        NODE_CLASS_MAPPINGS as _CLASSES,
    )
    from .chromagrade import (
        NODE_DISPLAY_NAME_MAPPINGS as _NAMES,
    )

    NODE_CLASS_MAPPINGS.update(_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_NAMES)
    logger.info("ChromaGrade %s: registered %d node(s)", __version__, len(NODE_CLASS_MAPPINGS))
except Exception as exc:  # pragma: no cover - import-time guard
    logger.error("ChromaGrade %s failed to load: %s: %s", __version__, type(exc).__name__, exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
