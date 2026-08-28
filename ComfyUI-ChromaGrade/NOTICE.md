# Attribution and licensing

ChromaGrade itself is MIT (see `LICENSE`). It bundles **no model weights, no
third-party source code, no datasets and no image assets**. Every fixture used
by the tests and the contact sheet is generated procedurally in
`tests/fixtures.py`.

Everything below is a method this package **implements from published
descriptions**. None of it is vendored code, and none of it imposes a licence
on this repository.

## Methods implemented

| Method | Source | Where it is used |
|---|---|---|
| Linear Monge–Kantorovich colour mapping | F. Pitié, A. Kokaram, *The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer*, CVMP 2007 | `stats.mkl_transform` — the closed-form transport core |
| Iterative Distribution Transfer (IDT) | F. Pitié, A. Kokaram, R. Dahyot, *Automated colour grading using colour distribution transfer*, CVIU 107(1):123–137, 2007 | `transport.IDTChain` — the non-Gaussian refinement |
| Grain/gradient preservation after transfer ("regrain") | same paper | `guided.restore_detail` solves the same problem with a guided filter instead of the paper's multiscale Jacobi solver |
| Guided image filtering | K. He, J. Sun, X. Tang, *Guided Image Filtering*, ECCV 2010 / TPAMI 35(6), 2013 | `guided.guided_filter` — the O(1) box-filter formulation |
| Oklab colour space | B. Ottosson, *A perceptual color space for image processing* (2020), <https://bottosson.github.io/posts/oklab/>. Released by the author as public domain / MIT. | `colorspace` — the working space for every distribution stage |
| Lightness-preserving gamut clipping | B. Ottosson, *Gamut clipping* (2021), <https://bottosson.github.io/posts/gamutclipping/> | `gamut.gamut_map_to_linear` |
| Shades-of-Grey illuminant estimation | G. Finlayson, E. Trezzi, *Shades of Gray and Colour Constancy*, Color Imaging Conference 2004 | `pipeline._estimate_illuminant` |
| Monotone cubic interpolation | F. N. Fritsch, R. E. Carlson, *Monotone Piecewise Cubic Interpolation*, SIAM J. Numer. Anal. 17(2), 1980 | `transport.MonotoneMap` |
| Colour transfer between images (baseline only) | E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley, *Color Transfer between Images*, IEEE CG&A 21(5), 2001 | `tests/baseline.py` — implemented purely to be measured against; not part of the node |
| SSIM | Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 13(4), 2004 | `tests/metrics.py` |

## Methods evaluated and deliberately not used

* **Neural Preset for Color Style Transfer** — Ke, Liu, Zhu, Zhao, Lau, CVPR
  2023 (<https://github.com/ZHKKKe/NeuralPreset>). Strong results and a
  genuinely relevant architecture, but the repository is licensed
  **CC BY-NC-SA 4.0**, which forbids commercial use, and pretrained weight
  availability is not clearly stated. Shipping it would put a non-commercial
  restriction on every ComfyUI user's output pipeline. Excluded on licensing
  grounds, not technical ones.
* **Deep photorealistic style transfer (PhotoWCT / WCT2 and successors)** —
  large VGG-family weights, heavier VRAM, and a real risk of semantic bleed
  from the reference into the target, which is precisely the failure this node
  exists to avoid.
* **Learned 3D LUT prediction (Zeng et al. and derivatives)** — attractive, but
  requires trained weights and a paired dataset to reproduce; the LUT
  representation itself, which is the useful part, is used here without them.

## Benchmark reference

The After Effects plugin *AI Color Match* (<https://bskl.xyz/aicolormatch>) was
used as a **product benchmark only** — what a good reference-based matcher
should be able to do, and what its published behaviour is (local GPU analysis
against a reference frame; exposure, contrast and white balance; a strength
control; LUT export in the Pro tier). No part of its implementation was
inspected, reverse engineered or reproduced, and none of its marketing claims
are treated here as technical fact.
