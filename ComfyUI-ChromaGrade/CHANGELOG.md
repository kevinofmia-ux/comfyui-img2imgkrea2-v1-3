# Changelog

## 1.0.0

First release.

* **AI Color Match (ChromaGrade)** node: `TARGET_IMAGE` + `COLOR_REFERENCE` in,
  one `IMAGE` out, eight working controls.
* Pipeline: Shades-of-Grey illuminant estimation and von Kries adaptation in
  cone space, geometric-mean exposure matching, a monotone slope-limited tone
  curve on Oklab lightness, linear Monge–Kantorovich transport, deterministic
  Iterative Distribution Transfer, per-lightness-band chroma residuals,
  skin-locus hue protection, a neutral-axis chroma cap, and hue- and
  lightness-preserving gamut mapping with an adaptive shoulder.
* The whole thing is baked into a smoothed 33³ 3D LUT and applied with
  tetrahedral interpolation, so structure preservation is a property of the
  representation rather than an approximation.
* Optional guided-filter detail restoration is the only spatial stage.
* Deterministic: no RNG anywhere. The published IDT algorithm's random
  rotations are replaced by a fixed Fibonacci-sphere construction.
* No model weights, no downloads, no network access, no compiled extensions.
* 109 tests, a Reinhard baseline comparison, a contact-sheet renderer and a
  runtime/memory benchmark.
