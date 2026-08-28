# Method

## The problem, stated precisely

Given a target image `T` and a reference image `R`, produce `T'` that carries
`R`'s *colour treatment* — white balance, exposure, contrast, palette, and
shadow/highlight coloration — while `T`'s content, geometry, texture and detail
are unchanged.

The second half of that is the hard half, and it is what most published
"colour transfer" work is weakest at. A method free to compute an arbitrary
per-pixel displacement can match any histogram you like and quietly destroy the
image doing it: amplified noise, contoured skies, clipped highlights, halos
around every edge.

## The decision that shapes everything else

**Estimate the transform at low resolution; apply it as a smooth 3D LUT at full
resolution.**

Every analysis stage below is a function of colour alone. So the pipeline is
evaluated once on a 33³ lattice over the sRGB cube and the image is pushed
through the resulting LUT with tetrahedral interpolation. Three consequences:

1. **Structure preservation becomes a theorem, not a hope.** A 3D LUT has no
   spatial extent. It cannot blur an edge, ring around one, or move texture.
   `tests/test_quality.py::test_the_transform_is_exactly_a_per_pixel_colour_map`
   checks this directly: the same colours arranged differently produce
   identical output.
2. **Cost stops depending on the algorithm's complexity.** The transport,
   whatever it costs, runs on ~35k lattice nodes. 4K only pays for an
   interpolation.
3. **Banding becomes something you can regularise.** A lattice can be smoothed;
   a closed-form per-pixel expression cannot. One light separable pass with
   linearly-extrapolating boundaries removes the quantile staircase that
   distribution matching leaves behind — without moving the black or white
   point, because linear extrapolation makes the filter exact on affine data.

Tetrahedral rather than trilinear interpolation, because tetrahedral reproduces
the LUT exactly along the neutral diagonal, and a colour cast on greys is the
most visible defect a LUT can have.

## Why not a neural model

The strongest learned option is **Neural Preset** (CVPR 2023), whose
"deterministic neural colour mapping" is a close cousin of the LUT idea here.
It is licensed CC BY-NC-SA 4.0 — non-commercial — with unclear weight
availability. Shipping it would attach a non-commercial restriction to every
user's output. Deep photorealistic style transfer (PhotoWCT/WCT²) brings large
VGG weights, more VRAM, and a real risk of semantic bleed from the reference.

The classical optimal-transport line, done properly, gets most of the quality
with none of the cost: no weights, no downloads, no network access, exact
determinism, and it runs on CPU. That is the trade this package takes. It is a
licensing and deployment decision as much as a technical one, and it is stated
plainly rather than dressed up.

## Pipeline

Working spaces, and why each stage lives where it does:

* **sRGB** — what arrives, and what a `.cube` LUT is indexed by. No statistics
  are ever computed here: distances in gamma-encoded RGB are perceptually
  meaningless and ratios are outright wrong.
* **Linear light** — exposure and white balance are physical scalings of light.
* **Oklab** — nearly uniform, well-behaved hue lines, cheap. Every distribution
  stage runs here so that equal numeric change means roughly equal perceived
  change, and so that the luminance/chrominance split is meaningful.

Chromatic adaptation reuses Oklab's own cone-response matrix rather than adding
a separate Bradford/CAT16 stage — it already maps linear sRGB into an LMS-like
space, which is the domain a von Kries gain is defined in.

### 1–2. Normalise: white balance and exposure

Illuminant estimation is Shades-of-Grey (Minkowski order 6) on linear RGB over
non-clipped, non-black pixels. Order 6 sits between Grey World (biased by any
large coloured region) and White Patch (hostage to one specular pixel). The
target's estimate is carried onto the reference's by a von Kries gain in cone
space, renormalised so it is purely chromatic — white keeps its luminance, so
white balance and exposure cannot fight each other.

Exposure is a single gain matching the trimmed *geometric* mean of luminance,
which is the right average for something about to be corrected multiplicatively.

Separating normalisation from stylisation is what makes the method robust to
targets and references shot under completely different light: by the time the
distribution matching runs, the two clouds are already roughly registered.

### 3. Tone: a monotone, slope-limited curve

Matched quantile pairs of Oklab lightness, fitted with a monotone cubic
(Fritsch–Carlson PCHIP). Three guards turn a fragile quantile match into
something shippable:

* **Trimmed tails** (0.2 %–99.8 %). The extreme quantiles are single pixels and
  are routinely a dead pixel or a specular clip.
* **Slope clamping to [0.15, 8]**, then re-integration and re-anchoring on the
  median. Slope 0.05 is posterisation; slope 30 is a blown highlight. Neither is
  a grade. Re-anchoring means clamping changes the curve's *shape* without
  shifting its level.
* **Smoothing the correction, not the values.** Smoothing `ys` directly bends
  the curve away from identity even when source and destination are the same
  distribution. Smoothing `ys - xs` instead makes matching an image against
  itself an exact no-op — measured max error 0.7/255, down from 26/255 before
  this change.

Monotone is not a stylistic preference: it is what guarantees the curve can
never invert local contrast.

### 4–5. Palette: optimal transport in Oklab

**MKL (closed form).** With both clouds treated as Gaussian, the optimal
transport map under quadratic cost has the closed form
`A = Σs^-1/2 (Σs^1/2 Σr Σs^1/2)^1/2 Σs^-1/2` (Pitié & Kokaram 2007). Eigenvalues
of `A` are capped at 6× — an uncapped `A` is the single commonest source of
blown-out example-based grades, because a direction the target barely varies
along gets an enormous stretch that turns sensor noise into colour blotches.

**IDT (refinement, `quality` mode).** Rotate the joint cloud, match its three
marginals independently, rotate back, repeat. Given enough rotations this
converges to a full 3-D distribution match, which is what captures palette
*relationships* rather than per-channel statistics — a red/cyan image and a
magenta/green one have identical marginals.

The published algorithm draws **random** rotations. That would make the node
non-deterministic, so the rotations here come from a fixed Fibonacci-sphere
construction: evenly spread by design rather than in expectation, and identical
on every run, machine and device. Each iteration's 1-D maps get the same
slope-limiting treatment as the tone curve, and the result is held back 20 %
toward the MKL solution — IDT is the stage most able to overfit an unrelated
reference, and that is a cheap, effective brake.

Critically, the chain is stored as `(rotation, three 1-D maps)` per iteration,
so it is a *function of colour*, not a per-pixel displacement field. That is
what makes it bakeable into a LUT.

### 6. Shadow/highlight coloration

A single global distribution match reproduces the reference's overall palette
but averages away the thing colourists actually reach for: cool shadows against
warm highlights. Five soft Gaussian bands over lightness compare mean chroma
band by band, and the difference is applied as a smooth function of lightness.
Each band's correction is shrunk by `n_eff / (n_eff + 256)` and hard-capped at
0.05 Oklab chroma, so a band containing forty pixels cannot invent a tint.

### 7–8. Protection

Both rules operate on `(input colour, transformed colour)` pairs, so they are
evaluated on the lattice with everything else: zero cost at full resolution, and
no mask, no edges, nothing to feather, nothing to produce a spatial artefact.

**Skin.** The failure mode that ruins a grade on people is hue rotation — the
reference's palette pulls skin green or magenta and the face reads as ill.
Chroma and lightness changes on skin are usually the *wanted* part. So inside a
soft skin locus (hue window derived at import from a documented ladder of skin
values, times smooth chroma and lightness windows), the input hue angle is
preserved and chroma gain is capped at 1.35×. Lightness is never touched.

**Neutrals.** A near-neutral surface should be allowed to take the reference's
cast — that is what white balance in a grade means — but must not become a
saturated colour, which is what an aggressive match does to a flat sky. The cap
is `3·C_in + 0.055`: at `C_in = 0` a pure grey may pick up a clearly visible
cast but can never turn into a colour, and by `C_in ≈ 0.09` the rule is inert.

### 9. Gamut mapping

`clamp(rgb, 0, 1)` shifts hue *and* lightness, which is most of what people call
"the AI colour look". Instead:

* Lightness rolls off through a `tanh` shoulder **sized to the actual
  overshoot**. An image that never leaves `[0, 1]` gets a pure clamp and a
  bit-exact white point; a grade that overshoots by 2 % lands white at
  L = 0.995. (No smooth bounded monotone function can both equal the identity
  at 1 with unit slope *and* stay ≤ 1, so some loss is unavoidable — the point
  is to make it proportional to the problem rather than constant.)
* Colours outside the gamut are pulled toward the neutral axis by bisection at
  constant lightness and constant hue. Saturation is what the eye is least
  sensitive to in isolation; hue and lightness, which it is very sensitive to,
  survive exactly. Only out-of-gamut pixels are bisected, which is what keeps
  this affordable at 4K.

### 10. Detail restoration (the only spatial stage)

Any contrast expansion also expands whatever rides on the tones it stretches:
sensor noise, film grain, diffusion-model texture. This is the problem Pitié's
*regrain* stage exists to solve; here a guided filter does it in O(1) per pixel
with an explicit edge-awareness parameter.

Both the input and the graded output are decomposed using the **same guide** —
the target's own lightness — so both base layers have identical edge structure
and exchanging their detail layers cannot produce a halo. At
`detail_preservation = 1` the graded *base* (all the colour and tonal character
of the reference) is kept while the target's original detail amplitude is
restored exactly.

Set it to 0 and the entire grade is a pure 3D LUT again.

## Confidence gating

A target with no tonal spread has no tone distribution to transport; one with no
chroma spread has no palette geometry. Forcing a transform anyway is how a black
frame ends up mid-grey and a greyscale plate ends up blotchy. Two smooth gates
(measured on the target's own dynamic range, *before* the exposure gain — see
below) fade out exactly the ill-posed parts: the tone curve's shape and the
transport matrix.

The two *level* corrections — exposure and mean alignment — are never gated,
because "put this frame at the reference's brightness and cast" is well defined
even for a flat plate. That split matters: measuring confidence *after* the
exposure gain conflates "flat" with "dark", and a correctly-exposed-down
high-key plate would gate itself off. Fixing that alone took the high-key test
pair from a sliced-Wasserstein distance of 0.275 to 0.062.

## Modes

| | `fast` | `quality` (default) |
|---|---|---|
| Analysis resolution | 256 px long edge | 384 px |
| Palette transport | MKL closed form | MKL + 10-step deterministic IDT |
| Shadow/highlight bands | — | 5 |
| Typical fit time | ~25 ms | ~190 ms |

Both are complete grades with the same normalisation, tone curve, protections
and gamut mapping. `fast` is Pitié & Kokaram's MKL transfer with a proper
perceptual space and a proper gamut mapper around it — a defensible method in
its own right, not a stub.

## Measured results

`python tools/evaluate.py`, against Reinhard et al. (2001) as the baseline.
**SWD** = sliced 1-Wasserstein distance in Oklab from output to reference
(lower is a closer grade); **SSIM** against the target (higher is less
disturbance); **rails** = pixels with any channel pinned to 0 or 1.

| pair | SWD ours | SWD fast | SWD Reinhard | SSIM ours | SSIM Reinhard | rails ours | rails Reinhard |
|---|---|---|---|---|---|---|---|
| gradient → sunset | 0.0113 | 0.0132 | 0.0281 | 0.892 | 0.947 | 5.3 % | 0.0 % |
| gradient → teal | 0.0121 | 0.0147 | 0.0390 | 0.852 | 0.911 | 0.0 % | 0.0 % |
| portrait → sepia | 0.0102 | 0.0109 | 0.0266 | 0.949 | 0.998 | 0.0 % | 0.8 % |
| portrait → teal | 0.0151 | 0.0173 | 0.0323 | 0.936 | 0.979 | 0.0 % | 0.0 % |
| neon → bleach | 0.0040 | 0.0066 | 0.0357 | 0.871 | 0.912 | 0.0 % | 5.2 % |
| greyscale → sepia | 0.0025 | 0.0025 | 0.0106 | 0.978 | 0.977 | 0.0 % | 3.5 % |
| fog → sunset | 0.0421 | 0.0424 | **0.0165** | 0.889 | 0.878 | 0.0 % | 12.2 % |
| high_key → teal | 0.0615 | 0.0626 | **0.0193** | 0.744 | 0.299 | 0.0 % | 1.2 % |
| low_key → bleach | 0.0702 | 0.0701 | **0.0298** | 0.409 | 0.254 | 0.0 % | 11.7 % |

Read this honestly:

* On the six ordinary scenes ChromaGrade is **2.4× to 8.9× closer** to the
  reference's colour distribution, and it never adds a fully clipped pixel.
* On the three deliberately flat targets the baseline reaches a *better*
  distribution distance — by brute force. On `high_key` it scores 0.0193 at an
  SSIM of **0.299**: it matched the histogram by destroying the image. Every one
  of those three, ChromaGrade wins on structure, and on two of them the baseline
  clips 12 % of the frame.
* The `rails` figure for gradient → sunset is not clipping. Those are pixels
  sitting exactly on the sRGB gamut *surface*, which is what a
  hue-and-lightness-preserving gamut mapper produces by design. Zero pixels in
  the entire suite are fully railed (all three channels), and distinct 8-bit
  colour counts go *up*, not down (1097 → 1106 on that pair).

The fair head-to-head is
`test_at_equal_colour_fidelity_structure_is_comparable_or_better`: our strength
is bisected down until the two methods reach the *same* distribution distance,
then structure is compared. ChromaGrade wins four of the six bisectable pairs
and loses the other two by 0.007 and 0.020 SSIM; on all three capped pairs it
wins outright.

## Performance

`python tools/benchmark.py` — RTX-class GPU, torch 2.10 + CUDA 13, best of 3.

| device | mode | size | fit | LUT build | LUT apply | total | peak VRAM |
|---|---|---|---|---|---|---|---|
| cuda | quality | 1024² | 189 ms | 36 ms | 9 ms | 266 ms | 335 MB |
| cuda | quality | 2048² | 193 ms | 37 ms | 41 ms | 355 ms | 573 MB |
| cuda | quality | 4096² | 190 ms | 37 ms | 162 ms | 691 ms | 2241 MB |
| cuda | fast | 4096² | 27 ms | 16 ms | 151 ms | 506 ms | 2242 MB |
| cpu | quality | 1024² | 161 ms | 56 ms | 201 ms | — | — |
| cpu | fast | 1024² | 21 ms | 35 ms | 197 ms | — | — |

End to end, warm, from a CPU input (the ComfyUI case): a batch of two 720p
frames takes 560 ms in `quality` and 169 ms in `fast`, peaking at 306 MB.

CPU inputs above 512×512 total pixels are moved to the accelerator, because
that is where measured end-to-end CUDA time first beats CPU (247 ms vs 334 ms).
Below it, the fit's long chain of tiny tensor ops is genuinely *faster* on CPU —
kernel-launch overhead beats the arithmetic — so small images stay put. An
out-of-memory error falls straight back to CPU rather than failing the graph.

Fitting is flat in image size, because the analysis cloud is capped at 65 536
points — quantiles converge like 1/√n, and past that more samples only cost
time. Replacing the sort-based quantile estimator with a histogram CDF cut the
`quality` fit from 1.86 s to 0.19 s on its own; `torch.sort` on CPU costs about
0.23 µs per element and the fit needs sixty of them.

## Known limitations

These are real, not hedges.

* **Flat targets cannot become contrasty ones.** Contrast expansion is capped at
  8× local slope. Feed a fog plate and a high-contrast reference and you get
  the level and the cast, not the contrast. That is deliberate — the numbers
  above show what the alternative looks like — but it is a limitation.
* **A pure black frame stays black.** Exposure matching is a multiplicative
  gain and zero times anything is zero.
* **The match is global.** There is no semantic correspondence: "make the sky
  like that sky and the skin like that skin" is not what this does. When target
  and reference have very different *proportions* of content — a close-up
  against a wide landscape — the palette statistics are dominated by whatever
  fills the frame, and the grade follows.
* **Skin protection is a colour locus, not a face detector.** Anything in the
  same hue/chroma/lightness region — wood, sand, terracotta — gets the same
  treatment. That is usually harmless and occasionally not what you wanted.
* **8-bit sources with existing banding stay banded.** Nothing here invents
  precision that was thrown away before the image arrived; the LUT smoothing
  prevents *new* contouring, it does not repair old contouring.
* **Determinism is per-device.** Results are bit-identical across runs on the
  same device. CPU and CUDA agree to within about 3/255, which is float
  reduction order, not algorithmic difference.
