"""The grade itself: fit a colour transform from two images, then apply it.

Pipeline shape, in order. Every stage is a function of colour alone, which is
what lets the whole thing be baked into a 3D LUT (see :mod:`chromagrade.lut`).

    sRGB -> linear
      1. von Kries chromatic adaptation      (white balance)
      2. exposure gain                       (level normalisation)
    linear -> Oklab
      3. monotone slope-limited tone curve   (contrast, black/white point)
      4. linear Monge-Kantorovich transport  (palette, global colour geometry)
      5. iterative distribution transfer     (non-Gaussian palette structure)
      6. lightness-banded chroma residual    (shadow/highlight coloration)
      7. skin-locus hue protection
      8. neutral-axis chroma cap
      9. saturation trim
    Oklab -> gamut-mapped linear -> sRGB

Stages 1-2 are the *normalise* half and 3-6 the *stylise* half; separating them
is what makes the method robust to targets and references that were shot under
completely different light, because by the time the distribution matching runs,
the two clouds are already roughly registered.

Stage 5 and 6 only run in ``quality`` mode. Stage 4 alone (``fast``) is already
a complete, defensible grade -- it is Pitie & Kokaram's MKL transfer with a
proper perceptual space, tone curve and gamut mapper around it.

After the LUT is applied at full resolution, one optional spatial stage runs:
detail restoration (:mod:`chromagrade.guided`), which undoes the noise/grain
amplification that any contrast expansion causes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import lut as lut_mod
from .colorspace import (
    linear_to_lms,
    linear_to_oklab,
    linear_to_srgb,
    lms_to_linear,
    luminance,
    oklab_to_linear,
    srgb_to_linear,
    srgb_to_oklab,
)
from .gamut import gamut_map_to_linear
from .guided import restore_detail
from .protect import apply_neutral_guard, apply_skin_protection
from .stats import mkl_transform, quantiles, trimmed_log_mean, weighted_mean, weighted_mean_cov
from .transport import IDTChain, MonotoneMap, build_quantile_map

__all__ = ["MatchParams", "GradeTransform", "fit_transform", "color_match", "MODES", "LUT_SIZE"]

MODES = ("quality", "fast")

# 33 is the de-facto standard cube size for colour LUTs and is plenty for a
# transform this smooth: measured worst-case round-trip error against direct
# evaluation is well under one 8-bit code value (see tests/test_lut.py).
LUT_SIZE = 33

_MODE_SETTINGS = {
    "quality": {"analysis_px": 384, "idt_iterations": 10, "bands": 5},
    "fast": {"analysis_px": 256, "idt_iterations": 0, "bands": 0},
}

# How much of the IDT refinement to keep on top of the MKL result. IDT is a much
# tighter distribution match but is also the stage most able to overfit an
# unrelated reference; holding a fifth of the MKL result back is a cheap and
# very effective brake.
_IDT_WEIGHT = 0.8

# Band residuals are shrunk by n_eff / (n_eff + _BAND_PRIOR) so that a lightness
# band containing forty pixels cannot dictate the grade's shadow tint.
_BAND_PRIOR = 256.0
_BAND_MAX_SHIFT = 0.05  # Oklab chroma units
_BAND_WEIGHT = 0.7

_WB_GAIN_LIMIT = (0.45, 2.2)
# About +/- 3.3 stops. Wide enough to carry a high-key plate down to a crushed
# reference (or the reverse) in one move, which is the whole job of the
# normalisation half; the tone curve is then only asked for shape.
_EXPOSURE_LIMIT = (0.1, 10.0)

# Per-segment slope bounds on the tone curve. The lower bound is the guard
# against posterisation and crushed blacks; the upper is the guard against
# blown highlights and noise amplification. Eight is deliberately generous --
# real grades do reach it, and `detail_preservation` exists specifically to
# clean up the grain that a steep curve lifts.
_TONE_SLOPE_RANGE = (0.15, 8.0)

# Quantiles converge like 1/sqrt(n); 64k samples put the standard error of every
# knot below a thousandth of the range, which is finer than the curves are
# smoothed to. Past that, more samples only cost time.
_MAX_ANALYSIS_POINTS = 65_536

# Total batch pixels above which a CPU input is moved to the accelerator.
# 512x512 is where measured end-to-end CUDA time first beats CPU (247 ms vs
# 334 ms); below it the fit's many small kernels dominate and the CPU is ahead.
_GPU_PIXEL_THRESHOLD = 1 << 18

# Confidence gates. A target with no tonal spread has no tone distribution to
# transport, and one with no chroma spread has no palette geometry; forcing a
# transform anyway is how a black frame ends up mid-grey and a greyscale plate
# ends up blotchy. Below the low threshold the ill-posed part of the transform
# is switched off entirely, above the high threshold it is fully trusted, and
# in between it fades. Both bands sit far below any real image: 0.05 of Oklab
# lightness is about 13 code values end to end.
_TONE_CONFIDENCE_BAND = (0.006, 0.050)
_PALETTE_CONFIDENCE_BAND = (0.004, 0.030)


def _clamp(value, lo: float, hi: float, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return min(max(f, lo), hi)


@dataclass
class MatchParams:
    """User-facing controls. Every field is honoured; none are decorative."""

    mode: str = "quality"
    strength: float = 1.0
    white_balance: float = 1.0
    tonal_transfer: float = 1.0
    palette_transfer: float = 1.0
    skin_protection: float = 0.5
    detail_preservation: float = 0.5
    saturation: float = 1.0

    def normalised(self) -> "MatchParams":
        """Coerce whatever arrived into something the pipeline can trust.

        Out-of-range values clamp; unparseable or non-finite ones fall back to
        the field's default. NaN in particular has to be caught explicitly:
        Python's ``min``/``max`` propagate it silently, and a NaN weight would
        otherwise poison the exposure gain and hand back a black frame.
        """
        return MatchParams(
            mode=self.mode if self.mode in MODES else "quality",
            strength=_clamp(self.strength, 0.0, 1.0, 1.0),
            white_balance=_clamp(self.white_balance, 0.0, 1.0, 1.0),
            tonal_transfer=_clamp(self.tonal_transfer, 0.0, 1.0, 1.0),
            palette_transfer=_clamp(self.palette_transfer, 0.0, 1.0, 1.0),
            skin_protection=_clamp(self.skin_protection, 0.0, 1.0, 0.5),
            detail_preservation=_clamp(self.detail_preservation, 0.0, 1.0, 0.5),
            saturation=_clamp(self.saturation, 0.0, 2.0, 1.0),
        )


@dataclass
class GradeTransform:
    """A fitted colour transform, evaluable on any sRGB colour."""

    params: MatchParams
    wb_gain: torch.Tensor  # [3] von Kries gains in Oklab's LMS cone space
    exposure_gain: float
    tone: MonotoneMap
    mkl_a: torch.Tensor  # [3, 3]
    mkl_b: torch.Tensor  # [3]
    idt: IDTChain | None = None
    idt_weight: float = _IDT_WEIGHT
    band_centres: torch.Tensor | None = None  # [K]
    band_sigma: float = 0.16
    band_shift: torch.Tensor | None = None  # [K, 2]
    notes: dict = field(default_factory=dict)

    def apply_srgb(self, srgb: torch.Tensor) -> torch.Tensor:
        """Grade ``[..., 3]`` gamma-encoded sRGB in ``[0, 1]``."""
        p = self.params
        shape = srgb.shape
        x = srgb.reshape(-1, 3)
        device, dtype = x.device, x.dtype

        lab_orig = srgb_to_oklab(x)

        lin = srgb_to_linear(x)
        gain = self.wb_gain.to(device=device, dtype=dtype)
        if p.white_balance > 0.0:
            gain = gain.pow(p.white_balance)
            lin = lms_to_linear(linear_to_lms(lin) * gain)
        if abs(self.exposure_gain - 1.0) > 1e-6:
            lin = lin * (self.exposure_gain ** p.tonal_transfer)

        lab = linear_to_oklab(lin)

        lightness = self.tone(lab[..., 0:1])
        staged = torch.cat([lightness, lab[..., 1:3]], dim=-1)

        moved = staged @ self.mkl_a.to(device=device, dtype=dtype).transpose(0, 1) + self.mkl_b.to(
            device=device, dtype=dtype
        )
        if self.idt is not None and len(self.idt) > 0 and self.idt_weight > 0.0:
            refined = self.idt(moved)
            moved = torch.lerp(moved, refined, self.idt_weight)

        ab = moved[..., 1:3]
        if self.band_shift is not None and self.band_centres is not None:
            ab = ab + self._band_offset(lightness)

        if p.palette_transfer < 1.0:
            ab = torch.lerp(staged[..., 1:3], ab, p.palette_transfer)

        out = torch.cat([lightness, ab], dim=-1)
        out = apply_skin_protection(lab_orig, out, p.skin_protection)
        out = apply_neutral_guard(lab_orig, out)
        if abs(p.saturation - 1.0) > 1e-6:
            out = torch.cat([out[..., 0:1], out[..., 1:3] * p.saturation], dim=-1)

        return linear_to_srgb(gamut_map_to_linear(out)).clamp(0.0, 1.0).reshape(shape)

    def _band_offset(self, lightness: torch.Tensor) -> torch.Tensor:
        centres = self.band_centres.to(device=lightness.device, dtype=lightness.dtype)
        shift = self.band_shift.to(device=lightness.device, dtype=lightness.dtype)
        w = torch.exp(-0.5 * ((lightness - centres) / self.band_sigma) ** 2)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return w @ shift


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def _estimate_illuminant(lin: torch.Tensor, p: float = 6.0) -> torch.Tensor:
    """Shades-of-Grey illuminant estimate on linear RGB, normalised to Y = 1.

    Minkowski order 6 sits between Grey World (p=1, biased by any large
    coloured region) and White Patch (p=inf, hostage to a single specular
    pixel), and is the order the shades-of-grey literature settles on. Clipped
    and near-black pixels are excluded because neither carries illuminant
    information: a clipped pixel's ratios are destroyed and a black pixel's are
    noise.
    """
    y = luminance(lin)
    valid = (lin.amax(dim=-1, keepdim=True) < 0.95) & (y > 2e-3)
    valid = valid.squeeze(-1)
    sel = lin[valid] if int(valid.sum()) >= 64 else lin
    est = sel.clamp_min(0.0).pow(p).mean(dim=0).clamp_min(1e-8).pow(1.0 / p)
    est = est / luminance(est).clamp_min(1e-8)
    if not bool(torch.isfinite(est).all()):
        return torch.ones(3, device=lin.device, dtype=lin.dtype)
    return est


def _white_balance_gain(lin_target: torch.Tensor, lin_reference: torch.Tensor) -> torch.Tensor:
    """Luminance-preserving von Kries gains carrying target light -> reference light."""
    ill_t = _estimate_illuminant(lin_target)
    ill_r = _estimate_illuminant(lin_reference)
    lms_t = linear_to_lms(ill_t).clamp_min(1e-6)
    lms_r = linear_to_lms(ill_r).clamp_min(1e-6)
    gain = (lms_r / lms_t).clamp(*_WB_GAIN_LIMIT)

    # Renormalise so the gain is purely chromatic: white must keep its
    # luminance, otherwise white balance and exposure fight each other.
    white = torch.ones(3, device=lin_target.device, dtype=lin_target.dtype)
    scaled = lms_to_linear(linear_to_lms(white) * gain)
    k = (luminance(scaled) / luminance(white)).clamp_min(1e-6)
    return (gain / k).clamp(*_WB_GAIN_LIMIT)


def _band_residuals(
    mapped: torch.Tensor,
    reference: torch.Tensor,
    lightness: torch.Tensor,
    n_bands: int,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-lightness-band mean-chroma corrections (shadow/highlight coloration).

    A single global distribution match reproduces the reference's *overall*
    palette but averages away the thing colourists actually reach for -- cool
    shadows against warm highlights. Comparing mean chroma band by band and
    applying the difference as a smooth function of lightness restores it,
    while the shrinkage and the hard cap keep a sparsely populated band from
    inventing a tint.
    """
    device = mapped.device
    centres = torch.linspace(0.08, 0.95, n_bands, device=device, dtype=mapped.dtype)
    shifts = torch.zeros(n_bands, 2, device=device, dtype=mapped.dtype)
    map_ab, ref_ab = mapped[:, 1:3], reference[:, 1:3]
    ref_l = reference[:, 0:1]
    for k in range(n_bands):
        c = centres[k]
        w_m = torch.exp(-0.5 * ((lightness - c) / sigma) ** 2).reshape(-1)
        w_r = torch.exp(-0.5 * ((ref_l - c) / sigma) ** 2).reshape(-1)
        mean_m, n_m = weighted_mean(map_ab, w_m)
        mean_r, n_r = weighted_mean(ref_ab, w_r)
        shrink = (n_m / (n_m + _BAND_PRIOR)) * (n_r / (n_r + _BAND_PRIOR))
        delta = (mean_r - mean_m) * shrink * _BAND_WEIGHT
        norm = float(delta.norm())
        if norm > _BAND_MAX_SHIFT:
            delta = delta * (_BAND_MAX_SHIFT / norm)
        shifts[k] = delta.to(shifts.dtype)
    return centres, shifts


def _smoothstep(value: float, lo: float, hi: float) -> float:
    t = min(max((value - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _spread(x: torch.Tensor, lo: float = 0.02, hi: float = 0.98) -> float:
    q = quantiles(x.reshape(-1), torch.tensor([lo, hi], device=x.device))
    return float(q[1] - q[0])


def _confidences(lab: torch.Tensor) -> tuple[float, float]:
    """How much of the ill-posed transform this target can actually support.

    Returns ``(tone, palette)`` in ``[0, 1]``. Both are 1 for any ordinary
    photograph and fall to 0 only for genuinely degenerate input -- a solid
    colour, a two-value graphic, a perfectly neutral plate.

    What they gate is strictly the part of the transform that needs a
    *distribution* to be well posed: the tone curve's shape and the transport
    matrix. The two *level* corrections -- the exposure gain and the mean
    alignment -- are never gated, because "put this frame at the reference's
    brightness and cast" is well defined even for a flat plate. That split is
    what lets a near-constant target still land in the right place instead of
    refusing to move.
    """
    tone = _smoothstep(_spread(lab[:, 0]), *_TONE_CONFIDENCE_BAND)
    chroma_spread = 0.5 * (_spread(lab[:, 1]) + _spread(lab[:, 2]))
    palette = _smoothstep(chroma_spread, *_PALETTE_CONFIDENCE_BAND)
    return tone, palette


def fit_transform(target_srgb: torch.Tensor, reference_srgb: torch.Tensor, params: MatchParams) -> GradeTransform:
    """Fit a :class:`GradeTransform` from two ``[N, 3]`` sRGB point clouds."""
    params = params.normalised()
    settings = _MODE_SETTINGS[params.mode]
    device = target_srgb.device
    dtype = torch.float32

    tgt = target_srgb.reshape(-1, 3).to(dtype)
    ref = reference_srgb.reshape(-1, 3).to(dtype)

    lin_t = srgb_to_linear(tgt)
    lin_r = srgb_to_linear(ref)

    wb_gain = _white_balance_gain(lin_t, lin_r)
    lin_t_wb = lms_to_linear(linear_to_lms(lin_t) * wb_gain)

    log_t = trimmed_log_mean(luminance(lin_t_wb))
    log_r = trimmed_log_mean(luminance(lin_r))
    exposure = float(torch.exp(log_r - log_t))
    if not math.isfinite(exposure):
        exposure = 1.0
    exposure = min(max(exposure, _EXPOSURE_LIMIT[0]), _EXPOSURE_LIMIT[1])

    # Confidence is measured on the target's *own* dynamic range, before the
    # exposure gain. Measuring afterwards conflates "flat" with "dark": a
    # correctly exposed-down bright plate would look low-contrast in Oklab and
    # gate itself off, which is how a high-key image used to refuse to be
    # graded at all.
    lab_t_raw = linear_to_oklab(lin_t_wb)
    conf_tone, conf_palette = _confidences(lab_t_raw)

    lab_t = linear_to_oklab(lin_t_wb * exposure)
    lab_r = linear_to_oklab(lin_r)

    tone = build_quantile_map(
        lab_t[:, 0],
        lab_r[:, 0],
        n_knots=41,
        slope_range=_TONE_SLOPE_RANGE,
        smooth=2,
        identity_blend=1.0 - params.tonal_transfer * conf_tone,
    )
    staged = torch.cat([tone(lab_t[:, 0:1]), lab_t[:, 1:3]], dim=-1)

    mean_s, cov_s, _ = weighted_mean_cov(staged)
    mean_d, cov_d, _ = weighted_mean_cov(lab_r)
    a64, _ = mkl_transform(mean_s, cov_s, mean_d, cov_d)
    if conf_palette < 1.0:
        # Shrink only the matrix toward identity, then re-derive the offset so
        # the cloud means still land on each other exactly. Losing the mean
        # alignment here would mean losing the reference's cast, which is the
        # one thing a degenerate target *can* still receive.
        eye = torch.eye(3, dtype=a64.dtype, device=a64.device)
        a64 = eye + (a64 - eye) * conf_palette
    b64 = mean_d - mean_s @ a64.transpose(0, 1)

    mkl_a = a64.to(dtype).to(device)
    mkl_b = b64.to(dtype).to(device)
    moved = staged @ mkl_a.transpose(0, 1) + mkl_b

    idt: IDTChain | None = None
    idt_weight = 0.0
    if settings["idt_iterations"] > 0 and conf_palette > 1e-3:
        idt = IDTChain.fit(moved, lab_r, iterations=settings["idt_iterations"])
        if len(idt) > 0:
            idt_weight = _IDT_WEIGHT * conf_palette
            moved = torch.lerp(moved, idt(moved), idt_weight)
        else:
            idt = None

    band_centres = band_shift = None
    sigma = 0.16
    if settings["bands"] > 0 and conf_palette > 1e-3:
        band_centres, band_shift = _band_residuals(
            moved, lab_r, staged[:, 0:1], settings["bands"], sigma
        )
        band_shift = band_shift * conf_palette

    return GradeTransform(
        params=params,
        wb_gain=wb_gain.to(dtype),
        exposure_gain=exposure,
        tone=tone,
        mkl_a=mkl_a,
        mkl_b=mkl_b,
        idt=idt,
        idt_weight=idt_weight,
        band_centres=band_centres,
        band_sigma=sigma,
        band_shift=band_shift,
        notes={
            "mode": params.mode,
            "exposure_gain": exposure,
            "wb_gain": [float(v) for v in wb_gain],
            "idt_steps": 0 if idt is None else len(idt),
            "confidence_tone": conf_tone,
            "confidence_palette": conf_palette,
        },
    )


# --------------------------------------------------------------------------- #
# Image-level entry point
# --------------------------------------------------------------------------- #


def _analysis_points(image: torch.Tensor, max_edge: int) -> torch.Tensor:
    """Downsample ``[B, H, W, 3]`` to an analysis cloud ``[N, 3]``.

    Area resampling, not nearest or bilinear: the analysis is a *distribution*
    estimate, and area averaging is the only common resampler that preserves
    the mean of every region it collapses. Nearest would alias the histogram;
    bilinear would bias it toward whatever the sampling grid happened to land
    on.
    """
    b, h, w, c = image.shape
    long_edge = max(h, w)
    if long_edge > max_edge:
        scale = max_edge / float(long_edge)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        small = F.interpolate(image.permute(0, 3, 1, 2), size=(nh, nw), mode="area")
        pts = small.permute(0, 2, 3, 1).reshape(-1, c)
    else:
        pts = image.reshape(-1, c)

    if pts.shape[0] > _MAX_ANALYSIS_POINTS:
        stride = int(math.ceil(pts.shape[0] / _MAX_ANALYSIS_POINTS))
        pts = pts[::stride]
    return pts.contiguous()


def _build_lut(transform: GradeTransform, size: int, device: torch.device) -> torch.Tensor:
    coords = lut_mod.lattice_coordinates(size, device, torch.float32)
    values = transform.apply_srgb(coords)
    lut = values.reshape(size, size, size, 3)
    lut = lut_mod.smooth_lut(lut, weight=1.0, passes=1)
    return lut.clamp(0.0, 1.0)


def _restore_detail_full(original: torch.Tensor, graded: torch.Tensor, amount: float) -> torch.Tensor:
    """Re-run the base/detail exchange at full resolution, in place, per item.

    One frame at a time and writing back over ``graded``: at 4K each of these
    intermediates is around 100 MB, and holding a whole batch of them is the
    difference between comfortable and out of memory.
    """
    for i in range(graded.shape[0]):
        lab_out = srgb_to_oklab(graded[i : i + 1])
        l_in = srgb_to_oklab(original[i : i + 1])[..., 0:1]
        l_new = restore_detail(l_in, lab_out[..., 0:1], amount)
        del l_in
        lab_out = torch.cat([l_new, lab_out[..., 1:3]], dim=-1)
        del l_new
        graded[i : i + 1] = linear_to_srgb(gamut_map_to_linear(lab_out)).clamp(0.0, 1.0)
    return graded


def _reference_groups(n_target: int, n_reference: int) -> list[list[int]]:
    """Which reference frames feed each output frame.

    * one reference           -> broadcast to every target
    * matching batch sizes    -> pairwise
    * anything else           -> every reference is pooled into one combined
      distribution and broadcast, which is both well defined and genuinely
      useful (a multi-frame reference is a more stable grade than one frame)
    """
    if n_reference == 1:
        return [[0]] * n_target
    if n_reference == n_target:
        return [[i] for i in range(n_target)]
    pooled = list(range(n_reference))
    return [pooled] * n_target


def _pick_device(image: torch.Tensor) -> torch.device:
    """Where to do the work.

    Below the threshold an image stays wherever it arrived: the fit is a long
    chain of tiny tensor ops, and on those a GPU is actually *slower* than a CPU
    (kernel launch overhead beats the arithmetic), so a round trip buys nothing.
    Above it the accelerator wins decisively, because LUT application and the
    guided filter are both bandwidth-bound and embarrassingly parallel --
    measured, 4K LUT application is 162 ms on CUDA against 3.5 s on CPU.

    The threshold is total pixels across the batch, not per frame: eight small
    frames are as much work as one large one.
    """
    if image.device.type != "cpu":
        return image.device
    if image.shape[0] * image.shape[1] * image.shape[2] < _GPU_PIXEL_THRESHOLD:
        return image.device
    try:  # pragma: no cover - depends on a live ComfyUI
        import comfy.model_management as mm

        dev = mm.get_torch_device()
        if dev is not None and dev.type != "cpu":
            return dev
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return image.device


def _validate(name: str, image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor IMAGE, got {type(image).__name__}")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"{name} must be a [batch, height, width, channels] IMAGE, got shape {tuple(image.shape)}")
    if image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] == 0:
        raise ValueError(f"{name} is empty (shape {tuple(image.shape)}); connect an image with non-zero size")
    if image.shape[3] not in (1, 3, 4):
        raise ValueError(
            f"{name} has {image.shape[3]} channels; expected 1 (grey), 3 (RGB) or 4 (RGBA)"
        )
    return image


def _to_rgb(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split an IMAGE into RGB and an optional alpha channel."""
    c = image.shape[3]
    if c == 1:
        return image.expand(-1, -1, -1, 3).contiguous(), None
    if c == 4:
        return image[..., :3].contiguous(), image[..., 3:4].contiguous()
    return image, None


def color_match(
    target: torch.Tensor,
    reference: torch.Tensor,
    params: MatchParams | None = None,
) -> torch.Tensor:
    """Transfer ``reference``'s colour treatment onto ``target``.

    Returns an IMAGE with ``target``'s batch size, height, width, channel count,
    dtype and device.
    """
    params = (params or MatchParams()).normalised()

    target = _validate("target_image", target)
    reference = _validate("color_reference", reference)

    out_device = target.device
    out_dtype = target.dtype
    out_channels = target.shape[3]

    if params.strength <= 0.0:
        return target.clone()

    work_device = _pick_device(target)
    try:
        return _run(target, reference, params, work_device, out_device, out_dtype, out_channels)
    except torch.cuda.OutOfMemoryError:  # pragma: no cover - hardware dependent
        if work_device.type == "cpu":
            raise
        torch.cuda.empty_cache()
        return _run(target, reference, params, torch.device("cpu"), out_device, out_dtype, out_channels)


def _run(
    target: torch.Tensor,
    reference: torch.Tensor,
    params: MatchParams,
    work_device: torch.device,
    out_device: torch.device,
    out_dtype: torch.dtype,
    out_channels: int,
) -> torch.Tensor:
    tgt = target.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
    ref = reference.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)

    tgt_rgb, alpha = _to_rgb(tgt)
    ref_rgb, _ = _to_rgb(ref)

    settings = _MODE_SETTINGS[params.mode]
    max_edge = settings["analysis_px"]

    ref_cache: dict[tuple[int, ...], torch.Tensor] = {}
    groups = _reference_groups(tgt_rgb.shape[0], ref_rgb.shape[0])

    graded = torch.empty_like(tgt_rgb)
    for i, group in enumerate(groups):
        key = tuple(group)
        ref_pts = ref_cache.get(key)
        if ref_pts is None:
            ref_pts = _analysis_points(ref_rgb[group], max_edge)
            ref_cache[key] = ref_pts
        tgt_pts = _analysis_points(tgt_rgb[i : i + 1], max_edge)

        transform = fit_transform(tgt_pts, ref_pts, params)
        table = _build_lut(transform, LUT_SIZE, work_device)
        table = lut_mod.blend_toward_identity(table, params.strength)
        graded[i : i + 1] = lut_mod.apply_lut(tgt_rgb[i : i + 1], table)

    if params.detail_preservation > 0.0:
        graded = _restore_detail_full(tgt_rgb, graded, params.detail_preservation)

    graded = torch.nan_to_num(graded, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    if out_channels == 1:
        # A single-channel IMAGE came in, so a single channel goes out: take the
        # graded lightness and re-encode it as a neutral. Doing this through
        # Oklab rather than a naive channel average keeps the tonal part of the
        # grade intact instead of averaging it away.
        lightness = srgb_to_oklab(graded)[..., 0:1]
        zeros = torch.zeros_like(lightness)
        neutral = torch.cat([lightness, zeros, zeros], dim=-1)
        graded = linear_to_srgb(oklab_to_linear(neutral))[..., 0:1].clamp(0.0, 1.0)
    elif alpha is not None:
        graded = torch.cat([graded, alpha], dim=-1)

    return graded.to(device=out_device, dtype=out_dtype)
