#!/usr/bin/env python3
"""VRAM estimation MODEL for the LTX-2 music-video pipeline (Plan 09.9-09/+19).

This is the pure, side-effect-free half of the VRAM guard. It holds the
single-source-of-truth VRAM constants and the estimation math (no ComfyUI /
nvidia-smi / subprocess calls) so it can be imported anywhere without pulling
in GPU-manager runtime deps or creating import cycles.

The other half (`mv_vram.py`) imports these symbols, adds the nvidia-smi
queries and the warm-card detection (which needs `mv_comfyui`), and re-exports
everything so existing call sites (`from mv_vram import ...`) keep working.

Plan 09.9-19 (mv-vram-frame-guard) ADDS a frame-count term to the activation
estimate: DiT refine KV-cache scales with `num_frames`, so a 15s clip costs
~1.5x and an 18s clip ~1.8x the 10s activation. The resolution-only guard
could not see this and over-approved long clips. The frame term is calibrated
from live OOM-boundary probe peaks (see debug session mv-vram-frame-guard).
"""

from __future__ import annotations

from typing import NamedTuple

# ── Resolution + two-stage + VRAM guard constants (Plan 09.9-09) ──

TWO_STAGE_LONG_EDGE_THRESHOLD = 768
"""Auto-enable two-stage sampling when long edge exceeds this."""

VRAM_SAFETY_MARGIN_MB = 300  # ~300MB headroom — activation headroom check is the real gate
"""Minimum free VRAM (MB) required beyond estimated need."""

TWO_STAGE_VRAM_OVERHEAD_FRAC = 0.15
"""DEPRECATED (kept for import compatibility). The old 15% two-stage multiplier
was bogus: both single- and two-stage paths run exactly ONE full-res KSampler
(single-stage node 3; two-stage refine node 41), so the VRAM PEAK is identical.
Two-stage only retains a couple of extra small latent buffers — modelled as a
flat additive (TWO_STAGE_RETAINED_LATENT_MB) instead. See modotte-oide-1080p-oom
debug session + specialist review."""

TWO_STAGE_RETAINED_LATENT_MB = 500
"""Flat VRAM overhead for the base + upscaled latent buffers retained across the
two-stage refine KSampler. Small vs the full-res activation peak."""

TEXT_ENCODER_VRAM_MB = 12000
"""GPU VRAM held by the Gemma-3-12B text encoder when device='default'.
Pinning it to CPU (device='cpu' in the LTXAVTextEncoderLoader) frees this."""

TEXT_ENCODER_CPU_MP_THRESHOLD = 3.0
"""Offload the text encoder to CPU when generation megapixels >= this. Raised
from 1.5 to 3.0 for Path B (Plan 09.9-18): the conv net refine (950MB) has
far lower activation than the old full DiT KSampler that required the encoder
offload. At 3.0 the encoder stays on GPU for all resolutions up to ~4K,
eliminating the CPU-bound text encoding bottleneck."""

VRAM_FALLBACK_W = 1088
VRAM_FALLBACK_H = 608
"""Fallback 16:9 resolution when 1080p two-stage doesn't fit."""

TARGET_W = 1920
TARGET_H = 1080
"""Final output resolution (after crop/scale)."""

# LTX-2 model VRAM footprint — single source of truth, imported by mv_comfyui.
LTX2_VRAM_MB = 38000

# Recalibrated VRAM model components (single source of truth for both the
# up-front guard estimate and the per-clip activation-headroom re-check).
LTX2_BASE_WEIGHTS_MB = 28500
"""Weights + video/audio VAE + text-projection + ComfyUI CUDA overhead
(text encoder EXCLUDED — added separately when GPU-resident).
Recalibrated from measured ComfyUI baseline ~28.5GB (modotte-oide 2026-07-11)."""

LTX2_ACT_PER_MP_MB = 2500
"""Path B conv net refine activation cost per megapixel. Recalibrated from 4500
(Path A low-res base pass) to 2500 for Path B's VRGDG upscale refine
(Plan 09.9-18): the upscaler's 950MB conv net has much lower activation than
the full DiT KSampler. Cold-card estimate at 1920x1088 with encoder GPU:
28500 + 12000 + 950 + (2500*2.09) + 500 = ~47.2GB, fits in 48GB. Per-clip
gate (warm): 5225 + 500 + 300 = ~6GB. Re-anchored from measured nvidia-smi
peak once Path B runs end-to-end."""

LTX2_ACT_PER_FRAME_MB = 30
"""Frame-count term (MB per generated frame) for the DiT refine KV-cache, added
in Plan 09.9-19 (mv-vram-frame-guard). The resolution-only guard could not see
frame-driven cost, so it approved 15s exactly as 10s even though refine VRAM
scales ~linearly with `num_frames` (10s->241, 15s->361=1.5x, 18s->433=1.8x).

Calibrated WARM from the OOM-boundary probe (debug session mv-vram-frame-guard,
2026-07-12): reliable warm peaks 10s=27315, 12s=27219, 14s=28531, 16s=29811 MiB;
slope from the 12->14 and 14->16 pairs = ~27 MB/frame. Rounded UP to 30 (≈10%
headroom) so the estimate lands conservatively ABOVE the real peak (safe). The
18s datum in the raw probe file is corrupt (short-circuited result), excluded;
the independent real 18s measurement was 31155 MiB, and
18100 + 30*441 + 950 = 32280 MiB (<4% high)."""

# I2I model VRAM footprint (kept for API completeness / re-export).
I2I_VRAM_MB = 30000

UPSCALER_CONV_NET_MB = 950
"""VRAM for the ltx-2.3-spatial-upscaler-x2-1.1 conv net (LatentUpscaleModelLoader).
The 950MB LatentUpsampler conv net loaded alongside the base DiT for the combined
base-gen+upscale job (Plan 09.9-16). Small vs the DiT weights — <1GB (matches the
plan-checker expectation)."""

COMBINED_RESIDENT_BASE_MB = 18100
"""Warm-resident VRAM base (MiB) for the combined base-gen+upscale graph (Path B,
09.9-16), FOLDED from model weights + video/audio VAEs + Gemma text encoder +
MP-dependent refine activation. Warm-calibrated from the OOM-boundary probe
(debug session mv-vram-frame-guard, 2026-07-12): reliable warm peaks
10s=27315, 12s=27219, 14s=28531, 16s=29811 MiB; 18s real=31155 MiB. The combined
estimate 18s = 18100 + 30*441 + 950 = 32280 MiB est vs 31155 MiB real (<4%, safe).

IMPORTANT: the text encoder (~12GB) and the MP-dependent refine activation are
FOLDED INTO THIS BASE. Do NOT add TEXT_ENCODER_VRAM_MB or LTX2_ACT_PER_MP_MB
separately in the combined path — that double-counts and over-estimates ~49.7GB
for 18s vs the real 31.2GB (the original bug this debug fixes). See `_estimate_combined_vram_mb`."""

COMBINED_VRAM_CEILING_MB = 46000
"""Hard ceiling for the combined base-gen+upscale graph: 48GB card − 2GB safety.
Plan 09.9-16 budget premise. If a combined estimate exceeds this it is a PLANNING
FINDING (surface to operator), NOT an automatic revert to Option A."""

# Frame-count formula — mirrors gpu-manager/src/workflow_ltx2._ltx2_num_frames
# (LTXVImgToVideo `length` input has step=8, min=9). Isolated here so the guard
# can derive frame count WITHOUT importing the gpu-manager workflow module.
LTX2_FRAME_RATE = 24
LTX2_FRAME_STRIDE = 8


def _ltx2_guard_num_frames(length_s: float) -> int:
    """Frame count for a clip of `length_s` seconds, rounded to 8k+1.

    Mirrors ``workflow_ltx2._ltx2_num_frames`` so the VRAM guard can compute the
    frame-driven activation term without importing the ComfyUI workflow graph.
    """
    raw = length_s * LTX2_FRAME_RATE
    k = round((raw - 1) / LTX2_FRAME_STRIDE)
    if k < 1:
        k = 1
    return LTX2_FRAME_STRIDE * k + 1


VramGuardResult = NamedTuple(
    "VramGuardResult",
    [("ok", bool), ("eff_w", int), ("eff_h", int), ("reason", str), ("two_stage", bool)],
)
# `two_stage` reflects the resolution the guard actually approved. On a
# single-stage fallback it is False even when the caller requested two-stage,
# so the caller must honour guard.two_stage (not the original request).


def _two_stage_base(w: int, h: int) -> tuple[int, int]:
    """Return two-stage base resolution (16:9, /32-valid).

    Used as the cheap first-pass sample before the LatentUpscale refine.
    For 16:9 this is 512x288 (512/32=16, 288/32=9).
    """
    return (512, 288)


def _text_encoder_on_cpu(width: int, height: int) -> bool:
    """Decide whether the Gemma-3-12B text encoder should be pinned to CPU.

    True at high resolution (>= TEXT_ENCODER_CPU_MP_THRESHOLD megapixels) where
    the full-res DiT KSampler activation dominates and the ~12GB encoder must
    vacate GPU to avoid OOM. False at low resolution (encoder stays on GPU —
    faster, and there is plenty of headroom).

    Env override: MV_TEXT_ENCODER_CPU=1 forces CPU offload regardless of
    resolution. Used for validation/e2e runs on a warm card (ComfyUI + a
    prior-stage model already resident) where the up-front VRAM guard measures
    less free VRAM than its cold-card design assumption. CPU offload drops the
    estimate by ~12GB so a low-res base generation still clears the guard
    without OOM. Does NOT change generation resolution or the upscale graph.
    """
    if __import__("os").getenv("MV_TEXT_ENCODER_CPU", "").lower() in ("1", "true", "yes"):
        return True
    mp = (width * height) / 1_000_000
    return mp >= TEXT_ENCODER_CPU_MP_THRESHOLD


def _estimate_vram_mb(
    width: int, height: int, two_stage: bool,
    text_encoder_cpu: bool = False, num_frames: int = 0,
) -> int:
    """Estimate total GPU VRAM (MB) an LTX-2 clip generation will consume.

    Compared by the guard against nvidia-smi *free* VRAM measured BEFORE the
    LTX-2 model is resident (so free ~= whole-card budget, ~48.5GB with the
    LLM hibernated and ComfyUI freshly started).

    Two physically-motivated components (Plan 09.9-09 recalibration):
      base   = weights + VAEs + text-projection + ComfyUI CUDA overhead
               (~26GB), PLUS the ~12GB Gemma text encoder IF it stays on GPU.
      activation = LTX2_ACT_PER_MP_MB * megapixels   (resolution-dependent
               full-res KSampler refine) PLUS, when a frame count is known,
               num_frames * LTX2_ACT_PER_FRAME_MB    (frame-dependent DiT
               refine KV-cache, added in Plan 09.9-19).

    `num_frames=0` (legacy / unknown length) adds NO frame term — the guard's
    caller must pass the real frame count (via `_ltx2_guard_num_frames`) to
    protect against frame-driven OOM. The up-front guard passes the frame count
    of the run's configured max clip length; the per-clip gate passes the
    actual clip's frame count.

    Both single- and two-stage run ONE full-res KSampler => same peak; two-stage
    only adds TWO_STAGE_RETAINED_LATENT_MB for retained base/upscale latents.
    """
    mp = (width * height) / 1_000_000
    base = LTX2_BASE_WEIGHTS_MB + (0 if text_encoder_cpu else TEXT_ENCODER_VRAM_MB)
    est = base + int(LTX2_ACT_PER_MP_MB * mp)
    if num_frames > 0:
        est += int(num_frames * LTX2_ACT_PER_FRAME_MB)

    if two_stage:
        est += TWO_STAGE_RETAINED_LATENT_MB

    return est


def _estimate_combined_vram_mb(
    base_w: int, base_h: int, target_w: int, target_h: int,
    text_encoder_cpu: bool = True, num_frames: int = 0,
) -> int:
    """Estimate peak GPU VRAM (MiB) for the combined base-gen+upscale graph (Path B, 09.9-16).

    ONE chained ComfyUI job: base samples at LOW res (base_w x base_h), then the
    VRGDG upscale refine samples at HIGH res (target_w x target_h). The DiT is
    loaded ONCE (shared by base + refine), so its weights are counted once.

    Estimate = COMBINED_RESIDENT_BASE_MB
             + int(num_frames * LTX2_ACT_PER_FRAME_MB)   # frame-driven refine KV-cache
             + UPSCALER_CONV_NET_MB                        # spatial upscaler conv net

    Calibrated WARM from the OOM-boundary probe (mv-vram-frame-guard): 18s real
    peak 31155 MiB; this formula gives 18100 + 30*441 + 950 = 32280 MiB (<4% high,
    conservative/safe). Validated 10s/12s/14s/16s within a few % (see debug session).

    `base_w/base_h/target_w/target_h/text_encoder_cpu` are retained in the signature
    for call-site compatibility. They are NO LONGER re-added: the text encoder and the
    MP-dependent refine activation are FOLDED INTO COMBINED_RESIDENT_BASE_MB. Adding
    TEXT_ENCODER_VRAM_MB / LTX2_ACT_PER_MP_MB / LTX2_BASE_WEIGHTS_MB separately here
    DOUBLE-COUNTS and over-estimated ~49.7GB for 18s vs the real 31.2GB (the original
    bug). `num_frames=0` (legacy) omits the frame term.
    """
    return COMBINED_RESIDENT_BASE_MB + int(num_frames * LTX2_ACT_PER_FRAME_MB) + UPSCALER_CONV_NET_MB


def _activation_headroom_mb(
    width: int, height: int, two_stage: bool, num_frames: int = 0
) -> int:
    """Minimum free VRAM (MB) the per-clip gate requires AFTER the model is
    resident — just the full-res KSampler activation transient + safety margin.

    Deliberately EXCLUDES the ~26GB base weights + text encoder: those are
    already loaded on clips 2..N, so requiring the full estimate would
    double-count resident weights and spuriously fail (see mv_clip note). This
    floor mainly catches a re-appeared orphan LLM squatter mid-batch — free VRAM
    would collapse below the activation transient and the gate fails fast
    instead of OOMing the KSampler.

    The frame term (Plan 09.9-19) makes this floor scale with the clip's actual
    frame count: a longer clip has a larger KV-cache transient, so the gate
    requires proportionally more free VRAM and will SKIP (return None, no
    corruption) a clip whose frame-driven activation would exceed headroom —
    exactly the OOM protection the old resolution-only floor lacked.
    """
    mp = (width * height) / 1_000_000
    act = int(LTX2_ACT_PER_MP_MB * mp)
    if num_frames > 0:
        act += int(num_frames * LTX2_ACT_PER_FRAME_MB)
    if two_stage:
        act += TWO_STAGE_RETAINED_LATENT_MB
    return act + VRAM_SAFETY_MARGIN_MB
