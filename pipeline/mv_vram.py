#!/usr/bin/env python3
"""VRAM guard + resolution/two-stage helpers (Plan 09.9-09 / +19).

This module is the import-surface for the rest of the pipeline. The pure
estimation math and constants live in `mv_vram_model` (a leaf with no
ComfyUI / nvidia-smi / subprocess deps, so it can't create an import cycle
with `mv_comfyui`). This module re-exports that surface and adds the parts
that DO need the GPU runtime: nvidia-smi queries, warm-card detection, and
the public guard entry points.

Plan 09.9-19 (mv-vram-frame-guard) made the guard FRAME-COUNT aware: every
estimate / headroom / guard function accepts `num_frames` (default 0 = legacy
resolution-only behaviour) so the DiT refine KV-cache cost — which scales with
generated frames — is now part of the budget. The up-front guard takes the
run's configured max clip length; the per-clip gate takes the actual clip's
frame count.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

from mv_mvconst import UPSCALE_FACTOR

# Re-export the pure model surface so `from mv_vram import ...` keeps working
# for every existing caller (mv_clip, generate_music_video_pipeline, mv_post,
# test_music_video_pipeline, mv_comfyui).
from mv_vram_model import (  # noqa: F401
    COMBINED_VRAM_CEILING_MB,
    I2I_VRAM_MB,
    LTX2_ACT_PER_FRAME_MB,
    LTX2_ACT_PER_MP_MB,
    LTX2_BASE_WEIGHTS_MB,
    LTX2_FRAME_RATE,
    LTX2_FRAME_STRIDE,
    LTX2_VRAM_MB,
    TEXT_ENCODER_CPU_MP_THRESHOLD,
    TEXT_ENCODER_VRAM_MB,
    TWO_STAGE_LONG_EDGE_THRESHOLD,
    TWO_STAGE_RETAINED_LATENT_MB,
    TWO_STAGE_VRAM_OVERHEAD_FRAC,
    TARGET_W,
    TARGET_H,
    UPSCALER_CONV_NET_MB,
    VRAM_FALLBACK_H,
    VRAM_FALLBACK_W,
    VRAM_SAFETY_MARGIN_MB,
    VramGuardResult,
    _activation_headroom_mb,
    _estimate_combined_vram_mb,
    _estimate_vram_mb,
    _ltx2_guard_num_frames,
    _text_encoder_on_cpu,
    _two_stage_base,
)

logger = logging.getLogger(__name__)


def _get_free_vram_mb() -> int | None:
    """Query nvidia-smi for free VRAM in MB. Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed: %s", result.stderr.strip())
            return None
        # Parse first GPU line: "49140, 38663" (comma-separated)
        line = result.stdout.strip().split("\n")[0].replace(",", "")
        parts = line.split()
        total = int(parts[0])
        used = int(parts[1])
        return total - used
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError) as e:
        logger.warning("nvidia-smi unavailable: %s", e)
        return None


def _resolution_filter(gen_w: int, gen_h: int) -> str | None:
    """Return the ffmpeg video filter to normalize generation res to TARGET_WxTARGET_H.

    - 1920x1088 -> crop to 1920x1080 (drop 8 bottom rows)
    - 1088x608  -> scale to 1920x1080
    - 1920x1080 -> None (already target, no filter needed)
    """
    if gen_w == TARGET_W and gen_h > TARGET_H:
        return f"crop={TARGET_W}:{TARGET_H}:0:0"
    if (gen_w, gen_h) != (TARGET_W, TARGET_H):
        return f"scale={TARGET_W}:{TARGET_H}"
    return None


def _plan_clip_resolution(
    width: int, height: int, two_stage: bool | None,
    max_length_s: int | None = None,
    combined: bool = True,
) -> tuple[int, int, bool, int, int, bool]:
    """Resolve the effective generation resolution for a clip.

    Decides two-stage vs single-stage (auto if two_stage is None), runs the
    VRAM guard, and returns:
        (gen_w, gen_h, use_two_stage, base_width, base_height, text_encoder_cpu)

    `max_length_s` is the run's configured MAX clip length (seconds). It is
    converted to a frame count and budgeted into the guard so a too-long max
    segment is refused (falling back to 1088x608 or aborting) BEFORE any clip
    is generated — this is the Plan 09.9-19 frame-count protection at the
    up-front gate. Pass None to fall back to the legacy resolution-only check.

    text_encoder_cpu tells the caller whether to pin the ~12GB Gemma text
    encoder to CPU (device="cpu") for the clip workflow. It is True when the
    effective resolution crosses the MP threshold OR whenever a VRAM fallback
    occurred (the fallback is a VRAM-constrained degrade and always offloads).
    The guard's estimate is computed with the SAME decision so the budget
    matches real VRAM usage.

    Raises RuntimeError if the VRAM guard fails (no fallback fits) so the
    caller can abort the whole run instead of OOM-ing the 48GB card.
    """
    if two_stage is None:
        use_two_stage = max(width, height) > TWO_STAGE_LONG_EDGE_THRESHOLD
    else:
        use_two_stage = two_stage

    # Frame count for the guard budget: the configured max clip length (so the
    # worst-case clip is what the up-front guard validates against). Legacy
    # callers that pass None omit the frame term entirely.
    guard_frames = _ltx2_guard_num_frames(max_length_s) if max_length_s else 0

    # Offload the ~12GB text encoder to CPU at high resolution (the actual
    # 1080p OOM fix). Budget the guard for the requested-resolution decision.
    text_encoder_cpu = _text_encoder_on_cpu(width, height)
    guard = _vram_guard_check(width, height, use_two_stage, text_encoder_cpu,
                              num_frames=guard_frames, combined=combined)
    if not guard.ok:
        raise RuntimeError(
            f"VRAM guard failed — {guard.reason}. Aborting to avoid OOM on the 48GB card."
        )
    gen_w, gen_h = guard.eff_w, guard.eff_h
    # Honour the guard's approved sampling mode — a single-stage fallback sets
    # this False even when the caller requested two-stage.
    use_two_stage = guard.two_stage
    if use_two_stage:
        base_width, base_height = _two_stage_base(gen_w, gen_h)
    else:
        base_width, base_height = gen_w, gen_h
    # A fallback (effective res != requested) is VRAM-constrained → force CPU
    # encoder to match the guard's fallback estimate; otherwise use the
    # resolution-based decision for the effective res.
    fell_back = (gen_w, gen_h) != (width, height)
    text_encoder_cpu = fell_back or _text_encoder_on_cpu(gen_w, gen_h)
    return gen_w, gen_h, use_two_stage, base_width, base_height, text_encoder_cpu


def _vram_guard_check(
    width: int, height: int, two_stage: bool,
    text_encoder_cpu: bool = False, num_frames: int = 0,
    combined: bool = False,
) -> VramGuardResult:
    """Check if enough VRAM is available for the target resolution + length.

    Falls back to VRAM_FALLBACK_W x VRAM_FALLBACK_H (1088x608, native 16:9)
    if 1080p two-stage doesn't fit. Raises RuntimeError if even the fallback
    doesn't fit — never OOM the 48GB card.

    `num_frames` makes the check frame-count aware (Plan 09.9-19): the budget
    now includes the DiT refine KV-cache cost that scales with generated
    frames, so a long clip at an otherwise-OK resolution is refused (or
    downgraded) instead of OOMing. Pass 0 for the legacy resolution-only check.

    Warm-card detection: if ComfyUI is already running (Stage 4+), the up-front
    estimate uses _activation_headroom_mb() (transient activation only) instead
    of _estimate_vram_mb() (full cold-card peak). The cold-card formula
    double-counts resident ComfyUI VRAM (weights, encoder, VAEs) that nvidia-smi
    "free" already excludes. The headroom check is the same formula used by the
    per-clip gate (check_vram_gate) — battle-tested for warm cards.

    Returns VramGuardResult(ok, eff_w, eff_h, reason, two_stage). eff_w/eff_h
    are the resolution that should actually be generated; two_stage is the
    sampling mode the guard approved (False on a single-stage fallback).
    """
    free = _get_free_vram_mb()
    if free is None:
        logger.warning("nvidia-smi unavailable; skipping VRAM guard")
        return VramGuardResult(True, width, height,
                               "nvidia-smi unavailable; skipping guard", two_stage)

    # Combined base-gen+upscale (Path B) graph: ONE DiT loaded once, refine at 2x
    # target res. The legacy single-stage `_estimate_vram_mb` formula double-counts
    # resident weights + text encoder + MP activation (~59GB for 18s) and would WRONGLY
    # reject a genuinely-fitting long clip (18s fits at ~31GB warm). Route the budget
    # through the WARM-calibrated combined formula (mv-vram-frame-guard debug session):
    # 18s = 18100 + 30*441 + 950 = 32280 MiB (<4% above the 31155 MiB real peak, safe).
    if combined:
        target_w = width * UPSCALE_FACTOR
        target_h = height * UPSCALE_FACTOR
        import mv_comfyui  # lazy to avoid circular import

        comfyui_warm = mv_comfyui._comfyui_is_ready()
        if comfyui_warm:
            needed = _activation_headroom_mb(target_w, target_h, False, num_frames)
            est_label = "activation-only combined (warm card, frame-aware)"
        else:
            needed = _estimate_combined_vram_mb(
                width, height, target_w, target_h, text_encoder_cpu, num_frames,
            )
            est_label = (f"combined graph (text_encoder="
                         f"{'cpu' if text_encoder_cpu else 'gpu'}, frame-aware)")
        if free >= needed + VRAM_SAFETY_MARGIN_MB:
            return VramGuardResult(
                True, width, height,
                f"VRAM OK: {free}MB free, {needed}MB needed ({est_label})",
                two_stage,
            )
        return VramGuardResult(
            False, width, height,
            f"VRAM insufficient (combined): {free}MB free, {needed}MB needed "
            f"(+ {VRAM_SAFETY_MARGIN_MB}MB margin), frames={num_frames}",
            two_stage,
        )

    # Warm-card detection: if ComfyUI is already running, use activation-only
    # estimate. The cold-card formula double-counts resident ComfyUI VRAM.
    import mv_comfyui  # lazy to avoid circular import

    comfyui_warm = mv_comfyui._comfyui_is_ready()
    if comfyui_warm:
        needed = _activation_headroom_mb(width, height, two_stage, num_frames)
        est_label = "activation-only (warm card, frame-aware)"
    else:
        needed = _estimate_vram_mb(width, height, two_stage, text_encoder_cpu,
                                   num_frames)
        est_label = (f"full cold-card (text_encoder="
                     f"{'cpu' if text_encoder_cpu else 'gpu'}, frame-aware)")

    if free >= needed + VRAM_SAFETY_MARGIN_MB:
        return VramGuardResult(True, width, height,
                               f"VRAM OK: {free}MB free, {needed}MB needed ({est_label})",
                               two_stage)

    # Fallback chain: 1920x1088 -> 1088x608 SINGLE-STAGE. REFUSE 1080p rather
    # than approve a config that OOMs. The fallback budgets for the SAME
    # text-encoder decision the clip workflow will make at the fallback res
    # (via _text_encoder_on_cpu) so the estimate matches real usage, AND carries
    # the frame count so a too-long clip is also refused at the fallback res.
    if width == TARGET_W and height == TARGET_H + 8:  # 1920x1088
        # Defense-in-depth: native 1080p is no longer REQUESTED by the pipeline
        # (Path A, Plan 09.9-13 generates low-res + scales to 1080p output), but
        # if anything ever asks for 1920x1088 we still refuse and degrade rather
        # than OOM the 48GB card.
        # The fallback is a VRAM-constrained degrade, so force the ~12GB text
        # encoder onto CPU too (the caller's _plan_clip_resolution mirrors this
        # by setting text_encoder_cpu=True whenever a fallback occurred).
        fb_needed = (_activation_headroom_mb(VRAM_FALLBACK_W, VRAM_FALLBACK_H,
                                            False, num_frames)
                     if comfyui_warm
                     else _estimate_vram_mb(VRAM_FALLBACK_W, VRAM_FALLBACK_H,
                                            False, True, num_frames))
        if free >= fb_needed + VRAM_SAFETY_MARGIN_MB:
            logger.warning(
                "VRAM guard: REFUSING 1080p, falling back to %dx%d SINGLE-STAGE "
                "(1080p needs %dMB, fallback needs %dMB, %dMB free, frames=%d)",
                VRAM_FALLBACK_W, VRAM_FALLBACK_H, needed, fb_needed, free, num_frames,
            )
            return VramGuardResult(
                True, VRAM_FALLBACK_W, VRAM_FALLBACK_H,
                f"fallback to {VRAM_FALLBACK_W}x{VRAM_FALLBACK_H} single-stage",
                False,
            )

    return VramGuardResult(
        False, width, height,
        f"VRAM insufficient: {free}MB free, {needed}MB needed "
        f"(+ {VRAM_SAFETY_MARGIN_MB}MB margin), frames={num_frames}",
        two_stage,
    )
