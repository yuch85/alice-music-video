#!/usr/bin/env python3
"""Per-segment LTX-2 clip generation (Plan 09.9-10).

Extracted from ``mv_clip.py`` per STYLE.md (single responsibility, <=300 LOC).
Owns ``_generate_clip``: build + queue one clip with its retry/VRAM-gate loop.
``mv_clip.py`` re-exports it so callers are unaffected.

ComfyUI helpers route via ``mv_comfyui.<name>`` (patchable); ``build_ltx2_workflow``
is imported lazily so tests can patch ``src.workflow_ltx2.build_ltx2_workflow``.

``_route_segment`` (Plan 09.9-25-01) is the D-02 hybrid dispatcher: it decides
per segment whether to call the existing LTX-2 ``_generate_clip`` or the HuMo 14B
generator (plan 02's ``generate_humo_clip``), and applies the D-01 16s HuMo
default clip duration. HuMo generation itself is delegated — the router only
calls the shared interface; it never queues ComfyUI directly for HuMo.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import mv_audio
import mv_comfyui
import mv_mvconst
import mv_upscale
from mv_black import _generate_black_frame
from mv_segment import ClipSegment

logger = logging.getLogger(__name__)

try:
    from workflows.workflow_ltx2 import build_ltx2_workflow  # noqa: E402  (re-export)
except ImportError:
    build_ltx2_workflow = None  # type: ignore[assignment]

# ── HuMo 14B generation constants (Plan 09.9-25-01, D-01) ──
# Default clip duration for HuMo segments: alias of the canonical 16s default in
# mv_mvconst (single source of truth — NOT a second literal, so the two never drift).
HUMO_DEFAULT_CLIP_DURATION_S = mv_mvconst.DEFAULT_CLIP_DURATION_S
# D-01: safety fallback for drift-prone segments (8s; 6s is the YC floor).
HUMO_FALLBACK_CLIP_DURATION_S = 8
# 16:9, both divisible by 8 (benchmark-approved upscale path).
HUMO_GEN_WIDTH = 848
HUMO_GEN_HEIGHT = 480
# Real-ESRGAN scale-to-fit target (no center-crop → no zoom).
HUMO_UPSCALED_WIDTH = 1920
HUMO_UPSCALED_HEIGHT = 1080
# Deterministic default seed (run-8 validated).
HUMO_DEFAULT_SEED = 42


def _generate_clip(
    segment: ClipSegment,
    scene_prompt: str,
    reference_portrait: Path,
    clip_index: int,
    output_dir: Path,
    shot_type: str = "singer",
    segment_prompt: str | None = None,
    vocals_stem: Path | None = None,
    gen_w: int = 1920,
    gen_h: int = 1088,
    use_two_stage: bool = False,
    base_width: int = 1920,
    base_height: int = 1088,
    text_encoder_device: str | None = None, upscale: bool = True,
    target_w: int | None = None, target_h: int | None = None,
    original_audio: Path | None = None,
    use_lipdub: bool = True,
    duration_s: int | None = None,
    ltx_motion_prompt: str = "",
    reference_image_2: str = "",
    use_vrdg_sigmas: bool = False,
) -> Path | None:
    """Generate one LTX-2 clip for a segment.

    Args:
        segment: ClipSegment with timing and lyrics.
        scene_prompt: Base scene description.
        reference_portrait: Reference portrait path.
        clip_index: 1-based clip index.
        output_dir: Output directory.
        shot_type: "singer" | "broll" | "instrumental" | "black".
        segment_prompt: Per-segment prompt override (controlled mode).
        vocals_stem: Vocals stem WAV for Audio VAE conditioning (None = off).
        gen_w, gen_h: Effective generation resolution (planned ONCE by caller).
        use_two_stage: Whether to use two-stage sampling.
        base_width, base_height: Two-stage base resolution.
        use_lipdub: False => clean LTX identity (Lipdub OFF, node "1a" omitted),
            decoupling identity from the separate HUMO lip-sync model (09.9-20).
            Default True keeps production Lipdub ON.
        duration_s: Per-clip duration override (seconds). When provided, wins
            over the segment's own duration and the LTX-2 4-18s clamp. None
            (default) keeps the existing segment-derived clamp (backward
            compatible; uses this only when the orchestrator passes an explicit
            per-clip duration, e.g. from 09.9-25-04's clip_durations wiring).

    Returns the output MP4 path, or None if generation failed/skipped.
    """
    from mv_clip import (  # lazy import: avoids a top-level cycle with mv_clip
        LTX2_MAX_LENGTH_S, LTX2_MIN_LENGTH_S, _energy_motion_phrase,
    )

    if shot_type == "black":
        return _generate_black_frame(segment.duration, clip_index, output_dir,
                                     width=gen_w, height=gen_h)

    prompt = segment_prompt or scene_prompt

    # Resolve effective duration. An explicit per-clip override wins; otherwise
    # the segment's own duration is clamped to the LTX-2 range (4-18s).
    if duration_s is not None:
        length_s = int(round(duration_s))
        logger.info("Segment %d: per-clip duration override %ds applied",
                    clip_index, length_s)
    else:
        length_s = int(round(segment.duration))
        if length_s > LTX2_MAX_LENGTH_S:
            length_s = LTX2_MAX_LENGTH_S
            logger.info("Segment %d: duration %.1fs exceeds LTX-2 max (%ds), clamping",
                        clip_index, segment.duration, LTX2_MAX_LENGTH_S)
        elif length_s < LTX2_MIN_LENGTH_S:
            length_s = LTX2_MIN_LENGTH_S
            logger.info("Segment %d: duration %.1fs below LTX-2 min (%ds), extending",
                        clip_index, segment.duration, LTX2_MIN_LENGTH_S)

    # Pre-roll/tail-loss padding: generate extra frames, trim after generation
    padded_length_s = length_s + (mv_comfyui.PRE_ROLL_FRAMES + mv_comfyui.TAIL_LOSS_FRAMES) / 24.0
    logger.info("Clip %d: generating %ds + %d pre-roll + %d tail-loss (padded %.1fs)",
                clip_index, length_s, mv_comfyui.PRE_ROLL_FRAMES,
                mv_comfyui.TAIL_LOSS_FRAMES, padded_length_s)

    # Crop audio segment for Audio VAE conditioning (if vocals stem present)
    audio_path: str | None = None
    audio_file_basename: str | None = None
    cropped_audio_path: Path | None = None
    if vocals_stem is not None and vocals_stem.exists():
        audio_file = mv_audio._crop_audio_segment(
            vocals_stem, segment.start, segment.end, output_dir, clip_index)
        if audio_file.exists():
            audio_file_basename = audio_file.name
            audio_path = audio_file_basename
            cropped_audio_path = audio_file
            logger.info("Clip %d: audio conditioning with %s", clip_index, audio_file)

    # F7 (S9): explicit motion prompt takes precedence over the weak
    # energy-conditioned phrase (LTX-2 is I2V and barely moves without it).
    # GATED to LTX-2 (HuMo uses audio conditioning, never a motion prompt).
    if ltx_motion_prompt:
        prompt = f"{prompt.rstrip('. ')}. {ltx_motion_prompt}"
        logger.info("Clip %d: explicit LTX motion prompt: %s", clip_index, ltx_motion_prompt)
    else:
        energy_phrase = _energy_motion_phrase(cropped_audio_path)
        if energy_phrase:
            prompt = f"{prompt.rstrip('. ')}. {energy_phrase}"
            logger.info("Clip %d: energy-conditioned motion: %s", clip_index, energy_phrase)

    # Copy original audio to ComfyUI input/ for CreateVideo output (Option B)
    output_audio_basename: str | None = None
    output_audio_path: Path | None = None
    if original_audio is not None and original_audio.exists():
        comfyui_input = Path(mv_comfyui.COMFYUI_OUTPUT_DIR) / "input"
        comfyui_input.mkdir(parents=True, exist_ok=True)
        output_audio_basename = f"orig_audio_{clip_index:03d}_{int(time.time())}_{original_audio.name}"
        output_audio_path = comfyui_input / output_audio_basename
        shutil.copy2(original_audio, output_audio_path)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu-manager"))
    # Path B: one combined base-gen+upscale job (Plan 09.9-16, latent→latent).
    # F5 (S7-1088): LTX-2 base res MUST be divisible by 32 (VAE spatial
    # downsample), so base_h stays 544 (=17*32) and target_h = 2*544 = 1088.
    # The per-clip intermediate is 1920x1088; the ffmpeg composite crops it to
    # 1920x1080 via _resolution_filter(1920,1088)="crop=1920:1080:0:0"
    # (mv_vram.py), so the SHIPPED final_output.mp4 is exactly 1080. Setting
    # base_h=540 (to force target_h=1080) would break the LTX-2 VAE (540 is not
    # /32) — do NOT do that. 1088 is an intermediate-only dimension, already
    # corrected at the composite boundary.
    tw = target_w or 1920
    th = target_h or 1088
    use_combined = upscale and mv_upscale._upscale_model_present()
    base_w2, base_h2 = tw // 2, th // 2

    # Decide text-encoder placement from the PLANNED generation resolution
    from mv_vram import (
        _text_encoder_on_cpu, _activation_headroom_mb, _ltx2_guard_num_frames,
    )
    if text_encoder_device is None:
        text_encoder_device = "cpu" if _text_encoder_on_cpu(gen_w, gen_h) else "default"
    # Plan 09.9-19 frame-guard: free-VRAM floor scales with the padded frame count
    clip_frames = _ltx2_guard_num_frames(padded_length_s)
    if use_combined:
        # Calculate headroom on base resolution, not target.
        # Two-stage generates at base_w2/base_h2 then upscales the latent.
        # Peak VRAM is at the base-resolution KSampler activation, not the
        # full-resolution target. Using tw/th (1920x1088) overestimates by ~48%
        # and spuriously fails the gate when ComfyUI's own model loading
        # leaves ~9GB free (enough for 7.4GB base, not 10.8GB target).
        clip_min_free_mb = _activation_headroom_mb(base_w2, base_h2, True, clip_frames)
    else:
        clip_min_free_mb = _activation_headroom_mb(gen_w, gen_h, use_two_stage, clip_frames)
    logger.info("Clip %d: text_encoder=%s (gen %dx%d), free floor=%dMB",
                clip_index, text_encoder_device, gen_w, gen_h, clip_min_free_mb)

    MAX_CLIP_RETRIES = 2
    RETRY_BACKOFF_S = 30
    dest = None
    ref_name = None
    ref_path = None
    dialogue_text = ""

    for attempt in range(1 + MAX_CLIP_RETRIES):
        if attempt > 0:
            logger.info("Clip %d: clearing ComfyUI state before retry (attempt %d)",
                        clip_index, attempt + 1)
            mv_comfyui._reset_comfyui_state()

        if not mv_comfyui._check_vram_gate(min_free_mb=clip_min_free_mb):
            if attempt < MAX_CLIP_RETRIES:
                logger.warning("Clip %d: VRAM gate failed (attempt %d/%d), waiting %ds",
                               clip_index, attempt + 1, 1 + MAX_CLIP_RETRIES, RETRY_BACKOFF_S)
                time.sleep(RETRY_BACKOFF_S)
                continue
            logger.warning("Skipping segment %d — VRAM gate failed after %d attempts",
                           clip_index, 1 + MAX_CLIP_RETRIES)
            return None

        if ref_path is not None:
            try:
                ref_path.unlink()
            except OSError:
                pass
            ref_path = None
            ref_name = None

        if shot_type in ("singer", "instrumental") and reference_portrait.exists():
            comfyui_input = Path(mv_comfyui.COMFYUI_OUTPUT_DIR) / "input"
            comfyui_input.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            ref_name = f"clip_{clip_index:03d}_ref_{ts}.jpg"
            ref_path = comfyui_input / ref_name
            shutil.copy2(reference_portrait, ref_path)

        if shot_type == "singer":
            dialogue_text = segment.text

        try:
            prefix = f"clip_{clip_index:03d}"
            workflow, out_prefix, dest_suffix, clip_timeout = mv_upscale.build_clip_workflow(
                use_combined=use_combined, prompt=prompt, dialogue_text=dialogue_text,
                ref_name=ref_name, padded_length_s=padded_length_s, audio_path=audio_path,
                tw=tw, th=th, base_w=base_w2, base_h=base_h2, gen_w=gen_w, gen_h=gen_h,
                use_two_stage=use_two_stage, base_width=base_width, base_height=base_height,
                text_encoder_device=text_encoder_device, neg_suffix="",
                use_lipdub=use_lipdub,
                output_audio_filename=output_audio_basename,
                use_vrdg_sigmas=use_vrdg_sigmas,
            )

            prompt_id = mv_comfyui._queue_workflow(workflow)
            logger.info("Clip %d queued (attempt %d/%d, prompt_id=%s, len=%.1fs, %s, %s, audio=%s)",
                        clip_index, attempt + 1, 1 + MAX_CLIP_RETRIES, prompt_id,
                        padded_length_s, shot_type, "B-combined" if use_combined else "A-base",
                        audio_path or "none")
            history = mv_comfyui._poll_completion(prompt_id, timeout=clip_timeout)

            output_path = mv_comfyui._find_output_file(history, out_prefix, "mp4")
            dest = output_dir / "clips" / f"{prefix}{dest_suffix}"
            shutil.copy2(output_path, dest)

            if mv_comfyui.PRE_ROLL_FRAMES > 0 or mv_comfyui.TAIL_LOSS_FRAMES > 0:
                trimmed = output_dir / "clips" / f"{prefix}_trimmed.mp4"
                if mv_audio._trim_padding_frames(dest, trimmed):
                    dest.unlink()
                    shutil.move(trimmed, dest)
                    logger.info("Clip %d trimmed: %s", clip_index, dest)
                elif trimmed.exists():
                    trimmed.unlink()

            logger.info("Clip %d saved: %s (shot_type=%s, attempt %d/%d)",
                        clip_index, dest, shot_type, attempt + 1, 1 + MAX_CLIP_RETRIES)

            if mv_comfyui.comfyui_client._consecutive_failures > 0:
                logger.info("Clip %d succeeded — resetting circuit breaker (was %d)",
                            clip_index, mv_comfyui.comfyui_client._consecutive_failures)
                mv_comfyui.comfyui_client._consecutive_failures = 0

            try:
                if ref_path is not None:
                    ref_path.unlink()
            except OSError:
                pass
            if audio_file_basename is not None:
                try:
                    (Path(mv_comfyui.COMFYUI_OUTPUT_DIR) / "input" / audio_file_basename).unlink()
                except OSError:
                    pass
            if output_audio_path is not None:
                try:
                    output_audio_path.unlink()
                except OSError:
                    pass
            return dest

        except Exception as e:
            logger.error("Clip %d generation failed (attempt %d/%d): %s",
                         clip_index, attempt + 1, 1 + MAX_CLIP_RETRIES, e)
            if attempt < MAX_CLIP_RETRIES:
                is_conn = any(kw in str(e).lower() for kw in [
                    "connection refused", "urlopen", "broken pipe",
                    "connection reset", "connection closed"])
                logger.warning("Clip %d: %s error — backing off %ds before retry",
                               clip_index, "ComfyUI connection" if is_conn else "non-connection",
                               RETRY_BACKOFF_S)
                time.sleep(RETRY_BACKOFF_S)
            else:
                logger.error("Clip %d: all %d attempts exhausted — skipping segment",
                             clip_index, 1 + MAX_CLIP_RETRIES)

    if ref_path is not None:
        try:
            ref_path.unlink()
        except OSError:
            pass
    if output_audio_path is not None:
        try:
            output_audio_path.unlink()
        except OSError:
            pass
    return dest


def _route_segment(
    segment: "ClipSegment",
    scene_prompt: str,
    reference_portrait: Path,
    clip_index: int,
    output_dir: Path,
    *,
    vocal_presence: bool,
    has_lyrics: bool,
    segment_prompt: str | None = None,
    vocals_stem: Path | None = None,
    original_audio: Path | None = None,
    duration_s: int | None = None,
    force_engine: "EngineChoice | None" = None,
    ltx_motion_prompt: str = "",
    reference_image_2: str = "",
    skip_upscale: bool = False,
    bg_plate_path: "Path | None" = None,
    use_vrdg_sigmas: bool = False,
    use_lipdub: bool = True,
    cascade_start: float | None = None,
    cascade_end: float | None = None,
) -> "Path | None":
    """Route one segment to HuMo 14B or the existing LTX-2 path (D-02 hybrid).

    This is the single dispatch point added by Plan 09.9-25-01. It decides the
    engine per segment and applies the D-01 16s HuMo default clip duration. It
    performs NO video generation itself — HuMo segments are delegated to plan 02's
    ``generate_humo_clip`` (which routes via slingshot/exec); LTX-2 segments go to
    the existing ``_generate_clip``. The router never queues ComfyUI directly for
    HuMo (threat T-09.9-25-01: no silent fallback — crop failures raise instead).

    Args:
        segment: ClipSegment with timing/lyrics.
        scene_prompt: Base scene description.
        reference_portrait: Reference portrait path (HuMo + LTX-2 singer).
        clip_index: 1-based clip index.
        output_dir: Output directory.
        vocal_presence: Whether the segment has detected vocal presence.
        has_lyrics: Whether the segment carries lyrics.
        segment_prompt: Per-segment prompt override (controlled mode).
        vocals_stem: Vocals stem WAV for audio conditioning.
        original_audio: Full original audio for LTX-2 mux.
        duration_s: Per-clip duration override (seconds). For HuMo, wins over the
            D-01 16s default; for LTX-2, wins over the segment's own duration and
            the 4-18s clamp. None = engine default.
        force_engine: When set, overrides ``classify_segment_engine`` (D-02
            extension) — e.g. force LTX-2 B-roll even on lyric-bearing segments.
        ltx_motion_prompt: Explicit LTX-2 motion direction (F7/S9); overrides the
            energy-conditioned phrase. GATED to LTX-2 (HuMo ignores it).
        reference_image_2: Optional second LTX-2 reference image for variety.

    Returns the output clip Path, or None if generation was skipped/failed.
    """
    # Lazy import avoids a top-level cycle (mv_clip re-exports _generate_clip
    # from this module, so importing mv_clip at load time would recurse).
    from mv_clip import EngineChoice, classify_segment_engine  # noqa: E402  (D-02)

    # 1. Resolve engine: explicit override beats the classifier.
    if force_engine is not None:
        engine = force_engine
    else:
        engine = classify_segment_engine(segment, vocal_presence, has_lyrics)
    logger.info("Segment %d: routing engine=%s (vocal=%s, lyrics=%s, force=%s)",
                clip_index, engine, vocal_presence, has_lyrics, force_engine)

    # Cascade timing override: when cascade_start/cascade_end are provided,
    # create a copy of the segment with overridden timing for audio cropping.
    # This ensures conditioning_audio == assembly_audio (Invariant 3).
    effective_segment = segment
    if cascade_start is not None and cascade_end is not None:
        effective_segment = dataclasses.replace(
            segment,
            start=cascade_start,
            end=cascade_end,
            duration=cascade_end - cascade_start,
        )
        logger.info(
            "Segment %d: cascade timing override [%.3f, %.3f) "
            "(plan: [%.3f, %.3f))",
            clip_index, cascade_start, cascade_end,
            segment.start, segment.end,
        )

   # 2. LTX-2 path: delegate to the existing generator unchanged.
    if engine == "ltx2":
        return _generate_clip(
            effective_segment, scene_prompt, reference_portrait, clip_index, output_dir,
            segment_prompt=segment_prompt, vocals_stem=vocals_stem,
            original_audio=original_audio, duration_s=duration_s,
            ltx_motion_prompt=ltx_motion_prompt,
            reference_image_2=reference_image_2,
            use_vrdg_sigmas=use_vrdg_sigmas,
            use_lipdub=use_lipdub,
        )

    # 3. HuMo path: crop the audio segment, then delegate to generate_humo_clip.
    #    An explicit per-clip duration wins; otherwise the D-01 16s default applies.
    #    F1 (S3): clip length is bounded by the vocal-slice window so HuMo does
    #    not animate a long silent tail (idle/loop mouth artifact). The composite
    #    overlay uses eof_action=pass (mv_post.py) so a shorter clip shows black
    #    until the next clip's audio start instead of freeze-framing.
    humo_duration_s = min(duration_s if duration_s is not None else HUMO_DEFAULT_CLIP_DURATION_S,
                          int(round(effective_segment.end - effective_segment.start)))

    # Crop the audio segment for HuMo. A missing vocals stem is a routing error,
    # not a silent fallback to LTX-2 (T-09.9-25-01). The stem precondition is
    # `vocals_stem is not None`; whether the crop itself produces a file is
    # validated after the call.
    if vocals_stem is None:
        raise RuntimeError(
            f"Segment {clip_index}: HuMo routing requires a vocals stem; got None"
        )
    cropped_audio_path = mv_audio._crop_audio_segment(
        vocals_stem, effective_segment.start, effective_segment.end, output_dir, clip_index,
    )
    if not cropped_audio_path.exists():
        raise RuntimeError(
            f"Segment {clip_index}: audio crop for HuMo failed at {cropped_audio_path}"
        )

    humo_prompt = segment_prompt or scene_prompt

    # Lazy import of the shared contract so plan-02 code need not exist yet in
    # Wave 1 — DO NOT rename generate_humo_clip.
    from mv_humo_gen import generate_humo_clip

    return generate_humo_clip(
        audio_segment_wav=cropped_audio_path,
        reference_path=reference_portrait,
        prompt=humo_prompt,
        output_dir=output_dir,
        clip_index=clip_index,
        duration_s=humo_duration_s,
        width=HUMO_GEN_WIDTH,
        height=HUMO_GEN_HEIGHT,
        seed=HUMO_DEFAULT_SEED,
        skip_upscale=skip_upscale,
        bg_plate_path=bg_plate_path,
    )


# LTX-2 frame rate constant (24fps throughout pipeline)
_LTX2_FRAME_RATE = 24


def _write_clip_manifest(
    *,
    clip_index: int,
    clip_path: Path,
    segment: ClipSegment,
    cascade_position: float,
    measured_duration: float,
    frame_count: int,
    scene_prompt: str,
    reference_portrait: Path,
    shot_type: str,
    output_dir: Path,
    seed: int | None = None,
) -> Path:
    """Write a JSON manifest for a generated clip.

    The manifest records cascade positions, measured durations, and generation
    parameters so downstream assembly can reuse them without re-probing.
    Written immediately after generation (before the next clip starts).

    Args:
        clip_index: 1-based clip index.
        clip_path: Path to the generated clip MP4.
        segment: Original ClipSegment (plan timing, not cascade override).
        cascade_position: Where this clip starts in the cascade timeline.
        measured_duration: Actual video duration from ffprobe.
        frame_count: Number of frames in the generated video.
        scene_prompt: Scene description used for generation.
        reference_portrait: Reference portrait path used.
        shot_type: Shot type string (e.g. "singer", "instrumental").
        output_dir: Output directory (manifest written to clips/ subfolder).
        seed: Random seed used for generation (optional).

    Returns:
        Path to the written manifest JSON file.
    """
    manifest = {
        "clip_index": clip_index,
        "segment_duration": round(segment.duration, 3),
        "measured_duration": round(measured_duration, 3),
        "frame_count": frame_count,
        "fps": _LTX2_FRAME_RATE,
        "conditioning_audio_start": round(cascade_position, 3),
        "conditioning_audio_end": round(cascade_position + segment.duration, 3),
        "cascade_position": round(cascade_position, 3),
        "cascade_next": round(cascade_position + measured_duration, 3),
        "generation_parameters": {
            "engine": "ltx2",
            "model_version": "ltx-2.3",
            "seed": seed if seed is not None else 42,
            "steps": 9,
            "scheduler": "euler",
        },
        "ref_image_path": str(reference_portrait),
        "shot_type": shot_type,
        "lyric_text": segment.text,
        "clip_path": str(clip_path),
    }

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = clips_dir / f"clip_{clip_index:03d}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Clip %d manifest written: %s", clip_index, manifest_path)
    return manifest_path
