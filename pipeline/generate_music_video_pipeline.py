#!/usr/bin/env python3
"""Automated LTX Music Video Pipeline Orchestrator.

End-to-end pipeline:
  Stage 1 — Demucs stem separation (MTV-04)
  Stage 2 — Whisper beat/lyric alignment (MTV-05)
  Stage 3 — Scene-consistent portrait generation (MTV-06)
  Stage 4 — LTX-2 clip generation per segment (MTV-06)
  Stage 5 — FFmpeg compositing (MTV-07)

VRAM safety via gpu-manager ensure_service_ready gate (MTV-08).

This module is a THIN FACADE / entrypoint (Plan 09.9-10): it preserves the
CLI and re-exports every original top-level symbol so existing importers
and the test `from ... import` form keep resolving. All logic lives in the
`mv_*.py` modules.

Usage:
  python generate_music_video_pipeline.py \
      --input <audio> \
      --output ./output \
      --portrait <portrait> \
      --scene-prompt "singer on stage, concert lighting" \
      [--max-segment-s 10] [--dry-run]

  # Generic reusable invocation (Plan 09.9-25-04, D-06): render ANY
  # song with a per-clip prompt list, a single 16:9 portrait (or per-clip
  # references), an explicit vocals stem, variable per-clip durations, and an
  # optional per-clip engine override:
  python generate_music_video_pipeline.py \
      --input <audio> \
      --output ./output \
      --portrait <portrait> \
      --audio-stem <vocals.wav> \
      --prompts "scene A" --prompts "scene B" --prompts "scene C" \
      --clip-duration-s 16 \
      --clip-durations 16 --clip-durations 8 --clip-durations 12 \
      --per-clip-engines humo --per-clip-engines ltx2 --per-clip-engines humo
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

# ── CUDA library path (must be set before any GPU imports) ─────────
# onnxruntime-gpu ships nvidia CUDA libs (cuBLAS, cuDNN, etc.) as
# site-packages/nvidia/cu*/lib/ but these aren't on the system ldconfig path.
# Set LD_LIBRARY_PATH so faster-whisper + onnxruntime-gpu can find them.
# Demucs (Stage 1) uses its own venv's CUDA libs — it runs as a subprocess
# with LD_LIBRARY_PATH cleared (see _run_demucs_separation).
# CUDA libs: derive from THIS interpreter's own site-packages so discovery is
# portable (no hardcoded venv paths). ctranslate2 dlopen's libcublas at runtime
# and needs LD_LIBRARY_PATH because its rpath does not reference nvidia wheels,
# and there is no system CUDA on this box.
#
# NOTE (Finding 1): glibc caches LD_LIBRARY_PATH at process STARTUP, so this
# in-process patch is a best-effort FALLBACK only — it cannot fix the
# `libcublas.so.12 not found` error when the process launched without the
# correct value. The AUTHORITATIVE fix is the launch-time LD_LIBRARY_PATH
# injection in gpu-manager's music-video MCP tools (uv venv nvidia lib dirs
# passed via the subprocess env=), which lets the fresh python process cache
# the correct search paths at startup.
import site as _site, sysconfig as _sysconfig
_nv_root = Path(_sysconfig.get_path("platlib")) / "nvidia"
_seen_libs = set()
_nv_lib_components = []
for _cu_dir in sorted(_nv_root.glob("*/lib")):
    _resolved = str(_cu_dir.resolve())
    if not _cu_dir.is_dir() or _resolved in _seen_libs:
        continue
    _seen_libs.add(_resolved)
    _nv_lib_components.append(_resolved)

# Build LD_LIBRARY_PATH from a component list so matching is by path
# component (no substring false-skip on lib/ vs lib64/ prefixes) and join
# with ':' (no trailing colon, which would make the linker search cwd).
# Our dirs are prepended ahead of any pre-existing entries.
_existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
_existing_set = set(_existing)
_nv_lib_components = [p for p in _nv_lib_components if p not in _existing_set]
os.environ["LD_LIBRARY_PATH"] = ":".join(_nv_lib_components + _existing)

# ── Logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Re-exports from mv_* modules (Plan 09.9-10 facade) ──────────────

from mv_segment import (
    WordSegment, ClipSegment, ShotType, SegmentPlan,
    _group_words_into_segments, _flush_segment, _write_segment_plan, _read_segment_plan,
    _fill_coverage_gaps, _split_long_segments, _validate_segment_plan,
    _classify_vocal_regions, _merge_short_segments,
)
from mv_audio import (
    _run_demucs_separation, _transcribe_with_whisper, _crop_audio_segment,
    _trim_padding_frames, _find_existing_vocals, _load_existing_transcript,
    _get_audio_duration,
    DEMUCS_BIN, WHISPER_MODEL_ID,
)
from mv_lut import (
    _download_default_lut,
    POST_PROCESS_LUT_DIR,
    DEFAULT_LUT_NAME,
    DEFAULT_GRAIN_INTENSITY,
    DEFAULT_SHARPEN_STRENGTH,
)
from mv_post import (
    _apply_post_processing,
    _composite_with_ffmpeg,
    _composite_timeline_canvas,
)
from mv_recovery import reconcile_plan_with_audio
from mv_vram import (
    VramGuardResult, _two_stage_base, _plan_clip_resolution,
    _get_free_vram_mb, _estimate_vram_mb, _vram_guard_check, _resolution_filter,
    TWO_STAGE_LONG_EDGE_THRESHOLD, VRAM_SAFETY_MARGIN_MB, TWO_STAGE_VRAM_OVERHEAD_FRAC,
    VRAM_FALLBACK_W, VRAM_FALLBACK_H, TARGET_W, TARGET_H, LTX2_VRAM_MB, I2I_VRAM_MB,
)
from mv_comfyui import (
    ComfyUIClient, _comfyui_post, _comfyui_get, _comfyui_is_ready,
    _queue_workflow, _poll_completion, _find_output_file, _validate_and_migrate_ltx2_models,
    _start_comfyui_via_gpu_manager, _wait_for_comfyui_ready, _check_vram_gate,
    _reset_comfyui_state, PRE_ROLL_FRAMES, TAIL_LOSS_FRAMES, MAX_CONSECUTIVE_COMFYUI_FAILURES,
    COMFYUI_HOST, COMFYUI_PORT, COMFYUI_BASE, COMFYUI_OUTPUT_DIR,
)
import mv_comfyui
from mv_slingshot import (
    SlingshotClient, _atexit_wake, _sigterm_handler, _register_slingshot_recovery,
    _RECOVERY_SLINGSHOT, GPU_MANAGER_HOST, GPU_MANAGER_PORT, GPU_MANAGER_BASE,
    is_local_llm_active,
)
from mv_prompt import (
    _load_creative_inputs, _get_local_llm_endpoint, _refine_prompts,
    _refine_prompts_with_prewrite,
    _build_broll_prompts,
    _T2I_SYSTEM_PROMPT, _I2V_SYSTEM_PROMPT,
)
from mv_clip import (
    _generate_clip, _generate_scene_portrait,
    build_ltx2_workflow, NEG_SUFFIX_6TERM, LTX2_MAX_LENGTH_S, LTX2_MIN_LENGTH_S,
)
from mv_black import _generate_black_frame
from mv_shot import (
    _get_pose_variation, _cycle_motion_templates, _build_motion_prompt, _assign_shot_type,
    POSE_VARIATIONS, CAMERA_MOTION_TEMPLATES, CHARACTER_ACTION_TEMPLATES, ENERGY_TEMPLATES,
)
from mv_refs import (
    _generate_segment_ref, _generate_segment_refs, _write_references_manifest,
    _generate_approval_dashboard,
)
from mv_clip_generate import _route_segment, _write_clip_manifest

import mv_mvconst



def _get_clip_duration(clip_path: Path) -> float:
    """Return video clip duration in seconds via ffprobe.

    Thin wrapper around ``_get_audio_duration`` — ffprobe works for both
    audio and video files. Used by the cascade loop to measure actual
    LTX-2 output duration after 8k+1 frame quantization.

    Args:
        clip_path: Path to the generated MP4 clip.

    Returns:
        Duration in seconds, or 0.0 on failure.
    """
    return _get_audio_duration(clip_path)


# LTX-2 frame quantization bound: stride=8 frames at 24fps = 333ms, half = 167ms
LTX2_QUANTIZATION_BOUND_S = 8.0 / 24.0  # ~0.167s


def _validate_parametric_inputs(
    segments: "list[Any]",
    prompts: "list[str] | None",
    per_clip_references: "list[Path] | None",
    clip_duration_s: int,
    clip_durations: "list[int] | None",
    per_clip_engines: "list[str] | None",
) -> None:
    """Fail-fast validation of the generic parametric surface (D-01/D-02/D-04/D-05).

    Must be called AFTER segment augmentation (coverage-fill + split) so the
    effective segment count matches how clips are generated. Raises ``ValueError``
    with an explicit message on any length/floor/ceiling/value mismatch. No bare
    except — every branch names the offending parameter (threat T-09.9-04-02:
    prompt/reference length mismatch must abort before any GPU spend).

    Args:
        segments: Effective (augmented) segment list.
        prompts: Optional per-segment prompt list.
        per_clip_references: Optional per-segment reference list.
        clip_duration_s: Global default clip duration (rejects below floor).
        clip_durations: Optional per-clip duration list.
        per_clip_engines: Optional per-segment engine override list.
    """
    seg_count = len(segments)

    if prompts is not None and len(prompts) != seg_count:
        raise ValueError(
            f"prompts length {len(prompts)} != segment count {seg_count}"
        )

    if per_clip_references is not None and len(per_clip_references) != seg_count:
        raise ValueError(
            f"per_clip_references length {len(per_clip_references)} != segment count {seg_count}"
        )

    if clip_duration_s is not None and clip_duration_s < mv_mvconst.CLIP_DURATION_FLOOR_S:
        raise ValueError(
            f"clip_duration_s {clip_duration_s} below floor {mv_mvconst.CLIP_DURATION_FLOOR_S}"
        )

    if clip_durations is not None:
        if len(clip_durations) != seg_count:
            raise ValueError(
                f"clip_durations length {len(clip_durations)} != segment count {seg_count}"
            )
        for d in clip_durations:
            if d < mv_mvconst.CLIP_DURATION_FLOOR_S or d > mv_mvconst.CLIP_DURATION_CEILING_S:
                raise ValueError(
                    f"clip_durations entry {d} outside "
                    f"[{mv_mvconst.CLIP_DURATION_FLOOR_S}, {mv_mvconst.CLIP_DURATION_CEILING_S}]"
                )

    if per_clip_engines is not None:
        if len(per_clip_engines) != seg_count:
            raise ValueError(
                f"per_clip_engines length {len(per_clip_engines)} != segment count {seg_count}"
            )
        for e in per_clip_engines:
            if e not in ("humo", "ltx2"):
                raise ValueError(
                    f"per_clip_engines entry {e!r} not in {{'humo','ltx2'}}"
                )


def run_pipeline(
    input_audio: str,
    output_dir: str,
    portrait: str,
    scene_prompt: str,
    max_segment_s: float = 10.0,
    dry_run: bool = False,
    mode: str = "auto",
    resume: bool = False,
    storyconcept_path: str | None = None,
    themestyle_path: str | None = None,
    subjectsandscenes_path: str | None = None,
    lyrics_path: str | None = None,
    lut_path: str | None = None,
    grain_intensity: float = DEFAULT_GRAIN_INTENSITY,
    sharpen_strength: float = DEFAULT_SHARPEN_STRENGTH,
    skip_post_process: bool = False,
    width: int = mv_mvconst.DEFAULT_LOWRES_W,
    height: int = mv_mvconst.DEFAULT_LOWRES_H,
    max_clip_s: int = mv_mvconst.DEFAULT_MAX_CLIP_S,
    two_stage: bool | None = None,
    upscale: bool = True,
    apply_lut: bool = False,
    pre_roll_frames: int = PRE_ROLL_FRAMES,
    use_lipdub: bool = True,
    # ── Plan 09.9-25-04 parametrization (D-01/D-04/D-05/D-06) ──
    # Generic, reusable surface so the pipeline can render ANY song/reference,
    # not just a hardcoded one. No song/reference names appear anywhere here.
    audio_stem: Path | None = None,
    prompts: list[str] | None = None,
    portrait_path: Path | None = None,
    per_clip_references: list[Path] | None = None,
    clip_duration_s: int | None = None,
    clip_durations: list[int] | None = None,
    per_clip_engines: list[str] | None = None,
    output_path: Path | None = None,
    # ── Plan 09.9-26 (F7 / S9): LTX-2 explicit motion prompt ──
    # LTX-2 is I2V (strength=1.0) and barely moves without an explicit motion
    # direction. This overrides the weak energy-conditioned phrase so b-roll /
    # instrumental clips show real movement. GATED to LTX-2 only (HuMo uses
    # audio conditioning, not a motion prompt). reference_image_2 enables a
    # second reference for variety (optional, LTX-2 only).
    ltx_motion_prompt: str = "",
    reference_image_2: str = "",
    # ── Pre-written prompts (pre-written prompt support) ──
    prompts_file: Path | None = None,
    # ── Pilot mode (pilot mode) ──
    max_clips: int | None = None,
) -> dict[str, Any]:
    """Execute the full music video pipeline.

    Args:
        input_audio: Path to input audio file.
        output_dir: Output directory for stems, clips, final video.
        portrait: Reference portrait path (legacy alias for portrait_path).
        scene_prompt: Visual scene description for LTX-2 (legacy alias for the
            single global prompt; superseded by ``prompts`` when per-clip prompts
            are supplied).
        max_segment_s: Max segment length in seconds.
        dry_run: Print plan without executing GPU work.
        mode: "auto" (default, fully automated) or "controlled" (per-segment refs + approval gate).
        resume: Resume from segment_plan.json (skip stages 1-3, generate clips + composite).
        storyconcept_path: Optional path to story concept text file.
        themestyle_path: Optional path to theme/style text file.
        subjectsandscenes_path: Optional path to subjects/scenes text file.
        lyrics_path: Optional path to lyrics text file.
        lut_path: Optional path to .cube LUT file for color grading.
        grain_intensity: Film grain intensity 0.0-10.0 (default: 0.5).
        sharpen_strength: Sharpening strength 0.0-1.5 (default: 0.4).
        skip_post_process: Skip post-processing chain entirely.
        width: Base generation width (default: 960 low-res; 2x -> 1080p output).
        height: Base generation height (default: 544 low-res; 2x -> 1080p output).
        max_clip_s: Max single-clip duration in seconds (default: 6) for VRAM
            headroom; longer segments are split to <= this many seconds.
        two_stage: Force two-stage (True), force single-stage (False), or auto (None).
        apply_lut: Enable the Cine Grade 3D LUT + color mixer. Off by default so
            output matches the 09.9-16 reference clips (neutral grade, no tint).
        pre_roll_frames: Pre-roll frames trimmed from each clip; the final audio
            is delayed by the same amount to keep lips aligned (see
            mv_comfyui.PRE_ROLL_FRAMES).
        use_lipdub: When False, LTX clips carry a CLEAN identity (Lipdub OFF,
            node "1a" omitted) so identity is decoupled from the separate HUMO
            lip-sync model (09.9-20). Default True keeps production Lipdub ON.
        audio_stem: Explicit full vocals stem (Demucs-separated). When None, the
            pipeline derives vocals via Demucs as today.
        prompts: Per-segment scene prompts aligned to segment order (D-05). When
            None, the single ``scene_prompt`` is replicated across segments.
        portrait_path: Single 16:9 reference (D-04 default). Supersedes the
            legacy ``portrait`` string param.
        per_clip_references: Optional per-segment references aligned to segment
            order (D-04 option). When provided, length MUST match segment count,
            and ``references_manifest.json`` is emitted for manual QA.
        clip_duration_s: Global default clip duration in seconds (D-01); used for
            every clip unless ``clip_durations`` overrides per clip. Default 16.
        clip_durations: Optional per-clip duration list (variable clip duration),
            indexed by segment order. Each value MUST be within
            ``[CLIP_DURATION_FLOOR_S, CLIP_DURATION_CEILING_S]`` and the list
            length MUST equal segment count. When None, every clip uses
            ``clip_duration_s``.
        per_clip_engines: Optional per-segment engine override list (D-02
            extension), indexed by segment order. Each value MUST be exactly
            ``'humo'`` or ``'ltx2'``; when None, the classifier decides per
            segment. Threaded into ``_route_segment`` as ``force_engine``.
        output_path: Explicit output directory (Path). Supersedes ``output_dir``
            string. When both are given, ``output_path`` wins.
        ltx_motion_prompt: Explicit LTX-2 motion direction (F7/S9). Overrides the
            weak energy-conditioned motion phrase so b-roll/instrumental clips
            show real movement. GATED to LTX-2 (HuMo uses audio conditioning).
        reference_image_2: Optional second reference image (LTX-2 only) for scene
            variety. Passed through to ``_route_segment``; unused by HuMo.

    Returns dict with results: segments, clips, output path, timings.
    """
    # Resolve the generic output dir/portrait alongside legacy args (portrait_path
    # and output_path are the new canonical names; portrait/output_dir kept for
    # backward CLI compat).
    input_path = Path(input_audio).resolve()
    resolved_output = Path(output_path) if output_path is not None else Path(output_dir)
    output_path_obj = resolved_output.resolve()
    portrait_path_obj = Path(portrait_path) if portrait_path is not None else Path(portrait)
    portrait_path = portrait_path_obj.resolve()
    output_path = output_path_obj

    # Validate inputs
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    if not portrait_path.exists():
        raise FileNotFoundError(f"Portrait not found: {portrait_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "clips").mkdir(exist_ok=True)

    result: dict[str, Any] = {
        "input_audio": str(input_path),
        "output_dir": str(output_path),
        "segments": [],
        "clips": [],
        "output_path": None,
        "timings": {},
        "mode": mode,
    }

    # ── Resume mode: read segment plan and generate clips ───────
    if resume:
        # BUG FIX 09.9-33-debug: Load --prompts-file before resume early-return.
        # The prompts_file loading below (line ~548) was never reached in resume mode,
        # causing --prompts-file to be silently ignored.
        effective_prompts = prompts
        if prompts_file is not None and prompts_file.exists() and effective_prompts is None:
            import json as _json
            _data = _json.loads(prompts_file.read_text())
            effective_prompts = _data.get("prompts")
            if effective_prompts:
                logger.info("Loaded pre-written prompts for resume: %d prompts from %s",
                            len(effective_prompts), prompts_file)
        return _run_resume_mode(
            output_path, result,
            skip_post_process=skip_post_process,
            lut_path=lut_path,
            grain_intensity=grain_intensity,
            sharpen_strength=sharpen_strength,
            width=width,
            height=height,
            two_stage=two_stage,
            upscale=upscale,
            apply_lut=apply_lut,
            pre_roll_frames=pre_roll_frames,
            use_lipdub=use_lipdub,
            use_vrdg_sigmas=True,
            prompts=effective_prompts,
            per_clip_references=per_clip_references,
            clip_duration_s=clip_duration_s,
            clip_durations=clip_durations,
            per_clip_engines=per_clip_engines,
            ltx_motion_prompt=ltx_motion_prompt,
            reference_image_2=reference_image_2,
            max_clips=max_clips,
            audio_stem=audio_stem,
        )

    # ── Stage 1: Demucs stem separation ─────────────────────────
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("Stage 1: Demucs stem separation")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY-RUN] Would run Demucs on %s", input_path)
        result["segments"].append(
            {"start": 0, "end": 10, "text": "[dry-run placeholder]", "duration": 10}
        )
        result["timings"]["demucs"] = 0
    else:
        # Check for existing vocals stem to skip re-running Demucs
        existing_vocals = _find_existing_vocals(output_path)
        if existing_vocals:
            logger.info("Stage 1 SKIPPED — reusing existing vocals stem: %s", existing_vocals)
            vocals_path = existing_vocals
            result["timings"]["demucs"] = 0
        else:
            stems = _run_demucs_separation(input_path, output_path, two_stems="vocals")
            vocals_path = stems.get("vocals")
            if not vocals_path:
                raise RuntimeError("Demucs did not produce a vocals stem")
            result["timings"]["demucs"] = round(time.monotonic() - t0, 1)
            logger.info("Vocals stem: %s", vocals_path)

    # ── Stage 2: Whisper transcription ──────────────────────────
    # Load creative inputs early so b-roll prompts are available for gap-fill.
    creative_inputs = _load_creative_inputs({
        "storyconcept_path": storyconcept_path,
        "themestyle_path": themestyle_path,
        "subjectsandscenes_path": subjectsandscenes_path,
        "lyrics_path": lyrics_path,
    })

    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("Stage 2: Whisper beat/lyric alignment")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY-RUN] Would run Whisper on vocals stem")
    else:
        # Check for existing transcript to skip re-running Whisper
        cached = _load_existing_transcript(output_path)
        if cached:
            segments, vocals_path = cached
            logger.info("Stage 2 SKIPPED — reusing existing transcript.json (%d segments)", len(segments))
            result["timings"]["whisper"] = 0

            # Run cached segments through augmentation pipeline (gap-fill + merge + split).
            audio_duration = _get_audio_duration(input_path)
            broll_prompts = _build_broll_prompts(creative_inputs) if creative_inputs else []
            segments = _fill_coverage_gaps(segments, audio_duration, broll_prompts)
            segments = _merge_short_segments(segments)
            segments = _split_long_segments(segments, max_s=max_clip_s)
            _validate_segment_plan(segments, audio_duration=audio_duration)

            result["segments"] = [
                {
                    "index": i,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "duration": round(s.duration, 3),
                    "text": s.text,
                    "word_count": 0,
                }
                for i, s in enumerate(segments)
            ]
        else:
            words = _transcribe_with_whisper(vocals_path)

            # Get audio duration and classify vocal vs instrumental regions.
            audio_duration = _get_audio_duration(input_path)
            vocal_words, last_vocal_end = _classify_vocal_regions(
                words, audio_duration
            )

            # Group only vocal-region words into segments.
            segments = _group_words_into_segments(
                vocal_words, max_segment_s=max_segment_s
            )

            # Fill coverage gaps (including instrumental outro) with b-roll.
            # Build b-roll prompts from creative inputs loaded below (Stage 2.5).
            # For the initial transcript save, use simple prompts; the full
            # augmentation with creative-input-driven b-roll happens after Stage 2.5.
            broll_prompts = _build_broll_prompts(creative_inputs) if creative_inputs else [
                "Countryside landscape with gentle breeze",
                "City street at night with neon reflections",
                "Nature scene with flowing water",
                "Atmospheric environmental shot",
            ]
            segments = _fill_coverage_gaps(
                segments, audio_duration, broll_prompts
            )

            # Merge short segments (including b-roll fillers) after gap-fill
            # so all segments exist before merging (Refinement 3).
            segments = _merge_short_segments(segments)

            # Split any segment longer than max_clip_s.
            segments = _split_long_segments(segments, max_s=max_clip_s)

            # Validate the segment plan against audio duration.
            _validate_segment_plan(segments, audio_duration=audio_duration)

        # Save transcript.json
        transcript = {
            "source": str(input_path),
            "audio_source": str(vocals_path),  # the actual file Whisper transcribed
            "vocals_stem": str(vocals_path),
            "total_duration": round(segments[-1].end if segments else 0, 2),
            "segment_count": len(segments),
            "segments": [
                {
                    "index": i,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "duration": round(s.duration, 3),
                    "text": s.text,
                    "word_count": len(s.words),
                }
                for i, s in enumerate(segments)
            ],
        }
        transcript_path = output_path / "transcript.json"
        transcript_path.write_text(json.dumps(transcript, indent=2))
        logger.info("Transcript saved: %s (%d segments)", transcript_path, len(segments))

        result["segments"] = transcript["segments"]
        result["timings"]["whisper"] = round(time.monotonic() - t0, 1)

    # ── Dry-run: print plan and exit ────────────────────────────
    if dry_run:
        logger.info("=" * 60)
        logger.info("[DRY-RUN] Pipeline plan:")
        logger.info("=" * 60)
        logger.info("Input: %s", input_path)
        logger.info("Output: %s", output_path)
        logger.info("Portrait: %s", portrait_path)
        logger.info("Scene prompt: %s", scene_prompt)
        logger.info("Max segment length: %.1fs", max_segment_s)
        logger.info("Mode: %s", mode)
        logger.info("")
        logger.info("Stages:")
        logger.info("  1. Demucs separation -> %s/htdemucs/", output_path)
        logger.info("  2. Whisper transcription -> %s/transcript.json", output_path)
        logger.info("  3. Scene portrait generation (Qwen I2I)")
        if mode == "controlled":
            logger.info("  4.5. Per-segment reference images (Qwen I2I)")
            logger.info("  4.5.5. Approval gate (segment_plan.json)")
        logger.info("  4. LTX-2 clip generation per segment")
        logger.info("  5. FFmpeg compositing -> %s/final_output.mp4", output_path)
        logger.info("")
        logger.info("Segments (%d total):", len(result["segments"]))
        for seg in result["segments"]:
            logger.info("  [%ds-%ds] %s", seg["start"], seg["end"], seg["text"][:60])
        return result

    # ── Stage 2.5: LLM prompt refinement (before Slingshot hibernates LLM) ──
    # creative_inputs already loaded at Stage 2 start for b-roll prompt building.
    refined_prompts: list[dict[str, str]] = []

    # Check for pre-written prompts (pre-written prompt support).
    # When pre-written VRDG prose prompts exist, use them directly for
    # singer segments and generate LLM prompts for b-roll/instrumental.
    # This avoids LLM chain-of-thought artifacts and prompt count mismatch.
    prewritten_data: dict[str, list] | None = None
    if prompts_file is not None and prompts_file.exists():
        prewritten_data = json.loads(prompts_file.read_text())
        logger.info("Loaded pre-written prompts: %d prompts from %s",
                     len(prewritten_data.get("prompts", [])), prompts_file)

    if prewritten_data and prewritten_data.get("prompts"):
        logger.info("=" * 60)
        logger.info("Stage 2.5: Pre-written prompt mapping (%d segments)", len(segments))
        logger.info("=" * 60)
        t0 = time.monotonic()
        refined_prompts = _refine_prompts_with_prewrite(
            segments,
            prewritten_data["prompts"],
            prewritten_data.get("beat_types", ["singer"] * len(prewritten_data["prompts"])),
            scene_prompt,
        )
        result["timings"]["prompt_refinement"] = round(time.monotonic() - t0, 1)
        logger.info("Pre-written prompt mapping complete (%.1fs)",
                     result["timings"]["prompt_refinement"])
    elif creative_inputs or scene_prompt:
        logger.info("=" * 60)
        logger.info("Stage 2.5: LLM prompt refinement (%d segments)", len(segments))
        logger.info("=" * 60)
        t0 = time.monotonic()
        refined_prompts = _refine_prompts(segments, creative_inputs, scene_prompt)
        result["timings"]["prompt_refinement"] = round(time.monotonic() - t0, 1)
        logger.info("Prompt refinement complete (%.1fs)", result["timings"]["prompt_refinement"])

    # ── Slingshot: Hibernate local LLM before GPU-intensive work ──
    slingshot_enabled = os.getenv("SLINGSHOT_ENABLED", "true").lower() not in ("false", "0", "no")
    slingshot_active = False
    slingshot = SlingshotClient() if slingshot_enabled else None

    # Cloud-LLM guard (Finding 3): on a cloud session  (cloud-LLM) there
    # is no local LLM to preserve, so skip ALL slingshot operations. Otherwise
    # the atexit/SIGTERM recovery could load an untracked llama.cpp orphan that
    # squats GPU VRAM. is_local_llm_active() queries gpu-manager /slingshot/status.
    if slingshot is not None and not is_local_llm_active():
        logger.info(
            "Slingshot: no local LLM active (cloud session) — disabling slingshot "
            "hibernate + recovery for this run"
        )
        slingshot_enabled = False
        slingshot = None

    # Register for SIGTERM recovery (atexit component unregistered on failure
    # in the finally block below — see the 1080p-OOM debug session 2026-07-11).
    if slingshot is not None:
        _register_slingshot_recovery(slingshot)

    if slingshot_enabled:
        slingshot_active = slingshot.ensure_hibernate()
        if not slingshot_active:
            logger.info("Slingshot hibernate skipped — continuing without LLM hibernation")
    else:
        logger.info("Slingshot disabled by SLINGSHOT_ENABLED env var")

    try:
        # ── Model pre-flight: validate LTX-2.3 file locations ──
        if not _validate_and_migrate_ltx2_models():
            raise RuntimeError(
                "LTX-2.3 model validation failed. Check logs for missing models."
            )

        # ── Stage 3: Scene-locked portrait ──────────────────────
        t0 = time.monotonic()
        logger.info("=" * 60)
        logger.info("Stage 3: Scene-locked portrait generation")
        logger.info("=" * 60)

        scene_portrait = _generate_scene_portrait(portrait_path, scene_prompt)
        result["timings"]["portrait"] = round(time.monotonic() - t0, 1)
        logger.info("Scene portrait: %s", scene_portrait)

        # ── Controlled mode: generate per-segment refs ──────────
        if mode == "controlled":
            return _run_controlled_mode(
                output_path, portrait_path, scene_portrait, segments,
                scene_prompt, slingshot, slingshot_active, result, input_path,
                refined_prompts,
                creative_inputs=creative_inputs,
                width=width,
                height=height,
                max_clip_s=max_clip_s,
                two_stage=two_stage,
            )

        # ── Auto mode: generate clips directly ──────────────────
        return _run_auto_mode(
            segments, scene_portrait, scene_prompt, output_path, input_path,
            slingshot, slingshot_active, result, vocals_path, refined_prompts,
            creative_inputs=creative_inputs,
            skip_post_process=skip_post_process,
            lut_path=lut_path,
            grain_intensity=grain_intensity,
            sharpen_strength=sharpen_strength,
            width=width,
            height=height,
            max_clip_s=max_clip_s,
            two_stage=two_stage,
            upscale=upscale,
            apply_lut=apply_lut,
            pre_roll_frames=pre_roll_frames,
            use_lipdub=use_lipdub,
            use_vrdg_sigmas=True,
            prompts=prompts,
            per_clip_references=per_clip_references,
            clip_duration_s=clip_duration_s,
            clip_durations=clip_durations,
            per_clip_engines=per_clip_engines,
        )

    finally:
        # ── Abort cleanup: stop ComfyUI on failed runs ──
        # If no clips were generated, ComfyUI is likely still holding VRAM
        # (crashed, aborted, guard failed). Stop it before waking the LLM
        # to avoid VRAM thrash (ComfyUI 10-40GB + LLM 38GB > 48GB card).
        clips_generated = result.get("clips", [])
        if not clips_generated and mv_comfyui._comfyui_is_ready():
            logger.info("Abort cleanup: stopping ComfyUI (failed run, no clips)")
            try:
                mv_comfyui._stop_comfyui_via_gpu_manager()
            except Exception:
                logger.warning("Abort cleanup: ComfyUI stop failed (best-effort)")

        # ── Slingshot: Wake local LLM after GPU work (idempotent) ──
        # Always wake — slingshot is not None only for alia-local sessions
        # (cloud sessions set it to None at line 380). Claude Code needs the
        # LLM back to continue, even on failed runs.
        if slingshot is not None and slingshot_active:
            output_path_str = result.get("output_path", "")
            slingshot.ensure_wake(task_name="music_video", output_path=output_path_str)



def _run_auto_mode(
    segments: list[ClipSegment],
    scene_portrait: Path,
    scene_prompt: str,
    output_path: Path,
    input_path: Path,
    slingshot: SlingshotClient | None,
    slingshot_active: bool,
    result: dict[str, Any],
    vocals_stem: Path | None = None,
    refined_prompts: list[dict[str, str]] | None = None,
    creative_inputs: dict[str, str] | None = None,
    skip_post_process: bool = False,
    lut_path: str | None = None,
    grain_intensity: float = DEFAULT_GRAIN_INTENSITY,
    sharpen_strength: float = DEFAULT_SHARPEN_STRENGTH,
    width: int = mv_mvconst.DEFAULT_LOWRES_W,
    height: int = mv_mvconst.DEFAULT_LOWRES_H,
    max_clip_s: int = mv_mvconst.DEFAULT_MAX_CLIP_S,
    two_stage: bool | None = None,
    upscale: bool = True,
    apply_lut: bool = False,
    pre_roll_frames: int = PRE_ROLL_FRAMES,
    use_lipdub: bool = True,
    use_vrdg_sigmas: bool = True,
    # ── Plan 09.9-25-04 parametric inputs ──
    prompts: list[str] | None = None,
    per_clip_references: list[Path] | None = None,
    clip_duration_s: int | None = None,
    clip_durations: list[int] | None = None,
    per_clip_engines: list[str] | None = None,
) -> dict[str, Any]:
    """Run auto mode — generate clips directly from single portrait."""
    # ── Stage 4: Clip generation ────────────────────────────
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("Stage 4: LTX-2 clip generation (%d segments)", len(segments))
    logger.info("=" * 60)

    # Segments are already augmented (gap-fill + merge + split) in Stage 2.
    # Re-validate before GPU spend as defense-in-depth.
    audio_duration = _get_audio_duration(input_path)

    # Plan 09.9-25-04: fail-fast validation against the EFFECTIVE segment count
    # (after augmentation) before any GPU work / resolution planning.
    _validate_parametric_inputs(
        segments, prompts, per_clip_references,
        segments, prompts, per_clip_references,
        clip_duration_s, clip_durations, per_clip_engines,
    )

    # Plan 09.9-33-01: regression validation — abort before any GPU spend.
    _validate_segment_plan(segments, audio_duration=audio_duration)
    logger.info("Segment plan validation passed (%d segments, %.1fs total)",
                len(segments), sum(s.duration for s in segments))

    # Plan 09.9-09: resolve effective generation resolution ONCE up front.
    # Raises RuntimeError (aborting the run) if the VRAM guard fails so we
    # never OOM the 48GB card. The planned values are passed to every clip
    # (the LTX-2 model is not yet resident here, so free VRAM is the true
    # budget). gen_w/gen_h also feed the composite crop below.
    # Plan 09.9-19 (mv-vram-frame-guard): budget the up-front guard for the
    # LONGEST clip this run will generate. If even the worst-case segment (clamped
    # to LTX2_MAX_LENGTH_S) won't fit at this resolution, the guard refuses /
    # falls back BEFORE any clip is rendered — instead of OOMing mid-run.
    _seg_durs = [seg.duration for seg in segments]
    guard_max_s = min(max(_seg_durs, default=10.0), float(LTX2_MAX_LENGTH_S))
    gen_w, gen_h, use_two_stage, base_width, base_height, text_encoder_cpu = \
        _plan_clip_resolution(width, height, two_stage, max_length_s=guard_max_s)
    text_encoder_device = "cpu" if text_encoder_cpu else "default"
    # Path B delivery target = 2x the low-res base (960x544 -> 1920x1088 -> crop 1080).
    target_w = gen_w * mv_mvconst.UPSCALE_FACTOR
    target_h = gen_h * mv_mvconst.UPSCALE_FACTOR

    clip_paths: list[Path] = []
    clip_timings: list[tuple[float, float]] = []

    # Plan 09.9-33-02: cascade timing — each clip's conditioning audio comes
    # from the cascade position (cumulative measured video durations), not the
    # Whisper timestamp. This ensures conditioning_audio == assembly_audio.
    cascade_pos = 0.0

    for i, seg in enumerate(segments):
        # Check circuit breaker before each clip
        if mv_comfyui.comfyui_client._consecutive_failures >= MAX_CONSECUTIVE_COMFYUI_FAILURES:
            logger.error(
                "Circuit breaker OPEN — aborting clip generation at segment %d/%d. "
                "Generated %d/%d clips.",
                i + 1, len(segments), len(clip_paths), len(segments),
            )
            break

        logger.info(
            "Segment %d/%d: [%0.1fs-%0.1fs] '%s...'",
            i + 1, len(segments), seg.start, seg.end, seg.text[:40],
        )

        # Plan 09.9-25-04 (D-05): use the per-clip prompt when supplied, else the
        # refined video_prompt, else fall back to the single scene_prompt.
        clip_prompt = scene_prompt
        if prompts and i < len(prompts):
            clip_prompt = prompts[i]
        elif refined_prompts and i < len(refined_prompts):
            clip_prompt = refined_prompts[i]["video_prompt"]

        # Plan 09.9-25-04 (D-04): per-clip reference when supplied, else the
        # single portrait default.
        reference = per_clip_references[i] if per_clip_references else scene_portrait
        # Plan 09.9-25-04 (D-01 extension): variable per-clip duration.
        clip_dur = (clip_durations[i] if (clip_durations and i < len(clip_durations))
                     else clip_duration_s)
        # Plan 09.9-25-04 (D-02 extension): per-clip engine override.
        seg_engine = per_clip_engines[i] if per_clip_engines else None

        # Plan 09.9-33-02: compute cascade timing for this segment.
        cascade_start = cascade_pos
        cascade_end = cascade_pos + seg.duration

        # Single dispatch point for BOTH HuMo and LTX-2 (hybrid router from
        # plan 01). duration_s=clip_dur now reaches HuMo (closing the prior
        # wiring gap where 16s was hardcoded); force_engine enables B-roll-on-
        # lyrics. Do NOT call _generate_clip directly from the orchestrator.
        clip_path = _route_segment(
            seg, clip_prompt, reference, i + 1, output_path,
            vocal_presence=bool(vocals_stem),
            has_lyrics=bool(seg.text and seg.text.strip()),
            segment_prompt=clip_prompt,
            vocals_stem=vocals_stem,
            original_audio=input_path,
            duration_s=clip_dur,
            force_engine=seg_engine,
            ltx_motion_prompt=ltx_motion_prompt,
            reference_image_2=reference_image_2,
            use_vrdg_sigmas=use_vrdg_sigmas,
            use_lipdub=use_lipdub,
            cascade_start=cascade_start,
            cascade_end=cascade_end,
        )
        if clip_path:
            clip_paths.append(clip_path)

            # Plan 09.9-33-02: measure actual video duration, write manifest,
            # advance cascade position. Failed clips do NOT advance cascade_pos.
            video_dur = _get_clip_duration(clip_path)
            seg.measured_duration = video_dur

            # Plan 09.9-33-02: use cascade timing for composite placement,
            # not original segment plan positions. With --max-clips N the
            # segment plan spans only the first N segments but the canvas
            # would be full audio duration, leaving black frames.
            clip_timings.append((cascade_start, cascade_start + video_dur))
            frame_count = int(round(video_dur * 24))

            _write_clip_manifest(
                clip_index=i + 1,
                clip_path=clip_path,
                segment=seg,
                cascade_position=cascade_pos,
                measured_duration=video_dur,
                frame_count=frame_count,
                scene_prompt=clip_prompt,
                reference_portrait=reference,
                shot_type=seg.shot_type or "singer",
                output_dir=output_path,
            )

            cascade_pos += video_dur
            logger.info(
                "Clip %d: measured %.3fs (%d frames), cascade advances to %.3fs",
                i + 1, video_dur, frame_count, cascade_pos,
            )

            # Duration fidelity validation (Invariant 4, Plan 09.9-33)
            _validate_clip_duration(i + 1, video_dur, seg.duration)

            # Bug 2 instrumentation — log duration discrepancies (logging only)
            _log_duration_instrumentation(
                clip_index=i,
                cascade_start=cascade_start,
                cascade_end=cascade_end,
                measured_duration=video_dur,
                plan_duration=seg.duration,
            )

            result["clips"].append({
                "index": i + 1,
                "path": str(clip_path),
                "segment": {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text,
                },
            })
        else:
            logger.warning("Segment %d skipped (generation failed)", i + 1)

    # Plan 09.9-25-04 (D-04): emit references_manifest.json ONLY when per-clip
    # references were supplied (never for the default single-portrait path).
    if per_clip_references is not None:
        manifest_path = _write_references_manifest(
            output_path, per_clip_references, [seg.start for seg in segments]
        )
        logger.info("Per-clip references manifest written: %s", manifest_path)

    result["timings"]["clips"] = round(time.monotonic() - t0, 1)
    logger.info("Cascade complete: %.3fs total (%d clips)", cascade_pos, len(clip_paths))
    logger.info("Generated %d/%d clips", len(clip_paths), len(segments))

    # ── Stage 5: FFmpeg compositing ─────────────────────────
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("Stage 5: FFmpeg compositing")
    logger.info("=" * 60)

    final_output = _composite_with_ffmpeg(
        clip_paths, input_path, output_path,
        skip_post_process=skip_post_process,
        lut_path=lut_path,
        grain_intensity=grain_intensity,
        sharpen_strength=sharpen_strength,
        gen_width=target_w,
        gen_height=target_h,
        clip_timings=clip_timings,
        audio_duration=cascade_pos if cascade_pos > 0 else audio_duration,
        apply_lut=apply_lut,
        pre_roll_frames=pre_roll_frames,
    )
    result["output_path"] = str(final_output)
    result["timings"]["ffmpeg"] = round(time.monotonic() - t0, 1)

    # ── Summary ─────────────────────────────────────────────────
    _log_summary(result, clip_paths, segments)
    return result



def _run_controlled_mode(
    output_path: Path,
    portrait_path: Path,
    scene_portrait: Path,
    segments: list[ClipSegment],
    scene_prompt: str,
    slingshot: SlingshotClient | None,
    slingshot_active: bool,
    result: dict[str, Any],
    input_path: Path,
    refined_prompts: list[dict[str, str]] | None = None,
    creative_inputs: dict[str, str] | None = None,
    width: int = mv_mvconst.DEFAULT_LOWRES_W,
    height: int = mv_mvconst.DEFAULT_LOWRES_H,
    max_clip_s: int = mv_mvconst.DEFAULT_MAX_CLIP_S,
    two_stage: bool | None = None,
    upscale: bool = True,
) -> dict[str, Any]:
    """Run controlled mode — generate per-segment refs, write plan, exit for approval."""
    # ── Stage 4.5: Generate per-segment refs ──────────────────
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("Stage 4.5: Per-segment reference images (%d segments)", len(segments))
    logger.info("=" * 60)

    # Segments are already augmented in Stage 2.
    seg_plans = _generate_segment_refs(
        scene_portrait, scene_prompt, segments, output_path, refined_prompts,
    )
    result["timings"]["segment_refs"] = round(time.monotonic() - t0, 1)

    # Generate approval dashboard (approval dashboard)
    dashboard_path = _generate_approval_dashboard(seg_plans, output_path, scene_prompt)

    # Write segment plan
    plan_path = _write_segment_plan(seg_plans, output_path,
                                    result["input_audio"],
                                    str(portrait_path),
                                    scene_prompt, refined_prompts,
                                    width=width, height=height, two_stage=two_stage)

    # Save transcript
    results_path = output_path / "pipeline_results.json"
    results_path.write_text(json.dumps(result, indent=2))

    logger.info("=" * 60)
    logger.info("CONTROLLED MODE — APPROVAL GATE")
    logger.info("=" * 60)
    logger.info("Segment plan written: %s", plan_path)
    logger.info("Approval dashboard: %s", dashboard_path)
    logger.info("Segments: %d", len(seg_plans))
    logger.info("Ref images: %d/%d generated", sum(1 for p in seg_plans if p.ref_image_path), len(seg_plans))
    logger.info("")
    logger.info("Review the approval dashboard in your browser:")
    logger.info("  file://%s", dashboard_path)
    logger.info("Select candidate images and click 'Export Selections'.")
    logger.info("Edit segment_plan.json to adjust shot_type or prompt per segment.")
    logger.info("After approval, resume with: --resume --output %s", output_path)

    result["segment_plan_path"] = str(plan_path)
    result["segment_plans"] = [
        {
            "index": p.index,
            "start": round(p.start, 3),
            "end": round(p.end, 3),
            "text": p.text,
            "shot_type": p.shot_type,
            "prompt": p.prompt,
            "ref_image_path": p.ref_image_path,
            "candidate_paths": p.candidate_paths,
            "status": p.status,
        }
        for p in seg_plans
    ]
    result["approval_dashboard"] = str(dashboard_path)

    return result



def _run_resume_mode(
    output_path: Path,
    result: dict[str, Any],
    skip_post_process: bool = False,
    lut_path: str | None = None,
    grain_intensity: float = DEFAULT_GRAIN_INTENSITY,
    sharpen_strength: float = DEFAULT_SHARPEN_STRENGTH,
    width: int = mv_mvconst.DEFAULT_LOWRES_W,
    height: int = mv_mvconst.DEFAULT_LOWRES_H,
    max_clip_s: int = mv_mvconst.DEFAULT_MAX_CLIP_S,
    two_stage: bool | None = None,
    upscale: bool = True,
    apply_lut: bool = False,
    pre_roll_frames: int = PRE_ROLL_FRAMES,
    use_lipdub: bool = True,
    use_vrdg_sigmas: bool = True,
    # ── Plan 09.9-25-04 parametric inputs ──
    prompts: list[str] | None = None,
    per_clip_references: list[Path] | None = None,
    clip_duration_s: int | None = None,
    clip_durations: list[int] | None = None,
    per_clip_engines: list[str] | None = None,
    ltx_motion_prompt: str = "",
    reference_image_2: str = "",
    max_clips: int | None = None,
    audio_stem: Path | None = None,
) -> dict[str, Any]:
    """Resume from segment_plan.json — generate clips per shot type, composite."""
    plan_data = _read_segment_plan(output_path)

    # Recover persisted resolution (Plan 09.9-09). Prefer stored values so a
    # resumed run keeps the same resolution it was planned with.
    width = plan_data.get("width", width)
    height = plan_data.get("height", height)
    two_stage = plan_data.get("two_stage", two_stage)

    input_path = Path(plan_data["input_audio"])
    portrait_path = Path(plan_data["portrait"])
    scene_prompt = plan_data["scene_prompt"]

    # BUG 09.9-17: reconcile the plan against the real audio BEFORE any
    # max_clips truncation. The reconciliation compares the full plan span
    # against the real audio duration. If we truncate first, reconciliation
    # sees a false mismatch and tries to re-transcribe (needs CUDA/cuBLAS).
    plan_data = reconcile_plan_with_audio(
        output_path, input_path, plan_data, max_clip_s=max_clip_s,
    )

    # Plan 09.9-33-04: pilot mode — limit to first N segments AFTER reconciliation.
    if max_clips is not None and max_clips < len(plan_data["segments"]):
        logger.info("PILOT MODE: limiting to first %d of %d segments",
                     max_clips, len(plan_data["segments"]))
        plan_data["segments"] = plan_data["segments"][:max_clips]
        # Slice prompts/references/durations to match truncated segment count
        if prompts is not None:
            prompts = prompts[:max_clips]
        if per_clip_references is not None:
            per_clip_references = per_clip_references[:max_clips]
        if clip_durations is not None:
            clip_durations = clip_durations[:max_clips]
        if per_clip_engines is not None:
            per_clip_engines = per_clip_engines[:max_clips]

    logger.info("=" * 60)
    logger.info("RESUME MODE — Generating clips from segment plan")
    logger.info("=" * 60)
    logger.info("Segments: %d", len(plan_data["segments"]))

    # Plan 09.9-25-04: fail-fast validation against the EFFECTIVE segment count
    _validate_parametric_inputs(
        plan_data["segments"], prompts, per_clip_references,
        clip_duration_s, clip_durations, per_clip_engines,
    )

    width = plan_data.get("width", width)
    height = plan_data.get("height", height)
    two_stage = plan_data.get("two_stage", two_stage)

    # Try to find vocals stem — explicit --audio-stem takes priority,
    # then fall back to transcript.json or htdemucs output.
    # BUG FIX 09.9-33: audio_stem was never passed to _run_resume_mode(),
    # causing all clips to generate with audio=none (no lip-sync conditioning).
    vocals_stem: Path | None = audio_stem
    if vocals_stem is None:
        cached = _load_existing_transcript(output_path)
        if cached:
            _, vocals_stem = cached
    if vocals_stem is None:
        existing_vocals = _find_existing_vocals(output_path)
        if existing_vocals:
            vocals_stem = existing_vocals

    # Slingshot: hibernate before clip generation
    slingshot_enabled = os.getenv("SLINGSHOT_ENABLED", "true").lower() not in ("false", "0", "no")
    slingshot_active = False
    slingshot = SlingshotClient() if slingshot_enabled else None

    # Cloud-LLM guard (Finding 3): skip ALL slingshot ops when no local LLM is
    # active, so the atexit/SIGTERM recovery cannot load an orphan llama.cpp.
    if slingshot is not None and not is_local_llm_active():
        logger.info(
            "Slingshot: no local LLM active (cloud session) — disabling slingshot "
            "for resume run"
        )
        slingshot_enabled = False
        slingshot = None

    if slingshot_enabled:
        slingshot_active = slingshot.ensure_hibernate()

    # Register for SIGTERM recovery (atexit unregistered on failure in finally)
    if slingshot is not None:
        _register_slingshot_recovery(slingshot)

    # Validate model files before GPU work
    if not _validate_and_migrate_ltx2_models():
        raise RuntimeError(
            "LTX-2.3 model validation failed. Check logs for missing models."
        )

    try:
        # ── Stage 4: Clip generation per shot type ────────────
        t0 = time.monotonic()
        logger.info("=" * 60)
        logger.info("Stage 4: LTX-2 clip generation (%d segments)", len(plan_data["segments"]))
        logger.info("=" * 60)

        # Plan 09.9-09: resolve effective generation resolution ONCE up front.
        # Raises RuntimeError (aborting the run) if the VRAM guard fails so we
        # never OOM the 48GB card. Planned values passed to every clip; the
        # LTX-2 model is not yet resident here, so free VRAM is the true budget.
        # Plan 09.9-19 (mv-vram-frame-guard): budget the up-front guard for the
        # longest planned segment (clamped to LTX2_MAX_LENGTH_S) so a too-long max
        # segment is refused / downgraded before any clip is rendered.
        _seg_durs = [s["end"] - s["start"] for s in plan_data["segments"]]
        guard_max_s = min(max(_seg_durs, default=10.0), float(LTX2_MAX_LENGTH_S))
        gen_w, gen_h, use_two_stage, base_width, base_height, text_encoder_cpu = \
            _plan_clip_resolution(width, height, two_stage, max_length_s=guard_max_s)
        text_encoder_device = "cpu" if text_encoder_cpu else "default"
        # Path B delivery target = 2x the low-res base (960x544 -> 1920x1088 -> crop 1080).
        target_w = gen_w * mv_mvconst.UPSCALE_FACTOR
        target_h = gen_h * mv_mvconst.UPSCALE_FACTOR

        clip_paths: list[Path] = []
        clip_timings: list[tuple[float, float]] = []

        # Plan 09.9-33-02: cascade timing — each clip's conditioning audio comes
        # from the cascade position (cumulative measured video durations), not the
        # Whisper timestamp. This ensures conditioning_audio == assembly_audio.
        cascade_pos = 0.0

        for seg_i, seg_data in enumerate(plan_data["segments"]):
            # Check circuit breaker before each clip
            if mv_comfyui.comfyui_client._consecutive_failures >= MAX_CONSECUTIVE_COMFYUI_FAILURES:
                logger.error(
                    "Circuit breaker OPEN — aborting clip generation at segment %d/%d. "
                    "Generated %d/%d clips.",
                    seg_i + 1, len(plan_data["segments"]),
                    len(clip_paths), len(plan_data["segments"]),
                )
                break

            i = seg_data["index"]
            shot_type = seg_data.get("shot_type", "singer")
            seg_prompt = seg_data.get("prompt", scene_prompt)
            ref_path = seg_data.get("ref_image_path")

            # Build ClipSegment from plan
            seg = ClipSegment(
                start=seg_data["start"],
                end=seg_data["end"],
                text=seg_data["text"],
                duration=seg_data["end"] - seg_data["start"],
                words=[],
            )

            # Plan 09.9-25-04 (D-05): per-clip prompt when supplied, else the
            # segment plan prompt, else the single scene_prompt.
            clip_prompt = seg_prompt
            if prompts and i < len(prompts):
                clip_prompt = prompts[i]

            # Plan 09.9-25-04 (D-04): per-clip reference when supplied, else the
            # segment's own ref_image_path, else the single portrait default.
            if per_clip_references and i < len(per_clip_references):
                reference = per_clip_references[i]
            else:
                reference = Path(ref_path) if ref_path and Path(ref_path).exists() else portrait_path

            # Plan 09.9-25-04 (D-01 extension): variable per-clip duration.
            clip_dur = (clip_durations[i] if (clip_durations and i < len(clip_durations))
                         else clip_duration_s)
            # Plan 09.9-25-04 (D-02 extension): per-clip engine override.
            # Plan 09.9-28-02: fall back to pipeline_engine from segment_plan.json.
            seg_engine = (
                per_clip_engines[i] if (per_clip_engines and i < len(per_clip_engines))
                else seg_data.get("pipeline_engine")
            )

            logger.info(
                "Segment %d: [%0.1fs-%0.1fs] shot_type=%s, '%s...'",
                i, seg.start, seg.end, shot_type, seg.text[:40],
            )

            # Plan 09.9-33-02: compute cascade timing for this segment.
            cascade_start = cascade_pos
            cascade_end = cascade_pos + seg.duration

            # Single dispatch point for BOTH HuMo and LTX-2 (hybrid router from
            # plan 01). duration_s=clip_dur reaches HuMo; force_engine enables
            # B-roll-on-lyrics. Do NOT call _generate_clip directly.
            clip_path = _route_segment(
                seg, clip_prompt, reference, i, output_path,
                vocal_presence=bool(vocals_stem),
                has_lyrics=bool(seg.text and seg.text.strip()),
                segment_prompt=clip_prompt,
                vocals_stem=vocals_stem,
                original_audio=input_path,
                duration_s=clip_dur,
                force_engine=seg_engine,
                ltx_motion_prompt=ltx_motion_prompt,
                reference_image_2=reference_image_2,
                use_vrdg_sigmas=use_vrdg_sigmas,
                use_lipdub=use_lipdub,
                cascade_start=cascade_start,
                cascade_end=cascade_end,
            )

            if clip_path:
                clip_paths.append(clip_path)

                # Plan 09.9-33-02: measure actual video duration, write manifest,
                # advance cascade position. Failed clips do NOT advance cascade_pos.
                video_dur = _get_clip_duration(clip_path)
                seg.measured_duration = video_dur
                frame_count = int(round(video_dur * 24))
                # Use cascade timing for composite placement, not segment plan.
                clip_timings.append((cascade_start, cascade_start + video_dur))

                _write_clip_manifest(
                    clip_index=i,
                    clip_path=clip_path,
                    segment=seg,
                    cascade_position=cascade_pos,
                    measured_duration=video_dur,
                    frame_count=frame_count,
                    scene_prompt=clip_prompt,
                    reference_portrait=reference,
                    shot_type=shot_type,
                    output_dir=output_path,
                )

                cascade_pos += video_dur
                logger.info(
                    "Clip %d: measured %.3fs (%d frames), cascade advances to %.3fs",
                    i, video_dur, frame_count, cascade_pos,
                )

                # Duration fidelity validation (Invariant 4, Plan 09.9-33)
                _validate_clip_duration(i, video_dur, seg.duration)

                # Bug 2 instrumentation — log duration discrepancies (logging only)
                _log_duration_instrumentation(
                    clip_index=i,
                    cascade_start=cascade_start,
                    cascade_end=cascade_end,
                    measured_duration=video_dur,
                    plan_duration=seg.duration,
                )

                result["clips"].append({
                    "index": i,
                    "path": str(clip_path),
                    "shot_type": shot_type,
                    "segment": {
                        "start": seg_data["start"],
                        "end": seg_data["end"],
                        "text": seg_data["text"],
                    },
                })
            else:
                logger.warning("Segment %d skipped (generation failed)", i)

        # Hard abort if circuit breaker opened — raise so atexit recovery fires
        if mv_comfyui.comfyui_client._consecutive_failures >= MAX_CONSECUTIVE_COMFYUI_FAILURES:
            raise RuntimeError(
                f"Pipeline aborted: VRAM circuit breaker opened. "
                f"Generated {len(clip_paths)}/{len(plan_data['segments'])} clips."
            )

        logger.info("Cascade complete: %.3fs total (%d clips)", cascade_pos, len(clip_paths))

        # Plan 09.9-25-04 (D-04): emit references_manifest.json ONLY when per-
        # clip references were supplied (never for the default single-portrait
        # path). Indexed by segment order.
        if per_clip_references is not None:
            manifest_indices = [s["index"] for s in plan_data["segments"]]
            manifest_refs = [per_clip_references[i] for i in manifest_indices if i < len(per_clip_references)]
            manifest_starts = [s["start"] for s in plan_data["segments"]]
            manifest_path = _write_references_manifest(
                output_path, manifest_refs, manifest_starts
            )
            logger.info("Per-clip references manifest written: %s", manifest_path)

        result["timings"]["clips"] = round(time.monotonic() - t0, 1)
        logger.info("Generated %d/%d clips", len(clip_paths), len(plan_data["segments"]))
        result["segments"] = plan_data["segments"]

        # ── Stage 5: FFmpeg compositing (timeline-aware, Plan 09.9-12) ──
        t0 = time.monotonic()
        logger.info("=" * 60)
        logger.info("Stage 5: FFmpeg compositing")
        logger.info("=" * 60)

        # Plan 09.9-33-02: use cascade timing from generation loop for both
        # clip placement and canvas duration. With --max-clips N the segment
        # plan spans only the first N segments; using full audio_duration would
        # create a canvas with black frames after the last clip.
        audio_duration = _get_audio_duration(input_path)
        canvas_duration = cascade_pos if cascade_pos > 0 else audio_duration

        final_output = _composite_with_ffmpeg(
            clip_paths, input_path, output_path,
            skip_post_process=skip_post_process,
            lut_path=lut_path,
            grain_intensity=grain_intensity,
            sharpen_strength=sharpen_strength,
            gen_width=target_w,
            gen_height=target_h,
            clip_timings=clip_timings,
            audio_duration=canvas_duration,
            apply_lut=apply_lut,
            pre_roll_frames=pre_roll_frames,
        )
        result["output_path"] = str(final_output)
        result["timings"]["ffmpeg"] = round(time.monotonic() - t0, 1)

    finally:
        # ── Abort cleanup: stop ComfyUI on failed runs ──
        clips_generated = result.get("clips", [])
        if not clips_generated and mv_comfyui._comfyui_is_ready():
            logger.info("Abort cleanup: stopping ComfyUI (failed run, no clips)")
            try:
                mv_comfyui._stop_comfyui_via_gpu_manager()
            except Exception:
                logger.warning("Abort cleanup: ComfyUI stop failed (best-effort)")

        # Always wake — slingshot is not None only for alia-local sessions.
        # Claude Code needs the LLM back even on failed runs.
        if slingshot is not None and slingshot_active:
            output_path_str = result.get("output_path", "")
            slingshot.ensure_wake(task_name="music_video", output_path=output_path_str)

    # ── Summary ─────────────────────────────────────────────────
    _log_summary(result, clip_paths, plan_data["segments"])
    return result



def _validate_clip_duration(
    clip_index: int,
    measured_duration: float,
    requested_duration: float,
) -> None:
    """Validate generated clip duration matches requested duration within LTX-2 bounds.

    LTX-2 quantizes frame counts to 8k+1, so the maximum per-clip error is
    half-stride = 4 frames = 167ms. A deviation exceeding 2x this bound
    indicates a duration propagation bug (e.g., fixed 16s override).

    Args:
        clip_index: 1-based clip index.
        measured_duration: Actual video duration from ffprobe.
        requested_duration: Segment duration that was requested for generation.
    """
    quantization_bound = LTX2_QUANTIZATION_BOUND_S
    actual_diff = abs(measured_duration - requested_duration)
    if actual_diff > quantization_bound * 2:
        logger.error(
            "Clip %d: duration deviation %.1fms exceeds %.1fms bound. "
            "Requested %.3fs, got %.3fs. Possible duration propagation bug.",
            clip_index,
            actual_diff * 1000,
            quantization_bound * 2 * 1000,
            requested_duration,
            measured_duration,
        )


# ── Bug 2 instrumentation (technical debt, 2026-07-24) ──────────────
# Bug 2: conditioning audio duration (seg.duration / plan) may differ from
# measured video duration (8k+1 frame quantization). Deferred —
# pilot passed acceptance without fix. Instrumentation logs discrepancy for
# future reference; does NOT change runtime behaviour.
# Revisit if: lip-sync drift reappears at longer clip counts or higher bitrates.
_DUR_DISCREPANCY_WARN_THRESHOLD_S: float = float(
    os.environ.get("MV_DURATION_DISCREPANCY_WARN_S", "0.5")
)


def _log_duration_instrumentation(
    clip_index: int,
    cascade_start: float,
    cascade_end: float,
    measured_duration: float,
    plan_duration: float,
) -> None:
    """Log duration instrumentation for Bug 2 technical debt tracking.

    This is a LOG-ONLY function — it does not change any runtime behaviour.
    It records the discrepancy between planned conditioning audio duration
    and measured video duration for future reference.

    Args:
        clip_index: 0-based clip index.
        cascade_start: Cascade position at clip start (authoritative timeline).
        cascade_end: cascade_start + plan_duration (conditioning audio end).
        measured_duration: Actual video duration after 8k+1 quantization.
        plan_duration: Original segment plan duration (seg.duration).
    """
    conditioning_dur = cascade_end - cascade_start
    diff_ms = abs(conditioning_dur - measured_duration) * 1000
    diff_pct = (
        (diff_ms / (conditioning_dur * 1000)) * 100 if conditioning_dur > 0 else 0
    )

    logger.info(
        "Clip %d duration: conditioning=%.3fs video=%.3fs cascade=%.3fs plan=%.3fs "
        "diff=%.1fms (%.1f%%)",
        clip_index,
        conditioning_dur,
        measured_duration,
        conditioning_dur,
        plan_duration,
        diff_ms,
        diff_pct,
    )

    if diff_ms > _DUR_DISCREPANCY_WARN_THRESHOLD_S * 1000:
        logger.warning(
            "Clip %d: conditioning-video duration discrepancy %.1fms (%.1f%%) "
            "exceeds threshold %.1fms. "
            "Conditioning audio [%.3f, %.3f) = %.3fs; video = %.3fs. "
            "Technical debt — Bug 2 (09.9-33). See architecture docs.",
            clip_index,
            diff_ms,
            diff_pct,
            _DUR_DISCREPANCY_WARN_THRESHOLD_S * 1000,
            cascade_start,
            cascade_end,
            conditioning_dur,
            measured_duration,
        )


def _log_summary(
    result: dict[str, Any],
    clip_paths: list[Path],
    segments: list | dict,
) -> None:
    """Log pipeline completion summary."""
    total_time = sum(result["timings"].values())
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Output: %s", result.get("output_path", "N/A"))
    logger.info("Clips: %d/%d", len(clip_paths), len(segments))
    logger.info("Total time: %.1fs", total_time)
    for stage, duration in result["timings"].items():
        logger.info("  %s: %.1fs", stage, duration)

    # Save pipeline results
    output_dir = Path(result["output_dir"])
    results_path = output_dir / "pipeline_results.json"
    results_path.write_text(json.dumps(result, indent=2))



def main() -> int:
    """CLI entry point for the music video pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated LTX Music Video Pipeline — "
                    "Demucs -> Whisper -> LTX-2 clips -> FFmpeg composite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full pipeline (auto mode)
  python generate_music_video_pipeline.py \\
      --input song.mp3 \\
      --output ./output \\
      --portrait portrait.jpg \\
      --scene-prompt "singer on stage, concert lighting"

  # Controlled mode: generate refs, exit for approval
  python generate_music_video_pipeline.py \\
      --input song.mp3 \\
      --output ./output \\
      --portrait portrait.jpg \\
      --scene-prompt "singer on stage" \\
      --mode controlled

  # Resume from segment plan: generate clips + composite
  python generate_music_video_pipeline.py \\
      --input song.mp3 \\
      --output ./output \\
      --portrait portrait.jpg \\
      --scene-prompt "singer on stage" \\
      --resume

  # Dry run (print plan only)
  python generate_music_video_pipeline.py \\
      --input song.mp3 \\
      --output ./output \\
      --portrait portrait.jpg \\
      --scene-prompt "test scene" \\
      --dry-run
""",
    )
    parser.add_argument(
        "--input", required=True,
        help="Input audio file path (MP3, WAV, etc.)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for stems, clips, and final video",
    )
    parser.add_argument(
        "--portrait", required=True,
        help="Reference portrait path (canonical sister portrait)",
    )
    parser.add_argument(
        "--scene-prompt", required=True,
        help="Scene description for LTX-2 generation",
    )
    parser.add_argument(
        "--max-segment-s", type=float, default=10.0,
        help="Maximum segment length in seconds (default: 10, range: 4-18)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print pipeline plan without executing GPU work",
    )
    parser.add_argument(
        "--mode", choices=["auto", "controlled"], default="auto",
        help="Pipeline mode: 'auto' (fully automated, default) or "
             "'controlled' (per-segment refs + approval gate)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from segment_plan.json — skip stages 1-3, "
             "generate clips + composite",
    )
    parser.add_argument(
        "--max-clips", type=int, default=None,
        help="Limit clip generation to first N segments (pilot mode). "
             "When set, only the first N segments are generated and assembled.",
    )
    parser.add_argument(
        "--storyconcept", default=None,
        help="Path to story concept text file (for LLM prompt refinement)",
    )
    parser.add_argument(
        "--themestyle", default=None,
        help="Path to theme/style text file (for LLM prompt refinement)",
    )
    parser.add_argument(
        "--subjectsandscenes", default=None,
        help="Path to subjects/scenes text file (for LLM prompt refinement)",
    )
    parser.add_argument(
        "--lyrics", default=None,
        help="Path to lyrics text file (for LLM prompt refinement)",
    )
    parser.add_argument(
        "--prompts-file", default=None,
        help="Path to pre-written prompts JSON file (prompts.json). "
             "When provided, maps pre-written VRDG prose prompts to singer "
             "segments and skips LLM refinement for those segments. "
             "Format: {\"prompts\": [...], \"beat_types\": [...]}.",
    )

    # Post-processing args ()
    parser.add_argument(
        "--lut", default=None,
        help="Path to .cube LUT file for color grading (default: Cine_Grade.cube)",
    )
    parser.add_argument(
        "--grain-intensity", type=float, default=DEFAULT_GRAIN_INTENSITY,
        help=f"Film grain intensity 0.0-10.0 (default: {DEFAULT_GRAIN_INTENSITY})",
    )
    parser.add_argument(
        "--sharpen-strength", type=float, default=DEFAULT_SHARPEN_STRENGTH,
        help=f"Sharpening strength 0.0-1.5 (default: {DEFAULT_SHARPEN_STRENGTH})",
    )
    parser.add_argument(
        "--skip-post-process", action="store_true",
        help="Skip post-processing chain (color grading, grain, sharpening)",
    )
    parser.add_argument(
        "--apply-lut", action="store_true",
        help="Apply the Cine Grade 3D LUT + color mixer. Default off: neutral "
             "grade (no tint) matching the 09.9-16 reference clips.",
    )

    # Resolution + two-stage args (Plan 09.9-09)
    parser.add_argument(
        "--width", type=int, default=mv_mvconst.DEFAULT_LOWRES_W,
        help="Base generation width (default 960; 2x -> 1920 1080p output)",
    )
    parser.add_argument(
        "--height", type=int, default=mv_mvconst.DEFAULT_LOWRES_H,
        help="Base generation height (default 544; 2x -> 1088 then crop to 1080)",
    )
    parser.add_argument(
        "--low-res", action="store_true", default=True,
        help="Generate base clips at low-res (default True), then upscale to "
             "1080p output via the ffmpeg stitch scale branch (Path A bislerp). "
             "Native 1080p generation is disabled (it OOMs the 48GB card).",
    )
    parser.add_argument(
        "--upscale", dest="upscale", action="store_true", default=True,
        help="Latent-upscaler toggle (Path B, coming in 09.9-14). In Wave 1 the "
             "upscaler model is absent, so the pipeline falls back to Path A "
             "bislerp decode regardless of this flag.",
    )
    parser.add_argument(
        "--no-upscale", dest="upscale", action="store_false",
        help="Disable the (future) latent upscaler; use Path A bislerp only.",
    )
    parser.add_argument(
        "--max-clip-s", type=int, default=mv_mvconst.DEFAULT_MAX_CLIP_S,
        help="Max single-clip duration in seconds (default 6); longer segments "
             "are split to <= this many seconds for VRAM headroom.",
    )
    parser.add_argument(
        "--two-stage", dest="two_stage", action="store_true", default=None,
        help="Force two-stage sampling. Auto if long-edge>768.",
    )
    parser.add_argument(
        "--no-two-stage", dest="two_stage", action="store_false",
        help="Force single-stage. Overrides auto threshold.",
    )
    parser.add_argument(
        "--no-lipdub", dest="no_lipdub", action="store_true", default=False,
        help="Decouple identity from lip sync (09.9-20): run the LTX clip with "
             "Lipdub OFF (clean reference identity, node '1a' omitted). Default "
             "False keeps production Lipdub ON. The separate HUMO talking-head "
             "model then drives true lip sync.",
    )

    # ── Plan 09.9-25-04 parametrization flags (D-01/D-02/D-04/D-05/D-06) ──
    # Generic, reusable surface so the pipeline can render ANY song/reference,
    # not just a hardcoded one. No song/reference names appear here.
    parser.add_argument(
        "--audio-stem", default=None,
        help="Explicit full vocals stem (Demucs-separated WAV). When omitted, "
             "the pipeline derives vocals via Demucs as today.",
    )
    parser.add_argument(
        "--prompts", action="append", default=None,
        help="Per-segment scene prompt (D-05), repeated once per segment or as a "
             "JSON list[str]. Aligned to segment order; required for per-clip "
             "prompt conditioning (do NOT collapse to a single global prompt).",
    )
    parser.add_argument(
        "--portrait-path", default=None,
        help="Single 16:9 reference path (D-04 default). Supersedes --portrait.",
    )
    parser.add_argument(
        "--per-clip-references", action="append", default=None,
        help="Optional per-segment reference path (D-04 option), repeated per "
             "segment or as a JSON list[str]. When supplied, references_manifest.json "
             "is emitted for manual QA. Length must match segment count.",
    )
    parser.add_argument(
        "--per-clip-references-file", default=None,
        help="Path to JSON file with per-segment reference mappings. "
             "Format: [{\"clip_index\": N, \"reference_path\": \"...\"}, ...]. "
             "Alternative to repeated --per-clip-references flags.",
    )
    parser.add_argument(
        "--clip-duration-s", type=int, default=None,
        help=f"Global default clip duration in seconds (D-01); default "
             f"{mv_mvconst.DEFAULT_CLIP_DURATION_S}. Each clip uses this unless "
             f"--clip-durations overrides per clip.",
    )
    parser.add_argument(
        "--clip-durations", action="append", default=None,
        help="Optional per-clip duration in seconds (D-01 extension), repeated "
             "per segment or as a JSON list[int]. Each value in "
             f"[{mv_mvconst.CLIP_DURATION_FLOOR_S},{mv_mvconst.CLIP_DURATION_CEILING_S}].",
    )
    parser.add_argument(
        "--per-clip-engines", action="append", default=None,
        help="Optional per-segment engine override (D-02 extension): 'humo' or "
             "'ltx2', repeated per segment or as a JSON list[str]. When omitted, "
             "the classifier decides per segment.",
    )
    parser.add_argument(
        "--output-path", default=None,
        help="Explicit output directory (Path form). Supersedes --output.",
    )
    parser.add_argument(
        "--ltx-motion-prompt", default="",
        help="F7/S9: explicit LTX-2 motion direction (overrides energy-conditioned "
             "phrase). GATED to LTX-2 only (HuMo uses audio conditioning).",
    )
    parser.add_argument(
        "--reference-image-2", default="",
        help="F7/S9: optional second LTX-2 reference image for scene variety "
             "(LTX-2 only).",
    )

    args = parser.parse_args()

    # ── Plan 09.9-25-04 parametrization flags (D-01/D-02/D-04/D-05/D-06) ──
    # Helpers to parse repeated-or-JSON list args.
    def _parse_list_str(value: str | list | None, cast):
        if value is None:
            return None
        # action="append" returns a list of strings from argparse.
        if isinstance(value, list):
            # If a single JSON array string was passed, the list has one element.
            if len(value) == 1:
                try:
                    parsed = json.loads(value[0])
                    if isinstance(parsed, list):
                        return [cast(x) for x in parsed]
                except (json.JSONDecodeError, TypeError):
                    pass
            # Already a list of individual values — return as-is.
            return [cast(x) for x in value]
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [cast(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        # Fall back to a single-element list (JSON scalar) or comma-split.
        if isinstance(value, str) and "," in value:
            return [cast(x.strip()) for x in value.split(",")]
        return [cast(value)]

    prompts = _parse_list_str(args.prompts, str) if args.prompts is not None else None
    per_clip_references = (
        [Path(p) for p in _parse_list_str(args.per_clip_references, str)]
        if args.per_clip_references is not None else None
    )
    # Load per-clip references from JSON file if provided
    if per_clip_references is None and args.per_clip_references_file is not None:
        _ref_data = json.loads(Path(args.per_clip_references_file).read_text())
        # Sort by clip_index to ensure correct segment ordering
        _ref_data.sort(key=lambda x: x.get("clip_index", 0))
        per_clip_references = [Path(r["reference_path"]) for r in _ref_data]
        logger.info("Loaded %d per-clip references from %s",
                     len(per_clip_references), args.per_clip_references_file)
    clip_durations = _parse_list_str(args.clip_durations, int) if args.clip_durations is not None else None
    per_clip_engines = _parse_list_str(args.per_clip_engines, str) if args.per_clip_engines is not None else None
    audio_stem = Path(args.audio_stem) if args.audio_stem is not None else None

    try:
        result = run_pipeline(
            input_audio=args.input,
            output_dir=args.output,
            portrait=args.portrait,
            scene_prompt=args.scene_prompt,
            max_segment_s=args.max_segment_s,
            max_clip_s=args.max_clip_s,
            dry_run=args.dry_run,
            mode=args.mode,
            resume=args.resume,
            max_clips=args.max_clips,
            storyconcept_path=args.storyconcept,
            themestyle_path=args.themestyle,
            subjectsandscenes_path=args.subjectsandscenes,
            lyrics_path=args.lyrics,
            lut_path=args.lut,
            grain_intensity=args.grain_intensity,
            sharpen_strength=args.sharpen_strength,
            skip_post_process=args.skip_post_process,
            width=args.width,
            height=args.height,
            two_stage=args.two_stage,
            upscale=args.upscale,
            apply_lut=args.apply_lut,
            pre_roll_frames=PRE_ROLL_FRAMES,
            use_lipdub=not args.no_lipdub,
            audio_stem=audio_stem,
            prompts=prompts,
            portrait_path=Path(args.portrait_path) if args.portrait_path else None,
            per_clip_references=per_clip_references,
            clip_duration_s=args.clip_duration_s,
            clip_durations=clip_durations,
            per_clip_engines=per_clip_engines,
            output_path=Path(args.output_path) if args.output_path else None,
            ltx_motion_prompt=args.ltx_motion_prompt,
            reference_image_2=args.reference_image_2,
            prompts_file=Path(args.prompts_file) if args.prompts_file else None,
        )

        if not args.dry_run and result.get("output_path"):
            print(f"\nFinal output: {result['output_path']}")
        elif result.get("segment_plan_path"):
            print(f"\nSegment plan: {result['segment_plan_path']}")
            print(f"Review refs in {args.output}/refs/, edit segment_plan.json, then resume with --resume")
        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except RuntimeError as e:
        logger.error("Pipeline error: %s", e)
        return 2
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        if _RECOVERY_SLINGSHOT is not None:
            _RECOVERY_SLINGSHOT.ensure_wake()
        return 130



if __name__ == "__main__":
    sys.exit(main())
