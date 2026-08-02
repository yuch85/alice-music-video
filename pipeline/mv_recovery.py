"""Plan/audio reconciliation for the music-video pipeline (BUG 09.9-17).

A ``segment_plan.json`` is generated for one specific audio. Resuming that plan
against a *different* audio (e.g. a full song's plan reused for a 45s short)
leaves the clip timings wrong: clips land at the wrong offsets, later clips fall
outside the canvas, and the gaps between clips render as held static frames.

This module owns the single responsibility of detecting that mismatch and, when
found, re-transcribing the real audio and rewriting the plan with correct,
contiguous timestamps. It is imported by generate_music_video_pipeline's resume
path. Kept separate (STYLE.md: single responsibility, < 300 LOC).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mv_mvconst
from mv_audio import (
    _get_audio_duration,
    _run_demucs_separation,
    _transcribe_with_whisper,
)
from mv_segment import (
    _fill_coverage_gaps,
    _group_words_into_segments,
    _split_long_segments,
)

logger = logging.getLogger(__name__)

# A plan is considered stale when its span diverges from the audio by more than
# the larger of an absolute floor and a proportional slack.
_MISMATCH_ABS_FLOOR_S = 2.0
_MISMATCH_REL_SLACK = 0.05


def _plan_span(plan_data: dict[str, Any]) -> float:
    """Largest segment end time in the plan (the plan's implied audio length)."""
    return max((float(s["end"]) for s in plan_data["segments"]), default=0.0)


def _is_mismatch(audio_duration: float, span: float) -> bool:
    """True when the plan span and the real audio duration disagree enough that
    resuming would misplace clips."""
    if span <= 0 or audio_duration <= 0:
        return False
    tolerance = max(_MISMATCH_ABS_FLOOR_S, _MISMATCH_REL_SLACK * span)
    return abs(audio_duration - span) > tolerance


def _rebuild_plan_for_audio(
    output_path: Path,
    input_path: Path,
    base_plan: dict[str, Any],
    max_clip_s: int,
) -> dict[str, Any]:
    """Re-transcribe ``input_path`` and rewrite segment_plan.json with correct
    timings, preserving portrait / scene_prompt / resolution from ``base_plan``.

    Ref images fall back to the canonical portrait — the resume clip loop already
    falls back to the portrait when a ref_image_path is missing, so no extra Qwen
    ref generation is needed. Returns the rewritten plan dict.
    """
    logger.info("Recovery: re-running Demucs + Whisper on actual audio %s", input_path)
    # Never reuse a cached vocals stem — it may belong to the audio the stale
    # plan was built from. Always separate the real input.
    stems = _run_demucs_separation(input_path, output_path, two_stems="vocals")
    vocals_path = stems.get("vocals")
    if not vocals_path:
        raise RuntimeError("Recovery Demucs did not produce a vocals stem")

    words = _transcribe_with_whisper(vocals_path)
    grouped = _group_words_into_segments(words, max_segment_s=max_clip_s)
    audio_duration = _get_audio_duration(input_path)
    # Tile 0 -> audio_duration contiguously (the shape the resume loop expects)
    # and split anything over the LTX-2 cap.
    augmented = _split_long_segments(
        _fill_coverage_gaps(grouped, audio_duration, []),
        max_s=max_clip_s,
    )

    portrait = base_plan["portrait"]
    scene_prompt = base_plan["scene_prompt"]
    plan_data: dict[str, Any] = {
        "mode": base_plan.get("mode", "controlled"),
        "input_audio": str(input_path),
        "portrait": portrait,
        "scene_prompt": scene_prompt,
        "refined_prompts": None,
        "width": base_plan.get("width", mv_mvconst.DEFAULT_LOWRES_W),
        "height": base_plan.get("height", mv_mvconst.DEFAULT_LOWRES_H),
        "two_stage": base_plan.get("two_stage"),
        "segments": [
            {
                "index": i + 1,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "shot_type": getattr(seg, "shot_type", None)
                or ("singer" if getattr(seg, "words", None) else "broll"),
                "prompt": scene_prompt,
                "ref_image_path": portrait,
                "status": "approved",
            }
            for i, seg in enumerate(augmented)
        ],
    }
    plan_path = output_path / "segment_plan.json"
    plan_path.write_text(json.dumps(plan_data, indent=2))
    logger.info(
        "Recovery: rewrote segment_plan.json with %d segments spanning %.1fs",
        len(augmented), audio_duration,
    )
    return plan_data


def reconcile_plan_with_audio(
    output_path: Path,
    input_path: Path,
    plan_data: dict[str, Any],
    *,
    max_clip_s: int = mv_mvconst.DEFAULT_MAX_CLIP_S,
) -> dict[str, Any]:
    """Return a plan whose timings match ``input_path``.

    If the plan already matches the audio, it is returned unchanged. On a
    mismatch, the plan is re-transcribed and rewritten (default), or the run is
    aborted when ``MV_PLAN_MISMATCH=abort``.
    """
    audio_duration = _get_audio_duration(input_path)
    span = _plan_span(plan_data)
    if not _is_mismatch(audio_duration, span):
        return plan_data

    logger.warning(
        "PLAN/AUDIO MISMATCH: segment_plan.json spans %.1fs but input audio '%s' "
        "is %.1fs. Plan was built for a different audio — resuming blindly would "
        "produce held static frames.",
        span, input_path, audio_duration,
    )
    if os.getenv("MV_PLAN_MISMATCH", "recover").lower() == "abort":
        raise RuntimeError(
            f"Plan/audio mismatch (plan spans {span:.1f}s, audio is "
            f"{audio_duration:.1f}s). Set MV_PLAN_MISMATCH=recover to auto-fix, "
            f"or regenerate segment_plan.json for this audio."
        )
    try:
        return _rebuild_plan_for_audio(output_path, input_path, plan_data, max_clip_s)
    except Exception as exc:
        raise RuntimeError(
            f"Plan/audio mismatch auto-recovery failed ({exc}). Regenerate "
            f"segment_plan.json for this audio before resuming."
        ) from exc
