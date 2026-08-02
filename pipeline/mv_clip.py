#!/usr/bin/env python3
"""Clip generation: scene portrait + facade to the clip orchestration (Plan 09.9-10).

This module owns the scene-portrait generation helper and re-exports
``_generate_clip`` (the per-segment ComfyUI build + retry loop), which now lives
in ``mv_clip_generate.py`` per STYLE.md (single responsibility, <=300 LOC each).
Callers of ``mv_clip._generate_clip`` are unaffected by the split.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal

import mv_comfyui
from mv_black import _generate_black_frame  # noqa: F401  (kept for backward imports)
from mv_segment import ClipSegment  # noqa: F401  (kept for backward imports)

logger = logging.getLogger(__name__)

# Engine choice type for the D-02 hybrid router. "humo" routes to the HuMo 14B
# talking-head generator (plan 02's generate_humo_clip); "ltx2" keeps the
# existing LTX-2 path. Imported by mv_clip_generate._route_segment.
EngineChoice = Literal["humo", "ltx2"]


def classify_segment_engine(
    segment: "ClipSegment",
    vocal_presence: bool,
    has_lyrics: bool,
) -> "EngineChoice":
    """Decide which engine generates one segment (D-02 hybrid router).

    Single decision point for per-segment engine selection. Parametrized ONLY
    by per-segment metadata — never by song name or reference image, so the
    pipeline stays generic for future reuse (D-06).

    Rule (LTX-first architecture):
        - "ltx2" for ALL segments by default. LTX-2.3 handles all clip types,
          including lip-sync singer clips (proven in extensive A/B
          comparison). The combined workflow with golden settings produces
          quality comparable to HuMo for singer segments.
        - "humo" is only reached via explicit ``force_engine="humo"`` override
          in ``_route_segment`` (e.g. per-clip engine flag or segment plan
          ``pipeline_engine`` field).

    Reused by ``mv_clip_generate._route_segment``; do not inline this logic.
    """
    return "ltx2"

# Negative prompt suffix (from server_dialogue.py, locked per CONTEXT.md)
NEG_SUFFIX_6TERM = (
    ". No visible text, no signage, no readable inscriptions, "
    "no labels, no book titles, no writing any kind in frame."
)

# LTX-2 clip length bounds (clamped in _generate_clip).
LTX2_MAX_LENGTH_S = 18
LTX2_MIN_LENGTH_S = 4

# build_ltx2_workflow is re-exported from here for facade compatibility.
try:
    from workflows.workflow_ltx2 import build_ltx2_workflow  # noqa: E402  (re-export target)
except ImportError:
    build_ltx2_workflow = None  # type: ignore[assignment]

# _generate_clip lives in mv_clip_generate (STYLE split); re-exported so
# existing callers (generate_music_video_pipeline.py) resolve it from here.
from mv_clip_generate import _generate_clip  # noqa: E402,F401


# ── Per-segment audio-energy → motion phrasing (Option C / BUG A) ──
# Replaces the static "one hand gesturing naturally" motion text with
# energy-conditioned phrasing so clip motion matches the audio mood.
# Calibrated on initial run — adjust here, not inline.
_ENERGY_CALM_CENTROID_HZ = 3300.0
_ENERGY_VIGOROUS_CENTROID_HZ = 4300.0
_ENERGY_ONSET_RATE_PER_S = 1.0
_ENERGY_RMS_FLOOR = 0.12
_ENERGY_ONSET_SMOOTH_WINDOW = 5
_ENERGY_FRAME_RATE_HZ = 10.0

def _energy_motion_phrase(audio_path) -> str:
    """Return an energy-conditioned motion phrase from a cropped vocals WAV.

    Buckets the clip into gentle / moderate / vigorous motion from RMS energy,
    frame-difference onset rate, and spectral centroid. Returns "" when audio is
    unavailable or analysis fails, so callers can append it unconditionally.
    """
    if audio_path is None:
        return ""
    p = Path(audio_path)
    if not p.exists() or p.stat().st_size == 0:
        return ""
    try:
        import wave as _wave
        import numpy as _np

        with _wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            data = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
        elif sw == 1:
            data = (_np.frombuffer(raw, dtype=_np.uint8).astype(_np.float32) - 128.0) / 128.0
        else:
            return ""
        if nch > 1:
            data = data.reshape(-1, nch).mean(axis=1)
        if data.size == 0:
            return ""
        dur = data.size / float(sr)
        if dur <= 0:
            return ""

        rms = float(_np.sqrt(_np.mean(data ** 2)))

        hop = max(1, int(0.1 * sr))
        n_frames = max(1, data.size // hop)
        framed = data[: n_frames * hop].reshape(n_frames, hop)
        frame_rms = _np.sqrt(_np.mean(framed ** 2, axis=1))
        local = _np.convolve(
            frame_rms, _np.ones(_ENERGY_ONSET_SMOOTH_WINDOW) / _ENERGY_ONSET_SMOOTH_WINDOW,
            mode="same",
        )
        local[local == 0] = 1e-6
        onset_rate = float((frame_rms > 1.5 * local).sum()) / dur

        win = _np.hanning(hop)
        specs = _np.abs(_np.fft.rfft(framed * win, axis=1))
        freqs = _np.fft.rfftfreq(hop, d=1.0 / sr)
        centroid = float(
            (_np.sum(specs * freqs, axis=1) / (_np.sum(specs, axis=1) + 1e-9)).mean()
        )
    except Exception as e:  # noqa: BLE001 — never let audio analysis break a clip
        logger.warning("Energy analysis failed for %s: %s", p, e)
        return ""

    energetic = (
        (centroid > _ENERGY_VIGOROUS_CENTROID_HZ)
        or (centroid > 3700.0 and onset_rate > _ENERGY_ONSET_RATE_PER_S)
        or (rms > _ENERGY_RMS_FLOOR and centroid > 3600.0)
    )
    calm = centroid < _ENERGY_CALM_CENTROID_HZ
    if energetic:
        return (
            "with vigorous, rhythmic body movement and energetic, expressive "
            "hand gestures; dynamic, animated motion that matches the driving beat"
        )
    if calm:
        return (
            "with gentle, subtle swaying and minimal, soft hand gestures; "
            "calm, intimate, understated movement that breathes with the melody"
        )
    return "with natural, relaxed movement and light, easy hand gestures"


def _generate_scene_portrait(
    canonical_portrait: Path, scene_prompt: str, motion_phrase: str | None = None
) -> Path:
    """Generate a scene-locked body-frame portrait using Qwen I2I.

    Mirrors alice_generate_scene_locked_portrait_impl from server_dialogue.py
    but uses direct ComfyUI HTTP API instead of the async MCP tool path. This
    portrait is the reference image for all LTX-2 clips, ensuring character
    consistency across scenes (scene_locked_portrait pattern). Falls back to the
    canonical portrait directly if generation fails.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu-manager"))
        from workflows.workflows import build_qwen_gguf_i2i_workflow
    except ImportError as e:
        logger.warning(
            "Could not import Qwen I2I workflow builder (%s). "
            "Using canonical portrait directly.", e
        )
        output_portrait = Path("/tmp") / f"scene_portrait_{int(time.time())}.jpg"
        shutil.copy2(canonical_portrait, output_portrait)
        return output_portrait

    motion_instruction = motion_phrase or (
        "Captured mid-speaking, mouth slightly open, mid-syllable, "
        "one hand gesturing naturally."
    )
    full_prompt = f"{scene_prompt}. {motion_instruction}"

    if not mv_comfyui._check_vram_gate():
        logger.warning("VRAM gate failed for portrait generation. Using canonical portrait.")
        output_portrait = Path("/tmp") / f"scene_portrait_{int(time.time())}.jpg"
        shutil.copy2(canonical_portrait, output_portrait)
        return output_portrait

    comfyui_input = Path(mv_comfyui.COMFYUI_OUTPUT_DIR) / "input"
    comfyui_input.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    input_name = f"scene_portrait_ref_{ts}.jpg"
    input_path = comfyui_input / input_name
    shutil.copy2(canonical_portrait, input_path)

    try:
        workflow = build_qwen_gguf_i2i_workflow(
            prompt=full_prompt,
            image_paths=[input_name],
            steps=4,
        )

        prompt_id = mv_comfyui._queue_workflow(workflow)
        logger.info("Portrait generation queued (prompt_id=%s)", prompt_id)
        history = mv_comfyui._poll_completion(prompt_id, timeout=300)

        output_path = mv_comfyui._find_output_file(history, "alice_i2i_gguf", "jpg")
        logger.info("Portrait generated: %s", output_path)
        return output_path

    except Exception as e:
        logger.warning("Portrait generation failed (%s). Using canonical portrait.", e)
        output_portrait = Path("/tmp") / f"scene_portrait_{int(time.time())}.jpg"
        shutil.copy2(canonical_portrait, output_portrait)
        return output_portrait
    finally:
        try:
            input_path.unlink()
        except OSError:
            pass
