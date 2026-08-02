#!/usr/bin/env python3
"""Creative-input loading + two-stage LLM prompt refinement (Plan 09.9 Plan 06).

Module is kept <= 400 lines per STYLE.md (approved 400 ceiling deviation
from the 300 default). Logic is byte-for-byte identical to the original
`generate_music_video_pipeline.py` block this module was split from.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from mv_shot import _build_motion_prompt, _cycle_motion_templates
from mv_gpu_manager import GPU_MANAGER_BASE, is_local_llm_active

logger = logging.getLogger(__name__)

# System prompts for two-stage LLM refinement (from VRGDG pattern)
_T2I_SYSTEM_PROMPT = (
    "Create one text-to-image prompt from the user input. "
    "Use the current visual prompt as the main scene foundation. "
    "Keep the main action, subject, and setting from the current visual prompt "
    "unless the user clearly changes them. "
    "Do not add camera movement or video directions — this is a still image prompt. "
    "Output ONLY the refined image prompt as a single paragraph."
)

_I2V_SYSTEM_PROMPT = (
    "Convert the user's text-to-image prompt into a dynamic image-to-video prompt. "
    "Add fast cinematic motion with natural pacing by giving the subject a clear "
    "action sequence, expressive body movement, strong gestures, and camera movement. "
    "The output should describe what happens in the video, not just what the scene "
    "looks like. Output ONLY the refined video prompt as a single paragraph."
)


def _load_creative_inputs(path_dict: dict[str, str | None]) -> dict[str, str]:
    """Read optional creative input files.

    Takes a dict mapping key names to file paths (or None).
    Returns dict mapping key name to file text content.
    Skips gracefully if path is None or file does not exist.
    """
    result: dict[str, str] = {}
    for key, path_str in path_dict.items():
        if path_str is None:
            continue
        p = Path(path_str)
        if p.is_file():
            result[key] = p.read_text().strip()
        else:
            logger.warning("Creative input not found: %s (%s)", key, path_str)
    return result


def _build_broll_prompts(creative_inputs: dict[str, str]) -> list[str]:
    """Build one B-roll prompt per location listed in the subjects/scenes file.

    Each prompt is prefixed with the themestyle line so the auto-inserted
    coverage fillers (intro / mid / outro) share the song's visual language.
    ``creative_inputs`` maps the same keys as ``_load_creative_inputs`` but its
    values are the file TEXT contents (the themestyle and subjects/scenes text).

    Returns the list of prompts (typically 6). If the subjects/scenes text is
    absent, returns [] and logs a warning — coverage still runs and fillers get
    empty-text prompts as a degraded fallback.
    """
    theme_line = ""
    raw_theme = (creative_inputs.get("themestyle_path") or "").strip()
    if raw_theme:
        for line in raw_theme.splitlines():
            line = line.strip()
            if line:
                theme_line = line
                break
        # Strip a leading "Visual style:" label if present.
        m = re.match(r"^visual\s*style\s*:\s*(.+)$", theme_line, re.IGNORECASE)
        if m:
            theme_line = m.group(1).strip()

    raw_locations = creative_inputs.get("subjectsandscenes_path", "")
    if not raw_locations:
        logger.warning(
            "No subjectsandscenes_path provided — B-roll fillers will use "
            "empty prompts"
        )
        return []

    # Take only the portion before the "Rules for B-roll:" marker (case-insensitive).
    marker = re.split(r"rules\s+for\s+b-roll\s*:", raw_locations, flags=re.IGNORECASE)
    locations_text = marker[0]

    prompts: list[str] = []
    # Numbered entries: "1. Location name — description" (em/en/hyphen dash).
    for loc in re.finditer(r"^\s*\d+\.\s*(.+?)\s*[—–-]", locations_text, re.MULTILINE):
        name = loc.group(1).strip()
        if not name:
            continue
        prompts.append(f"{theme_line} {name}" if theme_line else name)
    return prompts


def _get_local_llm_endpoint() -> tuple[str, str]:
    """Discover the active local LLM endpoint.

    Returns (endpoint_url, model_name) tuple.
    Tries gpu-manager /local-llm/status first, then env var, then default.
    """
    # Try gpu-manager /local-llm/status
    try:
        url = f"{GPU_MANAGER_BASE}/local-llm/status"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            port = data.get("port")
            model_name = data.get("served_model_name", "qwen3.6-35b-a3b")
            if port:
                return (
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    model_name,
                )
    except Exception as e:
        logger.debug("gpu-manager /local-llm/status unavailable: %s", e)

    # Try env var
    env_endpoint = os.getenv("LOCAL_LLM_ENDPOINT")
    if env_endpoint:
        return (env_endpoint, os.getenv("LOCAL_LLM_MODEL", "qwen3.6-35b-a3b"))

    # Default fallback
    return ("http://127.0.0.1:8029/v1/chat/completions", "qwen3.6-35b-a3b")


def _refine_prompts_with_prewrite(
    segments: list[ClipSegment],
    prewritten_prompts: list[str],
    prewritten_beat_types: list[str],
    scene_prompt: str,
) -> list[dict[str, str]]:
    """Map pre-written prompts to segments and generate LLM prompts for gaps.

    Pre-written prompts (e.g. from prompts.json) are beat-level VRDG prose
    prompts. This function maps them to segments based on timing overlap
    and shot type, then generates LLM prompts for b-roll/instrumental
    segments that don't have a pre-written prompt.

    Args:
        segments: List of ClipSegment (augmented, final count).
        prewritten_prompts: Pre-written VRDG prose prompts (one per beat).
        prewritten_beat_types: Beat type ("singer", "broll") for each prompt.
        scene_prompt: Fallback scene description for LLM-generated prompts.

    Returns list of dicts: [{"image_prompt": "...", "video_prompt": "..."}, ...]
    """
    from mv_shot import _build_motion_prompt, _cycle_motion_templates

    if not segments:
        return []

    # Matching strategy: index-based when counts align, timing-based as fallback.
    # When pre-written prompt count equals segment count, map 1:1 by position.
    # This is the expected mode for per-segment prompt files (e.g. 27 prompts
    # for 27 segments). When counts differ (e.g. 17 storyboard beats for 27
    # segments), fall back to timing-window overlap.
    prompt_map: dict[int, str] = {}

    if len(prewritten_prompts) == len(segments):
        # Index-based matching — direct 1:1 alignment.
        for i, seg in enumerate(segments):
            seg_idx = seg.index if hasattr(seg, "index") else i
            prompt_map[seg_idx] = prewritten_prompts[i]
    else:
        # Timing-based matching — build windows and find overlaps.
        # We need clip_durations to reconstruct beat timing. Use a default of 10s
        # if not available (caller should provide durations in prompts.json).
        beat_windows: list[tuple[float, float]] = []
        t = 0.0
        for bt in prewritten_beat_types:
            # Estimate beat end from start + typical duration
            # Use 10s as default; actual timing is approximate for matching
            beat_windows.append((t, t + 10.0))
            t += 10.0

        # Map each segment to a pre-written prompt by timing overlap and shot type.
        for seg in segments:
            seg_idx = seg.index if hasattr(seg, "index") else 0
            seg_type = seg.shot_type if hasattr(seg, "shot_type") else "singer"

            # Find overlapping pre-written prompt
            for pi, (bs, be) in enumerate(beat_windows):
                if pi >= len(prewritten_prompts):
                    break
                # Check timing overlap
                if seg.start < be and seg.end > bs:
                    # Use pre-written prompt for singer segments
                    if seg_type in ("singer", "instrumental") and prewritten_beat_types[pi] == "singer":
                        prompt_map[seg_idx] = prewritten_prompts[pi]
                        break

    # Generate prompts for all segments.
    results: list[dict[str, str]] = []
    for i, seg in enumerate(segments):
        seg_idx = seg.index if hasattr(seg, "index") else i
        seg_type = seg.shot_type if hasattr(seg, "shot_type") else "singer"

        if seg_idx in prompt_map:
            # Use pre-written prompt directly — no LLM refinement needed.
            # Pre-written prompts are already in VRDG prose style.
            vrpdg_prompt = prompt_map[seg_idx]
            results.append({
                "image_prompt": vrpdg_prompt,
                "video_prompt": vrpdg_prompt,
            })
        elif seg_type in ("broll", "black"):
            # B-roll segments: use scene prompt + lyrics as fallback.
            # No LLM refinement needed for simple b-roll.
            broll_text = seg.text[:80] if seg.text else ""
            fallback = f"{scene_prompt}. {broll_text}" if broll_text else scene_prompt
            results.append({
                "image_prompt": fallback,
                "video_prompt": fallback,
            })
        else:
            # Instrumental or unmatched singer: use scene prompt fallback.
            motion = _cycle_motion_templates(i)
            motion_prompt = _build_motion_prompt(
                motion["camera_motion"], motion["character_action"], motion["energy"]
            )
            fallback_img = f"{scene_prompt}. {seg.text[:80]}"
            fallback_vid = f"{scene_prompt}. {seg.text[:80]}. {motion_prompt}"
            results.append({
                "image_prompt": fallback_img,
                "video_prompt": fallback_vid,
            })

    return results


def _refine_prompts(
    segments: list[ClipSegment],
    creative_inputs: dict[str, str],
    scene_prompt: str,
) -> list[dict[str, str]]:
    """Two-stage LLM prompt refinement using the local LLM.

    Stage 1 (T2I): For each segment, call the local LLM with creative inputs
    to produce a refined image prompt.
    Stage 2 (I2V): For each segment, take the T2I output + motion template
    to produce a refined video prompt with cinematic motion.

    Falls back to combining scene_prompt + lyrics + motion template when
    the LLM is unavailable.

    Returns list of dicts: [{"image_prompt": "...", "video_prompt": "..."}, ...]
    """
    if not segments:
        return []

    # Cloud-LLM session (no local LLM active): skip refinement as a clean
    # no-op instead of spamming connection-refused warnings (Finding 5). The
    # generic fallback prompts are still returned so downstream B-roll uses
    # scene_prompt + lyrics. The legit-local-LLM-down path is untouched: if a
    # local LLM IS active but the HTTP call fails, the try/except below still
    # falls back gracefully.
    local_llm_active = is_local_llm_active()
    if not local_llm_active:
        logger.info(
            "Prompt refinement skipped — no local LLM active (cloud session); "
            "using generic fallback prompts"
        )
        results: list[dict[str, str]] = []
        for seg in segments:
            motion = _cycle_motion_templates(0)
            motion_prompt = _build_motion_prompt(
                motion["camera_motion"], motion["character_action"], motion["energy"]
            )
            results.append({
                "image_prompt": f"{scene_prompt}. {seg.text[:80]}",
                "video_prompt": f"{scene_prompt}. {seg.text[:80]}. {motion_prompt}",
            })
        return results

    endpoint_url, model_name = _get_local_llm_endpoint()
    results = []

    for i, seg in enumerate(segments):
        motion = _cycle_motion_templates(i)
        motion_prompt = _build_motion_prompt(
            motion["camera_motion"], motion["character_action"], motion["energy"]
        )

        # Build user prompt for Stage 1 (T2I)
        parts = []
        if creative_inputs.get("storyconcept_path"):
            parts.append(f"Story concept: {creative_inputs['storyconcept_path']}")
        if creative_inputs.get("themestyle_path"):
            parts.append(f"Theme/style: {creative_inputs['themestyle_path']}")
        if creative_inputs.get("subjectsandscenes_path"):
            parts.append(f"Subjects/scenes: {creative_inputs['subjectsandscenes_path']}")
        if creative_inputs.get("lyrics_path"):
            parts.append(f"Lyrics: {creative_inputs['lyrics_path']}")
        parts.append(f"Base scene: {scene_prompt}")
        parts.append(f"Segment lyrics: {seg.text[:120]}")
        parts.append(f"Segment index: {i}")
        user_prompt_t2i = "\n".join(parts)

        # Fallback prompts (used when LLM unavailable)
        fallback_image = f"{scene_prompt}. {seg.text[:80]}"
        fallback_video = f"{scene_prompt}. {seg.text[:80]}. {motion_prompt}"

        # Stage 1: T2I refinement
        image_prompt = fallback_image
        try:
            body = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "system", "content": _T2I_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt_t2i},
                ],
                "max_tokens": 256,
                "temperature": 0.8,
            }).encode("utf-8")
            req = urllib.request.Request(
                endpoint_url, data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                llm_data = json.loads(resp.read().decode("utf-8"))
                choices = llm_data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    # Use reasoning_content if present (llama.cpp --reasoning-budget
                    # separates thinking from content), otherwise use content.
                    # Strip <thinking> tags as defense-in-depth.
                    c = msg.get("reasoning_content") or msg.get("content", "")
                    if isinstance(c, str):
                        c = re.sub(r"<thinking>.*?</thinking>", "", c, flags=re.DOTALL).strip()
                    if c:
                        image_prompt = c
                    else:
                        logger.warning("Segment %d: unexpected LLM response format", i)
                else:
                    logger.warning("Segment %d: no LLM choices", i)
        except Exception as e:
            logger.warning(
                "Segment %d: LLM T2I refinement failed (%s), using fallback", i, e
            )

        # Stage 2: I2V refinement
        video_prompt = fallback_video
        try:
            user_prompt_i2v = f"Image prompt: {image_prompt}\nMotion template: {motion_prompt}"
            body = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "system", "content": _I2V_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt_i2v},
                ],
                "max_tokens": 256,
                "temperature": 0.8,
            }).encode("utf-8")
            req = urllib.request.Request(
                endpoint_url, data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                llm_data = json.loads(resp.read().decode("utf-8"))
                choices = llm_data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    c = msg.get("reasoning_content") or msg.get("content", "")
                    if isinstance(c, str):
                        c = re.sub(r"<thinking>.*?</thinking>", "", c, flags=re.DOTALL).strip()
                    if c:
                        video_prompt = c
                    else:
                        logger.warning("Segment %d: unexpected LLM response format", i)
                else:
                    logger.warning("Segment %d: no LLM choices", i)
        except Exception as e:
            logger.warning(
                "Segment %d: LLM I2V refinement failed (%s), using fallback", i, e
            )

        results.append({
            "image_prompt": image_prompt,
            "video_prompt": video_prompt,
        })

    return results
