#!/usr/bin/env python3
"""Shot-type + pose/motion template helpers .

Module is kept <= 400 lines per STYLE.md (approved 400 ceiling deviation
from the 300 default). Logic is byte-for-byte identical to the original
`generate_music_video_pipeline.py` block this module was split from.
"""

from __future__ import annotations

import logging
from typing import Any

from mv_segment import ClipSegment, ShotType

logger = logging.getLogger(__name__)

# ── Controlled mode helpers ──

# Pose variations for per-segment ref generation
POSE_VARIATIONS = [
    "close-up, face filling the frame, intimate framing",
    "medium shot, waist-up, natural stance",
    "profile angle, side view, three-quarter turn",
    "wide shot, full body, stage presence",
    "three-quarter view, slight angle, dynamic pose",
    "over-the-shoulder perspective, depth of field",
    "low angle looking up, powerful stance",
    "high angle looking down, vulnerable framing",
]

# ── Motion templates () ──
# Cinematic motion language (verbs, not framing descriptions) that
# LTX-2 responds to better than static framing cues.

CAMERA_MOTION_TEMPLATES = [
    "Slow tracking shot following the subject",
    "Gentle dolly-in pushing closer",
    "Crane shot rising above",
    "Handheld follow cam with subtle shake",
    "Static wide shot with slow zoom-in",
    "Orbiting arc around the subject",
    "Low-angle push-in building intensity",
    "Overhead tilt-down revealing the scene",
    "Whip pan transitioning focus",
    "Slow push-in with subtle Dutch angle",
    "Static medium shot with rack focus shift",
    "Dolly zoom (vertigo effect) subtle",
]

CHARACTER_ACTION_TEMPLATES = [
    "Throws their head back then turns toward camera",
    "Steps forward with confidence, arms gesturing",
    "Leans in close, then pulls back",
    "Spins slowly, hair flowing",
    "Walks toward camera with purpose",
    "Sways gently to an invisible rhythm",
    "Looks over shoulder, then faces forward",
    "Raises hands expressively then drops them",
    "Paces side to side with energy",
    "Stands still, then bursts into movement",
    "Turns away, then looks back over shoulder",
    "Gestures outward, inviting the viewer in",
]

ENERGY_TEMPLATES = [
    "High energy, dynamic movement throughout",
    "Builds from calm to intense",
    "Steady moderate energy",
    "Starts explosive, settles into groove",
    "Slow burn that crescendos",
    "Pulsing rhythm with pauses",
    "Relaxed and effortless flow",
    "Frenetic bursts between calm moments",
    "Cinematic slow-motion feel",
    "Driving forward momentum",
    "Ebb and flow like breathing",
    "Sustained intensity with micro-variations",
]


def _get_pose_variation(index: int) -> str:
    """Cycle through pose variations for segment index."""
    return POSE_VARIATIONS[index % len(POSE_VARIATIONS)]


def _cycle_motion_templates(index: int) -> dict[str, str]:
    """Cycle through motion templates by segment index (modulo length).

    Returns dict with camera_motion, character_action, energy keys.
    """
    return {
        "camera_motion": CAMERA_MOTION_TEMPLATES[index % len(CAMERA_MOTION_TEMPLATES)],
        "character_action": CHARACTER_ACTION_TEMPLATES[index % len(CHARACTER_ACTION_TEMPLATES)],
        "energy": ENERGY_TEMPLATES[index % len(ENERGY_TEMPLATES)],
    }


def _build_motion_prompt(
    camera_motion: str, character_action: str, energy: str
) -> str:
    """Combine three template elements into a single motion instruction string."""
    return f"Camera: {camera_motion}. Subject: {character_action}. Pacing: {energy}."


def _assign_shot_type(
    seg: ClipSegment,
    index: int,
    total: int,
    prev_end: float,
) -> str:
    """Assign shot_type based on segment content and position.

    Heuristic:
    - First and last segments → always "singer" (bookends).
    - Gap > 1.5s before segment → "broll" (instrumental break).
    - Short segments (< 4s) in the middle → alternate "singer" / "instrumental".
    - Default → "singer" with pose variation.

    This ensures visual variety without losing lyrical segments.
    """
    # Bookends are always singer
    if index == 0 or index == total - 1:
        return ShotType.SINGER.value

    # Gap before segment → b-roll (instrumental/mood break)
    gap = seg.start - prev_end
    if gap > 1.5:
        return ShotType.BROLL.value

    # Short middle segments → instrumental for variety
    if seg.duration < 4.0:
        return ShotType.INSTRUMENTAL.value

    return ShotType.SINGER.value
