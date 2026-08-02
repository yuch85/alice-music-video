#!/usr/bin/env python3
"""Visual storyboard classification helper — Stage 4.5 of the music video pipeline.

Classifies each beat with visual prioritisation metadata: narrative importance,
visual weight, performance focus, emotional/visual intensity, shot scale,
camera movement, coverage strategy, and rationale.

Module is kept <= 600 lines per STYLE.md (YC-approved 400 ceiling deviation
from the 300 default, extended for the 16-field dataclass and full
classification pipeline).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from mv_beats import BeatEntry, group_beats_by_section
from mv_utils import extract_and_repair_json  # noqa: F401 — re-export

logger = logging.getLogger(__name__)

# Constants
_WEIGHT_LOW = (2, 4)
_WEIGHT_MEDIUM = (3, 6)
_WEIGHT_HIGH = (7, 10)
_WEIGHT_CRITICAL = (8, 10)
_WEIGHT_INSTRUMENTAL = (2, 5)

# Performance focus defaults by section
_FOCUS_CHORUS = {"singer_pct": 70, "narrative_pct": 10, "broll_pct": 5,
                  "environment_pct": 10, "symbolic_pct": 5, "montage_pct": 0}
_FOCUS_VERSE = {"singer_pct": 50, "narrative_pct": 30, "broll_pct": 10,
                 "environment_pct": 10, "symbolic_pct": 0, "montage_pct": 0}
_FOCUS_INTRO = {"singer_pct": 20, "narrative_pct": 10, "broll_pct": 40,
                 "environment_pct": 30, "symbolic_pct": 0, "montage_pct": 0}
_FOCUS_BRIDGE = {"singer_pct": 60, "narrative_pct": 10, "broll_pct": 5,
                  "environment_pct": 15, "symbolic_pct": 10, "montage_pct": 0}
_FOCUS_OUTRO = {"singer_pct": 30, "narrative_pct": 10, "broll_pct": 20,
                 "environment_pct": 30, "symbolic_pct": 10, "montage_pct": 0}
_FOCUS_INSTRUMENTAL = {"singer_pct": 10, "narrative_pct": 5, "broll_pct": 30,
                        "environment_pct": 40, "symbolic_pct": 15, "montage_pct": 0}

_FOCUS_MAP: dict[str, dict[str, int]] = {
    "chorus": _FOCUS_CHORUS,
    "verse": _FOCUS_VERSE,
    "pre_chorus": _FOCUS_VERSE,
    "intro": _FOCUS_INTRO,
    "bridge": _FOCUS_BRIDGE,
    "outro": _FOCUS_OUTRO,
    "instrumental": _FOCUS_INSTRUMENTAL,
}


@dataclass
class StoryboardEntry:
    """Visual storyboard classification for a single beat.

    15 fields covering narrative importance, visual weight, performance focus,
    intensity metrics, shot recommendations, and rationale.
    """

    beat_index: int
    start: float
    end: float
    lyrics: str
    section: str
    narrative_importance: Literal["Low", "Medium", "High", "Critical"]
    visual_weight: int  # 1-10
    performance_focus: dict[str, int]
    emotional_intensity: int  # 1-5
    visual_energy: int  # 1-5
    primary_viewer_focus: str
    recommended_shot_duration_strategy: Literal[
        "Single", "Two-Shot", "Fast Montage", "Slow Cinematic"
    ]
    recommended_shot_scale: Literal[
        "ECU", "CU", "MCU", "Medium", "Full", "Wide", "Extreme Wide"
    ]
    camera_movement_intensity: Literal[
        "Locked", "Push", "Tracking", "Crane", "Orbit", "Dynamic"
    ]
    coverage_strategy: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoryboardEntry:
        """Deserialize from a plain dict."""
        return cls(
            beat_index=int(data.get("beat_index", 0)),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            lyrics=str(data.get("lyrics", "")),
            section=str(data.get("section", "")),
            narrative_importance=str(data.get("narrative_importance", "Medium")),
            visual_weight=int(data.get("visual_weight", 5)),
            performance_focus=dict(data.get("performance_focus", _FOCUS_VERSE)),
            emotional_intensity=int(data.get("emotional_intensity", 3)),
            visual_energy=int(data.get("visual_energy", 3)),
            primary_viewer_focus=str(data.get("primary_viewer_focus", "Face")),
            recommended_shot_duration_strategy=str(
                data.get("recommended_shot_duration_strategy", "Single")
            ),
            recommended_shot_scale=str(
                data.get("recommended_shot_scale", "MCU")
            ),
            camera_movement_intensity=str(
                data.get("camera_movement_intensity", "Push")
            ),
            coverage_strategy=str(data.get("coverage_strategy", "Single take")),
            rationale=str(data.get("rationale", "")),
        )


def classify_song_structure(beats: list[BeatEntry]) -> list[BeatEntry]:
    """Label each beat's section with music-video-aware heuristics.

    Recognise: intro, verse, pre-chorus, chorus, bridge, instrumental break,
    outro. Uses lyric content analysis and energy curves.

    Args:
        beats: List of BeatEntry objects.

    Returns:
        Same list with section labels populated.
    """
    if not beats:
        return beats

    total_duration = beats[-1].end if beats else 0.0
    first_lyric_idx = _find_first_lyric(beats)
    last_lyric_idx = _find_last_lyric(beats)

    for i, beat in enumerate(beats):
        if beat.section:
            continue  # Already classified

        if not beat.lyrics:
            # No lyrics — check position
            if i < first_lyric_idx:
                beat.section = "intro"
            elif i > last_lyric_idx:
                beat.section = "outro"
            else:
                beat.section = "instrumental"
        elif beat.energy == "high":
            beat.section = "chorus"
        elif beat.energy == "medium":
            # Check if this is a pre-chorus (building toward chorus)
            if _is_pre_chorus_position(i, beats):
                beat.section = "pre_chorus"
            else:
                beat.section = "verse"
        else:
            # Low energy with lyrics — likely verse or outro
            if i > last_lyric_idx - 2:
                beat.section = "outro"
            else:
                beat.section = "verse"

    logger.info("Classified song structure for %d beats", len(beats))
    return beats


def assign_visual_weight(beats: list[BeatEntry]) -> list[StoryboardEntry]:
    """Compute visual weight 1-10 for each beat.

    Based on section type, lyrical significance, energy level, and
    narrative importance.

    Args:
        beats: List of BeatEntry objects.

    Returns:
        List of StoryboardEntry objects with visual weights assigned.
    """
    entries: list[StoryboardEntry] = []

    for i, beat in enumerate(beats):
        section = beat.section.lower().replace(" ", "_")
        weight_range = _get_weight_range(section)
        weight = _compute_weight(beat, weight_range)

        # Determine narrative importance from weight
        if weight >= 9:
            importance = "Critical"
        elif weight >= 7:
            importance = "High"
        elif weight >= 4:
            importance = "Medium"
        else:
            importance = "Low"

        entry = StoryboardEntry(
            beat_index=i,
            start=beat.start,
            end=beat.end,
            lyrics=beat.lyrics,
            section=beat.section,
            narrative_importance=importance,
            visual_weight=weight,
            performance_focus=_get_default_focus(section),
            emotional_intensity=_map_energy_to_intensity(beat.energy),
            visual_energy=_map_energy_to_visual(beat.energy),
            primary_viewer_focus=_get_default_focus_target(section),
            recommended_shot_duration_strategy="Single",
            recommended_shot_scale="MCU",
            camera_movement_intensity="Push",
            coverage_strategy="",
            rationale="",
        )
        entries.append(entry)

    return entries


def assign_performance_focus(entry: StoryboardEntry) -> StoryboardEntry:
    """Set performance focus percentages based on section type.

    Choruses: high singer (60-80%). Verses: balanced (singer 40-60%,
    narrative 20-40%). B-roll: environment dominant. Bridge: emotional
    focus on singer or couple.

    Args:
        entry: StoryboardEntry to update.

    Returns:
        Updated StoryboardEntry.
    """
    section = entry.section.lower().replace(" ", "_")
    entry.performance_focus = _get_default_focus(section)
    return entry


def assign_shot_scale_and_movement(entry: StoryboardEntry) -> StoryboardEntry:
    """Recommend shot scale and camera movement.

    High emotional intensity + close viewer focus -> ECU/CU.
    Wide environments -> Wide/Extreme Wide.
    High visual energy -> Dynamic/Orbit camera.
    Low energy -> Locked/Push.

    Args:
        entry: StoryboardEntry to update.

    Returns:
        Updated StoryboardEntry.
    """
    # Shot scale based on viewer focus and intensity
    focus = entry.primary_viewer_focus.lower()
    if focus in ("eyes",):
        entry.recommended_shot_scale = "ECU"
    elif focus in ("face", "hands"):
        entry.recommended_shot_scale = "CU"
    elif focus in ("guitar", "couple"):
        entry.recommended_shot_scale = "MCU"
    elif focus in ("landscape",):
        entry.recommended_shot_scale = "Wide"
    else:
        entry.recommended_shot_scale = "Medium"

    # Camera movement based on visual energy
    if entry.visual_energy >= 4:
        entry.camera_movement_intensity = "Dynamic"
    elif entry.visual_energy >= 3:
        entry.camera_movement_intensity = "Tracking"
    elif entry.visual_energy >= 2:
        entry.camera_movement_intensity = "Push"
    else:
        entry.camera_movement_intensity = "Locked"

    return entry


def assign_coverage_strategy(entry: StoryboardEntry) -> StoryboardEntry:
    """Set the coverage strategy field.

    For beats whose duration exceeds ~10 seconds, record a durational
    timeline split describing how the long beat is covered by multiple
    shots/sub-segments. For shorter beats, use simple strategies.

    Args:
        entry: StoryboardEntry to update.

    Returns:
        Updated StoryboardEntry.
    """
    duration = entry.end - entry.start

    if duration > 10.0:
        mid = duration / 2
        entry.coverage_strategy = (
            f"Split at {mid:.0f}s: A-angle ({entry.primary_viewer_focus} close-up) "
            f"then B-angle (environment wide)"
        )
        entry.recommended_shot_duration_strategy = "Two-Shot"
    elif entry.section.lower().replace(" ", "_") == "chorus" and duration > 6.0:
        entry.coverage_strategy = (
            f"Montage: {entry.primary_viewer_focus} -> wide -> {entry.primary_viewer_focus}"
        )
        entry.recommended_shot_duration_strategy = "Fast Montage"
    elif duration > 8.0:
        entry.coverage_strategy = "Slow cinematic hold"
        entry.recommended_shot_duration_strategy = "Slow Cinematic"
    else:
        entry.coverage_strategy = "Single take"
        entry.recommended_shot_duration_strategy = "Single"

    return entry


def candidate_count_from_weight(visual_weight: int) -> tuple[int, int]:
    """Return (min_candidates, max_candidates) for reference image generation.

    Weight 1-3 -> (2, 3). Weight 4-6 -> (3, 4). Weight 7-10 -> (5, 8).

    Args:
        visual_weight: Visual weight score 1-10.

    Returns:
        Tuple of (min_candidates, max_candidates).
    """
    if visual_weight <= 0:
        visual_weight = 1
    if visual_weight > 10:
        visual_weight = 10

    if visual_weight <= 3:
        return (2, 3)
    elif visual_weight <= 6:
        return (3, 4)
    else:
        return (5, 8)


def build_storyboard(
    beats: list[BeatEntry],
    treatment_path: str,
    output_path: str,
) -> list[StoryboardEntry]:
    """Orchestrate full storyboard classification.

    Run classification pipeline, generate all fields, write storyboard.md.

    Args:
        beats: List of BeatEntry objects.
        treatment_path: Path to the director's treatment markdown file.
        output_path: Path for the output storyboard.md file.

    Returns:
        List of StoryboardEntry objects.
    """
    # Step 1: Classify song structure
    beats = classify_song_structure(beats)

    # Step 2: Assign visual weights
    entries = assign_visual_weight(beats)

    # Step 3: Assign performance focus, shot scale, movement, coverage
    for entry in entries:
        entry = assign_performance_focus(entry)
        entry = assign_shot_scale_and_movement(entry)
        entry = assign_coverage_strategy(entry)

    # Step 4: Generate rationale based on section + lyrics
    for entry in entries:
        entry.rationale = _generate_rationale(entry)

    # Step 5: Write storyboard.md
    _write_storyboard_md(entries, treatment_path, output_path)

    return entries


# ── Private helpers ──


def _find_first_lyric(beats: list[BeatEntry]) -> int:
    """Find index of first beat with lyrics."""
    for i, beat in enumerate(beats):
        if beat.lyrics:
            return i
    return len(beats)


def _find_last_lyric(beats: list[BeatEntry]) -> int:
    """Find index of last beat with lyrics."""
    for i in range(len(beats) - 1, -1, -1):
        if beats[i].lyrics:
            return i
    return -1


def _is_pre_chorus_position(idx: int, beats: list[BeatEntry]) -> bool:
    """Check if this beat is in a pre-chorus position.

    Pre-chorus typically appears right before a chorus section and
    shows building energy. Only classify as pre-chorus if there are
    at least 2 consecutive building beats before the chorus, to avoid
    misclassifying a single verse beat as pre-chorus.
    """
    if idx + 2 >= len(beats):
        return False
    # Check if next two beats are high energy or chorus
    next_beat = beats[idx + 1]
    next_next = beats[idx + 2]
    is_chorus_ahead = (
        next_beat.energy == "high" or next_beat.section == "chorus"
    ) and (
        next_next.energy == "high" or next_next.section == "chorus"
    )
    return is_chorus_ahead


def _get_weight_range(section: str) -> tuple[int, int]:
    """Get the visual weight range for a section type."""
    ranges = {
        "chorus": _WEIGHT_HIGH,
        "bridge": (6, 8),
        "verse": _WEIGHT_MEDIUM,
        "pre_chorus": (5, 7),
        "intro": _WEIGHT_LOW,
        "outro": _WEIGHT_LOW,
        "instrumental": _WEIGHT_INSTRUMENTAL,
    }
    return ranges.get(section, _WEIGHT_MEDIUM)


def _compute_weight(beat: BeatEntry, weight_range: tuple[int, int]) -> int:
    """Compute visual weight within the given range.

    Factors: energy level, lyrics density, position in song.
    """
    min_w, max_w = weight_range
    mid = (min_w + max_w) / 2

    if beat.energy == "high":
        return max_w
    elif beat.energy == "low":
        return min_w
    else:
        return int(mid)


def _get_default_focus(section: str) -> dict[str, int]:
    """Get default performance focus for a section type."""
    defaults = _FOCUS_MAP.get(section, _FOCUS_VERSE)
    return dict(defaults)  # Return copy


def _get_default_focus_target(section: str) -> str:
    """Get default primary viewer focus for a section type."""
    targets = {
        "chorus": "Face",
        "verse": "Face",
        "pre_chorus": "Face",
        "intro": "Landscape",
        "bridge": "Eyes",
        "outro": "Landscape",
        "instrumental": "Landscape",
    }
    return targets.get(section, "Face")


def _map_energy_to_intensity(energy: str) -> int:
    """Map energy label to emotional intensity 1-5."""
    mapping = {"low": 2, "medium": 3, "high": 5}
    return mapping.get(energy, 3)


def _map_energy_to_visual(energy: str) -> int:
    """Map energy label to visual energy 1-5."""
    mapping = {"low": 2, "medium": 3, "high": 4}
    return mapping.get(energy, 3)


def _generate_rationale(entry: StoryboardEntry) -> str:
    """Generate directorial rationale linking choice to lyrics/story."""
    section = entry.section
    focus = entry.primary_viewer_focus

    rationales = {
        "intro": f"Establishing shot sets mood before singer appears",
        "verse": f"Narrative storytelling with {focus.lower()} focus supports lyrical content",
        "pre_chorus": f"Building intensity toward emotional peak",
        "chorus": f"Hero moment — emotional peak demands intimate {focus.lower()} shot",
        "bridge": f"Emotional contrast — unique visual approach for section shift",
        "outro": f"Resolution and fade — pulling back to wider perspective",
        "instrumental": f"Atmospheric visual breathing room between lyrical sections",
    }
    return rationales.get(section, f"Visual approach supports {section} section")


def _write_storyboard_md(
    entries: list[StoryboardEntry],
    treatment_path: str,
    output_path: str,
) -> None:
    """Write storyboard.md with song structure summary and details table."""
    sections = {}
    for entry in entries:
        sections.setdefault(entry.section, []).append(entry)

    lines = ["# Visual Storyboard", "", "## Song Structure"]
    for section_name, section_entries in sections.items():
        avg_weight = sum(e.visual_weight for e in section_entries) // len(
            section_entries
        )
        start_time = section_entries[0].start
        end_time = section_entries[-1].end
        lines.append(
            f"- **{section_name.title()}** "
            f"({_fmt_time(start_time)} - {_fmt_time(end_time)}): "
            f"{len(section_entries)} beats, avg weight {avg_weight}"
        )

    lines.append("")
    lines.append("## Storyboard Details")
    lines.append("")
    lines.append(
        "| Beat | Time | Section | Importance | Weight | "
        "| Singer% | Narrative% | B-roll% | Env% | Symbolic% | Montage% | "
        "| Emo | Energy | Focus | Duration | Scale | Camera | "
        "| Coverage | Rationale |"
    )
    lines.append(
        "|------|------|---------|------------|--------|"
        "|---------|------------|---------|------|-----------|----------|"
        "|-----|--------|-------|----------|-------|--------|"
        "|----------|-----------|"
    )

    for entry in entries:
        pf = entry.performance_focus
        lines.append(
            f"| B{entry.beat_index + 1:02d} | "
            f"{_fmt_time(entry.start)}-{_fmt_time(entry.end)} | "
            f"{entry.section} | {entry.narrative_importance} | "
            f"{entry.visual_weight} | "
            f"{pf.get('singer_pct', 0)} | {pf.get('narrative_pct', 0)} | "
            f"{pf.get('broll_pct', 0)} | {pf.get('environment_pct', 0)} | "
            f"{pf.get('symbolic_pct', 0)} | {pf.get('montage_pct', 0)} | "
            f"{entry.emotional_intensity} | {entry.visual_energy} | "
            f"{entry.primary_viewer_focus} | "
            f"{entry.recommended_shot_duration_strategy} | "
            f"{entry.recommended_shot_scale} | "
            f"{entry.camera_movement_intensity} | "
            f"{entry.coverage_strategy} | {entry.rationale} |"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    logger.info("Storyboard written: %s (%d entries)", output, len(entries))


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS time string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"
