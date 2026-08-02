#!/usr/bin/env python3
"""Beat sheet generation helper — Stage 4 of the music video pipeline.

Translates Whisper timestamps into a structured beat sheet with narrative,
emotional, and visual purpose per beat. Supports timeline gap-filling
and clip duration splitting.

Module is kept <= 400 lines per STYLE.md (YC-approved 400 ceiling deviation
from the 300 default).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mv_utils import parse_markdown_table  # noqa: F401 — re-export for consumers

logger = logging.getLogger(__name__)

# Constants
_MAX_CLIP_DURATION_S = 18.0
_GAP_THRESHOLD_S = 0.5


@dataclass
class BeatEntry:
    """Single beat in the beat sheet.

    12 PRD fields plus `section` for song structure tracking.
    """

    start: float
    end: float
    duration: float
    lyrics: str
    narrative_purpose: str
    emotional_purpose: str
    visual_purpose: str
    energy: str
    dominant_character: str
    transition_from: str
    transition_to: str
    detailed_notes: str
    section: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeatEntry:
        """Deserialize from a plain dict."""
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            duration=float(data.get("duration", 0.0)),
            lyrics=str(data.get("lyrics", "")),
            narrative_purpose=str(data.get("narrative_purpose", "")),
            emotional_purpose=str(data.get("emotional_purpose", "")),
            visual_purpose=str(data.get("visual_purpose", "")),
            energy=str(data.get("energy", "")),
            dominant_character=str(data.get("dominant_character", "--")),
            transition_from=str(data.get("transition_from", "--")),
            transition_to=str(data.get("transition_to", "--")),
            detailed_notes=str(data.get("detailed_notes", "")),
            section=str(data.get("section", "")),
        )


def parse_whisper_timestamps(transcript_path: str) -> list[BeatEntry]:
    """Read transcript.json and convert ClipSegment entries to BeatEntry objects.

    Initially populates timing and lyrics fields; narrative/emotional/visual
    purposes are empty strings to be filled by the LLM later.

    Args:
        transcript_path: Path to the Whisper transcript JSON file.

    Returns:
        List of BeatEntry objects sorted by start time.
    """
    data = json.loads(Path(transcript_path).read_text())
    segments = data.get("segments", [])

    beats: list[BeatEntry] = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg.get("text", "")).strip()
        beats.append(BeatEntry(
            start=start,
            end=end,
            duration=end - start,
            lyrics=text,
            narrative_purpose="",
            emotional_purpose="",
            visual_purpose="",
            energy="",
            dominant_character="--",
            transition_from="--",
            transition_to="--",
            detailed_notes="",
            section="",
        ))

    beats.sort(key=lambda b: b.start)
    logger.info("Parsed %d beat entries from transcript", len(beats))
    return beats


def group_beats_by_section(
    beats: list[BeatEntry],
) -> dict[str, list[BeatEntry]]:
    """Group beats into song structure sections.

    Heuristics:
    - intro: beats before the first lyric-bearing beat
    - verse: beats with moderate-length lyrics
    - chorus: beats with short, repetitive lyrics (high energy)
    - bridge: beats with contrast (emotional shift)
    - instrumental: beats with no lyrics
    - outro: beats after the last lyric-bearing beat

    Args:
        beats: List of BeatEntry objects.

    Returns:
        Dict mapping section name to list of BeatEntry objects.
    """
    groups: dict[str, list[BeatEntry]] = {}

    for beat in beats:
        section = beat.section
        if not section:
            # Auto-classify based on content
            if not beat.lyrics:
                section = "instrumental"
            elif beat.start < 5.0 and not beat.lyrics:
                section = "intro"
            else:
                section = "verse"
            beat.section = section

        groups.setdefault(section, []).append(beat)

    logger.info("Grouped beats into sections: %s", list(groups.keys()))
    return groups


def assign_energy_level(
    beats: list[BeatEntry],
    audio_duration: float,
) -> list[BeatEntry]:
    """Assign energy labels based on segment position and lyrics density.

    Bookends tend to be lower energy, middle builds. This is a heuristic
    that the LLM refines.

    Args:
        beats: List of BeatEntry objects.
        audio_duration: Total audio duration in seconds.

    Returns:
        List of BeatEntry objects with energy labels assigned.
    """
    if not beats or audio_duration <= 0:
        return beats

    for i, beat in enumerate(beats):
        if beat.energy:
            continue  # Already assigned

        # Position-based energy: bookends low, middle high
        pos = beat.start / audio_duration
        lyrics_density = len(beat.lyrics) / max(beat.duration, 0.1)

        if beat.section == "chorus":
            beat.energy = "high"
        elif beat.section in ("intro", "outro"):
            beat.energy = "low"
        elif beat.section == "bridge":
            beat.energy = "high"
        elif pos < 0.2 or pos > 0.8:
            beat.energy = "low"
        elif lyrics_density > 1.0:
            beat.energy = "high"
        else:
            beat.energy = "medium"

    return beats


def fill_timeline_gaps(
    beats: list[BeatEntry],
    audio_duration: float,
) -> list[BeatEntry]:
    """Insert instrumental BeatEntry objects for gaps in the timeline.

    Ensures the beat sheet covers the full audio timeline contiguously
    from 0 to audio_duration.

    Args:
        beats: List of BeatEntry objects sorted by start time.
        audio_duration: Total audio duration in seconds.

    Returns:
        List of BeatEntry objects with gaps filled.
    """
    if not beats or audio_duration <= 0:
        return beats

    result: list[BeatEntry] = []
    prev_end = 0.0

    for beat in beats:
        gap = beat.start - prev_end
        if gap > _GAP_THRESHOLD_S:
            result.append(BeatEntry(
                start=prev_end,
                end=beat.start,
                duration=gap,
                lyrics="(instrumental)",
                narrative_purpose="",
                emotional_purpose="",
                visual_purpose="",
                energy="low",
                dominant_character="--",
                transition_from="--",
                transition_to="--",
                detailed_notes="",
                section="instrumental",
            ))
        result.append(beat)
        prev_end = beat.end

    # Fill trailing gap to audio_duration
    trailing = audio_duration - prev_end
    if trailing > _GAP_THRESHOLD_S:
        result.append(BeatEntry(
            start=prev_end,
            end=audio_duration,
            duration=trailing,
            lyrics="(instrumental)",
            narrative_purpose="",
            emotional_purpose="",
            visual_purpose="",
            energy="low",
            dominant_character="--",
            transition_from="--",
            transition_to="--",
            detailed_notes="",
            section="instrumental",
        ))

    logger.info(
        "Filled timeline gaps: %d -> %d beats (duration %.1fs)",
        len(beats), len(result), audio_duration,
    )
    return result


def split_long_clips(
    beats: list[BeatEntry],
    max_duration: float = _MAX_CLIP_DURATION_S,
) -> list[BeatEntry]:
    """Split any beat exceeding max_duration into sub-beats.

    Each sub-beat inherits the parent's narrative/emotional/visual purpose
    and dominant_character. Transition fields maintain continuity.

    Args:
        beats: List of BeatEntry objects.
        max_duration: Maximum clip duration in seconds (default 18s).

    Returns:
        List of BeatEntry objects with long beats split.
    """
    if max_duration <= 0:
        return list(beats)

    result: list[BeatEntry] = []
    for beat in beats:
        if beat.duration <= max_duration:
            result.append(beat)
            continue

        num_sub = int(beat.duration // max_duration) + (
            1 if beat.duration % max_duration > 1e-9 else 0
        )
        for k in range(num_sub):
            sub_start = beat.start + k * max_duration
            sub_end = (
                beat.start + (k + 1) * max_duration
                if k < num_sub - 1
                else beat.end
            )
            sub_duration = sub_end - sub_start

            # Set transition continuity
            if k == 0:
                t_from = beat.transition_from
            else:
                t_from = "continuation"
            if k == num_sub - 1:
                t_to = beat.transition_to
            else:
                t_to = "continuation"

            result.append(BeatEntry(
                start=sub_start,
                end=sub_end,
                duration=sub_duration,
                lyrics=beat.lyrics if k == 0 else "",
                narrative_purpose=beat.narrative_purpose,
                emotional_purpose=beat.emotional_purpose,
                visual_purpose=beat.visual_purpose,
                energy=beat.energy,
                dominant_character=beat.dominant_character,
                transition_from=t_from,
                transition_to=t_to,
                detailed_notes=beat.detailed_notes,
                section=beat.section,
            ))

    logger.info(
        "Split long clips: %d -> %d beats (max %.1fs)",
        len(beats), len(result), max_duration,
    )
    return result


def build_beat_sheet(
    beats: list[BeatEntry],
    treatment_path: str,
    output_path: str,
) -> Path:
    """Orchestrate beat sheet creation.

    Read treatment for narrative context, populate narrative/emotional/visual
    purpose fields via LLM prompt, and write beat_sheet.md.

    Args:
        beats: List of BeatEntry objects.
        treatment_path: Path to the director's treatment markdown file.
        output_path: Path for the output beat_sheet.md file.

    Returns:
        Path to the written beat_sheet.md file.
    """
    treatment_text = ""
    if Path(treatment_path).is_file():
        treatment_text = Path(treatment_path).read_text().strip()

    # Group beats by section for the output
    sections = group_beats_by_section(beats)

    # Build markdown output
    lines = ["# Beat Sheet", "", "## Sections"]
    for section_name, section_beats in sections.items():
        start_time = section_beats[0].start if section_beats else 0.0
        end_time = section_beats[-1].end if section_beats else 0.0
        lines.append(
            f"- **{section_name.title()}** "
            f"({_fmt_time(start_time)} - {_fmt_time(end_time)}): "
            f"{len(section_beats)} beats"
        )

    lines.append("")
    lines.append("## Beat Details")
    lines.append("")
    lines.append(
        "| Time | Duration | Lyrics | Narrative | Emotional | Visual | "
        "| Energy | Character | Transition From | Transition To | Detailed Notes |"
    )
    lines.append(
        "|------|----------|----------|-----------|-----------|--------|"
        "|--------|-----------|-----------------|---------------|--------------|"
    )

    for beat in beats:
        lines.append(
            f"| {_fmt_time(beat.start)}-{_fmt_time(beat.end)} | "
            f"{beat.duration:.1f}s | {beat.lyrics} | "
            f"{beat.narrative_purpose} | {beat.emotional_purpose} | "
            f"{beat.visual_purpose} | {beat.energy} | "
            f"{beat.dominant_character} | {beat.transition_from} | "
            f"{beat.transition_to} | {beat.detailed_notes} |"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    logger.info("Beat sheet written: %s (%d beats)", output, len(beats))
    return output


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS time string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"
