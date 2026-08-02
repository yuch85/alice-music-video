#!/usr/bin/env python3
"""Prerequisite check functions for mv_validation.

Each function checks one prerequisite and updates the ValidationReport.
Kept separate from mv_validation.py to stay within 300 LOC per file.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from mv_validation_content import (
    ValidationReport,
    validate_storyboard_completeness,
    check_timeline_coverage,
    check_aspect_ratio,
    validate_beat_content,
    _count_beats,
)

logger = logging.getLogger(__name__)

_PORTRAIT_MIN_SIZE_BYTES = 10_000
_TIMELINE_GAP_THRESHOLD_S = 0.5


def _find_audio_file(project_dir: Path) -> Path | None:
    """Find the source audio file in the project directory."""
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        for f in project_dir.glob(f"*{ext}"):
            if f.is_file() and f.stat().st_size > 0:
                return f
    return None


def _find_portrait_file(project_dir: Path) -> Path | None:
    """Find the portrait file in the project directory."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        for f in project_dir.glob(f"portrait*{ext}"):
            if f.is_file():
                return f
    for ext in (".jpg", ".jpeg", ".png"):
        for f in project_dir.glob(f"*{ext}"):
            if f.is_file() and f.stat().st_size > _PORTRAIT_MIN_SIZE_BYTES:
                return f
    return None


def _check_source_audio(project_dir: Path, report: ValidationReport) -> None:
    """Check 1: Source audio file exists and is readable."""
    audio = _find_audio_file(project_dir)
    if audio and audio.stat().st_size > 0:
        report.pass_checks.append(f"Source audio file exists: {audio.name}")
    else:
        report.missing_items.append(
            "Source audio file — re-run /music_video (Stage 0) or place audio in project folder"
        )


def _check_portrait(project_dir: Path, report: ValidationReport) -> None:
    """Check 2: Portrait file exists and is a valid image."""
    portrait = _find_portrait_file(project_dir)
    if portrait and portrait.stat().st_size >= _PORTRAIT_MIN_SIZE_BYTES:
        report.pass_checks.append(f"Portrait file exists: {portrait.name}")
    else:
        report.missing_items.append("Portrait file (min 10KB)")


def _check_lyrics(project_dir: Path, report: ValidationReport) -> None:
    """Check 3: Lyrics/transcript available."""
    transcript = project_dir / "transcript.json"
    lyrics_dir = project_dir / "lyrics"
    has_transcript = transcript.exists() and transcript.stat().st_size > 0
    has_lyrics = lyrics_dir.exists() and any(lyrics_dir.iterdir())

    if has_transcript or has_lyrics:
        source = "transcript.json" if has_transcript else "lyrics/"
        report.pass_checks.append(f"Lyrics/transcript available: {source}")
    else:
        report.missing_items.append("Lyrics or transcript.json")


def _check_transcript_structure(project_dir: Path, report: ValidationReport) -> None:
    """Check 4: Transcript is valid JSON with contiguous segment coverage."""
    transcript = project_dir / "transcript.json"
    if not transcript.exists():
        return

    try:
        data = json.loads(transcript.read_text())
        segments = data.get("segments", [])
        if not segments:
            report.fail_checks.append(
                "transcript.json has no segments — rollback to BEATS stage"
            )
            return

        sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))
        for i in range(1, len(sorted_segs)):
            gap = sorted_segs[i].get("start", 0) - sorted_segs[i - 1].get("end", 0)
            if gap > _TIMELINE_GAP_THRESHOLD_S:
                report.fail_checks.append(
                    f"transcript.json gap > {_TIMELINE_GAP_THRESHOLD_S}s "
                    f"between segments {i - 1} and {i}"
                )
                return

        report.pass_checks.append(
            f"Transcript structure valid: {len(segments)} segments, contiguous"
        )
    except (json.JSONDecodeError, KeyError) as exc:
        report.fail_checks.append(
            f"transcript.json malformed: {exc} — rollback to BEATS stage"
        )


def _check_transcript_source(project_dir: Path, report: ValidationReport) -> None:
    """Check 4b: Transcript audio_source field indicates which file was transcribed.

    If the transcript was generated from a stem file (vocals.wav), timestamps
    are song-relative and need a prologue offset to align with video time.
    If generated from the full video, timestamps are already in video time.
    This check warns (does not block) when the source is a stem file, because
    the downstream beat sheet stage handles the offset.
    """
    transcript = project_dir / "transcript.json"
    if not transcript.exists():
        return

    try:
        data = json.loads(transcript.read_text())
        # Check both field names — pipeline uses "source", modotte uses "audio_source"
        audio_source = data.get("audio_source") or data.get("source", "UNKNOWN")

        if "vocals" in audio_source or "stem" in audio_source:
            report.warning_checks.append(
                f"Transcript source is a stem file ({audio_source}) — "
                f"timestamps are song-relative. Ensure prologue offset is applied "
                f"in beat sheet stage."
            )
        elif data.get("segments"):
            report.pass_checks.append(
                f"Transcript source: {audio_source} (timestamps aligned)"
            )
    except (json.JSONDecodeError, KeyError):
        pass  # Non-blocking warning check


def _check_refs_manifest(project_dir: Path, report: ValidationReport) -> None:
    """Check 5: Approved reference images exist."""
    manifest = project_dir / "refs" / "manifest.json"
    if not manifest.exists():
        report.missing_items.append("refs/manifest.json")
        return

    try:
        data = json.loads(manifest.read_text())
        beats = data.get("beats", [])
        report.pass_checks.append(f"Refs manifest: {len(beats)} beat entries")
    except json.JSONDecodeError as exc:
        report.fail_checks.append(f"refs/manifest.json malformed: {exc}")


def _check_prompts(project_dir: Path, report: ValidationReport) -> None:
    """Check 6: Prompts exist for all beats."""
    prompts_dir = project_dir / "prompts"
    if not prompts_dir.exists():
        report.missing_items.append("prompts/ directory")
        return

    prompt_files = list(prompts_dir.glob("beat_*.md"))
    if prompt_files:
        report.pass_checks.append(f"Prompts: {len(prompt_files)} beat prompt files")
    else:
        report.missing_items.append("prompts/beat_*.md files")


def _check_continuity(project_dir: Path, report: ValidationReport) -> None:
    """Check 7: Continuity bible exists and is non-empty."""
    bible = project_dir / "continuity_bible.md"
    if bible.exists() and bible.stat().st_size > 0:
        report.pass_checks.append("Continuity bible exists")
    else:
        report.missing_items.append("continuity_bible.md")


def _check_treatment(project_dir: Path, report: ValidationReport) -> None:
    """Check 8: Treatment exists and is non-empty."""
    treatment = project_dir / "director_treatment.md"
    if treatment.exists() and treatment.stat().st_size > 0:
        report.pass_checks.append("Director treatment exists")
    else:
        report.missing_items.append("director_treatment.md")


def _check_beat_sheet(project_dir: Path, report: ValidationReport) -> None:
    """Check 9: Beat sheet exists."""
    beats = project_dir / "beat_sheet.md"
    if beats.exists():
        report.pass_checks.append("Beat sheet exists")
    else:
        report.missing_items.append("beat_sheet.md")


def _check_shot_list(project_dir: Path, report: ValidationReport) -> None:
    """Check 10: Shot list exists."""
    shots = project_dir / "shot_list.md"
    if shots.exists():
        report.pass_checks.append("Shot list exists")
    else:
        report.missing_items.append("shot_list.md")


def _check_storyboard(project_dir: Path, report: ValidationReport) -> None:
    """Check 11: Storyboard exists and is complete."""
    storyboard = project_dir / "storyboard.md"
    if not storyboard.exists():
        report.missing_items.append("storyboard.md")
        return

    beat_count = _count_beats(project_dir)
    gaps = validate_storyboard_completeness(storyboard, beat_count)
    if gaps:
        for gap in gaps[:5]:
            report.fail_checks.append(f"Storyboard gap: {gap}")
    else:
        report.pass_checks.append(
            f"Storyboard complete: {beat_count} beats, all fields populated"
        )


def _check_timeline(project_dir: Path, report: ValidationReport) -> None:
    """Check 12: Timeline coverage (segments tile 0->duration contiguously)."""
    transcript = project_dir / "transcript.json"
    if not transcript.exists():
        return

    try:
        data = json.loads(transcript.read_text())
        segments = data.get("segments", [])
        if not segments:
            return

        sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))
        audio_duration = sorted_segs[-1].get("end", 0)
        if check_timeline_coverage(transcript, audio_duration):
            report.pass_checks.append(
                f"Timeline coverage: 0 -> {audio_duration:.1f}s contiguous"
            )
        else:
            report.fail_checks.append("Timeline has gaps > 0.5s")
    except (json.JSONDecodeError, KeyError):
        pass


def _check_aspect_ratio(project_dir: Path, report: ValidationReport) -> None:
    """Check 13: Aspect ratio matches target resolution."""
    if check_aspect_ratio(1920, 1080, (16, 9)):
        report.pass_checks.append("Aspect ratio: 1920x1080 matches 16:9")
    else:
        report.fail_checks.append("Aspect ratio mismatch for 1920x1080")


def _check_fsm_state(project_dir: Path, report: ValidationReport) -> None:
    """Check 14: FSM state is PROMPTS or later."""
    index = project_dir / "index.md"
    if not index.exists():
        report.missing_items.append("index.md (FSM state)")
        return

    content = index.read_text()
    match = re.search(r"state:\s*(\w+)", content)
    if not match:
        report.fail_checks.append("index.md has no FSM state")
        return

    state = match.group(1)
    valid_states = ("PROMPTS", "VALIDATED", "GENERATING", "QC", "COMPLETE")
    if state in valid_states:
        report.pass_checks.append(f"FSM state: {state} (all creative stages approved)")
    else:
        report.fail_checks.append(
            f"FSM state is {state} — must be PROMPTS or later"
        )


def _check_beat_content(project_dir: Path, report: ValidationReport) -> None:
    """Check 15: Beat-by-beat content validation."""
    beats_path = project_dir / "beat_sheet.md"
    prompts_dir = project_dir / "prompts"
    continuity_path = project_dir / "continuity_bible.md"

    if not beats_path.exists():
        return

    violations = validate_beat_content(beats_path, prompts_dir, continuity_path)
    if violations:
        for v in violations[:5]:
            report.fail_checks.append(f"Beat content: {v}")
    else:
        report.pass_checks.append("Beat content: all durations valid, continuity OK")
