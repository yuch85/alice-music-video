#!/usr/bin/env python3
"""Content validation helpers for mv_validation.

Storyboard completeness, timeline coverage, aspect ratio, and
ValidationReport dataclass. Beat content validation in mv_validation_beat.py.
Kept separate from mv_validation.py to stay within 300 LOC per file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Re-export from beat module
from mv_validation_beat import (
    _count_beats,
    validate_beat_content,
)

_ASPECT_RATIO_TOLERANCE = 0.02


@dataclass
class ValidationReport:
    """Structured validation report with pass/fail/warning/missing classification.

    Properties:
        overall_pass: True only if no failures and no missing items.
          Warnings do not block generation.
        is_ready_for_generation: Alias for overall_pass.
    """

    pass_checks: list[str] = field(default_factory=list)
    fail_checks: list[str] = field(default_factory=list)
    warning_checks: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        """Return True only if no failures and no missing items.

        Warnings do not block generation — they are informational.
        """
        return len(self.fail_checks) == 0 and len(self.missing_items) == 0

    @property
    def is_ready_for_generation(self) -> bool:
        """Return True only if overall_pass."""
        return self.overall_pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "pass_checks": self.pass_checks,
            "fail_checks": self.fail_checks,
            "warning_checks": self.warning_checks,
            "missing_items": self.missing_items,
            "overall_pass": self.overall_pass,
        }


def validate_storyboard_completeness(
    storyboard_path: Path, beat_count: int
) -> list[str]:
    """Parse storyboard.md and verify every beat has all required fields.

    Args:
        storyboard_path: Path to storyboard.md.
        beat_count: Expected number of beats.

    Returns:
        List of gap descriptions (empty if all fields present).
    """
    if not storyboard_path.exists():
        return ["storyboard.md not found"]

    content = storyboard_path.read_text()
    lines = content.strip().split("\n")

    data_rows: list[list[str]] = []
    current_row: list[str] | None = None
    seen_separator = False
    for line in lines:
        if not line.startswith("|"):
            if seen_separator and current_row:
                data_rows.append(current_row)
                current_row = None
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]

        is_separator = bool(cells) and all(
            re.match(r"^-+$", c) for c in cells if c
        )
        if is_separator:
            seen_separator = True
            continue
        if not seen_separator:
            continue

        if cells and re.match(r"^B\d+", cells[0]):
            if current_row:
                data_rows.append(current_row)
            current_row = list(cells)
        elif current_row is not None:
            current_row.extend(cells)

    if current_row:
        data_rows.append(current_row)

    gaps: list[str] = []
    for row in data_rows:
        beat_id = row[0] if row else "unknown"

        if len(row) <= 4 or not row[4]:
            gaps.append(f"Beat {beat_id} missing visual_weight")

        if len(row) <= 3 or not row[3]:
            gaps.append(f"Beat {beat_id} missing narrative_importance")

        if len(row) > 10:
            try:
                pf_sum = sum(int(row[i]) for i in range(5, 11))
                if pf_sum != 100:
                    gaps.append(
                        f"Beat {beat_id} performance_focus sums to "
                        f"{pf_sum}, expected 100"
                    )
            except (ValueError, IndexError):
                gaps.append(f"Beat {beat_id} performance_focus invalid")
        else:
            gaps.append(f"Beat {beat_id} missing performance_focus")

        if len(row) <= 11 or not row[11]:
            gaps.append(f"Beat {beat_id} missing emotional_intensity")

        if len(row) <= 12 or not row[12]:
            gaps.append(f"Beat {beat_id} missing visual_energy")

        if len(row) <= 13 or not row[13]:
            gaps.append(f"Beat {beat_id} missing primary_viewer_focus")

        if len(row) <= 14 or not row[14]:
            gaps.append(
                f"Beat {beat_id} missing recommended_shot_duration_strategy"
            )

        if len(row) <= 15 or not row[15]:
            gaps.append(
                f"Beat {beat_id} missing recommended_shot_scale"
            )

        if len(row) <= 16 or not row[16]:
            gaps.append(
                f"Beat {beat_id} missing camera_movement_intensity"
            )

        if len(row) <= 17 or not row[17]:
            gaps.append(f"Beat {beat_id} missing coverage_strategy")

        if len(row) <= 18 or not row[18]:
            gaps.append(f"Beat {beat_id} missing rationale")

    if len(data_rows) < beat_count:
        gaps.append(
            f"Storyboard has {len(data_rows)} beats, expected {beat_count}"
        )

    return gaps


def check_timeline_coverage(
    transcript_path: Path, audio_duration: float, gap_threshold: float = 0.5
) -> bool:
    """Verify segments cover the full timeline contiguously.

    Args:
        transcript_path: Path to transcript.json.
        audio_duration: Total audio duration in seconds.
        gap_threshold: Maximum allowed gap between segments (default 0.5s).

    Returns:
        True if segments tile 0 -> audio_duration with no gaps.
    """
    if not transcript_path.exists():
        return False

    try:
        data = json.loads(transcript_path.read_text())
        segments = data.get("segments", [])
        if not segments:
            return False

        sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))

        if sorted_segs[0].get("start", 0) > gap_threshold:
            return False

        for i in range(1, len(sorted_segs)):
            gap = sorted_segs[i].get("start", 0) - sorted_segs[i - 1].get("end", 0)
            if gap > gap_threshold:
                return False

        last_end = sorted_segs[-1].get("end", 0)
        if audio_duration - last_end > gap_threshold:
            return False

        return True
    except (json.JSONDecodeError, KeyError):
        return False


def check_aspect_ratio(
    width: int,
    height: int,
    expected_ratio: tuple[int, int] = (16, 9),
    tolerance: float = 0.02,
) -> bool:
    """Validate resolution matches aspect ratio within tolerance.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        expected_ratio: Expected (width, height) ratio, e.g. (16, 9).
        tolerance: Maximum deviation as fraction of expected ratio.

    Returns:
        True if actual ratio is within tolerance of expected ratio.
    """
    if height == 0:
        return False

    actual_ratio = width / height
    expected_decimal = expected_ratio[0] / expected_ratio[1]
    deviation = abs(actual_ratio - expected_decimal) / expected_decimal

    return deviation <= tolerance
