#!/usr/bin/env python3
"""Pre-generation validation layer — Stage 8 of the music video pipeline.

Runs all prerequisite checks before GPU-intensive generation begins.
Validates storyboard completeness, timeline coverage, aspect ratio,
and beat-by-beat content (duration + continuity).

Module is kept <= 300 lines per STYLE.md.
Implementation split:
- mv_validation_prerequisites.py: 15 prerequisite check functions
- mv_validation_content.py: storyboard + beat content validation
"""

from __future__ import annotations

import logging
from pathlib import Path

from mv_validation_prerequisites import (
    _check_source_audio,
    _check_portrait,
    _check_lyrics,
    _check_transcript_structure,
    _check_transcript_source,
    _check_refs_manifest,
    _check_prompts,
    _check_continuity,
    _check_treatment,
    _check_beat_sheet,
    _check_shot_list,
    _check_storyboard,
    _check_timeline,
    _check_aspect_ratio,
    _check_fsm_state,
    _check_beat_content,
)
from mv_validation_content import (
    ValidationReport,
    validate_storyboard_completeness,
    check_timeline_coverage,
    check_aspect_ratio,
    validate_beat_content,
)

logger = logging.getLogger(__name__)

# Re-export constants for test compatibility
_MIN_BEAT_DURATION_S: float = 2.0
_MAX_BEAT_DURATION_S: float = 18.0
_TIMELINE_GAP_THRESHOLD_S: float = 0.5
_STORYBOARD_REQUIRED_FIELDS: list[str] = [
    "visual_weight",
    "narrative_importance",
    "performance_focus",
    "emotional_intensity",
    "visual_energy",
    "primary_viewer_focus",
    "recommended_shot_duration_strategy",
    "recommended_shot_scale",
    "camera_movement_intensity",
    "coverage_strategy",
    "rationale",
]


def validate_prerequisites(project_dir: Path) -> ValidationReport:
    """Run all prerequisite checks and return a ValidationReport.

    Checks 15 prerequisite items plus beat-by-beat content validation.
    Generation must NOT start if validation fails.

    Args:
        project_dir: Path to the music video project directory.

    Returns:
        ValidationReport with pass/fail/missing classification.
    """
    report = ValidationReport()

    _check_source_audio(project_dir, report)
    _check_portrait(project_dir, report)
    _check_lyrics(project_dir, report)
    _check_transcript_structure(project_dir, report)
    _check_transcript_source(project_dir, report)
    _check_refs_manifest(project_dir, report)
    _check_prompts(project_dir, report)
    _check_continuity(project_dir, report)
    _check_treatment(project_dir, report)
    _check_beat_sheet(project_dir, report)
    _check_shot_list(project_dir, report)
    _check_storyboard(project_dir, report)
    _check_timeline(project_dir, report)
    _check_aspect_ratio(project_dir, report)
    _check_fsm_state(project_dir, report)
    _check_beat_content(project_dir, report)

    return report
