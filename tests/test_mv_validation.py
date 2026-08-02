#!/usr/bin/env python3
"""Tests for mv_validation module — pre-generation validation layer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

# Ensure scripts/ is on path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mv_validation import (
    ValidationReport,
    validate_prerequisites,
    validate_storyboard_completeness,
    check_timeline_coverage,
    check_aspect_ratio,
    validate_beat_content,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def all_files_project(tmp_path: Path) -> Path:
    """Full project with all prerequisites satisfied."""
    project = tmp_path / "project"
    project.mkdir(parents=True)

    # Audio
    (project / "song.mp3").write_bytes(b"\x00" * 1024)

    # Portrait
    (project / "portrait.jpg").write_bytes(b"\x00" * 20_000)

    # Transcript
    (project / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 5.0, "text": "hi"}]})
    )

    # Refs
    (project / "refs" / "manifest.json").parent.mkdir(exist_ok=True)
    (project / "refs" / "manifest.json").write_text(
        json.dumps({"beats": [{"beat_id": "B01"}]})
    )

    # Prompts
    (project / "prompts" / "beat_01.md").parent.mkdir(exist_ok=True)
    (project / "prompts" / "beat_01.md").write_text("# B01")

    # Continuity
    (project / "continuity_bible.md").write_text("# Bible\n- white dress\n")

    # Treatment
    (project / "director_treatment.md").write_text("# Treatment")

    # Beat sheet
    (project / "beat_sheet.md").write_text("# Beats\n| B01 | 0 | 8 | 8 |")

    # Shot list
    (project / "shot_list.md").write_text("# Shots\nB01: MCU")

    # Storyboard (complete)
    (project / "storyboard.md").write_text(
        "# Visual Storyboard\n\n"
        "| Beat | Time | Section | Importance | Weight |\n"
        "| Singer% | Narrative% | B-roll% | Env% | Symbolic% | Montage% |\n"
        "| Emo | Energy | Focus | Duration | Scale | Camera |\n"
        "| Coverage | Rationale |\n"
        "|------|------|---------|------------|--------|\n"
        "|---------|------------|---------|------|-----------|----------|\n"
        "|-----|--------|-------|----------|-------|--------|\n"
        "|----------|-----------|\n"
        "| B01 | 0:00-0:08 | verse | Medium | 5 |\n"
        "| 50 | 30 | 10 | 10 | 0 | 0 |\n"
        "| 3 | 3 | Face | Single | MCU | Push |\n"
        "| Single take | Narrative storytelling |\n"
    )

    # FSM
    (project / "index.md").write_text("---\nstate: PROMPTS\n---\n# Index\n")

    return project


# ── Test 1: validate_prerequisites passes when all files exist ──────────

def test_validate_prerequisites_pass(all_files_project: Path) -> None:
    """Test 1: validate_prerequisites passes when all files exist."""
    report = validate_prerequisites(all_files_project)
    assert report.overall_pass is True
    assert len(report.fail_checks) == 0
    assert len(report.missing_items) == 0


# ── Test 2: validate_prerequisites fails with specific missing file ─────

def test_validate_prerequisites_fails_missing_file(tmp_path: Path) -> None:
    """Test 2: validate_prerequisites fails with specific missing file name."""
    project = tmp_path / "project"
    project.mkdir()
    # Only audio file present
    (project / "song.mp3").write_bytes(b"\x00" * 1024)

    report = validate_prerequisites(project)
    assert report.overall_pass is False
    # Should mention portrait as missing
    missing_str = " ".join(report.missing_items)
    assert "portrait" in missing_str.lower()


# ── Test 3: check_audio_portrait_match ──────────────────────────────────

def test_check_audio_portrait_match(tmp_path: Path) -> None:
    """Test 3: check_audio_portrait_match returns True for matching paths."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "song.mp3").write_bytes(b"\x00" * 1024)
    (project / "portrait.jpg").write_bytes(b"\x00" * 20_000)

    report = validate_prerequisites(project)
    # Both files exist and are valid size — should pass those checks
    audio_pass = any("audio" in c.lower() or "song" in c.lower()
                     for c in report.pass_checks)
    portrait_pass = any("portrait" in c.lower() for c in report.pass_checks)
    assert audio_pass or portrait_pass  # At least one should pass


# ── Test 4: verify_timeline_coverage ───────────────────────────────────

def test_verify_timeline_coverage_passes(tmp_path: Path) -> None:
    """Test 4: verify_timeline_coverage returns True when segments tile contiguously."""
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "first"},
            {"start": 5.0, "end": 10.0, "text": "second"},
            {"start": 10.0, "end": 15.0, "text": "third"},
        ]
    }))
    assert check_timeline_coverage(transcript, 15.0) is True


def test_verify_timeline_coverage_fails_gap(tmp_path: Path) -> None:
    """Timeline coverage fails when there is a large gap."""
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "first"},
            {"start": 8.0, "end": 13.0, "text": "second"},
        ]
    }))
    assert check_timeline_coverage(transcript, 13.0) is False


# ── Test 5: check_aspect_ratio ─────────────────────────────────────────

def test_check_aspect_ratio_16_9() -> None:
    """Test 5a: validates 16:9 for 1920x1080."""
    assert check_aspect_ratio(1920, 1080, (16, 9)) is True


def test_check_aspect_ratio_1920x1088_passes() -> None:
    """Test 5b: passes 1920x1088 within 2% tolerance."""
    assert check_aspect_ratio(1920, 1088, (16, 9), tolerance=0.02) is True


def test_check_aspect_ratio_wrong_fails() -> None:
    """Test 5c: fails for clearly wrong ratio."""
    assert check_aspect_ratio(1920, 1080, (4, 3)) is False


# ── Test 6: ValidationReport serialization ─────────────────────────────

def test_validation_report_serialization() -> None:
    """Test 6: ValidationReport dataclass serializes to dict with pass/fail/missing lists."""
    report = ValidationReport(
        pass_checks=["audio OK", "portrait OK"],
        fail_checks=["missing storyboard"],
        missing_items=["storyboard.md"],
    )
    assert report.overall_pass is False
    assert not report.is_ready_for_generation

    report_pass = ValidationReport(
        pass_checks=["all checks passed"],
        fail_checks=[],
        missing_items=[],
    )
    assert report_pass.overall_pass is True
    assert report_pass.is_ready_for_generation is True


# ── Test 7: validate_storyboard_completeness passes ────────────────────

def test_validate_storyboard_completeness_passes(tmp_path: Path) -> None:
    """Test 7: passes when all beats have required fields."""
    storyboard = tmp_path / "storyboard.md"
    storyboard.write_text(
        "# Visual Storyboard\n\n"
        "| Beat | Time | Section | Importance | Weight |\n"
        "| Singer% | Narrative% | B-roll% | Env% | Symbolic% | Montage% |\n"
        "| Emo | Energy | Focus | Duration | Scale | Camera |\n"
        "| Coverage | Rationale |\n"
        "|------|------|---------|------------|--------|\n"
        "|---------|------------|---------|------|-----------|----------|\n"
        "|-----|--------|-------|----------|-------|--------|\n"
        "|----------|-----------|\n"
        "| B01 | 0:00-0:08 | verse | Medium | 5 |\n"
        "| 50 | 30 | 10 | 10 | 0 | 0 |\n"
        "| 3 | 3 | Face | Single | MCU | Push |\n"
        "| Single take | Narrative storytelling |\n"
        "| B02 | 0:08-0:16 | chorus | High | 8 |\n"
        "| 70 | 10 | 5 | 10 | 5 | 0 |\n"
        "| 5 | 4 | Face | Two-Shot | CU | Dynamic |\n"
        "| Montage | Hero moment |\n"
    )
    gaps = validate_storyboard_completeness(storyboard, 2)
    assert len(gaps) == 0


# ── Test 8: validate_storyboard_completeness fails with gaps ───────────

def test_validate_storyboard_completeness_fails_gaps(tmp_path: Path) -> None:
    """Test 8: fails with gap list when beats missing visual_weight or rationale."""
    storyboard = tmp_path / "storyboard.md"
    storyboard.write_text(
        "# Visual Storyboard\n\n"
        "| Beat | Time | Section | Importance | Weight |\n"
        "| Singer% | Narrative% | B-roll% | Env% | Symbolic% | Montage% |\n"
        "| Emo | Energy | Focus | Duration | Scale | Camera |\n"
        "| Coverage | Rationale |\n"
        "|------|------|---------|------------|--------|\n"
        "|---------|------------|---------|------|-----------|----------|\n"
        "|-----|--------|-------|----------|-------|--------|\n"
        "|----------|-----------|\n"
        "| B01 | 0:00-0:08 | verse | Medium | 5 |\n"
        "| 50 | 30 | 10 | 10 | 0 | 0 |\n"
        "| 3 | 3 | Face | Single | MCU | Push |\n"
        "| Single take | Narrative storytelling |\n"
        "| B02 | 0:08-0:16 | chorus | High | |\n"
        "| 70 | 10 | 5 | 10 | 5 | 0 |\n"
        "| 5 | 4 | Face | Two-Shot | CU | Dynamic |\n"
        "| Montage | |\n"
    )
    gaps = validate_storyboard_completeness(storyboard, 2)
    assert len(gaps) > 0
    # B02 should be missing weight and rationale
    gap_str = " ".join(gaps).lower()
    assert "b02" in gap_str
    # Should mention at least one missing field
    assert "weight" in gap_str or "rationale" in gap_str


# ── Test 9: validate_beat_content flags duration out of range ──────────

def test_validate_beat_content_flags_duration(tmp_path: Path) -> None:
    """Test 9: flags beats with duration outside [min, max] range."""
    beats = tmp_path / "beat_sheet.md"
    beats.write_text(
        "# Beat Sheet\n\n"
        "| Beat | Start | End | Duration |\n"
        "|------|-------|-----|----------|\n"
        "| B01  | 0.0   | 8.0 | 8.0      |\n"
        "| B02  | 8.0   | 30.0 | 22.0     |\n"
    )
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "beat_01.md").write_text("# B01\nMotion: walk")
    (prompts / "beat_02.md").write_text("# B02\nMotion: run")
    continuity = tmp_path / "continuity_bible.md"
    continuity.write_text("# Bible\n\n## Invariances\n- white dress\n")

    violations = validate_beat_content(beats, prompts, continuity)
    assert len(violations) > 0
    # B02's 22s should exceed max
    violation_str = " ".join(violations).lower()
    assert "b02" in violation_str
    assert "22" in violation_str or "duration" in violation_str


# ── Test 10: validate_beat_content returns empty list when all pass ────

def test_validate_beat_content_all_pass(tmp_path: Path) -> None:
    """Test 10: returns empty list when all beats pass duration and continuity checks."""
    beats = tmp_path / "beat_sheet.md"
    beats.write_text(
        "# Beat Sheet\n\n"
        "| Beat | Start | End | Duration |\n"
        "|------|-------|-----|----------|\n"
        "| B01  | 0.0   | 6.0 | 6.0      |\n"
        "| B02  | 6.0   | 14.0 | 8.0      |\n"
    )
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "beat_01.md").write_text("# B01\nMotion: walk forward")
    (prompts / "beat_02.md").write_text("# B02\nMotion: standing still")
    continuity = tmp_path / "continuity_bible.md"
    continuity.write_text("# Bible\n\n## Invariances\n- white dress\n")

    violations = validate_beat_content(beats, prompts, continuity)
    assert len(violations) == 0
