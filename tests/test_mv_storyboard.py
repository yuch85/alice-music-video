#!/usr/bin/env python3
"""Tests for mv_storyboard — visual storyboard classification helper module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mv_beats import BeatEntry
from mv_storyboard import (
    StoryboardEntry,
    assign_coverage_strategy,
    assign_performance_focus,
    assign_shot_scale_and_movement,
    assign_visual_weight,
    build_storyboard,
    candidate_count_from_weight,
    classify_song_structure,
    extract_and_repair_json,
)


# ── Test 1: StoryboardEntry dataclass ──


class TestStoryboardEntry:
    """Test StoryboardEntry has all 15 fields and serializes to dict."""

    def test_all_fields(self) -> None:
        entry = StoryboardEntry(
            beat_index=0,
            start=0.0,
            end=5.0,
            lyrics="intro",
            section="intro",
            narrative_importance="Low",
            visual_weight=3,
            performance_focus={
                "singer_pct": 20,
                "narrative_pct": 10,
                "broll_pct": 40,
                "environment_pct": 30,
                "symbolic_pct": 0,
                "montage_pct": 0,
            },
            emotional_intensity=2,
            visual_energy=2,
            primary_viewer_focus="Landscape",
            recommended_shot_duration_strategy="Single",
            recommended_shot_scale="Wide",
            camera_movement_intensity="Locked",
            coverage_strategy="Single take",
            rationale="Establishing shot sets mood",
        )
        assert entry.beat_index == 0
        assert entry.visual_weight == 3
        assert entry.narrative_importance == "Low"

    def test_to_dict(self) -> None:
        entry = StoryboardEntry(
            beat_index=0,
            start=0.0,
            end=5.0,
            lyrics="intro",
            section="intro",
            narrative_importance="Low",
            visual_weight=3,
            performance_focus={
                "singer_pct": 20,
                "narrative_pct": 10,
                "broll_pct": 40,
                "environment_pct": 30,
                "symbolic_pct": 0,
                "montage_pct": 0,
            },
            emotional_intensity=2,
            visual_energy=2,
            primary_viewer_focus="Landscape",
            recommended_shot_duration_strategy="Single",
            recommended_shot_scale="Wide",
            camera_movement_intensity="Locked",
            coverage_strategy="Single take",
            rationale="Establishing shot",
        )
        d = entry.to_dict()
        assert d["visual_weight"] == 3
        assert d["performance_focus"]["singer_pct"] == 20
        assert len(d) == 16  # 16 fields: beat_index through rationale

    def test_from_dict(self) -> None:
        d = {
            "beat_index": 1,
            "start": 5.0,
            "end": 10.0,
            "lyrics": "verse",
            "section": "verse",
            "narrative_importance": "Medium",
            "visual_weight": 5,
            "performance_focus": {
                "singer_pct": 50,
                "narrative_pct": 30,
                "broll_pct": 10,
                "environment_pct": 10,
                "symbolic_pct": 0,
                "montage_pct": 0,
            },
            "emotional_intensity": 3,
            "visual_energy": 3,
            "primary_viewer_focus": "Face",
            "recommended_shot_duration_strategy": "Single",
            "recommended_shot_scale": "MCU",
            "camera_movement_intensity": "Push",
            "coverage_strategy": "Single take",
            "rationale": "Verse storytelling",
        }
        entry = StoryboardEntry.from_dict(d)
        assert entry.beat_index == 1
        assert entry.visual_weight == 5
        assert entry.primary_viewer_focus == "Face"


# ── Test 2: classify_song_structure ──


class TestClassifySongStructure:
    """Test classify_song_structure labels intro/verse/pre-chorus/chorus/etc."""

    def test_labels_sections(self) -> None:
        beats = [
            BeatEntry(
                start=0.0, end=5.0, duration=5.0, lyrics="",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="low", dominant_character="--", transition_from="--",
                transition_to="--", detailed_notes="", section="",
            ),
            BeatEntry(
                start=5.0, end=15.0, duration=10.0, lyrics="verse lyrics here",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="medium", dominant_character="singer", transition_from="--",
                transition_to="--", detailed_notes="", section="",
            ),
            BeatEntry(
                start=15.0, end=25.0, duration=10.0, lyrics="chorus lyrics",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="high", dominant_character="singer", transition_from="--",
                transition_to="--", detailed_notes="", section="",
            ),
        ]
        result = classify_song_structure(beats)
        # First beat (no lyrics) should be intro or instrumental
        assert result[0].section in ("intro", "instrumental")
        # Second beat (with lyrics, medium energy) should be verse
        assert result[1].section == "verse"
        # Third beat (high energy) should be chorus
        assert result[2].section == "chorus"


# ── Test 3: assign_visual_weight ──


class TestAssignVisualWeight:
    """Test assign_visual_weight maps chorus to higher weight, intro to lower."""

    def test_chorus_higher_weight(self) -> None:
        beats = [
            BeatEntry(
                start=0.0, end=5.0, duration=5.0, lyrics="",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="low", dominant_character="--", transition_from="--",
                transition_to="--", detailed_notes="", section="intro",
            ),
            BeatEntry(
                start=5.0, end=15.0, duration=10.0, lyrics="verse lyrics",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="medium", dominant_character="singer", transition_from="--",
                transition_to="--", detailed_notes="", section="verse",
            ),
            BeatEntry(
                start=15.0, end=25.0, duration=10.0, lyrics="chorus lyrics",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="high", dominant_character="singer", transition_from="--",
                transition_to="--", detailed_notes="", section="chorus",
            ),
        ]
        result = assign_visual_weight(beats)
        # Intro should be low weight (2-4)
        assert 2 <= result[0].visual_weight <= 4
        # Verse should be medium weight (3-6)
        assert 3 <= result[1].visual_weight <= 6
        # Chorus should be high weight (7-10)
        assert 7 <= result[2].visual_weight <= 10


# ── Test 4: build_storyboard ──


class TestBuildStoryboard:
    """Test build_storyboard produces list of StoryboardEntry with all fields."""

    def test_produces_entries(self, tmp_path: Path) -> None:
        beats = [
            BeatEntry(
                start=0.0, end=5.0, duration=5.0, lyrics="",
                narrative_purpose="atmosphere", emotional_purpose="calm",
                visual_purpose="wide", energy="low", dominant_character="--",
                transition_from="fade_in", transition_to="cut",
                detailed_notes="", section="intro",
            ),
            BeatEntry(
                start=5.0, end=15.0, duration=10.0, lyrics="verse lyrics",
                narrative_purpose="story", emotional_purpose="tension",
                visual_purpose="medium", energy="medium",
                dominant_character="singer", transition_from="cut",
                transition_to="dissolve", detailed_notes="", section="verse",
            ),
        ]
        treatment = tmp_path / "treatment.md"
        treatment.write_text("# Treatment\nTest treatment.")
        output = tmp_path / "storyboard.md"

        result = build_storyboard(beats, str(treatment), str(output))
        assert len(result) == 2
        assert all(isinstance(e, StoryboardEntry) for e in result)
        assert output.is_file()


# ── Test 5: candidate_count_from_weight ──


class TestCandidateCountFromWeight:
    """Test candidate_count_from_weight returns correct ranges."""

    def test_weight_ranges(self) -> None:
        assert candidate_count_from_weight(1) == (2, 3)
        assert candidate_count_from_weight(2) == (2, 3)
        assert candidate_count_from_weight(3) == (2, 3)
        assert candidate_count_from_weight(4) == (3, 4)
        assert candidate_count_from_weight(5) == (3, 4)
        assert candidate_count_from_weight(6) == (3, 4)
        assert candidate_count_from_weight(7) == (5, 8)
        assert candidate_count_from_weight(8) == (5, 8)
        assert candidate_count_from_weight(9) == (5, 8)
        assert candidate_count_from_weight(10) == (5, 8)

    def test_edge_cases(self) -> None:
        # Out of range should clamp
        assert candidate_count_from_weight(0) == (2, 3)
        assert candidate_count_from_weight(11) == (5, 8)


# ── Test 6: extract_and_repair_json ──


class TestExtractAndRepairJson:
    """Test JSON extraction and repair from LLM output."""

    def test_plain_json(self) -> None:
        result = extract_and_repair_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_array(self) -> None:
        result = extract_and_repair_json('[{"a": 1}, {"a": 2}]')
        assert len(result) == 2

    def test_markdown_code_fence(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = extract_and_repair_json(text)
        assert result == {"key": "value"}

    def test_trailing_comma_removal(self) -> None:
        text = '{"key": "value",}'
        result = extract_and_repair_json(text)
        assert result == {"key": "value"}

    def test_trailing_comma_in_array(self) -> None:
        text = '[{"a": 1}, {"a": 2},]'
        result = extract_and_repair_json(text)
        assert len(result) == 2

    def test_bracket_balancing(self) -> None:
        text = '{"key": "value"'  # Missing closing brace
        result = extract_and_repair_json(text)
        assert result == {"key": "value"}

    def test_nested_bracket_balancing(self) -> None:
        text = '[{"a": {"b": 1}}'  # Missing ]
        result = extract_and_repair_json(text)
        assert result == [{"a": {"b": 1}}]

    def test_invalid_json_returns_none(self) -> None:
        result = extract_and_repair_json("not json at all {{{")
        assert result is None
