#!/usr/bin/env python3
"""Tests for mv_beats — beat sheet generation helper module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mv_beats import (
    BeatEntry,
    assign_energy_level,
    build_beat_sheet,
    fill_timeline_gaps,
    group_beats_by_section,
    parse_markdown_table,
    parse_whisper_timestamps,
    split_long_clips,
)


# ── Test 1: BeatEntry dataclass serialization ──


class TestBeatEntry:
    """Test BeatEntry dataclass has 12 fields and serializes correctly."""

    def test_basics(self) -> None:
        b = BeatEntry(
            start=0.0,
            end=5.0,
            duration=5.0,
            lyrics="hello",
            narrative_purpose="intro",
            emotional_purpose="calm",
            visual_purpose="wide",
            energy="low",
            dominant_character="singer",
            transition_from="fade_in",
            transition_to="cut",
            detailed_notes="",
            section="intro",
        )
        assert b.start == 0.0
        assert b.end == 5.0
        assert b.duration == 5.0
        assert b.section == "intro"

    def test_to_dict(self) -> None:
        b = BeatEntry(
            start=0.0,
            end=5.0,
            duration=5.0,
            lyrics="hello",
            narrative_purpose="intro",
            emotional_purpose="calm",
            visual_purpose="wide",
            energy="low",
            dominant_character="singer",
            transition_from="fade_in",
            transition_to="cut",
            detailed_notes="",
            section="intro",
        )
        d = b.to_dict()
        assert d["start"] == 0.0
        assert d["lyrics"] == "hello"
        assert d["section"] == "intro"
        assert len(d) == 13  # 12 PRD fields + section

    def test_from_dict(self) -> None:
        d = {
            "start": 2.0,
            "end": 7.0,
            "duration": 5.0,
            "lyrics": "verse line",
            "narrative_purpose": "setup",
            "emotional_purpose": "tension",
            "visual_purpose": "medium",
            "energy": "medium",
            "dominant_character": "singer",
            "transition_from": "cut",
            "transition_to": "dissolve",
            "detailed_notes": "test note",
            "section": "verse",
        }
        b = BeatEntry.from_dict(d)
        assert b.start == 2.0
        assert b.lyrics == "verse line"
        assert b.section == "verse"

    def test_duration_computed(self) -> None:
        b = BeatEntry(
            start=0.0,
            end=8.0,
            duration=8.0,
            lyrics="test",
            narrative_purpose="test",
            emotional_purpose="test",
            visual_purpose="test",
            energy="low",
            dominant_character="--",
            transition_from="--",
            transition_to="--",
            detailed_notes="",
            section="intro",
        )
        assert b.duration == b.end - b.start


# ── Test 2: parse_whisper_timestamps ──


class TestParseWhisperTimestamps:
    """Test parse_whisper_timestamps returns list of BeatEntry from transcript.json."""

    def test_parses_segments(self, tmp_path: Path) -> None:
        transcript = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "intro music"},
                {"start": 5.0, "end": 12.0, "text": "first verse lyrics here"},
                {"start": 12.0, "end": 18.0, "text": "chorus singing loudly"},
            ],
            "vocals_stem": str(tmp_path / "vocals.wav"),
        }
        tp = tmp_path / "transcript.json"
        tp.write_text(json.dumps(transcript))

        beats = parse_whisper_timestamps(str(tp))
        assert len(beats) == 3
        assert isinstance(beats[0], BeatEntry)
        assert beats[0].start == 0.0
        assert beats[0].end == 5.0
        assert beats[0].lyrics == "intro music"


# ── Test 3: group_beats_by_section ──


class TestGroupBeatsBySection:
    """Test group_beats_by_section identifies intro/verse/chorus/bridge/outro."""

    def test_basic_grouping(self) -> None:
        beats = [
            BeatEntry(
                start=0.0, end=5.0, duration=5.0, lyrics="",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="low", dominant_character="--", transition_from="--",
                transition_to="--", detailed_notes="", section="intro",
            ),
            BeatEntry(
                start=5.0, end=15.0, duration=10.0, lyrics="verse lyrics here",
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
        groups = group_beats_by_section(beats)
        assert "intro" in groups
        assert "verse" in groups
        assert "chorus" in groups


# ── Test 4: assign_energy_level ──


class TestAssignEnergyLevel:
    """Test assign_energy_level maps audio segment to energy label."""

    def test_energy_labels(self) -> None:
        beats = [
            BeatEntry(
                start=0.0, end=5.0, duration=5.0, lyrics="",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="", dominant_character="--", transition_from="--",
                transition_to="--", detailed_notes="", section="intro",
            ),
            BeatEntry(
                start=5.0, end=15.0, duration=10.0, lyrics="verse lyrics",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="", dominant_character="singer", transition_from="--",
                transition_to="--", detailed_notes="", section="verse",
            ),
            BeatEntry(
                start=15.0, end=25.0, duration=10.0, lyrics="chorus lyrics",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="", dominant_character="singer", transition_from="--",
                transition_to="--", detailed_notes="", section="chorus",
            ),
        ]
        result = assign_energy_level(beats, 25.0)
        assert all(b.energy in ("low", "medium", "high") for b in result)


# ── Test 5: fill_timeline_gaps ──


class TestFillTimelineGaps:
    """Test fill_timeline_gaps inserts instrumental slots for gaps."""

    def test_fills_gaps(self) -> None:
        beats = [
            BeatEntry(
                start=0.0, end=5.0, duration=5.0, lyrics="intro",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="low", dominant_character="--", transition_from="--",
                transition_to="--", detailed_notes="", section="intro",
            ),
            BeatEntry(
                start=10.0, end=15.0, duration=5.0, lyrics="verse",
                narrative_purpose="", emotional_purpose="", visual_purpose="",
                energy="medium", dominant_character="singer", transition_from="--",
                transition_to="--", detailed_notes="", section="verse",
            ),
        ]
        result = fill_timeline_gaps(beats, 20.0)
        # Should have inserted gap between 5.0-10.0 and trailing gap 15.0-20.0
        total_end = result[-1].end
        assert total_end == 20.0
        # Check gap was filled
        gap_beats = [b for b in result if b.lyrics == "(instrumental)"]
        assert len(gap_beats) >= 1


# ── Test 6: split_long_clips ──


class TestSplitLongClips:
    """Test split_long_clips splits beats exceeding max_duration."""

    def test_splits_long_beat(self) -> None:
        beats = [
            BeatEntry(
                start=0.0, end=25.0, duration=25.0, lyrics="long verse",
                narrative_purpose="story", emotional_purpose="tension",
                visual_purpose="medium", energy="medium",
                dominant_character="singer", transition_from="fade_in",
                transition_to="cut", detailed_notes="", section="verse",
            ),
        ]
        result = split_long_clips(beats, max_duration=10.0)
        assert len(result) >= 2
        for b in result:
            assert b.duration <= 10.0 + 0.01


# ── Test 7: parse_markdown_table ──


class TestParseMarkdownTable:
    """Test parse_markdown_table with auto-repair features."""

    def test_basic_table(self) -> None:
        md = """| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |"""
        rows = parse_markdown_table(md)
        assert len(rows) == 2
        assert rows[0]["A"] == "1"
        assert rows[1]["B"] == "4"

    def test_mismatched_columns(self) -> None:
        md = """| A | B | C |
|---|---|---|
| 1 | 2 |
| 3 | 4 | 5 | 6 |"""
        rows = parse_markdown_table(md)
        assert len(rows) == 2
        # Row 1: missing C, should be padded
        assert rows[0]["A"] == "1"
        assert rows[0]["B"] == "2"
        assert rows[0]["C"] == ""
        # Row 2: extra column, should be truncated
        assert rows[1]["C"] == "5"

    def test_divider_rows_skipped(self) -> None:
        md = """| A | B |
|---|---|
|---|---|
| 1 | 2 |
|:--|:--:|
| 3 | 4 |"""
        rows = parse_markdown_table(md)
        assert len(rows) == 2

    def test_empty_lines_sanitized(self) -> None:
        md = """

| A | B |

|---|---|

| 1 | 2 |

| 3 | 4 |

"""
        rows = parse_markdown_table(md)
        assert len(rows) == 2

    def test_missing_header_raises(self) -> None:
        md = """| 1 | 2 |
| 3 | 4 |"""
        with pytest.raises(ValueError, match="header"):
            parse_markdown_table(md)
