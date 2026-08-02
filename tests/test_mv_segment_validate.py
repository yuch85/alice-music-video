"""Tests for mv_segment._validate_segment_plan regression validator."""

from mv_segment import _validate_segment_plan, ClipSegment, WordSegment
import pytest


# -- Fixtures --

def _valid_segments():
    """Return a valid non-overlapping contiguous segment list."""
    return [
        ClipSegment(start=0.0, end=10.0, text="hello world", duration=10.0),
        ClipSegment(start=10.0, end=20.0, text="foo bar", duration=10.0),
    ]


def _valid_segments_with_words():
    """Return valid segments with word lists attached."""
    words = [
        WordSegment("hello", 0.0, 0.5),
        WordSegment("world", 0.5, 1.0),
        WordSegment("foo", 10.0, 10.5),
        WordSegment("bar", 10.5, 11.0),
    ]
    segs = _valid_segments()
    segs[0].words = words[:2]
    segs[1].words = words[2:]
    return segs, words


# -- Test: overlap detection --

def test_detects_overlapping_segments():
    segs = [
        ClipSegment(start=0.0, end=10.0, text="a", duration=10.0),
        ClipSegment(start=9.0, end=20.0, text="b", duration=11.0),
    ]
    with pytest.raises(ValueError, match="(?i)overlap"):
        _validate_segment_plan(segs)


# -- Test: valid contiguous plan passes --

def test_valid_contiguous_plan_passes():
    segs = _valid_segments()
    _validate_segment_plan(segs)  # should not raise


# -- Test: negative gap (overlap) detection --

def test_detects_negative_gap():
    # segments where end > next start (i.e. overlap)
    segs = [
        ClipSegment(start=0.0, end=10.0, text="a", duration=10.0),
        ClipSegment(start=9.5, end=19.5, text="b", duration=10.0),
    ]
    with pytest.raises(ValueError, match="(?i)overlap"):
        _validate_segment_plan(segs)


# -- Test: duplicated words detection --

def test_detects_duplicated_words():
    words = [
        WordSegment("hello", 0.0, 0.5),
        WordSegment("world", 5.0, 5.5),
    ]
    seg1 = ClipSegment(start=0.0, end=10.0, text="a", duration=10.0)
    seg1.words = words
    seg2 = ClipSegment(start=10.0, end=20.0, text="b", duration=10.0)
    seg2.words = words  # same word objects duplicated
    with pytest.raises(ValueError, match="duplicat"):
        _validate_segment_plan([seg1, seg2], words=words)


# -- Test: skipped words detection --

def test_detects_skipped_words():
    words = [
        WordSegment("hello", 0.0, 0.5),
        WordSegment("world", 15.0, 15.5),
    ]
    seg1 = ClipSegment(start=0.0, end=10.0, text="a", duration=10.0)
    seg1.words = [words[0]]
    seg2 = ClipSegment(start=10.0, end=20.0, text="b", duration=10.0)
    seg2.words = []  # word at 15-15.5 is skipped
    with pytest.raises(ValueError, match="skip"):
        _validate_segment_plan([seg1, seg2], words=words)


# -- Test: valid durations --

def test_invalid_duration_raises():
    segs = [
        ClipSegment(start=0.0, end=10.0, text="a", duration=10.0),
        ClipSegment(start=10.0, end=20.0, text="b", duration=0.0),  # zero duration
    ]
    with pytest.raises(ValueError, match="duration"):
        _validate_segment_plan(segs)


def test_duration_mismatch_raises():
    segs = [
        ClipSegment(start=0.0, end=10.0, text="a", duration=10.0),
        ClipSegment(start=10.0, end=20.0, text="b", duration=5.0),  # wrong duration
    ]
    with pytest.raises(ValueError, match="duration"):
        _validate_segment_plan(segs)


# -- Test: timeline consistency --

def test_timeline_consistency_with_gaps():
    # Gaps are caught by timeline consistency check (validation runs after
    # _fill_coverage_gaps, so gaps should not exist in the augmented plan)
    segs = [
        ClipSegment(start=0.0, end=10.0, text="a", duration=10.0),
        ClipSegment(start=11.0, end=21.0, text="b", duration=10.0),
    ]
    with pytest.raises(ValueError, match="inconsisten"):
        _validate_segment_plan(segs)


def test_timeline_consistency_contiguous():
    segs = _valid_segments()
    _validate_segment_plan(segs)


# -- Test: monotonic timestamps --

def test_non_monotonic_segment_raises():
    segs = [
        ClipSegment(start=10.0, end=0.0, text="a", duration=-10.0),  # start > end
    ]
    with pytest.raises(ValueError, match="monotonic|duration"):
        _validate_segment_plan(segs)


# -- Test: empty segments --

def test_empty_segments_pass():
    _validate_segment_plan([])  # no segments = no violations


# -- Test: single segment --

def test_single_valid_segment_passes():
    segs = [ClipSegment(start=0.0, end=10.0, text="hello", duration=10.0)]
    _validate_segment_plan(segs)
