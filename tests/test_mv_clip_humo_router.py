"""Unit tests for the HuMo 14B hybrid router (Plan 09.9-25-01).

Tests the D-02 segment classifier and the D-01 16s HuMo default without any
live GPU / ComfyUI. ``generate_humo_clip`` (plan 02) is faked via a stub module
in ``sys.modules`` so ``_route_segment``'s lazy ``from mv_humo_gen import
generate_humo_clip`` resolves to a MagicMock.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from mv_clip import classify_segment_engine
from mv_clip_generate import (
    HUMO_DEFAULT_CLIP_DURATION_S,
    HUMO_GEN_HEIGHT,
    HUMO_GEN_WIDTH,
    _route_segment,
)
from mv_segment import ClipSegment


def _make_segment(start: float = 0.0, end: float = 5.0, text: str = "hello world") -> ClipSegment:
    """Build a minimal ClipSegment for routing tests."""
    return ClipSegment(start=start, end=end, text=text, duration=end - start, words=[])


def _install_fake_humo_gen() -> MagicMock:
    """Inject a fake ``mv_humo_gen`` module so the lazy import resolves.

    Returns a FRESH ``generate_humo_clip`` MagicMock (returning a sentinel Path)
    and registers it on the fake module, so each test gets an isolated mock and
    ``_route_segment``'s ``from mv_humo_gen import generate_humo_clip`` resolves.
    """
    fake = sys.modules.get("mv_humo_gen")
    if fake is None:
        fake = types.ModuleType("mv_humo_gen")
        sys.modules["mv_humo_gen"] = fake
    gen_mock = MagicMock(return_value=Path("/tmp/humo_out.mp4"))
    fake.generate_humo_clip = gen_mock
    return gen_mock


def test_classify_vocal_lyrics_is_humo() -> None:
    seg = _make_segment(text="singing the words")
    assert classify_segment_engine(seg, vocal_presence=True, has_lyrics=True) == "humo"


def test_classify_instrumental_is_ltx2() -> None:
    # No vocal presence -> ltx2 regardless of has_lyrics value.
    seg = _make_segment(text="")
    assert classify_segment_engine(seg, vocal_presence=False, has_lyrics=True) == "ltx2"
    assert classify_segment_engine(seg, vocal_presence=False, has_lyrics=False) == "ltx2"


def test_classify_broll_no_lyrics_is_ltx2() -> None:
    seg = _make_segment(text="")
    assert classify_segment_engine(seg, vocal_presence=False, has_lyrics=False) == "ltx2"


def test_route_humo_calls_generate_humo_clip_with_16s() -> None:
    gen_mock = _install_fake_humo_gen()
    cropped = Path("/tmp/cropped_humo_test.wav")
    cropped.write_text("fake")
    crop_mock = MagicMock(return_value=cropped)
    seg = _make_segment(start=0.0, end=4.0, text="la la la")
    with patch("mv_audio._crop_audio_segment", crop_mock), \
         patch("mv_humo_gen.generate_humo_clip", gen_mock):
        out = _route_segment(
            seg, "scene", Path("/tmp/ref.png"), 1, Path("/tmp/out"),
            vocal_presence=True, has_lyrics=True, vocals_stem=Path("/tmp/v.wav"),
        )
    assert out == Path("/tmp/humo_out.mp4")
    gen_mock.assert_called_once()
    _, kwargs = gen_mock.call_args
    assert kwargs["duration_s"] == HUMO_DEFAULT_CLIP_DURATION_S  # 16
    assert kwargs["duration_s"] == 16
    assert kwargs["width"] == HUMO_GEN_WIDTH   # 848
    assert kwargs["height"] == HUMO_GEN_HEIGHT  # 480
    crop_mock.assert_called_once()
    cropped.unlink(missing_ok=True)


def test_route_ltx2_delegates_to_generate_clip() -> None:
    ltx_mock = MagicMock(return_value=Path("/tmp/ltx_out.mp4"))
    gen_mock = _install_fake_humo_gen()
    seg = _make_segment(start=0.0, end=4.0, text="")
    with patch("mv_clip_generate._generate_clip", ltx_mock), \
         patch("mv_humo_gen.generate_humo_clip", gen_mock):
        out = _route_segment(
            seg, "scene", Path("/tmp/ref.png"), 2, Path("/tmp/out"),
            vocal_presence=False, has_lyrics=False,
        )
    assert out == Path("/tmp/ltx_out.mp4")
    ltx_mock.assert_called_once()
    gen_mock.assert_not_called()


def test_route_humo_forwards_explicit_duration_s() -> None:
    gen_mock = _install_fake_humo_gen()
    cropped = Path("/tmp/cropped_humo_test2.wav")
    cropped.write_text("fake")
    crop_mock = MagicMock(return_value=cropped)
    seg = _make_segment(start=0.0, end=4.0, text="la la")
    with patch("mv_audio._crop_audio_segment", crop_mock), \
         patch("mv_humo_gen.generate_humo_clip", gen_mock):
        _route_segment(
            seg, "scene", Path("/tmp/ref.png"), 3, Path("/tmp/out"),
            vocal_presence=True, has_lyrics=True,
            vocals_stem=Path("/tmp/v.wav"), duration_s=8,
        )
    gen_mock.assert_called_once()
    _, kwargs = gen_mock.call_args
    assert kwargs["duration_s"] == 8  # per-clip override beats 16s default
    cropped.unlink(missing_ok=True)


def test_route_force_engine_ltx2_overrides_classifier() -> None:
    ltx_mock = MagicMock(return_value=Path("/tmp/ltx_out.mp4"))
    gen_mock = _install_fake_humo_gen()
    seg = _make_segment(start=0.0, end=4.0, text="lyrics here")
    with patch("mv_clip_generate._generate_clip", ltx_mock), \
         patch("mv_humo_gen.generate_humo_clip", gen_mock):
        out = _route_segment(
            seg, "scene", Path("/tmp/ref.png"), 4, Path("/tmp/out"),
            vocal_presence=True, has_lyrics=True, force_engine="ltx2",
        )
    assert out == Path("/tmp/ltx_out.mp4")
    ltx_mock.assert_called_once()
    gen_mock.assert_not_called()
