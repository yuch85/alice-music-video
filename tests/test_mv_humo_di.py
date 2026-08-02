#!/usr/bin/env python3
"""Tests for mv_humo_di — DI interfaces for HuMo progressive validation.

Covers AudioProvisioner (real + mock) and SlingshotGate (real + mock)
protocols and their factory functions.

Tests use the mock implementations to verify structural correctness
without requiring GPU/ComfyUI/ffmpeg (except the real audio test
which validates ffmpeg is available).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.mv_humo_di import (
    AudioProvisioner,
    MockAudioProvisioner,
    MockSlingshotGate,
    RealAudioProvisioner,
    RealSlingshotGate,
    SlingshotGate,
    create_audio_provisioner,
    create_slingshot_gate,
)


# ── AudioProvisioner tests ──────────────────────────────────────────────────

class TestRealAudioProvisioner:
    """Test RealAudioProvisioner crops a WAV file correctly."""

    def test_prepare_segment_crops_wav(self, tmp_path: Path) -> None:
        # Create a source WAV (2 seconds of silence)
        source = tmp_path / "source.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "anullsrc=r=44100:cl=mono",
                "-t", "2", "-acodec", "pcm_s16le",
                str(source),
            ],
            check=True, capture_output=True,
        )
        provisioner = RealAudioProvisioner()
        result = provisioner.prepare_segment(
            source_path=source,
            start_s=0.5,
            duration_s=1.0,
            output_dir=tmp_path,
        )
        assert result.exists(), "Output WAV should exist"
        assert result.suffix == ".wav"
        # Verify duration is ~1.0s
        dur = _get_duration(result)
        assert 0.9 < dur < 1.2, f"Expected ~1.0s duration, got {dur}"

    def test_prepare_segment_raises_on_missing_source(self, tmp_path: Path) -> None:
        provisioner = RealAudioProvisioner()
        with pytest.raises(FileNotFoundError):
            provisioner.prepare_segment(
                source_path=tmp_path / "nonexistent.wav",
                start_s=0.0,
                duration_s=1.0,
                output_dir=tmp_path,
            )


class TestMockAudioProvisioner:
    """Test MockAudioProvisioner creates a stub WAV without real audio."""

    def test_prepare_segment_creates_stub(self, tmp_path: Path) -> None:
        # No real source needed — mock creates silent audio
        provisioner = MockAudioProvisioner()
        result = provisioner.prepare_segment(
            source_path=tmp_path / "fake_source.wav",
            start_s=5.0,
            duration_s=8.0,
            output_dir=tmp_path,
        )
        assert result.exists(), "Mock output WAV should exist"
        assert result.suffix == ".wav"
        # Verify duration matches requested
        dur = _get_duration(result)
        assert 7.5 < dur < 8.5, f"Expected ~8.0s duration, got {dur}"


# ── SlingshotGate tests ─────────────────────────────────────────────────────

class TestMockSlingshotGate:
    """Test MockSlingshotGate records enter/exit calls."""

    def test_context_manager_records_calls(self) -> None:
        gate = MockSlingshotGate(task_name="test", output_path="/tmp/out.mp4")
        assert gate.calls == []

        with gate:
            assert len(gate.calls) == 1
            assert gate.calls[0] == "enter"

        assert len(gate.calls) == 2
        assert gate.calls[1] == "exit"

    def test_context_manager_records_exception(self) -> None:
        gate = MockSlingshotGate(task_name="test", output_path="/tmp/out.mp4")
        with pytest.raises(ValueError):
            with gate:
                raise ValueError("test error")

        assert "exit_with_error" in gate.calls


class TestRealSlingshotGate:
    """Test RealSlingshotGate delegates to SlingshotClient."""

    def test_is_slingshot_gate_protocol(self) -> None:
        # Verify RealSlingshotGate satisfies the SlingshotGate protocol
        gate: SlingshotGate = RealSlingshotGate(
            task_name="test", output_path="/tmp/out.mp4"
        )
        assert hasattr(gate, "__enter__")
        assert hasattr(gate, "__exit__")


# ── Factory function tests ──────────────────────────────────────────────────

class TestFactories:
    """Test factory functions return correct implementations."""

    def test_create_audio_provisioner_real(self) -> None:
        prov = create_audio_provisioner(mock=False)
        assert isinstance(prov, RealAudioProvisioner)

    def test_create_audio_provisioner_mock(self) -> None:
        prov = create_audio_provisioner(mock=True)
        assert isinstance(prov, MockAudioProvisioner)

    def test_create_slingshot_gate_real(self) -> None:
        gate = create_slingshot_gate(mock=False)
        assert isinstance(gate, RealSlingshotGate)

    def test_create_slingshot_gate_mock(self) -> None:
        gate = create_slingshot_gate(mock=True)
        assert isinstance(gate, MockSlingshotGate)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_duration(path: Path) -> float:
    """Return audio/video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return 0.0
    return float(result.stdout.strip())
