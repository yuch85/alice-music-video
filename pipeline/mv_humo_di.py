#!/usr/bin/env python3
"""Dependency Injection interfaces for HuMo progressive validation (Plan 09.9-27-01).

Provides protocol-based DI for audio provisioning and Slingshot lifecycle
management. Real implementations wire into ffmpeg and gpu-manager; mock
implementations enable structural testing without GPU/ComfyUI dependencies.

STYLE.md: manual DI, no container. Factory functions create configured instances.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# ── Named constants (STYLE.md: no magic numbers/strings) ────────────────────

VOCALS_SAMPLE_RATE = 44100
VOCALS_AUDIO_CODEC = "pcm_s16le"
DEFAULT_VOCALS_SOURCE = Path(
    os.environ.get("MV_AUDIO_PATH", "/path/to/vocals.wav")
)
GPU_MANAGER_BASE_URL = "http://127.0.0.1:8090"


# ── Protocol 1: AudioProvisioner ────────────────────────────────────────────

class AudioProvisioner(Protocol):
    """Crop audio segments from a vocals stem for HuMo Audio VAE conditioning."""

    def prepare_segment(
        self,
        *,
        source_path: Path,
        start_s: float,
        duration_s: float,
        output_dir: Path,
    ) -> Path: ...


class RealAudioProvisioner:
    """Crop audio segments using ffmpeg.

    Uses list-based subprocess calls (no shell=True) per threat model
    T-09.9-27-01 mitigation.
    """

    def prepare_segment(
        self,
        *,
        source_path: Path,
        start_s: float,
        duration_s: float,
        output_dir: Path,
    ) -> Path:
        """Crop [start_s, start_s + duration_s] from the source WAV.

        Args:
            source_path: Path to the source vocals WAV file.
            start_s: Start time in seconds.
            duration_s: Duration of the segment in seconds.
            output_dir: Directory for the output WAV file.

        Returns:
            Path to the cropped WAV file.

        Raises:
            FileNotFoundError: If source_path does not exist.
            RuntimeError: If ffmpeg fails.
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Audio source not found: {source_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        end_s = start_s + duration_s
        output_file = output_dir / f"vocals_{start_s:.0f}s_{duration_s:.0f}s.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(source_path),
            "-ss", str(start_s),
            "-to", str(end_s),
            "-acodec", VOCALS_AUDIO_CODEC,
            "-ar", str(VOCALS_SAMPLE_RATE),
            str(output_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio crop failed (rc={result.returncode}): "
                f"{result.stderr[:500]}"
            )

        logger.info("Audio segment cropped: %s -> %s", source_path.name, output_file)
        return output_file


class MockAudioProvisioner:
    """Create silent WAV stubs using ffmpeg anullsrc.

    Used for structural testing without real audio files.
    """

    def prepare_segment(
        self,
        *,
        source_path: Path,
        start_s: float,
        duration_s: float,
        output_dir: Path,
    ) -> Path:
        """Generate a silent WAV of the requested duration.

        Args:
            source_path: Ignored (mock does not read the source).
            start_s: Used for output filename only.
            duration_s: Duration of the silent WAV in seconds.
            output_dir: Directory for the output WAV file.

        Returns:
            Path to the generated silent WAV file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"vocals_{start_s:.0f}s_{duration_s:.0f}s.wav"

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={VOCALS_SAMPLE_RATE}:cl=mono",
            "-t", str(duration_s),
            "-acodec", VOCALS_AUDIO_CODEC,
            str(output_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg silent WAV generation failed (rc={result.returncode}): "
                f"{result.stderr[:500]}"
            )

        logger.info("Mock audio stub created: %s (%.1fs)", output_file, duration_s)
        return output_file


# ── Protocol 2: SlingshotGate ───────────────────────────────────────────────

class SlingshotGate(Protocol):
    """Context manager for Slingshot hibernate/wake lifecycle."""

    def __enter__(self) -> "SlingshotGate": ...
    def __exit__(self, *exc_info: object) -> bool | None: ...


class RealSlingshotGate:
    """Wrap SlingshotClient hibernate/wake as a context manager.

    On enter: calls ensure_hibernate() to save LLM state and free VRAM.
    On exit: calls ensure_wake() to restore the LLM.

    Imports mv_slingshot lazily to keep this file importable without GPU deps.
    """

    def __init__(self, *, task_name: str = "humo_test", output_path: str = "") -> None:
        """Initialize the gate.

        Args:
            task_name: Task identifier passed to slingshot wake.
            output_path: Output file path passed to slingshot wake.
        """
        self._task_name = task_name
        self._output_path = output_path
        self._client: "object | None" = None

    def __enter__(self) -> RealSlingshotGate:
        from mv_slingshot import SlingshotClient  # deferred import

        self._client = SlingshotClient(base_url=GPU_MANAGER_BASE_URL)
        self._client.ensure_hibernate()
        return self

    def __exit__(self, *exc_info: object) -> bool | None:
        if self._client is not None:
            return self._client.ensure_wake(
                task_name=self._task_name,
                output_path=self._output_path,
            )
        return None


class MockSlingshotGate:
    """No-op SlingshotGate that records enter/exit calls.

    Used for structural testing without gpu-manager.
    """

    def __init__(self, *, task_name: str = "humo_test", output_path: str = "") -> None:
        """Initialize the mock gate.

        Args:
            task_name: Recorded for inspection.
            output_path: Recorded for inspection.
        """
        self.task_name = task_name
        self.output_path = output_path
        self.calls: list[str] = []

    def __enter__(self) -> MockSlingshotGate:
        self.calls.append("enter")
        return self

    def __exit__(self, *exc_info: object) -> bool | None:
        if exc_info[0] is not None:
            self.calls.append("exit_with_error")
        else:
            self.calls.append("exit")
        return None


# ── Factory functions ───────────────────────────────────────────────────────

def create_audio_provisioner(*, mock: bool = False) -> AudioProvisioner:
    """Create an AudioProvisioner implementation.

    Args:
        mock: If True, return MockAudioProvisioner.

    Returns:
        Configured AudioProvisioner instance.
    """
    if mock:
        return MockAudioProvisioner()
    return RealAudioProvisioner()


def create_slingshot_gate(
    *,
    mock: bool = False,
    task_name: str = "humo_test",
    output_path: str = "",
) -> SlingshotGate:
    """Create a SlingshotGate implementation.

    Args:
        mock: If True, return MockSlingshotGate.
        task_name: Task identifier for wake calls.
        output_path: Output file path for wake calls.

    Returns:
        Configured SlingshotGate instance.
    """
    if mock:
        return MockSlingshotGate(task_name=task_name, output_path=output_path)
    return RealSlingshotGate(task_name=task_name, output_path=output_path)
