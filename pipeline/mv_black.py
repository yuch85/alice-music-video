#!/usr/bin/env python3
"""Black-frame clip generation for the 'black' shot type (Plan 09.9-10).

Extracted from the original `generate_music_video_pipeline.py` block during the
STYLE.md-compliant split. Logic is byte-for-byte identical to the source.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _generate_black_frame(
    duration_s: float,
    clip_index: int,
    output_dir: Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Generate a black frame video via FFmpeg for 'black' shot type.

    Uses the generation resolution (width x height) so the black frame matches
    the real clips — the concat demuxer requires uniform resolution.
    """
    dest = output_dir / "clips" / f"clip_{clip_index:03d}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:d={duration_s:.2f}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "24",
        str(dest),
    ]
    logger.info("Generating black frame: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Black frame generation failed: %s", result.stderr)
        return dest  # Return path anyway, composite handles missing clips
    logger.info("Black frame saved: %s", dest)
    return dest
