#!/usr/bin/env python3
"""Post-processing filter-chain construction for the music-video pipeline.

Pure helpers that build ffmpeg video/audio filter strings (no subprocess, no
I/O) so they are unit-testable in isolation. Split from mv_post.py to keep that
module within its line budget and each module single-responsibility.
"""

from __future__ import annotations

from pathlib import Path

# Color-grade mixers (Plan 09.9-12 chain). CINE_GRADE applies a warm-red /
# cool-blue shift plus a 3D LUT; NEUTRAL is identity (no shift) to match the
# 09.9-16 reference clips, which were generated before the LUT grade existed.
CINE_GRADE_COLOR_MIXER = (
    "colorchannelmixer=rr=1.04:rg=0.02:rb=0.01:"
    "gr=0.01:gg=1.02:gb=0.01:"
    "br=0.01:bg=0.01:bb=1.06"
)
NEUTRAL_COLOR_MIXER = "colorchannelmixer=rr=1.0:gg=1.0:bb=1.0"

# Generation frame rate; pre-roll/tail-loss padding is measured in frames
# (mv_comfyui.PRE_ROLL_FRAMES). Converts the pre-roll offset into an ffmpeg
# audio-delay (microseconds) for lip-sync realignment.
POST_GRADE_FPS = 24


def _build_post_video_filter(
    res_filter: str | None,
    apply_lut: bool,
    lut_path: str | None,
    grain_intensity: float,
    sharpen_strength: float,
) -> tuple[str, bool]:
    """Build the single-pass post-processing video filter chain.

    Returns (filter_chain, lut_applied). When apply_lut is False the chain is
    neutral (identity color mixer, no 3D LUT) so output matches the 09.9-16
    reference clips; when True it applies the cinematic Cine Grade mixer and the
    3D LUT if the .cube file exists.
    """
    filters: list[str] = []
    if res_filter:
        filters.append(res_filter)
    filters.append("format=yuv420p")
    filters.append(CINE_GRADE_COLOR_MIXER if apply_lut else NEUTRAL_COLOR_MIXER)
    lut_applied = False
    if apply_lut and lut_path and Path(lut_path).exists():
        filters.append(f"lut3d={lut_path}")
        lut_applied = True
    grain_scale = grain_intensity * 5.0
    filters.append(f"noise=alls=10:allf=+t:alls={grain_scale:.1f}")
    sharpen_val = sharpen_strength * 0.5
    filters.append(f"unsharp=7:7:{sharpen_val:.2f}")
    return ",".join(filters), lut_applied


def _build_audio_delay_filter(pre_roll_frames: int) -> str | None:
    """Build an ffmpeg audio-delay filter to realign lips after pre-roll trim.

    Clip generation pads PRE_ROLL_FRAMES at the start then trims them from the
    video, but the final audio is muxed from frame 0 — so the trimmed video lags
    the audio by PRE_ROLL_FRAMES. Delaying the audio by the same amount realigns
    them. Returns None when no delay is needed.
    """
    if pre_roll_frames <= 0:
        return None
    delay_us = int(pre_roll_frames / POST_GRADE_FPS * 1_000_000)
    return f"adelay={delay_us}:all=1"
