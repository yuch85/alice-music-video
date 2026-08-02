#!/usr/bin/env python3
"""MV credits rendering and final encode helper.

Extracted from _mv_assemble_cascade.py to stay within 300 LOC limit (STYLE.md).
Shared constants and functions for end-credits sequences.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

FPS = 24
WIDTH, HEIGHT = 1920, 1080

FADE_TO_BLACK_S = 2.0
BLACK_PAUSE_S = 1.0
CREDITS_FADE_IN_S = 2.0
CREDITS_HOLD_S = 14.0
CREDITS_FADE_OUT_S = 2.0
TRAILING_BLACK_S = 1.0

CREDITS_LINES = [
    "Song Title", "", "(Version)", "", "",
    "Performed by", "", "ALICE", "", "",
    "Lyrics & Creative Direction", "", "ALICE", "", "",
    "An ALICE AI Production", "", "",
    "Created with AI", "", "",
    "© 2026 ALICE AI",
]

log = logging.getLogger(__name__)


def _get_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def render_credits_image(path: Path) -> None:
    """Render credits as a 1920x1080 PNG using ffmpeg drawtext."""
    font_size = 28
    line_h = 36
    total_h = len(CREDITS_LINES) * line_h
    start_y = (HEIGHT - total_h) // 2 + font_size

    font = None
    for f in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        if Path(f).exists():
            font = f
            break
    if font is None:
        font = "DejaVuSans"
        log.warning("No font file found, using default")

    drawtext_parts = []
    for idx, line in enumerate(CREDITS_LINES):
        y = start_y + idx * line_h
        if line.strip():
            escaped = line.replace("'", "\\'").replace("(", "\\(").replace(")", "\\)")
            escaped = escaped.replace("&", "\\&").replace(":", "\\:")
            escaped = escaped.replace("<", "\\<").replace(">", "\\>")
            escaped = escaped.replace("%", "\\%")
            drawtext_parts.append(
                f"drawtext=text='{escaped}':fontsize={font_size}:"
                f"fontcolor=white:x=(w-text_w)/2:y={y}:"
                f"fontfile={font}"
            )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:d=1:r={FPS}",
        "-vf", ",".join(drawtext_parts),
        "-frames:v", "1", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)


def build_final_with_credits(
    *,
    concat_path: Path,
    final_path: Path,
    total_duration: float,
) -> Path:
    """Add end-credits sequence (fade, black, credits, fade) with NVENC encode.

    Takes the concatenated clips video and produces the final MP4 with
    credits appended. Audio is padded with silence for the credits tail.
    """
    credits_total = CREDITS_FADE_IN_S + CREDITS_HOLD_S + CREDITS_FADE_OUT_S
    bridge_dur = FADE_TO_BLACK_S + BLACK_PAUSE_S
    final_total = total_duration + bridge_dur + credits_total + TRAILING_BLACK_S

    log.info(
        "Credits timing: clips=%.2fs, bridge=%.1fs, credits=%.1fs, total=%.2fs",
        total_duration, bridge_dur, credits_total, final_total,
    )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    credits_img = final_path.with_suffix(".credits.png")
    render_credits_image(credits_img)

    filter_complex = (
        f"[0:v]fade=t=out:st={total_duration - FADE_TO_BLACK_S}:"
        f"d={FADE_TO_BLACK_S}[vf1];"
        f"[2:v]setpts=PTS-STARTPTS[vf2];"
        f"[1:v]trim=duration={credits_total},"
        f"setpts=PTS-STARTPTS,"
        f"fade=t=in:st=0:d={CREDITS_FADE_IN_S},"
        f"fade=t=out:st={credits_total - CREDITS_FADE_OUT_S}:"
        f"d={CREDITS_FADE_OUT_S}[vf3];"
        f"[3:v]setpts=PTS-STARTPTS[vf4];"
        f"[vf1][vf2][vf3][vf4]concat=n=4:v=1:a=0[vout];"
        f"[0:a]apad=whole_dur={final_total*1000}[aout];"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(concat_path),
        "-loop", "1", "-t", str(credits_total), "-i", str(credits_img),
        "-f", "lavfi", "-i",
        f"color=c=black:s={WIDTH}x{HEIGHT}:d={bridge_dur}:r={FPS}",
        "-f", "lavfi", "-i",
        f"color=c=black:s={WIDTH}x{HEIGHT}:d={TRAILING_BLACK_S}:r={FPS}",
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "h264_nvenc", "-cq", "23", "-b:v", "0",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-t", str(final_total),
        str(final_path),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        log.error("build_final_with_credits failed: %s", result.stderr[-2000:])
        raise RuntimeError(f"build_final_with_credits failed (rc={result.returncode})")
    log.info("Final video with credits: %.0fs", time.time() - t0)

    credits_img.unlink(missing_ok=True)
    return final_path
