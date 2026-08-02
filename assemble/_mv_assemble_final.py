#!/usr/bin/env python3
"""Assemble final music video.

Steps:
1. Concatenate all 32 clips (clip_001_1080p.mp4 .. clip_032_1080p.mp4)
2. Map full audio (audio_original.mp3.bak, 200.04s)
3. Add fade-to-black after last clip
4. Render end credits sequence (black bg, white text, elegant)
5. Deliver final video
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
_MV = _REPO / "songs" / "music-videos" / "project-name"
_GEN = _MV / "gen-output"
_CLIPS = _GEN / "clips"
_AUDIO_FULL = _MV / "audio_original.mp3.bak"
_OUTPUT = _MV / "final"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

# ── Credits config ──
CREDITS_TEXT = """\
Song Title

(Version)

Performed by

ALICE

Lyrics & Creative Direction

ALICE

An ALICE AI Production

Created with AI

© 2026 ALICE AI"""

# ── Timing ──
# After clips: fade to black ~2s, pause ~1s, credits in, hold, credits out
FADE_TO_BLACK_S = 2.0       # 200.04 → 202.04
BLACK_PAUSE_S = 1.0         # 202.04 → 203.04
CREDITS_FADE_IN_S = 2.0     # 203.04 → 205.04 (credits fully visible)
CREDITS_HOLD_S = 14.0       # 205.04 → 219.04
CREDITS_FADE_OUT_S = 2.0    # 219.04 → 221.04
TRAILING_BLACK_S = 1.0      # 221.04 → 222.04 (final)

TOTAL_DURATION = 0  # computed after clip concat

FPS = 24


def get_clip_durations() -> list[float]:
    """Get duration of each clip_XXX_1080p.mp4."""
    durations = []
    for i in range(1, 33):
        clip = _CLIPS / f"clip_{i:03d}_1080p.mp4"
        if not clip.exists():
            raise FileNotFoundError(f"Missing: {clip}")
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
            capture_output=True, text=True, check=True,
        )
        durations.append(float(out.stdout.strip()))
    return durations


def create_concat_file(tmpdir: Path) -> Path:
    """Create FFmpeg concat input file."""
    f = tmpdir / "concat.txt"
    lines = []
    for i in range(1, 33):
        clip = _CLIPS / f"clip_{i:03d}_1080p.mp4"
        lines.append(f"file '{clip}'")
    f.write_text("\n".join(lines) + "\n")
    return f


def concatenate_clips(concat_file: Path, tmpdir: Path) -> tuple[Path, float]:
    """Concatenate all 32 clips into one video (no audio — added separately)."""
    output = tmpdir / "clips_concat.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-an",  # strip audio — we'll add full audio track
        str(output),
    ]
    log.info("Concatenating 32 clips...")
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    log.info("Concat done in %.0fs", time.time() - t0)

    # Get duration
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output)],
        capture_output=True, text=True, check=True,
    )
    dur = float(out.stdout.strip())
    return output, dur


def create_credits_image(path: Path) -> None:
    """Render credits as a 1920x1080 PNG using ffmpeg drawtext."""
    # Build drawtext filter chain — one per line, centered
    lines = CREDITS_TEXT.split("\n")
    h, w = 1080, 1920
    font_size = 28
    line_spacing = 42  # font_size + 14
    total_block_h = len(lines) * line_spacing
    start_y = (h - total_block_h) // 2 + font_size

    drawtext_filters = []
    for idx, line in enumerate(lines):
        y = start_y + idx * line_spacing
        # Escape special chars for ffmpeg
        escaped = line.replace("'", "\\'").replace("(", "\\(").replace(")", "\\)")
        escaped = escaped.replace("&", "\\&").replace(":", "\\:")
        drawtext_filters.append(
            f"drawtext=text='{escaped}':fontsize={font_size}:"
            f"fontcolor=white:x=(w-text_w)/2:y={y}:"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Regular.ttf"
        )

    filter_str = ",".join(drawtext_filters)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d=1:r={FPS}",
        "-vf", filter_str,
        "-frames:v", "1",
        str(path),
    ]
    log.info("Rendering credits image...")
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    log.info("Credits image: %s", path)


def build_final_video(
    clips_video: Path,
    clips_duration: float,
    credits_img: Path,
    tmpdir: Path,
) -> Path:
    """Build final video: clips → black → credits → black.

    Uses complex filtergraph:
    - clips video (with fadeout)
    - black bridge (fade from clips to full black)
    - credits (fade in/out)
    - audio (full track + silence for credits)
    """
    output = _OUTPUT / "project-name-final.mp4"
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    # Compute timing
    credits_start = clips_duration + FADE_TO_BLACK_S + BLACK_PAUSE_S
    total_dur = credits_start + CREDITS_FADE_IN_S + CREDITS_HOLD_S + CREDITS_FADE_OUT_S + TRAILING_BLACK_S

    log.info("Timing: clips=%.2fs, credits_start=%.2fs, total=%.2fs",
             clips_duration, credits_start, total_dur)

    # Create credits video (static image with duration, faded in/out)
    # The credits image is drawn on black, faded in over CREDITS_FADE_IN_S,
    # held, then faded out over CREDITS_FADE_OUT_S.
    credits_total = CREDITS_FADE_IN_S + CREDITS_HOLD_S + CREDITS_FADE_OUT_S

    # Filtergraph:
    # [0:v] clips → fade out last FADE_TO_BLACK_S → [clips_faded]
    # [color=black] black bridge for (FADE_TO_BLACK_S + BLACK_PAUSE_S) → [bridge]
    # [1:v] credits image → fadein + fadeout → [credits_faded]
    # [color=black] trailing black → [trail]
    # [clips_faded][bridge][credits_faded][trail] concat → [video]
    # [2:a] audio → adelay to align → [audio]

    bridge_dur = FADE_TO_BLACK_S + BLACK_PAUSE_S

    filter_complex = (
        # Clips with fade-out at the end
        f"[0:v]fade=t=out:st={clips_duration - FADE_TO_BLACK_S}:d={FADE_TO_BLACK_S}:alpha=0[clips_faded];"
        # Black bridge (fills gap between clips fade-out and credits)
        f"[color=black:s=1920x1080:d={bridge_dur}:r={FPS}:rate=24]null[bridge];"
        # Credits with fade-in and fade-out
        f"[1:v]loop=-1:size=1,tbt=1,fade=t=in:st=0:d={CREDITS_FADE_IN_S},"
        f"fade=t=out:st={credits_total - CREDITS_FADE_OUT_S}:d={CREDITS_FADE_OUT_S},"
        f"setpts=PTS-STARTPTS,trim=duration={credits_total}[credits_faded];"
        # Trailing black
        f"[color=black:s=1920x1080:d={TRAILING_BLACK_S}:r={FPS}:rate=24]null[trail];"
        # Concat all video segments
        f"[clips_faded][bridge][credits_faded][trail]concat=n=4:v=1:a=0[video];"
        # Audio: full track + silence for credits tail
        f"[2:a]adelay={int((total_dur - 200.04) * 1000)}|{int((total_dur - 200.04) * 1000)}[audio];"
    )

    cmd = [
        "ffmpeg", "-y",
        # Input 0: concatenated clips
        "-i", str(clips_video),
        # Input 1: credits image
        "-loop", "1", "-t", str(credits_total), "-i", str(credits_img),
        # Input 2: full audio
        "-i", str(_AUDIO_FULL),
        # Filter
        "-filter_complex", filter_complex,
        # Map outputs
        "-map", "[video]",
        "-map", "[audio]",
        # Encode
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-t", str(total_dur),
        str(output),
    ]

    log.info("Building final video (%.1fs)...", total_dur)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        log.error("FFmpeg stderr: %s", result.stderr[-2000:])
        raise RuntimeError(f"FFmpeg failed: {result.returncode}")
    log.info("Final video done in %.0fs", time.time() - t0)

    # Verify
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
         "-of", "default=noprint_wrappers=1", str(output)],
        capture_output=True, text=True, check=True,
    )
    log.info("Output: %s", out.stdout.strip())
    return output


def main() -> int:
    log.info("=" * 60)
    log.info("Music Video — Final Assembly")
    log.info("=" * 60)

    _OUTPUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mv_assemble_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Get clip durations
        durations = get_clip_durations()
        clips_total = sum(durations)
        log.info("32 clips total duration: %.2fs", clips_total)

        # Step 2: Concatenate clips
        concat_file = create_concat_file(tmpdir)
        clips_video, actual_dur = concatenate_clips(concat_file, tmpdir)
        log.info("Concatenated video: %.2fs", actual_dur)

        # Step 3: Render credits image
        credits_img = tmpdir / "credits.png"
        create_credits_image(credits_img)

        # Step 4: Build final video
        final = build_final_video(clips_video, actual_dur, credits_img, tmpdir)

        # Step 5: Copy to downloads for WhatsApp
        import shutil
        dl = _REPO / "downloads" / "project-name-final.mp4"
        shutil.copy2(final, dl)
        size_mb = dl.stat().st_size / (1024 * 1024)
        log.info("Copied to downloads: %.1f MB", size_mb)

        # Step 6: Publish to artifacts
        artifact = _REPO / "artifacts" / "project-artifact-id"
        shutil.copy2(final, artifact / "project-name-final.mp4")
        log.info("Published to artifacts")

    log.info("")
    log.info("=" * 60)
    log.info(f"ASSEMBLY COMPLETE — {final}")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
