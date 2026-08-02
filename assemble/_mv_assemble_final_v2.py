#!/usr/bin/env python3
"""Assemble final Modotte Oide Yui music video — optimized (stream copy concat).

Steps:
1. Concatenate 32 clips via stream copy (no re-encode)
2. Render credits image
3. Build final: clips (fade-out) → black → credits (fade in/out) → black
4. Add full audio track extended with silence for credits tail
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
_MV = _REPO / "songs" / "music-videos" / "modotte-oide-yui"
_GEN = _MV / "gen-output"
_CLIPS = _GEN / "clips"
_AUDIO_FULL = _MV / "audio_original.mp3.bak"
_OUTPUT = _MV / "final"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
)
# Force line-buffered stdout for real-time logging
logger = logging.getLogger()
for h in logger.handlers:
    h.flush = lambda: None  # ensure flush works

log = logging.getLogger(__name__)

# ── Credits text ──
CREDITS_LINES = [
    "Modotte Oide",
    "",
    "(Acoustic Plus)",
    "",
    "",
    "Performed by",
    "",
    "ALICE",
    "",
    "",
    "Lyrics & Creative Direction",
    "",
    "ALICE",
    "",
    "",
    "An ALICE AI Production",
    "",
    "",
    "Created with AI",
    "",
    "",
    "© 2026 ALICE AI",
]

# ── Timing ──
FADE_TO_BLACK_S = 2.0
BLACK_PAUSE_S = 1.0
CREDITS_FADE_IN_S = 2.0
CREDITS_HOLD_S = 14.0
CREDITS_FADE_OUT_S = 2.0
TRAILING_BLACK_S = 1.0

FPS = 24
WIDTH, HEIGHT = 1920, 1080


def get_clip_duration(clip: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def concat_stream_copy(tmpdir: Path) -> tuple[Path, float]:
    """Concatenate 32 clips via concat demuxer (video copy, audio re-encode)."""
    output = tmpdir / "clips_concat.mp4"

    # Create concat file list
    concat_file = tmpdir / "concat.txt"
    lines = []
    for i in range(1, 33):
        clip = _CLIPS / f"clip_{i:03d}_1080p.mp4"
        if not clip.exists():
            raise FileNotFoundError(f"Missing: {clip}")
        lines.append(f"file '{clip}'")
    concat_file.write_text("\n".join(lines) + "\n")

    # Use concat demuxer — re-mux only, re-encode audio to AAC for compatibility
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]

    log.info("Concatenating 32 clips (video copy + audio remux)...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("FFmpeg stderr: %s", result.stderr[-1000:])
        raise RuntimeError(f"Concat failed (rc={result.returncode})")
    log.info("Concat done in %.1fs", time.time() - t0)

    dur = get_clip_duration(output)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("Concatenated: %.2fs, %.1f MB", dur, size_mb)
    return output, dur


def render_credits_image(path: Path) -> None:
    """Render credits as 1920x1080 PNG."""
    font_size = 28
    line_h = 36
    total_h = len(CREDITS_LINES) * line_h
    start_y = (HEIGHT - total_h) // 2 + font_size

    # Check for font file
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    font = None
    for f in font_candidates:
        if Path(f).exists():
            font = f
            break
    if font is None:
        # Fallback: use default font (may produce warning)
        font = "DejaVuSans"
        log.warning("No font file found, using default")

    drawtext_filters = []
    for idx, line in enumerate(CREDITS_LINES):
        y = start_y + idx * line_h
        if line.strip():
            escaped = line.replace("'", "\\'").replace("(", "\\(").replace(")", "\\)")
            escaped = escaped.replace("&", "\\&").replace(":", "\\:")
            escaped = escaped.replace("<", "\\<").replace(">", "\\>")
            escaped = escaped.replace("%", "\\%")
            drawtext_filters.append(
                f"drawtext=text='{escaped}':fontsize={font_size}:"
                f"fontcolor=white:x=(w-text_w)/2:y={y}:"
                f"fontfile={font}"
            )

    filter_str = ",".join(drawtext_filters)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:d=1:r={FPS}",
        "-vf", filter_str,
        "-frames:v", "1",
        str(path),
    ]
    log.info("Rendering credits image...")
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    log.info("Credits image: %s", path)


def build_final(clips_video: Path, clips_duration: float, credits_img: Path,
                tmpdir: Path) -> Path:
    """Build final video with fade transitions and credits."""
    output = _OUTPUT / "modotte-oide-yui-final.mp4"
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    credits_start = clips_duration + FADE_TO_BLACK_S + BLACK_PAUSE_S
    credits_total = CREDITS_FADE_IN_S + CREDITS_HOLD_S + CREDITS_FADE_OUT_S
    bridge_dur = FADE_TO_BLACK_S + BLACK_PAUSE_S
    total_dur = clips_duration + bridge_dur + credits_total + TRAILING_BLACK_S

    log.info("Timing: clips=%.2fs, credits@%.2fs, total=%.2fs",
             clips_duration, credits_start, total_dur)

    silence_ms = max(0, int((total_dur - 200.04) * 1000))

    # Use lavfi inputs for black screens + credits image as separate inputs
    # Input 0: clips_video
    # Input 1: credits image (looped)
    # Input 2: black bridge (lavfi)
    # Input 3: trailing black (lavfi)
    filter_complex = (
        # Clips with fade-out
        f"[0:v]fade=t=out:st={clips_duration - FADE_TO_BLACK_S}:"
        f"d={FADE_TO_BLACK_S}[vf1];"
        # Black bridge (input 2)
        f"[2:v]setpts=PTS-STARTPTS[vf2];"
        # Credits with fade in/out (input 1, looped)
        f"[1:v]trim=duration={credits_total},"
        f"setpts=PTS-STARTPTS,"
        f"fade=t=in:st=0:d={CREDITS_FADE_IN_S},"
        f"fade=t=out:st={credits_total - CREDITS_FADE_OUT_S}:"
        f"d={CREDITS_FADE_OUT_S}[vf3];"
        # Trailing black (input 3)
        f"[3:v]setpts=PTS-STARTPTS[vf4];"
        # Concat all video
        f"[vf1][vf2][vf3][vf4]concat=n=4:v=1:a=0[vout];"
        # Audio: pad with silence to fill total duration
        f"[0:a]apad=whole_dur={total_dur*1000}[aout];"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clips_video),
        "-loop", "1", "-t", str(credits_total), "-i", str(credits_img),
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:d={bridge_dur}:r={FPS}",
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:d={TRAILING_BLACK_S}:r={FPS}",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_nvenc", "-cq", "23", "-b:v", "0",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-t", str(total_dur),
        str(output),
    ]

    log.info("Building final video (%.1fs)...", total_dur)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        log.error("FFmpeg stderr (last 2000 chars):")
        log.error(result.stderr[-2000:])
        raise RuntimeError(f"FFmpeg final build failed (rc={result.returncode})")
    log.info("Final video done in %.0fs", time.time() - t0)

    # Verify
    dur = get_clip_duration(output)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("Output: %.2fs, %.1f MB", dur, size_mb)
    return output


def main() -> int:
    log.info("=" * 60)
    log.info("Modotte Oide Yui — Final Assembly v2")
    log.info("=" * 60)

    _OUTPUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mv_assemble_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Concat clips (stream copy — fast)
        clips_video, clips_dur = concat_stream_copy(tmpdir)

        # Step 2: Render credits
        credits_img = tmpdir / "credits.png"
        render_credits_image(credits_img)

        # Step 3: Build final video
        final = build_final(clips_video, clips_dur, credits_img, tmpdir)

        # Step 4: Copy to downloads
        dl = _REPO / "downloads" / "modotte-oide-yui-final.mp4"
        shutil.copy2(final, dl)
        log.info("Copied to downloads: %.1f MB", dl.stat().st_size / (1024*1024))

        # Step 5: Publish to artifacts
        artifact = _REPO / "artifacts" / "55d180e3-d2aa-48ff-b64e-9c60103f04fc"
        shutil.copy2(final, artifact / "modotte-oide-yui-final.mp4")
        log.info("Published to artifacts")

    log.info("")
    log.info("=" * 60)
    log.info(f"ASSEMBLY COMPLETE — {final}")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
