#!/usr/bin/env python3
"""Fix audio stutter in final music video.

Root cause: Each clip has mono demucs vocals (not full track).
Concatenating 32 clips = 32 audio seams with discontinuous vocals.

Fix: Strip all clip audio, concat video only, overlay original stereo audio track.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
_MV = _REPO / "songs" / "music-videos" / "project-name"
_GEN = _MV / "gen-output"
_CLIPS = _GEN / "clips"
_SEGMENT_PLAN = _GEN / "segment_plan.json"
# Original full stereo audio track
_AUDIO_ORIGINAL = _MV / "audio_original.mp3.bak"  # 200.04s
_AUDIO_SONG = _MV / "audio.mp3"  # 155.83s
_OUTPUT = _MV / "final"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

# ── Credits text ──
CREDITS_LINES = [
    "Song Title",
    "",
    "(Version)",
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
NUM_CLIPS = 32


def get_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def strip_audio_and_concat(tmpdir: Path) -> tuple[Path, float]:
    """Strip audio from all clips and concatenate video-only (stream copy)."""
    output = tmpdir / "clips_video_only.mp4"

    # Create concat file list
    concat_file = tmpdir / "concat.txt"
    lines = []
    for i in range(1, NUM_CLIPS + 1):
        clip = _CLIPS / f"clip_{i:03d}_1080p.mp4"
        if not clip.exists():
            raise FileNotFoundError(f"Missing: {clip}")
        lines.append(f"file '{clip}'")
    concat_file.write_text("\n".join(lines) + "\n")

    # Concat demuxer with video copy + drop all audio (-an)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy",
        "-an",  # strip all audio
        str(output),
    ]

    log.info("Concatenating %d clips (video copy, audio stripped)...", NUM_CLIPS)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("FFmpeg stderr (last 2000 chars):")
        log.error(result.stderr[-2000:])
        raise RuntimeError(f"Concat failed (rc={result.returncode})")
    elapsed = time.time() - t0
    log.info("Concat done in %.1fs", elapsed)

    dur = get_duration(output)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("Video-only concat: %.2fs, %.1f MB", dur, size_mb)
    return output, dur


def render_credits_image(path: Path) -> None:
    """Render credits as 1920x1080 PNG."""
    font_size = 28
    line_h = 36
    total_h = len(CREDITS_LINES) * line_h
    start_y = (HEIGHT - total_h) // 2 + font_size

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


def build_final_with_original_audio(
    video_only: Path,
    clips_duration: float,
    credits_img: Path,
    tmpdir: Path,
) -> Path:
    """Build final: video + original audio track + credits.

    Audio strategy:
    - Use original audio.mp3 (155.83s full stereo track) for song portion
    - Extend with silence for outro clips and credits tail
    - Single audio source = zero seam artifacts
    """
    output = _OUTPUT / "project-name-final.mp4"
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    # Timing calculations
    credits_start = clips_duration + FADE_TO_BLACK_S + BLACK_PAUSE_S
    credits_total = CREDITS_FADE_IN_S + CREDITS_HOLD_S + CREDITS_FADE_OUT_S
    bridge_dur = FADE_TO_BLACK_S + BLACK_PAUSE_S
    total_dur = clips_duration + bridge_dur + credits_total + TRAILING_BLACK_S

    # Original audio duration
    audio_dur = get_duration(_AUDIO_SONG)
    log.info("Original audio: %.2fs, Video clips: %.2fs, Total output: %.2fs",
             audio_dur, clips_duration, total_dur)

    # Audio: original song + silence pad to fill total duration
    # The song (155.83s) covers the song clips. Outro clips (28-32) + bridge + credits = silence.
    silence_needed = max(0, total_dur - audio_dur)
    log.info("Silence padding: %.2fs (for outro + bridge + credits)", silence_needed)

    filter_complex = (
        # Video: clips with fade-out at end
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
        # Audio: original track + apad silence to fill total duration
        f"[4:a]apad=whole_dur={total_dur*1000}[aout];"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only),                          # 0: video clips
        "-loop", "1", "-t", str(credits_total),
        "-i", str(credits_img),                          # 1: credits image
        "-f", "lavfi", "-i",
        f"color=c=black:s={WIDTH}x{HEIGHT}:d={bridge_dur}:r={FPS}",  # 2: bridge
        "-f", "lavfi", "-i",
        f"color=c=black:s={WIDTH}x{HEIGHT}:d={TRAILING_BLACK_S}:r={FPS}",  # 3: trailing
        "-i", str(_AUDIO_SONG),                          # 4: original audio
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_nvenc", "-cq", "23", "-b:v", "0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        "-t", str(total_dur),
        str(output),
    ]

    log.info("Building final video (%.1fs, original audio)...", total_dur)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        log.error("FFmpeg stderr (last 2000 chars):")
        log.error(result.stderr[-2000:])
        raise RuntimeError(f"Final build failed (rc={result.returncode})")
    log.info("Final video done in %.0fs", time.time() - t0)

    # Verify
    dur = get_duration(output)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("Output: %.2fs, %.1f MB", dur, size_mb)

    # Verify audio streams
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,channels,sample_rate,duration",
         "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )
    import json as j
    audio_info = j.loads(probe.stdout)["streams"][0]
    log.info("Audio: %s, %dch, %dHz, %.2fs",
             audio_info["codec_name"], audio_info["channels"],
             audio_info["sample_rate"], audio_info["duration"])

    return output


def main() -> int:
    log.info("=" * 60)
    log.info("Music Video — Audio Seam Fix")
    log.info("=" * 60)

    _OUTPUT.mkdir(parents=True, exist_ok=True)

    # Back up existing final
    final = _OUTPUT / "project-name-final.mp4"
    if final.exists():
        backup = _OUTPUT / "project-name-final-bak.mp4"
        shutil.move(str(final), str(backup))
        log.info("Backed up existing final to %s", backup)

    with tempfile.TemporaryDirectory(prefix="mv_fix_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Strip audio + concat video
        video_only, clips_dur = strip_audio_and_concat(tmpdir)

        # Step 2: Render credits
        credits_img = tmpdir / "credits.png"
        render_credits_image(credits_img)

        # Step 3: Build final with original audio
        output = build_final_with_original_audio(
            video_only, clips_dur, credits_img, tmpdir
        )

        # Step 4: Copy to downloads
        dl = _REPO / "downloads" / "project-name-final.mp4"
        shutil.copy2(output, dl)
        log.info("Copied to downloads: %.1f MB", dl.stat().st_size / (1024*1024))

        # Step 5: Publish to artifacts
        artifact = _REPO / "artifacts" / "project-artifact-id"
        if artifact.exists():
            shutil.copy2(output, artifact / "project-name-final.mp4")
            log.info("Published to artifacts")

    log.info("")
    log.info("=" * 60)
    log.info(f"ASSEMBLY COMPLETE — {output}")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
