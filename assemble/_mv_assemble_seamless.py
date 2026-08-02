#!/usr/bin/env python3
"""Assemble final MV with seamless audio — non-overlapping slices from original track.

Fixes the audio stutter at clip boundaries caused by overlapping segment plans.

Approach:
1. Concat all 32 clips as video-only (stream copy, -an)
2. Build audio track from original audio.mp3 using non-overlapping boundaries
   - Clips 1-27 (song): exact slices from original track, extended to match
     video duration via loop fill for LTX 8k+1 frame padding
   - Clips 28-32 (outro): use clip's own audio
3. Merge video + audio with credits tail
4. Verify zero silence gaps at all 31 clip boundaries

See .planning/debug/mv-lip-sync-drift.md for full investigation.
"""
from __future__ import annotations

import json
import logging
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
_MV = _REPO / "songs" / "music-videos" / "modotte-oide-yui"
_GEN = _MV / "gen-output"
_CLIPS = _GEN / "clips"
_AUDIO = _MV / "audio.mp3"  # 155.83s original stereo track
_SEGMENT_PLAN = _GEN / "segment_plan.json"
_OUTPUT = _MV / "final"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
)
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
NUM_CLIPS = 32
SONG_CLIP_COUNT = 27  # clips 1-27 are song, 28-32 are outro


def get_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def compute_non_overlapping_boundaries() -> list[dict]:
    """Compute non-overlapping audio boundaries from segment plan.

    For clips 1-27 (song clips), each clip's audio end = next clip's audio start.
    This eliminates the overlapping regions that caused stutter.

    Returns list of {index, audio_start, audio_end} for clips 1-27.
    """
    plan = json.loads(_SEGMENT_PLAN.read_text())
    segments = plan["segments"]  # 27 segments, index 1-27

    # Sort by index
    segments.sort(key=lambda s: s["index"])

    boundaries = []
    for i, seg in enumerate(segments):
        audio_start = seg["start"]
        # Non-overlapping: end = next segment's start, or seg's end if last
        if i + 1 < len(segments):
            audio_end = segments[i + 1]["start"]
        else:
            audio_end = seg["end"]
        boundaries.append({
            "index": seg["index"],
            "audio_start": audio_start,
            "audio_end": audio_end,
            "duration": audio_end - audio_start,
        })

    return boundaries


def verify_no_overlaps(boundaries: list[dict]) -> bool:
    """Verify non-overlapping boundaries."""
    ok = True
    for i in range(len(boundaries) - 1):
        a_end = boundaries[i]["audio_end"]
        b_start = boundaries[i + 1]["audio_start"]
        gap = b_start - a_end
        if abs(gap) > 0.01:
            log.error("BOUNDARY ISSUE: clip %d ends %.3f, clip %d starts %.3f (gap=%.3f)",
                      boundaries[i]["index"], a_end,
                      boundaries[i + 1]["index"], b_start, gap)
            ok = False
    # Verify coverage
    total = sum(b["duration"] for b in boundaries)
    log.info("Non-overlapping song audio total: %.3fs (original: ~155.78s)", total)
    return ok


def extract_audio_slices(
    tmpdir: Path,
    boundaries: list[dict],
) -> list[Path]:
    """Extract audio slices for clips 1-32.

    For song clips (1-27): slice from original track, extend to match video
    duration using loop fill for 8k+1 frame padding.
    For outro clips (28-32): extract clip's own audio.
    """
    audio_files = []

    for i in range(NUM_CLIPS):
        clip_idx = i + 1
        clip_file = _CLIPS / f"clip_{clip_idx:03d}_1080p.mp4"
        video_dur = get_duration(clip_file)
        output = tmpdir / f"audio_{i:03d}.aac"

        if clip_idx <= SONG_CLIP_COUNT:
            # Song clip: non-overlapping slice from original track
            b = boundaries[clip_idx - 1]  # boundaries[0] = clip 1
            audio_dur = b["duration"]

            if audio_dur < video_dur - 0.01:
                # Video is longer due to LTX 8k+1 frame padding.
                # Extract audio slice, then apad extends with last-sample
                # loop fill to match video duration.
                pad_ms = int((video_dur - audio_dur) * 1000)
                whole_dur_ms = int(video_dur * 1000)
                log.info("Clip %03d: audio %.3fs -> apad to %.3fs (+%dms loop fill)",
                         clip_idx, audio_dur, video_dur, pad_ms)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(b["audio_start"]),
                    "-i", str(_AUDIO),
                    "-t", str(audio_dur),
                    "-af", f"apad=whole_dur={whole_dur_ms}",
                    "-c:a", "aac", "-b:a", "192k",
                    str(output),
                ]
            else:
                # Audio duration >= video duration (or close enough).
                # Extract exactly video_dur from original track.
                log.info("Clip %03d: audio %.3fs (video %.3fs) — exact slice",
                         clip_idx, audio_dur, video_dur)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(b["audio_start"]),
                    "-i", str(_AUDIO),
                    "-t", str(video_dur),
                    "-c:a", "aac", "-b:a", "192k",
                    str(output),
                ]
        else:
            # Outro clip (28-32): use clip's own audio
            log.info("Clip %03d: outro — clip audio (%.3fs)", clip_idx, video_dur)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(clip_file),
                "-vn", "-c:a", "aac", "-b:a", "192k",
                str(output),
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.returncode != 0:
            log.error("Clip %03d audio extract failed: %s",
                      clip_idx, result.stderr[-500:])
            raise RuntimeError(f"Audio extract failed for clip {clip_idx}")
        audio_files.append(output)

    return audio_files


def concat_video_only(tmpdir: Path) -> tuple[Path, float]:
    """Concatenate 32 clips as video-only (stream copy)."""
    output = tmpdir / "video_only.mp4"

    concat_file = tmpdir / "concat.txt"
    lines = [f"file '{_CLIPS / f'clip_{i:03d}_1080p.mp4'}'" for i in range(1, NUM_CLIPS + 1)]
    concat_file.write_text("\n".join(lines) + "\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy", "-an",
        str(output),
    ]

    log.info("Concatenating %d clips (video-only stream copy)...", NUM_CLIPS)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("FFmpeg stderr: %s", result.stderr[-1000:])
        raise RuntimeError(f"Concat failed (rc={result.returncode})")
    log.info("Concat done in %.1fs", time.time() - t0)

    dur = get_duration(output)
    log.info("Video-only concat: %.2fs", dur)
    return output, dur


def concat_audio(audio_files: list[Path], tmpdir: Path) -> tuple[Path, float]:
    """Concatenate audio slices into a single audio track."""
    output = tmpdir / "audio_only.aac"

    concat_file = tmpdir / "audio_concat.txt"
    concat_file.write_text("\n".join(
        f"file '{af}'" for af in audio_files
    ) + "\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:a", "copy",
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        log.error("FFmpeg stderr: %s", result.stderr[-1000:])
        raise RuntimeError(f"Audio concat failed (rc={result.returncode})")

    dur = get_duration(output)
    log.info("Audio concat: %.2fs", dur)
    return output, dur


def merge_video_audio(video_file: Path, audio_file: Path, tmpdir: Path) -> tuple[Path, float]:
    """Merge video + audio with -shortest."""
    output = tmpdir / "clips_merged.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_file),
        "-i", str(audio_file),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("FFmpeg stderr: %s", result.stderr[-1000:])
        raise RuntimeError(f"Merge failed (rc={result.returncode})")

    dur = get_duration(output)
    log.info("Merged video+audio: %.2fs", dur)
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


def build_final(clips_merged: Path, clips_duration: float,
                tmpdir: Path) -> Path:
    """Build final video: clips (fade-out) -> black -> credits -> trailing black."""
    output = _OUTPUT / "modotte-oide-yui-final.mp4"
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    # Backup existing final
    if output.exists():
        bak = output.with_suffix(".bak")
        shutil.copy2(output, bak)
        log.info("Backed up existing final to %s", bak)

    credits_img = tmpdir / "credits.png"
    render_credits_image(credits_img)

    credits_start = clips_duration + FADE_TO_BLACK_S + BLACK_PAUSE_S
    credits_total = CREDITS_FADE_IN_S + CREDITS_HOLD_S + CREDITS_FADE_OUT_S
    bridge_dur = FADE_TO_BLACK_S + BLACK_PAUSE_S
    total_dur = clips_duration + bridge_dur + credits_total + TRAILING_BLACK_S

    log.info("Timing: clips=%.2fs, credits@%.2fs, total=%.2fs",
             clips_duration, credits_start, total_dur)

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
        # Audio: pad with silence for credits tail
        f"[0:a]apad=whole_dur={total_dur*1000}[aout];"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clips_merged),
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

    log.info("Building final video (%.1fs, NVENC)...", total_dur)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        log.error("FFmpeg stderr (last 2000 chars):")
        log.error(result.stderr[-2000:])
        raise RuntimeError(f"Final build failed (rc={result.returncode})")
    log.info("Final video done in %.0fs", time.time() - t0)

    dur = get_duration(output)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("Output: %.2fs, %.1f MB", dur, size_mb)
    return output


def verify_audio_seams(final_video: Path) -> bool:
    """Check for silence gaps at all clip boundaries in the final video."""
    log.info("=" * 60)
    log.info("Audio seam verification")
    log.info("=" * 60)

    # Extract audio to WAV
    import tempfile as tf
    with tf.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wav_path = Path(wf.name)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(final_video),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(wav_path),
    ], capture_output=True, check=True)

    with open(wav_path, "rb") as f:
        f.read(44)
        data = f.read()
    wav_path.unlink()

    bytes_per_sample = 4
    samples_per_sec = 44100

    def get_rms(time_s: float, window: float = 0.1) -> float:
        start = int(time_s * samples_per_sec * bytes_per_sample)
        n = int(window * samples_per_sec)
        chunk = data[start:start + n * bytes_per_sample]
        if not chunk:
            return 0.0
        samples = struct.unpack("<%dh" % (len(chunk) // 2), chunk)
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    # Compute clip boundaries (cumulative video durations)
    clip_durs = []
    for i in range(1, NUM_CLIPS + 1):
        clip_file = _CLIPS / f"clip_{i:03d}_1080p.mp4"
        clip_durs.append(get_duration(clip_file))

    boundaries = []
    cumulative = 0.0
    for dur in clip_durs[:-1]:
        cumulative += dur
        boundaries.append(cumulative)

    log.info("Checking %d clip boundaries...", len(boundaries))
    ok = True
    gaps_found = 0
    for b in boundaries:
        rms = get_rms(b)
        status = "OK" if rms >= 100 else "SILENCE GAP"
        if rms < 100:
            ok = False
            gaps_found += 1
        log.info("  Boundary %.3fs: RMS = %.1f — %s", b, rms, status)

    if ok:
        log.info("PASS: Zero silence gaps at all %d boundaries.", len(boundaries))
    else:
        log.info("FAIL: %d silence gaps detected.", gaps_found)

    return ok


def main() -> int:
    log.info("=" * 60)
    log.info("Modotte Oide Yui — Seamless Audio Assembly")
    log.info("=" * 60)

    # Step 1: Compute non-overlapping boundaries
    log.info("")
    log.info("Step 1: Computing non-overlapping audio boundaries...")
    boundaries = compute_non_overlapping_boundaries()
    ok = verify_no_overlaps(boundaries)
    if not ok:
        log.error("Overlapping boundaries detected! Aborting.")
        return 1

    # Log boundaries
    for b in boundaries:
        log.info("  Clip %02d: %.2f -> %.2f (%.2fs)",
                 b["index"], b["audio_start"], b["audio_end"], b["duration"])

    with tempfile.TemporaryDirectory(prefix="mv_seamless_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 2: Concat video-only
        log.info("")
        log.info("Step 2: Concatenating video-only...")
        video_file, video_dur = concat_video_only(tmpdir)

        # Step 3: Extract audio slices
        log.info("")
        log.info("Step 3: Extracting audio slices...")
        audio_files = extract_audio_slices(tmpdir, boundaries)

        # Step 4: Concat audio
        log.info("")
        log.info("Step 4: Concatenating audio...")
        audio_file, audio_dur = concat_audio(audio_files, tmpdir)

        # Step 5: Merge video + audio
        log.info("")
        log.info("Step 5: Merging video + audio...")
        clips_merged, merged_dur = merge_video_audio(video_file, audio_file, tmpdir)

        # Step 6: Verify seams on merged clips (before credits)
        log.info("")
        log.info("Step 6: Verifying audio seams...")
        seams_ok = verify_audio_seams(clips_merged)

        # Step 7: Build final with credits
        log.info("")
        log.info("Step 7: Building final video with credits...")
        final = build_final(clips_merged, merged_dur, tmpdir)

        # Step 8: Copy to downloads
        dl = _REPO / "downloads" / "modotte-oide-yui-final.mp4"
        shutil.copy2(final, dl)
        log.info("Copied to downloads: %.1f MB", dl.stat().st_size / (1024 * 1024))

        # Step 9: Publish to artifacts
        artifact = _REPO / "artifacts" / "55d180e3-d2aa-48ff-b64e-9c60103f04fc"
        artifact.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, artifact / "modotte-oide-yui-final.mp4")
        log.info("Published to artifacts")

    log.info("")
    log.info("=" * 60)
    if seams_ok:
        log.info("ASSEMBLY COMPLETE — zero audio seams verified")
    else:
        log.info("ASSEMBLY COMPLETE — WARNING: audio seam issues detected")
    log.info("Final: %s", final)
    log.info("=" * 60)

    return 0 if seams_ok else 1


if __name__ == "__main__":
    sys.exit(main())
