#!/usr/bin/env python3
"""Assemble final MV by trimming video frames to match non-overlapping audio.

For each clip:
1. Compute non-overlapping audio boundary from segment plan
2. Trim video to audio duration (drop extra 8k+1 frames from end)
3. Extract audio slice from original track
4. Merge trimmed video + audio (re-encode video via NVENC)

Clips 28-32 (outro): keep as-is (own audio, no trimming needed).

See .planning/debug/mv-lip-sync-drift.md for investigation.
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

# ── Credits ──
CREDITS_LINES = [
    "Modotte Oide", "", "(Acoustic Plus)", "", "",
    "Performed by", "", "ALICE", "", "",
    "Lyrics & Creative Direction", "", "ALICE", "", "",
    "An ALICE AI Production", "", "",
    "Created with AI", "", "",
    "© 2026 ALICE AI",
]

FADE_TO_BLACK_S = 2.0
BLACK_PAUSE_S = 1.0
CREDITS_FADE_IN_S = 2.0
CREDITS_HOLD_S = 14.0
CREDITS_FADE_OUT_S = 2.0
TRAILING_BLACK_S = 1.0
FPS = 24
WIDTH, HEIGHT = 1920, 1080
NUM_CLIPS = 32
SONG_CLIP_COUNT = 27


def get_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def get_non_overlapping_boundaries() -> list[dict]:
    """Compute non-overlapping audio boundaries for clips 1-27."""
    plan = json.loads(_SEGMENT_PLAN.read_text())
    segments = sorted(plan["segments"], key=lambda s: s["index"])

    boundaries = []
    for i, seg in enumerate(segments):
        audio_start = seg["start"]
        audio_end = segments[i + 1]["start"] if i + 1 < len(segments) else seg["end"]
        boundaries.append({
            "index": seg["index"],
            "audio_start": audio_start,
            "audio_end": audio_end,
            "audio_dur": audio_end - audio_start,
        })
    return boundaries


def trim_and_merge_clip(clip_idx: int, audio_dur: float, audio_start: float,
                        tmpdir: Path) -> Path:
    """Trim video to audio_dur, extract audio slice, merge. Re-encode video."""
    clip_file = _CLIPS / f"clip_{clip_idx:03d}_1080p.mp4"
    output = tmpdir / f"trimmed_{clip_idx:03d}.mp4"
    video_dur = get_duration(clip_file)

    trim_dur = audio_dur  # trim video to match audio
    frames_dropped = int(round((video_dur - audio_dur) * FPS))

    log.info("Clip %03d: video %.3fs -> trim to %.3fs (%d frames), audio %.2f-%.2fs",
             clip_idx, video_dur, trim_dur, frames_dropped, audio_start, audio_start + audio_dur)

    # Trim video + extract audio from original track in one pass
    cmd = [
        "ffmpeg", "-y",
        # Video input — trim to audio duration
        "-i", str(clip_file),
        # Audio input — slice from original track
        "-ss", str(audio_start), "-i", str(_AUDIO),
        # Trim video to audio duration, reset timestamps
        "-filter_complex",
        f"[0:v]trim=duration={trim_dur},setpts=PTS-STARTPTS[v];"
        f"[1:a]asetpts=PTS-STARTPTS[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "h264_nvenc", "-cq", "23", "-b:v", "0",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(trim_dur),
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("Clip %03d failed: %s", clip_idx, result.stderr[-500:])
        raise RuntimeError(f"Trim failed for clip {clip_idx}")

    return output


def process_outro_clip(clip_idx: int, tmpdir: Path) -> Path:
    """Copy outro clip as-is (own audio, no trim needed)."""
    clip_file = _CLIPS / f"clip_{clip_idx:03d}_1080p.mp4"
    output = tmpdir / f"trimmed_{clip_idx:03d}.mp4"

    # Stream copy — no re-encode for outro
    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_file),
        "-c:v", "copy", "-c:a", "copy",
        str(output),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    log.info("Clip %03d: outro (stream copy)", clip_idx)
    return output


def concat_clips(tmpdir: Path, clip_count: int) -> tuple[Path, float]:
    """Concatenate trimmed clips via concat demuxer."""
    output = tmpdir / "concat.mp4"
    concat_file = tmpdir / "concat.txt"

    lines = [f"file 'trimmed_{i:03d}.mp4'" for i in range(1, clip_count + 1)]
    concat_file.write_text("\n".join(lines) + "\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("Concat failed: %s", result.stderr[-1000:])
        raise RuntimeError("Concat failed")

    dur = get_duration(output)
    log.info("Concatenated: %.2fs", dur)
    return output, dur


def render_credits_image(path: Path) -> None:
    """Render credits as 1920x1080 PNG."""
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

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:d=1:r={FPS}",
        "-vf", ",".join(drawtext_filters),
        "-frames:v", "1", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)


def build_final(clips_concat: Path, clips_dur: float, tmpdir: Path) -> Path:
    """Build final: clips (fade-out) -> black -> credits -> trailing black."""
    output = _OUTPUT / "modotte-oide-yui-final.mp4"
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    if output.exists():
        shutil.copy2(output, output.with_suffix(".bak"))

    credits_img = tmpdir / "credits.png"
    render_credits_image(credits_img)

    credits_total = CREDITS_FADE_IN_S + CREDITS_HOLD_S + CREDITS_FADE_OUT_S
    bridge_dur = FADE_TO_BLACK_S + BLACK_PAUSE_S
    total_dur = clips_dur + bridge_dur + credits_total + TRAILING_BLACK_S

    log.info("Timing: clips=%.2fs, total=%.2fs", clips_dur, total_dur)

    filter_complex = (
        f"[0:v]fade=t=out:st={clips_dur - FADE_TO_BLACK_S}:"
        f"d={FADE_TO_BLACK_S}[vf1];"
        f"[2:v]setpts=PTS-STARTPTS[vf2];"
        f"[1:v]trim=duration={credits_total},setpts=PTS-STARTPTS,"
        f"fade=t=in:st=0:d={CREDITS_FADE_IN_S},"
        f"fade=t=out:st={credits_total - CREDITS_FADE_OUT_S}:"
        f"d={CREDITS_FADE_OUT_S}[vf3];"
        f"[3:v]setpts=PTS-STARTPTS[vf4];"
        f"[vf1][vf2][vf3][vf4]concat=n=4:v=1:a=0[vout];"
        f"[0:a]apad=whole_dur={total_dur*1000}[aout];"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clips_concat),
        "-loop", "1", "-t", str(credits_total), "-i", str(credits_img),
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:d={bridge_dur}:r={FPS}",
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:d={TRAILING_BLACK_S}:r={FPS}",
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "h264_nvenc", "-cq", "23", "-b:v", "0",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-t", str(total_dur), str(output),
    ]

    log.info("Building final (%.1fs, NVENC)...", total_dur)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        log.error("FFmpeg: %s", result.stderr[-2000:])
        raise RuntimeError("Final build failed")
    log.info("Done in %.0fs", time.time() - t0)

    return output


def verify_seams(video_path: Path) -> bool:
    """Verify zero silence gaps at clip boundaries."""
    import tempfile as tf

    with tf.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wav = Path(wf.name)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(wav),
    ], capture_output=True, check=True)

    with open(wav, "rb") as f:
        f.read(44)
        data = f.read()
    wav.unlink()

    def get_rms(t: float) -> float:
        samples_per_byte = 44100 * 2  # bytes per second for 16bit stereo
        start = int(t * samples_per_byte)
        chunk = data[start:start + int(0.1 * samples_per_byte)]
        if not chunk:
            return 0.0
        import struct as st
        samples = st.unpack("<%dh" % (len(chunk) // 2), chunk)
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    # Get trimmed clip durations
    clip_durs = []
    for i in range(1, NUM_CLIPS + 1):
        clip = _CLIPS / f"clip_{i:03d}_1080p.mp4"
        if i <= SONG_CLIP_COUNT:
            # Trimmed duration = audio duration
            b = get_non_overlapping_boundaries()[i - 1]
            clip_durs.append(b["audio_dur"])
        else:
            clip_durs.append(get_duration(clip))

    boundaries = []
    cum = 0.0
    for d in clip_durs[:-1]:
        cum += d
        boundaries.append(cum)

    log.info("Checking %d boundaries...", len(boundaries))
    ok = True
    for b in boundaries:
        rms = get_rms(b)
        status = "OK" if rms >= 100 else "GAP"
        if rms < 100:
            ok = False
        log.info("  %.3fs: RMS=%.1f %s", b, rms, status)

    return ok


def main() -> int:
    log.info("=" * 60)
    log.info("MV Assembly — Trim Video to Match Audio")
    log.info("=" * 60)

    boundaries = get_non_overlapping_boundaries()

    # Log what we're doing
    total_frames = 0
    for b in boundaries:
        clip_file = _CLIPS / f"clip_{b['index']:03d}_1080p.mp4"
        video_dur = get_duration(clip_file)
        frames = int(round((video_dur - b["audio_dur"]) * FPS))
        total_frames += frames
        log.info("  Clip %02d: %.3fs -> %.3fs (drop %d frames)",
                 b["index"], video_dur, b["audio_dur"], frames)
    log.info("Total frames dropped: %d (~%.1fs)", total_frames, total_frames / FPS)

    with tempfile.TemporaryDirectory(prefix="mv_trim_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Trim song clips + extract audio
        log.info("")
        log.info("Step 1: Trimming song clips (1-27)...")
        for i, b in enumerate(boundaries, 1):
            trim_and_merge_clip(i, b["audio_dur"], b["audio_start"], tmpdir)

        # Step 2: Copy outro clips
        log.info("")
        log.info("Step 2: Processing outro clips (28-32)...")
        for i in range(28, NUM_CLIPS + 1):
            process_outro_clip(i, tmpdir)

        # Step 3: Concatenate
        log.info("")
        log.info("Step 3: Concatenating %d clips...", NUM_CLIPS)
        concat_file, concat_dur = concat_clips(tmpdir, NUM_CLIPS)

        # Step 4: Verify seams
        log.info("")
        log.info("Step 4: Verifying audio seams...")
        seams_ok = verify_seams(concat_file)

        # Step 5: Build final with credits
        log.info("")
        log.info("Step 5: Building final video...")
        final = build_final(concat_file, concat_dur, tmpdir)

        # Step 6: Deliver
        dl = _REPO / "downloads" / "modotte-oide-yui-final.mp4"
        shutil.copy2(final, dl)
        log.info("Downloads: %.1f MB", dl.stat().st_size / (1024 * 1024))

        artifact = _REPO / "artifacts" / "55d180e3-d2aa-48ff-b64e-9c60103f04fc"
        artifact.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, artifact / "modotte-oide-yui-final.mp4")
        log.info("Published to artifacts")

    log.info("")
    log.info("=" * 60)
    log.info(f"COMPLETE — seams {'PASS' if seams_ok else 'FAIL'}")
    log.info(f"Final: {final}")
    log.info("=" * 60)

    return 0 if seams_ok else 1


if __name__ == "__main__":
    sys.exit(main())
