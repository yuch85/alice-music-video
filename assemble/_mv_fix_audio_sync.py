#!/usr/bin/env python3
"""Re-trim audio in already-trimmed clips to fix audio sync bug.

The existing clip_XXX_up.mp4 files have video correctly trimmed (4 pre-roll
+ 4 tail-loss frames removed) but audio was not trimmed. This script trims
only the audio stream to match the existing video duration, producing
clip_XXX_up_fixed.mp4 files.

Usage:
    uv run python scripts/_mv_fix_audio_sync.py \
        --input songs/music-videos/project-name/gen-output/clips/ \
        --output songs/music-videos/project-name/gen-output-option-a/clips/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _ffprobe_json(path: str | Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "stream=codec_type,duration,nb_frames,width,height",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def _get_stream_duration(data: dict, codec_type: str) -> float:
    for stream in data.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return float(stream["duration"])
    return 0.0


def fix_audio_sync(input_clip: Path, output_clip: Path) -> bool:
    """Trim audio to match video duration in an already-video-trimmed clip.

    Strategy: copy video stream unchanged, trim audio to video duration
    using atrim + asetpts, re-encode audio to aac.
    """
    data = _ffprobe_json(str(input_clip))
    vid_dur = _get_stream_duration(data, "video")
    aud_dur = _get_stream_duration(data, "audio")

    if vid_dur <= 0:
        print(f"  SKIP: no video stream in {input_clip.name}")
        return False

    if abs(vid_dur - aud_dur) < 0.005:
        print(f"  SKIP: {input_clip.name} already synced "
              f"(video={vid_dur:.3f}s, audio={aud_dur:.3f}s)")
        # Just copy the file
        import shutil
        shutil.copy2(input_clip, output_clip)
        return True

    print(f"  FIX: {input_clip.name} video={vid_dur:.3f}s, audio={aud_dur:.3f}s "
          f"(diff={abs(vid_dur - aud_dur):.3f}s)")

    # Trim audio to match video duration. Copy video stream.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_clip),
        "-c:v", "copy",
        "-af", f"atrim=end={vid_dur:.6f},asetpts=PTS-STARTPTS",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "0:a:0",
        str(output_clip),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {input_clip.name}: {result.stderr[:200]}")
        return False

    # Verify result
    out_data = _ffprobe_json(str(output_clip))
    out_vid = _get_stream_duration(out_data, "video")
    out_aud = _get_stream_duration(out_data, "audio")
    diff = abs(out_vid - out_aud)

    if diff < 1.0 / 24.0:
        print(f"  OK: {output_clip.name} video={out_vid:.3f}s, "
              f"audio={out_aud:.3f}s (diff={diff:.6f}s)")
        return True
    else:
        print(f"  WARN: {output_clip.name} still has mismatch "
              f"video={out_vid:.3f}s, audio={out_aud:.3f}s (diff={diff:.3f}s)")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix audio sync in trimmed clips")
    parser.add_argument("--input", required=True, help="Input clips directory")
    parser.add_argument("--output", required=True, help="Output clips directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(input_dir.glob("clip_*_up.mp4"))
    if not clips:
        print(f"No clips found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(clips)} clips to fix")

    success_count = 0
    for clip in clips:
        out_name = clip.name.replace("_up.mp4", "_up_fixed.mp4")
        out_path = output_dir / out_name
        if fix_audio_sync(clip, out_path):
            success_count += 1

    print(f"\nFixed {success_count}/{len(clips)} clips")
    if success_count != len(clips):
        sys.exit(1)


if __name__ == "__main__":
    main()
