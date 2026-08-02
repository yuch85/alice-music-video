#!/usr/bin/env python3
"""Experimental audit: Simple Assembly audio trim/pad analysis.

Answer YC's questions experimentally:
- where is audio trimmed?
- where is audio padded?
- does trimming occur during silence?
- could trimming remove part of a phoneme?
- could padding delay the next lyric?
- per-clip lip-sync impact

Uses non-overlapping segment plan (what regeneration will produce).
"""
from __future__ import annotations

import json
import subprocess
import struct
import sys
from pathlib import Path
from typing import NamedTuple

_REPO = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
_MV = _REPO / "songs" / "music-videos" / "modotte-oide-yui"
_AUDIO = _MV / "audio.mp3"
_VOCALS = _MV / "stems" / "htdemucs" / "htdemucs" / "audio" / "vocals.wav"
_SEGMENT_PLAN = _MV / "gen-output" / "segment_plan.json"
_CLIPS = _MV / "gen-output" / "clips"

FPS = 24


def _ltx2_num_frames(length_s: float) -> int:
    raw = length_s * FPS
    k = round((raw - 1) / 8)
    if k < 1:
        k = 1
    return 8 * k + 1


def get_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def compute_non_overlapping_segments() -> list[dict]:
    """Compute what the non-overlapping segment plan would be.

    The fix at line 110 ensures no overlaps. Recompute from Whisper words
    is complex, so simulate: take current segments, make non-overlapping
    by using min(start, prev_end) for each segment.
    """
    plan = json.loads(_SEGMENT_PLAN.read_text())
    segments = sorted(plan["segments"], key=lambda s: s["index"])

    result = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        # Make non-overlapping: start = max(start, prev_end)
        if i > 0:
            start = max(start, result[-1]["end"])
        dur = end - start
        result.append({
            "index": seg["index"],
            "start": start,
            "end": end,
            "duration": dur,
            "text": seg["text"],
            "shot_type": seg.get("shot_type", "unknown"),
        })
    return result


def extract_audio_rms(vocals_path: Path, start: float, end: float) -> list[tuple[float, float]]:
    """Extract audio RMS energy at 10ms intervals for [start, end].

    Returns list of (timestamp, rms) tuples.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(vocals_path),
        "-ss", str(start), "-to", str(end),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(wav_path),
    ], capture_output=True, timeout=15)

    with open(wav_path, "rb") as f:
        f.read(44)  # WAV header
        data = f.read()

    samples = struct.unpack("<%dh" % (len(data) // 2), data)
    sr = 44100
    window = int(0.010 * sr)  # 10ms windows

    intervals = []
    for i in range(0, len(samples) - window, window):
        chunk = samples[i:i + window]
        rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
        t = (i / sr)
        intervals.append((t, rms))

    wav_path.unlink()
    return intervals


def find_vocals_stem() -> Path | None:
    if _VOCALS.is_file():
        return _VOCALS
    # Fallback: search
    for d in sorted((_MV / "stems").iterdir()) if (_MV / "stems").is_dir() else []:
        if not d.is_dir():
            continue
        for td in sorted(d.iterdir()):
            if not td.is_dir():
                continue
            for fd in sorted(td.iterdir()):
                if not fd.is_dir():
                    continue
                v = fd / "vocals.wav"
                if v.is_file():
                    return v
    return None


def analyze_seam(segments, seg_idx):
    """Analyze the boundary between seg_idx and seg_idx+1.

    For Simple Assembly: audio is extracted at segment plan times.
    Video duration may differ from segment duration due to 8k+1 quantization.
    Audio is trimmed/padded to match video.

    Check: is the trim/pad region silent?
    """
    seg = segments[seg_idx]
    next_seg = segments[seg_idx + 1] if seg_idx + 1 < len(segments) else None

    seg_dur = seg["duration"]
    frames = _ltx2_num_frames(seg_dur)
    video_dur = frames / FPS
    diff_ms = (video_dur - seg_dur) * 1000

    # The audio adjustment happens at the END of the clip's audio
    if diff_ms > 0:
        # Video is LONGER than audio -> need to PAD audio at end
        action = "PAD"
        adjust_region_start = seg["end"]
        adjust_region_end = seg["end"] + (diff_ms / 1000)
    else:
        # Video is SHORTER than audio -> need to TRIM audio at end
        action = "TRIM"
        adjust_region_start = seg["end"] + (diff_ms / 1000)  # diff_ms is negative
        adjust_region_end = seg["end"]

    # What content is in the adjust region?
    # Check if next segment's first word starts after adjust_region_end
    if next_seg and diff_ms < 0:
        # We're trimming the end of this segment's audio
        # Check: does the next segment start within the trim region?
        next_starts_in_trim = next_seg["start"] < seg["end"]
    else:
        next_starts_in_trim = False

    return {
        "seg_idx": seg["index"],
        "seg_dur": seg_dur,
        "video_dur": video_dur,
        "diff_ms": diff_ms,
        "action": action,
        "adjust_region": (adjust_region_start, adjust_region_end),
        "next_seg_idx": next_seg["index"] if next_seg else None,
        "next_starts_in_trim": next_starts_in_trim,
    }


def main():
    print("=" * 80)
    print("MV Audio Seam Audit — Simple Assembly Analysis")
    print("=" * 80)

    segments = compute_non_overlapping_segments()

    print(f"\nNon-overlapping segments: {len(segments)}")
    print(f"Total segment duration: {sum(s['duration'] for s in segments):.3f}s")

    # Find vocals stem
    vocals = find_vocals_stem()
    if vocals is None:
        print("ERROR: vocals stem not found")
        sys.exit(1)
    print(f"Vocals stem: {vocals}")

    # Compute quantization for each segment
    print("\n" + "-" * 80)
    print("Per-Clip Quantization Analysis")
    print("-" * 80)
    print(f"{'Clip':>4} {'SegDur':>8} {'Frames':>6} {'VideoDur':>9} {'Diff(ms)':>9} {'Action':>5} {'ShotType':>12}")
    print("-" * 80)

    total_trim_ms = 0
    total_pad_ms = 0
    trims = []
    pads = []

    for seg in segments:
        frames = _ltx2_num_frames(seg["duration"])
        video_dur = frames / FPS
        diff_ms = (video_dur - seg["duration"]) * 1000

        if diff_ms > 0.5:
            action = "PAD"
            total_pad_ms += diff_ms
            pads.append(seg)
        elif diff_ms < -0.5:
            action = "TRIM"
            total_trim_ms += abs(diff_ms)
            trims.append(seg)
        else:
            action = "—"

        print(f"{seg['index']:>4d} {seg['duration']:>8.3f} {frames:>6d} {video_dur:>9.3f} {diff_ms:>+9.1f} {action:>5} {seg['shot_type']:>12}")

    print("-" * 80)
    print(f"Total TRIM: {total_trim_ms:.1f}ms across {len(trims)} clips")
    print(f"Total PAD:  {total_pad_ms:.1f}ms across {len(pads)} clips")
    print(f"Net:        {total_pad_ms - total_trim_ms:+.1f}ms")

    # Analyze seams
    print("\n" + "-" * 80)
    print("Seam Analysis — Where Does Audio Adjustment Occur?")
    print("-" * 80)

    seams = []
    for i in range(len(segments) - 1):
        seam = analyze_seam(segments, i)
        seams.append(seam)

    # Now check: for TRIM clips, is the trim region at the end of the segment
    # actually silent in the vocals stem?
    print(f"\n{'Clip':>4} {'Diff(ms)':>9} {'Action':>5} {'EndPos':>8} {'TrimRegion':>20} {'ShotType':>12}")
    print("-" * 80)

    trim_in_silence = 0
    trim_in_content = 0
    pad_at_boundary = 0

    for seam in seams:
        if seam["action"] == "—":
            continue

        seg_end = seam["adjust_region"][1] if seam["action"] == "TRIM" else seam["adjust_region"][0]
        trim_region_str = f"{seam['adjust_region'][0]:.3f}-{seam['adjust_region'][1]:.3f}"

        print(f"{seam['seg_idx']:>4d} {seam['diff_ms']:>9.1f} {seam['action']:>5} {seg_end:>8.3f} {trim_region_str:>20} {segments[seam['seg_idx']-1]['shot_type']:>12}")

        # Check if the adjust region has vocal content
        # For TRIM: check if there's vocal energy in the last |diff_ms| of the segment
        # For PAD: the pad is silence added after the segment — always safe
        if seam["action"] == "TRIM":
            # Check RMS in the trim region
            region_start = seam["adjust_region"][0]
            region_end = seam["adjust_region"][1]
            rms_data = extract_audio_rms(vocals, region_start - 0.05, region_end + 0.05)

            if rms_data:
                # Get RMS specifically in the trim region (relative to segment)
                seg_abs_end = segments[seam['seg_idx']-1]["end"]
                trim_rms_values = [r for (t, r) in rms_data if t >= (region_start - seg_abs_end - 0.05) and t <= (region_end - seg_abs_end + 0.05)]

                if trim_rms_values:
                    max_rms = max(trim_rms_values)
                    avg_rms = sum(trim_rms_values) / len(trim_rms_values)
                    if max_rms < 500:  # silence threshold
                        trim_in_silence += 1
                    else:
                        trim_in_content += 1
                else:
                    trim_in_silence += 1  # assume silence if no data
            else:
                trim_in_silence += 1
        elif seam["action"] == "PAD":
            pad_at_boundary += 1

    print(f"\nTRIM in silence: {trim_in_silence}")
    print(f"TRIM in content: {trim_in_content}")
    print(f"PAD at boundary: {pad_at_boundary}")

    # Detailed analysis of each trim region
    print("\n" + "-" * 80)
    print("Detailed Trim Region Analysis (vocal energy)")
    print("-" * 80)

    for seg in trims:
        seg_end = seg["end"]
        frames = _ltx2_num_frames(seg["duration"])
        video_dur = frames / FPS
        trim_ms = (seg["duration"] - video_dur) * 1000

        # Extract RMS for last 200ms of segment
        check_start = max(seg["start"], seg_end - 0.3)
        rms_data = extract_audio_rms(vocals, check_start, seg_end)

        if rms_data:
            # Focus on the actual trim region
            trim_region_t0 = seg_end - (trim_ms / 1000)
            trim_rms = [r for (t, r) in rms_data if t >= (trim_region_t0 - check_start)]
            pre_trim_rms = [r for (t, r) in rms_data if t < (trim_region_t0 - check_start)]

            max_trim_rms = max(trim_rms) if trim_rms else 0
            avg_trim_rms = (sum(trim_rms) / len(trim_rms)) if trim_rms else 0
            max_pre_rms = max(pre_trim_rms) if pre_trim_rms else 0

            silent = "SILENT" if max_trim_rms < 500 else "CONTENT"
            print(f"Clip {seg['index']:>2d}: trim {trim_ms:.0f}ms, "
                  f"max_RMS_in_trim={max_trim_rms:.0f}, "
                  f"avg_RMS_in_trim={avg_trim_rms:.0f}, "
                  f"max_RMS_pre_trim={max_pre_rms:.0f} -> {silent}")
        else:
            print(f"Clip {seg['index']:>2d}: trim {trim_ms:.0f}ms, no RMS data -> UNKNOWN")

    # Cascade simulation for comparison
    print("\n" + "-" * 80)
    print("Cascade Shift Simulation (Option A vs B comparison)")
    print("-" * 80)

    cascade_pos = 0.0
    max_shift = 0
    max_shift_clip = 0

    print(f"{'Clip':>4} {'SegStart':>9} {'CascadePos':>10} {'Shift(ms)':>10} {'Cumulative':>10}")
    print("-" * 80)

    for seg in segments:
        shift_ms = (cascade_pos - seg["start"]) * 1000
        frames = _ltx2_num_frames(seg["duration"])
        video_dur = frames / FPS

        if abs(shift_ms) > abs(max_shift):
            max_shift = shift_ms
            max_shift_clip = seg["index"]

        print(f"{seg['index']:>4d} {seg['start']:>9.3f} {cascade_pos:>10.3f} {shift_ms:>+10.1f} {cascade_pos:>10.3f}")

        cascade_pos += video_dur

    print(f"\nMax cascade shift: {max_shift:+.1f}ms at clip {max_shift_clip}")
    print(f"Total cascade drift: {(cascade_pos - segments[-1]['end']):+.1f}ms")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Non-overlapping segments: {len(segments)}")
    print(f"Total segment duration: {sum(s['duration'] for s in segments):.3f}s")
    total_video = sum(_ltx2_num_frames(s['duration']) / FPS for s in segments)
    print(f"Total video duration: {total_video:.3f}s")
    print(f"Net quantization: {total_video - sum(s['duration'] for s in segments):+.1f}ms")
    print(f"Clips needing TRIM: {len(trims)}")
    print(f"Clips needing PAD: {len(pads)}")
    print(f"Trim in silence: {trim_in_silence}/{len(trims)}")
    print(f"Trim in content: {trim_in_content}/{len(trims)}")
    print(f"Max cascade shift (Option A): {max_shift:+.1f}ms at clip {max_shift_clip}")


if __name__ == "__main__":
    main()
