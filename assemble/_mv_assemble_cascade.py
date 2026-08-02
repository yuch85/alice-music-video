#!/usr/bin/env python3
"""Cascade assembly script — Option B architecture.

Reads clip manifests (or computes from segment plan) and assembles the song
portion. Audio cropped at cascade position, video never trimmed (Invariant 1).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from _mv_credits import build_final_with_credits

_REPO = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
CHAIN_TOLERANCE = 1e-3

REQUIRED_MANIFEST_KEYS = [
    "clip_index", "cascade_position", "cascade_next",
    "measured_duration", "clip_path",
    "conditioning_audio_start", "conditioning_audio_end",
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


# ── Manifest loading ───────────────────────────────────────────────────────

def load_manifests(clips_dir: Path) -> list[dict[str, object]]:
    """Scan for clip manifest JSON files and validate the cascade chain.

    Validates: chain continuity, clip file existence, duration consistency.
    Raises ``FileNotFoundError`` if no manifests or clips missing.
    Raises ``ValueError`` on chain break or duration mismatch.
    """
    manifest_files = sorted(clips_dir.glob("clip_*_manifest.json"))
    if not manifest_files:
        raise FileNotFoundError(f"No clip manifests found in {clips_dir}")

    manifests: list[dict[str, object]] = []
    for mf in manifest_files:
        data = json.loads(mf.read_text())
        for key in REQUIRED_MANIFEST_KEYS:
            if key not in data:
                raise ValueError(f"{mf.name}: missing required key '{key}'")
        manifests.append(data)

    manifests.sort(key=lambda m: int(m["clip_index"]))

    for i in range(len(manifests) - 1):
        expected = manifests[i]["cascade_next"]
        actual = manifests[i + 1]["cascade_position"]
        if abs(expected - actual) > CHAIN_TOLERANCE:
            raise ValueError(
                f"Chain break at clip {manifests[i]['clip_index']} -> "
                f"{manifests[i + 1]['clip_index']}: "
                f"expected cascade_position={expected}, got {actual} "
                f"(gap={actual - expected:.3f}s)"
            )

    for m in manifests:
        clip_path = Path(m["clip_path"])
        if not clip_path.exists():
            raise FileNotFoundError(
                f"Clip file missing: {clip_path} (clip {m['clip_index']})"
            )

    for m in manifests:
        computed = m["cascade_next"] - m["cascade_position"]
        actual = m["measured_duration"]
        if abs(computed - actual) > CHAIN_TOLERANCE:
            raise ValueError(
                f"Clip {m['clip_index']}: measured_duration={actual} "
                f"!= cascade_next - cascade_position={computed:.3f}"
            )

    total_dur = sum(m["measured_duration"] for m in manifests)
    first_pos = manifests[0]["cascade_position"]
    last_next = manifests[-1]["cascade_next"]
    log.info(
        "Loaded %d manifests: %.3fs total, cascade %.3f-%.3fs",
        len(manifests), total_dur, first_pos, last_next,
    )
    return manifests


def _compute_cascade_from_plan(
    clips_dir: Path, segment_plan: Path,
) -> list[dict[str, object]]:
    """Fallback: compute cascade positions from segment plan + clip measurements."""
    plan = json.loads(segment_plan.read_text())
    segments = sorted(plan["segments"], key=lambda s: s["index"])

    manifests: list[dict[str, object]] = []
    cascade_pos = 0.0

    for seg in segments:
        idx = seg["index"]
        clip_file = clips_dir / f"clip_{idx:03d}_1080p.mp4"
        if not clip_file.exists():
            raise FileNotFoundError(f"Clip file missing: {clip_file}")

        video_dur = _get_duration(clip_file)
        cascade_next = cascade_pos + video_dur

        manifests.append({
            "clip_index": idx,
            "cascade_position": cascade_pos,
            "cascade_next": cascade_next,
            "measured_duration": video_dur,
            "clip_path": str(clip_file),
            "conditioning_audio_start": cascade_pos,
            "conditioning_audio_end": cascade_next,
        })
        cascade_pos = cascade_next

    log.info(
        "Computed cascade from segment plan: %d clips, %.3fs total",
        len(manifests), cascade_pos,
    )
    return manifests


def _write_manifests(manifests: list[dict[str, object]], clips_dir: Path) -> None:
    """Persist computed manifests as clip_NNN_manifest.json files."""
    for m in manifests:
        mf = clips_dir / f"clip_{int(m['clip_index']):03d}_manifest.json"
        mf.write_text(json.dumps(m, indent=2) + "\n")
    log.info("Wrote %d manifest files to %s", len(manifests), clips_dir)


# ── FFmpeg helpers ─────────────────────────────────────────────────────────

def mux_clip_with_audio(
    *,
    clip_path: Path,
    audio_path: Path,
    audio_start: float,
    audio_end: float,
    output_path: Path,
) -> Path:
    """Mux a clip video with an audio slice from the original track.

    Video is stream-copied (no re-encode) to preserve generated frames
    exactly (Invariant 1). Audio is re-encoded to AAC for compatibility.
    """
    duration = audio_end - audio_start
    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-ss", str(audio_start), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(duration),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("mux_clip_with_audio failed: %s", result.stderr[-500:])
        raise RuntimeError(f"mux_clip_with_audio failed (rc={result.returncode})")
    return output_path


def concat_clips(
    *,
    muxed_paths: list[Path],
    output_path: Path,
) -> tuple[Path, float]:
    """Concatenate muxed clips using the concat demuxer (stream copy).

    No video re-encode, no frame modification. Audio re-encoded to AAC.
    Returns (output_path, duration_seconds).
    """
    concat_file = output_path.with_suffix(".txt")
    concat_file.write_text(
        "\n".join(f"file '{p}'" for p in muxed_paths) + "\n"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("concat_clips failed: %s", result.stderr[-1000:])
        raise RuntimeError(f"concat_clips failed (rc={result.returncode})")

    dur = _get_duration(output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Concatenated %d clips: %.2fs, %.1f MB", len(muxed_paths), dur, size_mb)
    return output_path, dur


# ── Credits (re-export from _mv_credits) ───────────────────────────────────

def add_credits(
    *,
    concat_path: Path,
    final_path: Path,
    total_duration: float,
) -> Path:
    """Add end-credits sequence with NVENC encode. Delegates to _mv_credits."""
    return build_final_with_credits(
        concat_path=concat_path,
        final_path=final_path,
        total_duration=total_duration,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point for cascade assembly."""
    parser = argparse.ArgumentParser(description="Cascade assembly (Option B)")
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "songs" / "music-videos" / "modotte-oide-yui" / "gen-output"),
        help="Path to gen-output directory",
    )
    parser.add_argument(
        "--include-credits",
        action="store_true",
        help="Add end credits and produce final NVENC encode",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    audio_path = output_dir.parent / "audio.mp3"
    segment_plan = output_dir / "segment_plan.json"

    if not audio_path.exists():
        log.error("Audio file not found: %s", audio_path)
        return 1

    # Load or compute manifests
    try:
        manifests = load_manifests(clips_dir)
    except (FileNotFoundError, ValueError) as e:
        log.info("Manifest load failed (%s) — computing cascade from segment plan", e)
        if not segment_plan.exists():
            log.error("Segment plan not found: %s", segment_plan)
            return 1
        manifests = _compute_cascade_from_plan(clips_dir, segment_plan)
        _write_manifests(manifests, clips_dir)

    # Mux each clip with its audio slice
    muxed_paths: list[Path] = []
    for i, m in enumerate(manifests, 1):
        idx = int(m["clip_index"])
        mux_out = Path(f"/tmp/mv_cascade_mux_{idx:03d}.mp4")
        log.info("[%d/%d] Muxing clip %03d (%.3fs)", i, len(manifests), idx,
                 m["measured_duration"])
        mux_clip_with_audio(
            clip_path=Path(m["clip_path"]),
            audio_path=audio_path,
            audio_start=float(m["cascade_position"]),
            audio_end=float(m["cascade_next"]),
            output_path=mux_out,
        )
        muxed_paths.append(mux_out)

    # Concatenate
    concat_path = Path("/tmp/mv_cascade_concat.mp4")
    log.info("Concatenating %d muxed clips...", len(muxed_paths))
    _, concat_dur = concat_clips(muxed_paths=muxed_paths, output_path=concat_path)
    for p in muxed_paths:
        p.unlink(missing_ok=True)

    if args.include_credits:
        final_path = output_dir.parent / "final" / "modotte-oide-yui-cascade.mp4"
        add_credits(concat_path=concat_path, final_path=final_path,
                    total_duration=concat_dur)
        concat_path.unlink(missing_ok=True)

    log.info("Assembly complete: %.2fs, %d clips", concat_dur, len(manifests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
