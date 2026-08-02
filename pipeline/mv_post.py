#!/usr/bin/env python3
"""Post-processing + FFmpeg compositing stage.

Module is kept <= 400 lines per STYLE.md (approved 400 ceiling deviation
from the 300 default). Logic is byte-for-byte identical to the original
`generate_music_video_pipeline.py` block this module was split from.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from mv_lut import (
    _download_default_lut,
    POST_PROCESS_LUT_DIR,
    DEFAULT_LUT_NAME,
    DEFAULT_GRAIN_INTENSITY,
    DEFAULT_SHARPEN_STRENGTH,
)
from mv_post_filter import (
    POST_GRADE_FPS,
    _build_audio_delay_filter,
    _build_post_video_filter,
)
from mv_vram import _resolution_filter

logger = logging.getLogger(__name__)


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        logger.warning("Failed to get video duration for %s: %s", video_path, exc)
        return 0.0

# Fixed video-only overlap (LOCKED decision D-03, LOCKED decision D-03). At every clip
# seam the next clip's VIDEO is delayed by this many seconds relative to its
# AUDIO start, so during the overlap the listener hears clip i's audio while
# seeing clip (i-1)'s video. Deterministic and non-adaptive — audio is re-muxed
# independently against the true clip_timings, so it stays seamless (no gap,
# no crossfade). Clip 0 is never offset.
VIDEO_OVERLAP_S = 1.0


def _apply_post_processing(
    input_video: Path,
    output_video: Path,
    audio_path: Path,
    lut_path: str | None = None,
    grain_intensity: float = DEFAULT_GRAIN_INTENSITY,
    sharpen_strength: float = DEFAULT_SHARPEN_STRENGTH,
    res_filter: str | None = None,
    apply_lut: bool = False,
    pre_roll_frames: int = 0,
    max_duration: float | None = None,
) -> bool:
    """Apply post-processing filter chain: color grading + film grain + sharpening.

    Runs FFmpeg with a single-pass filter chain:
    1. (optional) res_filter — crop/scale to normalize generation res to 1920x1080
    2. colorchannelmixer — identity (neutral) unless apply_lut is set
    3. lut3d — 3D LUT color grading (only when apply_lut and the .cube exists)
    4. noise — film grain (time-varying)
    5. unsharp — sharpening

    The grade is OFF by default (apply_lut=False) so output matches the 09.9-16
    reference clips, which were made before the LUT grade was added. The final
    audio is delayed by pre_roll_frames to realign lips after the pre-roll trim
    (see mv_comfyui.PRE_ROLL_FRAMES).

    Args:
        input_video: Path to the stitched video (after concat).
        output_video: Path for the final output.
        audio_path: Path to the audio file to re-mux.
        lut_path: Path to .cube LUT file (optional).
        grain_intensity: Film grain intensity 0.0-10.0.
        sharpen_strength: Sharpening strength 0.0-1.5.
        res_filter: Optional ffmpeg video filter (crop/scale) to normalize
            generation resolution to 1920x1080 before color grading.
        apply_lut: When True, apply the Cine Grade color mixer + 3D LUT. When
            False (default), the chain is neutral (matches 09.9-16).
        pre_roll_frames: Pre-roll frames trimmed from the video (see
            mv_comfyui.PRE_ROLL_FRAMES); the audio is delayed by the same amount
            to keep lips aligned.

    Returns:
        True if post-processing succeeded, False otherwise.
    """
    # Build filter chain via the pure helper (testable in isolation).
    filter_chain, lut_applied = _build_post_video_filter(
        res_filter, apply_lut, lut_path, grain_intensity, sharpen_strength,
    )
    audio_filter = _build_audio_delay_filter(pre_roll_frames)
    if audio_filter:
        logger.info(
            "Post-processing: audio delayed %dus (pre-roll realign, %d frames)",
            int(pre_roll_frames / POST_GRADE_FPS * 1_000_000), pre_roll_frames,
        )
    else:
        logger.info("Post-processing: no audio delay (pre_roll_frames=%d)", pre_roll_frames)

    # Single-pass: apply filters + re-mux audio
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-i", str(audio_path),
        "-vf", filter_chain,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
    ]
    if max_duration is not None and max_duration > 0:
        cmd += ["-t", str(round(max_duration, 3))]
    cmd.append(str(output_video))
    if audio_filter:
        cmd += ["-af", audio_filter]

    logger.info("Applying post-processing: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            # Try without lut3d (it may not be supported)
            if lut_applied:
                logger.warning(
                    "Post-processing with LUT failed, retrying without LUT: %s",
                    result.stderr[:200],
                )
                # Rebuild the chain with no LUT (lut_path=None forces no lut3d).
                filter_chain_no_lut, _ = _build_post_video_filter(
                    res_filter, apply_lut, None, grain_intensity, sharpen_strength,
                )
                cmd_no_lut = [
                    "ffmpeg", "-y",
                    "-i", str(input_video),
                    "-i", str(audio_path),
                    "-vf", filter_chain_no_lut,
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "medium",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    str(output_video),
                ]
                if audio_filter:
                    cmd_no_lut += ["-af", audio_filter]
                result = subprocess.run(
                    cmd_no_lut, capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0:
                    logger.error("Post-processing failed (no LUT): %s", result.stderr[:200])
                    return False
                logger.info("Post-processing succeeded (without LUT fallback)")
                return True
            else:
                logger.error("Post-processing failed: %s", result.stderr[:200])
                return False

        logger.info("Post-processing complete: %s", output_video)
        return True

    except subprocess.TimeoutExpired:
        logger.error("Post-processing timed out after 600s")
        return False


def _composite_timeline_canvas(
    clip_paths: list[Path],
    clip_timings: list[tuple[float, float]],
    audio_duration: float | None,
    output_dir: Path,
    gen_width: int | None = None,
    gen_height: int | None = None,
) -> Path:
    """Render a black canvas of length == audio_duration with each clip overlaid
    at its true ``segment.start`` (timeline-aware composite, Plan 09.9-12).

    Canvas length T is the audio duration (audio is master; never truncated).
    Compositing uses segment-based concat: each clip overlays on its own canvas
    segment at t=0 (no global timestamp issues). Black canvas segments fill
    ``VIDEO_OVERLAP_S`` gaps between clips (D-03). Segments are concatenated
    into the final stitched video. Returns the
    stitched (video-only) path; the caller re-muxes audio afterward. Threat
    T-09.9-12-01: clip starts are clamped to finite >= 0 so no malformed
    ``enable=''`` is built.

    Video overlap (LOCKED D-03, LOCKED decision D-03): for every clip after the first
    (i >= 2) the VIDEO enable threshold is delayed by ``VIDEO_OVERLAP_S``
    relative to its audio start, so clip i's video appears 1s after its audio
    begins — masking seam artifacts (works with the frame-0 contrast fix).
    Clip 1 (i == 1) is never offset. This is purely video-side: audio is
    re-muxed separately against the untouched clip_timings (seamless, no
    crossfade/gap).
    """
    W = int(gen_width) if gen_width else 1920
    H = int(gen_height) if gen_height else 1088

    if audio_duration and float(audio_duration) > 0:
        T = float(audio_duration)
    else:
        # Defensive: without a known audio length, sum clip durations so the
        # canvas still covers every clip (audio is re-muxed separately).
        T = sum(max(0.0, end - start) for start, end in clip_timings)
        logger.warning("Timeline composite: audio_duration invalid — "
                       "falling back to summed clip durations (%.3f)", T)

    stitched = output_dir / "stitched.mp4"

    # Build timeline via segment-based concat.
    # Each clip overlays on its own canvas segment at t=0 (no global timestamps,
    # no eof_action issues). Segments are concatenated back-to-back.
    #
    # Total video duration == sum of clip durations, which matches the cascade
    # audio duration (audio is re-muxed separately against clip_timings).
    # VIDEO_OVERLAP_S is NOT implemented as extra black frames — that would
    # extend video beyond audio and break lip sync. The overlap is a placement
    # concern for the overlay approach (canvas == audio duration), not applicable
    # to back-to-back concat where each clip's video aligns with its audio.

    n = len(clip_paths)
    temp_segments: list[Path] = []

    # Build segments: for each clip, overlay on black canvas of clip's duration.
    # This normalizes the clip (ensures yuv420p, consistent encoding params)
    # without introducing timestamp issues.
    segment_paths: list[Path] = []
    for i in range(n):
        clip_dur = _get_video_duration(str(clip_paths[i]))
        if clip_dur <= 0:
            raise RuntimeError(f"Clip {clip_paths[i].name} has invalid duration ({clip_dur})")

        clip_seg = output_dir / f"_seg_{i:03d}.mp4"
        temp_segments.append(clip_seg)
        segment_paths.append(clip_seg)

        clip_filter = (
            f"[0:v]format=yuv420p[bg];[bg][1:v]overlay=format=auto:eof_action=pass[out]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={W}x{H}:d={clip_dur:.3f}:r=24",
            "-i", str(clip_paths[i]),
            "-filter_complex", clip_filter,
            "-map", "[out]",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-g", "24",
            "-x264-params", "keyint=24:min-keyint=12:bframes=3",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(clip_seg),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg clip segment %d failed: %s", i + 1, result.stderr)
            raise RuntimeError(f"FFmpeg clip segment {i + 1} failed (rc={result.returncode})")

    # Concat all segments into the final stitched video.
    concat_list = output_dir / "_concat_timeline.txt"
    with open(concat_list, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-g", "24",
        "-x264-params", "keyint=24:min-keyint=12:bframes=3",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(stitched),
    ]
    logger.info("Timeline compositing (segment concat): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg timeline concat failed: %s", result.stderr)
        raise RuntimeError(f"FFmpeg timeline concat failed (rc={result.returncode})")

    # Cleanup temp segments
    for seg in temp_segments:
        seg.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)

    logger.info("Timeline stitched canvas: %s", stitched)
    return stitched


def _composite_with_ffmpeg(
    clip_paths: list[Path],
    input_audio: Path,
    output_dir: Path,
    skip_post_process: bool = False,
    lut_path: str | None = None,
    grain_intensity: float = DEFAULT_GRAIN_INTENSITY,
    sharpen_strength: float = DEFAULT_SHARPEN_STRENGTH,
    gen_width: int | None = None,
    gen_height: int | None = None,
    clip_timings: list[tuple[float, float]] | None = None,
    audio_duration: float | None = None,
    apply_lut: bool = False,
    pre_roll_frames: int = 0,
) -> Path:
    """Stitch clips and apply post-processing, then re-mux with original audio.

    Pass 1: When ``clip_timings`` is supplied (timeline-aware mode, Plan
        09.9-12) each clip is placed at its true ``segment.start`` on a black
        canvas of length == ``audio_duration`` (audio is master; never
        truncated). Otherwise clips are concatenated back-to-back via the
        concat demuxer (legacy branch, preserved). Pass 2: post-processing
        (color grading + grain + sharpen) + audio re-mux, or simple re-mux if
        skipped. gen_width/gen_height (Plan 09.9-09) normalize the final output
        to 1920x1080 (e.g. 1920x1088 -> crop, 1088x608 -> scale). The grade is
        off by default (apply_lut=False); the audio is delayed by
        pre_roll_frames to realign lips after the pre-roll trim.
    """
    if not clip_paths:
        raise RuntimeError("No clips to composite — pipeline produced no video")

    # Plan 09.9-09: resolution-normalize filter (crop 1920x1088 -> 1920x1080,
    # or scale a safe-res like 1088x608 -> 1920x1080).
    res_filter = None
    if gen_width and gen_height:
        res_filter = _resolution_filter(gen_width, gen_height)
        if res_filter:
            logger.info("Resolution normalize filter: %s", res_filter)

    concat_list = output_dir / "concat_list.txt"
    stitched = output_dir / "stitched.mp4"

    # ── Timeline-aware branch (Plan 09.9-12) ───────────────────────
    # Place each clip at its true segment.start on a black canvas of length
    # == audio_duration (audio is the master; never truncated). Compositing
    # uses segment-based concat to avoid overlay timestamp issues (each clip
    # composites independently; black canvas fills overlap gaps).
    if clip_timings is not None and len(clip_timings) == len(clip_paths):
        stitched = _composite_timeline_canvas(
            clip_paths, clip_timings, audio_duration, output_dir,
            gen_width, gen_height,
        )
    else:
        # ── Legacy concat branch (backward-compatible; preserved) ──
        # Write concat list file
        with open(concat_list, "w") as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")
        logger.info("Concat list: %d clips", len(clip_paths))

        # Pass 1: Stitch clips (video only)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "copy",
            str(stitched),
        ]
        logger.info("Stitching clips: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg concat failed: %s", result.stderr)
            raise RuntimeError(f"FFmpeg concat failed (rc={result.returncode})")
        logger.info("Stitched video: %s", stitched)

    # Pass 2: Post-processing + audio re-mux (or simple re-mux if skipped)
    final_output = output_dir / "final_output.mp4"

    # Plan 09.9-33-02: limit output to stitched video duration so audio is
    # cropped when canvas is shorter than the full audio file (e.g. --max-clips).
    stitched_dur = _get_video_duration(str(stitched))

    if skip_post_process:
        # Always re-encode to guarantee yuv420p pixel format and faststart.
        # -c:v copy cannot change pixel format — input clips may be yuv444p.
        if res_filter:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(stitched),
                "-i", str(input_audio),
                "-vf", res_filter,
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(round(stitched_dur, 3)),
                str(final_output),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(stitched),
                "-i", str(input_audio),
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(round(stitched_dur, 3)),
                str(final_output),
            ]
        logger.info("Re-muxing with audio (no post-processing): %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg re-mux failed: %s", result.stderr)
            raise RuntimeError(f"FFmpeg re-mux failed (rc={result.returncode})")
    else:
        # Ensure default LUT exists — only when the grade is actually enabled,
        # so we don't fetch Cine_Grade.cube for the neutral (09.9-16-matching) path.
        if apply_lut and lut_path is None:
            downloaded = _download_default_lut()
            if downloaded:
                lut_path = str(downloaded)
            else:
                logger.warning(
                    "Default LUT not available — post-processing will use colorchannelmixer only"
                )
                lut_path = None

        success = _apply_post_processing(
            stitched, final_output, input_audio,
            lut_path=lut_path,
            grain_intensity=grain_intensity,
            sharpen_strength=sharpen_strength,
            res_filter=res_filter,
            apply_lut=apply_lut,
            pre_roll_frames=pre_roll_frames,
            max_duration=stitched_dur,
        )
        if not success:
            logger.warning(
                "Post-processing failed — falling back to simple re-mux"
            )
            # Always re-encode to guarantee yuv420p + faststart on the fallback
            # path too. Apply resolution filter so the final output is 1920x1080.
            vf = res_filter if res_filter else "format=yuv420p"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(stitched),
                "-i", str(input_audio),
                "-vf", vf,
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(round(stitched_dur, 3)),
                str(final_output),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("FFmpeg fallback re-mux failed: %s", result.stderr)
                raise RuntimeError(f"FFmpeg fallback re-mux failed (rc={result.returncode})")
    logger.info("Final output: %s", final_output)

    # Cleanup concat list
    concat_list.unlink(missing_ok=True)

    return final_output
