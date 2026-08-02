#!/usr/bin/env python3
"""Audio stage: Demucs stem separation, Whisper transcription, audio crop/trim.

Module is kept <= 400 lines per STYLE.md (approved 400 ceiling deviation
from the 300 default). Logic is byte-for-byte identical to the original
`generate_music_video_pipeline.py` block this module was split from.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

from mv_segment import ClipSegment, WordSegment
from mv_comfyui import (
    COMFYUI_OUTPUT_DIR,
    PRE_ROLL_FRAMES,
    TAIL_LOSS_FRAMES,
)

logger = logging.getLogger(__name__)

DEMUCS_BIN = os.environ.get("DEMUCS_BIN", "/path/to/demucs")
WHISPER_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"


def _run_demucs_separation(
    input_audio: Path, output_dir: Path, two_stems: str | None = None
) -> dict[str, Path]:
    """Run Demucs to separate stems.

    Returns dict mapping stem name to file path.
    With --two-stems vocals: {'vocals': Path, 'no_vocals': Path}
    Without: {'vocals': Path, 'drums': Path, 'bass': Path, 'other': Path}
    """
    htdemucs_dir = output_dir / "htdemucs"
    htdemucs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [DEMUCS_BIN, "-o", str(htdemucs_dir), str(input_audio)]
    if two_stems:
        cmd.extend(["--two-stems", two_stems])

    logger.info("Running Demucs: %s", " ".join(cmd))
    # Clear LD_LIBRARY_PATH for Demucs — it uses its own venv's CUDA libs.
    # Our LD_LIBRARY_PATH points to ComfyUI's CUDA libs for onnxruntime-gpu.
    _demucs_env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=_demucs_env)
    if result.returncode != 0:
        logger.error("Demucs failed: %s", result.stderr)
        raise RuntimeError(f"Demucs failed (rc={result.returncode}): {result.stderr}")

    # Parse output — Demucs creates subdirectories per model
    # Structure: htdemucs/<model_name>/<track>/<stem>.wav
    stems: dict[str, Path] = {}
    model_dirs = sorted(htdemucs_dir.iterdir())
    if not model_dirs:
        raise RuntimeError(f"No output from Demucs in {htdemucs_dir}")

    model_dir = model_dirs[0]  # Usually 'htdemucs' or 'htdemucs_6s'
    track_dirs = sorted(model_dir.iterdir())
    for track_dir in track_dirs:
        # Stem files are <stem>.wav directly in the track directory
        for wav_file in sorted(track_dir.glob("*.wav")):
            stem_name = wav_file.stem  # filename without .wav
            stems[stem_name] = wav_file

    logger.info("Demucs stems: %s", list(stems.keys()))
    return stems


def _transcribe_with_whisper(audio_path: Path) -> list[WordSegment]:
    """Transcribe audio using faster-whisper with word-level timestamps.

    Returns list of WordSegment (text, start, end) sorted by start time.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper not installed. Install with: "
            "pip install faster-whisper"
        ) from e

    logger.info("Loading Whisper model %s...", WHISPER_MODEL_ID)
    model = WhisperModel(
        WHISPER_MODEL_ID, device="cuda", compute_type="float16"
    )
    logger.info("Whisper model loaded.")

    segments, info = model.transcribe(
        str(audio_path), word_timestamps=True
    )

    words: list[WordSegment] = []
    for seg in segments:
        for word in seg.words:
            if word.word.strip():
                words.append(WordSegment(
                    text=word.word.strip(),
                    start=word.start,
                    end=word.end,
                ))

    words.sort(key=lambda w: w.start)
    logger.info("Transcribed %d words (%.1fs audio, lang=%s)", len(words), info.duration, info.language)

    # Release the in-process Whisper CUDA model so it does not hold ~2-3 GiB
    # on device 0 across the rest of the run (it competes with ComfyUI's
    # WanVideo graph on the shared GPU; 09.9-25-05 clip-2 OOM investigation).
    del model
    gc.collect()
    if _has_torch:
        torch.cuda.empty_cache()
    return words


def _crop_audio_segment(
    audio_file: Path,
    start: float,
    end: float,
    output_dir: Path,
    index: int,
) -> Path:
    """Crop a segment of audio from the vocals stem for Audio VAE conditioning.

    Uses ffmpeg to extract [start, end] range, outputs 44100Hz PCM WAV.
    The output is placed in ComfyUI input/ directory for LoadAudio node.

    Args:
        audio_file: Path to the vocals stem WAV file.
        start: Start time in seconds.
        end: End time in seconds.
        output_dir: Output directory for the pipeline.
        index: 1-based segment index for filename uniqueness.

    Returns:
        Path to the cropped audio file in ComfyUI input/ directory.
    """
    # Create output subdirectory for audio segments
    audio_seg_dir = output_dir / "audio_segments"
    audio_seg_dir.mkdir(parents=True, exist_ok=True)

    # Local output path
    local_dest = audio_seg_dir / f"audio_seg_{index:03d}.wav"

    # Crop with ffmpeg — 44100Hz PCM WAV
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_file),
        "-ss", str(start),
        "-to", str(end),
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        str(local_dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Audio crop failed for segment %d: %s", index, result.stderr)
        return local_dest  # Return path anyway, caller handles missing files

    # Copy to ComfyUI input/ directory for LoadAudio node
    comfyui_input = Path(COMFYUI_OUTPUT_DIR) / "input"
    comfyui_input.mkdir(parents=True, exist_ok=True)
    comfyui_dest = comfyui_input / f"audio_seg_{index:03d}.wav"
    shutil.copy2(local_dest, comfyui_dest)

    logger.info("Audio segment %d cropped: %s -> %s", index, local_dest, comfyui_dest)
    return comfyui_dest


def _trim_padding_frames(
    input_clip: Path,
    output_clip: Path,
) -> bool:
    """Trim pre-roll and tail-loss padding frames from a generated clip.

    Trims BOTH video and audio symmetrically so the output clip has matching
    stream durations (format_duration == video_duration == audio_duration).
    The trimmed clip is the "normalized" artifact used for all downstream
    timing, cascade positions, and compositing.

    Video is trimmed using frame indices (start_frame / end_frame) via the
    ``trim`` video filter. Audio is trimmed using the equivalent time
    boundaries via the ``atrim`` audio filter. Both streams are re-encoded
    (libx264 for video, aac for audio) to produce a self-consistent clip.

    The raw LTX output is an intermediate artifact; pre-roll and tail-loss
    frames are known generation artifacts removed during normalization. All
    timing operations in the pipeline use normalized clip durations, not
    raw generation durations.

    Args:
        input_clip: Path to the generated clip with padding.
        output_clip: Path for the trimmed (normalized) output clip.

    Returns:
        True if trim succeeded, False otherwise.
    """
    _TRIM_FPS = 24.0  # LTX-2 outputs 24fps

    if PRE_ROLL_FRAMES == 0 and TAIL_LOSS_FRAMES == 0:
        return True  # No padding to trim

    # Probe total frame count (duration * fps; LTX-2 outputs 24fps).
    total_frames = 0
    try:
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(input_clip)],
            capture_output=True, text=True, timeout=30,
        )
        if dur.returncode == 0:
            total_frames = int(round(float(dur.stdout.strip()) * _TRIM_FPS))
    except (subprocess.TimeoutExpired, ValueError, OSError):
        total_frames = 0

    start_frame = PRE_ROLL_FRAMES
    end_frame = (
        total_frames - TAIL_LOSS_FRAMES
        if total_frames > TAIL_LOSS_FRAMES
        else 0
    )

    # Track whether we have a valid end boundary (both start AND end trim).
    has_end_boundary = end_frame > start_frame

    if has_end_boundary:
        vf = (
            f"trim=start_frame={start_frame}:"
            f"end_frame={end_frame},setpts=PTS-STARTPTS"
        )
    elif start_frame > 0:
        # Not enough frames to trim both ends — fall back to pre-roll only.
        vf = f"trim=start_frame={start_frame},setpts=PTS-STARTPTS"
    else:
        return True  # nothing to trim

    # Compute matching audio trim boundaries in seconds.
    # The audio trim mirrors the video trim: skip the first N frames worth
    # of audio, and cut before the last M frames worth of audio.
    audio_start_s = start_frame / _TRIM_FPS
    if has_end_boundary:
        audio_end_s = end_frame / _TRIM_FPS
        af = (
            f"atrim=start={audio_start_s:.6f}:"
            f"end={audio_end_s:.6f},asetpts=PTS-STARTPTS"
        )
    else:
        # Pre-roll-only fallback: trim from start, no end boundary.
        af = f"atrim=start={audio_start_s:.6f},asetpts=PTS-STARTPTS"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_clip),
        "-vf", vf,
        "-af", af,
        "-c:a", "aac",
        "-vsync", "vfr",
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_clip),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("FFmpeg trim failed: %s", result.stderr)
        return False
    logger.info("Trimmed padding frames: %s -> %s", input_clip, output_clip)
    return True


def _find_existing_vocals(output_dir: Path) -> Path | None:
    """Find existing vocals stem in htdemucs/ output.

    Returns the vocals.wav path if it exists, None otherwise.
    """
    htdemucs_dir = output_dir / "htdemucs"
    if not htdemucs_dir.is_dir():
        return None
    # Demucs structure: htdemucs/<model_name>/<track>/vocals.wav
    for model_dir in sorted(htdemucs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for track_dir in sorted(model_dir.iterdir()):
            vocals = track_dir / "vocals.wav"
            if vocals.is_file():
                return vocals
    return None


def _load_existing_transcript(output_dir: Path) -> tuple[list[ClipSegment], Path] | None:
    """Load existing transcript.json and return (segments, vocals_path).

    Returns None if transcript.json doesn't exist or is invalid.
    """
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.is_file():
        return None
    try:
        data = json.loads(transcript_path.read_text())
        vocals_path = Path(data["vocals_stem"])
        if not vocals_path.is_file():
            logger.warning("Vocals stem missing: %s — skipping transcript reuse", vocals_path)
            return None
        segments = []
        for seg_data in data["segments"]:
            segments.append(ClipSegment(
                start=seg_data["start"],
                end=seg_data["end"],
                text=seg_data["text"],
                duration=seg_data["duration"],
                words=[],
            ))
        logger.info("Loaded %d segments from existing transcript.json", len(segments))
        return (segments, vocals_path)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Invalid transcript.json: %s — will regenerate", e)
        return None


def _get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe (mirrors the existing
    ffprobe duration pattern used elsewhere in this module).

    Returns 0.0 on failure or parse error so callers fall back to the last
    segment end (the composite never truncates audio regardless).
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(audio_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ffprobe duration failed: %s", result.stderr[:200])
            return 0.0
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError) as e:
        logger.warning("ffprobe duration error: %s", e)
        return 0.0
