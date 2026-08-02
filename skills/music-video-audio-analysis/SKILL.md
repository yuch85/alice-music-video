---
name: music-video-audio-analysis
description: Run Demucs stem separation and Whisper transcription — prerequisite for beat planning
allowed-tools: [Read, Write, Bash]
---

# Audio Analysis Skill (Stage 4)

## Prerequisites

- FSM state must be `AUDIO_ANALYSIS`.
- `continuity_bible.md` exists and is `APPROVED`.
- Source audio file `audio.mp3` exists in the project root.

## Purpose

Extract the vocal and instrumental stems via Demucs, then transcribe the vocal stem with Whisper to produce word-level timestamped lyrics. These artifacts are the **input** for the Beat Sheet stage — without them, segment boundaries, segment types, and b-roll placement cannot be meaningfully planned.

## Process

### 1. Demucs Stem Separation

Run Demucs on the source audio to separate vocal and instrumental stems:

```bash
cd ~/alice/scripts
uv run python -c "
from mv_audio import run_demucs_separation
run_demucs_separation(
    audio_path='<project_dir>/audio.mp3',
    output_dir='<project_dir>'
)
"
```

This produces:
- `vocals.wav` — isolated vocal stem (for Whisper transcription + LTX conditioning)
- `instrumental.wav` — instrumental stem (for final assembly)

**GPU environment**: Do not run bare `uv run` for GPU-accelerated calls — the uv venv CUDA library paths may not include system NVIDIA libraries, causing `libcublas.so` errors. Use the `mv_audio` module which configures `LD_LIBRARY_PATH` correctly. If your environment uses a GPU lifecycle manager, ensure it is active before running.

### 2. Whisper Transcription

Transcribe the vocal stem with Whisper to generate word-level timestamps:

```bash
cd ~/alice/scripts
uv run python -c "
from mv_audio import transcribe_with_whisper
transcribe_with_whisper(
    audio_path='<project_dir>/vocals.wav',
    output_path='<project_dir>/lyrics/transcript.json'
)
"
```

This produces:
- `lyrics/transcript.json` — timestamped lyrics with word-level timing

**CRITICAL — Transcript source tracking**: The `transcript.json` MUST include an `"audio_source"` field recording the exact file path that was transcribed (e.g., `"vocals.wav"`). Downstream stages use this to determine whether timestamps are song-relative (stem source) or video-absolute (full video source). If Whisper transcribes `vocals.wav`, timestamps start at song=0s and need a prologue offset to align with video time. If it transcribes the full video audio, timestamps are already in video time. Without this field, there is no way to detect misalignment.

**GPU environment**: Same `LD_LIBRARY_PATH` requirement as Demucs — use the `mv_audio` module.

### 3. Verify Outputs

Check that all three artifacts exist and are non-empty:
- `vocals.wav` — vocal stem
- `instrumental.wav` — instrumental stem
- `lyrics/transcript.json` — timestamped transcription

If any artifact is missing or empty, diagnose and retry before proceeding.

### 4. Derive Audio Duration

Obtain the audio duration for use in timeline gap-filling. Run this before Step 5:

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 <project_dir>/audio.mp3
```

The output is the duration in seconds (e.g., `200.34`). Pass this value as `audio_duration` to `fill_timeline_gaps()`.

**Note:** This is the source song duration. For the full video timeline duration (including prologue and epilogue), use `audio_full.mp3` from Step 6. The beat sheet must use the full timeline duration, not the source song duration.

### 5. Generate Audio Region Classification

Run the mv-planner to classify vocal vs instrumental regions:

```bash
cd ~/alice/scripts
uv run python -c "
from mv_beats import parse_whisper_timestamps, fill_timeline_gaps, split_long_clips
import json

with open('<project_dir>/lyrics/transcript.json') as f:
    transcript = json.load(f)

beats = parse_whisper_timestamps(transcript)
beats = fill_timeline_gaps(beats, audio_duration=<duration from Step 4>)
beats = split_long_clips(beats, max_duration=18)

# Print summary for review
for b in beats:
    print(f\"{b['start']:.1f}s-{b['end']:.1f}s: {b.get('lyrics','(instrumental)')[:60]}\")
"
```

This provides a preliminary timeline view for the Beats interview.

## Output

Three artifacts in the project directory:
- `vocals.wav` — vocal stem
- `instrumental.wav` — instrumental stem
- `lyrics/transcript.json` — Whisper timestamped transcription

### 6. Create Full-Timeline Audio Track

The MV timeline uses absolute video time (e.g., 0-30s prologue, 30-230s song, 230-236s epilogue). The source audio file contains only the song — it has no prologue or epilogue silence. If audio slicing uses absolute video time offsets into the song-only file, shots beyond the song duration get empty audio, and prologue shots get wrong audio content.

**Before beat planning and generation, create a "full timeline" audio track** where time 0 aligns with video time 0:

```bash
# Create audio_full.mp3 = prologue_silence + song + epilogue_silence
# Example: 30s prologue silence + 200s song + 6s epilogue silence = 236s total
ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100:d=30 \
       -i audio.mp3 \
       -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100:d=6 \
       -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]" \
       -map "[out]" audio_full.mp3
```

Where:
- `d=30` = prologue duration in seconds (from treatment/storyboard)
- `audio.mp3` = the source song
- `d=6` = epilogue duration in seconds (from treatment/storyboard)

The resulting `audio_full.mp3` has the property that any absolute video time offset (e.g., 150s) maps directly to the correct position in the audio file. This is the file that downstream stages (beat planning, clip generation, final assembly) must use for audio slicing.

**Timing parameters** come from the treatment/storyboard. If the treatment specifies a prologue of 30 seconds and an epilogue of 6 seconds, those are the values to use. If the MV has no prologue or epilogue, the corresponding duration is 0 and `audio_full.mp3` is identical to `audio.mp3`.

**When to create this:** After the storyboard defines the timeline structure (prologue/song/epilogue boundaries), but before beat planning begins. The beat sheet's time offsets must reference `audio_full.mp3`, not `audio.mp3`.

## FSM Update

On success, transition `AUDIO_ANALYSIS` -> `APPROVED` which advances to `BEATS`. Use `mv_fsm_cli.py transition <project_dir> AUDIO_ANALYSIS APPROVED`.

## Failure Modes

- **Demucs fails**: Check GPU availability. Retry after ComfyUI idle.
- **Whisper fails**: Same GPU check. Also verify `vocals.wav` is non-silent.
- **libcublas error**: Never run bare `uv run` — always use the `mv_audio` module which sets up `LD_LIBRARY_PATH` correctly. If your environment has a GPU lifecycle manager, it handles this automatically.
