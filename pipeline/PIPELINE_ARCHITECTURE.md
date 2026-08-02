# MV Pipeline -- Architecture Reference

Canonical reference for the music video pipeline. The 10-clip pilot passed acceptance (2026-07-24). This document describes the accepted design -- do not deviate from it without explicit architectural review.

---

## 1. Overview

End-to-end flow from audio input to final video:

```
Input Audio
  |
  v
+-------------------+
| Stage 1: Demucs   |  Stem separation (vocals vs accompaniment)
|                   |  Output: vocals.wav, no_vocals.wav
+-------------------+
  |
  v
+-------------------+
| Stage 2: Whisper  |  Word-level transcription with timestamps
|                   |  Output: WordSegment list (text, start, end)
+-------------------+
  |
  v
+-------------------+
| Segment Planning  |  Group words into ClipSegments (~6s each)
|                   |  Fill coverage gaps (b-roll for silent regions)
|                   |  Split segments exceeding LTX-2 max (18s)
|                   |  Output: segment_plan.json
+-------------------+
  |
  v
+-------------------+
| Stage 3: Portrait |  Scene-locked reference portrait (Qwen I2I)
|                   |  (Skipped in resume mode -- uses plan data)
+-------------------+
  |
  v
+-------------------+
| Stage 4: Clips    |  Per-segment LTX-2.3 generation
|                   |  Each clip: audio-conditioned, reference-guided
|                   |  Resolution: 960x544 base -> 1920x1088 upscale
|                   |  Post-generation: trim padding frames
|                   |  Output: clip_NNN.mp4 (normalized)
+-------------------+
  |
  v
+-------------------+
| Stage 5: Composite|  Segment-based concat on timeline canvas
|                   |  Re-mux original master audio
|                   |  Post-processing: sharpen, grain, color
|                   |  Resolution crop: 1920x1088 -> 1920x1080
|                   |  Output: final_output.mp4
+-------------------+
```

Key design principle: **each clip is independently generated.** There is no inter-clip dependency during generation. Timing and continuity are established during compositing.

---

## 2. Execution Modes

### Auto Mode (`run_pipeline()` -> `_run_auto_mode()`)

Full pipeline: Demucs -> Whisper -> segment planning -> portrait -> clips -> composite.

Used for initial runs when no segment plan exists. Orchestrated by `generate_music_video_pipeline.py:run_pipeline()` lines 257-692. Auto-delegates to `_run_auto_mode()` (line 648) or `_run_controlled_mode()` (line 636) depending on `--mode`.

### Resume Mode (`run_pipeline()` -> `_run_resume_mode()`)

Reads `segment_plan.json` and `transcript.json` from a previous run. Skips stages 1-3 (Demucs, Whisper, portrait). Generates clips per the saved plan, then composites.

Production runs use resume mode. Invocation: `--resume` flag.

Entry point: `generate_music_video_pipeline.py:_run_resume_mode()` line 1035.

Resumes preserve:
- Segment timing (start, end, duration)
- Per-segment prompts and reference images
- Shot types and engine assignments
- Resolution and two-stage settings (recovered from plan data, lines 1067-1069)

### Controlled Mode

Generates per-segment reference images, writes `segment_plan.json`, and exits for human approval. After approval, resume mode generates clips.

Used during pre-production for creative control. Not used in production.

### Pilot Mode

Identical to resume mode with `--max-clips N` flag. Truncates the segment plan to the first N segments (line 1084-1096). Pilot and production execute the **same code path** -- only clip count differs.

---

## 3. Clip Generation

### Engine

LTX-2.3 via ComfyUI. Each clip is an independent ComfyUI workflow submission.

### Generation Resolution

- Base: 960x544 (low-res DiT pass)
- Two-stage upscale: 1920x1088 (latent-to-latent, x2 spatial)
- Final crop: 1920x1080 (8px top/bottom crop in compositing)

The 544 height is chosen because it is divisible by 32 (LTX-2 VAE spatial downsample requirement). 540 would break VAE. See `mv_clip_generate.py` lines 176-184.

### VRDG Sigmas

The two-stage upscale uses VRGDG (VRGameDevGirl) refinement sigmas: `0.909375, 0.725, 0.0` (3 sigmas, 2 DiT passes). Defined in `mv_mvconst.py:LTX2_UPSCALE_REFINEMENT_SIGMAS` line 56.

High starting sigma (0.909375) ensures the conv-net upscaler output is strongly re-noised and reconstructed by the DiT, preventing pixelation. Full-denoise terminus (0.0) ensures clean output.

### Audio Conditioning

Each clip receives a cropped segment of the vocals stem WAV. The crop boundaries come from **cascade positions** (see section 4), not Whisper timestamps.

Audio conditioning flows through ComfyUI's Audio VAE node. The model uses the audio to drive lip-sync and motion characteristics.

Code: `mv_clip_generate.py:_generate_clip()` lines 140-151.

### Duration Handling

LTX-2 quantizes frame counts to 8k+1. At 24fps this means a quantization step of 8/24 = 333ms. The actual video duration may differ from the requested duration by up to ~167ms (half-stride).

Requested duration is clamped: min 4s, max 18s. See `mv_clip_generate.py` lines 117-132.

### Pre-roll / Tail-loss Padding

Each clip generates extra frames:
- `PRE_ROLL_FRAMES = 4` (mv_comfyui.py line 39)
- `TAIL_LOSS_FRAMES = 4` (mv_comfyui.py line 42)

Padded length = requested duration + (4 + 4) / 24fps = requested + 0.333s.

After generation, `_trim_padding_frames()` removes these frames (see section 5).

### Retry Logic

Each clip has up to 2 retries (`MAX_CLIP_RETRIES = 2`). On failure, ComfyUI state is reset before retry. VRAM gate is checked before each attempt. 30s backoff between retries.

Code: `mv_clip_generate.py:_generate_clip()` lines 218-325.

### Hybrid Router

`_route_segment()` (line 339) is the single dispatch point. All segments default to LTX-2.3 with audio vocal stem conditioning for lip-sync. HuMo 14B is available as a fallback, routed when:
- `force_engine` parameter (explicit override)
- `classify_segment_engine()` (automatic: vocal presence + lyrics)

The router creates a copy of the segment with cascade timing overrides (lines 408-421) so conditioning audio uses cascade positions.

---

## 4. Cascade Timing

### Problem

Whisper timestamps (segment plan) and actual video durations diverge due to LTX-2's 8k+1 frame quantization. If conditioning audio is cropped from Whisper positions but clips are assembled at measured durations, conditioning and assembly audio will not match -- causing lip-sync drift.

### Solution

Cascade timing. A running position counter (`cascade_pos`) advances by **measured video durations**, not plan durations. Each clip's conditioning audio is cropped from `[cascade_pos, cascade_pos + seg.duration]`.

```
cascade_pos = 0.0
for each segment:
    cascade_start = cascade_pos
    cascade_end   = cascade_pos + seg.duration    # plan duration for audio crop
    clip = generate(segment with cascade timing override)
    video_dur = measure(clip)                      # actual duration
    clip_timings.append((cascade_start, cascade_start + video_dur))
    cascade_pos += video_dur                       # advance by MEASURED duration
```

Code: `_run_auto_mode()` lines 790-880, `_run_resume_mode()` lines 1184-1316.

### Key Properties

1. **Conditioning audio** is cropped from cascade positions, not Whisper timestamps.
2. **Assembly positions** use cascade positions (clip_timings list).
3. **Canvas duration** = final `cascade_pos` value (sum of all measured durations).
4. Failed clips do NOT advance `cascade_pos`.

### Invariant 3

`conditioning_audio == assembly_audio` for each clip. The audio the model sees during generation is the same audio region that appears in the final composite. This is enforced by passing `cascade_start` and `cascade_end` to `_route_segment()`, which creates a dataclass copy of the segment with overridden timing (lines 408-421 of mv_clip_generate.py).

---

## 5. Normalization

### `_trim_padding_frames()` (mv_audio.py line 182)

After generation, each clip has pre-roll and tail-loss padding frames. This function removes them from **both video and audio streams symmetrically**.

Trim logic:
- Video: `trim=start_frame=PRE_ROLL_FRAMES:end_frame=total_frames-TAIL_LOSS_FRAMES`
- Audio: `atrim=start=(PRE_ROLL_FRAMES/24):end=((total-TAIL_LOSS)/24)`

Both streams use the same frame boundaries converted to time, ensuring the output clip has matching video and audio durations.

The normalized clip is the **authoritative video artifact**. All downstream timing, cascade positions, and compositing use normalized clip durations, not raw generation durations.

Code: `mv_audio.py:_trim_padding_frames()` lines 182-280. Called from `mv_clip_generate.py:_generate_clip()` lines 277-284.

### Fallback

If `PRE_ROLL_FRAMES == 0` and `TAIL_LOSS_FRAMES == 0`, normalization is a no-op (returns True immediately). If total frame count is too small for both-end trim, it falls back to pre-roll-only trim (line 243).

---

## 6. Audio Architecture

Three distinct audio roles in the pipeline:

### Conditioning Audio

- **Source:** Vocals stem (Demucs-separated)
- **Purpose:** Drives lip-sync and motion during LTX-2 generation
- **Cropped from:** Cascade positions `[cascade_start, cascade_end]`
- **Format:** 44100Hz PCM WAV, placed in ComfyUI input directory
- **Lifecycle:** Used during generation only. Deleted after clip generation (line 299-303 of mv_clip_generate.py).

Code: `mv_audio.py:_crop_audio_segment()` lines 128-179.

### Embedded Clip Audio

- **Source:** Same as conditioning audio (copied to ComfyUI as `orig_audio_NNN.wav`)
- **Purpose:** Embedded in the generated MP4 by ComfyUI's CreateVideo node
- **Lifecycle:** Temporary. Present in individual clip MP4 files. **Discarded during assembly.** The compositing stage does not use clip audio streams.

Note: The embedded audio is a side effect of ComfyUI's video output, not an intentional pipeline design. It exists in clip files but is not used downstream.

### Master Audio

- **Source:** Original input audio file (the song)
- **Purpose:** Final soundtrack in the composite video
- **Lifecycle:** Passed to `_composite_with_ffmpeg()` as `input_audio` parameter. Re-muxed into the final output during post-processing (line 119 of mv_post.py).
- **Never modified.** The pipeline never edits, crops, or processes the master audio.

The master audio is the single source of truth for the final soundtrack. Audio continuity (no gaps, no pops) is guaranteed because the master audio is a single continuous file, not stitched from per-clip audio segments.

---

## 7. Compositing

### Segment Concat Mode

Compositing uses `_composite_timeline_canvas()` (mv_post.py line 188). Each clip is overlaid on its own black canvas segment, then segments are concatenated.

Process:
1. For each clip, create a black canvas segment of the clip's duration
2. Overlay the clip on the black canvas (normalizes encoding, ensures yuv420p)
3. Concatenate all segments with FFmpeg concat demuxer
4. Result: stitched video (video-only, no audio)

Code: `mv_post.py:_composite_timeline_canvas()` lines 188-315.

### Why Not Overlay

The previous approach (single black canvas with overlay filters and enable ranges) had timestamp issues and black frame gaps. Segment concat avoids global timestamp problems by compositing each clip independently at t=0.

### Audio Re-mux

After stitching, the master audio is re-muxed with the video in `_apply_post_processing()` (line 116-131 of mv_post.py). The audio is delayed by `pre_roll_frames` worth of microseconds to realign lips after the pre-roll trim.

Audio delay calculation: `pre_roll_frames / 24fps * 1,000,000` microseconds. Code: `mv_post_filter.py:_build_audio_delay_filter()`.

### Canvas Duration

Canvas duration = `cascade_pos` (sum of all measured clip durations). When `--max-clips N` is used, the canvas is shorter than the full audio duration. The output is cropped to the stitched video duration (`-t` parameter, line 130 of mv_post.py).

### Post-processing Filter Chain

Applied in `_apply_post_processing()` (mv_post.py line 58):

1. Resolution filter: crop 1920x1088 -> 1920x1080 (via `_resolution_filter()`)
2. Color grading: `colorchannelmixer` (neutral by default, LUT-enabled when `--apply-lut`)
3. 3D LUT: `lut3d` filter (optional, Cine Grade)
4. Film grain: `noise` filter (time-varying, default intensity 0.5)
5. Sharpening: `unsharp` filter (default strength 0.4)

Default grade is OFF (`apply_lut=False`) to match 09.9-16 reference clips.

### Fallback

If post-processing fails, a simple re-mux path is used (lines 464-488 of mv_post.py). Same resolution filter, no grain/sharpen/LUT. Ensures a valid output even when post-processing fails.

---

## 8. Resolution

### Pipeline Resolution Flow

```
960 x 544  -- LTX-2 base generation (DiT)
    |
    v  (x2 latent upscaler)
1920 x 1088  -- Two-stage upscale result (intermediate)
    |
    v  (crop in compositing: crop=1920:1080:0:0)
1920 x 1080  -- Final output (SHIPPED resolution)
```

The 1920x1088 intermediate is necessary because:
- LTX-2 VAE requires dimensions divisible by 32
- 544 = 17 * 32 (valid), 540 = 15 * 36 (invalid for VAE)
- 2 * 544 = 1088 (upscale result)
- 1088 - 1080 = 8px crop (4px top, 4px bottom via `crop=1920:1080:0:0`)

Resolution filter built in `mv_vram.py:_resolution_filter()`. Applied in compositing via `res_filter` parameter.

### VRAM Guard

Up-front VRAM check before any clip generation. Plans resolution based on longest segment. Falls back to lower resolution if VRAM is insufficient. Aborts if even lowest resolution won't fit.

Code: `_plan_clip_resolution()` in mv_vram.py. Called at lines 768-769 (auto) and 1171-1172 (resume).

---

## 9. Bug 2 Technical Debt

### Issue

Conditioning audio duration (based on segment plan duration) may differ from video duration (based on LTX-2's 8k+1 frame quantization).

Example from 5-clip pilot:
- Clip 1: plan 6.52s, measured 7.042s, diff -522ms (-1.9% temporal compression)
- Clips 2-5: plan ~6.26s, measured 6.042s, diff ~278ms (~4.6% temporal compression)

### Status

**Deferred.** Pilot passed acceptance. No observable lip-sync drift. LTX-2 appears to handle the duration mismatch gracefully.

### Instrumentation

`_log_duration_instrumentation()` (generate_music_video_pipeline.py line 1435) logs per-clip:
- Conditioning audio duration
- Measured video duration
- Cascade duration (authoritative)
- Absolute and percentage discrepancy

Emits WARNING if discrepancy exceeds threshold (default 500ms, configurable via `MV_DURATION_DISCREPANCY_WARN_S` env var).

**No runtime behavior change.** Purely observational.

### Revisit Conditions

1. Lip-sync drift at scale (10+ clips, full 27-clip run)
2. Warning threshold exceeded consistently
3. Model change (new LTX version, different quantization)
4. User-reported lip-sync inaccuracies

Full details: `.planning/debug/mv-bug2-technical-debt.md`

---

## 10. Instrumentation

### Duration Logging

After each clip generation:
- Measured duration via ffprobe
- Frame count (duration * 24fps)
- Cascade position advancement

Log format: `Clip N: measured X.XXXs (F frames), cascade advances to Y.YYYs`

### Clip Manifests

`_write_clip_manifest()` (mv_clip_generate.py line 485) writes a JSON manifest per clip:
- Cascade position and next position
- Conditioning audio start/end
- Measured duration, frame count, FPS
- Generation parameters (engine, model, seed, steps, scheduler)
- Reference image path, shot type, lyric text

Written to `clips/clip_NNN_manifest.json`.

### Pipeline Results

`_log_summary()` (generate_music_video_pipeline.py line 1490) writes `pipeline_results.json`:
- Input audio path
- Output directory and final output path
- Segment list with timing
- Clip list with paths
- Per-stage timing breakdown
- Mode (auto/resume/controlled)

### Duration Validation

`_validate_clip_duration()` (generate_music_video_pipeline.py line 1394) checks each clip's measured duration against requested duration. Error logged if deviation exceeds 2x quantization bound (~333ms).

Code: lines 1394-1421. Called after each successful clip generation.

---

## File Map

| File | Responsibility |
|------|----------------|
| `generate_music_video_pipeline.py` | Main orchestrator, CLI, mode routing |
| `mv_clip_generate.py` | Per-clip generation, hybrid router, manifests |
| `mv_audio.py` | Demucs, Whisper, audio crop, padding trim |
| `mv_post.py` | Compositing, post-processing, re-mux |
| `mv_comfyui.py` | ComfyUI client, constants (PRE_ROLL, TAIL_LOSS) |
| `mv_mvconst.py` | Magic numbers (resolution, duration, sigmas) |
| `mv_vram.py` | VRAM guard, resolution planning |
| `mv_segment.py` | Segment data classes, plan I/O |
| `mv_recovery.py` | Plan reconciliation with audio |
| `mv_prompt.py` | LLM prompt refinement |
| `mv_shot.py` | Shot variety (pose, motion, camera) |
| `mv_upscale.py` | Two-stage upscale workflow building |
| `mv_black.py` | Black frame generation (fallback) |
| `mv_lut.py` | LUT download, post-processing defaults |
| `mv_post_filter.py` | FFmpeg filter chain builders |
| `mv_slingshot.py` | Local LLM hibernate/wake during pipeline |
