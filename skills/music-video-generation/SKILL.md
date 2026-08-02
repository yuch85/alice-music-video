---
name: music-video-generation
description: Consume approved keyframes and run LTX-2.3 pipeline (HuMo 14B optional fallback) — LTX handles temporal animation only; does NOT rewrite generation code
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# /music-video-generation

Video clip generation (Stage 9). Thin wrapper that maps approved keyframes and pre-production artefacts to the LTX-2.3 pipeline's parameters (HuMo 14B available as opt-in per-clip override via `--engine humo`). Does NOT rewrite generation code.

## LTX Role in the Keyframe Pipeline

LTX-2.3 is responsible **only for temporal animation** — camera movement, facial motion, body motion, and environmental motion. It does NOT design shots, control framing, or determine composition. Those responsibilities belong to QEI (which generates the keyframe) and Human QC (which approves it).

**Ref-image dominance**: In I2V mode, LTX inherits first-frame composition from the approved keyframe more strongly than it honors scene prompt text. If the keyframe is a close-up, the output will be a close-up clip regardless of what the prompt says about "wide shot." The lever is the keyframe, not prompt text. See the `ltx-prompting` skill for the full ref-image dominance principle.

**Motion prompts are required**: LTX clips are near-static without explicit motion prompts. Every motion prompt must describe body/scene movement. Avoid passivity tokens (see `ltx-prompting` skill's passivity token blacklist — phrases like "holds gaze" and "composed stillness" produce frozen-face output).

## Prerequisites

- FSM state must be `GENERATING`.
- `validation.md` shows PASS.
- All artefacts approved.

## Process

1. Read `validation.md` — confirm PASS.

2. Read `refs/manifest.json` — extract per-clip approved keyframe paths (pointing to `approvals/Beat_<ID>/candidate_*.png`).

3. Read `prompts/beat_{NN}.md` — extract Keyframe Prompts (for QEI keyframe generation) and Motion Prompts (for LTX clip generation) per beat.

4. Read `shot_list.md` — extract shot types (singer/broll/instrumental/black).

5. Read `beat_sheet.md` — extract segment timings.

6. Read `continuity_bible.md` — extract continuity rules (weather progression, wardrobe behavior, prop placement). Do NOT prepend full character description strings to prompts — the keyframe already encodes static appearance. Only use continuity data that affects motion (e.g., wet fabric behavior, wind effects, weather state).

7. Assemble pipeline parameters:
   - `input_audio`: `audio_full.mp3` (full-timeline audio, NOT `audio.mp3`). The full-timeline track includes prologue silence and epilogue silence, so absolute video time offsets from the beat sheet map correctly to audio positions. See `music-video-audio-analysis` Step 6.
   - `portrait_path`: canonical portrait
   - `per_clip_references`: from manifest
   - `prompts`: motion prompts from prompts/
   - `clip_durations`: from beat sheet timings
   - `per_clip_engines`: default ltx2 for all clips. Override to humo per-clip via shot_list annotation when lip-sync precision is the top priority (edge case fallback).
   - **Audio conditioning per clip**: Read `**Audio**` field from each prompt file.
     When `**Audio**: silence` is present, pass `None` as audio to LTX for that
     clip (no audio conditioning = no lip-sync mouth movement). This is required
     for all non-singing characters to prevent unwanted mouth animation. SINGER
     shots without the flag receive the full audio track (default behavior).

8. Invoke existing pipeline via MCP tool (`music_video_generate_refs` + `music_video_generate_clips`) or direct CLI:
   ```bash
   cd $MV_REPO_DIR
   uv run python scripts/generate_music_video_pipeline.py \
       --input <audio> --output <project_dir> --portrait <portrait> \
       --scene-prompt "<base scene description>" \
       --width 1920 --height 1088
   ```

   **Why 1088, not 1080?** LTX-2 base height must be divisible by 32 (VAE spatial downsample), so base_h = 544 (= 17×32). The combined workflow's 2x generative upscale produces 1088 (= 2×544). The final ffmpeg composite crops each clip from 1920×1088 to 1920×1080 in post-production. Do NOT change the pipeline height to 1080 — it will break VAE divisibility.

9. After generation completes, write `generation_manifest.md` (immutable record).

## Generation Manifest

Written to `generation_manifest.md` as the immutable proof of reproducibility.

### Required Fields

- Model versions (LTX-2.3 model hash — primary; HuMo 14B model hash — include only if HuMo was used for any clip).
- Workflow version (ComfyUI workflow JSON version)
- Per-clip: seed, prompt hash, approved keyframe path, engine used, resolution, duration
- Global settings: VRDG sigmas, CFG, sampler, steps
- Generation date, GPU used
- **Git commit hash**: obtained dynamically via `git rev-parse HEAD` (subprocess)
- Pipeline parameters (all run_pipeline args)

### Immutability Enforcement

After writing `generation_manifest.md`:
1. Raise a validation error on any subsequent write attempt
2. Set file permissions to read-only (`chmod 444 generation_manifest.md`) as secondary safety net

The manifest is the "immutable proof of reproducibility" — once written, it is never modified.

## Critical Constraint

This skill does NOT rewrite any generation code. It consumes pre-production artefacts and passes them as PARAMETERS to the existing pipeline. The existing pipeline (`generate_music_video_pipeline.py` and `mv_*` modules) is untouched.

## Engine Selection (LTX-2.3 Default, HuMo Fallback)

LTX-2.3 is the default engine for all clip types. This was validated by extensive A/B testing: LTX scored balanced across all dimensions (lip sync, camera motion, identity preservation, background fidelity, composability) and produced the best video in weeks of iteration.

HuMo 14B remains available as an opt-in fallback for edge cases where LTX underperforms (e.g., extreme lip-sync precision requirements). To use HuMo for a specific clip, annotate the shot list entry with `engine: humo` or pass `per_clip_engines` with `"humo"` for that segment index.

LTX golden settings (locked): model `ltx-2.3-22b-distilled-1.1-Q6_K.gguf`, VRDG V5.1 sigma schedule, euler sampler, CFG=1.0, `use_vrdg_sigmas=True`, `use_lipdub=False`, `text_encoder_device="default"` (GPU). GPU encoder safe at 1080p — frame-count probe measured 31.2GB peak VRAM (18s clip, 1920x1088, combined workflow) with ~17GB headroom on 48GB card. CPU was conservative carryover from single-stage OOM; reverted to GPU subsequently.

## FSM Update

On generation complete, transition GENERATING -> QC:
```bash
python scripts/mv_fsm_cli.py transition <project_dir> GENERATING APPROVED
```

Do not implement inline state updates.

## GPU Operations

### GPU Lifecycle Management

Video clip generation requires substantial GPU resources. If your environment runs
other GPU workloads (e.g., a local LLM), a GPU lifecycle manager can suspend them
before generation and restore them afterward, ensuring restoration even after
abnormal termination.

- If a GPU lifecycle manager is configured, it handles LLM hibernate/wake automatically.
- If no GPU lifecycle manager is configured, the pipeline proceeds normally.
  Ensure sufficient VRAM is available before starting generation.
- Use the `mv_audio` module for any GPU-accelerated audio operations to ensure
  correct CUDA library paths.

> **Alice implementation note:** In the private Alice environment, a GPU lifecycle
> manager automatically hibernates the local LLM before clip generation and restores
> it afterward. This behaviour is specific to Alice's deployment and is not required
> by the public pipeline.
