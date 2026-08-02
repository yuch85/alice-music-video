---
name: generate-music-video
description: Generate a beat-aligned music video from a song using the LTX-2 pipeline (Demucs -> Whisper -> scene-locked portrait -> LTX-2 clips -> FFmpeg composite). Covers mode choice, the human inputs to collect, where to write them, the resolution/aspect gotcha, and the controlled-run recipe.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# /generate-music-video

Generate a beat-aligned music video from an audio track using the LTX-2 pipeline. The pipeline separates stems (Demucs), aligns lyrics (Whisper), builds a scene-locked reference portrait, generates one LTX-2 clip per beat, then composites with FFmpeg.

## When to use
- The user wants a music video from a song in `songs/`.
- They want a quick draft without the structured pre-production workflow.

**NOTE:** This is the legacy quick-run skill. For new projects, use `/music_video` which provides the structured FSM-driven workflow with stage gating, human approval gates, and the keyframe pipeline.

## Structured Workflow

For a structured creative pre-production workflow, use `/music_video` instead. The new workflow provides:
creative interview, director's treatment, continuity bible, beat sheet, visual storyboard,
shot list, reference image approval, prompt generation, pre-generation validation, and final QC.

`/generate-music-video` remains available for quick, automated runs.

## Two modes
- **Auto** — fully automated one-shot pipeline. Quick draft only.
- **Controlled** (recommended) — `music_video_generate_refs` -> user approves per-beat refs -> `music_video_generate_clips`. Per-beat creative control.

## Collect these human inputs FIRST

1. **Audio file** — Path to the song in `songs/`. Confirm variant.
2. **Canonical portrait** — MUST be a real photo. AI-generated portraits drift from canonical identity.
3. **Character + clothing description** — Consistent string prepended to every HuMo segment prompt (e.g. "a woman in a white dress").
4. **Camera motion preference** — From `CAMERA_MOTION_TEMPLATES`: tracking, dolly-in, crane, handheld, zoom-in, orbit, push-in, tilt-down, whip pan, Dutch angle, rack focus, dolly zoom.
5. **Scene prompt / environment** — Base scene description. Vary backgrounds for visual interest.
6. **Mode** — Auto or controlled. Default: controlled.
7. **Aspect & resolution** — Default 960x544 base, final 1920x1080 via 2x Path B generative upscale + crop. Do NOT pass 1920x1088 as base.
8. **Max segment length** — Default 10s (range 4-18s). HuMo clips: 4-8s for best identity.

## Where inputs live
- Per-project: `songs/music-videos/<project>/`
- Creative inputs: `storyconcept.txt`, `subjectsandscenes.txt`, `themestyle.txt`, `index.md`
- Outputs: `stems/`, `refs/`, `segment_plan.json`, `clips/`, `final_output.mp4`

## Pipeline Entry Points

The pipeline can be invoked via MCP tools (if configured) or directly through the CLI.

| Mode | Entry Point | Key Args |
|------|-------------|----------|
| Auto (one-shot) | `generate_music_video_pipeline.py` | input_audio_path, output_dir, portrait_path, scene_prompt, max_segment_s, width=960, height=544 |
| Controlled: refs | `music_video_generate_refs` (MCP) or CLI | + optional storyconcept_path / themestyle_path / subjectsandscenes_path / lyrics_path |
| Controlled: clips | `music_video_generate_clips` (MCP) or CLI | output_dir (reads width/height from segment_plan.json) |

> **Alice implementation note:** In the private Alice environment, MCP tools
> (`alice_generate_music_video`, `music_video_generate_refs`, `music_video_generate_clips`)
> wrap the CLI entry points and handle GPU lifecycle management automatically.

## Direct local launch (no MCP)

```bash
cd $MV_REPO_DIR
uv run python scripts/generate_music_video_pipeline.py \
    --input songs/<project>/<song>.mp3 \
    --output songs/music-videos/<project> \
    --portrait songs/music-videos/<project>/<portrait>.jpg \
    --scene-prompt "<base scene description>" \
    --width 960 --height 544 --two-stage
```

The `mv_audio` module handles `LD_LIBRARY_PATH` automatically. For direct CLI runs,
inject the uv-venv NVIDIA library directories at launch time
(see `reference_mv_pipeline_run_libcublas_gotcha.md`).

## Controlled-run recipe
1. Ensure portrait + audio exist; write creative-input files if B-roll/narrative wanted.
2. Run reference image generation (stages 1-4.5) with creative-input paths.
   Returns `segment_plan.json` + `ref_images`.
3. Show the user the ref images / segment plan; get approval.
4. Run clip generation (stages 4-5) on the same `output_dir`.
5. Verify: `ffprobe` the `final_output.mp4` (codec, duration matches audio, resolution).

## GPU Lifecycle Management

The pipeline may require exclusive GPU access during clip generation. If your
environment runs a local LLM alongside the video generation pipeline, a GPU
lifecycle manager can suspend the LLM before rendering and restore it afterward.

- Check GPU availability before a run; ensure ComfyUI/LTX services are free.
- If no GPU lifecycle manager is configured, the pipeline proceeds normally —
  VRAM contention is the operator's responsibility.
- Long idle stages during clip generation are expected.

> **Alice implementation note:** In the private Alice environment, a GPU lifecycle
> manager automatically hibernates the local LLM before clip generation and restores
> it afterward. This behaviour is specific to Alice's deployment and is not required
> by the public pipeline.

## Additional operational detail

Pipeline internals (resolution node-trace, VRAM math, HuMo parameters, LTX golden settings,
camera motion formulas, audio flow, engine comparison) are documented in
`reference_mv_legacy_pipeline_details.md`.
