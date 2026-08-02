# Pipeline Architecture

## Pipeline Stages

The music video pipeline executes in ordered stages. Each stage produces artifacts that the next stage consumes. The pipeline is resume-aware: if interrupted, it can pick up from the last completed stage.

### 1. Audio Analysis

Stem separation via Demucs isolates vocals, music, bass, and other stems. Whisper transcribes the vocals for beat/lyric alignment.

**Input:** Music track (MP3, WAV, FLAC)
**Output:** Separated stems, transcript with timestamps

**Modules:** `pipeline/mv_audio.py`

### 2. Beat Detection

Generates a structured beat sheet from the transcript and audio timestamps. Each beat is a timed segment with associated lyrics and metadata.

**Input:** Transcript, audio file
**Output:** `transcript.json` (beat sheet)

**Modules:** `pipeline/mv_beats.py`

### 3. Storyboard

Visual classification of each beat: weight, performance focus, shot scale, emotional intensity, camera movement. This is a creative planning step driven by Claude Code skills.

**Input:** Beat sheet, creative brief
**Output:** `storyboard.json`

**Modules:** `pipeline/mv_storyboard.py`, `skills/music-video-storyboard`

### 4. Shot Planning

Assigns shot types, poses, and camera motion per segment. Determines which segments use HuMo (singing/talking) vs. LTX (b-roll, atmosphere).

**Input:** Storyboard, treatment
**Output:** `segment_plan.json`, `plan.json`

**Modules:** `pipeline/mv_shot.py`, `pipeline/mv_segment.py`, `skills/music-video-shots`

### 5. Reference Images

Generates per-segment keyframes using Qwen Image Edit or ComfyUI image workflows. Each segment gets one or more reference images that guide the video generation.

**Input:** Storyboard, character portraits, scene prompts
**Output:** `keyframes/` directory

**Modules:** `pipeline/mv_refs.py`, `skills/music-video-reference-images`

### 6. Prompt Refinement

Two-stage LLM refinement: first generates text-to-image prompts, then image-to-video motion prompts. Optimized for the target generation model.

**Input:** Reference images, storyboard
**Output:** `prompts.json`

**Modules:** `pipeline/mv_prompt.py`, `skills/music-video-prompts`

### 7. Validation

Runs 15+ prerequisite checks before GPU-intensive generation begins. Validates storyboard completeness, timeline coverage, aspect ratio consistency, and beat-by-beat content (duration, continuity).

**Input:** All planning artifacts
**Output:** Pass/fail report

**Modules:** `pipeline/mv_validation.py`, `pipeline/mv_validation_prerequisites.py`, `pipeline/mv_validation_content.py`, `skills/music-video-validation`

### 8. Clip Generation

Routes each segment to the appropriate engine:
- **LTX-2.3** for b-roll, atmosphere, and non-speaking shots
- **HuMo 14B** for singing lip-sync and talking-head shots
- Hybrid clips may combine both

Generation respects VRAM constraints with automatic fallback.

**Input:** Reference images, prompts, segment plan
**Output:** `clips/` directory

**Modules:** `pipeline/mv_clip_generate.py`, `pipeline/mv_clip.py`, `pipeline/mv_humo_gen.py`, `pipeline/mv_ltx2.py`, `skills/music-video-generation`

### 9. Quality Control

Evaluates generated clips for quality. Identifies clips that need regeneration (e.g., frozen frames, poor lip-sync, incoherent motion).

**Input:** Generated clips
**Output:** QC report, regeneration list

**Modules:** `skills/music-video-qc`

### 10. Assembly

FFmpeg-based cascade timing assembles the clips into the final video. Applies the continuous audio overlay (the Yui gold standard), color grading via LUTs, and burns in ASS subtitles and credits.

**Input:** Clips, audio stems, storyboard
**Output:** Final video (MP4)

**Modules:** `assemble/_mv_assemble_seamless.py`, `assemble/_mv_burn_subs_and_credits.py`, `assemble/_mv_credits.py`

## Key Design Principles

### Planning Drives Generation

Creative decisions precede every GPU pass. The pipeline does not generate video until the storyboard, shot list, and prompts are finalized. This prevents wasteful trial-and-error generation.

### Resume-Aware

An FSM (finite state machine) tracks pipeline progress. If a run is interrupted, the pipeline resumes from the last completed stage rather than starting over.

**Module:** `pipeline/mv_fsm.py`, `pipeline/mv_fsm_persist.py`

### VRAM-Aware

The pipeline detects available VRAM and adjusts accordingly:
- Two-stage upscale (960x544 -> 1920x1088) on 48GB+ GPUs
- Single-stage generation on 24GB GPUs
- Automatic clip duration capping to avoid OOM

**Modules:** `pipeline/mv_vram.py`, `pipeline/mv_vram_model.py`

### Hybrid Routing

Not every clip needs the same engine. The pipeline routes segments based on content:
- HuMo for singing/talking (lip-sync required)
- LTX-2.3 for b-roll, atmosphere, establishing shots
- Per-clip engine override via `--per-clip-engines`

**Module:** `pipeline/mv_clip_generate.py`

## Claude Code Skills

Pre-production is driven by Claude Code skills. Each skill guides the user through a structured planning phase:

| Skill | Purpose |
|-------|---------|
| `music-video-project` | Initialize project structure |
| `music-video-interview` | Creative brief and direction |
| `music-video-treatment` | Treatment document |
| `music-video-continuity` | Continuity bible (character, setting, props) |
| `music-video-audio-analysis` | Audio analysis guidance |
| `music-video-beats` | Beat sheet creation |
| `music-video-storyboard` | Visual storyboard |
| `music-video-shots` | Shot list and segment planning |
| `music-video-reference-images` | Keyframe generation prompts |
| `music-video-prompts` | Prompt refinement (T2I + I2V) |
| `music-video-validation` | Pre-flight checks |
| `music-video-generation` | Clip generation orchestration |
| `music-video-qc` | Quality control |

## Directory Structure

```
pipeline/          Core pipeline modules
  generate_music_video_pipeline.py  Main entry point / CLI
  mv_audio.py                 Audio analysis (Demucs, Whisper)
  mv_beats.py                 Beat detection
  mv_storyboard.py            Storyboard processing
  mv_shot.py                  Shot planning
  mv_segment.py               Segment management
  mv_refs.py                  Reference image handling
  mv_prompt.py                Prompt refinement
  mv_clip.py                  Clip management
  mv_clip_generate.py         Clip generation orchestrator
  mv_humo_gen.py              HuMo talking-head generation
  mv_humo_di.py               HuMo dependency injection
  mv_humo_postprocess.py      HuMo post-processing
  mv_comfyui.py               ComfyUI client wrapper
  mv_comfyui_client.py        ComfyUI HTTP client
  mv_upscale.py               Upscaling logic
  mv_lut.py                   Color grading LUTs
  mv_post.py                  Post-processing utilities
  mv_post_filter.py           Post-filter operations
  mv_validation.py            Validation orchestrator
  mv_validation_prerequisites.py  Prerequisite checks
  mv_validation_content.py    Content validation
  mv_beat_validation.py       Beat-specific validation
  mv_fsm.py                   Finite state machine
  mv_fsm_persist.py           FSM state persistence
  mv_fsm_cli.py               FSM CLI interface
  mv_vram.py                  VRAM detection
  mv_vram_model.py            VRAM-aware model selection
  mv_mvconst.py               Pipeline constants
  mv_utils.py                 Shared utilities
  mv_recovery.py              Recovery from failures
  mv_slingshot.py             Pipeline launcher
  mv_black.py                 Black frames utility

assemble/          Assembly and post-processing
  _mv_assemble_seamless.py    Seamless clip assembly
  _mv_assemble_final.py       Final assembly (legacy)
  _mv_assemble_final_v2.py    Final assembly v2
  _mv_assemble_cascade.py     Cascade timing assembly
  _mv_assemble_trim.py        Trim and align
  _mv_burn_subs_and_credits.py  Subtitle and credits burn-in
  _mv_credits.py              Credits generation
  _mv_audit_audio_seams.py    Audio seam detection
  _mv_fix_audio_seams.py      Audio seam repair
  _mv_fix_audio_sync.py       Audio sync correction
  _mv_build_*_dashboard.py    Dashboard generators

workflows/         ComfyUI integration
  workflows.py                Core workflow builders
  workflow_humo.py            HuMo workflow builder
  workflow_ltx2.py            LTX-2.3 workflow builder
  workflow_ltx2_upscale.py    LTX-2.3 upscale subgraph
  workflow_dialogue.py        Dialogue-driven generation
  workflow_ovi.py             Ovi I2AV workflow
  comfyui_node_compat.py      Node schema validation
  humo_models.py              HuMo model constants
  comfyui/                    Pre-exported workflow JSONs

skills/            Claude Code skills for pre-production
tests/             Test suite
```

## Constants

Magic numbers are centralized in `pipeline/mv_mvconst.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_CLIP_DURATION_S` | 16 | Default HuMo clip duration |
| `HUMO_FALLBACK_CLIP_DURATION_S` | 8 | Fallback for drift-prone segments |
| `CLIP_DURATION_FLOOR_S` | 6 | Hard minimum clip length |
| `CLIP_DURATION_CEILING_S` | 18 | VRAM ceiling per clip |
| `DEFAULT_LOWRES_W/H` | 960x544 | Base resolution for two-stage upscale |
| `UPSCALE_FACTOR` | 2 | Generative upscaler scale factor |
