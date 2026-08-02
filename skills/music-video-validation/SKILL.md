---
name: music-video-validation
description: Pre-generation validation — verify all prerequisites including storyboard completeness before GPU spend
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# /music-video-validation

Pre-generation validation gate (Stage 8). Runs all prerequisite checks before GPU-intensive generation begins. Blocks generation if any check fails.

## Prerequisites

- FSM state must be `PROMPTS` or later (all creative stages approved).
- All prior artefacts must exist and be valid.

## Process

1. Run `mv_validation.validate_prerequisites(project_dir)` which executes 16 checks:
   - **Source audio file** — exists, readable, size > 0. Terminal error if missing (no auto-regeneration).
  - **Full-timeline audio** — `audio_full.mp3` exists and its duration matches the expected full video timeline (prologue + song + epilogue). If the MV has a prologue or epilogue, `audio_full.mp3` must be longer than `audio.mp3`. If missing, block generation — clip audio slicing will be incorrect.
   - **Portrait file** — exists, valid image (size >= 10KB).
   - **Lyrics/transcript** — `transcript.json` or `lyrics/` directory present.
   - **Transcript validation** — valid JSON with segment entries, contiguous coverage (no gaps > 0.5s).
   - **Approved keyframes** — `refs/manifest.json` has entries matching beat count. Each approved keyframe must have passed QC for identity, continuity, framing, and lighting. Every keyframe must be 16:9 landscape (e.g., 1920x1088) — portrait keyframes will cause LTX to invent left/right content. Note: 1088 (not 1080) is the pipeline intermediate — base_h=544 divisible by 32 for VAE, upscaled 2x to 1088, cropped to 1080 in post.
   - **Prompts** — `prompts/beat_{NN}.md` files exist for all beats. Each file must contain both `## Keyframe Prompt` and `## Motion Prompt` sections.
   - **Keyframe prompts** — Every shot that requires a QEI-generated keyframe (XW, Wide, Full, ECU, two-character, wet-state) must have a Keyframe Prompt. Shots that use the canonical portrait directly (MCU, some CU) may omit Keyframe Prompts.
   - **Continuity bible** — exists, non-empty.
   - **Treatment** — exists, non-empty.
   - **Beat sheet** — exists.
   - **Shot list** — exists.
   - **Storyboard completeness** — `storyboard.md` exists AND every beat has all 15 fields:
     `visual_weight`, `narrative_importance`, `performance_focus` (6 keys summing to 100),
     `emotional_intensity`, `visual_energy`, `primary_viewer_focus`,
     `recommended_shot_duration_strategy`, `recommended_shot_scale`,
     `camera_movement_intensity`, `coverage_strategy`, `rationale`.
   - **Timeline coverage** — segments tile 0 -> full video duration contiguously (duration of `audio_full.mp3`, not `audio.mp3`).
   - **Aspect ratio** — matches target resolution (16:9 for 1920x1080, 2% tolerance).
   - **FSM state** — `PROMPTS` or later.
   - **Beat-by-beat content validation**:
     - Duration within [2s, 18s] generation limits.
     - Two-tier continuity check:
       - Tier 1 (BLOCKING): Deterministic string matching on invariance rules.
       - Tier 2 (WARNING): LLM-based semantic check (deferred).

2. If any check fails, report specific missing items and block generation.

3. If storyboard has gaps, list missing fields per beat. Do NOT proceed to generation until storyboard is complete.

4. If all pass, write `validation.md` with the structured report.

## Output

`validation.md` — structured validation report with pass/fail/missing classification.

## FSM Update

On validation pass, transition VALIDATED -> GENERATING:
```bash
python scripts/mv_fsm_cli.py transition <project_dir> VALIDATED APPROVED
```

On failure, stay at VALIDATED with specific error list. Do not implement inline state updates.

## Critical

Validation is a GATE. Generation must NOT start if validation fails. The generation skill checks `validation.md` before proceeding.

## Module Structure

- `scripts/mv_validation.py` — public API and ValidationReport dataclass
- `scripts/mv_validation_prerequisites.py` — 15 prerequisite check functions
- `scripts/mv_validation_content.py` — storyboard parser, timeline, aspect ratio
- `scripts/mv_validation_beat.py` — beat content validation + continuity checks
