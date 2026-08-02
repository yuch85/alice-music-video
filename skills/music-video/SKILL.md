---
name: music-video
description: Orchestrate music video pre-production workflow — FSM-driven, stage-gated, resumable
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# /music_video — Music Video Pre-production Orchestrator

Single user-facing command that orchestrates the entire music video pre-production
workflow. Loads FSM state from disk, determines the current stage, and delegates
to the appropriate subordinate skill.

**This skill is ORCHESTRATION only.** It contains NO creative logic, NO generation
logic, NO prompt engineering. It loads state, delegates to skills, handles approvals.

## Pipeline Architecture

The music video pipeline follows a keyframe-based workflow:

1. **Pre-production** (stages INTERVIEW through SHOTS): Creative direction, beat planning, visual classification, cinematography assignments.
2. **Keyframe generation** (stage IMAGE_APPROVAL): QEI generates shot-specific keyframe candidates from canonical portraits. Keyframe Prompts (QEI prompts) are composed inline during this stage (Step 4 of the reference-images skill), using the shot list and continuity bible. Human QC approves one keyframe per shot.
3. **Keyframe Prompt Recording** (stage PROMPTS — first sub-stage): Record the final approved Keyframe Prompt text into `prompts/beat_{NN}.md` files. These prompts were built during IMAGE_APPROVAL; this step captures them as durable artifacts for downstream reference and potential revision.
4. **Motion prompt generation** (stage PROMPTS — second sub-stage): Compose Motion Prompts (temporal animation) from the approved keyframes. The keyframe defines the initial frame (identity, wardrobe, environment, framing, pose, expression); the motion prompt describes only what changes over time. Composition is determined by the keyframe, not the prompt.
5. **Validation** (stage VALIDATED): Pre-generation gate verifies all prerequisites.
6. **Generation** (stage GENERATING): LTX animates from approved keyframes using Motion Prompts. LTX handles temporal animation only — not shot design.
7. **QC** (stage QC): Technical quality checks on the final video.

See the `music-video-reference-images` skill for the full keyframe pipeline and tool responsibility matrix.

## FSM Stages (13 stages)

```
INTERVIEW -> TREATMENT -> CONTINUITY -> AUDIO_ANALYSIS -> BEATS -> STORYBOARD -> SHOTS
-> IMAGE_APPROVAL -> PROMPTS -> VALIDATED -> GENERATING -> QC -> COMPLETE
```

| Stage | Subordinate Skill | Output File |
|-------|-------------------|-------------|
| INTERVIEW | music-video-interview | project.md |
| TREATMENT | music-video-treatment | director_treatment.md |
| CONTINUITY | music-video-continuity | continuity_bible.md |
| AUDIO_ANALYSIS | music-video-audio-analysis | vocals.wav, instrumental.wav, lyrics/transcript.json, audio_full.mp3 |
| BEATS | music-video-beats | beat_sheet.md |
| STORYBOARD | music-video-storyboard | storyboard.md |
| SHOTS | music-video-shots | shot_list.md |
| IMAGE_APPROVAL | music-video-reference-images | approvals/ (keyframes) + refs/ (manifest) |
| PROMPTS | music-video-prompts | prompts/ |
| VALIDATED | music-video-validation | validation.md |
| GENERATING | music-video-generation | clips/ + final/ |
| QC | music-video-qc | qc_report.md |
| COMPLETE | — (report completion) | — |

## Dispatch Table

```
STAGE_TO_SKILL = {
    MVStage.INTERVIEW: "music-video-interview",
    MVStage.TREATMENT: "music-video-treatment",
    MVStage.CONTINUITY: "music-video-continuity",
    MVStage.AUDIO_ANALYSIS: "music-video-audio-analysis",
    MVStage.BEATS: "music-video-beats",
    MVStage.STORYBOARD: "music-video-storyboard",
    MVStage.SHOTS: "music-video-shots",
    MVStage.IMAGE_APPROVAL: "music-video-reference-images",
    MVStage.PROMPTS: "music-video-prompts",
    MVStage.VALIDATED: "music-video-validation",
    MVStage.GENERATING: "music-video-generation",
    MVStage.QC: "music-video-qc",
}
```

## Workflow

### 1. Check for Existing Project

Look for a project folder under `songs/music-videos/` matching the song name.
If the user provides a project name, search for it. If multiple matches exist,
ask the user to disambiguate.

### 2. If Project Exists — Resume from Disk State

1. Load `songs/music-videos/<Project>/index.md`.
2. Read the FSM state block (current stage + per-stage statuses).
3. Determine the current stage.
4. **If current stage is COMPLETE**: Report completion, show final output path.
   Display the `qc_report.md` summary and the `final/` directory contents.
5. **Otherwise**: Delegate to the appropriate subordinate skill (see Dispatch Table).

### 3. If Project Does NOT Exist — Initialize

Delegate to the `music-video-project` skill (Stage 0) for initialization:
- Pre-initialization asset search (existing audio, lyrics, portraits, refs)
- Project folder creation under `songs/music-videos/<Project>/`
- Directory structure setup (`lyrics/`, `refs/`, `concepts/`, `prompts/`,
  `clips/`, `final/`, `approvals/`)
- `index.md` creation with FSM state initialized to INTERVIEW
- Source asset placement (audio files, lyrics)

After initialization, the FSM state will be INTERVIEW. Delegate to
`music-video-interview` (Stage 1).

### 4. Stage Delegation

Based on the FSM state, invoke the corresponding subordinate skill. Each skill
performs its work and writes its output file. After the subordinate skill completes:

1. Re-read FSM state from `index.md`.
2. **If not COMPLETE**: Report the current stage and what happens next.
3. **If COMPLETE**: Celebrate and show the final output path.

### 5. Approval Gates

After each creative stage, present the output to the user and wait for approval
before transitioning FSM state. The stages requiring approval are:

- **INTERVIEW** -> `project.md` — creative direction
- **TREATMENT** -> `director_treatment.md` — immutable after approval
- **CONTINUITY** -> `continuity_bible.md` — injected into every prompt
- **AUDIO_ANALYSIS** -> `vocals.wav`, `instrumental.wav`, `lyrics/transcript.json` — audio artifacts verified before beat planning
- **BEATS** -> `beat_sheet.md` — timestamped layout
- **STORYBOARD** -> `storyboard.md` — visual classification per beat
- **SHOTS** -> `shot_list.md` — cinematography plan
- **IMAGE_APPROVAL** -> keyframe candidates per shot — user selects and QC's approved keyframes

For each approval gate:
1. Show the output (file content, images, or dashboard).
2. Ask: "Approve this and move to the next stage?"
3. **If APPROVED**: Run `uv run python scripts/mv_fsm_cli.py transition <project_dir> <stage> APPROVED`
4. **If REJECTED**: Run `uv run python scripts/mv_fsm_cli.py transition <project_dir> <stage> REJECTED`
   Then re-dispatch the same subordinate skill for revision.

## Resumability

Every invocation loads FSM state from disk. No state lives only in context.
If the session drops, the next invocation resumes from the last approved stage.

The FSM enforces stage gating: downstream stages cannot execute without approved
prerequisites. The `mv_fsm_cli.py` CLI handles all state transitions.

## Rectification Protocol

When validation (Stage 8 / VALIDATED) reports missing or failed assets, execute
this rectification protocol:

### Step 1: Read the Validation Report

Validation outputs a structured remediation report in `validation.md` listing:
- (a) Which beats/stages have missing assets
- (b) Which stage is responsible for each missing asset
- (c) Whether the asset can be auto-regenerated

### Step 2: Print Remediation Instructions

Present clear, actionable remediation instructions to the user:
- What failed and why
- What will happen next
- Estimated time for the fix

### Step 3: Trigger Targeted Rollback

Invoke the rollback CLI to set FSM state back to the stage responsible for the
missing asset. All stages strictly after the target are reset to NOT_STARTED:

```bash
uv run python scripts/mv_fsm_cli.py rollback <project_dir> --to <target_stage>
```

### Step 4: Re-dispatch Subordinate Skill

Re-dispatch the corresponding subordinate skill, targeting only the missing or
failed beat indices. For example, if prompts for beats B07 and B12 are missing,
regenerate only those two beats' prompts — not the entire stage.

### Step 5: Handle Non-Auto-Regeneratable Assets

If the missing asset cannot be auto-regenerated (e.g., source audio file is
missing from disk), halt and guide the user:
- Option A: Re-run initialization (`/music_video` -> Stage 0)
- Option B: Manually place the missing file

This is a terminal error condition — the orchestrator cannot proceed without
the source asset.

## User Interaction Model

### First Run
```
"What song would you like to make a music video for?"
-> Project initialization (music-video-project)
-> Creative interview (music-video-interview)
```

### Resume
```
"Where are we with <project>?"
-> Load index.md FSM state
-> Continue from current stage
```

### Status Check
```
"Show me the status of <project>"
-> Display index.md stage table
-> uv run python scripts/mv_fsm_cli.py status <project_dir>
```

## CLI Reference

All FSM operations go through `scripts/mv_fsm_cli.py`:

| Command | Purpose |
|---------|---------|
| `init <dir>` | Create project folder + index.md |
| `status <dir>` | Show current stage + per-stage table |
| `get <dir>` | Return current stage as a single line |
| `transition <dir> <stage> <status>` | APPROVED/REJECTED transition |
| `set-status <dir> <stage> <status>` | Set status without transitioning |
| `rollback <dir> [--to <stage>]` | Roll back, reset downstream stages |

Always run from the repo root (`$MV_REPO_DIR` (e.g., the repo root)) with `uv run python`.

## Backward Compatibility

The existing `/generate-music-video` skill remains available for quick, automated
runs without the structured pre-production workflow. For the full creative
pre-production workflow with stage gating and human approval, use `/music_video`.
