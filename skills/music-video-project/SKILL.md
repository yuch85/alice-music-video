---
name: music-video-project
description: Initialize a new music video project folder with FSM state tracking
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# /music-video-project

Initialize a new music video project folder with FSM state tracking and folder schema. This skill runs FIRST when the user invokes `/music_video`.

## When to use

- The user wants to start a new music video project.
- The user invokes `/music_video` for a song that does not yet have a project folder.
- A project folder exists but needs state recovery (check `index.md` first).

## Inputs to collect from user

Collect these inputs BEFORE any file operations. Suggest defaults, require confirmation.

1. **Project name** — Used as the folder name under `songs/music-videos/`. Lowercase, hyphens for spaces (e.g. `song-title`).

2. **Audio file path** — Path to the full song file (`.mp3`, `.wav`, `.flac`). Must be a valid, readable file.

3. **Reference portrait path** — Canonical portrait image (`.jpg`, `.png`). MUST be a real photo — AI-generated portraits drift from canonical identity. If missing, suggest generating via QEI from a real source photo.

4. **Lyrics file** (optional) — Plain text lyrics (`.txt`, `.lrc`, `.srt`). If not provided at init, the Beat Sheet stage will auto-generate `transcript.json` via Whisper on the vocals stem.

5. **Target resolution** — Default `1920x1080`. Confirm.

6. **Aspect ratio** — Default `16:9`. Confirm.

7. **Prologue/Epilogue timing** — Does the MV include a prologue (ambient/silence before the song) or epilogue (content after the song ends)? Record durations in seconds. Default: 0s prologue, 0s epilogue. These values are used to create `audio_full.mp3` (the full-timeline audio track) during audio analysis.

## Pre-initialization Asset Search

Before creating the project folder, search for existing assets:

1. Scan `songs/` and `songs/music-videos/` for directories matching the project name (case-insensitive, fuzzy match on common variations like hyphens vs spaces).

2. Within matching directories, identify:
   - Audio files: `*.mp3`, `*.wav`, `*.flac`
   - Lyrics files: `*.txt`, `*.lrc`, `*.srt`
   - Portrait/reference images: `*.jpg`, `*.png`, `*.jpeg`

3. Present discovered assets to the user:
   > "Found existing assets for '<project>': [list]. Confirm reuse? [y/n]"

4. On confirmation, use the discovered asset paths. On decline, use user-provided paths.

## Boundary Validation (before creating the project)

1. **Audio file validation**: Verify the user-provided audio path points to a valid, readable, non-zero file. If missing or unreadable, interactively prompt for the correct path (suggest searching under `songs/`). Do not fail silently.

2. **Raw Lyrics** (source asset, optional at init): If provided, copy to `lyrics/` subfolder. If missing at init, check whether `transcript.json` will be auto-generatable via Whisper at Beat Sheet stage. If **neither** raw lyrics nor auto-generation is possible (e.g., no audio file), warn the user that the Creative Interview stage will have no lyrics to derive story defaults from — but allow initialization to proceed.

3. **Timed Lyrics / transcript.json** (pipeline artifact, NOT generated at init): Init allocates the `lyrics/` folder for this artifact; it is populated later at Beat Sheet stage via Whisper on the vocals stem.

**Distinction**: Raw Lyrics = source asset provided by the user (optional at init, copied to `lyrics/`). Timed Lyrics = pipeline artifact auto-generated at Beat Sheet stage via Whisper (written as `lyrics/transcript.json`). Downstream stages (beats, storyboard) consume `transcript.json`, not raw lyrics.

## Actions

Execute these steps in order:

### 1. Create project folder

```
songs/music-videos/<Project>/
```

### 2. Create all subdirectories (from PRD schema)

```
lyrics/       Raw source lyrics + Whisper transcript.json (generated later)
refs/         Reference materials, seed portrait, generated reference images
concepts/     Staged concepts for storyboard / styling
prompts/      Output prompt files (.json/text) per beat
clips/        Rendered video clips per beat
final/        Combined / composited outputs
approvals/    Beat-level directories for candidate images and approvals
```

### 3. Copy/relocate physical assets to canonical project paths

- Copy input audio to project root as `audio.mp3` (or preserve original filename). All downstream stages reference this canonical path.
- If source lyrics/TXT provided, copy to `lyrics/` (e.g., `lyrics/lyrics.txt`). Preserve original filename. Note: `lyrics/` stores raw source lyrics as a read-only reference artifact. Processed timestamps with timing data come from Whisper's `transcript.json`. Downstream stages consume `transcript.json`, not `lyrics/`.
- Copy seed portrait to `refs/portrait.jpg`. This is the canonical portrait path used by all downstream stages (QEI, generation).
- If multiple characters, copy each to `refs/<character>-portrait.jpg`. These are the canonical portraits — persistent assets that never change.
- If additional reference images provided, copy to `refs/` with descriptive filenames.

**Persistent asset philosophy:** Only canonical character portraits and continuity-state variants (e.g., wet-state for rain scenes) are persistent assets. Shot-specific keyframes are generated per-shot via QEI during Stage 6 and are not pre-generated at init. Do not create large reference libraries.

### 4. Write `index.md` using the template

Use `project_template.md` as the base. Fill in the actual metadata (audio path, lyrics path, portrait path, resolution, aspect ratio, creation date). This is the canonical state file, NOT documentation.

### 5. Record canonical asset paths

Record in the Metadata section of `index.md`:
- Audio: canonical path in project root
- Lyrics: canonical path in `lyrics/` folder
- Portrait: canonical path in `refs/` folder

### 6. Initialize FSM state

Run: `mv_fsm_cli.py init <project_dir>`

This sets FSM state to INTERVIEW with all stages NOT_STARTED and writes the FSM JSON block to `index.md`.

### 7. Register project in `songs/music-videos/index.md`

Append the project to the Projects section of the master index.

## Output

Project folder ready for Stage 1 (Creative Interview). FSM state = INTERVIEW.

## Resumability

If the project folder already exists with `index.md`:
1. Load existing state: `mv_fsm_cli.py status <project_dir>`
2. Report current stage to the user
3. Do NOT overwrite existing files
4. Continue from the current stage

## FSM Integration

Use `mv_fsm.MusicVideoFSM` for state management. After each stage transition, persist the FSM state to `index.md` via `mv_fsm_cli.py`. Never manipulate the FSM JSON block with regex — always use the CLI.
