---
name: music-video-beats
description: Create beat sheet from Whisper timestamps — narrative, emotional, visual purpose per beat
allowed-tools: [Read, Write, Edit, Bash]
---

# Beat Sheet Skill (Stage 4)

## Prerequisites

- FSM state must be `BEATS`.
- `AUDIO_ANALYSIS` stage must be `APPROVED`.
- `continuity_bible.md` exists and is `APPROVED`.
- Audio analysis artifacts must exist (produced by AUDIO_ANALYSIS stage):
  - `lyrics/transcript.json` — Whisper timestamped transcription
  - `vocals.wav` — vocal stem
  - `instrumental.wav` — instrumental stem
  - `audio_full.mp3` — full-timeline audio (prologue silence + song + epilogue silence). See `music-video-audio-analysis` Step 6. If this file is missing, the beat sheet cannot use correct time offsets.

**If AUDIO_ANALYSIS is not APPROVED, do not proceed.** The Whisper timestamps are critical for calculating segment/beat durations and determining structural sections (chorus boundaries, verse transitions). Without them, meaningful beat planning is impossible. Transition AUDIO_ANALYSIS to APPROVED first, or run the audio analysis via the `music-video-audio-analysis` skill.

## Process

1. Load `lyrics/transcript.json` and parse Whisper timestamps into beat entries via `mv_beats.parse_whisper_timestamps()`
2. **Determine the prologue offset** — Check `transcript.json`'s `"audio_source"` field:
   - If source is a stem file (`vocals.wav`, `instrumental.wav`): timestamps are song-relative. Add the prologue duration (from treatment/storyboard, or the project config constant like `PROLOGUE_DURATION_S`) to all timestamps.
   - If source is the full video (`final_output_v6.mp4`, `audio_full.mp3`): timestamps are already in video time. No offset needed.
   - If `"audio_source"` is missing: **assume song-relative** (the default Whisper output) and apply prologue offset. Log a warning.
3. **Run timeline gap-filling** (`mv_beats.fill_timeline_gaps`) — insert instrumental slots for gaps between lyric segments so the timeline is contiguous from 0 to full video duration (including prologue and epilogue). Use `audio_full.mp3` duration, not `audio.mp3` duration.
4. **Run clip duration splitting** (`mv_beats.split_long_clips`) — split any beat exceeding the max clip duration (18s default) into sub-beats that fit the generation pipeline's constraints
5. For each beat, determine: narrative purpose (what story beat), emotional purpose (what feeling), visual purpose (what we see), energy level, dominant character, transition from/to, detailed notes
6. Use LLM to refine purposes based on treatment + continuity bible context
7. Write `beat_sheet.md` — structured markdown table with all 12 fields per beat (11 PRD fields + section)

**Critical timing rule:** All time offsets in `beat_sheet.md` are absolute video time (0 = start of video, which may be prologue silence). They must map directly to positions in `audio_full.mp3`. Never use `audio.mp3` (song-only) offsets for the beat sheet — that causes shots beyond the song duration to have empty audio, and prologue shots to have wrong audio content.

## Edge Cases

- **Instrumental-only songs** (no vocals): When `transcript.json` is empty or contains no words, create beats from the instrumental structure. Use `ffprobe` to derive `audio_duration`, then split the timeline into segments of 8-18s. Mark all beats as `(instrumental)` with B-roll or environmental focus. Classify sections by energy/dynamics shifts rather than lyric-based structure (intro, build, drop, outro).
- **Whisper transcription failures**: If `transcript.json` is missing, empty, or clearly wrong (e.g., wrong language), re-run Whisper via the `music-video-audio-analysis` skill before proceeding. Do not fabricate timestamps.
- **Overlapping lyrics** (harmonies, duets, backing vocals): Whisper may merge overlapping voices into a single line. When the treatment specifies multiple characters singing, note the overlap in the `Detailed Notes` column and mark the dominant character per beat. If both characters share a line, use `"Both"` as the dominant character and describe the interaction (e.g., "harmonizing", "call-and-response").

## Output Format (beat_sheet.md)

```markdown
# Beat Sheet — <Project Name>

## Sections
- **Intro** (0:00 - 0:15): [description]
- **Verse 1** (0:15 - 0:45): [description]
...

## Beat Details

| Time | Duration | Lyrics | Narrative | Emotional | Visual | Energy | Character | Transition From | Transition To | Detailed Notes |
|------|----------|--------|-----------|-----------|--------|--------|-----------|-----------------|---------------|-------|
| 0:00-0:05 | 5.0s | (instrumental) | Opening atmosphere | Anticipation | Wide establishing shot | Low | — | Fade in | Cut | — |
| 0:05-0:12 | 7.0s | "lyrics here" | Introduce protagonist | Longing | Medium close-up, singer | Medium | Singer | Cut | Dissolve | — |
```

## FSM Update

On approval, transition `BEATS` -> `STORYBOARD`. Use `mv_fsm_cli.py transition <project_dir> <stage> APPROVED` to update FSM state. Do not implement inline state updates.
