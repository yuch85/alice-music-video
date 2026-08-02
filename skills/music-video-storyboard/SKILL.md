---
name: music-video-storyboard
description: Create visual storyboard — classify each beat with importance, weight, focus, intensity, shot scale, camera movement, coverage strategy
allowed-tools: [Read, Write, Edit, Bash]
---

# Visual Storyboard Skill (Stage 4.5)

## Prerequisites

- FSM state must be `STORYBOARD`.
- `beat_sheet.md` exists and is `APPROVED`.

## Input

Read `beat_sheet.md` + `director_treatment.md` + `continuity_bible.md`.

## Process

1. Parse beat sheet entries using `mv_beats.parse_markdown_table()`
2. Classify song structure (intro/verse/pre-chorus/chorus/bridge/instrumental/outro) via `mv_storyboard.classify_song_structure()`
3. For each beat, compute: narrative importance, visual weight, performance focus, emotional intensity, visual energy, primary viewer focus, recommended shot scale, camera movement intensity, coverage strategy
4. Use LLM to generate rationale for each beat explaining why this visual approach best supports the lyric
5. Write `storyboard.md` — structured markdown table

## Output Format (storyboard.md)

```markdown
# Visual Storyboard — <Project Name>

## Song Structure
- **Intro** (0:00 - 0:15): 2 beats, avg weight 3
- **Verse 1** (0:15 - 0:45): 4 beats, avg weight 5
- **Pre-Chorus** (0:45 - 1:00): 2 beats, avg weight 7
- **Chorus** (1:00 - 1:30): 4 beats, avg weight 9
...

## Storyboard Details

| Beat | Time | Section | Importance | Weight | Singer% | Narrative% | B-roll% | Env% | Symbolic% | Montage% | Emo | Energy | Focus | Duration | Scale | Camera | Coverage | Rationale |
|------|------|---------|------------|--------|---------|------------|---------|------|-----------|----------|-----|--------|-------|----------|-------|--------|----------|-----------|
| B01 | 0:00-0:05 | Intro | Low | 3 | 20 | 10 | 40 | 30 | 0 | 0 | 2 | 2 | Landscape | Single | Wide | Locked | Single take | Establishing shot sets mood before singer appears |
| B05 | 1:00-1:08 | Chorus | Critical | 9 | 70 | 10 | 5 | 10 | 5 | 0 | 5 | 5 | Eyes | Single | CU | Dynamic | Hero moment — emotional peak demands intimate close-up |
```

## Output Fields

The validation skill (`music-video-validation`, lines 34-38) requires 15 fields per beat.
Every row in `storyboard.md` must include all of them:

| Field | Description | Valid Values |
|-------|-------------|--------------|
| `Beat` | Unique beat identifier | `B01`, `B02`, … |
| `Time` | Start-end timestamp | `0:00-0:05` |
| `Section` | Song structure section | `Intro`, `Verse 1`, `Pre-Chorus`, `Chorus`, `Bridge`, `Instrumental`, `Outro` |
| `Importance` | Narrative importance | `Low`, `Medium`, `High`, `Critical` |
| `Weight` | Visual weight — drives keyframe candidate count | `1-10` integer |
| `Singer%` | % of frame devoted to singer performance | `0-100` (see performance_focus below) |
| `Narrative%` | % devoted to story-driven visual content | `0-100` |
| `B-roll%` | % devoted to supplementary B-roll imagery | `0-100` |
| `Env%` | % devoted to environment/landscape | `0-100` |
| `Symbolic%` | % devoted to abstract/metaphorical visuals | `0-100` |
| `Montage%` | % devoted to rapid-cut montage sequences | `0-100` |
| `Emo` | Emotional intensity | `1-5` integer |
| `Energy` | Visual energy level | `1-5` integer |
| `Focus` | Primary viewer focus | `Eyes`, `Face`, `Upper body`, `Full body`, `Landscape`, `Object` |
| `Duration` | Recommended shot duration strategy | `Single`, `Two-Shot`, `Fast Montage`, `Slow Cinematic` |
| `Scale` | Primary shot scale | `XW`, `Wide`, `Full`, `Med`, `MCU`, `CU`, `ECU` |
| `Camera` | Camera movement intensity | `Locked`, `Subtle`, `Moderate`, `Dynamic` |
| `Coverage` | Coverage strategy | `Single take`, `A/B cut`, `Multi-angle`, `Montage` |
| `Rationale` | Why this visual approach serves the lyric | Free text (see quality guidance below) |

**`performance_focus` breakdown:** `Singer%`, `Narrative%`, `B-roll%`, `Env%`, `Symbolic%`, and `Montage%` must sum to 100 for each beat.

**`camera_movement_intensity` scale:** `Locked` (no movement), `Subtle` (micro-adjustments, slow drift), `Moderate` (visible push-in/dolly), `Dynamic` (active tracking, crane, whip pan).

**`coverage_strategy` options:** `Single take` (one continuous shot), `A/B cut` (two angles cut on action), `Multi-angle` (3+ angles for complex beats), `Montage` (rapid sequence of distinct images).

## Classification Heuristics

Use these rules of thumb when classifying beats. The continuity bible provides environment and lighting context that should inform every classification decision — e.g., a beat set in rain should reflect that in its visual energy and coverage strategy.

- **Choruses:** Strong singer performance focus (60-80% singer), close-ups (CU/MCU), higher emotional intensity (4-5), dynamic camera movement. The chorus is the emotional peak — the visuals should match.
- **Verses:** More narrative and environmental storytelling (lower singer %, wider shots). Moderate camera movement. Emotional intensity 2-3.
- **Pre-Choruses:** Transitional — build intensity from the verse toward the chorus. Increasing weight, moderate-to-dynamic camera. Performance focus shifts from narrative toward singer.
- **Bridges:** Emotional contrast — use a unique shot scale and camera movement that differs from the surrounding sections. If the verses are wide and the chorus is tight, the bridge might be a slow cinematic single take.
- **Intros/Outros:** Environment-heavy (high Env%), locked or subtle camera. Establishing shots for intro, resolving imagery for outro.
- **Instrumental sections:** B-roll and environment focus. No singer % — use the time for visual storytelling through landscape, detail shots, or symbolic imagery.
- **Visual weight** directly drives candidate image counts in the IMAGE_APPROVAL stage: weight 1-3 = 2-3 candidates, 4-6 = 3-4, 7-10 = 5-8 candidates.

## Rationale Quality Guidance

The rationale field explains why each visual approach best supports the lyric. A good rationale:

- **References the specific lyric** — quote or paraphrase the line, don't speak generally.
- **Connects lyric to visual choice** — "the lyric 'walking through fire' suggests a warm color palette and forward camera motion" not "this is a dramatic moment".
- **Justifies the performance focus split** — why 70% singer and 30% environment for this beat?
- **Is 1-2 sentences** — concise but substantive.

A bad rationale:

- "This is an important beat" (no connection to lyric or visual choice)
- "The singer is singing" (restates the obvious, no reasoning)
- "Close-up because it's the chorus" (generic rule, not beat-specific)

## FSM Update

On approval, transition `STORYBOARD` -> `SHOTS`. Use `mv_fsm_cli.py transition <project_dir> <stage> APPROVED` to update FSM state. Do not implement inline state updates.

## `concepts/` Directory

The storyboard skill stores intermediate visual concept drafts in `concepts/` before finalizing `storyboard.md`. These are working drafts (e.g., `concepts/beat_B05_draft_v1.md`) that capture exploratory visual directions before the final storyboard entry is locked. They are not required artefacts — `storyboard.md` is the canonical output — but they provide an auditable trail of creative decisions.

## Batch LLM Queries

When using the LLM for rationale generation, batch queries into sections of 10-15 beats and request JSON array structures from the local LLM endpoint to prevent timeouts and token window spills. Do not query one beat at a time. Use `extract_and_repair_json()` to parse batch LLM responses.
