---
name: music-video-shots
description: Create cinematography shot list from storyboard — camera, lens, composition, movement, lighting, mood per beat
allowed-tools: [Read, Write, Edit, Bash]
---

# Shot List Skill (Stage 5)

Translates storyboard visual metadata into concrete cinematography assignments.
The shot list no longer reads the beat sheet directly — it consumes the storyboard's
recommended shot scale, camera movement intensity, primary viewer focus, visual
weight, and coverage strategy, and maps those into operational cinematography plans.

## Prerequisites

- FSM state must be `SHOTS`.
- `storyboard.md` exists and is `APPROVED`.
- `continuity_bible.md` exists (for camera rules and lighting reference).
- `director_treatment.md` exists (for camera philosophy and visual style).

## Input

Read `storyboard.md` + `continuity_bible.md` + `director_treatment.md`.

## Process

1. **Parse storyboard entries** — Use `parse_markdown_table` from `mv_beats.py`
   (import it) to parse `storyboard.md`, NOT LLM-based parsing. This ensures
   deterministic, correct parsing of user-edited markdown tables. Each row carries:
   recommended shot scale, camera movement intensity, primary viewer focus,
   visual weight, coverage strategy, recommended shot duration strategy, and rationale.

2. **Check coverage indicators** — For each beat, examine the storyboard's
   `recommended_shot_duration_strategy` and `coverage_strategy`. If the strategy
   is `Two-Shot`, `Fast Montage`, or the coverage strategy describes splits,
   generate an alternative list of shot assignments covering each sub-segment.
   Single-beat shots get one primary assignment.

3. **Translate storyboard recommendations into all 10 cinematography parameters**:

   - **Primary Shot**: Map storyboard's `recommended_shot_scale` to concrete shot type.
     ECU -> Extreme Close-up (eyes, lips). CU -> Close-up (face). MCU -> Medium
     Close-up (head and shoulders). Medium -> Waist-up. Full -> Full body.
     Wide -> Wide shot showing environment. Extreme Wide -> Establishing landscape.

   - **Camera Settings**: Sensor size implication, shutter speed feel, ISO feel.
     Examples: "full-frame, clean low-ISO look" or "APS-C, slight grain for texture".

   - **Lens Selection**: Focal length and character. Close viewer focus (eyes, face)
     -> 85mm telephoto compression. Medium shots -> 50mm normal. Wide environment
     -> 24mm wide-angle. Environment sweep -> 16mm ultra-wide.

   - **Composition Structure**: Derive from coverage strategy and viewer focus.
     Single subject -> rule of thirds. Symmetrical scene -> center-framed.
     Leading lines for tracking shots. Negative space for emotional isolation.

   - **Camera Movement Detail**: Map storyboard's `camera_movement_intensity` with
     specifics. Locked -> Static tripod. Push -> Slow motorized push-in (~2cm/s).
     Tracking -> Lateral dolly tracking. Crane -> Vertical crane rise.
     Orbit -> 360 arc around subject. Dynamic -> Handheld with purposeful
     micro-movement.

   - **Lighting Style**: From continuity bible lighting rules + emotional intensity.
     High intensity -> dramatic key/fill ratio (2:1 or higher). Low intensity ->
     soft even lighting (1:1). Directional notes (front-lit, backlit, side-lit).

   - **Framing Method**: How the subject is placed within the frame. Center-framed,
     rule of thirds, headroom/leadroom considerations, eye-level vs high/low framing.

   - **Foreground/Background Setup**: What appears in the foreground (blur, elements
     framing the subject) and background (environment depth, bokeh quality,
     spatial context).

   - **Mood Target**: From storyboard's emotional intensity + rationale. Translate
     emotional quality into visual mood (e.g., "intimate and warm", "cold and
     isolating", "energetic and electric").

   - **Cinematic Reasoning**: Why this specific setup supports the beat. References
     the storyboard's rationale and the beat's narrative/emotional purpose.
     Explains the creative decision.

4. **Provide exactly three options per beat**:
   - **Primary**: The main recommendation derived directly from storyboard data.
   - **Alternative A**: A different angle/composition that still serves the beat.
   - **Alternative B**: A contrasting approach offering creative variety.
   - **Alternative C**: A third distinct option (e.g., different lens, framing,
     or movement combination).
   All three alternatives must be distinct from each other and from the primary.

5. **Enforce shot variety** — No two consecutive beats should have identical
   shot types unless the storyboard rationale explicitly calls for it.

6. **Cross-reference continuity bible** — Shot list must comply with camera rules,
   lens conventions, and lighting defaults defined in the continuity bible.

## Audio Conditioning Decision

Every shot requires an audio conditioning decision. This is a first-class property
of the shot, decided alongside lens, framing, and DOF — not an afterthought.

The generation pipeline reads `**Audio**` from each prompt file. When set to
`silence`, the clip is generated without audio conditioning (no music passed to
LTX), which prevents lip-sync mouth movement on non-singing characters.

**Default behavior:** Music audio is passed to LTX (lip-sync ON). SINGER shots
do not need an explicit `**Audio**` field — music conditioning is the default.

**Silence flag:** Add `**Audio**: silence` to any shot where the character(s) are
NOT singing. This tells the generator to pass `None` as audio to LTX.

### Decision Table

| Shot Type | Audio Conditioning | Flag Required? | Reason |
|-----------|-------------------|----------------|--------|
| SINGER | Music (default) | No — default | Character is singing, lip-sync needed |
| NARRATIVE (face visible CU/MCU) | Silence | Yes | Character not singing, prevent mouth movement |
| NARRATIVE (rear view / wide / face not visible) | Silence | Yes | Belt-and-suspenders safety |
| CONVERGENCE (faces visible at medium range) | Silence | Yes | Neither character is singing |
| CONVERGENCE (distant figures) | Silence | Yes | Safety |
| BROLL | Silence | Yes | No characters present |
| Prologue (SH-001..006) | Ambient | Handled by code | No music yet in timeline |
| Epilogue (e.g. SH-047) | Ambient | Handled by code | Rain ambience, no music |
| TTS shots (e.g. SH-048) | TTS only | Handled by code | Spoken dialogue, separate path |

### How to Apply

For each shot in the shot list, determine the shot type based on:
1. **Who is in frame** — singer, narrative character, both, or no one
2. **Is the face visible** — CU/MCU makes mouth movement obvious; wide shots
   are less sensitive but still require the flag for consistency
3. **Is anyone singing** — only the designated singer sings

Then add `**Audio**: silence` to the prompt file for any non-singing shot.
The flag goes in the shot metadata header, alongside Duration and Weather:

```
**Weather**: Dry, natural daylight
**Duration**: 6s
**Audio**: silence
```

## Storyboard-to-Shot Mapping Rules

- Storyboard's `recommended_shot_scale` is the PRIMARY input for shot type assignment.
- Storyboard's `camera_movement_intensity` is the PRIMARY input for movement assignment.
- Storyboard's `primary_viewer_focus` influences angle and lens feel.
- Storyboard's `coverage_strategy` determines alternative differentiation (single
  take = unified approach, multiple angles = varied perspectives, montage =
  rapid-cut variety).
- Storyboard's `visual_weight` is recorded in shot list for downstream reference
  generation.
- Storyboard's `rationale` is NOT copied verbatim — it INFORMS the cinematography
  choices.

## Output Format (shot_list.md)

```markdown
# Shot List — <Project Name>

## Shot Assignments

### Beat B01 (0:00-0:05) — Intro, Weight 3

**Primary**: Extreme wide | Full-frame | 16mm ultra-wide | Symmetry | Static crane rise | Cool dawn, front-lit | Center-framed | Open landscape background, shallow foreground | Anticipation | Establishing shot sets mood before singer appears
**Alternative A**: Wide shot | APS-C | 24mm wide | Rule of thirds | Slow lateral tracking | Cool dawn, side-lit | Subject left-third | Distant mountains, mist foreground | Mystery | Same scene, different spatial relationship
**Alternative B**: Medium wide | Full-frame | 35mm | Leading lines | Static | Cool dawn, backlit | Center-framed | Road leading to horizon | Solitude | Emphasizes isolation through perspective
**Alternative C**: Full body | Full-frame | 50mm | Rule of thirds | Slow push-in | Cool dawn, soft front | Subject right-third | Blurred treeline | Contemplation | Introduces human element early

### Beat B05 (1:00-1:08) — Chorus, Weight 9

**Primary**: Close-up | Full-frame | 85mm telephoto | Rule of thirds | Slow push-in | Warm key, dramatic ratio | Eye-level, face right-third | Soft bokeh, warm background | Intimacy | Hero moment demands intimate connection
**Alternative A**: MCU | Full-frame | 50mm | Center-framed | Static | Warm key, even | Eye-level | Shallow depth of field | Warmth | Slightly more context while maintaining intimacy
**Alternative B**: ECU | Full-frame | 100mm macro | Extreme tight | Micro push | Warm rim light | Eye-level | Complete background blur | Intensity | Maximum emotional proximity
**Alternative C**: CU | APS-C | 35mm | Off-center | Handheld micro | Warm key, side-lit | Slightly low angle | Moderate bokeh | Raw | Grittier, more documentary feel
```

## FSM Update

On approval, transition `SHOTS` -> `IMAGE_APPROVAL`. Use `mv_fsm_cli.py transition <project_dir> <stage> APPROVED` to update FSM state. Do not implement inline state updates.

## Integration Note

The shot list feeds the keyframe generation skill (Stage 6). Each shot's
assignment determines what kind of keyframe is needed (framing,
angle, composition). The shot list carries forward the storyboard's `visual_weight`
for candidate count determination.

**Keyframe requirements by shot scale:**
- MCU and Medium shots: Canonical portrait often works directly (same framing as portrait).
- XW, Wide, Full, ECU shots: Require shot-specific keyframes generated via QEI from the canonical portrait.
- Rain/wet shots: May require continuity-state variants (e.g., wet-state portrait) as the QEI source.
- Two-character shots: QEI generates the complete two-character keyframe (composition, framing, relative positioning). LTX only animates.
- Environment-only shots: No character keyframe needed.

See the `music-video-reference-images` skill for the full keyframe pipeline.
