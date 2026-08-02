---
name: music-video-prompts
description: Generate keyframe, motion, negative, camera prompts in LTX VRDG prose style from treatment + continuity + storyboard + shot list + approved keyframe (HuMo formula available as fallback)
allowed-tools: [Read, Write, Edit]
---

# Prompt Generation Skill (Stage 7)

Generates four prompt types per beat (keyframe, motion, negative, camera) by consuming
all accumulated creative documents including the storyboard. Prompts are engine-formatted
(HuMo formula vs LTX VRDG prose) and include storyboard context injection.

The PROMPTS stage has two sub-stages that run in order:

1. **Keyframe Prompt Recording** — Record the final approved Keyframe Prompt text into `prompts/beat_{NN}.md` files. These prompts were originally composed during IMAGE_APPROVAL (Step 4 of the reference-images skill) and used to generate the keyframes. This sub-stage captures them as durable artifacts for downstream reference and potential revision. This step does NOT generate keyframe prompts de novo.
2. **Motion Prompts** — Compose temporal animation descriptions (for LTX clip generation). These require the approved keyframe. The keyframe defines the initial frame (identity, wardrobe, environment, lighting, framing, composition, pose, expression); the motion prompt describes only what changes over time. See Golden Rules below.

## Prerequisites

- FSM state must be `PROMPTS`.
- All prior artefacts approved: treatment, continuity, beat sheet, storyboard, shot list.
- For Motion Prompts sub-stage: approved keyframes in `approvals/` with `manifest.json` in `refs/` (required — motion prompts describe change from the keyframe's initial state).

## Input

Read ALL accumulated artefacts from the project directory:

1. `director_treatment.md` — visual concept, style guide
2. `continuity_bible.md` — character, wardrobe, lighting, colour, environment rules
3. `storyboard.md` — narrative importance, visual weight, performance focus, emotional/visual intensity, viewer focus, shot scale, camera movement, coverage strategy, rationale
4. `shot_list.md` — cinematography parameters per beat
5. `refs/manifest.json` — approved keyframe paths with visual weight

## Process

### Step 1: Parse All Artefacts

Parse `storyboard.md` using `mv_beats.parse_markdown_table()` (deterministic, NOT LLM-based).
Parse `shot_list.md` similarly. Load `refs/manifest.json` for approved keyframe paths.

### Step 2: Generate Four Prompt Types Per Beat

For each beat, generate:

**1. Keyframe Prompt** — Text-to-image / image-to-image prompt for QEI keyframe generation.
Describes the scene as a static composition: framing, pose, camera angle, character placement,
wardrobe, environment, lighting, weather state. This prompt feeds QEI to produce the keyframe.
It does NOT describe temporal motion. Incorporates storyboard's primary viewer focus and rationale.

**16:9 output requirement:** QEI inherits dimensions from the input reference image. Ensure all
reference images are canvas-expanded to 16:9 before QEI (see `music-video-reference-images`
Step 2). The resulting keyframe MUST be 16:9 — LTX video generation outputs at 16:9, and
portrait keyframes cause compositional artifacts.

**2. Motion Prompt** — Image-to-video prompt describing temporal motion for the clip.
Describes what happens OVER TIME: camera movement, character body motion, facial animation,
environmental motion. This prompt does NOT control composition or framing — those are
determined by the approved keyframe (ref-image dominance). See the `ltx-prompting` skill
for the passivity token blacklist and motion verb recommendations, and the Golden Rules
below for QEI-first workflow guidance.

**Key architectural principle:** QEI determines where the shot starts. LTX determines
how the shot evolves. The motion prompt should NOT re-describe the static content of the
keyframe (identity, wardrobe, environment, lighting, framing, composition, pose, expression).
It should describe what CHANGES during the shot.

Incorporates camera movement from shot list, character action from storyboard rationale,
energy level from storyboard visual_energy.

- **Default (LTX-2.3):** VRDG prose style — camera as grammatical subject, multi-sentence narrative, couples motion with what it reveals. Use "singing with passion" (NOT "lip syncing naturally").
- **HuMo override (if segment annotated `engine: humo`):** Use the HuMo prompt formula (see HuMo Override section below).

**3. Negative Prompt** — What to avoid. Derived from continuity bible (e.g., "no red
clothing" if wardrobe rules say white dress). Prevents visual inconsistencies.

**4. Camera Prompt** — Explicit camera movement direction.

- **Default (LTX-2.3):** VRDG prose style — camera movement woven into the narrative prose.
- **HuMo override:** Camera motion phrase appended at end of prompt.

### Step 3: Storyboard Context Injection

Every prompt incorporates storyboard data:

- **`performance_focus`** determines prompt emphasis:
  - High `singer_pct` -> singer-centric action description
  - High `narrative_pct` -> story-driven action
  - High `broll_pct` -> environmental focus
  - High `environment_pct` -> landscape/atmosphere emphasis
- **`symbolic_pct`** — when >0%, inject abstract/metaphorical visual elements matching the treatment's symbolism section
- **`montage_pct`** — when >0%, apply rapid-cut layout modifiers and sequence multiple visual concepts
- **`recommended_shot_duration_strategy`** applies layout-specific modifiers:
  - `Single` -> one cohesive scene
  - `Two-Shot` -> A/B cut with match-on-action
  - `Fast Montage` -> staccato rhythm, multiple subjects
  - `Slow Cinematic` -> long-take language, minimal edits
- **`emotional_intensity`** modulates prompt language intensity (subtle vs dramatic descriptors)
- **`visual_energy`** modulates motion prompt dynamism (static vs sweeping movement)
- **`primary_viewer_focus`** ensures the prompt keeps attention on the right element
- **`rationale`** informs the creative direction of the prompt without being copied verbatim

### Step 4: Continuity Bible Injection (Motion Prompts)

The approved keyframe already encodes the character's identity, wardrobe, and environment.
Motion prompts do NOT need to re-describe static appearance. Only include continuity
information that affects HOW the character or environment moves over time:

- **Wardrobe behavior in motion:** Hair moves in wind, wet fabric clings and shifts, loose
  straps swing. These details tell LTX how clothing animates, not what it looks like.
- **Prop interaction:** Flask in hand, backpack on shoulders, phone in fingers. Only mention
  props that the character actively uses during the shot.
- **Weather state for animation:** Rain falls, mist drifts, wind moves foliage. The keyframe
  captures a frozen instant; the prompt tells LTX how weather evolves.

Do NOT prepend a full character + clothing description string to every motion prompt.
The keyframe is the continuity anchor. The prompt adds motion on top of it.

### Step 5: Format Prompts (LTX VRDG Prose Default)

All prompts use LTX VRDG prose style by default. This is the proven format — extensive A/B testing showed LTX VRDG prose produces superior camera motion, identity preservation, and composability.

**LTX VRDG Prose Rules:**
- Camera as grammatical subject ("the camera pushes in slowly...")
- Multi-sentence narrative structure
- Couple camera movement with what it reveals
- Use "singing with passion" for singer clips (NOT "lip syncing naturally")
- **Describe change, not state.** The approved keyframe encodes the initial frame
  (identity, wardrobe, props, environment, lighting, framing, pose, expression).
  The motion prompt describes what happens over time. Never re-describe what the
  keyframe already shows. See Golden Rules below for the full QEI-first workflow.

**HuMo Override (when segment specifies `engine: humo`):**
Follow the HuMo prompt formula: `[Shot Type] [Character Description] [Action] [Environment] [Framing] [Camera Motion]`

### Step 6: Write Prompt Files

Write one file per beat: `prompts/beat_{NN}.md`

```markdown
# Beat {NN} — {time range}

**Lyrics**: "{lyrics}"
**Section**: {from storyboard}
**Visual Weight**: {from storyboard}
**Shot Type**: {from shot list}
**Approved Keyframe**: approvals/Beat_<ID>/candidate_<approved>.png
**Audio**: {silence | (omit if singer shot — music is default)}

## Keyframe Prompt
{keyframe prompt text}

**Source ref**: {path to canonical portrait or continuity-state variant}
**Target framing**: {XW/Wide/Full/Med/MCU/CU/ECU}
**Lens**: {focal length}
**Weather**: {weather state}
**Lighting**: {lighting state}

## Motion Prompt
{motion prompt text}

## Negative Prompt
{negative prompt text}

## Camera Prompt
{camera prompt text}

## Storyboard Context
**Narrative Importance**: {from storyboard}
**Performance Focus**: Singer {pct}%, Narrative {pct}%, B-roll {pct}%, Environment {pct}%, Symbolic {pct}%, Montage {pct}%
**Shot Duration Strategy**: {from storyboard}
**Emotional Intensity**: {1-5}
**Visual Energy**: {1-5}
**Viewer Focus**: {from storyboard}
**Rationale**: {from storyboard}

## Continuity Rules (injected)
{relevant rules from continuity bible}
```

### Audio Conditioning Field

The `**Audio**` field controls whether music audio is passed to LTX during clip
generation. This determines whether the character's mouth animates (lip-sync).

- **Omit the field** (or leave blank) for SINGER shots — music audio is the default,
  LTX receives the audio track and animates mouth movement.
- **`**Audio**: silence`** for all non-singing shots — the generator passes `None`
  as audio to LTX, preventing unwanted mouth movement.

The audio conditioning decision is made during shot planning (Stage 5 / SHOTS).
See the `music-video-shots` skill's Audio Conditioning Decision section for the
decision table. Do not decide audio conditioning here — inherit it from the shot
plan.

### Step 7: Dashboard Integration

When the user pastes the JSON block from `approval_dashboard.html`:

1. Parse the JSON array
2. For each beat entry: record the approved candidate path in `refs/manifest.json`
3. If `revised_*_prompt` fields are present and non-empty, apply them as overrides to the generated prompts. If empty, use the originally generated prompt.
4. Write the final `prompts/beat_{NN}.md` files with the approved/overridden prompts
5. Update FSM state

## Golden Rules for LTX Motion Prompts (QEI-First Workflow)

In the MV pipeline, every shot has an approved QEI keyframe that defines the initial
frame. LTX is responsible only for temporal animation. These rules govern how motion
prompts are written given that architectural fact.

1. **Describe change, not state.** The keyframe encodes identity, wardrobe, props,
   environment, lighting, framing, composition, character positions, expression, and
   pose. Your prompt describes what happens over time. If a detail does not change
   during the shot, it does not belong in the motion prompt.

2. **One character identifier is enough.** Start with a bare name or role reference
   ("Character A", "Character B", "the two figures"). Do not re-describe clothing, hair
   style, or physical appearance. LTX reads these from the keyframe.

3. **Lead with motion, not scene.** The first sentence should describe movement or
   action, not a static scene description. The keyframe IS the scene description.
   Opening with "Close-up on the character's face in the rain" wastes tokens the keyframe
   already provides.

4. **Weather is for animation, not decoration.** Mention weather only when it creates
   visible motion: "rain streaks across her face", "mist drifts across the valley",
   "wind moves through her hair". Do not re-state weather that is frozen in the
   keyframe without describing how it moves.

5. **Camera movement is mandatory when the shot list specifies it.** Use the LTX-safe
   phrases from the storyboard's LTX reliability assessment (e.g., "camera slowly
   pushes in", "camera dollies laterally alongside"). Couple camera movement with
   what it reveals, per VRDG prose style.

6. **Static shots need micro-movement.** When the shot list calls for a static camera
   on a face, the prompt MUST include natural micro-movement language to avoid
   frozen-face artifacts. See the `ltx-prompting` skill's passivity token blacklist
   for phrases that cause frozen-face output and their rewrites. Use "natural blink
   cadence, eyes tracking" and "subtle micro-expressions shifting" instead of
   "holds gaze" or "still expression".

7. **Singer clips: "singing with passion", never "lip syncing".** Describe the
   emotional quality of the vocal performance. LTX animates mouth movement from
   the audio input; the prompt guides expression, not phonetics.

7a. **Non-singer clips: ensure `**Audio**: silence` is set.** When the shot
    contains a character who is NOT singing, the prompt file MUST include
    `**Audio**: silence` in its metadata header. Without this flag, LTX receives
    the music audio and will animate the character's mouth, producing an
    incorrect lip-sync effect. This is not optional — every NARRATIVE,
    CONVERGENCE, and BROLL shot requires the silence flag.

8. **Two-character shots: animate both.** When the keyframe contains two characters,
   describe what each one does. "Character A's shoulders ease, Character B's eyes soften" not
   "they look at each other". Specific micro-actions per character produce better
   temporal animation than generic group descriptions.

9. **Environment-only shots: describe what moves.** For landscape/broll shots with
   no character reference, the prompt IS the scene description (no keyframe to
   inherit from). Include environment + what moves: clouds gathering, water flowing,
   mist drifting. These are the exception to Rule 1 — no keyframe means you must
   describe the scene.

10. **Keep it tight: 40-80 words for character shots.** With the keyframe handling
    static description, motion prompts can be shorter than the `ltx-prompting` skill's
    general 80-120 word recommendation. Every word should drive motion. The 200-word
    hard ceiling still applies (see `ltx-prompting` skill). Environment-only shots
    may need slightly more words since they lack a keyframe anchor.

## Prompt Extraction Rules for Stage 9 Generation

The generation skill (Stage 9) reads `prompts/beat_{NN}.md` files and extracts prompts:

- **Keyframe generation (Stage 6):** Feed the "Keyframe Prompt" to QEI for keyframe generation
- **Clip generation (Stage 9 video clips):** Combine "Motion Prompt" + "Camera Prompt" into a single string: `"[Motion Prompt]. [Camera Prompt]"` — concatenated with a period and space separator
- **Negative Prompts:** Tracked as metadata. Used to ensure no contradictions. NOT concatenated with motion/camera prompts — passed as a separate negative prompt parameter

## Output

- `prompts/beat_{NN}.md` — one file per beat, containing all 4 prompt types plus storyboard context and continuity rules

## FSM Update

On completion, transition `PROMPTS` -> `VALIDATED`.
Use `mv_fsm_cli.py transition <project_dir> <stage> APPROVED` to update FSM state.
Do not implement inline state updates.
