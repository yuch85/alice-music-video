---
name: music-video-interview
description: Conduct creative interview for music video pre-production (story, theme, visual style, symbolism)
allowed-tools:
  - Read
  - Write
  - Edit
---

# /music-video-interview

Conduct a structured creative interview that produces `project.md` — the raw creative input document for the music video. This is Stage 1 of the FSM workflow (INTERVIEW).

## FSM Prerequisites

- FSM state must be `INTERVIEW`.
- Project folder exists with `index.md`.
- Either raw lyrics (source TXT in `lyrics/`) or `transcript.json` (Whisper-timed lyrics) must exist. If neither exists, prompt the user to provide lyrics or confirm they will describe the story conversationally. Raw lyrics can be auto-converted to `transcript.json` later via Whisper at the Beat Sheet stage, so raw lyrics at init is sufficient.

## Interview Flow

The interview is **CONVERSATIONAL**, not a form. Ask one topic at a time, building on previous answers. Suggest defaults based on lyrics analysis (if lyrics are available in the project folder).

### Pre-Interview

1. Read `lyrics/` content if available. Extract themes, emotions, and narrative elements.
2. Read any existing creative input files (`storyconcept.txt`, `themestyle.txt`, `subjectsandscenes.txt`) for context.
3. Briefly summarize what you found, then begin the interview.

### Interview Topics (12 Topics)

Ask each topic as a conversational question. Wait for the user's response before moving to the next topic. Suggest defaults based on audio/lyrics analysis.

#### 1. Story / Narrative

"What is the music video about? What story does it tell?"

- Derive from lyrics if available.
- Structure: Beginning, Middle, Ending, Twist (if any).
- What literally happens from start to finish?

#### 2. Theme

"What is the emotional core? What feeling should the viewer have?"

- Core thematic elements: Loss, Freedom, Love, Isolation, Memory, Growing up, etc.
- What is the video trying to say?

#### 3. Emotional Arc

"How does the emotion change through the song from beginning to end?"

- Starting emotion (intro/verse 1).
- Midpoint / turning point emotion.
- Concluding emotion (outro).
- How does the viewer's emotional journey progress?

#### 4. Character Arc

"Who are the key characters, and how do they change?"

- Who changes over the course of the video?
- Who learns something?
- Who remains static (anchor, observer, foil)?
- Key relationships between characters.

#### 5. Visual Style

"What is the overall visual language?"

- Cinematic, documentary, surreal, minimalist, etc.?
- Reference films or visual aesthetics.
- Film/anime/photography/director references.
- Music video and painter influences.

#### 6. Color Palette

"What is the color language of the video?"

- Warm/cool, saturated/muted, specific colors?
- Primary color palette and accent highlights.
- How does color evolve through the video?

#### 7. Camera Approach

"How should the camera behave?"

- Handheld/static, close-up/wide preference, movement style?
- Standard movement style: Static, Handheld, Dreamlike, Documentary, Energetic, Formal.
- Lens language (wide-angle intimacy, telephoto compression, etc.).

#### 8. Editing Rhythm

"How should the video be cut?"

- Fast cuts, long takes, match cuts, transitions?
- Cutting philosophy: Slow, Fast, Long takes, Match cuts, Jump cuts, Cross-cutting, Montage.

#### 9. Symbolism / Motifs

"What recurring visual symbols or metaphors should appear?"

- Weather, Nature, Objects, Architecture, Animals, Repeated imagery.
- Visual callbacks that connect beginning to end.

#### 10. Wardrobe

"What do the characters wear and carry?"

- **Clothing** — garments, colors, layers, style.
- **Accessories** — jewelry, bags, watches, glasses.
- **Hair** — style, color, length.
- **Props** — key items characters interact with or carry.

#### 11. Environment / Setting

"Where does the video take place?"

- Indoor/outdoor, urban/natural?
- Specific locations and environments.
- Background elements and spatial relationships.

### Post-Interview

After all 12 topics are covered:

1. Compile all answers into `project.md`.
2. Present `project.md` to the user for review.
3. On approval, transition the FSM:

```bash
python scripts/mv_fsm_cli.py transition <project_dir> INTERVIEW APPROVED
```

Do not implement inline state updates. Use the FSM CLI tool.

## Output: `project.md`

Write `project.md` in the project folder. Structured markdown with sections for each topic:

```markdown
# Creative Interview — [Project Name]

## Story / Narrative
[User's response]

## Theme
[User's response]

## Emotional Arc
[User's response]

## Character Arc
[User's response]

## Visual Style
[User's response]

## Color Palette
[User's response]

## Camera Approach
[User's response]

## Editing Rhythm
[User's response]

## Symbolism / Motifs
[User's response]

## Wardrobe
[User's response]

## Environment / Setting
[User's response]
```

This file is the RAW creative input — it feeds the Treatment stage (Stage 2).

## FSM Transition

After user approves `project.md`:

```bash
python scripts/mv_fsm_cli.py transition <project_dir> INTERVIEW APPROVED
```

This transitions INTERVIEW -> TREATMENT in the FSM state machine. The full FSM cycle is:

INTERVIEW -> TREATMENT -> CONTINUITY -> BEATS -> STORYBOARD -> SHOTS -> IMAGE_APPROVAL -> PROMPTS -> VALIDATED -> GENERATING -> QC -> COMPLETE

## Key Design Principles

- **Conversational**: Ask one topic at a time. Do not present all 12 topics as a form.
- **Suggest defaults**: Use lyrics analysis and existing creative input files to suggest starting points.
- **Build on answers**: Each topic should reference previous answers where relevant (e.g., Wardrobe references Character Arc).
- **Raw input**: `project.md` is raw creative material. It will be refined into `director_treatment.md` at Stage 2.
