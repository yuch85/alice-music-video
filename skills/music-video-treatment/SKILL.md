---
name: music-video-treatment
description: Create director's treatment from interview output — immutable after approval
allowed-tools:
  - Read
  - Write
  - Edit
---

# /music-video-treatment

Create the director's treatment document from the creative interview output. This is Stage 2 of the FSM workflow (TREATMENT).

## FSM Prerequisites

- FSM state must be `TREATMENT`.
- `project.md` exists and is `APPROVED`.

## Input

Read `project.md` (the creative interview output from Stage 1).

## Output: `director_treatment.md`

Write `director_treatment.md` — a polished director's treatment document. Exactly 13 sections, in this order:

```markdown
# Director's Treatment — [Project Name]

## 1. Theme
Core thematic statement. What is this video fundamentally about?

## 2. Story
What literally happens — beginning, middle, ending, twist (if any).
A clear narrative arc from start to finish.

## 3. Characters
Who appears, their role, appearance, personality.
How each character serves the story.

## 4. Visual Philosophy
Overall visual language and aesthetic.
The visual grammar that every shot must follow.

## 5. Camera Philosophy
Movement style, framing approach, lens language.
How the camera behaves as a storytelling tool.

## 6. Editing Philosophy
Cutting rhythm, transition style, temporal structure.
How scenes connect and flow.

## 7. Colour Progression
How color evolves through the video.
From opening palette to closing palette.

## 8. Lighting Progression
How lighting evolves through the video.
Quality, direction, and mood of light over time.

## 9. Music Video Style
Genre/style classification.
Performance, narrative, abstract, or hybrid.

## 10. Location Guide
Where each section takes place.
Environment details and spatial relationships.

## 11. Performance Direction
How the singer performs.
Acting notes, emotional delivery, physical presence.

## 12. Ending
How the video concludes.
Final image, fade, resolution.

## 13. Creative Intent
Director's personal vision.
What this video is trying to achieve beyond the literal story.
```

## Immutability Enforcement

After user approval, `director_treatment.md` is LOCKED. All downstream stages (continuity, beats, storyboard, shots, prompts) MUST reference the treatment. No stage may contradict it.

### Two-Layer Immutability

**Primary enforcement (validation error):** In any downstream skill or module, check the FSM state before any write to `director_treatment.md`. If the treatment is `APPROVED`, raise a validation error before the operation. This is the primary gate — checked by the FSM before any write.

**Secondary safety net (chmod 444):** Set file permissions to read-only (`chmod 444 director_treatment.md`) after approval as a secondary safety net. The validation error is the authoritative enforcement; chmod is defense-in-depth.

Document this two-layer enforcement mechanism in all downstream skills.

### Rejection Flow

If the user rejects the treatment:
1. Present specific sections for revision.
2. Revise based on feedback.
3. Re-present for approval.
4. Do NOT transition FSM until approved.

## FSM Transition

After user approves `director_treatment.md`:

```bash
python scripts/mv_fsm_cli.py transition <project_dir> TREATMENT APPROVED
chmod 444 <project_dir>/director_treatment.md
```

This transitions TREATMENT -> CONTINUITY in the FSM state machine. Mark `director_treatment.md` as immutable.

Do not implement inline state updates. Use the FSM CLI tool.

## Key Design Principles

- **Derived from interview**: Every section in the treatment must be traceable to `project.md`. Do not invent new creative directions.
- **Polished, not raw**: The treatment is a professional document — refined language, clear structure, actionable direction.
- **Immutable after approval**: Once approved, this document is the creative authority for all downstream stages.
- **Referenceable**: Downstream stages (continuity bible, beat sheet, storyboard, shot list, prompts) must cite specific sections of the treatment.
