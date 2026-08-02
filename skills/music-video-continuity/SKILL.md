---
name: music-video-continuity
description: Create continuity bible — character, wardrobe, time, weather, lighting, camera, colour, environment rules
allowed-tools:
  - Read
  - Write
  - Edit
---

# /music-video-continuity

Create the continuity bible — the persistent reference document injected into every downstream prompt. This is Stage 3 of the FSM workflow (CONTINUITY).

## FSM Prerequisites

- FSM state must be `CONTINUITY`.
- `director_treatment.md` exists and is `APPROVED` (immutable).

## Input

Read both:
- `director_treatment.md` — the immutable creative authority.
- `project.md` — the raw creative interview input.

## Output: `continuity_bible.md`

Write `continuity_bible.md` — the continuity reference document. Include ALL 14 PRD categories:

```markdown
# Continuity Bible — [Project Name]

## 1. Character Appearance Descriptions
Physical description, age, ethnicity, build, distinguishing features.
Must be consistent across ALL clips.

## 2. Wardrobe Definitions
Exact clothing description per scene/time.
What changes and what stays constant.

## 3. Hair & Makeup Specifications
Hair style, color, length.
Makeup details: natural, dramatic, none.
Per character.

## 4. Accessories & Props Lists
Jewelry, bags, watches, glasses.
Key objects characters interact with.
Per scene.

## 5. Time of Day Constraints
When does each section take place?
Lighting implications per time period.

## 6. Weather Configurations
Rain, fog, clear, etc.
Per scene.

## 7. Season Context
What season?
Environmental indicators: foliage, snow, humidity, light quality.

## 8. Lighting Setup Defaults
Warm/cool, natural/artificial, key light direction.
Default lighting approach per scene type.

## 9. Camera and Lens Language Conventions
Lens choices, framing preferences, movement style.
Derived from treatment.

## 10. Film Look Characteristics
Grain, contrast, color grading style, film stock emulation.
The overall photographic aesthetic.

## 11. Colour Palette Mappings
Palette, saturation, contrast.
Per scene/section.

## 12. Environment Rules and Constraints
Setting details, background elements, spatial relationships.
What must appear in every shot of a given location.

## 13. Invariances
Elements that must NEVER change across the entire video.
Examples: character's eye color, specific prop, core outfit elements.

## 14. Variations
Elements that INTENTIONALLY evolve over the course of the video.
Examples: lighting gets warmer, clothing gets disheveled, weather changes.
```

## Prompt Injection

This file is INJECTED into every prompt generated at Stage 7 (Prompt Generation). It is the single source of truth for visual consistency.

Downstream prompt generation reads `continuity_bible.md` and prepends its rules to every segment prompt. The continuity bible ensures that every AI-generated clip follows the same character descriptions, wardrobe, color palette, and environment rules.

Document this injection mechanism in the skill so downstream stages know to consume this file.

## FSM Transition

After user approves `continuity_bible.md`:

```bash
python scripts/mv_fsm_cli.py transition <project_dir> CONTINUITY APPROVED
```

This transitions CONTINUITY -> BEATS in the FSM state machine. The continuity bible is a prerequisite for the Storyboard stage (Stage 4.5) which consumes its rules for visual classification.

Do not implement inline state updates. Use the FSM CLI tool.

## Continuity-State Variants

When a character's appearance changes over the course of the video (e.g., gets wet in rain, changes clothes, hair becomes disheveled), document these as continuity-state variants in the Variations section (category 14). These variants may require persistent continuity-state portrait assets.

A continuity-state variant is a persistent asset (not a per-shot keyframe) when:
- The state change affects multiple shots across different segments
- Consistency of the state (wetness level, clothing behavior, hair state) matters across those shots

Example: "Ishi wet-state" — ivory chiffon dress, visibly damp and clinging slightly, wet hair (same style, dampened), natural makeup washed slightly by rain. This single variant covers all rain shots for Ishi, ensuring consistent wetness level.

## Key Design Principles

- **Complete coverage**: All 14 categories must be present. No gaps.
- **Specific, not vague**: "warm amber light" not "nice lighting". "white cotton dress, knee-length" not "casual outfit".
- **Traceable to treatment**: Every rule must be derivable from `director_treatment.md`. Do not invent new creative directions.
- **Actionable for prompts**: Each rule must be usable as a constraint in AI image/video generation prompts.
- **Invariances vs Variations**: Clearly distinguish what stays the same from what intentionally changes. This prevents the AI from either freezing everything or drifting randomly.
- **Wardrobe must be specific**: The wardrobe definition must be exact (color, garment type, fabric). Vague definitions ("earth tones", "natural colors") cause inconsistent outputs across QEI generations. See the `music-video-reference-images` skill Step 3 (Validate Wardrobe Lock).
