---
name: ltx-prompting
description: Rubric for writing effective LTX-2 prompts — paragraph structure, shot vocabulary, emotion rewriting, motion cue injection, NEG_SUFFIX rules, ref-image dominance principle.
allowed-tools:
  - Read
---

# /ltx-prompting

Rubric for composing `scene_prompt` strings passed to `alice_generate_dialogue_clip_ltx2`.
The internal `_prepare_ltx_prompt()` helper applies these rules automatically for code
callers — this skill is for humans composing prompts or auditing what the helper does.

This rubric applies to both dialogue clips and music video motion prompts. For MV
production, see the keyframe pipeline in the `music-video-reference-images` skill:
the approved keyframe determines composition; the LTX motion prompt adds temporal
animation only.

---

## 6-Element Structure

Effective LTX-2 scene prompts follow this order within a single paragraph:

| Element | Description | Example |
|---------|-------------|---------|
| **Shot type** | Framing directive for camera | `Close-up on Alice's face` |
| **Scene description** | Environment, lighting, setting | `in a dim glass-walled conference room, neon ambient` |
| **Character action** | Physical action — what the subject is doing | `She turns to face camera, reaches for a document` |
| **Emotion (as physical cue)** | Internal state expressed physically — NOT bare label | `jaw tightens, eyes narrow` (NOT "angry") |
| **Camera movement** | How the camera moves during the shot | `Camera slow push-in on her face` |
| **Audio/ambient** | What the viewer hears beyond dialogue | `ambient hum of ventilation, footsteps receding` |

All 6 elements are optional, but including all 6 gives LTX-2 the most signal. Lead with
shot type + scene to anchor composition before character details.

**MV exception:** For music video production with approved keyframes, skip elements 1-2
(shot type, scene description) — the keyframe provides these. Lead with element 3
(character action). See the `music-video-prompts` Golden Rule 3 ("Lead with motion, not
scene") for the full MV-specific guidance.

---

## Vocabulary Tables

### Shot Types

| Term | Framing |
|------|---------|
| `close-up` | Face filling most of frame |
| `medium shot` | Waist-up |
| `wide shot` | Full body + environment visible |
| `over-the-shoulder` | Subject B framed over Subject A's shoulder |
| `Dutch angle` | Camera tilted ~15-30° for unease |

### Camera Motion

| Term | Effect |
|------|--------|
| `slow push-in` | Gradual zoom toward subject — intimacy, tension |
| `tracking` | Camera moves laterally with subject |
| `static` | No camera movement — austere, clinical |
| `handheld drift` | Subtle organic sway — documentary feel |

### Motion Verbs (prefer these over static verbs)

**Good — imply motion:** gestures, turns, steps, lifts, lowers, reaches, glances, leans,
tilts, shifts weight, opens, closes, nods.

**Avoid — imply static pose:** stands, waits, holds, sits (without a secondary action),
remains, stays, is.

---

## Emotion → Physical Cue Rewrites

Bare emotion labels generate abstract internal states that LTX-2 cannot render visually.
Replace them with the physical manifestation. `_prepare_ltx_prompt` performs this
automatically; use these rewrites when writing prompts by hand.

| Bare Label (avoid) | Physical Cue (use instead) |
|--------------------|---------------------------|
| sad | face crumples slightly, takes a slow breath |
| angry | jaw tightens, eyes narrow |
| happy | corners of mouth lift, shoulders ease |
| nervous | fingers fidget, glances sidelong |
| surprised | eyes widen, head tilts back slightly |
| confident | chin lifted, shoulders back, steady gaze forward |

---

## Passivity Token Blacklist

These phrases cause LTX-2 to produce frozen-face output — the subject appears as a still
image rather than an animated character:

- `holds gaze`
- `still expression`
- `composed stillness`
- `frozen expression`
- `frozen face`

`_prepare_ltx_prompt` rewrites all of the above automatically. Include this list so
prompt authors avoid them in the first place rather than relying on the helper to fix
them post hoc.

Rewrites applied by the helper:

| Passivity Token | Replaced With |
|-----------------|---------------|
| `holds gaze` / `hold gaze` | `natural blink cadence, eyes tracking` |
| `still expression` | `subtle micro-expressions shifting` |
| `composed stillness` | `natural micro-movement, weight shifting` |
| `frozen expression` | `animated expression` |
| `frozen face` | `animated expression` |

---

## The 200-Word Cap

LTX-2 has a hard scene prompt length ceiling of 200 words (Lightricks README + empirical
testing). `_prepare_ltx_prompt` truncates with a warning if the input exceeds this.

**Recommendation:** aim for 80–120 words. A dense, well-ordered 100-word prompt
out-performs a sprawling 180-word one. 200 is the ceiling, not the target.

If you need to describe multiple simultaneous elements (environment + character + camera),
prioritise physical character action over environment detail — LTX-2 I2AV weights the
reference portrait image more heavily than text when both are in conflict (see below).

---

## NEG_SUFFIX Rules

`NEG_SUFFIX_6TERM` is the fixed 6-term negative prompt appended when `suppress_text=True`
(the default):

```
no text, no subtitles, no captions, no watermarks, no logos, no visible text
```

These are broad category suppressors. The 6-term lock is non-negotiable.

**Do NOT** add format-name tokens such as `no news chyron`, `no title card`,
`no lower third`. Empirically, appending a specific format name introduces that exact
visual schema into the output via the pink-elephant effect — LTX-2's text encoder
activates on the format concept even under negation. The 6 broad terms suppress the
underlying category without naming the specific schema. Extending the list causes more
text-artifact problems than it solves.

If text artifacts appear despite NEG_SUFFIX, the fix is to remove text-describing phrases
from the positive prompt (e.g., `courtroom nameplate` → `nameplate`, drop it entirely),
not to extend the negative prompt.

---

## What the Prompt Cannot Override

**Ref-image dominance.** In I2AV mode, LTX-2 inherits first-frame composition from the
reference portrait image more strongly than it honors `scene_prompt` text. If the portrait
is a tight headshot, the output will be a tight headshot clip regardless of what the
prompt says about "wide shot" or "medium shot". The lever is the reference image, not
prompt text. Design the reference portrait for the desired framing before calling the tool.

**16:9 keyframe requirement (MV pipeline).** LTX video generation outputs at 16:9 landscape
(e.g., 960x544). If the input keyframe is portrait-oriented, LTX must invent left/right
content, causing compositional artifacts. All QEI keyframes MUST be 16:9 — ensure reference
images are canvas-expanded to 16:9 before QEI (see `music-video-reference-images` Step 2).

**Text rendering.** LTX-2 does not generate readable text. Rely on `NEG_SUFFIX_6TERM` for
text suppression. Do not describe visible text elements (signs, labels, chyrons) in the
positive prompt — they will appear as garbled glyphs if at all.

---

## Empirical Conflicts: Guide vs Local Findings

| Conflict | Official Guide | Our Empirical Finding | Resolution |
|----------|----------------|----------------------|------------|
| NEG_SUFFIX scope | Implied: enumerate specific things to suppress | Adding format-name tokens (news chyron, title card) introduces those exact schemas — pink-elephant effect | Keep 6-term lock; do not extend with format names |
| Ref-image dominance | Silent (prompt controls scene composition) | I2AV first-frame composition inherits from reference portrait strongly — text prompt cannot override framing | Design the reference image for the desired framing; document as ref-image dominance in skill/docs |
| Passivity tokens | Silent (no guidance) | "holds gaze", "composed stillness" reliably cause frozen-face artifacts across multiple test renders | Helper rewrites automatically; blacklist included here for human awareness |

---

## Worked Example

**Raw input (poor prompt):**

> Alice stands at the courtroom lectern, composed stillness, looking sad. She holds gaze toward the judge.

**Issues:**
- `composed stillness` → passivity token (frozen-face risk)
- `sad` → bare emotion label (not visually renderable)
- `holds gaze` → passivity token (frozen-face risk)

**After `_prepare_ltx_prompt`:**

> Alice stands at the courtroom lectern, natural micro-movement, weight shifting, looking face crumples slightly, takes a slow breath. She natural blink cadence, eyes tracking toward the judge.

**Warnings emitted:**
```
Passivity cue rewritten: replaced '\bcomposed stillness\b' with motion cue.
Emotion label rewritten: replaced '\bsad\b' with physical cue.
Passivity cue rewritten: replaced '\bholds? gaze\b' with motion cue.
```

**Better raw prompt (no helper intervention needed):**

> Medium shot. Alice at a polished courtroom lectern, late-afternoon light through tall windows. She sets down a folder, turns slightly toward the bench, jaw tightens, eyes steady. Camera slow push-in on her face. Low ambient echo of the courtroom.
