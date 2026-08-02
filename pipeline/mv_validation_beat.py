#!/usr/bin/env python3
"""Beat-by-beat content validation for mv_validation.

Duration checks and two-tier continuity validation (deterministic + LLM).
Kept separate to stay within 300 LOC per file.
"""

from __future__ import annotations

import re
from pathlib import Path

_MIN_BEAT_DURATION_S = 2.0
_MAX_BEAT_DURATION_S = 18.0


def validate_beat_content(
    beats_path: Path,
    prompts_dir: Path,
    continuity_path: Path,
    min_duration: float = _MIN_BEAT_DURATION_S,
    max_duration: float = _MAX_BEAT_DURATION_S,
) -> list[str]:
    """Beat-by-beat content validation.

    For each beat:
    (a) Verify configured duration is within [min_duration, max_duration].
    (b) Run two-tier continuity check:
        - Tier 1: Deterministic string matching (BLOCKING)
        - Tier 2: LLM-based semantic check (WARNING) — deferred

    Args:
        beats_path: Path to beat_sheet.md.
        prompts_dir: Path to prompts/ directory.
        continuity_path: Path to continuity_bible.md.
        min_duration: Minimum valid beat duration in seconds.
        max_duration: Maximum valid beat duration in seconds.

    Returns:
        List of violation descriptions with tier labels.
    """
    violations: list[str] = []

    if not beats_path.exists():
        return violations

    beats = _parse_beat_sheet(beats_path)
    if not beats:
        return violations

    invariance_rules: list[str] = []
    if continuity_path.exists():
        invariance_rules = _extract_invariance_rules(continuity_path)

    prompts = _load_prompts(prompts_dir)

    for beat_id, duration in beats.items():
        if duration < min_duration or duration > max_duration:
            violations.append(
                f"Beat {beat_id} duration {duration:.1f}s "
                f"outside [{min_duration:.0f}s, {max_duration:.0f}s] [BLOCKING]"
            )

        motion_prompt = prompts.get(beat_id, "")
        for rule in invariance_rules:
            contradiction = _check_invariance_contradiction(rule, motion_prompt)
            if contradiction:
                violations.append(
                    f"Beat {beat_id} prompt contradicts continuity: "
                    f"{contradiction} [BLOCKING]"
                )

    return violations


def _count_beats(project_dir: Path) -> int:
    """Count beats from beat_sheet.md."""
    beats_path = project_dir / "beat_sheet.md"
    if not beats_path.exists():
        return 0
    beats = _parse_beat_sheet(beats_path)
    return len(beats)


def _parse_beat_sheet(beats_path: Path) -> dict[str, float]:
    """Parse beat_sheet.md and return {beat_id: duration} mapping."""
    beats: dict[str, float] = {}
    content = beats_path.read_text()

    for line in content.split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 4 and cells[0].startswith("B"):
            try:
                duration = float(cells[3])
                beats[cells[0]] = duration
            except (ValueError, IndexError):
                continue

    return beats


def _extract_invariance_rules(continuity_path: Path) -> list[str]:
    """Extract invariance rules from continuity_bible.md."""
    rules: list[str] = []
    content = continuity_path.read_text()

    in_invariance_section = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("##"):
            in_invariance_section = "invariance" in stripped.lower()
            continue
        if in_invariance_section and stripped.startswith("-"):
            rule = stripped.lstrip("- ").strip()
            if rule:
                rules.append(rule)

    return rules


def _load_prompts(prompts_dir: Path) -> dict[str, str]:
    """Load motion prompts from prompts/ directory."""
    prompts: dict[str, str] = {}
    if not prompts_dir.exists():
        return prompts

    for p in prompts_dir.glob("beat_*.md"):
        stem = p.stem
        beat_num = stem.replace("beat_", "").zfill(2)
        beat_id = f"B{beat_num}"
        prompts[beat_id] = p.read_text()

    return prompts


def _check_invariance_contradiction(
    rule: str, motion_prompt: str
) -> str | None:
    """Check if a motion prompt contradicts an invariance rule.

    Uses deterministic substring matching. Returns contradiction description
    or None if no contradiction found.
    """
    if not motion_prompt:
        return None

    rule_lower = rule.lower()
    prompt_lower = motion_prompt.lower()

    if "dress" in rule_lower and "dress" in prompt_lower:
        rule_colors = _extract_colors(rule_lower)
        prompt_colors = _extract_colors(prompt_lower)
        conflicting = set(rule_colors) & set(prompt_colors)
        if not conflicting and rule_colors and prompt_colors:
            return (
                f"character wearing {', '.join(prompt_colors)} "
                f"but bible specifies {', '.join(rule_colors)}"
            )

    return None


def _extract_colors(text: str) -> list[str]:
    """Extract color words from text."""
    colors = [
        "red", "blue", "green", "yellow", "black", "white",
        "orange", "purple", "pink", "brown", "gray", "grey",
        "navy", "teal", "maroon", "cyan", "magenta", "gold",
        "silver", "ivory", "cream", "beige", "tan",
    ]
    return [c for c in colors if c in text]
