#!/usr/bin/env python3
"""Persistence helpers for mv_fsm — read/write FSM state to index.md.

Handles JSON block extraction, robust repair, and template generation.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

from mv_fsm import MVStage, MusicVideoFSM

_INDEX_FILE = "index.md"
_BLOCK_START = "```json\n"
_BLOCK_END = "\n```"


def repair_json(raw: str) -> str:
    """Attempt to repair common JSON syntax errors before parsing.

    Handles BOM stripping, trailing commas, unbalanced braces,
    and non-printable control characters.
    """
    repaired = raw.replace("﻿", "")
    repaired = "".join(
        ch for ch in repaired if ch.isprintable() or ch in ("\n", "\r", "\t")
    ).strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += "}" * max(0, open_braces) + "]" * max(0, open_brackets)
    return repaired


def parse_json_block(text: str) -> dict[str, object]:
    """Extract and parse a JSON code block from markdown text.

    Uses robust JSON repair for resilience against manual editing errors.
    """
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if not match:
        match = re.search(r"\{\s*\"current\"", text)
        if match:
            json_str = text[match.start():]
        else:
            raise ValueError(
                "No FSM JSON block found in index.md. "
                "Run 'mv_fsm_cli.py init' to reinitialize."
            )
    else:
        json_str = match.group(1)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = repair_json(json_str)
        try:
            print(
                "Warning: FSM state in index.md had syntax errors — "
                "auto-repaired. Review recommended.",
                file=sys.stderr,
            )
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse FSM state JSON: {exc}. "
                "Run 'mv_fsm_cli.py init' to reinitialize."
            ) from exc


def build_json_block(data: dict[str, object]) -> str:
    """Format FSM state dict as a JSON code block."""
    return _BLOCK_START + json.dumps(data, indent=2) + _BLOCK_END


def load_fsm(project_dir: Path) -> tuple[MusicVideoFSM, Path, str]:
    """Load FSM state from project_dir/index.md.

    Returns (fsm, index_path, raw_content).
    """
    index_path = project_dir / _INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(
            f"No {_INDEX_FILE} found in {project_dir}. "
            "Run 'mv_fsm_cli.py init' first."
        )
    content = index_path.read_text()
    data = parse_json_block(content)
    fsm = MusicVideoFSM()
    fsm.from_dict(data)
    return fsm, index_path, content


def save_fsm(fsm: MusicVideoFSM, index_path: Path, current_content: str) -> None:
    """Update the FSM JSON block in index.md, preserving surrounding content."""
    data = fsm.to_dict()
    new_block = build_json_block(data)

    match = re.search(r"```json\s*\n(.*?)\n```", current_content, re.DOTALL)
    if match:
        new_content = (
            current_content[: match.start()]
            + new_block
            + current_content[match.end() :]
        )
    else:
        new_content = (
            current_content.rstrip()
            + "\n\n## FSM State JSON\n\n"
            + new_block
            + "\n"
        )
    index_path.write_text(new_content)


def get_template(project_name: str) -> str:
    """Return the index.md template for a new project."""
    fsm_block = build_json_block(MusicVideoFSM().to_dict())
    return textwrap.dedent(f"""\
# {project_name} — Music Video Project

## Metadata
- **Audio**: audio.mp3
- **Lyrics**: lyrics/
- **Portrait**: refs/portrait.jpg
- **Resolution**: 1920x1080
- **Aspect Ratio**: 16:9
- **Created**:
- **Current Stage**: INTERVIEW

## Stage Status

| Stage | Status | Artefact | Notes |
|-------|--------|----------|-------|
| INTERVIEW | NOT_STARTED | project.md | |
| TREATMENT | NOT_STARTED | director_treatment.md | |
| CONTINUITY | NOT_STARTED | continuity_bible.md | |
| AUDIO_ANALYSIS | NOT_STARTED | lyrics/transcript.json, vocals.wav, instrumental.wav | |
| BEATS | NOT_STARTED | beat_sheet.md | |
| STORYBOARD | NOT_STARTED | storyboard.md | |
| SHOTS | NOT_STARTED | shot_list.md | |
| IMAGE_APPROVAL | NOT_STARTED | refs/ | |
| PROMPTS | NOT_STARTED | prompts/ | |
| VALIDATED | NOT_STARTED | validation.md | |
| GENERATING | NOT_STARTED | clips/ | |
| QC | NOT_STARTED | qc_report.md | |
| COMPLETE | NOT_STARTED | final/ | |

## FSM State JSON

{fsm_block}
""")
