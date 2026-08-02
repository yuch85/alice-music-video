#!/usr/bin/env python3
"""Shared parsing utilities for the music video pipeline.

Markdown table parsing with auto-repair and JSON extraction/repair
from LLM output. Consumed by mv_beats, mv_storyboard, and downstream
shot list skills.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_markdown_table(md_text: str) -> list[dict[str, str]]:
    """Parse a markdown table into a list of dicts with full auto-repair.

    Handles mismatched column count (pad or truncate), divider row
    skipping, empty line sanitization, and header validation.

    Args:
        md_text: Markdown table text.

    Returns:
        List of dicts keyed by header row values.

    Raises:
        ValueError: If header row is missing, empty, or no separator row.
    """
    lines = [line.strip() for line in md_text.strip().splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return []

    header_line = None
    header_idx = -1
    for i, line in enumerate(lines):
        if "|" in line:
            header_line = line
            header_idx = i
            break

    if header_line is None:
        raise ValueError("No header row found in markdown table")

    headers = _split_row(header_line)
    if not headers or all(h.strip() == "" for h in headers):
        raise ValueError("Header row is empty in markdown table")
    headers = [h.strip() for h in headers]
    header_count = len(headers)

    # Require a separator row after the header
    has_separator = False
    for i in range(header_idx + 1, min(header_idx + 3, len(lines))):
        cells = _split_row(lines[i])
        if _is_divider_row(cells):
            has_separator = True
            break
    if not has_separator:
        raise ValueError(
            "No separator row found after header — not a valid markdown table"
        )

    rows: list[dict[str, str]] = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        cells = _split_row(line)

        if _is_divider_row(cells):
            continue

        if len(cells) < header_count:
            logger.warning(
                "Row %d has %d cells, expected %d — padding with empty strings",
                i + 1, len(cells), header_count,
            )
            cells.extend([""] * (header_count - len(cells)))
        elif len(cells) > header_count:
            logger.warning(
                "Row %d has %d cells, expected %d — truncating",
                i + 1, len(cells), header_count,
            )
            cells = cells[:header_count]

        row_dict = {headers[j]: cells[j].strip() for j in range(header_count)}
        rows.append(row_dict)

    return rows


def _split_row(line: str) -> list[str]:
    """Split a markdown table row on pipe delimiters."""
    line = line.strip().strip("|")
    cells = line.split("|")
    return [c.strip() for c in cells]


def _is_divider_row(cells: list[str]) -> bool:
    """Check if a row is a horizontal separator."""
    combined = "".join(cells).strip()
    if not combined:
        return True
    return bool(re.match(r"^[\s\|:-]+$", combined))


def extract_and_repair_json(text: str) -> Any | None:
    """Extract and repair JSON blocks from LLM output text.

    Handles markdown wrappers, partial outputs, trailing commas,
    and unbalanced brackets.

    Args:
        text: Raw LLM output text.

    Returns:
        Parsed JSON object, or None if unparseable.
    """
    if not text:
        return None

    cleaned = text.strip()
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # If text starts with [, try greedy match for nested structures
    if cleaned.lstrip().startswith("["):
        greedy_match = re.search(r"(?s)\[.*", cleaned)
        if greedy_match:
            repaired = _repair_json_string(greedy_match.group())
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    for pattern in [r"(?s)\[.*?\]", r"(?s)\{.*?\}"]:
        matches = re.findall(pattern, cleaned)
        if not matches:
            continue
        for match in sorted(matches, key=len, reverse=True):
            repaired = _repair_json_string(match)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue

    repaired = _repair_json_string(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _repair_json_string(text: str) -> str:
    """Apply JSON repair: trailing comma removal + bracket balancing."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    text += "]" * max(0, open_brackets)
    text += "}" * max(0, open_braces)
    return text
