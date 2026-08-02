#!/usr/bin/env python3
"""Build an HTML QC-review dashboard for MV clip generation prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
PROMPTS_DIR = BASE / "prompts"
KEYFRAMES_DIR = BASE / "keyframes"
SHOT_LIST = BASE / "shot_list.md"
OUTPUT = BASE / "prompt-review-dashboard.html"

NEW_SHOTS = {
    "SH-001", "SH-002", "SH-003", "SH-004", "SH-005", "SH-006", "SH-007",
    "SH-009", "SH-016", "SH-026", "SH-027", "SH-030", "SH-032", "SH-033",
    "SH-034", "SH-039", "SH-041", "SH-043", "SH-044", "SH-045", "SH-046",
    "SH-048",
}


def parse_shot_list() -> dict:
    text = SHOT_LIST.read_text()
    result: dict = {}
    current_section: str = ""
    current_shot: str | None = None

    # Single pass: track section changes and shot headers in order
    section_re = re.compile(r"^##\s+(S\d+\.?\d*)\s*--\s*(.+)$")
    shot_re = re.compile(
        r"^###\s+(SH-\d+)\s*\|\s*(S[\d.]+(?:-[\d.]+)?)\s*\|\s*(\d+)->(\d+)s\s*\|\s*(\d+)s\s*\|\s*(\S+)"
    )

    for line in text.split("\n"):
        sm = section_re.match(line)
        if sm:
            current_section = sm.group(2).strip()
            continue

        sm = shot_re.match(line)
        if sm:
            shot_id = sm.group(1)
            current_shot = shot_id
            result[shot_id] = {
                "section": current_section,
                "sub_id": sm.group(2),
                "timestamp": f"{sm.group(3)}->{sm.group(4)}s",
                "duration": int(sm.group(5)),
                "visual_type": sm.group(6),
                "lens": "",
                "framing": "",
            }
            continue

        # Parse bullet details for current shot
        if current_shot and current_shot in result:
            bm = re.match(r"- \*\*Lens\*\*\s*:\s*(.+)", line)
            if bm:
                result[current_shot]["lens"] = bm.group(1).strip()
            fm = re.match(r"- \*\*Framing\*\*\s*:\s*(.+)", line)
            if fm:
                result[current_shot]["framing"] = fm.group(1).strip()

    return result


def parse_prompt_file(path: Path) -> dict:
    text = path.read_text()

    header_m = re.match(r"#\s+(SH-\d+)\s*—\s*(.+?)\s*—\s*(\d+)s", text)
    shot_id = header_m.group(1) if header_m else path.stem.replace("shot_", "")
    section_name = header_m.group(2) if header_m else ""
    duration = int(header_m.group(3)) if header_m else 0

    def extract_field(name: str) -> str:
        m = re.search(rf"\*\*{name}\*\*\s*:\s*(.+)", text)
        return m.group(1).strip() if m else ""

    metadata = {
        "shot_type": extract_field("Shot type"),
        "characters": extract_field("Character(s)"),
        "ref_image": extract_field("Ref image"),
        "weather": extract_field("Weather"),
        "duration": duration,
    }

    no_qei_match = re.search(r"\*\*Keyframe\*\*:\s*(.+?)(?:\n|$)", text)
    has_no_qei = no_qei_match and "no QEI keyframe needed" in no_qei_match.group(1)

    keyframe_prompt = ""
    keyframe_meta: dict = {}
    kf_section = re.search(
        r"## Keyframe Prompt\s*\n\n(.*?)(?=\n## |\n---\n|\Z)",
        text, re.DOTALL,
    )
    if kf_section:
        kf_text = kf_section.group(1).strip()
        meta_start = re.search(r"\*\*Source ref\*\*", kf_text)
        if meta_start:
            keyframe_prompt = kf_text[:meta_start.start()].strip()
            kf_meta_text = kf_text[meta_start.start():]
        else:
            keyframe_prompt = kf_text
            kf_meta_text = ""
        for field in ("Source ref", "Target framing", "Lens", "Weather", "Lighting"):
            fm = re.search(rf"\*\*{field}\*\*\s*:\s*(.+)", kf_meta_text)
            if fm:
                keyframe_meta[field.lower().replace(" ", "_")] = fm.group(1).strip()

    motion_prompt = ""
    motion_meta: dict = {}
    mp_section = re.search(
        r"## Motion Prompt\s*\n\n(.*?)(?=\n## |\n---\n|\Z)",
        text, re.DOTALL,
    )
    if mp_section:
        mp_text = mp_section.group(1).strip()
        meta_start = re.search(r"\*\*Word count\*\*", mp_text)
        if meta_start:
            motion_prompt = mp_text[:meta_start.start()].strip()
            mp_meta_text = mp_text[meta_start.start():]
        else:
            motion_prompt = mp_text
            mp_meta_text = ""
        for field in ("Word count", "Passivity check", "Key considerations"):
            fm = re.search(rf"\*\*{field}\*\*\s*:\s*(.+)", mp_meta_text)
            if fm:
                motion_meta[field.lower().replace(" ", "_")] = fm.group(1).strip()

    return {
        "shot_id": shot_id,
        "section_name": section_name,
        "metadata": metadata,
        "has_keyframe_prompt": bool(keyframe_prompt) and not has_no_qei,
        "keyframe_prompt": keyframe_prompt,
        "keyframe_meta": keyframe_meta,
        "no_qei_note": no_qei_match.group(1).strip() if no_qei_match else "",
        "motion_prompt": motion_prompt,
        "motion_meta": motion_meta,
    }


def load_keyframe(shot_id: str) -> str | None:
    kf_path = KEYFRAMES_DIR / f"keyframe_shot_{shot_id}.jpg"
    if not kf_path.exists():
        return None
    return f"keyframes/keyframe_shot_{shot_id}.jpg"


# Section ordering from shot_list.md (parsed in order of appearance)
SECTION_ORDER = [
    "Prologue (Ambient, No Music, 0-30s)",
    "Humming / Instrumental Transition (0-12s)",
    "V1: Character Quiet Uncertainty (12-28s)",
    "V1 Cont.: Character Insert (28-32s)",
    "V1 Cont.: Character / Character Intercut (32-46s)",
    "Pre-C1 / Breathing Space (46-48s)",
    "C1: First Chorus (48-69s)",
    "Gap 1: Instrumental (69-79s)",
    "V2: Character Journey (79-100s)",
    "V2 Cont. / Pre-C2: Parallel Journeys (100-112s)",
    "C2: Second Chorus (112-130s)",
    "Bridge: Stumble & Turning Point (130-145s)",
    "Bridge: Character Reflection (145-149s)",
    "Bridge Cont.: Returning (149-155s)",
    "Final Chorus: Climax (155-177s)",
    "Final Character Punctuation (177-180s)",
    "Outro: aa (180-183s)",
    "Fade (183-200s)",
    "Epilogue (200-205s)",
]


def build_html(shots_data: list) -> str:
    template_path = Path(__file__).with_name("_mv_dashboard_template.html")
    template = template_path.read_text()
    shots_json = json.dumps(shots_data, ensure_ascii=False)
    html = template.replace("SHOTS_JSON_PLACEHOLDER", shots_json)
    return html


def main() -> None:
    shot_list = parse_shot_list()

    # Assign section order indices
    section_index: dict = {}
    for idx, sec in enumerate(SECTION_ORDER):
        section_index[sec] = idx

    prompt_files = sorted(PROMPTS_DIR.glob("shot_SH-*.md"))
    shots_data: list = []

    for pf in prompt_files:
        parsed = parse_prompt_file(pf)
        sid = parsed["shot_id"]
        sl = shot_list.get(sid, {})

        kf_b64 = load_keyframe(sid)

        shots_data.append({
            "shot_id": sid,
            "sub_id": sl.get("sub_id", parsed.get("section_name", "")),
            "section": sl.get("section", ""),
            "section_order": section_index.get(sl.get("section", ""), 999),
            "timestamp": sl.get("timestamp", ""),
            "duration": sl.get("duration", parsed["metadata"].get("duration", 0)),
            "visual_type": sl.get("visual_type", ""),
            "lens": sl.get("lens", ""),
            "framing": sl.get("framing", ""),
            "characters": parsed["metadata"].get("characters", ""),
            "is_new": sid in NEW_SHOTS,
            "has_keyframe_prompt": parsed["has_keyframe_prompt"],
            "keyframe_prompt": parsed["keyframe_prompt"],
            "keyframe_meta": parsed["keyframe_meta"],
            "no_qei_note": parsed["no_qei_note"],
            "motion_prompt": parsed["motion_prompt"],
            "motion_meta": parsed["motion_meta"],
            "keyframe_b64": kf_b64,
        })

    # Sort by section order, then shot number
    shots_data.sort(key=lambda s: (s["section_order"], s["shot_id"]))

    html = build_html(shots_data)
    OUTPUT.write_text(html)
    print(f"Dashboard written to {OUTPUT}")
    print(f"  Shots: {len(shots_data)}")
    print(f"  With keyframes: {sum(1 for s in shots_data if s['keyframe_b64'])}")
    print(f"  With KF prompts: {sum(1 for s in shots_data if s['has_keyframe_prompt'])}")
    print(f"  New/regenerated: {sum(1 for s in shots_data if s['is_new'])}")


if __name__ == "__main__":
    main()
