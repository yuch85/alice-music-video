#!/usr/bin/env python3
"""Update dashboard-clips.html SHOTS array with motion prompts from prompt files.

Reads each prompts/shot_SH-*.md file, extracts the motion prompt (between
## Motion Prompt and the **Word count** line), and patches the SHOTS JS array
in the dashboard HTML.
"""

import json
import re
import glob
import sys

DASHBOARD = "songs/music-videos/project-name/dashboard-clips.html"
PROMPT_DIR = "songs/music-videos/project-name/prompts"


def extract_motion_prompt(filepath: str) -> str | None:
    """Extract motion prompt from a shot prompt markdown file."""
    text = open(filepath).read()
    # Find content between ## Motion Prompt and **Word count**
    m = re.search(r"^## Motion Prompt\s*\n\n(.*?)\n\*\*Word count\*\*", text, re.DOTALL | re.MULTILINE)
    if not m:
        return None
    prompt = m.group(1).strip()
    # Remove trailing metadata lines that aren't part of the prompt
    return prompt


def main():
    # Load all prompt files
    prompts = {}
    for f in sorted(glob.glob(f"{PROMPT_DIR}/shot_SH-*.md")):
        shot_id = re.search(r"shot_(SH-\d+)\.md", f).group(1)
        motion = extract_motion_prompt(f)
        if motion:
            prompts[shot_id] = motion

    print(f"Loaded {len(prompts)} prompt files")

    # Read dashboard
    html = open(DASHBOARD).read()

    # Extract SHOTS array
    m = re.search(r'var SHOTS = (\[.*?\]);\s*function escapeHtml', html, re.DOTALL)
    if not m:
        print("ERROR: Could not find SHOTS array in dashboard", file=sys.stderr)
        sys.exit(1)

    shots_json = m.group(1)
    shots = json.loads(shots_json)

    # Update motion prompts
    updated = []
    for shot in shots:
        sid = shot["shot_id"]
        if sid in prompts:
            old = shot["motion_prompt"]
            new = prompts[sid]
            if old != new:
                shot["motion_prompt"] = new
                updated.append(sid)
                print(f"  Updated {sid}")
            else:
                print(f"  Unchanged {sid}")
        else:
            print(f"  Skip {sid} (no prompt file)")

    if not updated:
        print("No prompts changed.")
        return

    # Write back
    new_json = json.dumps(shots, ensure_ascii=False)
    new_html = html[:m.start(1)] + new_json + html[m.end(1):]
    open(DASHBOARD, "w").write(new_html)
    print(f"\nUpdated {len(updated)} shots: {', '.join(updated)}")
    print(f"Dashboard: {DASHBOARD}")


if __name__ == "__main__":
    main()
