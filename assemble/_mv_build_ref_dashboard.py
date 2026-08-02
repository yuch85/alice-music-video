#!/usr/bin/env python3
"""Build a per-segment reference image dashboard for the 18-segment plan.

Maps existing refs (from the old 27-segment plan) to new segments by time overlap.
Outputs a self-contained HTML dashboard for user sign-off.

Usage:
    cd ~/alice && uv run scripts/_mv_build_ref_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path.home() / "alice" / "songs" / "music-videos" / "project-name"
NEW_PLAN = PROJECT / "gen-output-18seg" / "transcript.json"
OLD_PLAN = PROJECT / "gen-output" / "segment_plan.json"
REFS_DIR = PROJECT / "gen-output" / "refs"
OUT = PROJECT / "ref_dashboard.html"


def map_refs_to_segments() -> list[dict]:
    """Map old refs to new segments by time overlap.

    Uses per-segment refs (ref_NNN.jpg) from gen-output/refs/ rather than
    the shared portrait.jpg. Maps by matching old segment time ranges to
    new segment time ranges.
    """
    new = json.loads(NEW_PLAN.read_text())
    old = json.loads(OLD_PLAN.read_text())

    old_segs = old.get("segments", [])
    new_segs = new.get("segments", [])

    # Build a lookup: old segment index -> ref_NNN.jpg path
    # The refs are numbered by old segment index (ref_001.jpg for old seg #1).
    old_ref_map: dict[int, str] = {}
    for fname in sorted(REFS_DIR.iterdir()):
        if fname.suffix.lower() in (".jpg", ".jpeg", ".png"):
            stem = fname.stem  # e.g. "ref_001"
            if stem.startswith("ref_"):
                try:
                    idx = int(stem.split("_")[1])
                    # Use relative path from PROJECT directory for the HTML.
                    old_ref_map[idx] = f"gen-output/refs/{fname.name}"
                except (ValueError, IndexError):
                    pass

    segments_data: list[dict] = []
    for ns in new_segs:
        # Find old segments that overlap with this new segment's time range.
        candidates = []
        for os_ in old_segs:
            if os_["start"] < ns["end"] and os_["end"] > ns["start"]:
                old_idx = os_["index"]
                if old_idx in old_ref_map:
                    ref_path = old_ref_map[old_idx]
                    if (PROJECT / ref_path).exists():
                        candidates.append({
                            "label": f"Old #{old_idx}",
                            "path": ref_path,
                            "old_index": old_idx,
                            "old_time": f"{os_['start']:.1f}-{os_['end']:.1f}s",
                            "old_shot_type": os_.get("shot_type", "singer"),
                        })

        # If no time-overlap candidates (e.g. broll outro), use ref_candidates
        # from transcript.json if available.
        if not candidates and "ref_candidates" in ns:
            for rc in ns["ref_candidates"]:
                rc_path = rc["path"]
                if (PROJECT / rc_path).exists():
                    candidates.append({
                        "label": f"Old #{rc['old_index']}",
                        "path": rc_path,
                        "old_index": rc["old_index"],
                        "old_time": rc.get("old_time", "N/A"),
                        "old_shot_type": "broll-reuse",
                    })

        # Deduplicate by path.
        seen: set[str] = set()
        unique: list[dict] = []
        for c in candidates:
            if c["path"] not in seen:
                seen.add(c["path"])
                unique.append(c)

        segments_data.append({
            "index": ns["index"],
            "start": round(ns["start"], 1),
            "end": round(ns["end"], 1),
            "duration": round(ns["duration"], 1),
            "shot_type": ns["shot_type"],
            "text": ns["text"][:150],
            "word_count": ns.get("word_count", 0),
            "candidates": unique,
            "has_refs": len(unique) > 0,
        })

    return segments_data


def build_html(segments_data: list[dict]) -> str:
    """Build self-contained HTML dashboard."""
    segments_json = json.dumps(segments_data, indent=2)

    # Stats
    total = len(segments_data)
    singer = sum(1 for s in segments_data if s["shot_type"] == "singer")
    broll = sum(1 for s in segments_data if s["shot_type"] == "broll")
    with_refs = sum(1 for s in segments_data if s["has_refs"])
    needs_refs = total - with_refs

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Music Video — Reference Dashboard</title>
<style>
:root {
  --bg: #0a0a0a; --surface: #141414; --border: #2a2a2a;
  --text: #e0e0e0; --muted: #888; --accent: #4a9eff;
  --approved: #22c55e; --rejected: #ef4444; --pending: #f59e0b;
  --singer: #4a9eff; --broll: #f59e0b;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  line-height: 1.5; padding: 2rem; max-width: 1400px; margin: 0 auto;
}
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
.stats { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.8rem;
}
.stat strong { color: var(--accent); font-size: 1.1rem; display: block; }
.stat.warn strong { color: var(--pending); }

/* Timeline */
.timeline { margin-bottom: 2rem; }
.timeline-bar {
  height: 36px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; position: relative; overflow: hidden; margin-bottom: 0.4rem;
}
.timeline-seg {
  position: absolute; top: 0; height: 100%; display: flex;
  align-items: center; justify-content: center; font-size: 0.6rem;
  color: #fff; border-right: 1px solid var(--bg); cursor: pointer;
  transition: opacity 0.15s;
}
.timeline-seg:hover { opacity: 0.8; }
.timeline-seg.singer { background: #4a9eff88; }
.timeline-seg.broll { background: #f59e0b88; }
.timeline-labels {
  display: flex; justify-content: space-between;
  font-size: 0.7rem; color: var(--muted);
}

/* Segment cards */
.segment {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; margin-bottom: 1rem; overflow: hidden;
}
.segment-header {
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.75rem 1rem; background: rgba(255,255,255,0.02);
  border-bottom: 1px solid var(--border); flex-wrap: wrap;
}
.seg-num { font-size: 1.1rem; font-weight: 700; min-width: 2rem; }
.seg-num.singer { color: var(--singer); }
.seg-num.broll { color: var(--broll); }
.badge {
  font-size: 0.65rem; padding: 0.1rem 0.4rem; border-radius: 3px;
  text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;
}
.badge.singer { background: #4a9eff22; color: var(--singer); }
.badge.broll { background: #f59e0b22; color: var(--broll); }
.time { color: var(--muted); font-size: 0.85rem; }
.duration { color: var(--muted); font-size: 0.85rem; margin-left: 0.5rem; }
.words-count { color: var(--muted); font-size: 0.8rem; }
.segment-body { display: flex; gap: 1rem; padding: 1rem; flex-wrap: wrap; }
.lyrics { flex: 1; min-width: 250px; font-size: 0.85rem; color: var(--muted); line-height: 1.7; }
.ref-area { flex: 1; min-width: 280px; }
.ref-area h4 { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.5rem; font-weight: 500; }
.candidates { display: flex; gap: 0.75rem; flex-wrap: wrap; }
.candidate {
  position: relative; flex: 1; min-width: 180px; max-width: 300px;
}
.candidate img {
  width: 100%; border-radius: 6px; border: 2px solid transparent;
  display: block; background: #000;
}
.candidate.selected img { border-color: var(--accent); }
.candidate-info {
  font-size: 0.7rem; color: var(--muted); margin-top: 0.25rem;
}
.candidate label {
  display: block; margin-top: 0.35rem; text-align: center; cursor: pointer;
}
.candidate input[type="radio"] { display: none; }
.candidate .radio-label {
  padding: 0.3rem 0.75rem; border-radius: 4px; background: rgba(255,255,255,0.05);
  font-size: 0.75rem; transition: background 0.2s;
}
.candidate.selected .radio-label { background: var(--accent); color: #fff; }
.no-refs {
  font-size: 0.8rem; color: var(--pending); padding: 0.5rem 0;
}
.segment-footer {
  display: flex; justify-content: flex-end; gap: 0.5rem;
  padding: 0.5rem 1rem; border-top: 1px solid var(--border);
}
.btn {
  border: 1px solid var(--border); background: transparent; color: var(--text);
  padding: 0.3rem 0.75rem; border-radius: 6px; cursor: pointer; font-size: 0.75rem;
  transition: all 0.15s;
}
.btn:hover { background: rgba(255,255,255,0.1); }
.btn.approve:hover { border-color: var(--approved); color: var(--approved); }
.btn.reject:hover { border-color: var(--rejected); color: var(--rejected); }

/* Export bar */
.export-bar {
  position: sticky; bottom: 0; background: var(--surface);
  border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem 1.5rem; margin-top: 2rem;
  display: flex; justify-content: space-between; align-items: center;
  backdrop-filter: blur(10px); z-index: 100;
}
.export-stats { display: flex; gap: 1.5rem; font-size: 0.85rem; }
.export-btn {
  background: var(--accent); color: #fff; border: none;
  padding: 0.5rem 1.5rem; border-radius: 8px; cursor: pointer; font-size: 0.9rem;
}
.export-btn:hover { background: #3a8eef; }
#output {
  display: none; position: fixed; bottom: 4rem; right: 2rem;
  background: #000; color: #0f0; padding: 1rem; border-radius: 8px;
  max-width: 600px; max-height: 400px; overflow: auto;
  font-family: monospace; font-size: 0.75rem; z-index: 99;
  border: 1px solid var(--border);
}
.note {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;
  font-size: 0.85rem;
}
.note h3 { font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--accent); }
.note ul { margin-left: 1.25rem; }
.note li { margin-bottom: 0.25rem; }
</style>
</head>
<body>
<h1>Music Video — Reference Dashboard</h1>
<p class="subtitle">Per-segment reference image review for user sign-off before LTX generation</p>

<div class="stats">
  <div class="stat"><strong>SEGMENTS_PLACEHOLDER</strong>Segments</div>
  <div class="stat"><strong>SINGER_PLACEHOLDER</strong>Singer</div>
  <div class="stat"><strong>BROLL_PLACEHOLDER</strong>B-roll</div>
  <div class="stat"><strong>WITH_REFS_PLACEHOLDER</strong>With Refs</div>
  <div class="stat warn"><strong>NEEDS_REFS_PLACEHOLDER</strong>Need Refs</div>
</div>

<div class="note">
  <h3>Ref Mapping Notes</h3>
  <ul>
    <li>Existing refs from the old 27-segment plan are mapped by time overlap to the new 18-segment plan.</li>
    <li>Singer segments (0-13) may have mapped refs from the old plan. Review for suitability.</li>
    <li>B-roll segments (14-16) cover the instrumental outro (155.8-200.0s). New refs needed — old plan had no outro coverage.</li>
    <li>Select one ref per segment (or flag "need new ref") before generation.</li>
  </ul>
</div>

<h2 style="font-size:1rem;margin-bottom:0.75rem;">Timeline</h2>
<div class="timeline">
  <div class="timeline-bar">TIMELINE_PLACEHOLDER</div>
  <div class="timeline-labels">
    <span>0.0s</span><span>50s</span><span>100s</span><span>150s</span><span>200s</span>
  </div>
</div>

<h2 style="font-size:1rem;margin-bottom:0.75rem;">Segments</h2>
<div id="segments"></div>

<div class="export-bar">
  <div class="export-stats">
    <span>Selected: <strong id="sel-count">0</strong>/<strong>SEGMENTS_PLACEHOLDER</strong></span>
  </div>
  <button class="export-btn" onclick="exportSelections()">Export Selections</button>
</div>
<pre id="output"></pre>

<script>
const segments = SEGMENTS_JSON_PLACEHOLDER;
const audioDur = 200.04;

function render() {
    // Timeline
    var tl = document.querySelector('.timeline-bar');
    segments.forEach(function(seg) {
        var div = document.createElement('div');
        var left = (seg.start / audioDur * 100).toFixed(1);
        var width = (seg.duration / audioDur * 100).toFixed(1);
        div.className = 'timeline-seg ' + seg.shot_type;
        div.style.left = left + '%';
        div.style.width = width + '%';
        div.title = 'Seg ' + seg.index + ': ' + seg.start + '-' + seg.end + 's';
        div.textContent = seg.index;
        div.onclick = function() {
            document.getElementById('seg-' + seg.index).scrollIntoView({behavior:'smooth', block:'center'});
        };
        tl.appendChild(div);
    });

    // Segment cards
    var container = document.getElementById('segments');
    segments.forEach(function(seg) {
        var card = document.createElement('div');
        card.className = 'segment';
        card.id = 'seg-' + seg.index;

        var candidatesHtml = '';
        if (seg.candidates && seg.candidates.length > 0) {
            seg.candidates.forEach(function(c, ci) {
                candidatesHtml += '<div class="candidate" id="seg-' + seg.index + '-cand-' + ci + '">' +
                    '<img src="' + c.path + '" alt="Ref ' + c.label + '" onerror="this.style.display=\\'none\\'">' +
                    '<div class="candidate-info">' + c.label + ' (' + c.old_time + ')</div>' +
                    '<label><input type="radio" name="seg-' + seg.index + '" value="' + ci + '" onchange="selectCandidate(' + seg.index + ', ' + ci + ')" class="seg-radio">' +
                    '<span class="radio-label">Select</span></label></div>';
            });
        } else {
            candidatesHtml = '<div class="no-refs">No mapped refs — new ref needed for this segment.</div>';
        }

        card.innerHTML = '<div class="segment-header">' +
            '<span class="seg-num ' + seg.shot_type + '">' + seg.index + '</span>' +
            '<span class="badge ' + seg.shot_type + '">' + seg.shot_type + '</span>' +
            '<span class="time">' + seg.start + 's - ' + seg.end + 's</span>' +
            '<span class="duration">' + seg.duration + 's</span>' +
            (seg.word_count > 0 ? '<span class="words-count">' + seg.word_count + ' words</span>' : '') +
            '</div>' +
            '<div class="segment-body">' +
            '<div class="lyrics">' + (seg.text || '—') + '</div>' +
            '<div class="ref-area"><h4>Reference Image</h4><div class="candidates">' + candidatesHtml + '</div></div>' +
            '</div>' +
            '<div class="segment-footer">' +
            '<button class="btn approve" onclick="this.closest(\\'.segment\\').classList.add(\\'approved\\')">Approve</button>' +
            '<button class="btn reject" onclick="this.closest(\\'.segment\\').classList.add(\\'rejected\\')">Reject</button>' +
            '</div>';
        container.appendChild(card);
    });
    updateCount();
}

function selectCandidate(idx, ci) {
    var radios = document.querySelectorAll('[name="seg-' + idx + '"]');
    radios.forEach(function(r) {
        r.closest('.candidate').classList.remove('selected');
    });
    document.querySelector('input[name="seg-' + idx + '"][value="' + ci + '"]').closest('.candidate').classList.add('selected');
    updateCount();
}

function updateCount() {
    var count = 0;
    segments.forEach(function(seg, idx) {
        var sel = document.querySelector('input[name="seg-' + idx + '"]:checked');
        if (sel) count++;
    });
    document.getElementById('sel-count').textContent = count;
}

function exportSelections() {
    var selections = segments.map(function(seg, idx) {
        var sel = document.querySelector('input[name="seg-' + idx + '"]:checked');
        var card = document.getElementById('seg-' + idx);
        var status = card.classList.contains('approved') ? 'approved' :
                     card.classList.contains('rejected') ? 'rejected' : 'pending';
        return {
            segment: seg.index,
            start: seg.start,
            end: seg.end,
            duration: seg.duration,
            shot_type: seg.shot_type,
            status: status,
            selected_candidate: sel ? parseInt(sel.value) : -1,
            ref_path: sel ? seg.candidates[parseInt(sel.value)].path : null,
            ref_label: sel ? seg.candidates[parseInt(sel.value)].label : null,
            needs_new_ref: !sel && !seg.has_refs
        };
    });
    var output = document.getElementById('output');
    output.textContent = JSON.stringify(selections, null, 2);
    output.style.display = 'block';
    navigator.clipboard.writeText(JSON.stringify(selections, null, 2)).catch(function() {});
}

render();
</script>
</body>
</html>"""

    # Build timeline HTML
    audio_dur = 200.04
    timeline_html = ""
    for seg in segments_data:
        left = (seg["start"] / audio_dur) * 100
        width = (seg["duration"] / audio_dur) * 100
        st = seg["shot_type"]
        timeline_html += f'<div class="timeline-seg {st}" style="left:{left:.1f}%;width:{width:.1f}%" title="Seg {seg["index"]}: {seg["start"]}-{seg["end"]}s">{seg["index"]}</div>\n'
    html = html.replace("SEGMENTS_PLACEHOLDER", str(total))
    html = html.replace("SINGER_PLACEHOLDER", str(singer))
    html = html.replace("BROLL_PLACEHOLDER", str(broll))
    html = html.replace("WITH_REFS_PLACEHOLDER", str(with_refs))
    html = html.replace("NEEDS_REFS_PLACEHOLDER", str(needs_refs))
    html = html.replace("TIMELINE_PLACEHOLDER", timeline_html)
    html = html.replace("SEGMENTS_JSON_PLACEHOLDER", segments_json)
    return html


def main() -> None:
    segments_data = map_refs_to_segments()
    html = build_html(segments_data)
    OUT.write_text(html)
    print(f"Dashboard saved: {OUT}")
    print(f"Segments: {len(segments_data)}")
    for s in segments_data:
        refs = len(s["candidates"])
        print(f"  [{s['index']:2d}] {s['start']:5.1f}-{s['end']:5.1f}s {s['shot_type']:8s} {refs} ref(s)")


if __name__ == "__main__":
    main()
