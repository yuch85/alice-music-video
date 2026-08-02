#!/usr/bin/env python3
"""Regenerate clips.html dashboard for all 32 clips (27 original + 5 outro).

Reads segment_plan.json + outro_prompts.json + index.md for metadata.
Outputs HTML to artifact directory.
"""
import json
from pathlib import Path

MV_DIR = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
ARTIFACT = Path(os.environ.get("MV_ARTIFACT_DIR", "/path/to/artifacts"))

# Load segment plan
plan = json.loads((MV_DIR / "gen-output" / "segment_plan.json").read_text())
outro = json.loads((MV_DIR / "prompts" / "outro_prompts.json").read_text())

segments = plan["segments"]

# Build clip entries: 27 from segment_plan + 5 outro
clips = []
for seg in segments:
    idx = seg["index"]
    start = seg["start"]
    end = seg["end"]
    dur = end - start
    shot = seg.get("shot_type", "singer")
    text = seg.get("text", "")
    ref_path = seg.get("ref_image_path")
    has_ref = ref_path is not None and Path(ref_path).exists()
    ref_file = f"ref_{idx:03d}.jpg" if has_ref else ""

    # Mark QC retries
    tag = " 🔄" if idx in (5, 10, 11, 13, 16) else ""

    # Format time
    def fmt_time(s):
        m = int(s // 60)
        sec = int(s % 60)
        return f"{m}:{sec:02d}"

    clips.append({
        "idx": idx,
        "shot": shot,
        "start": fmt_time(start),
        "end": fmt_time(end),
        "dur": f"{dur:.1f}s",
        "dur_raw": dur,
        "text": text,
        "ref": ref_file,
        "tag": tag,
        "is_outro": False,
    })

# Outro clips (28-32)
outro_starts = [155.83, 166.03, 176.03, 186.23, 196.23]
for i, (start, dur) in enumerate(zip(outro_starts, outro["clip_durations"])):
    idx = 28 + i
    end = start + dur
    beat = outro["beat_numbers"][i]
    shot = outro["beat_types"][i]

    def fmt_time(s):
        m = int(s // 60)
        sec = int(s % 60)
        return f"{m}:{sec:02d}"

    clips.append({
        "idx": idx,
        "shot": shot,
        "start": fmt_time(start),
        "end": fmt_time(end),
        "dur": f"{dur:.1f}s",
        "dur_raw": dur,
        "text": f"Beat {beat} — {shot}",
        "ref": f"ref_{idx:03d}.jpg",
        "tag": " 🆕",
        "is_outro": True,
    })

total_dur = sum(c["dur_raw"] for c in clips)
n_singer = sum(1 for c in clips if c["shot"] == "singer")
n_instrumental = sum(1 for c in clips if c["shot"] == "instrumental")
n_broll = sum(1 for c in clips if c["shot"] == "broll")

# Generate HTML
cards_html = ""
for c in clips:
    outro_marker = ' data-outro="true"' if c["is_outro"] else ""
    ref_html = f'<img src="{c["ref"]}" class="ref-thumb" loading="lazy" alt="ref image">' if c["ref"] else ""
    tag_html = f'<span style="font-size:0.75rem;" title="Regenerated">{c["tag"]}</span>' if c["tag"] else ""

    cards_html += f'''<div class="card{" outro" if c["is_outro"] else ""}" data-segment="{c["idx"]}">
  <div class="card-header">
    <span class="seg-num">#<strong>{c["idx"]}</strong></span>
    <span class="badge {c["shot"]}">{c["shot"]}</span>
    <span class="time">{c["start"]} – {c["end"]}</span>
    <span class="duration">{c["dur"]}</span>
    {tag_html}
  </div>
  <div class="card-body">
    <div class="video-wrap">
      <video src="clip_{c["idx"]:03d}_1080p.mp4" controls preload="metadata" playsinline></video>
    </div>
    <div class="card-meta">
      {ref_html}
      <p class="lyrics">{c["text"]}</p>
    </div>
  </div>
  <div class="card-actions">
    <button class="btn approve" onclick="setStatus({c["idx"]},1)">✓</button>
    <button class="btn reject" onclick="setStatus({c["idx"]},-1)">✗</button>
  </div>
</div>
'''

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Music Video — Clip QC Dashboard (32 clips)</title>
<style>
  :root {{
    --bg: #0a0a0a; --surface: #141414; --border: #2a2a2a;
    --text: #e0e0e0; --muted: #888; --accent: #4a9eff;
    --approved: #22c55e; --rejected: #ef4444; --pending: #f59e0b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    line-height: 1.5; padding: 2rem;
    max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--muted); margin-bottom: 1rem; font-size: 0.95rem; }}
  .stats {{
    display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap;
  }}
  .stat {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.75rem 1.25rem; font-size: 0.85rem;
  }}
  .stat strong {{ color: var(--accent); font-size: 1.2rem; display: block; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; margin-bottom: 1.5rem; overflow: hidden;
    transition: border-color 0.2s;
  }}
  .card.outro {{ border-left: 3px solid var(--accent); }}
  .card.approved {{ border-color: var(--approved); }}
  .card.rejected {{ border-color: var(--rejected); }}
  .card-header {{
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.75rem 1rem; background: rgba(255,255,255,0.03);
    border-bottom: 1px solid var(--border);
  }}
  .seg-num {{ font-size: 1.1rem; min-width: 2.5rem; }}
  .badge {{
    font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 4px;
    text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;
  }}
  .badge.singer {{ background: #4a9eff33; color: #4a9eff; }}
  .badge.instrumental {{ background: #a78bfa33; color: #a78bfa; }}
  .badge.broll {{ background: #f59e0b33; color: #f59e0b; }}
  .time {{ color: var(--muted); font-size: 0.85rem; }}
  .duration {{ color: var(--muted); font-size: 0.85rem; margin-left: auto; }}
  .card-body {{ display: flex; gap: 1rem; padding: 1rem; align-items: flex-start; flex-wrap: wrap; }}
  .video-wrap {{ flex: 1; min-width: 320px; max-width: 640px; }}
  .video-wrap video {{ width: 100%; border-radius: 8px; background: #000; }}
  .card-meta {{ display: flex; gap: 0.75rem; align-items: flex-start; flex: 1; min-width: 200px; }}
  .ref-thumb {{ width: 80px; height: 80px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }}
  .lyrics {{ font-size: 0.85rem; color: var(--muted); line-height: 1.6; }}
  .card-actions {{ display: flex; gap: 0.5rem; padding: 0.5rem 1rem 1rem; justify-content: flex-end; }}
  .btn {{
    border: 1px solid var(--border); background: transparent; color: var(--text);
    width: 36px; height: 36px; border-radius: 8px; cursor: pointer; font-size: 1.1rem;
    transition: all 0.15s;
  }}
  .btn:hover {{ background: rgba(255,255,255,0.1); }}
  .btn.approve:hover {{ border-color: var(--approved); color: var(--approved); }}
  .btn.reject:hover {{ border-color: var(--rejected); color: var(--rejected); }}
  .summary {{
    position: sticky; bottom: 0; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1rem 1.5rem; margin-top: 2rem;
    display: flex; justify-content: space-between; align-items: center;
    backdrop-filter: blur(10px);
  }}
  .summary-stats {{ display: flex; gap: 1.5rem; font-size: 0.9rem; }}
  .summary-stats .ok {{ color: var(--approved); }}
  .summary-stats .bad {{ color: var(--rejected); }}
  .summary-stats .pending {{ color: var(--pending); }}
  .play-all {{
    background: var(--accent); color: #fff; border: none;
    padding: 0.5rem 1.5rem; border-radius: 8px; cursor: pointer; font-size: 0.9rem;
  }}
  .section-label {{
    color: var(--accent); font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.1em; margin: 2rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  @media (max-width: 768px) {{
    body {{ padding: 1rem; }}
    .card-body {{ flex-direction: column; }}
    .video-wrap {{ max-width: 100%; }}
  }}
</style>
</head>
<body>
<h1>Music Video — Clip QC</h1>
<p class="subtitle">32 LTX-2.3 clips · 1920×1080 · 88 BPM · Generated 2026-07-22 · 🔄 = QC retry · 🆕 = outro</p>

<div class="stats">
  <div class="stat"><strong>32</strong>total clips</div>
  <div class="stat"><strong>{n_singer}</strong>singer</div>
  <div class="stat"><strong>{n_instrumental}</strong>instrumental</div>
  <div class="stat"><strong>{n_broll}</strong>b-roll</div>
  <div class="stat"><strong>{total_dur:.0f}s</strong>total duration</div>
</div>

<div class="section-label">Original (beats 1-17)</div>
{cards_html.split('data-outro="true"')[0]}
<div class="section-label">Outro (beats 18-22)</div>
{"".join(c for c in cards_html.split('\n') if 'data-outro' in c or (any(k in c for k in ['28','29','30','31','32']) and 'card' in c))}

<div class="summary">
  <div class="summary-stats">
    <span class="ok" id="approved-count">Approved: 0</span>
    <span class="bad" id="rejected-count">Rejected: 0</span>
    <span class="pending" id="pending-count">Pending: 32</span>
  </div>
  <button class="play-all" onclick="playAll()">▶ Play All</button>
</div>

<script>
const total = 32;
const status = {{}};
let approved = 0, rejected = 0;

function setStatus(idx, val) {{
  const card = document.querySelector(`[data-segment="idx"]`);
  const prev = status[idx];
  if (prev === 1) approved--;
  if (prev === -1) rejected--;

  if (status[idx] === val) {{
    delete status[idx]; card.classList.remove(val === 1 ? 'approved' : 'rejected');
  }} else {{
    status[idx] = val;
    card.classList.remove('approved', 'rejected');
    if (val === 1) {{ approved++; card.classList.add('approved'); }}
    if (val === -1) {{ rejected++; card.classList.add('rejected'); }}
  }}

  document.getElementById('approved-count').textContent = `Approved: ${{approved}}`;
  document.getElementById('rejected-count').textContent = `Rejected: ${{rejected}}`;
  document.getElementById('pending-count').textContent = `Pending: ${{total - approved - rejected}}`;
}}

function playAll() {{
  const videos = document.querySelectorAll('video');
  let i = 0;
  function playNext() {{
    if (i >= videos.length) return;
    videos[i].play();
    videos[i].onended = () => {{ i++; playNext(); }};
  }}
  playNext();
}}
</script>
</body>
</html>'''

# Write to artifact
ARTIFACT.mkdir(parents=True, exist_ok=True)
(ARTIFACT / "clips.html").write_text(html)
print(f"Dashboard written: {len(clips)} clips, {total_dur:.0f}s total")
print(f"  singer={n_singer}, instrumental={n_instrumental}, broll={n_broll}")
