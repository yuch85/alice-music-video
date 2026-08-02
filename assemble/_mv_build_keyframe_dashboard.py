#!/usr/bin/env python3
"""Build keyframe QC review dashboard for music video.

Reads shot metadata from prompt files and shot_list.md,
scans keyframe directory, generates a self-contained HTML dashboard.

Usage:
    uv run python scripts/_mv_build_keyframe_dashboard.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
PROMPTS_DIR = PROJECT_DIR / "prompts"
KEYFRAMES_DIR = PROJECT_DIR / "keyframes"
SHOT_LIST = PROJECT_DIR / "shot_list.md"
OUTPUT = PROJECT_DIR / "keyframe-review-dashboard.html"

# Track which shots were newly generated in 2026-07-31 regen
NEW_SHOTS = {
    "SH-001", "SH-002", "SH-003", "SH-004",
    "SH-005", "SH-006", "SH-007",
    "SH-009", "SH-016",
    "SH-026", "SH-027",
    "SH-030",
    "SH-032", "SH-033", "SH-034",
    "SH-039",
    "SH-041", "SH-043",
    "SH-044", "SH-045", "SH-046", "SH-048",
}


def parse_shot_list() -> dict:
    """Parse shot_list.md for section, timestamp, duration, visualType."""
    text = SHOT_LIST.read_text()
    shots = {}

    current_section = ""
    for line in text.split("\n"):
        # Section headers like "## S0 -- Prologue (Ambient, No Music, 0-30s)"
        sec_match = re.match(r"^## (.+?) -- (.+)", line)
        if sec_match:
            current_section = sec_match.group(2).strip()
            continue

        # Shot entries: "### SH-001 | S0-1 | 0->5s | 5s | NARRATIVE"
        shot_match = re.match(
            r"^### (SH-\d+) \| (.+?) \| (.+?) \| (\S+) \| (\S+)", line
        )
        if shot_match:
            shot_id = shot_match.group(1)
            sub_id = shot_match.group(2)
            timestamp = shot_match.group(3)
            duration = shot_match.group(4)
            visual_type = shot_match.group(5)
            shots[shot_id] = {
                "sub_id": sub_id,
                "section": current_section,
                "timestamp": timestamp,
                "duration": duration,
                "visual_type": visual_type,
            }
    return shots


def parse_prompt_file(path: Path) -> dict:
    """Extract metadata from a prompt file."""
    text = path.read_text()
    result = {}

    shot_type = re.search(r"\*\*Shot type\*\*: (.+)", text)
    result["shot_type"] = shot_type.group(1).strip() if shot_type else ""

    chars = re.search(r"\*\*Character\(s\)\*\*: (.+)", text)
    result["characters"] = chars.group(1).strip() if chars else ""

    weather = re.search(r"\*\*Weather\*\*: (.+)", text)
    result["weather"] = weather.group(1).strip() if weather else ""

    lens = re.search(r"\*\*Lens\*\*: (.+)", text)
    result["lens"] = lens.group(1).strip() if lens else ""

    framing = re.search(r"\*\*Target framing\*\*: (.+)", text)
    if not framing:
        framing = re.search(r"\*\*Framing\*\*: (.+)", text)
    result["framing"] = framing.group(1).strip() if framing else ""

    return result


def build_shot_data() -> list[dict]:
    """Build complete shot data for all 48 shots."""
    shot_list = parse_shot_list()
    shots = []

    for i in range(1, 49):
        shot_id = f"SH-{i:03d}"
        prompt_file = PROMPTS_DIR / f"shot_{shot_id}.md"
        keyframe_file = KEYFRAMES_DIR / f"keyframe_shot_{shot_id}.jpg"

        meta = shot_list.get(shot_id, {})
        prompt_meta = {}
        if prompt_file.exists():
            prompt_meta = parse_prompt_file(prompt_file)

        has_keyframe = keyframe_file.exists()
        is_new = shot_id in NEW_SHOTS

        shots.append({
            "id": shot_id,
            "sub_id": meta.get("sub_id", ""),
            "section": meta.get("section", ""),
            "timestamp": meta.get("timestamp", ""),
            "duration": meta.get("duration", ""),
            "visual_type": meta.get("visual_type", ""),
            "shot_type": prompt_meta.get("shot_type", ""),
            "characters": prompt_meta.get("characters", ""),
            "weather": prompt_meta.get("weather", ""),
            "lens": prompt_meta.get("lens", ""),
            "framing": prompt_meta.get("framing", ""),
            "has_keyframe": has_keyframe,
            "is_new": is_new,
        })

    return shots


def group_by_section(shots: list[dict]) -> dict[str, list[dict]]:
    """Group shots by section, preserving order."""
    grouped = {}
    section_order = []
    for s in shots:
        sec = s["section"] or "Unclassified"
        if sec not in grouped:
            grouped[sec] = []
            section_order.append(sec)
        grouped[sec].append(s)
    # Return ordered
    return {sec: grouped[sec] for sec in section_order}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MV Keyframe QC — Music Video</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #1a1a2e;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
}

/* Header */
.header {
  background: #16213e;
  border-bottom: 2px solid #e94560;
  padding: 20px 32px;
  position: sticky;
  top: 0;
  z-index: 100;
}
.header h1 {
  font-size: 20px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 6px;
}
.status-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 13px;
  flex-wrap: wrap;
}
.status-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
  font-size: 12px;
}
.badge-total { background: #2c3e6d; color: #7ec8e3; }
.badge-approved { background: #1b7a3d; color: #fff; }
.badge-rejected { background: #e94560; color: #fff; }
.badge-pending { background: #555; color: #ccc; }

/* Toolbar */
.toolbar {
  background: #0f1a30;
  padding: 10px 32px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  border-bottom: 1px solid #2a2a4a;
  position: sticky;
  top: 82px;
  z-index: 99;
}
.filter-btn {
  padding: 5px 14px;
  background: transparent;
  color: #888;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.filter-btn:hover { color: #ccc; border-color: #555; }
.filter-btn.active { color: #e94560; border-color: #e94560; }
.toolbar-spacer { flex: 1; }
.toolbar-info { font-size: 12px; color: #666; }

/* Content */
.container {
  max-width: 1500px;
  margin: 0 auto;
  padding: 24px 32px 80px;
}

/* Section grouping */
.section-group { margin-bottom: 36px; }
.section-label {
  font-size: 14px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 14px;
  padding-bottom: 6px;
  border-bottom: 1px solid #2a2a4a;
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.section-count {
  font-size: 12px;
  color: #888;
  font-weight: 400;
}

/* QC Grid */
.qc-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}
@media (max-width: 1200px) { .qc-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 768px) {
  .qc-grid { grid-template-columns: 1fr; }
  .header, .toolbar, .container { padding-left: 16px; padding-right: 16px; }
}

.qc-card {
  background: #16213e;
  border-radius: 6px;
  overflow: hidden;
  border: 2px solid transparent;
  transition: border-color 0.2s;
}
.qc-card.qc-approved { border-color: #2ecc71; }
.qc-card.qc-reject { border-color: #e74c3c; }
.qc-card.qc-new { border-left: 3px solid #f1c40f; }
.qc-card.hidden { display: none; }

.qc-card-img {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #0a0a1a;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.qc-card-img img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.qc-card-img .placeholder {
  color: #444;
  font-size: 12px;
  text-align: center;
}

.qc-card-body { padding: 10px 12px; }

.qc-card-top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 4px;
}
.qc-card-top .shot-id {
  font-size: 14px;
  font-weight: 700;
  color: #fff;
}
.qc-card-top .shot-time {
  font-size: 11px;
  color: #888;
}
.qc-card-sub {
  font-size: 11px;
  color: #666;
  margin-bottom: 4px;
}

.qc-tags {
  display: flex;
  gap: 4px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.qc-tag {
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  padding: 1px 5px;
  border-radius: 3px;
  letter-spacing: 0.04em;
}
.tag-narrative { background: #3d2c1e; color: #e8a84c; }
.tag-singer { background: #2c3e6d; color: #7ec8e3; }
.tag-broll { background: #1e3d2c; color: #6ddb8f; }
.tag-convergence { background: #3d1e3d; color: #d47ae5; }
.tag-new { background: #3d3d1e; color: #f1c40f; }

.qc-card-meta {
  font-size: 11px;
  color: #999;
  margin-bottom: 8px;
  line-height: 1.4;
}
.qc-card-meta strong { color: #bbb; font-weight: 500; }

/* QC Controls */
.qc-actions {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
}
.qc-btn {
  flex: 1;
  padding: 5px 4px;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  background: #0f1a30;
  color: #aaa;
}
.qc-btn:hover { border-color: #555; }
.qc-btn.approve { color: #2ecc71; border-color: #2ecc7144; }
.qc-btn.approve:hover, .qc-btn.approve.selected { background: #2ecc7122; color: #2ecc71; border-color: #2ecc71; }
.qc-btn.reject { color: #e74c3c; border-color: #e74c3c44; }
.qc-btn.reject:hover, .qc-btn.reject.selected { background: #e74c3c22; color: #e74c3c; border-color: #e74c3c; }

.reject-reason {
  display: none;
  margin-bottom: 6px;
}
.reject-reason.show { display: block; }
.reject-reason select {
  width: 100%;
  padding: 4px 6px;
  background: #0f1a30;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  color: #e0e0e0;
  font-size: 11px;
}

.qc-notes {
  width: 100%;
  min-height: 36px;
  background: #0f1a30;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  color: #e0e0e0;
  font-family: inherit;
  font-size: 11px;
  padding: 6px 8px;
  resize: vertical;
}
.qc-notes:focus { outline: none; border-color: #e94560; }

/* Export Section */
.export-section {
  margin-top: 40px;
  padding: 24px 32px;
  background: #16213e;
  border-radius: 8px;
}
.export-title {
  font-size: 15px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 12px;
}
.export-btn {
  padding: 10px 24px;
  background: #e94560;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  margin: 0 6px 10px;
  transition: background 0.15s;
}
.export-btn:hover { background: #d63050; }
.export-btn.secondary {
  background: #2a2a4a;
}
.export-btn.secondary:hover { background: #3a3a5a; }

.export-preview {
  margin-top: 16px;
  max-height: 400px;
  overflow: auto;
  background: #0f1a30;
  border: 1px solid #2a2a4a;
  border-radius: 6px;
  padding: 16px;
  display: none;
}
.export-preview pre {
  font-size: 11px;
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  white-space: pre-wrap;
  word-break: break-all;
  color: #bbb;
}

.toast {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: #2ecc71;
  color: #fff;
  padding: 10px 20px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  opacity: 0;
  transform: translateY(10px);
  transition: opacity 0.2s, transform 0.2s;
  pointer-events: none;
  z-index: 999;
}
.toast.show { opacity: 1; transform: translateY(0); }

/* Summary bar */
.summary-bar {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
  padding: 10px 16px;
  background: #0f1a30;
  border-radius: 6px;
  font-size: 13px;
}
.summary-item { display: flex; align-items: center; gap: 4px; }
.summary-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.dot-approved { background: #2ecc71; }
.dot-rejected { background: #e74c3c; }
.dot-pending { background: #555; }
.dot-new { background: #f1c40f; }
</style>
</head>
<body>

<div class="header">
  <h1>MV Keyframe QC Review &mdash; Music Video</h1>
  <div class="status-bar">
    <span><span class="status-badge badge-total" id="badge-total">48</span> total</span>
    <span><span class="status-badge badge-approved" id="badge-approved">0</span> approved</span>
    <span><span class="status-badge badge-rejected" id="badge-rejected">0</span> rejected</span>
    <span><span class="status-badge badge-pending" id="badge-pending">48</span> pending</span>
  </div>
</div>

<div class="toolbar">
  <button class="filter-btn active" data-filter="all">All</button>
  <button class="filter-btn" data-filter="pending">Pending</button>
  <button class="filter-btn" data-filter="new">New Only</button>
  <button class="filter-btn" data-filter="approved">Approved</button>
  <button class="filter-btn" data-filter="rejected">Rejected</button>
  <span class="toolbar-spacer"></span>
  <span class="toolbar-info" id="filter-info">Showing 48/48</span>
</div>

<div class="container">
  <div class="summary-bar">
    <div class="summary-item"><span class="summary-dot dot-approved"></span> Approved</div>
    <div class="summary-item"><span class="summary-dot dot-rejected"></span> Rejected</div>
    <div class="summary-item"><span class="summary-dot dot-pending"></span> Pending</div>
    <div class="summary-item"><span class="summary-dot dot-new"></span> Newly Generated</div>
  </div>
  <div id="shot-grid"></div>

  <div class="export-section">
    <div class="export-title">Export QC Decisions</div>
    <button class="export-btn" onclick="exportJSON()">Preview Decisions</button>
    <button class="export-btn secondary" onclick="copyJSON()">Copy JSON</button>
    <button class="export-btn secondary" onclick="downloadJSON()">Download JSON</button>
    <button class="export-btn secondary" onclick="downloadCSV()">Download CSV</button>
    <div class="export-preview" id="export-preview"><pre id="export-text"></pre></div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const SHOTS = SHOT_DATA_PLACEHOLDER;

// State
const decisions = {};
let currentFilter = "all";

// Load from localStorage
try {
  const saved = localStorage.getItem("mv-qc-decisions");
  if (saved) Object.assign(decisions, JSON.parse(saved));
} catch(e) {}

function saveDecisions() {
  localStorage.setItem("mv-qc-decisions", JSON.stringify(decisions));
  updateCounts();
}

function updateCounts() {
  let approved = 0, rejected = 0, pending = 0;
  SHOTS.forEach(s => {
    const d = decisions[s.id];
    if (d && d.status === "approved") approved++;
    else if (d && d.status === "rejected") rejected++;
    else pending++;
  });
  document.getElementById("badge-approved").textContent = approved;
  document.getElementById("badge-rejected").textContent = rejected;
  document.getElementById("badge-pending").textContent = pending;
}

function setDecision(shotId, status) {
  if (!decisions[shotId]) decisions[shotId] = {};
  decisions[shotId].status = status;

  // Update button states
  const card = document.getElementById("card-" + shotId);
  card.querySelector(".qc-btn.approve").classList.toggle("selected", status === "approved");
  card.querySelector(".qc-btn.reject").classList.toggle("selected", status === "rejected");
  card.classList.toggle("qc-approved", status === "approved");
  card.classList.toggle("qc-reject", status === "rejected");

  // Show/hide reject reason
  const reasonDiv = card.querySelector(".reject-reason");
  reasonDiv.classList.toggle("show", status === "rejected");

  saveDecisions();
  applyFilter();
}

function setRejectReason(shotId, reason) {
  if (!decisions[shotId]) decisions[shotId] = { status: "rejected" };
  decisions[shotId].reject_reason = reason;
  saveDecisions();
}

function setNotes(shotId, notes) {
  if (!decisions[shotId]) decisions[shotId] = {};
  decisions[shotId].notes = notes || null;
  saveDecisions();
}

function applyFilter() {
  let visible = 0;
  SHOTS.forEach(s => {
    const card = document.getElementById("card-" + s.id);
    if (!card) return;
    const d = decisions[s.id];
    let show = false;

    switch(currentFilter) {
      case "all": show = true; break;
      case "pending": show = (!d || d.status !== "approved" && d.status !== "rejected"); break;
      case "new": show = s.is_new; break;
      case "approved": show = d && d.status === "approved"; break;
      case "rejected": show = d && d.status === "rejected"; break;
    }

    card.classList.toggle("hidden", !show);
    if (show) visible++;
  });
  document.getElementById("filter-info").textContent = "Showing " + visible + "/" + SHOTS.length;
}

function renderDashboard() {
  const grid = document.getElementById("shot-grid");
  let html = "";

  // Group by section
  const sections = {};
  SHOTS.forEach(s => {
    const sec = s.section || "Unclassified";
    if (!sections[sec]) sections[sec] = [];
    sections[sec].push(s);
  });

  for (const [sec, shots] of Object.entries(sections)) {
    html += `<div class="section-group">`;
    html += `<div class="section-label">${escHtml(sec)} <span class="section-count">(${shots.length} shots)</span></div>`;
    html += `<div class="qc-grid">`;

    shots.forEach(s => {
      const d = decisions[s.id] || {};
      const typeClass = "tag-" + s.visual_type.toLowerCase();
      const isNewTag = s.is_new ? '<span class="qc-tag tag-new">NEW</span>' : '';

      html += `
      <div class="qc-card ${s.is_new ? 'qc-new' : ''} ${d.status === 'approved' ? 'qc-approved' : ''} ${d.status === 'rejected' ? 'qc-reject' : ''}" id="card-${s.id}">
        <div class="qc-card-img">
          ${s.has_keyframe
            ? `<img src="keyframes/keyframe_shot_${s.id}.jpg" alt="${s.id}">`
            : `<div class="placeholder">Keyframe<br>not found</div>`}
        </div>
        <div class="qc-card-body">
          <div class="qc-card-top">
            <span class="shot-id">${s.id}</span>
            <span class="shot-time">${escHtml(s.timestamp)} &middot; ${s.duration}</span>
          </div>
          <div class="qc-card-sub">${escHtml(s.sub_id)}</div>
          <div class="qc-tags">
            <span class="qc-tag ${typeClass}">${s.visual_type}</span>
            ${isNewTag}
          </div>
          <div class="qc-card-meta">
            <strong>${escHtml(s.characters)}</strong> &middot; ${escHtml(s.framing)} ${escHtml(s.lens)}
          </div>
          <div class="qc-actions">
            <button class="qc-btn approve ${d.status === 'approved' ? 'selected' : ''}" onclick="setDecision('${s.id}', 'approved')">✓ Approve</button>
            <button class="qc-btn reject ${d.status === 'rejected' ? 'selected' : ''}" onclick="setDecision('${s.id}', 'rejected')">✗ Reject</button>
          </div>
          <div class="reject-reason ${d.status === 'rejected' ? 'show' : ''}">
            <select onchange="setRejectReason('${s.id}', this.value)">
              <option value="">Reject reason...</option>
              <option value="too_bright" ${d.reject_reason === 'too_bright' ? 'selected' : ''}>Too bright / wrong lighting</option>
              <option value="wrong_angle" ${d.reject_reason === 'wrong_angle' ? 'selected' : ''}>Wrong angle / framing</option>
              <option value="wrong_character" ${d.reject_reason === 'wrong_character' ? 'selected' : ''}>Wrong character / face</option>
              <option value="composition" ${d.reject_reason === 'composition' ? 'selected' : ''}>Wrong composition</option>
              <option value="inconsistent" ${d.reject_reason === 'inconsistent' ? 'selected' : ''}>Inconsistent with other shots</option>
              <option value="artifact" ${d.reject_reason === 'artifact' ? 'selected' : ''}>Visual artifacts</option>
              <option value="other" ${d.reject_reason === 'other' ? 'selected' : ''}>Other</option>
            </select>
          </div>
          <textarea class="qc-notes" placeholder="Notes..." oninput="setNotes('${s.id}', this.value)">${escHtml(d.notes || '')}</textarea>
        </div>
      </div>`;
    });

    html += `</div></div>`;
  }

  grid.innerHTML = html;
  updateCounts();
  applyFilter();
}

function escHtml(s) {
  const div = document.createElement("div");
  div.textContent = s || "";
  return div.innerHTML;
}

// ===== EXPORT =====
let lastExport = null;

function gatherDecisions() {
  const results = [];
  SHOTS.forEach(s => {
    const d = decisions[s.id];
    if (d && (d.status === "approved" || d.status === "rejected")) {
      results.push({
        shot_id: s.id,
        section: s.section,
        timestamp: s.timestamp,
        duration: s.duration,
        visual_type: s.visual_type,
        characters: s.characters,
        status: d.status,
        reject_reason: d.reject_reason || null,
        notes: d.notes || null,
        is_new: s.is_new,
      });
    }
  });
  return {
    exported_at: new Date().toISOString(),
    project: "project-name",
    total_shots: SHOTS.length,
    decisions_count: results.length,
    qc_results: results,
  };
}

function exportJSON() {
  lastExport = gatherDecisions();
  const preview = document.getElementById("export-preview");
  const text = document.getElementById("export-text");
  text.textContent = JSON.stringify(lastExport, null, 2);
  preview.style.display = "block";
  showToast("Decisions exported — " + lastExport.decisions_count + " decisions");
}

function copyJSON() {
  if (!lastExport) exportJSON();
  navigator.clipboard.writeText(JSON.stringify(lastExport, null, 2)).then(() => {
    showToast("Copied to clipboard");
  }).catch(() => {
    const ta = document.createElement("textarea");
    ta.value = JSON.stringify(lastExport, null, 2);
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    showToast("Copied to clipboard");
  });
}

function downloadJSON() {
  if (!lastExport) exportJSON();
  const blob = new Blob([JSON.stringify(lastExport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mv-keyframe-qc-" + new Date().toISOString().slice(0, 10) + ".json";
  a.click();
  URL.revokeObjectURL(url);
  showToast("JSON downloaded");
}

function downloadCSV() {
  if (!lastExport) exportJSON();
  const headers = ["shot_id","section","timestamp","duration","visual_type","characters","status","reject_reason","notes","is_new"];
  const rows = lastExport.qc_results.map(r => headers.map(h => {
    let v = String(r[h] ?? "");
    if (v.includes(",") || v.includes('"') || v.includes("\n")) v = '"' + v.replace(/"/g, '""') + '"';
    return v;
  }).join(","));
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mv-keyframe-qc-" + new Date().toISOString().slice(0, 10) + ".csv";
  a.click();
  URL.revokeObjectURL(url);
  showToast("CSV downloaded");
}

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2200);
}

// ===== FILTER BUTTONS =====
document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    applyFilter();
  });
});

// ===== INIT =====
renderDashboard();
</script>
</body>
</html>"""


def main() -> int:
    shots = build_shot_data()
    shot_json = json.dumps(shots, ensure_ascii=False, indent=2)

    html = HTML_TEMPLATE.replace("SHOT_DATA_PLACEHOLDER", shot_json)
    OUTPUT.write_text(html, encoding="utf-8")

    # Stats
    with_kf = sum(1 for s in shots if s["has_keyframe"])
    new_count = sum(1 for s in shots if s["is_new"])
    print(f"Dashboard: {OUTPUT}")
    print(f"  Shots: {len(shots)}, Keyframes: {with_kf}, New: {new_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
