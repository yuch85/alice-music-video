#!/usr/bin/env python3
"""Per-segment reference image generation (Plan 09.9 Plan 04).

Module is kept <= 400 lines per STYLE.md (approved 400 ceiling deviation
from the 300 default). Logic is byte-for-byte identical to the original
`generate_music_video_pipeline.py` block this module was split from.

BLOCKER 1: internal ComfyUI-helper calls are routed via the `mv_comfyui.<name>`
module attribute so `mock.patch("mv_comfyui.<name>")` intercepts them.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import mv_comfyui
from mv_segment import SegmentPlan
from mv_shot import _assign_shot_type, _get_pose_variation

logger = logging.getLogger(__name__)


def _write_references_manifest(
    output_path: Path,
    references: list[Path],
    segment_starts: list[float],
) -> Path:
    """Emit ``references_manifest.json`` for manual QA of per-clip references.

    Plan 09.9-25-04 (D-04 option): when the caller supplies an explicit
    per-segment reference list, this surfaces the chosen reference per clip so
    the user can review each starting reference. It NEVER silently picks references — the
    list is supplied by the caller.

    Args:
        output_path: Output directory (manifest is written here).
        references: Per-clip reference paths (aligned to segment order).
        segment_starts: Start time (seconds) of each segment, aligned to
            ``references``.

    Returns the manifest Path.
    """
    manifest = [
        {
            "clip_index": i,
            "reference_path": str(ref),
            "segment_start_s": round(float(segment_starts[i]), 3) if i < len(segment_starts) else None,
        }
        for i, ref in enumerate(references)
    ]
    manifest_path = Path(output_path) / "references_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _generate_segment_ref(
    canonical_portrait: Path,
    scene_prompt: str,
    segment_text: str,
    segment_index: int,
    output_dir: Path,
    variation: str = "",
) -> Path | None:
    """Generate one reference image for a segment via Qwen I2I.

    Uses the canonical portrait as input, varies the prompt with
    a pose variation and segment lyrics for uniqueness.

    Args:
        canonical_portrait: Reference portrait path.
        scene_prompt: Base scene description.
        segment_text: Segment lyrics/text for context.
        segment_index: 0-based segment index.
        output_dir: Output directory for ComfyUI input files.
        variation: Optional composition variation string for candidate diversity.

    Returns output JPG path, or None if generation failed.
    """
    pose = _get_pose_variation(segment_index)
    # Build per-segment prompt: scene + lyrics context + pose + variation
    extra = f". {variation}" if variation else ""
    seg_prompt = f"{scene_prompt}. {segment_text[:80]}. {pose}{extra}"

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu-manager"))
        from workflows.workflows import build_qwen_gguf_i2i_workflow
    except ImportError as e:
        logger.warning("Cannot import Qwen I2I for segment ref %d: %s", segment_index, e)
        return None

    # VRAM gate with circuit breaker
    if not mv_comfyui._check_vram_gate():
        logger.warning("Skipping segment ref %d — VRAM gate failed", segment_index)
        return None

    # Prepare input
    comfyui_input = Path(mv_comfyui.COMFYUI_OUTPUT_DIR) / "input"
    comfyui_input.mkdir(parents=True, exist_ok=True)
    ts = int(time.time()) + segment_index
    ref_filename = f"seg_ref_{segment_index:03d}_{ts}.jpg"

    input_path = comfyui_input / ref_filename
    shutil.copy2(canonical_portrait, input_path)

    try:
        # Build workflow — 4-step Lightning, same as scene portrait
        # image_paths is a list of filenames in ComfyUI input/
        workflow = build_qwen_gguf_i2i_workflow(
            prompt=seg_prompt,
            image_paths=[ref_filename],
            steps=4,
        )

        prompt_id = mv_comfyui._queue_workflow(workflow)
        logger.info("Segment ref %d queued (prompt_id=%s)", segment_index, prompt_id)
        history = mv_comfyui._poll_completion(prompt_id, timeout=300)

        # Qwen GGUF I2I saves with prefix "alice_i2i_gguf"
        output_path = mv_comfyui._find_output_file(history, "alice_i2i_gguf", "jpg")
        logger.info("Segment ref %d generated: %s", segment_index, output_path)
        return output_path

    except Exception as e:
        logger.warning("Segment ref %d generation failed: %s", segment_index, e)
        return None

    finally:
        try:
            input_path.unlink()
        except OSError:
            pass


def _generate_segment_refs(
    canonical_portrait: Path,
    scene_prompt: str,
    segments: list[ClipSegment],
    output_dir: Path,
    refined_prompts: list[dict[str, str]] | None = None,
    candidates_per_segment: int = 3,
) -> list[SegmentPlan]:
    """Generate per-segment reference images and return segment plans.

    For each segment, generates ``candidates_per_segment`` reference images
    via Qwen I2I with varied poses and compositions. Shot types are assigned
    based on segment content and position (singer, broll, instrumental).

    If refined_prompts is provided, uses image_prompt for ref generation
    instead of scene_prompt + pose_variation.

    Candidates are stored under ``refs/candidates/ref_<N>_<a/b/c>.jpg`` for
    the approval dashboard (approval dashboard).
    """
    refs_dir = output_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = refs_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Pose variation templates for candidate diversity.
    candidate_variations = [
        "slightly turned to the left, natural stance",
        "facing forward, relaxed posture",
        "slightly turned to the right, one hand gesturing",
    ]

    plans: list[SegmentPlan] = []
    prev_end = 0.0  # track previous segment end for gap detection

    for i, seg in enumerate(segments):
        # Check circuit breaker before starting each ref
        if mv_comfyui.comfyui_client._consecutive_failures >= mv_comfyui.MAX_CONSECUTIVE_COMFYUI_FAILURES:
            logger.error(
                "Circuit breaker OPEN — aborting segment ref generation at segment %d/%d",
                i + 1, len(segments),
            )
            break

        # Assign shot type based on content/position — but honor an explicit
        # filler shot_type (e.g. "broll") from the coverage step. Because the
        # coverage step pre-fills the gap, the following lyric segment's prev_end
        # is contiguous (gap=0), so _assign_shot_type now correctly labels it
        # "singer" instead of mislabeling it "broll" (seg-7 fix).
        shot_type = seg.shot_type or _assign_shot_type(seg, i, len(segments), prev_end)

        logger.info("Generating refs for segment %d: [%s-%s] shot_type=%s %s...",
                     i + 1, seg.start, seg.end, shot_type, seg.text[:40])

        # Determine prompt for this segment
        seg_prompt = scene_prompt
        if refined_prompts and i < len(refined_prompts):
            seg_prompt = refined_prompts[i]["image_prompt"]

        # Generate candidate refs for singer/instrumental shots; skip for broll/black
        candidate_paths: list[str] = []
        if shot_type not in ("broll", "black"):
            num_candidates = min(candidates_per_segment, len(candidate_variations))
            for c in range(num_candidates):
                variation = candidate_variations[c]
                ref_path = _generate_segment_ref(
                    canonical_portrait, seg_prompt, seg.text, i, output_dir,
                    variation=variation,
                )
                if ref_path:
                    candidate_name = f"ref_{i + 1:03d}_{chr(97 + c)}.jpg"
                    local_candidate = candidates_dir / candidate_name
                    shutil.copy2(ref_path, local_candidate)
                    candidate_paths.append(str(local_candidate))

        # Primary ref is the first candidate (or None for broll/black)
        primary_ref = candidate_paths[0] if candidate_paths else None

        plans.append(SegmentPlan(
            index=i + 1,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            shot_type=shot_type,
            prompt=seg_prompt,
            ref_image_path=primary_ref,
            candidate_paths=candidate_paths,
            status="pending",
        ))

        prev_end = seg.end

    total_candidates = sum(len(p.candidate_paths) for p in plans)
    logger.info("Generated %d candidates for %d segments", total_candidates, len(plans))
    return plans


def _generate_approval_dashboard(
    plans: list[SegmentPlan],
    output_dir: Path,
    scene_prompt: str,
) -> Path:
    """Generate a self-contained HTML approval dashboard for reference images.

    Approval dashboard: renders per-segment sections with metadata,
    candidate images side-by-side, and radio buttons for selection.
    Includes a 'Export Selections' button that compiles choices to JSON.

    Args:
        plans: Segment plans with candidate_paths.
        output_dir: Output directory (dashboard written here).
        scene_prompt: Base scene description for display.

    Returns the dashboard HTML Path.
    """
    # Build JSON data for the dashboard.
    segments_data: list[dict[str, object]] = []
    for p in plans:
        if not p.candidate_paths:
            continue
        segments_data.append({
            "index": p.index,
            "start": round(p.start, 2),
            "end": round(p.end, 2),
            "duration": round(p.end - p.start, 2),
            "shot_type": p.shot_type,
            "text": p.text[:120] if p.text else "",
            "prompt": p.prompt[:200] if p.prompt else "",
            "candidates": [
                {"label": chr(97 + i), "path": f"refs/candidates/{Path(c).name}"}
                for i, c in enumerate(p.candidate_paths)
            ],
        })

    # Generate self-contained HTML.
    html_content = _build_dashboard_html(segments_data, scene_prompt)
    dashboard_path = output_dir / "approval_dashboard.html"
    dashboard_path.write_text(html_content)
    logger.info("Approval dashboard written: %s (%d segments with candidates)",
                dashboard_path, len(segments_data))
    return dashboard_path


def _build_dashboard_html(
    segments_data: list[dict[str, object]],
    scene_prompt: str,
) -> str:
    """Build self-contained HTML for the approval dashboard.

    Uses string replacement to avoid f-string/JavaScript curly brace conflicts.
    """
    segments_json = json.dumps(segments_data, indent=2)
    escaped_prompt = scene_prompt.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Music Video - Reference Image Approval</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 2rem; }
h1 { text-align: center; margin-bottom: 0.5rem; color: #fff; font-size: 1.5rem; }
.subtitle { text-align: center; color: #888; margin-bottom: 2rem; font-size: 0.9rem; }
.segment { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
.segment-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.segment-index { font-size: 1.2rem; font-weight: bold; color: #e94560; }
.segment-meta { font-size: 0.85rem; color: #888; }
.segment-meta span { margin-right: 1rem; }
.candidates { display: flex; gap: 1rem; flex-wrap: wrap; }
.candidate { flex: 1; min-width: 250px; max-width: 400px; }
.candidate img { width: 100%; border-radius: 4px; border: 2px solid transparent; }
.candidate.selected img { border-color: #e94560; }
.candidate label { display: block; margin-top: 0.5rem; text-align: center; cursor: pointer; }
.candidate input[type="radio"] { display: none; }
.candidate .radio-label { padding: 0.5rem; border-radius: 4px; background: #0f3460; transition: background 0.2s; }
.candidate.selected .radio-label { background: #e94560; color: #fff; }
.prompt-preview { font-size: 0.8rem; color: #666; margin-top: 0.5rem; font-style: italic; }
.export-btn { position: fixed; bottom: 2rem; right: 2rem; padding: 1rem 2rem; background: #e94560; color: #fff; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; z-index: 100; }
.export-btn:hover { background: #c81e45; }
#output { display: none; position: fixed; bottom: 2rem; right: 10rem; background: #000; color: #0f0; padding: 1rem; border-radius: 8px; max-width: 600px; max-height: 400px; overflow: auto; font-family: monospace; font-size: 0.8rem; z-index: 99; }
</style>
</head>
<body>
<h1>Reference Image Approval Dashboard</h1>
<p class="subtitle">SCENE_PROMPT_PLACEHOLDER</p>
<div id="segments"></div>
<button class="export-btn" onclick="exportSelections()">Export Selections</button>
<pre id="output"></pre>
<script>
const segments = SEGMENTS_JSON_PLACEHOLDER;
function render() {
    const container = document.getElementById('segments');
    segments.forEach((seg, idx) => {
        const div = document.createElement('div');
        div.className = 'segment';
        var candidatesHtml = '';
        seg.candidates.forEach(function(c, ci) {
            candidatesHtml += '<div class="candidate" id="seg-' + idx + '-cand-' + ci + '">' +
                '<img src="' + c.path + '" alt="Candidate ' + c.label + '">' +
                '<label><input type="radio" name="seg-' + idx + '" value="' + c.label + '" onchange="selectCandidate(' + idx + ', \'' + c.label + '\')">' +
                '<span class="radio-label">Select ' + c.label.toUpperCase() + '</span></label></div>';
        });
        div.innerHTML = '<div class="segment-header">' +
            '<span class="segment-index">Segment ' + seg.index + '</span>' +
            '<div class="segment-meta"><span>' + seg.start + 's-' + seg.end + 's (' + seg.duration + 's)</span>' +
            '<span>' + seg.shot_type + '</span></div></div>' +
            '<div class="candidates">' + candidatesHtml + '</div>' +
            '<div class="prompt-preview">' + seg.prompt + '</div>';
        container.appendChild(div);
    });
}
function selectCandidate(idx, label) {
    var candidates = document.querySelectorAll('[name="seg-' + idx + '"]');
    candidates.forEach(function(r) {
        r.closest('.candidate').classList.remove('selected');
    });
    document.querySelector('input[name="seg-' + idx + '"][value="' + label + '"]').closest('.candidate').classList.add('selected');
}
function exportSelections() {
    var selections = segments.map(function(seg, idx) {
        var selected = document.querySelector('input[name="seg-' + idx + '"]:checked');
        return {
            segment: seg.index,
            start: seg.start,
            end: seg.end,
            shot_type: seg.shot_type,
            approved_candidate: selected ? selected.value : "a",
            candidate_path: 'refs/candidates/ref_' + String(seg.index).padStart(3, '0') + '_' + (selected ? selected.value : 'a') + '.jpg'
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

    html = html.replace("SCENE_PROMPT_PLACEHOLDER", escaped_prompt)
    html = html.replace("SEGMENTS_JSON_PLACEHOLDER", segments_json)
    return html
