#!/usr/bin/env python3
"""Segment dataclasses + word-grouping + vocal classification + segment-plan I/O.

Functions: _group_words_into_segments, _classify_vocal_regions,
_merge_short_segments, _fill_coverage_gaps, _split_long_segments,
_validate_segment_plan, _write_segment_plan, _read_segment_plan.

Kept <= 400 lines per STYLE.md (YC-approved 400 ceiling deviation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import mv_mvconst

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    """Single word-level transcription result from Whisper."""

    text: str
    start: float
    end: float


@dataclass
class ClipSegment:
    """Grouped segment for one LTX-2 clip (4-10s)."""

    start: float
    end: float
    text: str
    duration: float
    words: list[WordSegment] = field(default_factory=list)
    measured_duration: float | None = None  # actual video duration after LTX-2 8k+1 quantization
    shot_type: str | None = None  # explicit shot type (e.g. "broll" filler from coverage)


# ── Controlled mode types (Phase 09.9 Plan 04) ──

class ShotType(str, Enum):
    """Shot type for a segment in controlled mode."""

    SINGER = "singer"  # Singer singing lyrics (LTX-2 I2AV + dialogue)
    BROLL = "broll"  # B-roll footage (LTX-2 T2V, no ref, no dialogue)
    INSTRUMENTAL = "instrumental"  # Instrumental mood piece (I2V/T2V, no dialogue)
    BLACK = "black"  # Black frame / fade (FFmpeg only, no LTX-2)


@dataclass
class SegmentPlan:
    """Per-segment plan for controlled mode."""

    index: int
    start: float
    end: float
    text: str  # lyrics for this segment
    shot_type: str  # "singer" | "broll" | "instrumental" | "black"
    prompt: str  # per-segment prompt override
    ref_image_path: str | None = None  # generated ref image (singer/instrumental)
    status: str = "pending"  # "pending" | "approved" | "rejected" | "generated"
    pipeline_engine: str | None = None  # "humo" | "ltx2" | None (classifier decides)
    candidate_paths: list[str] = field(default_factory=list)  # all candidate refs

def _group_words_into_segments(
    words: list[WordSegment], max_segment_s: float = 10.0
) -> list[ClipSegment]:
    """Group word-level timestamps into segments up to max_segment_s.

    Flushes on max duration exceeded or gaps > 2s. Single-word segments
    are kept as-is (may be < 4s). Last segment flushes even if short.
    """
    if not words:
        return []

    segments: list[ClipSegment] = []
    current_words: list[WordSegment] = []
    current_start = words[0].start

    for word in words:
        if current_words and (word.start - current_words[-1].end) > 2.0:
            if current_words:
                seg = _flush_segment(current_words, current_start)
                segments.append(seg)
            current_words = [word]
            current_start = word.start
            continue

        current_words.append(word)
        current_end = word.end
        current_duration = current_end - current_start

        if current_duration >= max_segment_s and len(current_words) > 1:
            seg = _flush_segment(current_words, current_start)
            segments.append(seg)
            current_words = []
            current_start = word.end

    if current_words:
        seg = _flush_segment(current_words, current_start)
        segments.append(seg)

    logger.info("Grouped %d words into %d segments", len(words), len(segments))
    return segments

def _flush_segment(
    words: list[WordSegment], start: float
) -> ClipSegment:
    """Flush accumulated words into a ClipSegment."""
    end = words[-1].end
    text = " ".join(w.text for w in words)
    return ClipSegment(
        start=start,
        end=end,
        text=text,
        duration=end - start,
        words=words,
    )

def _split_long_segments(
    segments: list[ClipSegment], max_s: int = 18
) -> list[ClipSegment]:
    """Split segments longer than ``max_s`` into contiguous sub-segments.

    ``max_s`` mirrors ``mv_clip.LTX2_MAX_LENGTH_S`` (18) as a plain literal
    to avoid circular imports. Word-bearing lyric segments (<= 10s) are kept
    intact. Wordless segments (e.g. B-roll fillers) are split.
    """
    if max_s <= 0:
        return list(segments)
    out: list[ClipSegment] = []
    for seg in segments:
        dur = seg.end - seg.start
        if seg.words or dur <= max_s:
            out.append(seg)
            continue
        n = int(dur // max_s) + (1 if dur % max_s > 1e-9 else 0)
        for k in range(n):
            s = seg.start + k * max_s
            e = seg.start + (k + 1) * max_s if k < n - 1 else seg.end
            out.append(ClipSegment(
                start=s, end=e, text=seg.text,
                duration=e - s, words=[], shot_type=seg.shot_type,
            ))
    return out

def _fill_coverage_gaps(
    segments: list[ClipSegment],
    audio_duration: float,
    broll_prompts: list[str],
    gap_threshold: float = 0.5,
) -> list[ClipSegment]:
    """Insert B-roll fillers so the plan tiles 0 -> audio_duration contiguously.

    Gaps > ``gap_threshold`` between segments get a broll filler. The trailing
    region to ``audio_duration`` also gets a filler. B-roll prompts cycle.
    Returns unchanged if ``audio_duration`` is 0/None. NaN/negative starts
    are clamped (threat T-09.9-12-01).
    """
    if not audio_duration or audio_duration <= 0:
        return list(segments)

    result: list[ClipSegment] = []
    prev_end = 0.0
    prompt_idx = 0
    n_prompts = len(broll_prompts)

    for seg in segments:
        seg_start = float(seg.start)
        if seg_start != seg_start or seg_start < 0.0:  # NaN / negative
            seg_start = max(0.0, prev_end)
        gap = seg_start - prev_end
        if gap > gap_threshold:
            prompt = broll_prompts[prompt_idx % n_prompts] if n_prompts else ""
            result.append(ClipSegment(
                start=prev_end, end=seg_start, text=prompt,
                duration=seg_start - prev_end, words=[], shot_type="broll",
            ))
            prompt_idx += 1
        result.append(seg)
        prev_end = float(seg.end)

    if audio_duration - prev_end > gap_threshold:
        prompt = broll_prompts[prompt_idx % n_prompts] if n_prompts else ""
        result.append(ClipSegment(
            start=prev_end, end=audio_duration, text=prompt,
            duration=audio_duration - prev_end, words=[], shot_type="broll",
        ))

    return result

def _classify_vocal_regions(
    words: list[WordSegment],
    audio_duration: float,
    gap_threshold: float = 3.0,
) -> tuple[list[WordSegment], float]:
    """Classify vocal vs instrumental regions from Whisper word timestamps.

    End of vocal timeline is authoritative for instrumental outro boundary.
    Interior gaps > gap_threshold split the timeline into regions.
    Returns (vocal_words, last_vocal_end).

    Filters out Whisper hallucinations — words with near-zero duration
    (start == end or end - start < 0.1s) are low-confidence fabrications
    common on instrumental passages (e.g. "ご視聴ありがとうございました"
    hallucinated on Japanese song outros).
    """
    if not words:
        return [], audio_duration

    # Filter out hallucinated words (zero/near-zero duration)
    MIN_WORD_DURATION_S = 0.1
    filtered_words = [w for w in words if w.end - w.start >= MIN_WORD_DURATION_S]
    hallucinated = len(words) - len(filtered_words)
    if hallucinated > 0:
        logger.info(
            "Filtered %d hallucinated words (duration < %.1fs)",
            hallucinated, MIN_WORD_DURATION_S,
        )

    # Split words into contiguous vocal regions separated by gaps.
    # This also handles Whisper hallucinations on instrumental passages:
    # e.g. "ご視聴ありがとうございました" can appear with real timestamps
    # on instrumental outros, separated from the last real vocal by a
    # large gap (~28s). Such trailing regions are discarded.
    regions: list[list[WordSegment]] = []
    region: list[WordSegment] = []
    for word in sorted(filtered_words, key=lambda w: w.start):
        if region and (word.start - region[-1].end) > gap_threshold:
            regions.append(region)
            region = []
        region.append(word)
    if region:
        regions.append(region)

    # Discard trailing hallucinated regions. The largest contiguous region
    # is the main singing body. Any region separated from it by a large gap
    # (>10s) and having few words (<=5) is a Whisper hallucination on
    # instrumental audio. Keep only regions up to and including the largest.
    if len(regions) >= 2:
        largest_idx = max(range(len(regions)), key=lambda i: len(regions[i]))
        if largest_idx < len(regions) - 1:
            # Keep only regions up to and including the largest
            kept_end = regions[largest_idx][-1].end
            discarded_words = sum(len(r) for r in regions[largest_idx + 1:])
            regions = regions[:largest_idx + 1]
            logger.info(
                "Discarded %d hallucinated words in trailing regions (after t=%.1fs)",
                discarded_words, kept_end,
            )

    vocal_words: list[WordSegment] = []
    for r in regions:
        vocal_words.extend(r)

    last_vocal_end = vocal_words[-1].end if vocal_words else audio_duration

    logger.info(
        "Vocal regions: %d/%d words (%d filtered, %d regions), last_vocal_end=%.1fs (audio=%.1fs)",
        len(vocal_words), len(words), hallucinated, len(regions),
        last_vocal_end, audio_duration,
    )
    return vocal_words, last_vocal_end

def _merge_short_segments(
    segments: list[ClipSegment],
    preferred_min: float = mv_mvconst.PLANNER_PREFERRED_MIN_DURATION_S,
    max_segment_s: float = 10.0,
) -> list[ClipSegment]:
    """Merge adjacent segments below preferred_min to avoid cascade drift.

    Merge with whichever neighbor respects max_segment_s. Prefer forward,
    then backward. B-roll fillers adjacent to short segments absorbed first.
    """
    if not segments:
        return []

    def _do_merge(idx_a: int, idx_b: int) -> None:
        a, b = segments[idx_a], segments[idx_b]
        st = "broll absorbed" if a.shot_type == "broll" else \
             "singer absorbed broll" if b.shot_type == "broll" else \
             f"short ({a.duration:.1f}s) merged forward" if a.duration < b.duration \
             else f"short ({b.duration:.1f}s) merged backward"
        segments[idx_a] = ClipSegment(
            start=min(a.start, b.start), end=max(a.end, b.end),
            text=f"{a.text} | {b.text} [merged: {st}]",
            duration=max(a.end, b.end) - min(a.start, b.start),
            words=a.words + b.words,
            shot_type=a.shot_type if a.shot_type and a.shot_type != "broll" else b.shot_type,
        )
        del segments[idx_b]

    original_len = len(segments)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg.duration >= preferred_min:
                i += 1
                continue
            prev = segments[i - 1] if i > 0 else None
            nxt = segments[i + 1] if i + 1 < len(segments) else None

            # Priority 1: if this segment is broll, absorb into adjacent singer.
            if seg.shot_type == "broll":
                if nxt and nxt.shot_type != "broll":
                    _do_merge(i, i + 1); changed = True; continue
                if prev and prev.shot_type != "broll":
                    _do_merge(i - 1, i); changed = True; continue
                # Both neighbors are broll — merge forward.
                if nxt:
                    _do_merge(i, i + 1); changed = True; continue
            # Priority 2: adjacent broll absorbed into this singer segment.
            if nxt and nxt.shot_type == "broll":
                _do_merge(i, i + 1); changed = True; continue
            if prev and prev.shot_type == "broll":
                _do_merge(i - 1, i); changed = True; continue
            # Priority 3: merge with shorter neighbor (semantic continuity).
            if nxt and (seg.duration + nxt.duration) <= max_segment_s:
                _do_merge(i, i + 1); changed = True; continue
            if prev and (prev.duration + seg.duration) <= max_segment_s:
                _do_merge(i - 1, i); changed = True; continue
            i += 1

    logger.info("Merge complete: %d -> %d segments", original_len, len(segments))
    return segments

def _write_segment_plan(
    plans: list[SegmentPlan],
    output_dir: Path,
    input_audio: str,
    portrait: str,
    scene_prompt: str,
    refined_prompts: list[dict[str, str]] | None = None,
    width: int = 1920,
    height: int = 1088,
    two_stage: bool | None = None,
) -> Path:
    """Write segment_plan.json for approval gate."""
    plan_data = {
        "mode": "controlled",
        "input_audio": input_audio,
        "portrait": portrait,
        "scene_prompt": scene_prompt,
        "refined_prompts": refined_prompts,
        "width": width,
        "height": height,
        "two_stage": two_stage,
        "segments": [
            {
                "index": p.index,
                "start": round(p.start, 3),
                "end": round(p.end, 3),
                "text": p.text,
                "shot_type": p.shot_type,
                "prompt": p.prompt,
                "ref_image_path": p.ref_image_path,
                "status": p.status,
                "pipeline_engine": p.pipeline_engine,
            }
            for p in plans
        ],
    }

    plan_path = output_dir / "segment_plan.json"
    plan_path.write_text(json.dumps(plan_data, indent=2))
    logger.info("Segment plan written: %s (%d segments)", plan_path, len(plans))
    return plan_path

def _read_segment_plan(output_dir: Path) -> dict:
    """Read segment_plan.json from output_dir."""
    plan_path = output_dir / "segment_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Segment plan not found: {plan_path}. Run --mode controlled first.")
    return json.loads(plan_path.read_text())

def _validate_segment_plan(
    segments: list[ClipSegment],
    *,
    audio_duration: float | None = None,
    words: list[WordSegment] | None = None,
    preferred_min: float = mv_mvconst.PLANNER_PREFERRED_MIN_DURATION_S,
) -> None:
    """Validate segment plan invariants. Raises ``ValueError`` on violation.

    Checks: starts at 0, no overlaps, monotonic, valid durations, contiguous,
    ends at audio_duration, word coverage, timeline consistency.
    Warns (non-fatal) on sub-preferred-min segments.
    """
    if not segments:
        return

    tol = mv_mvconst.PLANNER_TOLERANCE_S
    if segments[0].start > tol:
        raise ValueError(f"Plan starts at {segments[0].start:.3f}s, expected 0 (tol {tol}s)")

    for i in range(len(segments) - 1):
        if segments[i].end > segments[i + 1].start:
            raise ValueError(
                f"Overlap seg[{i}].end={segments[i].end} > "
                f"seg[{i+1}].start={segments[i+1].start}"
            )

    for i, seg in enumerate(segments):
        if seg.start >= seg.end:
            raise ValueError(f"Non-monotonic seg {i}: start={seg.start} >= end={seg.end}")
        expected = seg.end - seg.start
        if abs(seg.duration - expected) > 1e-6:
            raise ValueError(f"Duration mismatch seg {i}: {seg.duration} != {expected}")

    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if gap > tol:
            raise ValueError(f"Gap seg {i}->{i+1}: {gap:.3f}s (tol {tol}s)")

    if audio_duration is not None:
        last_end = segments[-1].end
        if audio_duration - last_end > tol:
            raise ValueError(
                f"Plan ends at {last_end:.3f}s, audio={audio_duration:.3f}s "
                f"(uncovered={audio_duration - last_end:.3f}s)"
            )

    if words is not None:
        assigned = sorted((w.text, w.start, w.end) for seg in segments for w in seg.words)
        input_words = sorted((w.text, w.start, w.end) for w in words)
        if assigned != input_words:
            raise ValueError("Word coverage mismatch: duplicated or skipped words")

    total = sum(seg.duration for seg in segments)
    span = segments[-1].end - segments[0].start
    if abs(total - span) > 1e-6:
        raise ValueError(f"Timeline inconsistency: sum={total} != span={span}")

    for i, seg in enumerate(segments):
        if seg.duration < preferred_min and seg.words:
            logger.warning(
                "Segment %d duration %.1fs < preferred %.1fs (cascade drift risk %.1fs)",
                i, seg.duration, preferred_min, preferred_min - seg.duration,
            )
