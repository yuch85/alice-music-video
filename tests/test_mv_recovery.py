"""Unit tests for mv_recovery — plan/audio reconciliation (BUG 09.9-17).

GPU-free: Demucs, Whisper, and audio-duration probing are all mocked. Verifies
the mismatch detector and the plan-rebuild path without touching hardware.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import mv_recovery
from mv_recovery import _is_mismatch, _plan_span, reconcile_plan_with_audio
from mv_segment import ClipSegment, WordSegment


def _plan(segments: list[tuple[float, float]]) -> dict:
    """Build a minimal plan dict with the given (start, end) segment spans."""
    return {
        "mode": "controlled",
        "input_audio": "/tmp/orig.mp3",
        "portrait": "/tmp/portrait.jpg",
        "scene_prompt": "a singer",
        "width": 960,
        "height": 544,
        "two_stage": True,
        "segments": [
            {"index": i + 1, "start": s, "end": e, "text": "x",
             "shot_type": "singer", "prompt": "p", "ref_image_path": None,
             "status": "pending"}
            for i, (s, e) in enumerate(segments)
        ],
    }


class TestMismatchDetection(unittest.TestCase):
    def test_matching_plan_is_not_mismatch(self):
        # Plan spans 45s, audio is 45s -> match.
        self.assertFalse(_is_mismatch(45.0, 45.0))

    def test_small_drift_within_tolerance(self):
        # 1s drift on a 45s plan is under max(2.0, 5%) -> not a mismatch.
        self.assertFalse(_is_mismatch(44.0, 45.0))

    def test_full_song_plan_vs_short_audio_is_mismatch(self):
        # The BUG 09.9-17 case: 200s plan, 45s audio.
        self.assertTrue(_is_mismatch(45.0, 200.04))

    def test_zero_guards_return_false(self):
        self.assertFalse(_is_mismatch(0.0, 45.0))
        self.assertFalse(_is_mismatch(45.0, 0.0))

    def test_plan_span_reads_max_end(self):
        self.assertEqual(_plan_span(_plan([(0, 6), (6, 12), (12, 20)])), 20.0)


class TestReconcile(unittest.TestCase):
    def test_matching_plan_returned_unchanged(self):
        plan = _plan([(0, 6), (6, 12)])
        with patch.object(mv_recovery, "_get_audio_duration", return_value=12.0):
            out = reconcile_plan_with_audio(Path("/tmp"), Path("/tmp/a.wav"), plan)
        self.assertIs(out, plan)

    def test_abort_mode_raises(self):
        plan = _plan([(0, 100), (100, 200)])
        with patch.object(mv_recovery, "_get_audio_duration", return_value=45.0), \
             patch.dict("os.environ", {"MV_PLAN_MISMATCH": "abort"}):
            with self.assertRaises(RuntimeError):
                reconcile_plan_with_audio(Path("/tmp"), Path("/tmp/a.wav"), plan)

    def test_recover_rebuilds_plan_for_actual_audio(self):
        stale = _plan([(0, 100), (100, 200)])  # 200s plan
        words = [
            WordSegment(text="hello", start=0.5, end=1.0),
            WordSegment(text="world", start=1.0, end=2.0),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            # Two audio-duration reads: reconcile() gate + rebuild tiling. Both 6s.
            with patch.object(mv_recovery, "_get_audio_duration", return_value=6.0), \
                 patch.object(mv_recovery, "_run_demucs_separation",
                              return_value={"vocals": out_dir / "vocals.wav"}), \
                 patch.object(mv_recovery, "_transcribe_with_whisper",
                              return_value=words):
                out = reconcile_plan_with_audio(
                    out_dir, Path("/tmp/short.wav"), stale, max_clip_s=6,
                )
            # Plan now points at the real audio and spans <= 6s.
            self.assertEqual(out["input_audio"], "/tmp/short.wav")
            self.assertLessEqual(max(s["end"] for s in out["segments"]), 6.0)
            # Resolution + portrait preserved from the base plan.
            self.assertEqual(out["width"], 960)
            self.assertEqual(out["portrait"], "/tmp/portrait.jpg")
            # Ref falls back to portrait so resume needs no Qwen ref-gen.
            self.assertEqual(out["segments"][0]["ref_image_path"], "/tmp/portrait.jpg")
            # segment_plan.json was rewritten on disk.
            written = json.loads((out_dir / "segment_plan.json").read_text())
            self.assertEqual(written["input_audio"], "/tmp/short.wav")

    def test_recover_raises_when_demucs_yields_no_vocals(self):
        stale = _plan([(0, 100), (100, 200)])
        with patch.object(mv_recovery, "_get_audio_duration", return_value=6.0), \
             patch.object(mv_recovery, "_run_demucs_separation", return_value={}):
            with self.assertRaises(RuntimeError):
                reconcile_plan_with_audio(Path("/tmp"), Path("/tmp/s.wav"), stale)


if __name__ == "__main__":
    unittest.main()
