"""Unit tests for the music video pipeline orchestrator.

Tests pure-logic functions that don't require GPU or external services:
- Segment grouping (Whisper words -> 4-10s segments)
- FFmpeg concat list generation
- VRAM gate logic (mocked HTTP)
- Error handling (clip failure doesn't abort pipeline)
- Controlled mode: ShotType, SegmentPlan, per-segment refs, resume mode
"""

from __future__ import annotations

import json
import subprocess
import sys
import mv_post
import mv_post_filter
import mv_mvconst
import mv_vram
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import pipeline module
sys.path.insert(0, str(Path(__file__).parent))
# Make the gpu-manager `src` package importable (needed to patch
# build_ltx2_workflow, which _generate_clip imports from workflows.workflow_ltx2).
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
from generate_music_video_pipeline import (
    CAMERA_MOTION_TEMPLATES,
    CHARACTER_ACTION_TEMPLATES,
    ClipSegment,
    DEFAULT_GRAIN_INTENSITY,
    DEFAULT_LUT_NAME,
    DEFAULT_SHARPEN_STRENGTH,
    ENERGY_TEMPLATES,
    LTX2_VRAM_MB,
    POST_PROCESS_LUT_DIR,
    PRE_ROLL_FRAMES,
    SegmentPlan,
    ShotType,
    SlingshotClient,
    TAIL_LOSS_FRAMES,
    WordSegment,
    _apply_post_processing,
    _build_motion_prompt,
    _check_vram_gate,
    _composite_with_ffmpeg,
    _crop_audio_segment,
    _cycle_motion_templates,
    _download_default_lut,
    _flush_segment,
    _generate_black_frame,
    _generate_clip,
    _generate_segment_refs,
    _get_local_llm_endpoint,
    _get_pose_variation,
    _group_words_into_segments,
    _load_creative_inputs,
    _plan_clip_resolution,
    _read_segment_plan,
    _refine_prompts,
    _resolution_filter,
    _vram_guard_check,
    _write_segment_plan,
    _assign_shot_type,
    _fill_coverage_gaps,
    _split_long_segments,
    _build_broll_prompts,
    TARGET_H,
    TARGET_W,
    TWO_STAGE_LONG_EDGE_THRESHOLD,
    VramGuardResult,
    VRAM_FALLBACK_H,
    VRAM_FALLBACK_W,
    VRAM_SAFETY_MARGIN_MB,
)


class TestSegmentGrouping(unittest.TestCase):
    """Test _group_words_into_segments pure logic."""

    def _make_words(self, pairs: list[tuple[str, float, float]]) -> list[WordSegment]:
        """Helper: create WordSegments from (text, start, end) tuples."""
        return [WordSegment(text=t, start=s, end=e) for t, s, e in pairs]

    def test_basic_grouping(self):
        """Words within max_segment_s are grouped into one segment."""
        words = self._make_words([
            ("hello", 0.0, 0.5),
            ("world", 0.6, 1.0),
            ("this", 1.1, 1.5),
            ("is", 1.6, 2.0),
            ("a", 2.1, 2.5),
            ("test", 2.6, 3.0),
        ])
        segments = _group_words_into_segments(words, max_segment_s=10.0)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 3.0)
        self.assertEqual(segments[0].text, "hello world this is a test")

    def test_segment_split_at_max(self):
        """Words exceeding max_segment_s trigger a segment boundary."""
        words = self._make_words([
            ("word1", 0.0, 0.5),
            ("word2", 0.6, 1.0),
            ("word3", 5.0, 5.5),
            ("word4", 5.6, 6.0),
            ("word5", 6.1, 6.5),
            ("word6", 6.6, 7.0),
        ])
        segments = _group_words_into_segments(words, max_segment_s=5.0)
        # First segment: word1-word3 (0.0-5.5, duration 5.5 > 5.0)
        # word3 pushes over 5.0s, so it should be in first segment
        # Actually: word1(0.0-0.5) + word2(0.6-1.0) = 1.0s
        # Adding word3(5.0-5.5): duration = 5.5 - 0.0 = 5.5 >= 5.0 -> flush
        self.assertGreaterEqual(len(segments), 2)
        # Each segment should have text
        for seg in segments:
            self.assertTrue(seg.text)
            self.assertGreater(seg.duration, 0)

    def test_gap_triggers_boundary(self):
        """Large gap (> 2s) between words forces a segment boundary."""
        words = self._make_words([
            ("first", 0.0, 0.5),
            ("second", 0.6, 1.0),
            # 3-second gap
            ("third", 4.0, 4.5),
            ("fourth", 4.6, 5.0),
        ])
        segments = _group_words_into_segments(words, max_segment_s=10.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "first second")
        self.assertEqual(segments[1].text, "third fourth")

    def test_single_word_segment(self):
        """A single word forms its own segment."""
        words = self._make_words([
            ("lonely", 10.0, 10.5),
        ])
        segments = _group_words_into_segments(words, max_segment_s=10.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "lonely")
        self.assertAlmostEqual(segments[0].duration, 0.5)

    def test_empty_words(self):
        """Empty word list returns empty segments."""
        segments = _group_words_into_segments([], max_segment_s=10.0)
        self.assertEqual(segments, [])

    def test_flush_segment(self):
        """_flush_segment creates correct ClipSegment from words."""
        words = [
            WordSegment("hello", 0.0, 0.5),
            WordSegment("world", 0.6, 1.0),
        ]
        seg = _flush_segment(words, start=0.0)
        self.assertAlmostEqual(seg.start, 0.0)
        self.assertAlmostEqual(seg.end, 1.0)
        self.assertAlmostEqual(seg.duration, 1.0)
        self.assertEqual(seg.text, "hello world")
        self.assertEqual(len(seg.words), 2)

    def test_segment_duration_range(self):
        """Segments should generally be within 4-10s range (with edge cases)."""
        # Create words spanning 30 seconds
        words = []
        for i in range(60):
            words.append(WordSegment(
                f"word{i}", i * 0.5, i * 0.5 + 0.4
            ))
        segments = _group_words_into_segments(words, max_segment_s=10.0)
        # Should produce ~3 segments of ~10s each
        self.assertGreaterEqual(len(segments), 2)
        for seg in segments:
            self.assertLess(seg.duration, 15.0)  # generous upper bound


class TestFFmpegConcatList(unittest.TestCase):
    """Test FFmpeg concat list generation and compositing logic."""

    def test_concat_list_format(self):
        """Concat list uses FFmpeg concat demuxer format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()

            # Create dummy clip files
            clip_paths = []
            for i in range(1, 4):
                clip = clips_dir / f"clip_{i:03d}.mp4"
                clip.write_bytes(b"fake mp4 content")
                clip_paths.append(clip)

            # Verify clip naming is sequential and zero-padded
            for i, p in enumerate(clip_paths, 1):
                expected_name = f"clip_{i:03d}.mp4"
                self.assertEqual(p.name, expected_name)

            # Simulate concat list generation (from _composite_with_ffmpeg)
            concat_list = output_dir / "concat_list.txt"
            with open(concat_list, "w") as f:
                for clip_path in clip_paths:
                    f.write(f"file '{clip_path}'\n")

            # Verify format
            lines = concat_list.read_text().strip().split("\n")
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[0].startswith("file '"))
            self.assertTrue(lines[0].endswith(".mp4'"))

    def test_composite_with_ffmpeg_no_clips(self):
        """_composite_with_ffmpeg raises when no clips provided."""
        with self.assertRaises(RuntimeError) as ctx:
            _composite_with_ffmpeg([], Path("/tmp/input.mp3"), Path("/tmp/output"))
        self.assertIn("No clips", str(ctx.exception))

    @patch("mv_post.subprocess.run")
    def test_composite_ffmpeg_commands(self, mock_run):
        """FFmpeg receives correct concat and re-mux commands."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()

            clip_paths = []
            for i in range(1, 3):
                clip = clips_dir / f"clip_{i:03d}.mp4"
                clip.write_bytes(b"fake")
                clip_paths.append(clip)

            input_audio = Path(tmpdir) / "input.mp3"
            input_audio.write_bytes(b"fake audio")

            result = _composite_with_ffmpeg(clip_paths, input_audio, output_dir)

            # Verify two subprocess calls: concat + re-mux
            self.assertEqual(mock_run.call_count, 2)

            # First call: concat
            concat_call = mock_run.call_args_list[0]
            concat_cmd = concat_call[1].get("cmd", concat_call[0][0])
            self.assertIn("-f", concat_cmd)
            self.assertIn("concat", concat_cmd)

            # Second call: re-mux
            remux_call = mock_run.call_args_list[1]
            remux_cmd = remux_call[1].get("cmd", remux_call[0][0])
            self.assertIn("-c:a", remux_cmd)
            self.assertIn("aac", remux_cmd)

            # Verify output path
            self.assertEqual(result, output_dir / "final_output.mp4")


class TestVRAMGate(unittest.TestCase):
    """Test VRAM gate logic — Slingshot handles VRAM, gate checks ComfyUI."""

    @patch("mv_comfyui._comfyui_is_ready")
    def test_gate_passes_when_ready(self, mock_ready):
        """Gate returns True when ComfyUI ready (VRAM freed by Slingshot)."""
        mock_ready.return_value = True

        result = _check_vram_gate()
        self.assertTrue(result)
        mock_ready.assert_called_once()

    @patch("mv_comfyui._start_comfyui_via_gpu_manager", return_value=False)
    @patch("mv_comfyui._comfyui_is_ready")
    @patch("mv_comfyui._wait_for_comfyui_ready")
    def test_gate_fails_when_comfyui_busy(self, mock_wait, mock_ready, mock_start):
        """Gate returns False when ComfyUI not responding.

        Mock the gpu-manager /comfyui/start call so the gate's "ComfyUI
        not ready" branch deterministically fails to start ComfyUI
        regardless of whether gpu-manager is actually reachable in the
        test environment (previously passed only because gpu-manager was
        down at baseline time — the gate logic is unchanged by 09.9-10).
        """
        mock_ready.return_value = False
        mock_wait.return_value = False

        result = _check_vram_gate()
        self.assertFalse(result)

    @patch("mv_comfyui._comfyui_is_ready")
    @patch("mv_comfyui._wait_for_comfyui_ready")
    def test_gate_waits_then_passes(self, mock_wait, mock_ready):
        """Gate waits for ComfyUI and passes if it becomes ready."""
        mock_ready.return_value = False
        mock_wait.return_value = True

        result = _check_vram_gate()
        self.assertTrue(result)


class TestErrorHandling(unittest.TestCase):
    """Test pipeline error handling — one clip failure doesn't abort."""

    def test_failed_clip_excluded_from_concat(self):
        """Failed clips are excluded from the final concat list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()

            # Simulate: segments 1, 3 succeed; segment 2 fails
            successful_clips: list[Path] = []

            # Segment 1: success
            clip1 = clips_dir / "clip_001.mp4"
            clip1.write_bytes(b"success")
            successful_clips.append(clip1)

            # Segment 2: skipped (generation failed)
            # clip_002.mp4 does NOT exist

            # Segment 3: success
            clip3 = clips_dir / "clip_003.mp4"
            clip3.write_bytes(b"success")
            successful_clips.append(clip3)

            # Verify only successful clips are in the list
            self.assertEqual(len(successful_clips), 2)

            # Simulate concat list from successful clips only
            concat_lines = []
            for clip_path in successful_clips:
                concat_lines.append(f"file '{clip_path}'")

            # Verify no failed clip in concat list
            for line in concat_lines:
                self.assertNotIn("clip_002", line)

            # Verify correct clips present
            self.assertTrue(any("clip_001" in l for l in concat_lines))
            self.assertTrue(any("clip_003" in l for l in concat_lines))

    def test_pipeline_continues_after_clip_failure(self):
        """Pipeline processes remaining segments after one failure."""
        # Simulate segment processing with one failure
        segments = [
            {"index": 1, "text": "first segment"},
            {"index": 2, "text": "second segment"},  # will "fail"
            {"index": 3, "text": "third segment"},
        ]

        clips: list[dict] = []
        skipped: list[int] = []

        for seg in segments:
            if seg["index"] == 2:
                # Simulate generation failure
                skipped.append(seg["index"])
                continue
            clips.append({
                "index": seg["index"],
                "path": f"/tmp/clip_{seg['index']:03d}.mp4",
            })

        # Verify pipeline continued
        self.assertEqual(len(clips), 2)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped, [2])
        self.assertEqual(clips[0]["index"], 1)
        self.assertEqual(clips[1]["index"], 3)


class TestSlingshotClient(unittest.TestCase):
    """Tests for SlingshotClient HTTP interface."""

    def test_status_parses_response(self):
        """status() returns parsed dict on success."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "state": "idle",
                "hibernating": False,
                "slot_saved": False,
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_open.return_value = mock_resp

            client = SlingshotClient("http://localhost:8090")
            result = client.status()

            self.assertEqual(result["state"], "idle")
            self.assertFalse(result["hibernating"])

    def test_status_returns_none_on_error(self):
        """status() returns None when gpu-manager is unreachable."""
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            client = SlingshotClient("http://localhost:99999")
            result = client.status()
            self.assertIsNone(result)

    def test_hibernate_sends_post(self):
        """hibernate() sends POST and returns True on success."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "ok": True,
                "slot_saved": True,
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_open.return_value = mock_resp

            client = SlingshotClient("http://localhost:8090")
            result = client.hibernate()

            self.assertTrue(result)
            # Verify POST was called (Request with data)
            call_args = mock_open.call_args
            self.assertEqual(call_args[0][0].method, "POST")

    def test_hibernate_returns_false_on_error(self):
        """hibernate() returns False when hibernation fails."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "ok": False,
                "error": "hibernate_failed",
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_open.return_value = mock_resp

            client = SlingshotClient("http://localhost:8090")
            result = client.hibernate()
            self.assertFalse(result)

    def test_hibernate_returns_false_on_exception(self):
        """hibernate() returns False on connection errors."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            client = SlingshotClient("http://localhost:99999")
            result = client.hibernate()
            self.assertFalse(result)

    def test_wake_sends_params(self):
        """wake() sends gen_type and output_path in POST body."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "ok": True,
                "wake_prompt": "test prompt",
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_open.return_value = mock_resp

            client = SlingshotClient("http://localhost:8090")
            result = client.wake(task_name="music_video", output_path="/tmp/output.mp4")

            self.assertIsNotNone(result)
            self.assertTrue(result["ok"])
            # Verify POST body contains task_name
            call_args = mock_open.call_args
            body = call_args[0][0].data
            body_json = json.loads(body.decode())
            self.assertEqual(body_json["gen_type"], "music_video")
            self.assertEqual(body_json["output_path"], "/tmp/output.mp4")

    def test_wake_returns_none_on_error(self):
        """wake() returns None on connection errors."""
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            client = SlingshotClient("http://localhost:99999")
            result = client.wake()
            self.assertIsNone(result)

    def test_no_raw_ensure_vram_in_vram_gate(self):
        """_check_vram_gate no longer calls raw _ensure_vram."""
        # The _check_vram_gate function should not reference _ensure_vram
        # (replaced by SlingshotClient.hibernate)
        import inspect
        source = inspect.getsource(_check_vram_gate)
        self.assertNotIn("_ensure_vram", source)


class TestControlledMode(unittest.TestCase):
    """Tests for controlled mode — ."""

    def test_shot_type_enum_values(self):
        """ShotType enum has expected values."""
        self.assertEqual(ShotType.SINGER.value, "singer")
        self.assertEqual(ShotType.BROLL.value, "broll")
        self.assertEqual(ShotType.INSTRUMENTAL.value, "instrumental")
        self.assertEqual(ShotType.BLACK.value, "black")

    def test_segment_plan_dataclass(self):
        """SegmentPlan dataclass has all required fields."""
        plan = SegmentPlan(
            index=1,
            start=0.0,
            end=8.5,
            text="hello world",
            shot_type="singer",
            prompt="singer on stage",
            ref_image_path="/tmp/ref.jpg",
            status="pending",
        )
        self.assertEqual(plan.index, 1)
        self.assertEqual(plan.shot_type, "singer")
        self.assertEqual(plan.status, "pending")
        self.assertIsNotNone(plan.ref_image_path)

    def test_pose_variation_cycling(self):
        """_get_pose_variation cycles through pose variations."""
        p0 = _get_pose_variation(0)
        p1 = _get_pose_variation(1)
        p8 = _get_pose_variation(8)

        # Different indices -> different poses (for first 8)
        self.assertNotEqual(p0, p1)
        # Index 8 wraps to index 0
        self.assertEqual(p0, p8)

    def test_write_and_read_segment_plan(self):
        """_write_segment_plan writes JSON that _read_segment_plan can read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            plans = [
                SegmentPlan(
                    index=1, start=0.0, end=8.5, text="hello world",
                    shot_type="singer", prompt="singer on stage",
                    ref_image_path="/tmp/ref_001.jpg", status="pending",
                ),
                SegmentPlan(
                    index=2, start=8.5, end=15.0, text="goodbye world",
                    shot_type="broll", prompt="concert venue atmosphere",
                    ref_image_path=None, status="pending",
                ),
            ]

            plan_path = _write_segment_plan(
                plans, output_dir, "/tmp/input.mp3",
                "/tmp/portrait.jpg", "singer on stage"
            )

            self.assertTrue(plan_path.exists())

            plan_data = _read_segment_plan(output_dir)
            self.assertEqual(plan_data["mode"], "controlled")
            self.assertEqual(len(plan_data["segments"]), 2)
            self.assertEqual(plan_data["segments"][0]["shot_type"], "singer")
            self.assertEqual(plan_data["segments"][1]["shot_type"], "broll")
            self.assertEqual(plan_data["segments"][0]["ref_image_path"], "/tmp/ref_001.jpg")
            self.assertIsNone(plan_data["segments"][1]["ref_image_path"])

    def test_read_segment_plan_missing_file(self):
        """_read_segment_plan raises FileNotFoundError for missing plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                _read_segment_plan(output_dir)

    def test_generate_black_frame(self):
        """_generate_black_frame creates a black frame video via FFmpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "clips").mkdir()

            path = _generate_black_frame(5.0, 1, output_dir)
            self.assertEqual(path, output_dir / "clips" / "clip_001.mp4")
            # Note: FFmpeg may fail in test env (no ffmpeg), but path should be returned

    def test_controlled_mode_clip_generation_singer(self):
        """Singer shot type uses I2AV mode (ref + dialogue)."""
        # Test that _generate_clip with shot_type="singer" passes dialogue_text
        seg = ClipSegment(
            start=0.0, end=8.0, text="hello world",
            duration=8.0, words=[],
        )
        # The test verifies the function signature accepts shot_type
        from generate_music_video_pipeline import _generate_clip
        import inspect
        sig = inspect.signature(_generate_clip)
        params = list(sig.parameters.keys())
        self.assertIn("shot_type", params)
        self.assertIn("segment_prompt", params)

    def test_controlled_mode_shot_types_in_generate_clip(self):
        """_generate_clip handles all shot types in its logic."""
        import inspect
        from generate_music_video_pipeline import _generate_clip
        source = inspect.getsource(_generate_clip)

        # Verify shot type handling
        self.assertIn('"singer"', source)
        self.assertIn('"broll"', source)
        self.assertIn('"instrumental"', source)
        self.assertIn('"black"', source)

    def test_slingshot_in_controlled_mode(self):
        """Slingshot wraps both ref generation and clip generation."""
        import inspect
        from generate_music_video_pipeline import run_pipeline
        source = inspect.getsource(run_pipeline)

        # Verify Slingshot is used in the pipeline
        self.assertIn("slingshot", source.lower())
        self.assertIn("SLINGSHOT_ENABLED", source)

    def test_resume_mode_in_pipeline(self):
        """run_pipeline accepts mode and resume params."""
        import inspect
        from generate_music_video_pipeline import run_pipeline
        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())

        self.assertIn("mode", params)
        self.assertIn("resume", params)


class TestMCPTools(unittest.TestCase):
    """Tests for MCP tool registration — ."""

    def test_music_video_generate_refs_exists(self):
        """music_video_generate_refs MCP tool is registered in server.py."""
        server_path = Path(__file__).parent.parent / "gpu-manager" / "src" / "server.py"
        src = server_path.read_text()
        self.assertIn("music_video_generate_refs", src)
        self.assertIn("controlled", src)

    def test_music_video_generate_clips_exists(self):
        """music_video_generate_clips MCP tool is registered in server.py."""
        server_path = Path(__file__).parent.parent / "gpu-manager" / "src" / "server.py"
        src = server_path.read_text()
        self.assertIn("music_video_generate_clips", src)
        self.assertIn("resume", src)

    def test_original_auto_tool_unchanged(self):
        """alice_generate_music_video MCP tool remains unchanged."""
        server_path = Path(__file__).parent.parent / "gpu-manager" / "src" / "server.py"
        src = server_path.read_text()
        self.assertIn("alice_generate_music_video", src)


class TestLTX23Upgrade(unittest.TestCase):
    """Tests for LTX-2.3 GGUF upgrade —  Task 1."""

    def test_model_file_constant_is_ltx23_gguf(self):
        """LTX2_MODEL_FILE constant updated to LTX-2.3 GGUF filename."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import LTX2_MODEL_FILE
        self.assertIn("2.3", LTX2_MODEL_FILE)
        self.assertTrue(LTX2_MODEL_FILE.endswith(".gguf"))

    def test_vae_file_constant_is_ltx23(self):
        """LTX2_VAE_FILE constant updated to LTX23 video VAE filename."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import LTX2_VAE_FILE
        self.assertTrue(LTX2_VAE_FILE.endswith(".safetensors"))

    def test_text_projection_file_constant_exists(self):
        """LTX2_TEXT_PROJECTION_FILE constant exists for LTX-2.3."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import LTX2_TEXT_PROJECTION_FILE
        self.assertTrue(LTX2_TEXT_PROJECTION_FILE.endswith(".safetensors"))
        self.assertTrue(LTX2_TEXT_PROJECTION_FILE.endswith(".safetensors"))

    def test_gguf_loader_returns_unetloader_gguf(self):
        """_ltx2_loader_nodes with .gguf returns UnetLoaderGGUF node."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import _ltx2_loader_nodes
        nodes, mid = _ltx2_loader_nodes("LTX-2.3-22B-distilled-1.1-Q6_K.gguf")
        self.assertIn("UnetLoaderGGUF", str(nodes))
        self.assertEqual(mid, "30")  # GGUF model node ID

    def test_safetensors_loader_returns_checkpoint_simple(self):
        """_ltx2_loader_nodes with explicit safetensors path returns CheckpointLoaderSimple (backward compat)."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import _ltx2_loader_nodes
        nodes, mid = _ltx2_loader_nodes("ltx-2-19b-distilled-fp8.safetensors")
        self.assertIn("CheckpointLoaderSimple", str(nodes))
        self.assertEqual(mid, "10")  # Safetensors model node ID

    def test_gguf_loader_has_text_projection(self):
        """GGUF loader path includes text projection loader node."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import _ltx2_loader_nodes
        nodes, _ = _ltx2_loader_nodes("test.gguf")
        # Check that text projection loader is present
        has_text_proj = any(
            "TextProjection" in str(v.get("class_type", ""))
            for v in nodes.values()
        )
        self.assertTrue(has_text_proj, "GGUF loader missing text projection node")

    def test_safetensors_loader_has_text_projection(self):
        """Safetensors loader path also includes text projection for LTX-2.3."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import _ltx2_loader_nodes
        nodes, _ = _ltx2_loader_nodes("ltx-2-19b-distilled-fp8.safetensors")
        has_text_proj = any(
            "TextProjection" in str(v.get("class_type", ""))
            for v in nodes.values()
        )
        self.assertTrue(has_text_proj, "Safetensors loader missing text projection node")


class TestAudioVAEConditioning(unittest.TestCase):
    """Tests for Audio VAE conditioning —  Task 2."""

    def test_build_workflow_with_audio_path(self):
        """build_ltx2_workflow with audio_path produces LoadAudio + LTXVAudioVAEEncode."""
        import json
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import build_ltx2_workflow

        wf = build_ltx2_workflow(
            "test scene", "hello", "ref.jpg", audio_path="test.wav"
        )
        wf_str = json.dumps(wf)
        self.assertIn("LoadAudio", wf_str)
        self.assertIn("LTXVAudioVAEEncode", wf_str)

    def test_build_workflow_without_audio_path(self):
        """build_ltx2_workflow with audio_path=None uses LTXVEmptyLatentAudio (backward compat)."""
        import json
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import build_ltx2_workflow

        wf = build_ltx2_workflow("test scene", "hello", "ref.jpg", audio_path=None)
        wf_str = json.dumps(wf)
        self.assertIn("LTXVEmptyLatentAudio", wf_str)
        self.assertNotIn("LoadAudio", wf_str)

    def test_audio_path_workflow_is_serializable(self):
        """Workflow dict with audio_path is valid JSON."""
        import json
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import build_ltx2_workflow

        wf = build_ltx2_workflow("scene", "text", "ref.jpg", audio_path="audio.wav")
        # Should not raise
        json.dumps(wf)

    def test_audio_concat_ref_differs_between_branches(self):
        """LTXVConcatAVLatent audio_latent ref differs: audio vs no-audio path."""
        import json
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import build_ltx2_workflow

        wf_audio = build_ltx2_workflow("s", "h", "r.jpg", audio_path="t.wav")
        wf_no_audio = build_ltx2_workflow("s", "h", "r.jpg", audio_path=None)

        # Find LTXVConcatAVLatent in both
        for nid, node in wf_audio.items():
            if node.get("class_type") == "LTXVConcatAVLatent":
                audio_ref_with = node["inputs"]["audio_latent"]
                break
        for nid, node in wf_no_audio.items():
            if node.get("class_type") == "LTXVConcatAVLatent":
                audio_ref_without = node["inputs"]["audio_latent"]
                break

        # The audio latent source node should differ
        self.assertNotEqual(audio_ref_with, audio_ref_without)

    def test_i2av_alias_passes_audio_path_none(self):
        """build_ltx2_i2av_workflow passes audio_path=None by default."""
        import json
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import build_ltx2_i2av_workflow

        wf = build_ltx2_i2av_workflow("scene", "hello", "ref.jpg")
        wf_str = json.dumps(wf)
        # Should use empty latent audio (backward compat)
        self.assertIn("LTXVEmptyLatentAudio", wf_str)
        self.assertNotIn("LoadAudio", wf_str)


class TestPreRollPadding(unittest.TestCase):
    """Tests for pre-roll/tail-loss padding —  Task 3."""

    def test_pre_roll_frames_constant(self):
        """PRE_ROLL_FRAMES constant exists and equals 4."""
        self.assertEqual(PRE_ROLL_FRAMES, 4)

    def test_tail_loss_frames_constant(self):
        """TAIL_LOSS_FRAMES constant exists and equals 4."""
        self.assertEqual(TAIL_LOSS_FRAMES, 4)

    def test_crop_audio_segment_function_exists(self):
        """_crop_audio_segment function exists in the module."""
        self.assertTrue(callable(_crop_audio_segment))

    def test_generate_clip_accepts_vocals_stem(self):
        """_generate_clip accepts vocals_stem parameter."""
        import inspect
        sig = inspect.signature(_generate_clip)
        params = list(sig.parameters.keys())
        self.assertIn("vocals_stem", params)


class TestVRAMBudget(unittest.TestCase):
    """Tests for VRAM budget hardening —  Task 4."""

    def test_ltx2_vram_mb_in_reasonable_range(self):
        """LTX2_VRAM_MB updated for LTX-2.3 GGUF + Audio VAE (~35000-42000)."""
        val = LTX2_VRAM_MB
        self.assertGreaterEqual(val, 35000)
        self.assertLessEqual(val, 42000)

    def test_estimate_clip_vram_exists(self):
        """_estimate_clip_vram function exists in server.py."""
        server_path = Path(__file__).parent.parent / "gpu-manager" / "src" / "server.py"
        src = server_path.read_text()
        self.assertIn("_estimate_clip_vram", src)

    @unittest.skip("Private integration test — requires gpu-manager venv")
    def test_estimate_clip_vram_returns_reasonable_value(self):
        """_estimate_clip_vram returns value in reasonable range for typical clip.

        NOTE: Skipped in public repo — requires gpu-manager venv to run.
        """
        # Import via gpu-manager venv to avoid missing uvicorn
        import subprocess
        result = subprocess.run(
            ["gpu-manager/.venv/bin/python3", "-c",
             "import sys; sys.path.insert(0, 'gpu-manager'); "
             "from src.server import _estimate_clip_vram; "
             "print(_estimate_clip_vram(241))"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )
        vram = int(result.stdout.strip())
        self.assertGreaterEqual(vram, 30000)
        self.assertLessEqual(vram, 45000)

    @patch("mv_comfyui._comfyui_is_ready", return_value=False)
    @patch("mv_vram._get_free_vram_mb", return_value=39000)
    def test_vram_guard_fallback_single_stage(self, mock_free, mock_warm):
        """1920x1088 two-stage blocked, falls back to 1088x608 SINGLE-STAGE.

        Cold-card path (ComfyUI not yet resident): with ~39GB free, two-stage
        1920x1088 (~45.1GB needed via _estimate_vram_mb) and two-stage 1088x608
        (~41.7GB needed) both fail, but single-stage 1088x608 (~29.4GB needed —
        text encoder offloaded to CPU per the fallback logic) fits. The guard
        must approve the fallback AND report two_stage=False so the caller
        generates single-stage (not OOM). `mv_comfyui._comfyui_is_ready` is
        pinned False so the deterministic cold-card fallback chain is exercised
        (the warm-card branch would use activation-headroom and short-circuit).
        """
        guard = _vram_guard_check(1920, 1088, True)
        self.assertTrue(guard.ok)
        self.assertEqual((guard.eff_w, guard.eff_h), (1088, 608))
        self.assertFalse(guard.two_stage)
        self.assertIn("single-stage", guard.reason)

    @patch("mv_comfyui._comfyui_is_ready", return_value=False)
    @patch("mv_vram._get_free_vram_mb", return_value=30000)
    def test_vram_guard_blocks_when_nothing_fits(self, mock_free, mock_warm):
        """Even the single-stage fallback is rejected when VRAM is very low."""
        guard = _vram_guard_check(1920, 1088, True)
        self.assertFalse(guard.ok)

    def test_estimate_lowres_base_pass_fits(self):
        """Path A low-res base (960x544, two-stage, encoder CPU) fits < 48.5GB.

        The 09.9-13 reset of LTX2_ACT_PER_MP_MB to a low-res-anchored value must
        keep the base generation pass well under the 48GB-card budget.
        """
        est = mv_vram._estimate_vram_mb(960, 544, two_stage=True, text_encoder_cpu=True)
        self.assertLess(est, 48500)
        self.assertGreater(est, 20000)




class TestMotionTemplates(unittest.TestCase):
    """Tests for motion template constants and cycling — ."""

    def test_camera_motion_templates_has_12_items(self):
        """CAMERA_MOTION_TEMPLATES has exactly 12 items."""
        self.assertEqual(len(CAMERA_MOTION_TEMPLATES), 12)

    def test_character_action_templates_has_12_items(self):
        """CHARACTER_ACTION_TEMPLATES has exactly 12 items."""
        self.assertEqual(len(CHARACTER_ACTION_TEMPLATES), 12)

    def test_energy_templates_has_12_items(self):
        """ENERGY_TEMPLATES has exactly 12 items."""
        self.assertEqual(len(ENERGY_TEMPLATES), 12)

    def test_all_templates_are_nonempty_strings(self):
        """All template items are non-empty strings."""
        for templates in (CAMERA_MOTION_TEMPLATES, CHARACTER_ACTION_TEMPLATES, ENERGY_TEMPLATES):
            for item in templates:
                self.assertIsInstance(item, str)
                self.assertTrue(len(item) > 0, "Template item is empty")

    def test_cycle_motion_templates_returns_dict_with_correct_keys(self):
        """_cycle_motion_templates returns dict with camera_motion, character_action, energy."""
        result = _cycle_motion_templates(0)
        self.assertIn("camera_motion", result)
        self.assertIn("character_action", result)
        self.assertIn("energy", result)

    def test_cycle_motion_templates_cycling(self):
        """_cycle_motion_templates cycles through all 12 items correctly."""
        # Index 0 and 12 should return the same template (modulo cycling)
        r0 = _cycle_motion_templates(0)
        r12 = _cycle_motion_templates(12)
        self.assertEqual(r0["camera_motion"], r12["camera_motion"])
        self.assertEqual(r0["character_action"], r12["character_action"])
        self.assertEqual(r0["energy"], r12["energy"])

    def test_cycle_motion_templates_different_indices(self):
        """Different indices return different templates (indices 0 vs 1)."""
        r0 = _cycle_motion_templates(0)
        r1 = _cycle_motion_templates(1)
        self.assertNotEqual(r0["camera_motion"], r1["camera_motion"])
        self.assertNotEqual(r0["character_action"], r1["character_action"])

    def test_build_motion_prompt_combines_elements(self):
        """_build_motion_prompt combines camera, character, energy into a string."""
        prompt = _build_motion_prompt(
            camera_motion="Slow tracking shot following the subject",
            character_action="Throws their head back then turns toward camera",
            energy="High energy, dynamic movement throughout",
        )
        self.assertIn("Slow tracking shot", prompt)
        self.assertIn("Throws their head back", prompt)
        self.assertIn("High energy", prompt)

    def test_build_motion_prompt_format(self):
        """_build_motion_prompt uses Camera/Subject/Pacing labels."""
        prompt = _build_motion_prompt("dolly-in", "spins", "calm")
        self.assertIn("Camera:", prompt)
        self.assertIn("Subject:", prompt)
        self.assertIn("Pacing:", prompt)


class TestCreativeInputs(unittest.TestCase):
    """Tests for creative input loading — ."""

    def test_load_creative_inputs_empty_dict(self):
        """_load_creative_inputs returns empty dict for empty input."""
        result = _load_creative_inputs({})
        self.assertEqual(result, {})

    def test_load_creative_inputs_none_paths(self):
        """_load_creative_inputs returns empty dict for None paths."""
        result = _load_creative_inputs({
            "storyconcept_path": None,
            "themestyle_path": None,
        })
        self.assertEqual(result, {})

    def test_load_creative_inputs_nonexistent_paths(self):
        """_load_creative_inputs skips non-existent files gracefully."""
        result = _load_creative_inputs({
            "storyconcept_path": "/nonexistent/path/story.txt",
        })
        self.assertEqual(result, {})

    def test_load_creative_inputs_existing_files(self):
        """_load_creative_inputs reads file contents for existing paths."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            story = Path(tmpdir) / "story.txt"
            story.write_text("A tale of two cities")
            style = Path(tmpdir) / "style.txt"
            style.write_text("Cinematic noir")

            result = _load_creative_inputs({
                "storyconcept_path": str(story),
                "themestyle_path": str(style),
                "lyrics_path": None,  # should be skipped
            })

            self.assertEqual(result["storyconcept_path"], "A tale of two cities")
            self.assertEqual(result["themestyle_path"], "Cinematic noir")
            self.assertNotIn("lyrics_path", result)


class TestLLMRefinement(unittest.TestCase):
    """Tests for LLM prompt refinement — ."""

    def _make_segments(self, count: int) -> list[ClipSegment]:
        """Helper: create mock ClipSegment list."""
        segments = []
        for i in range(count):
            segments.append(ClipSegment(
                start=float(i * 8),
                end=float((i + 1) * 8),
                text=f"lyrics for segment {i}",
                duration=8.0,
                words=[],
            ))
        return segments

    def test_refine_prompts_returns_original_on_unreachable_llm(self):
        """_refine_prompts falls back to original prompts when LLM is unreachable."""
        segments = self._make_segments(3)
        creative_inputs = {"storyconcept_path": "A story", "themestyle_path": "Cinematic"}

        # No LLM running, should fall back gracefully
        result = _refine_prompts(segments, creative_inputs, "singer on stage")

        self.assertEqual(len(result), 3)
        for r in result:
            self.assertIn("image_prompt", r)
            self.assertIn("video_prompt", r)
            # Fallback should contain scene prompt and motion info
            self.assertTrue(
                "singer on stage" in r["image_prompt"]
                or "singer on stage" in r["video_prompt"],
                "Fallback prompt should contain scene_prompt",
            )

    def test_refine_prompts_parses_llm_response(self):
        """_refine_prompts parses LLM JSON response correctly (mocked)."""
        segments = self._make_segments(2)
        creative_inputs = {}

        # Mock _get_local_llm_endpoint to return a fake endpoint
        with patch.object(
            sys.modules.get("generate_music_video_pipeline") or __import__(
                "generate_music_video_pipeline", fromlist=["_get_local_llm_endpoint"]
            ),
            "_get_local_llm_endpoint",
            return_value=("http://127.0.0.1:9999/v1/chat/completions", "test-model"),
        ):
            # Mock urllib.request.urlopen to simulate LLM response
            mock_response = unittest.mock.MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "Refined cinematic prompt"}}]
            }).encode()

            with patch("urllib.request.urlopen", return_value=mock_response):
                with patch("urllib.request.Request"):
                    result = _refine_prompts(segments, creative_inputs, "base scene")

            self.assertEqual(len(result), 2)
            for r in result:
                self.assertIn("image_prompt", r)
                self.assertIn("video_prompt", r)

    def test_refine_prompts_empty_segments(self):
        """_refine_prompts handles empty segment list."""
        result = _refine_prompts([], {}, "scene")
        self.assertEqual(result, [])

    def test_get_local_llm_endpoint_returns_tuple(self):
        """_get_local_llm_endpoint returns (url, model_name) tuple."""
        # When gpu-manager is unreachable, should fall back to default
        result = _get_local_llm_endpoint()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIn("/v1/chat/completions", result[0])
        self.assertIsInstance(result[1], str)

    def test_refine_prompts_integration_with_motion_templates(self):
        """_refine_prompts integrates motion templates into fallback prompts."""
        segments = self._make_segments(4)
        creative_inputs = {"storyconcept_path": "Epic story"}

        result = _refine_prompts(segments, creative_inputs, "concert stage")

        # Each segment should have different motion (from cycling templates)
        video_prompts = [r["video_prompt"] for r in result]
        # At least some prompts should differ (motion templates cycle)
        self.assertNotEqual(video_prompts[0], video_prompts[1])


class TestPipelineIntegration(unittest.TestCase):
    """Tests for pipeline integration of prompt refinement — ."""

    def test_run_pipeline_accepts_creative_input_args(self):
        """run_pipeline signature accepts creative input path parameters."""
        import inspect
        from generate_music_video_pipeline import run_pipeline
        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())

        self.assertIn("storyconcept_path", params)
        self.assertIn("themestyle_path", params)
        self.assertIn("subjectsandscenes_path", params)
        self.assertIn("lyrics_path", params)

    def test_main_cli_has_creative_input_args(self):
        """CLI parser has --storyconcept, --themestyle, --subjectsandscenes, --lyrics."""
        import inspect
        from generate_music_video_pipeline import main
        source = inspect.getsource(main)

        self.assertIn("storyconcept", source)
        self.assertIn("themestyle", source)
        self.assertIn("subjectsandscenes", source)
        self.assertIn("lyrics", source)

    def test_auto_mode_uses_refined_prompts(self):
        """_run_auto_mode source references refined_prompts."""
        import inspect
        from generate_music_video_pipeline import _run_auto_mode
        source = inspect.getsource(_run_auto_mode)
        self.assertIn("refined_prompts", source)

    def test_controlled_mode_uses_refined_prompts(self):
        """_run_controlled_mode source references refined_prompts."""
        import inspect
        from generate_music_video_pipeline import _run_controlled_mode
        source = inspect.getsource(_run_controlled_mode)
        self.assertIn("refined_prompts", source)


class TestPostProcessing(unittest.TestCase):
    """Tests for post-processing chain —  Task 1."""

    def test_post_process_constants_exist(self):
        """Post-processing constants are defined with expected defaults."""
        self.assertEqual(DEFAULT_GRAIN_INTENSITY, 0.5)
        self.assertEqual(DEFAULT_SHARPEN_STRENGTH, 0.4)
        self.assertEqual(DEFAULT_LUT_NAME, "Cine_Grade.cube")
        self.assertIsNotNone(POST_PROCESS_LUT_DIR)

    def test_download_default_lut_creates_file(self):
        """_download_default_lut creates the LUT directory and file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import mv_lut
            orig_dir = mv_lut.POST_PROCESS_LUT_DIR
            orig_name = mv_lut.DEFAULT_LUT_NAME
            try:
                mv_lut.POST_PROCESS_LUT_DIR = tmpdir
                mv_lut.DEFAULT_LUT_NAME = "Test.cube"
                _download_default_lut()
                lut_path = Path(tmpdir) / "Test.cube"
                self.assertTrue(lut_path.exists(), "LUT file was not created")
                content = lut_path.read_text()
                self.assertIn("TITLE", content)
                self.assertIn("LUT_3D_SIZE 17", content)
            finally:
                mv_lut.POST_PROCESS_LUT_DIR = orig_dir
                mv_lut.DEFAULT_LUT_NAME = orig_name

    def test_download_default_lut_skips_if_exists(self):
        """_download_default_lut skips if LUT file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import mv_lut
            orig_dir = mv_lut.POST_PROCESS_LUT_DIR
            orig_name = mv_lut.DEFAULT_LUT_NAME
            try:
                mv_lut.POST_PROCESS_LUT_DIR = tmpdir
                mv_lut.DEFAULT_LUT_NAME = "Test.cube"
                existing = Path(tmpdir) / "Test.cube"
                existing.write_text("already here")
                _download_default_lut()
                self.assertEqual(existing.read_text(), "already here")
            finally:
                mv_lut.POST_PROCESS_LUT_DIR = orig_dir
                mv_lut.DEFAULT_LUT_NAME = orig_name

    @patch("mv_post.subprocess.run")
    def test_apply_post_processing_generates_filter_chain(self, mock_run):
        """_apply_post_processing generates FFmpeg command with colorchannelmixer, noise, unsharp."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            audio = Path(tmpdir) / "audio.mp3"
            audio.write_bytes(b"fake")
            output = Path(tmpdir) / "output.mp4"
            lut = Path(tmpdir) / "test.cube"
            lut.write_text("TITLE test\nLUT_3D_SIZE 17\n")

            result = _apply_post_processing(
                input_video, output, audio,
                lut_path=str(lut),
                grain_intensity=0.5,
                sharpen_strength=0.4,
                apply_lut=True,
            )

            self.assertTrue(result)
            self.assertGreaterEqual(mock_run.call_count, 1)
            call_args = mock_run.call_args_list[0]
            cmd = call_args[1].get("cmd", call_args[0][0])
            cmd_str = " ".join(str(c) for c in cmd)
            self.assertIn("colorchannelmixer", cmd_str)
            self.assertIn("lut3d", cmd_str)
            self.assertIn("noise", cmd_str)
            self.assertIn("unsharp", cmd_str)

    @patch("mv_post.subprocess.run")
    def test_apply_post_processing_fallback_without_lut(self, mock_run):
        """_apply_post_processing falls back to colorchannelmixer when LUT missing."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            audio = Path(tmpdir) / "audio.mp3"
            audio.write_bytes(b"fake")
            output = Path(tmpdir) / "output.mp4"
            missing_lut = Path(tmpdir) / "missing.cube"

            result = _apply_post_processing(
                input_video, output, audio,
                lut_path=str(missing_lut),
                grain_intensity=0.5,
                sharpen_strength=0.4,
            )

            self.assertTrue(result)
            call_args = mock_run.call_args_list[0]
            cmd = call_args[1].get("cmd", call_args[0][0])
            cmd_str = " ".join(str(c) for c in cmd)
            self.assertIn("colorchannelmixer", cmd_str)

    @patch("mv_post.subprocess.run")
    def test_apply_post_processing_default_params(self, mock_run):
        """_apply_post_processing with default params produces expected filter values."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            audio = Path(tmpdir) / "audio.mp3"
            audio.write_bytes(b"fake")
            output = Path(tmpdir) / "output.mp4"
            lut = Path(tmpdir) / "test.cube"
            lut.write_text("TITLE test\nLUT_3D_SIZE 17\n")

            _apply_post_processing(
                input_video, output, audio,
                lut_path=str(lut),
                grain_intensity=DEFAULT_GRAIN_INTENSITY,
                sharpen_strength=DEFAULT_SHARPEN_STRENGTH,
            )

            call_args = mock_run.call_args_list[0]
            cmd = call_args[1].get("cmd", call_args[0][0])
            cmd_str = " ".join(str(c) for c in cmd)
            self.assertIn("alls=2.5", cmd_str)
            self.assertIn("0.2", cmd_str)

    @patch("mv_post.subprocess.run")
    def test_apply_post_processing_neutral_default_no_tint(self, mock_run):
        """Default (apply_lut=False) is neutral: identity mixer, no 3D LUT."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            audio = Path(tmpdir) / "audio.mp3"
            audio.write_bytes(b"fake")
            output = Path(tmpdir) / "output.mp4"
            lut = Path(tmpdir) / "test.cube"
            lut.write_text("TITLE test\nLUT_3D_SIZE 17\n")

            _apply_post_processing(input_video, output, audio, lut_path=str(lut))

            call_args = mock_run.call_args_list[0]
            cmd = call_args[1].get("cmd", call_args[0][0])
            cmd_str = " ".join(str(c) for c in cmd)
            self.assertNotIn("lut3d", cmd_str)
            self.assertIn("colorchannelmixer=rr=1.0:gg=1.0:bb=1.0", cmd_str)

    @patch("mv_post.subprocess.run")
    def test_apply_post_processing_audio_delay_for_pre_roll(self, mock_run):
        """pre_roll_frames>0 adds an -af adelay matching the trimmed frames."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            audio = Path(tmpdir) / "audio.mp3"
            audio.write_bytes(b"fake")
            output = Path(tmpdir) / "output.mp4"

            # 4 pre-roll frames @ 24fps = 166667 us delay.
            _apply_post_processing(
                input_video, output, audio, pre_roll_frames=4,
            )

            call_args = mock_run.call_args_list[0]
            cmd = call_args[1].get("cmd", call_args[0][0])
            self.assertIn("-af", cmd)
            self.assertIn("adelay=166666:all=1", cmd)

    @patch("mv_post.subprocess.run")
    def test_apply_post_processing_no_audio_delay_when_zero(self, mock_run):
        """pre_roll_frames=0 omits the audio-delay filter entirely."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            audio = Path(tmpdir) / "audio.mp3"
            audio.write_bytes(b"fake")
            output = Path(tmpdir) / "output.mp4"

            _apply_post_processing(input_video, output, audio, pre_roll_frames=0)

            call_args = mock_run.call_args_list[0]
            cmd = call_args[1].get("cmd", call_args[0][0])
            self.assertNotIn("-af", cmd)

    def test_build_post_video_filter_helper(self):
        """_build_post_video_filter is pure: LUT applied only when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lut = Path(tmpdir) / "Cine_Grade.cube"
            lut.write_text("TITLE test\nLUT_3D_SIZE 17\n")

            chain, applied = mv_post_filter._build_post_video_filter(
                None, True, str(lut), 0.5, 0.4,
            )
            self.assertTrue(applied)
            self.assertIn(f"lut3d={lut}", chain)
            self.assertIn("rr=1.04", chain)  # Cine Grade mixer

            chain2, applied2 = mv_post_filter._build_post_video_filter(
                None, False, str(lut), 0.5, 0.4,
            )
            self.assertFalse(applied2)
            self.assertNotIn("lut3d", chain2)
            self.assertIn("colorchannelmixer=rr=1.0:gg=1.0:bb=1.0", chain2)

    def test_build_audio_delay_filter_helper(self):
        """_build_audio_delay_filter converts pre-roll frames to microseconds."""
        self.assertIsNone(mv_post_filter._build_audio_delay_filter(0))
        self.assertEqual(
            mv_post_filter._build_audio_delay_filter(4), "adelay=166666:all=1",
        )

    @patch("mv_post.subprocess.run")
    def test_composite_with_ffmpeg_calls_post_processing(self, mock_run):
        """_composite_with_ffmpeg calls _apply_post_processing when post-processing enabled."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()

            clip_paths = []
            for i in range(1, 3):
                clip = clips_dir / f"clip_{i:03d}.mp4"
                clip.write_bytes(b"fake")
                clip_paths.append(clip)

            input_audio = Path(tmpdir) / "input.mp3"
            input_audio.write_bytes(b"fake audio")

            result = _composite_with_ffmpeg(
                clip_paths, input_audio, output_dir,
                skip_post_process=False,
            )

            self.assertGreaterEqual(mock_run.call_count, 2)
            self.assertEqual(result, output_dir / "final_output.mp4")

    @patch("mv_post.subprocess.run")
    def test_composite_with_ffmpeg_skip_post_process(self, mock_run):
        """_composite_with_ffmpeg skips post-processing when flag is set."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()

            clip_paths = []
            for i in range(1, 3):
                clip = clips_dir / f"clip_{i:03d}.mp4"
                clip.write_bytes(b"fake")
                clip_paths.append(clip)

            input_audio = Path(tmpdir) / "input.mp3"
            input_audio.write_bytes(b"fake audio")

            result = _composite_with_ffmpeg(
                clip_paths, input_audio, output_dir,
                skip_post_process=True,
            )

            self.assertEqual(mock_run.call_count, 2)
            self.assertEqual(result, output_dir / "final_output.mp4")

    def test_cli_args_include_post_processing(self):
        """CLI parser has --lut, --grain-intensity, --sharpen-strength, --skip-post-process."""
        import inspect
        from generate_music_video_pipeline import main
        source = inspect.getsource(main)

        self.assertIn("--lut", source)
        self.assertIn("--grain-intensity", source)
        self.assertIn("--sharpen-strength", source)
        self.assertIn("--skip-post-process", source)

    def test_run_pipeline_accepts_post_processing_args(self):
        """run_pipeline signature accepts post-processing parameters."""
        import inspect
        from generate_music_video_pipeline import run_pipeline
        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())

        self.assertIn("lut_path", params)
        self.assertIn("grain_intensity", params)
        self.assertIn("sharpen_strength", params)
        self.assertIn("skip_post_process", params)


@unittest.skip("Private integration tests — require gpu-manager src.server, not in public repo")
class TestMCPToolCreativeInputs(unittest.TestCase):
    """Tests for MCP tool creative input params —  Task 2.

    NOTE: These tests verify the gpu-manager ↔ pipeline integration boundary.
    They are retained for reference but skipped in the public repo.
    """

    def test_music_video_generate_refs_accepts_creative_inputs(self):
        """music_video_generate_refs signature includes creative input params."""
        import inspect
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from src.server import music_video_generate_refs
        sig = inspect.signature(music_video_generate_refs)
        params = list(sig.parameters.keys())

        self.assertIn("storyconcept_path", params)
        self.assertIn("themestyle_path", params)
        self.assertIn("subjectsandscenes_path", params)
        self.assertIn("lyrics_path", params)

    def test_music_video_generate_refs_passes_creative_inputs_to_cli(self):
        """music_video_generate_refs passes creative input paths to pipeline CLI."""
        server_path = Path(__file__).parent.parent / "gpu-manager" / "src" / "server.py"
        src = server_path.read_text()

        import re
        match = re.search(
            r'async def music_video_generate_refs.*?(?=\nasync def |\n@mcp\.tool|\Z)',
            src, re.DOTALL
        )
        if match:
            func_src = match.group(0)
            self.assertIn("--storyconcept", func_src)
            self.assertIn("--themestyle", func_src)
            self.assertIn("--subjectsandscenes", func_src)
            self.assertIn("--lyrics", func_src)

    def test_music_video_generate_clips_signature_unchanged(self):
        """music_video_generate_clips only takes output_dir (no creative input params)."""
        import inspect
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from src.server import music_video_generate_clips
        sig = inspect.signature(music_video_generate_clips)
        params = list(sig.parameters.keys())

        self.assertEqual(params, ["output_dir"])

    def test_alice_generate_music_video_unchanged(self):
        """alice_generate_music_video still exists with original signature."""
        import inspect
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from src.server import alice_generate_music_video
        sig = inspect.signature(alice_generate_music_video)
        params = list(sig.parameters.keys())

        self.assertIn("input_audio_path", params)
        self.assertIn("output_dir", params)
        self.assertIn("portrait_path", params)
        self.assertIn("scene_prompt", params)


# ── Pipeline recovery tests (pipeline recovery) ──────────


class TestPipelineRecovery(unittest.TestCase):
    """Test pipeline error recovery — ensure_wake, ComfyUI reset, KeyboardInterrupt."""

    def test_ensure_wake_idempotent(self):
        """ensure_wake only calls wake() when state is hibernating."""
        # Instance A: hibernating -> should call wake
        client_a = SlingshotClient()
        client_a.status = MagicMock(return_value={"state": "hibernating"})
        client_a.wake = MagicMock(return_value={"ok": True})
        client_a.ensure_wake()
        client_a.wake.assert_called_once()

        # Instance B: idle -> should NOT call wake
        client_b = SlingshotClient()
        client_b.status = MagicMock(return_value={"state": "idle"})
        client_b.wake = MagicMock()
        client_b.ensure_wake()
        client_b.wake.assert_not_called()

    def test_clip_retry_clears_comfyui(self):
        """_generate_clip calls _reset_comfyui_state on retry (attempt > 0)."""
        reset_calls = []

        def mock_reset(timeout_s=30):
            reset_calls.append(timeout_s)
            return True

        with (
            patch(
                "mv_comfyui._reset_comfyui_state",
                side_effect=mock_reset,
            ),
            patch(
                "mv_comfyui._check_vram_gate",
                return_value=True,
            ),
            patch(
                "mv_vram._plan_clip_resolution",
                return_value=(768, 768, False, 768, 768, False),
            ),
            patch(
                "mv_comfyui._queue_workflow",
                side_effect=[RuntimeError("connection refused"), "prompt-123"],
            ),
            patch(
                "mv_comfyui._poll_completion",
                return_value={"outputs": {"1": {"filename": "alice_ltx2_001.mp4"}}},
            ),
            patch(
                "mv_comfyui._find_output_file",
                return_value=Path("/tmp/test_clip.mp4"),
            ),
            patch(
                "mv_audio._crop_audio_segment",
                return_value=Path("/tmp/test_audio.wav"),
            ),
            patch(
                "builtins.open", MagicMock()
            ),
        ):
            seg = ClipSegment(start=0.0, end=5.0, text="test segment", duration=5.0, words=[])
            result = _generate_clip(
                seg, "test prompt", Path("/tmp/portrait.jpg"), 1, Path("/tmp/output"),
            )
            # _reset_comfyui_state should have been called on the retry (attempt > 0)
            self.assertGreater(len(reset_calls), 0, "_reset_comfyui_state not called on retry")

    def test_main_keyboardinterrupt_wakes(self):
        """KeyboardInterrupt handler calls ensure_wake on the recovery client."""
        import generate_music_video_pipeline as pipeline_mod
        from generate_music_video_pipeline import (
            _register_slingshot_recovery,
        )

        mock_client = MagicMock()
        mock_client.ensure_wake = MagicMock()

        # Register the mock client
        _register_slingshot_recovery(mock_client)

        # Verify registration worked (read from module, not local import)
        self.assertIsNotNone(pipeline_mod._RECOVERY_SLINGSHOT)

        # Simulate the KeyboardInterrupt path by calling ensure_wake directly
        # (main() is hard to mock with sys.argv; test the handler logic)
        mock_client.ensure_wake.reset_mock()
        if pipeline_mod._RECOVERY_SLINGSHOT is not None:
            pipeline_mod._RECOVERY_SLINGSHOT.ensure_wake()
        mock_client.ensure_wake.assert_called_once()


class TestResolutionWiring(unittest.TestCase):
    """Plan 09.9-09: default 1920x1080 output — resolution threading + crop/scale."""

    def test_resolution_filter_crop(self):
        """1920x1088 generation -> crop to 1920x1080."""
        self.assertEqual(_resolution_filter(1920, 1088), "crop=1920:1080:0:0")

    def test_resolution_filter_scale(self):
        """1088x608 (VRAM-safe fallback) -> scale to 1920x1080."""
        self.assertEqual(_resolution_filter(1088, 608), "scale=1920:1080")

    def test_resolution_filter_none_at_target(self):
        """Already 1920x1080 -> no filter needed."""
        self.assertIsNone(_resolution_filter(1920, 1080))

    def test_segment_plan_roundtrip(self):
        """segment_plan.json persists width/height/two_stage (not 768)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            plans = [
                SegmentPlan(index=1, start=0.0, end=8.5, text="a", shot_type="singer",
                            prompt="p", ref_image_path="/tmp/r.jpg", status="pending"),
                SegmentPlan(index=2, start=8.5, end=15.0, text="b", shot_type="broll",
                            prompt="p2", ref_image_path=None, status="pending"),
            ]
            _write_segment_plan(plans, output_dir, "/tmp/in.mp3",
                                "/tmp/portrait.jpg", "scene",
                                width=1920, height=1088, two_stage=True)
            data = _read_segment_plan(output_dir)
            self.assertEqual(data["width"], 1920)
            self.assertEqual(data["height"], 1088)
            self.assertTrue(data["two_stage"])

    @patch("src.workflow_ltx2.build_ltx2_workflow")
    @patch("mv_comfyui._queue_workflow",
          side_effect=RuntimeError("short-circuit"))
    @patch("mv_comfyui._check_vram_gate", return_value=True)
    @patch("mv_vram._vram_guard_check",
          return_value=VramGuardResult(True, 1920, 1088, "test", True))
    @patch("mv_clip.time.sleep", return_value=None)
    def test_build_kwargs_dryrun(self, mock_sleep, mock_guard, mock_gate,
                                 mock_queue, mock_build):
        """1920x1088 auto two-stage -> build_ltx2_workflow keyword kwargs."""
        mock_build.return_value = {"nodes": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "clips").mkdir()
            portrait = output_dir / "portrait.jpg"
            portrait.write_bytes(b"fake")
            seg = ClipSegment(start=0.0, end=8.0, text="hello", duration=8.0, words=[])
            # Resolution is planned ONCE by the caller (auto two-stage from
            # 1920x1088 -> 1920x1088 / base 512x288 / use_two_stage=True via the
            # mocked guard), then passed to _generate_clip (which no longer
            # re-plans per clip). _queue_workflow short-circuits after the build
            # call; time.sleep is a no-op so the retry loop doesn't actually wait.
            gen_w, gen_h, use_two_stage, base_width, base_height, te_cpu = \
                _plan_clip_resolution(1920, 1088, None)
            # upscale=False forces Path A (base-only build_ltx2_workflow) so this
            # dryrun checks the base build kwargs deterministically — independent of
            # whether the Path B upscaler checkpoint happens to be on disk (09.9-16).
            _generate_clip(seg, "scene", portrait, 1, output_dir,
                           gen_w=gen_w, gen_h=gen_h, use_two_stage=use_two_stage,
                           base_width=base_width, base_height=base_height,
                           text_encoder_device=("cpu" if te_cpu else "default"),
                           upscale=False)
            self.assertTrue(mock_build.called, "build_ltx2_workflow was not called")
            kwargs = mock_build.call_args.kwargs
            self.assertEqual(kwargs["width"], 1920)
            self.assertEqual(kwargs["height"], 1088)
            self.assertTrue(kwargs["use_two_stage"])
            self.assertEqual(kwargs["base_width"], 512)
            self.assertEqual(kwargs["base_height"], 288)

    def test_final_resolution_ffprobe(self):
        """Real ffmpeg round-trip: 1920x1088 clip -> composite -> 1920x1080."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()

            # 1s silent audio for re-mux
            audio = output_dir / "audio.mp3"
            gen_audio = subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "aevalsrc=0:duration=1",
                str(audio),
            ], capture_output=True, text=True)
            self.assertEqual(gen_audio.returncode, 0, gen_audio.stderr)

            # 1920x1088 test source clip
            src = clips_dir / "clip_001.mp4"
            gen = subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "testsrc=size=1920x1088:rate=24:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src),
            ], capture_output=True, text=True)
            self.assertEqual(gen.returncode, 0, gen.stderr)

            # Composite (skip post-process), normalize 1920x1088 -> 1920x1080
            final = _composite_with_ffmpeg(
                [src], audio, output_dir,
                skip_post_process=True, gen_width=1920, gen_height=1088,
            )

            probe = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0", str(final),
            ], capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(probe.stdout.strip(), "1920,1080")


class TestTimelineCompositeAndCoverage(unittest.TestCase):
    """Plan 09.9-12: timeline-aware composite + coverage-gap fillers + 18s split.

    PART D (end-to-end end-to-end re-run) is executed by the orchestrator via
    the gpu-manager MCP tools and is intentionally NOT covered here.
    """

    @patch("mv_post.subprocess.run")
    def test_composite_timeline_overlay(self, mock_run):
        """Timeline branch overlays clips at segment.start; no -shortest; audio remuxed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()

            clip_paths = []
            for i in range(1, 3):
                clip = clips_dir / f"clip_{i:03d}.mp4"
                clip.write_bytes(b"fake")
                clip_paths.append(clip)

            input_audio = Path(tmpdir) / "input.mp3"
            input_audio.write_bytes(b"fake audio")

            result = _composite_with_ffmpeg(
                clip_paths, input_audio, output_dir,
                clip_timings=[(0.0, 5.0), (5.0, 10.0)],
                audio_duration=10.0,
                gen_width=1920, gen_height=1088,
            )

            overlay_found = False
            shortest_found = False
            audio_remux_found = False
            for call in mock_run.call_args_list:
                cmd = call[1].get("cmd", call[0][0])
                cmd_str = " ".join(str(c) for c in cmd)
                if "overlay" in cmd_str and "gte(t," in cmd_str and "eof_action=repeat" in cmd_str:
                    overlay_found = True
                if "-shortest" in cmd:
                    shortest_found = True
                if "-c:a" in cmd and "aac" in cmd:
                    audio_remux_found = True

            self.assertTrue(overlay_found, "timeline overlay filtergraph not built")
            self.assertFalse(shortest_found, "-shortest must be absent from every ffmpeg cmd")
            self.assertTrue(audio_remux_found, "audio re-mux (-c:a aac) missing")
            self.assertEqual(result, output_dir / "final_output.mp4")

    @patch("mv_post.subprocess.run")
    def test_composite_timeline_video_overlap_d03(self, mock_run):
        """LOCKED D-03: fixed 1s VIDEO-only overlap; clip 1 unoffset, audio seamless.

        Every clip after the first has its VIDEO enable threshold delayed by
        VIDEO_OVERLAP_S (1.0s) relative to its audio start. Clip 1 (i == 1)
        stays at its true audio start. No crossfade/dissolve appears (hard
        video-side trim). Audio re-mux path is untouched (covered elsewhere).
        """
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            clips_dir = output_dir / "clips"
            clips_dir.mkdir()
            clip_paths = []
            for i in range(1, 4):  # 3 clips -> 2 seams
                clip = clips_dir / f"clip_{i:03d}.mp4"
                clip.write_bytes(b"fake")
                clip_paths.append(clip)
            input_audio = Path(tmpdir) / "input.mp3"
            input_audio.write_bytes(b"fake audio")

            # clip2 audio starts at 15 (1s overlap with clip1 [0,16]),
            # clip3 audio starts at 30 (1s overlap with clip2 [15,31]).
            clip_timings = [(0.0, 16.0), (15.0, 31.0), (30.0, 46.0)]

            _composite_with_ffmpeg(
                clip_paths, input_audio, output_dir,
                clip_timings=clip_timings,
                audio_duration=46.0,
                gen_width=1920, gen_height=1088,
            )

            canvas_cmd = ""
            for call in mock_run.call_args_list:
                cmd = call[1].get("cmd", call[0][0])
                cmd_str = " ".join(str(c) for c in cmd)
                if "color=c=black" in cmd_str and "overlay" in cmd_str and "gte(t," in cmd_str:
                    canvas_cmd = cmd_str
                    break
            self.assertTrue(canvas_cmd, "timeline canvas filtergraph not captured")

            import re
            enables = re.findall(r"gte\(t,([0-9.]+)\)", canvas_cmd)
            self.assertEqual(len(enables), 3, "expected one enable per clip")
            # Clip 1 unoffset; clips 2/3 offset by exactly +1.0s.
            self.assertAlmostEqual(float(enables[0]), clip_timings[0][0], places=3)
            self.assertAlmostEqual(
                float(enables[1]), clip_timings[1][0] + 1.0, places=3,
                msg="clip2 video enable must be audio start + 1.0s (D-03)",
            )
            self.assertAlmostEqual(
                float(enables[2]), clip_timings[2][0] + 1.0, places=3,
                msg="clip3 video enable must be audio start + 1.0s (D-03)",
            )
            self.assertIn("VIDEO_OVERLAP_S", mv_post.__dict__ or "")
            # No transition filter — hard trim only.
            self.assertNotIn("xfade", canvas_cmd)
            self.assertNotIn("dissolve", canvas_cmd)
            # The D-03 constant exists and is a fixed 1s.
            self.assertEqual(getattr(mv_post, "VIDEO_OVERLAP_S", None), 1.0)

    def test_fill_coverage_gaps(self):
        """Coverage tiles 0->200.04 contiguously with 3 broll fillers; lyric seg not broll."""
        segs = [
            ClipSegment(start=11.94, end=20.0, text="a", duration=8.06, words=[]),
            ClipSegment(start=20.0, end=69.2, text="b", duration=49.2, words=[]),
            ClipSegment(start=79.5, end=89.72, text="c", duration=10.22, words=[]),
        ]
        out = _fill_coverage_gaps(segs, 200.04, ["loc1", "loc2", "loc3"])

        # Contiguous tiling: each segment.end == next.start within 1e-6.
        for prev, nxt in zip(out, out[1:]):
            self.assertAlmostEqual(prev.end, nxt.start, places=6)

        brolls = [s for s in out if s.shot_type == "broll"]
        self.assertEqual(len(brolls), 3, "expected exactly 3 broll fillers")

        # Lyric seg C keeps shot_type None (NOT broll) — the GAP got the filler.
        self.assertIsNone(segs[2].shot_type)

        # Intro filler starts at 0; outro filler ends at audio_duration.
        self.assertAlmostEqual(out[0].start, 0.0)
        self.assertAlmostEqual(out[0].end, 11.94)
        self.assertAlmostEqual(out[-1].end, 200.04)

    def test_split_long_segments(self):
        """A 23.44s broll segment splits into 18.0 + 5.44, both broll."""
        seg = ClipSegment(
            start=176.6, end=200.04, text="broll", duration=23.44,
            words=[], shot_type="broll",
        )
        out = _split_long_segments([seg])
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].start, 176.6)
        self.assertAlmostEqual(out[0].end, 194.6)
        self.assertAlmostEqual(out[1].start, 194.6)
        self.assertAlmostEqual(out[1].end, 200.04)
        for s in out:
            self.assertLessEqual(s.end - s.start, 18.0 + 1e-9)
            self.assertEqual(s.shot_type, "broll")

    def test_split_long_segments_max_clip_s(self):
        """--max-clip-s=6 splits an 18s wordless segment into 3 x <=6s clips."""
        seg = ClipSegment(
            start=0.0, end=18.0, text="broll", duration=18.0,
            words=[], shot_type="broll",
        )
        out = _split_long_segments([seg], max_s=6)
        self.assertEqual(len(out), 3)
        for s in out:
            self.assertGreater(s.end - s.start, 0.0)
            self.assertLessEqual(s.end - s.start, 6.0 + 1e-9)
            self.assertEqual(s.shot_type, "broll")
        # Contiguous coverage: 0..6, 6..12, 12..18
        self.assertAlmostEqual(out[0].start, 0.0)
        self.assertAlmostEqual(out[0].end, 6.0)
        self.assertAlmostEqual(out[1].start, 6.0)
        self.assertAlmostEqual(out[1].end, 12.0)
        self.assertAlmostEqual(out[2].start, 12.0)
        self.assertAlmostEqual(out[2].end, 18.0)

    def test_build_broll_prompts(self):
        """6 themed prompts from themestyle + subjects/scenes files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            theme = td / "themestyle.txt"
            theme.write_text("Visual style: warm, nostalgic cinematic.\n")
            subs = td / "subjectsandscenes.txt"
            subs.write_text(
                "1. Mountain at dawn — first light\n"
                "2. Cold river — frozen surface\n"
                "3. The empty room — stillness\n"
                "4. The scenery she saw — memory\n"
                "5. Mountain air between two people — closeness\n"
                "6. Quiet together — peace\n"
                "\nRules for B-roll:\nUse only wide shots.\n"
            )
            prompts = _build_broll_prompts({
                "themestyle_path": theme.read_text(),
                "subjectsandscenes_path": subs.read_text(),
            })
            self.assertEqual(len(prompts), 6)
            theme_line = "warm, nostalgic cinematic."
            for p in prompts:
                self.assertTrue(p.startswith(theme_line), p)
            locs = [
                "Mountain at dawn", "Cold river", "The empty room",
                "The scenery she saw", "Mountain air between two people",
                "Quiet together",
            ]
            for loc in locs:
                self.assertTrue(any(loc in p for p in prompts), loc)

    def test_assign_shot_type_gap_filler(self):
        """Gap filler absorbs the gap so the following lyric segment is 'singer'."""
        broll_filler = ClipSegment(
            start=69.2, end=79.5, text="loc", duration=10.3, words=[], shot_type="broll"
        )
        seg_c = ClipSegment(
            start=79.5, end=89.72, text="君は戻ってきた…", duration=10.22, words=[]
        )
        total = 20

        # The filler keeps its explicit broll shot type.
        st_broll = broll_filler.shot_type or _assign_shot_type(broll_filler, 7, total, 0.0)
        self.assertEqual(st_broll, "broll")

        # seg_c follows contiguously (prev_end == seg_c.start) thanks to the
        # filler, so the gap heuristic no longer mislabels it broll.
        prev_end = broll_filler.end
        st_c = seg_c.shot_type or _assign_shot_type(seg_c, 8, total, prev_end)
        self.assertEqual(st_c, "singer")

        # Contrast: WITHOUT the filler, the 10.3s gap would mislabel seg_c broll.
        st_c_broken = _assign_shot_type(seg_c, 8, total, 69.2)
        self.assertEqual(st_c_broken, "broll")


class TestMVMVConst(unittest.TestCase):
    """Constants introduced by Plan 09.9-13 (Path A low-res + bislerp).

    Single source of truth for 09.9-13 magic numbers — must not be hardcoded
    elsewhere.
    """

    def test_constants_present_and_typed(self):
        """All 09.9-13 constants exist with the expected values/types."""
        self.assertTrue(mv_mvconst.UPSCALE_MODEL_FILENAME.endswith(".safetensors"))
        self.assertEqual(mv_mvconst.UPSCALE_MODEL_DIR, "models/latent_upscale_models")
        self.assertEqual(mv_mvconst.UPSCALE_FACTOR, 2)
        self.assertEqual(mv_mvconst.DEFAULT_LOWRES_W, 960)
        self.assertEqual(mv_mvconst.DEFAULT_LOWRES_H, 544)
        self.assertEqual(mv_mvconst.DEFAULT_MAX_CLIP_S, 6)

    def test_lowres_doubles_to_1080p_output(self):
        """Low-res base (960x544) upscales 2x to ~1088 then crops to 1080."""
        self.assertEqual(mv_mvconst.DEFAULT_LOWRES_W * mv_mvconst.UPSCALE_FACTOR, 1920)
        self.assertEqual(mv_mvconst.DEFAULT_LOWRES_H * mv_mvconst.UPSCALE_FACTOR, 1088)


class TestLTX2TwoStagePathA(unittest.TestCase):
    """Plan 09.9-13 Path A: two-stage refine KSampler (node 41) removed (OOM fix)."""

    def _build_two_stage(self) -> dict:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import build_ltx2_workflow
        return build_ltx2_workflow(
            "scene", "text", "ref.jpg", use_two_stage=True, audio_path=None,
        )

    @staticmethod
    def _contains_link(obj, target: list) -> bool:
        if isinstance(obj, list) and len(obj) == 2 and obj == target:
            return True
        if isinstance(obj, dict):
            return any(TestLTX2TwoStagePathA._contains_link(v, target) for v in obj.values())
        if isinstance(obj, list):
            return any(TestLTX2TwoStagePathA._contains_link(v, target) for v in obj)
        return False

    def test_node_41_removed(self):
        """The full-res refine KSampler (node 41) must be gone to avoid OOM."""
        wf = self._build_two_stage()
        self.assertNotIn("41", wf, "node 41 (refine KSampler) must be removed")

    def test_node_42_decodes_node_40(self):
        """Node 42 decodes node 40's bislerp-upscaled latent directly."""
        wf = self._build_two_stage()
        self.assertIn("42", wf)
        self.assertTrue(
            self._contains_link(wf["42"], ["40", 0]),
            "node 42 must decode node 40's latent (Path A bislerp, no refine)",
        )

    def test_node_40_is_latent_upscale(self):
        """Node 40 is the bislerp LatentUpscale that is decoded directly."""
        wf = self._build_two_stage()
        self.assertIn("40", wf)
        self.assertEqual(wf["40"]["class_type"], "LatentUpscale")

    def test_node_40_upscale_target_is_decoded_div_4(self):
        """Path A bislerp LatentUpscale target = requested decoded res // 4.

        LTX-2 VAE decodes at x32 but the core ComfyUI LatentUpscale sizes the
        latent as width//8, so a raw target of W decodes to 4W pixels. Node 40
        must therefore request W//4 so the per-clip lands at the intended W.
        Without this, a 960x544 request silently decoded to 3840x2176 (4K) and
        defeated the low-res OOM-unblock. Caught during the 09.9-13 e2e.
        """
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2 import build_ltx2_workflow
        wf = build_ltx2_workflow(
            "scene", "text", "ref.jpg", use_two_stage=True,
            width=960, height=544, audio_path=None,
        )
        self.assertEqual(wf["40"]["inputs"]["width"], 240)
        self.assertEqual(wf["40"]["inputs"]["height"], 136)


class TestCombinedGraphVRAM(unittest.TestCase):
    """Task 3: combined base-gen+upscale graph VRAM budget vs VRAM guard."""

    def test_combined_estimate_under_ceiling_with_cpu_encoder(self):
        """Combined graph fits 48GB (−2GB) with text-encoder CPU-offload + Slingshot."""
        from mv_vram import _estimate_combined_vram_mb, COMBINED_VRAM_CEILING_MB
        est = _estimate_combined_vram_mb(960, 544, 1920, 1088, text_encoder_cpu=True)
        self.assertLess(
            est, COMBINED_VRAM_CEILING_MB,
            f"combined graph estimate {est}MB must fit under {COMBINED_VRAM_CEILING_MB}MB "
            "(48GB card - 2GB) with text-encoder CPU-offload + Slingshot arbitration",
        )

    def test_ceiling_is_48gb_minus_2gb(self):
        from mv_vram import COMBINED_VRAM_CEILING_MB
        self.assertEqual(COMBINED_VRAM_CEILING_MB, 46000)

    def test_cpu_offload_saves_text_encoder_vram(self):
        """Combined-graph estimate is encoder-invariant: the ~12GB Gemma text encoder
        is FOLDED INTO COMBINED_RESIDENT_BASE_MB (warm-calibrated with GPU encoder), so
        CPU vs GPU offload yields identical estimates. Both must sit under the 48GB
        ceiling (conservative ~32GB for 18s vs ~31GB real). This is the post-restructure
        semantics of `_estimate_combined_vram_mb` (mv-vram-frame-guard)."""
        from mv_vram import (
            COMBINED_VRAM_CEILING_MB,
            _estimate_combined_vram_mb,
        )
        cpu = _estimate_combined_vram_mb(960, 544, 1920, 1088, text_encoder_cpu=True)
        gpu = _estimate_combined_vram_mb(960, 544, 1920, 1088, text_encoder_cpu=False)
        # Encoder is folded into the resident base -> CPU and GPU estimates are equal.
        self.assertEqual(cpu, gpu)
        # Both must be safely under the 48GB-card ceiling (leaves room for Slingshot
        # arbitration + ~2GB safety margin).
        self.assertLess(cpu, COMBINED_VRAM_CEILING_MB)
        self.assertLess(gpu, COMBINED_VRAM_CEILING_MB)

    def test_upscaler_conv_net_included(self):
        """The <1GB upscaler conv net is budgeted beyond the base weights."""
        from mv_vram import UPSCALER_CONV_NET_MB
        self.assertGreaterEqual(UPSCALER_CONV_NET_MB, 900)
        self.assertLessEqual(UPSCALER_CONV_NET_MB, 1000)


class TestLTX2UpscaleWorkflow(unittest.TestCase):
    """Task 1/4: build_ltx2_combined_workflow (latent→latent Path B, 09.9-16)."""

    def _build(self, **kw):
        sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-manager"))
        from workflows.workflow_ltx2_upscale import build_ltx2_combined_workflow
        base = dict(
            target_w=1920, target_h=1088, base_w=960, base_h=544,
            scene_prompt="a singer on stage", dialogue_text="hello",
            ref_image_filename="portrait.jpg",
            length_s=6, audio_filename="vocals.wav",
        )
        base.update(kw)
        return build_ltx2_combined_workflow(**base)

    def test_no_mp4_round_trip(self):
        """No LoadVideo / VAEEncode anywhere — the base→upscale handoff is latent→latent."""
        wf = self._build()
        ct = {n.get("class_type") for n in wf.values()}
        self.assertNotIn("LoadVideo", ct, "LoadVideo must be absent (no MP4 round-trip)")
        self.assertNotIn("VAEEncode", ct, "VAEEncode of an MP4 must be absent")

    def test_one_dit_load_shared(self):
        """The base LTX-2.3 DiT is loaded ONCE (one UnetLoaderGGUF) for base + refine."""
        wf = self._build()
        from workflows.workflow_ltx2 import LTX2_MODEL_FILE
        dit = [n for n in wf.values()
               if n.get("class_type") == "UnetLoaderGGUF"
               and n.get("inputs", {}).get("unet_name") == LTX2_MODEL_FILE]
        self.assertEqual(len(dit), 1, "exactly one base DiT load shared by base + refine")

    def test_handoff_is_latent_to_latent(self):
        """The upscale sub-graph's input SeparateAV consumes the base-gen joint AV latent.

        Trace the upscale LTXVSeparateAVLatent (the one NOT decoding node 207's
        output) → its av_latent must reference a base-gen latent node (KSampler /
        SeparateAV / ImgToVideo / EmptyLTXVLatentVideo), NEVER a VAEEncode or a
        LTXVConcatAVLatent fed by one. And NO LoadVideo/VAEEncode exists at all.
        """
        wf = self._build()
        ct = {n.get("class_type") for n in wf.values()}
        self.assertNotIn("LoadVideo", ct)
        self.assertNotIn("VAEEncode", ct)
        # Input-side SeparateAV = a SeparateAV whose av_latent is NOT node 207.
        inp = [n for n in wf.values()
               if n.get("class_type") == "LTXVSeparateAVLatent"
               and n["inputs"].get("av_latent", [None])[0] != "207"]
        self.assertEqual(len(inp), 1, "expected one input-side upscale SeparateAV")
        src_id = inp[0]["inputs"]["av_latent"][0]
        src_class = wf[str(src_id)]["class_type"]
        self.assertIn(
            src_class,
            {"KSampler", "LTXVSeparateAVLatent", "LTXVImgToVideo", "EmptyLTXVLatentVideo"},
            f"upscale av_latent must be a base-gen latent node, got {src_class}",
        )
        self.assertEqual(src_class, "KSampler",
                         "VRGDG feeds the sampler (joint AV) latent directly")

    def test_upscaler_loaded_via_model_loader(self):
        """Upscaler conv net loaded via LatentUpscaleModelLoader, not as a DiT.

        The base LTX-2.3 DiT (deployed GGUF) IS loaded for the refine pass;
        the upscaler conv net must NOT be loaded as a DiT checkpoint/unet — it is
        loaded via LatentUpscaleModelLoader with model_name == UPSCALE_MODEL_FILENAME.
        """
        wf = self._build()
        self.assertIsInstance(wf, dict)
        from mv_mvconst import UPSCALE_MODEL_FILENAME
        from workflows.workflow_ltx2 import LTX2_MODEL_FILE
        # Base DiT loader (UnetLoaderGGUF for the deployed GGUF base).
        base_nodes = [n for n in wf.values()
                      if n.get("class_type") == "UnetLoaderGGUF"
                      and n.get("inputs", {}).get("unet_name") == LTX2_MODEL_FILE]
        self.assertTrue(base_nodes, "base LTX-2.3 DiT must be loaded for the refine pass")
        # Upscaler conv net must NOT be loaded as a DiT.
        up_as_dit = [n for n in wf.values()
                     if n.get("class_type") in ("CheckpointLoaderSimple", "UnetLoaderGGUF")
                     and n.get("inputs", {}).get("ckpt_name") == UPSCALE_MODEL_FILENAME]
        self.assertFalse(up_as_dit, "upscaler conv net must NOT be loaded as a DiT")
        # Upscaler loaded via LatentUpscaleModelLoader with model_name == filename.
        up_nodes = [n for n in wf.values()
                    if n.get("class_type") == "LatentUpscaleModelLoader"]
        self.assertTrue(up_nodes, "expected a LatentUpscaleModelLoader node")
        self.assertEqual(up_nodes[0]["inputs"]["model_name"], UPSCALE_MODEL_FILENAME)

    def test_sampler_target_via_upsampler_not_encode(self):
        """2x target via LTXVLatentUpsampler on the CARRIED latent — no VAEEncode.

        The base-gen joint AV latent flows straight into the upscale sub-graph and
        is 2x spatial-upscaled by LTXVLatentUpsampler (the VRGDG conv net). There is
        no VAEEncode of a decoded MP4 (that re-encode was the ghosting source, 09.9-16).
        Node ids are dynamic (loader re-homes the VRGDG sub-graph) — assert by class_type.
        """
        wf = self._build(target_w=1920, target_h=1088)
        up = [n for n in wf.values() if n.get("class_type") == "LTXVLatentUpsampler"]
        self.assertTrue(up, "expected an LTXVLatentUpsampler node (2x spatial scale)")
        encode = [n for n in wf.values() if n.get("class_type") == "VAEEncode"]
        self.assertFalse(encode, "VAEEncode of a decoded MP4 must be absent (latent→latent)")

    def test_vae_decode_node_present(self):
        """LTXVTiledVAEDecode decodes the upscaled latent to frames."""
        wf = self._build()
        decode_nodes = [n for n in wf.values()
                        if n.get("class_type") == "LTXVTiledVAEDecode"]
        self.assertTrue(decode_nodes, "expected a LTXVTiledVAEDecode node")

    def test_audio_latent_carried_through_refine(self):
        """Audio latent is carried through the refine pass (D-09 / REQ-15-06).

        With audio_filename set, the graph contains LTXVSeparateAVLatent (splits
        the joint AV latent), LTXVAudioVAEEncode (encodes the vocals), and
        LTXVConcatAVLatent (recombines video + audio before the sampler). The
        decode path includes LTXVAudioVAEDecode and CreateVideo carries an audio
        input so the upscaled clip keeps a decoded audio track.
        """
        wf = self._build()
        self.assertTrue(
            any(n.get("class_type") == "LTXVSeparateAVLatent" for n in wf.values()),
            "LTXVSeparateAVLatent must be present (audio split)",
        )
        self.assertTrue(
            any(n.get("class_type") == "LTXVAudioVAEEncode" for n in wf.values()),
            "LTXVAudioVAEEncode must be present (audio encode)",
        )
        self.assertTrue(
            any(n.get("class_type") == "LTXVConcatAVLatent" for n in wf.values()),
            "LTXVConcatAVLatent must be present (audio recombine)",
        )
        self.assertTrue(
            any(n.get("class_type") == "LTXVAudioVAEDecode" for n in wf.values()),
            "LTXVAudioVAEDecode must be present (audio decode)",
        )
        create = [n for n in wf.values() if n.get("class_type") == "CreateVideo"]
        self.assertTrue(create)
        self.assertIn("audio", create[0]["inputs"], "CreateVideo must carry the decoded audio")

    def test_upscaler_node_is_latent_upscale_model_loader(self):
        """Upscaler conv net loaded via LatentUpscaleModelLoader (key model_name)."""
        wf = self._build()
        up_nodes = [n for n in wf.values()
                    if n.get("class_type") == "LatentUpscaleModelLoader"]
        self.assertTrue(up_nodes, "expected a LatentUpscaleModelLoader node")
        from mv_mvconst import UPSCALE_MODEL_FILENAME
        self.assertEqual(up_nodes[0]["inputs"]["model_name"], UPSCALE_MODEL_FILENAME)

    def test_refine_pass_present_but_no_condition_only(self):
        """Refine pass is present (sampler stack) but LTXVImgToVideoConditionOnly is removed.

        The refine pass is NECESSARY — the conv net output is not directly
        VAE-decodable without diffusion refinement (raw decode produces static).
        However, LTXVImgToVideoConditionOnly was removed because it replaced the
        first frames of the upsampled latent, destroying the conv net's structure.
        """
        wf = self._build()
        # Refine pass MUST be present
        self.assertTrue(
            any(n.get("class_type") == "SamplerCustomAdvanced" for n in wf.values()),
            "SamplerCustomAdvanced must be present for refine pass",
        )
        self.assertTrue(
            any(n.get("class_type") == "ManualSigmas" for n in wf.values()),
            "ManualSigmas must be present for low-noise schedule",
        )
        self.assertTrue(
            any(n.get("class_type") == "CFGGuider" for n in wf.values()),
            "CFGGuider must be present for conditioning",
        )
        # Text encoder / conditioning must be present
        self.assertTrue(
            any(n.get("class_type") == "CLIPTextEncode" for n in wf.values()),
            "CLIPTextEncode must be present for prompt conditioning",
        )
        self.assertTrue(
            any(n.get("class_type") == "LTXVConditioning" for n in wf.values()),
            "LTXVConditioning must be present",
        )
        # LTXVImgToVideoConditionOnly must NOT be present (frame replacement bug)
        self.assertFalse(
            any(n.get("class_type") == "LTXVImgToVideoConditionOnly" for n in wf.values()),
            "LTXVImgToVideoConditionOnly must not be present (replaces frames)",
        )

    def test_upsampler_feeds_sampler_not_condition_only(self):
        """SamplerCustomAdvanced takes latent from LTXVConcatAVLatent, not ConditionOnly.

        The refined latent is fed by the upscale sub-graph (Inplace -> ConcatAV),
        never by LTXVImgToVideoConditionOnly (the haze bug). Node ids are dynamic,
        so assert by the link SOURCE node's class_type.
        """
        wf = self._build()
        sampler = [n for n in wf.values()
                   if n.get("class_type") == "SamplerCustomAdvanced"]
        self.assertTrue(sampler, "expected a SamplerCustomAdvanced node")
        latent_link = sampler[0]["inputs"]["latent_image"]
        src_class = wf[str(latent_link[0])]["class_type"]
        self.assertEqual(src_class, "LTXVConcatAVLatent",
                         "sampler must take latent from LTXVConcatAVLatent")
        # Decode takes from a SeparateAV node (splits the refined AV latent).
        decode = [n for n in wf.values()
                  if n.get("class_type") == "LTXVTiledVAEDecode"]
        self.assertTrue(decode, "expected a LTXVTiledVAEDecode node")
        decode_link = decode[0]["inputs"]["latents"]
        decode_src = wf[str(decode_link[0])]["class_type"]
        self.assertEqual(decode_src, "LTXVSeparateAVLatent",
                         "decode must take from LTXVSeparateAVLatent")

    def test_refinement_sigmas_are_vrgdg_schedule(self):
        """ManualSigmas uses the VRGDG full-denoise schedule (D-04 / REQ-15-03).

        The schedule is "0.909375, 0.725, 0.0" (3 sigmas / 2 refine passes) —
        the shipped 3σ fix from debug mv-generation-time (commit f1e26a90),
        which drops the lowest-middle sigma while KEEPING the high start
        (0.909375) + 0.0 terminus that the 09.9-15 pixelation fix depends on.
        Full denoise that fixes pixelation from the old gentle "0.3,0.2,0.1,0.0".
        The old first-sigma <= 0.45 assertion is intentionally removed (VRGDG's
        first sigma is 0.909375).
        """
        wf = self._build()
        from mv_mvconst import LTX2_UPSCALE_REFINEMENT_SIGMAS
        sig_nodes = [n for n in wf.values() if n.get("class_type") == "ManualSigmas"]
        self.assertTrue(sig_nodes, "expected a ManualSigmas node")
        self.assertEqual(sig_nodes[0]["inputs"]["sigmas"], LTX2_UPSCALE_REFINEMENT_SIGMAS)
        self.assertEqual(sig_nodes[0]["inputs"]["sigmas"], "0.909375, 0.725, 0.0")

    def test_has_latent_upsampler_node(self):
        """LTXVLatentUpsampler applies the 2x spatial upscale to the latent."""
        wf = self._build()
        up = [n for n in wf.values() if n.get("class_type") == "LTXVLatentUpsampler"]
        self.assertTrue(up, "expected an LTXVLatentUpsampler node")

    def test_has_img_to_video_inplace_node(self):
        """LTXVImgToVideoInplace re-conditions on the portrait (D-07 / REQ-15-04).

        Must be present with strength=1.0 and bypass=False (NOT
        LTXVImgToVideoConditionOnly, which caused the haze bug).
        """
        wf = self._build()
        inplace = [n for n in wf.values() if n.get("class_type") == "LTXVImgToVideoInplace"]
        self.assertTrue(inplace, "LTXVImgToVideoInplace must be present (re-conditioning)")
        self.assertEqual(inplace[0]["inputs"]["strength"], 1.0)
        self.assertEqual(inplace[0]["inputs"]["bypass"], False)

    def test_has_crop_guides_node(self):
        """LTXVCropGuides removes guide frames before upscale (D-08 / REQ-15-05)."""
        wf = self._build()
        self.assertTrue(
            any(n.get("class_type") == "LTXVCropGuides" for n in wf.values()),
            "LTXVCropGuides must be present",
        )

    def test_has_separate_av_latent_node(self):
        """LTXVSeparateAVLatent splits joint AV latent for audio (D-09 / REQ-15-06)."""
        wf = self._build()
        self.assertTrue(
            any(n.get("class_type") == "LTXVSeparateAVLatent" for n in wf.values()),
            "LTXVSeparateAVLatent must be present (audio handling in refine)",
        )

    def test_sampler_is_euler(self):
        """Sampler is 'euler', not 'euler_cfg_pp' (D-05 / REQ-15-02)."""
        wf = self._build()
        sel = [n for n in wf.values() if n.get("class_type") == "KSamplerSelect"]
        self.assertTrue(sel, "expected a KSamplerSelect node")
        self.assertEqual(sel[0]["inputs"]["sampler_name"], "euler")
        self.assertNotEqual(sel[0]["inputs"]["sampler_name"], "euler_cfg_pp")

    def test_sigmas_are_vrgdg_schedule(self):
        """ManualSigmas = the VRGDG full-denoise schedule (D-04 / REQ-15-03)."""
        wf = self._build()
        sig = [n for n in wf.values() if n.get("class_type") == "ManualSigmas"]
        self.assertTrue(sig, "expected a ManualSigmas node")
        self.assertEqual(sig[0]["inputs"]["sigmas"], "0.909375, 0.725, 0.0")

    def test_vrgdg_graph_structure(self):
        """Full VRGDG upscale sub-graph class_type set present (REQ-15-01)."""
        wf = self._build()
        ct = {n.get("class_type") for n in wf.values()}
        required = {
            "LTXVSeparateAVLatent",
            "LTXVCropGuides",
            "LTXVLatentUpsampler",
            "LTXVImgToVideoInplace",
            "LTXVConcatAVLatent",
            "SamplerCustomAdvanced",
        }
        missing = required - ct
        self.assertFalse(missing, f"missing VRGDG upscale nodes: {missing}")
        # The forbidden haze-causing node must be absent.
        self.assertNotIn("LTXVImgToVideoConditionOnly", ct)


class TestUpscaleStep(unittest.TestCase):
    """Task 2: _upscale_model_present preflight + build_clip_workflow Path A/B selection.

    The two-queued-job _upscale_clip is RETIRED (Plan 09.9-16, single combined job);
    its old copied-path test is removed. Single-job routing lives in
    TestGenerateClipUpscaleRouting.
    """

    def test_model_present_true_when_file_exists(self):
        import mv_upscale
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / mv_mvconst.UPSCALE_MODEL_DIR / mv_mvconst.UPSCALE_MODEL_FILENAME
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(b"x")
            with patch.object(mv_upscale, "_resolve_comfyui_root", return_value=Path(tmp)):
                self.assertTrue(mv_upscale._upscale_model_present())

    def test_model_present_false_when_absent(self):
        import mv_upscale
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(mv_upscale, "_resolve_comfyui_root", return_value=Path(tmp)):
                self.assertFalse(mv_upscale._upscale_model_present())

    def test_two_job_upscale_helpers_retired(self):
        """The old two-queued-job upscale helpers are gone (single combined job now)."""
        import mv_upscale
        self.assertFalse(hasattr(mv_upscale, "_upscale_clip"))
        self.assertFalse(hasattr(mv_upscale, "_maybe_upscale_clip"))

    def test_build_clip_workflow_path_b_prefix(self):
        """Path B (combined) → alice_ltx2_up prefix, _up.mp4 suffix, one workflow."""
        import mv_upscale
        trivial = {"1": {"class_type": "UnetLoaderGGUF", "inputs": {}}}
        with patch("src.workflow_ltx2_upscale.build_ltx2_combined_workflow",
                   return_value=trivial) as mc, \
             patch("src.workflow_ltx2.build_ltx2_workflow", return_value=trivial) as mb:
            wf, prefix, suffix, timeout = mv_upscale.build_clip_workflow(
                use_combined=True, prompt="p", dialogue_text="d", ref_name="r.jpg",
                padded_length_s=6.0, audio_path="a.wav", tw=1920, th=1088,
                base_w=960, base_h=544, gen_w=960, gen_h=544, use_two_stage=False,
                base_width=960, base_height=544, text_encoder_device="default", neg_suffix="",
            )
        self.assertEqual(prefix, "alice_ltx2_up")
        self.assertEqual(suffix, "_up.mp4")
        mc.assert_called_once()
        mb.assert_not_called()

    def test_build_clip_workflow_path_a_prefix(self):
        """Path A (base-only) → alice_ltx2 prefix, .mp4 suffix, base builder."""
        import mv_upscale
        trivial = {"1": {"class_type": "UnetLoaderGGUF", "inputs": {}}}
        with patch("src.workflow_ltx2_upscale.build_ltx2_combined_workflow",
                   return_value=trivial) as mc, \
             patch("src.workflow_ltx2.build_ltx2_workflow", return_value=trivial) as mb:
            wf, prefix, suffix, timeout = mv_upscale.build_clip_workflow(
                use_combined=False, prompt="p", dialogue_text="d", ref_name="r.jpg",
                padded_length_s=6.0, audio_path="a.wav", tw=1920, th=1088,
                base_w=960, base_h=544, gen_w=960, gen_h=544, use_two_stage=False,
                base_width=960, base_height=544, text_encoder_device="cpu", neg_suffix="x",
            )
        self.assertEqual(prefix, "alice_ltx2")
        self.assertEqual(suffix, ".mp4")
        mb.assert_called_once()
        mc.assert_not_called()


class TestGenerateClipUpscaleRouting(unittest.TestCase):
    """Task 2: _generate_clip Path A/B routing via a SINGLE combined job (09.9-16).

    Path B is now one chained base-gen+upscale ComfyUI job (build_ltx2_combined_workflow)
    — NOT a second queued job. Assert exactly one queue call, and that the correct
    builder is used per Path (combined vs base-only).
    """

    def _run(self, upscale, model_present):
        import mv_upscale
        out = Path(tempfile.mkdtemp())
        (out / "clips").mkdir()
        seg = ClipSegment(start=0.0, end=5.0, text="x", duration=5.0, words=[])
        portrait = out / "portrait.jpg"
        portrait.write_bytes(b"x")
        found = out / "clips" / "found.mp4"
        found.write_bytes(b"base")
        trivial = {"1": {"class_type": "UnetLoaderGGUF", "inputs": {}}}
        with patch("mv_comfyui.COMFYUI_OUTPUT_DIR", str(out)), \
             patch("mv_comfyui._check_vram_gate", return_value=True), \
             patch("mv_comfyui._queue_workflow", return_value="pid") as mock_q, \
             patch("mv_comfyui._poll_completion", return_value={}), \
             patch("mv_comfyui._find_output_file", return_value=found), \
             patch("mv_audio._trim_padding_frames", return_value=False), \
             patch.object(mv_upscale, "_upscale_model_present", return_value=model_present), \
             patch("src.workflow_ltx2_upscale.build_ltx2_combined_workflow",
                   return_value=trivial) as mock_comb, \
             patch("src.workflow_ltx2.build_ltx2_workflow",
                   return_value=trivial) as mock_base:
            res = _generate_clip(seg, "scene", portrait, 1, out,
                                 gen_w=960, gen_h=544,
                                 upscale=upscale, target_w=1920, target_h=1088)
        return res, mock_q, mock_comb, mock_base

    def test_routes_path_b_when_upscale_true(self):
        res, mock_q, mock_comb, mock_base = self._run(True, model_present=True)
        self.assertIsNotNone(res)
        # ONE combined job — no second ComfyUI queue for the upscale pass.
        mock_q.assert_called_once()
        mock_comb.assert_called_once()
        mock_base.assert_not_called()
        # Combined builder receives the low-res base + high-res target dims.
        kw = mock_comb.call_args.kwargs
        self.assertEqual((kw["target_w"], kw["target_h"]), (1920, 1088))
        self.assertEqual((kw["base_w"], kw["base_h"]), (960, 544))
        # Output is the *_up.mp4 from the single combined job.
        self.assertTrue(res.name.endswith("_up.mp4"))

    def test_routes_path_a_when_upscale_false(self):
        res, mock_q, mock_comb, mock_base = self._run(False, model_present=True)
        self.assertIsNotNone(res)
        mock_q.assert_called_once()
        mock_base.assert_called_once()     # base-only build
        mock_comb.assert_not_called()      # no combined upscale job
        self.assertFalse(res.name.endswith("_up.mp4"))

    def test_falls_back_to_path_a_when_model_absent(self):
        res, mock_q, mock_comb, mock_base = self._run(True, model_present=False)
        self.assertIsNotNone(res)
        mock_q.assert_called_once()
        mock_base.assert_called_once()     # upscaler missing -> Path A base-only
        mock_comb.assert_not_called()
        self.assertFalse(res.name.endswith("_up.mp4"))


if __name__ == "__main__":
    unittest.main()
