---
name: music-video-qc
description: Final QC — runtime, resolution, FPS, black frames, frozen clips, timeline coverage, audio sync
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
---

# /music-video-qc

Final quality control (Stage 10). Verifies the compiled final video against the generation manifest.

## Prerequisites

- FSM state must be `QC`.
- `final_output.mp4` exists.
- `generation_manifest.md` exists.

## QC Checks

Run 8 quality checks using ffprobe/ffmpeg:

1. **Runtime**: Video duration matches full-timeline audio duration (`audio_full.mp3`, within 0.5s tolerance).
2. **Resolution**: Output is 1920x1080 (or configured resolution).
3. **FPS**: Frame rate is consistent (24/25/30 fps).
4. **Black Frames**: No extended black frames (> 2s) except intentional fades.
5. **Frozen Clips**: No frame-frozen segments (> 3s of identical frames).
6. **Timeline Coverage**: Video covers 0 -> full video duration with no gaps.
7. **Audio Sync**: Audio track present, matches full-timeline audio. Verify prologue section (if any) has silence/ambient content, not song content.
8. **Clip Count**: Number of clips matches segment count from manifest.

## Additional Timing Checks (post-v7 lessons)

9. **Audio-Video Duration Match**: Verify that the audio stream duration matches the video stream duration within 0.1s tolerance. Use ffprobe to extract both `format.duration` and per-stream `duration`. If audio is shorter than video, the final frames will hold the last audio packet (audible artifact). If longer, audio will be silently truncated.

   ```bash
   ffprobe -v error -show_entries stream=codec_type,duration -of json final_output.mp4
   ```

10. **Transcript Source Verification**: If subtitles are burned, verify the transcript's `"audio_source"` field matches the expected source. A transcript from `vocals.wav` (song-relative) used against a video with prologue will cause subtitles to appear ~30s early. Check:

   ```bash
   python -c "import json; t=json.load(open('lyrics/transcript.json')); print(t.get('audio_source','MISSING'))"
   ```

   If source is a stem file but the video has a prologue, the first lyric timestamp should be > prologue duration. If it's < prologue duration, the transcript source is wrong.

## Process

1. Run ffprobe on `final_output.mp4` for metadata (duration, resolution, FPS, streams):
   ```bash
   ffprobe -v quiet -print_format json -show_format -show_streams final_output.mp4
   ```

2. Detect black frames via ffmpeg `blackdetect` filter:
   ```bash
   ffmpeg -i final_output.mp4 -vf blackdetect=d=2:pix_threshold=0.01:picture_threshold=0.95 -f null - 2>&1
   ```
   Parse stderr for timeline timestamps.

3. Detect frozen clips via ffmpeg `freezedetect` filter:
   ```bash
   ffmpeg -i final_output.mp4 -vf freezedetect=d=3:freeze_thresh=0.01 -f null - 2>&1
   ```
   Parse stderr for freeze start/end timestamps.

4. Compare against generation manifest expectations.

5. Write `qc_report.md` with pass/fail per check, including black/freeze timestamps.

## Critical

Use ffmpeg `blackdetect` and `freezedetect` filters via subprocess and parse their stderr output for timeline anomalies. Do NOT build frame-by-frame loops in Python — ffmpeg handles this efficiently in C.

## Per-Clip QC Guidance

When individual clips fail QC, common rejection reasons from production experience:

| Rejection Reason | Example | Fix |
|-----------------|---------|-----|
| Repetitive setting | Lake shot following a lake shot | Swap to a different keyframe with a different environment |
| Wrong props | Guitar when character should not have one | Regenerate keyframe with explicit "no instrument" prompt |
| Unnatural objects | Microphone appearing from water | Regenerate keyframe with explicit negative prompt |
| Static scene | No visible camera motion potential | Rewrite motion prompt with explicit camera movement |
| Identity drift | Face doesn't match canonical portrait | Regenerate keyframe from canonical portrait with tighter identity lock |
| Wardrobe inconsistency | Wrong color or garment | Regenerate keyframe with explicit wardrobe string from Continuity Bible |

When regenerating a clip:
1. Check if the issue is with the keyframe (composition, identity, props) or the motion prompt (camera movement, body motion).
2. If keyframe issue: regenerate the keyframe via QEI, get Human QC approval, then regenerate the clip.
3. If motion prompt issue: rewrite the motion prompt and regenerate the clip with the same keyframe.

## FSM Update

On QC pass, transition QC -> COMPLETE:
```bash
python scripts/mv_fsm_cli.py transition <project_dir> QC APPROVED
```

On failure, stay at QC with specific issues for the user to address. Do not implement inline state updates.
