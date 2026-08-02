---
name: alice-lipsync
description: MuseTalk offline lip-sync for Alice-sister (Alia, Ishi, Tara, Koko) dialogue shots. Composites a synthesized-voice audio track onto a face-visible source video so the mouth matches the speech. Wraps `~/alice/scripts/alice_musetalk_offline.py` which drives MuseTalkEngine directly — bypasses the broken upstream `scripts/inference.py` mmpose dependency (BUG-012).
allowed-tools:
  - Bash
  - Read
---

# /alice-lipsync

Offline MuseTalk lip-sync for Alice-sister dialogue videos. Takes a
face-visible source clip + a synthesized voice track and produces an
MP4 where the subject's mouth matches the audio.

Wraps `~/alice/scripts/alice_musetalk_offline.py` — the engine-class
path that sidesteps the mmpose / mmcv / mmdet install that upstream
`scripts/inference.py` requires. See
`reference_musetalk_offline_inference.md` for the mmpose-avoidance
context; attempting to install mmpose via `openmim` downgraded
setuptools and broke the venv (2026-04-14), so this wrapper is the
durable path.

## Usage

```
/alice-lipsync <video_in> <audio_in> <video_out> [--fps 16]
```

**Arguments:**
- `<video_in>` — Path to face-visible source video (typically a WAN I2V
  output at 16fps, 832×480 or 480×832). Subject's head must be framed
  at head-and-shoulders or tighter — MuseTalk's Haar-cascade detector
  will fail on long-shot framing.
- `<audio_in>` — Path to the voice track. `.ogg`, `.wav`, or `.mp3`
  accepted. Internally librosa-resampled to 24kHz mono float32; any
  source sample rate works.
- `<video_out>` — Destination MP4. Synced face is composited onto the
  source frames; audio is muxed back via ffmpeg (libx264, aac).
- `--fps` — FPS of the source video. **Must match** or frames will
  drift. I2V default = 16; if you fed MuseTalk a 24fps or 25fps clip,
  pass it explicitly.

## How it works

The wrapper:
1. Loads `MuseTalkEngine` from `~/moot-bench/src/avatar/pipeline/`.
2. Calls `engine.prepare_avatar(role, video_path)` — Haar-cascade face
   detection over the source frames, builds a per-frame reference
   pool. Capped at 25 distinct references; longer clips cycle.
3. Loads audio at 24kHz mono via librosa.
4. `engine.generate_frames(role, pcm_24k)` runs the MuseTalk decoder;
   returns a list of np.uint8 BGR frames at source resolution.
5. Writes frames to a temp silent `.mp4` via OpenCV (`mp4v` fourcc).
6. `ffmpeg -i silent.mp4 -i audio -c:v libx264 -c:a aac -shortest out.mp4`
   muxes the voice back in.

## Invocation

Run through the moot-bench venv (Python 3.12, has all the CV/ML deps).
**`cd` into `$MOOT_BENCH_DIR/` first** — the underlying MuseTalk
VAE loader resolves `models/sd-vae` as a relative path at load time,
so running from anywhere else fails with
`OSError: models/sd-vae is not a local folder`.

```bash
cd $MOOT_BENCH_DIR && \
  $MOOT_BENCH_DIR/.venv/bin/python \
  $MV_REPO_DIR/scripts/alice_musetalk_offline.py \
  --fps 16 \
  --task alia:$MV_REPO_DIR/downloads/alia_shot4.mp4:$MV_REPO_DIR/downloads/alia_vo_line4.ogg:$MV_REPO_DIR/downloads/alia_shot4_synced.mp4
```

The `--task` arg is `role:video:audio:output` with colon separators.
`--task` can repeat — useful when you want one engine-init for a burst
of lipsync jobs across sisters.

Example multi-task burst:
```bash
$MOOT_BENCH_DIR/.venv/bin/python \
  $MV_REPO_DIR/scripts/alice_musetalk_offline.py \
  --fps 16 \
  --task alia:downloads/alia_shot4.mp4:downloads/alia_vo4.ogg:downloads/alia_shot4_synced.mp4 \
  --task alia:downloads/alia_shot6.mp4:downloads/alia_vo6.ogg:downloads/alia_shot6_synced.mp4 \
  --task koko:downloads/koko_ja.mp4:downloads/koko_vo_ja.ogg:downloads/koko_ja_synced.mp4
```

## Pre-flight checks

Before running:

1. **Source framing.** Head-and-shoulders or tighter. Full-body shots
   will have mouths too small for Haar-cascade to latch onto, and you
   get no-op output (or engine raises "no frames generated").
2. **Audio duration vs video duration.** `-shortest` in the ffmpeg mux
   cuts to whichever is shorter. If the voice is longer than the
   video, extend the clip first (WAN I2V with `length=` higher, or
   ffmpeg loop). If the video is longer, synth more voice.
3. **FPS match.** WAN I2V defaults to 16fps. If you've stitched clips
   at 25fps or 30fps elsewhere, pass `--fps` matching the source — no
   resampling happens internally, frames just drift.

## Known limits

- **Reference pool capped at 25 frames.** Long source clips loop through
  the same face refs, so very long clips may look subtly repetitive.
- **Haar-cascade detector only** — no mmpose landmark regression. Fine
  for front-facing close-ups; degrades on profile shots or occluded
  faces.
- **No mmpose = no upstream CLI.** Don't try `python -m scripts.inference`
  inside the MuseTalk repo — it dies on `from mmpose.apis import ...`.
  This wrapper is the only working offline path.
- **Venv discipline.** Use `$MOOT_BENCH_DIR/.venv/bin/python`
  explicitly. Don't `uv run` — the script path-hacks
  `moot-bench/src` and `OpenAvatarChat/src/handlers/avatar/musetalk/MuseTalk`
  into `sys.path` and expects the moot-bench deps (cv2, librosa,
  torch/CUDA, MuseTalk weights) already resolved in that venv.

## When to use vs. skip

**Use** when you have a face-visible Alice-sister I2V clip and a
generated voice line that needs mouth sync — the whole point of
the stitched Flux→WAN→GPT-SoVITS/Orpheus→MuseTalk pipeline.

**Skip** for:
- Narration over non-face shots (VO stays narrator-style; no sync needed).
- Shots where the face is too small / too profile / occluded — the
  detector will silently mis-lock or fail.
- Anything planned through LTX-2 / Ovi (unified video+audio models
  handle sync natively — though neither takes reference images today,
  so canonical sister-face work still goes through this stitch).

## Related memory

- `reference_musetalk_offline_inference.md` — why mmpose install is
  avoided and the engine-class wrapper pattern.
- `reference_ffmpeg_pipeline_gotchas.md` — filter_complex / BGM / SRT
  recipes for the downstream stitch step.
- `reference_canonical_sister_portraits.md` — canonical face URLs + the
  Flux-FaceID body-frame recipe upstream of I2V.
