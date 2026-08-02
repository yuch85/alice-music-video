#!/usr/bin/env python3
"""Constants for Plan 09.9-13 (Path A: low-res DiT + bislerp decode -> 1080p).

Single source of truth for the magic numbers this plan introduces, so they are
never hardcoded inline in the pipeline or workflow code. This is a leaf module:
it imports nothing heavy (no torch / ComfyUI / GPU deps) so it can be imported
at module load by ``generate_music_video_pipeline`` without side effects.
"""

from __future__ import annotations

# ── HuMo 14B clip duration (Plan 09.9-25-01, D-01) ──
# Canonical single source of truth for the default HuMo clip duration. Defined
# exactly once here (leaf module, no heavy imports) and aliased in
# mv_clip_generate.HUMO_DEFAULT_CLIP_DURATION_S so the two never drift.
DEFAULT_CLIP_DURATION_S = 16
"""Default HuMo 14B clip duration (seconds).

Within the 18s VRAM ceiling (09.9-19 probe); fewer clips / fewer stitch seams.
8s remains the safety fallback for drift-prone segments; 6s is the hard floor.
"""

# ── Variable per-clip duration bounds (Plan 09.9-25-04, D-01 extension) ──
# Safety fallback for drift-prone segments (8s; 6s is the hard floor). Defined
# once here as the single source of truth; plan 02 aliases it in
# mv_clip_generate.HUMO_FALLBACK_CLIP_DURATION_S so the two never drift.
HUMO_FALLBACK_CLIP_DURATION_S = 8

# document hard floor for any generated clip (6s). Below this the motion
# coherence degrades; the pipeline refuses shorter requests.
CLIP_DURATION_FLOOR_S = 6

# VRAM ceiling for a single clip (18s), from the 09.9-19 VRAM probe. Above
# this the 48GB card OOMs; the Long-CLIP splitter caps at LTX2_MAX_LENGTH_S
# which matches this ceiling.
CLIP_DURATION_CEILING_S = 18

UPSCALE_MODEL_FILENAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
"""Filename of the Lightricks LTX-2.3 generative spatial-upscaler (Path B)."""

UPSCALE_MODEL_DIR = "models/latent_upscale_models"
"""ComfyUI-relative dir the upscaler checkpoint loads from (Path B)."""

UPSCALE_FACTOR = 2
"""Generative upscaler doubles one spatial dimension (x2)."""

DEFAULT_LOWRES_W = 960
"""Default LOW-RES base generation width (2x -> 1920 output, cropped to 1080)."""

DEFAULT_LOWRES_H = 544
"""Default LOW-RES base generation height (2x -> 1088, cropped to 1080)."""

DEFAULT_MAX_CLIP_S = 6
"""Default maximum single-clip duration (seconds) for short-clip VRAM headroom."""

# ── Planner policy constants (model-aware, not hardcoded to LTX-2) ──
# Generation models with different minimum clip durations may require
# different planning policies. These constants decouple planner behavior
# from any single generation model's constraints.

PLANNER_PREFERRED_MIN_DURATION_S = 5.0
"""Preferred minimum segment duration for planning merges.

Segments below this threshold are merged with adjacent segments during
planning to avoid generation-time extension (which causes cascade drift).
Distinct from the hard generator minimum (e.g., LTX2_MIN_LENGTH_S = 4.0).
Different generation models may require different preferred minima.
"""

PLANNER_TOLERANCE_S = 0.5
"""Tolerance for plan boundary validation (start=0, end=audio_duration).

The planner output must begin at t=0 and end at audio_duration, each
within this tolerance. Used by _validate_segment_plan to enforce the
planner invariants (Refinement 5).
"""

LTX2_UPSCALE_REFINEMENT_SIGMAS = "0.909375, 0.725, 0.0"
"""Path B generative upscale REFINE sigma schedule (Plan 09.9-15, VRGDG reference).

FULL-DENOISE schedule derived from the VRGameDevGirl (VRGDG) LTX-2.3 Music Video
Creator upscale sub-graph. The 950MB ``ltx-2.3-spatial-upscaler-x2-1.1`` conv net
produces a 2x spatial latent that is NOT directly VAE-decodable; the base DiT must
refine it at near-full noise (first sigma 0.909375) so the DiT reconstructs coherent
pixels from the conv net's structure, guided by LTXVImgToVideoInplace re-conditioning
on the reference portrait.

REVISION (2026-07-12, debug mv-generation-time): reduced from 4 sigmas
("0.909375, 0.725, 0.421875, 0.0", 3 DiT passes) to 3 sigmas (2 DiT passes) to cut
per-clip generation time. Split-test timing showed the VRGDG 2x refine at full
1920x1088 was 69% of the ~220s/clip cost (152s of refine vs 68s base-gen). The HIGH
starting sigma (0.909375) and the 0.0 full-denoise terminus are PRESERVED — only the
lowest-middle sigma (0.421875) is dropped. This is what avoids regressing the 09.9-15
pixelation fix: blocky/frozen output came from a LOW starting sigma (<=0.45, the old
"0.3,0.2,0.1,0.0" schedule), NOT from step count. Keeping the high starting sigma
means the conv-net output is still strongly re-noised and reconstructed, only ~1 step
less refined — at most marginally softer, not pixelated.

The audio path (BUG A lip-sync) and ffmpeg bitrate (BUG B) are untouched by this
constant, so neither regresses. Visual sharpness of the 2-step refine is a
visual-QA item (human-verify checkpoint in the mv-generation-time debug session)."""

# ── Tiled VAE decode overlap values (M-4) ──
# Different VAE decode nodes use different parameter semantics for "overlap".
# ComfyUI-LTXVideo changed LTXVTiledVAEDecode from pixel-space to latent-space
# (breaking change, 2026-07-31). These constants prevent accidental mix-ups.

LTXV_TILED_VAE_OVERLAP = 8
"""Overlap for LTXVTiledVAEDecode node (latent-space, 1-8 range).

ComfyUI-LTXVideo changed this parameter from pixel-space (0-64) to
latent-space (1-8). Value 8 is correct for current ComfyUI-LTXVideo.
Used in workflow_ltx2_upscale.py for the Path B combined workflow.
"""

STANDARD_TILED_VAE_OVERLAP = 64
"""Overlap for VAEDecodeTiled node (pixel-space, 0-64 range).

Standard ComfyUI VAEDecodeTiled node uses pixel-space overlap.
Used in workflow_ltx2.py for non-LTXV tiled decode.
"""
