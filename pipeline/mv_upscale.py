#!/usr/bin/env python3
"""Path B upscale preflight (Plan 09.9-16).

Single responsibility: report whether the Lightricks
``ltx-2.3-spatial-upscaler-x2-1.1`` conv-net checkpoint is present on disk, so
``mv_clip._generate_clip`` can choose between the single combined base-gen+upscale
job (Path B) and the base-only Path A bislerp fallback.

Plan 09.9-16 RETIRED the old two-queued-job upscale step. Base generation and the
VRGDG upscale sub-graph now run as ONE chained ComfyUI job built by
``build_ltx2_combined_workflow`` (latent→latent handoff, no MP4 round-trip, no
inter-job model unload). The former ``_upscale_clip`` / ``_maybe_upscale_clip``
second-job orchestration is gone — this module keeps only the model preflight.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mv_comfyui
import mv_mvconst

logger = logging.getLogger(__name__)


def _resolve_comfyui_root() -> Path:
    """Resolve the ComfyUI root dir robustly from mv_comfyui (env COMFYUI_DIR)."""
    root = Path(getattr(mv_comfyui, "COMFYUI_OUTPUT_DIR", os.environ.get("COMFYUI_DIR", "/path/to/ComfyUI")))
    return root


def _upscale_model_present() -> bool:
    """Return True if the Path B upscaler checkpoint is present on disk.

    Logs a warning (no crash) and returns False when absent so the caller can
    fall back to the Path A bislerp clip. Never raises for a missing file.
    """
    root = _resolve_comfyui_root()
    model_path = root / mv_mvconst.UPSCALE_MODEL_DIR / mv_mvconst.UPSCALE_MODEL_FILENAME
    try:
        present = model_path.exists() and model_path.stat().st_size > 0
    except OSError as e:
        logger.warning("Upscaler model check failed (%s): %s", model_path, e)
        return False
    if present:
        logger.info("Upscaler model present: %s", model_path)
    else:
        logger.warning(
            "Upscaler model ABSENT — Path B upscale skipped, falling back to "
            "Path A bislerp. Expected: %s",
            model_path,
        )
    return present


def build_clip_workflow(
    *,
    use_combined: bool,
    prompt: str,
    dialogue_text: str,
    ref_name: str | None,
    padded_length_s: float,
    audio_path: str | None,
    tw: int,
    th: int,
    base_w: int,
    base_h: int,
    gen_w: int,
    gen_h: int,
    use_two_stage: bool,
    base_width: int,
    base_height: int,
    text_encoder_device: str,
    neg_suffix: str,
    use_lipdub: bool = True,
    output_audio_filename: str | None = None,
    use_vrdg_sigmas: bool = False,
) -> tuple[dict, str, str, int]:
    """Build the per-clip ComfyUI workflow and return (workflow, save_prefix, dest_suffix, timeout).

    Path B (``use_combined``): ONE chained base-gen+upscale job (Plan 09.9-16,
    latent→latent) — base samples at LOW res (``base_w`` x ``base_h``), refine
    outputs HIGH res (``tw`` x ``th``). Text encoder on CPU (golden setting from
    09.9-28 A/B tests) to avoid OOM. When ``use_vrdg_sigmas=True``, the base-gen
    sampling uses the VRDG V5.1 9-step sigma schedule instead of the default
    KSampler (8 steps, internal sigma computation).
    Path A fallback: base-only generation (bislerp scaled at the ffmpeg stitch).

    Builders are imported lazily so test patches on
    ``src.workflow_ltx2_upscale.build_ltx2_combined_workflow`` /
    ``src.workflow_ltx2.build_ltx2_workflow`` intercept at call time.
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu-manager"))
    if use_combined:
        # Keep float — _ltx2_num_frames computes 8k+1 from the raw frame count.
        # Rounding to int here drops the pre-roll/tail-loss padding.
        length_s = padded_length_s
        from workflows.workflow_ltx2_upscale import build_ltx2_combined_workflow

        wf = build_ltx2_combined_workflow(
            target_w=tw, target_h=th, base_w=base_w, base_h=base_h,
            scene_prompt=prompt, dialogue_text=dialogue_text,
            ref_image_filename=ref_name, length_s=length_s, audio_filename=audio_path,
            seed=None, use_tiled_vae=True,
            mute_audio=(audio_path is None), text_encoder_device="cpu",
            output_audio_filename=output_audio_filename,
            use_lipdub=use_lipdub,
            use_vrdg_sigmas=use_vrdg_sigmas,
        )
        return wf, "alice_ltx2_up", "_up.mp4", 900

    import workflows.workflow_ltx2 import build_ltx2_workflow

    # Path A rounds to int (no pre-roll padding needed for fallback bislerp path)
    length_s_a = int(round(padded_length_s))
    wf = build_ltx2_workflow(
        scene_prompt=prompt, dialogue_text=dialogue_text, ref_image_filename=ref_name,
        length_s=length_s_a, seed=None, mute_audio=(output_audio_filename is None),
        neg_suffix=neg_suffix,
        use_tiled_vae=True, width=gen_w, height=gen_h, use_two_stage=use_two_stage,
        base_width=base_width, base_height=base_height, audio_path=audio_path,
        text_encoder_device=text_encoder_device,
        output_audio_filename=output_audio_filename,
        use_vrdg_sigmas=use_vrdg_sigmas,
    )
    return wf, "alice_ltx2", ".mp4", 600
