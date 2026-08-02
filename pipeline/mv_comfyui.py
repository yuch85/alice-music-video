#!/usr/bin/env python3
"""ComfyUI interaction layer (Plan 09.9-10).

Thin constant + delegating-callable + re-export layer. All real logic lives
in `mv_comfyui_client.ComfyUIClient`; this module keeps MODULE-LEVEL
delegating callables for every helper the test patches (BLOCKER 1) so
`mock.patch("mv_comfyui.<name>")` intercepts internal calls made by other
modules (mv_clip / mv_refs) via `mv_comfyui.<name>` attribute lookup.

The single module-scope `comfyui_client` singleton is the ONLY module-level
mutable object (Manual DI). PRE_ROLL_FRAMES / TAIL_LOSS_FRAMES /
MAX_CONSECUTIVE_COMFYUI_FAILURES live here as the canonical source.

Module is kept <= 400 lines per STYLE.md (YC-approved 400 ceiling deviation
from the 300 default).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from mv_comfyui_client import ComfyUIClient
from mv_slingshot import GPU_MANAGER_BASE
from mv_vram import LTX2_VRAM_MB

logger = logging.getLogger(__name__)

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8199"))
COMFYUI_BASE = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

COMFYUI_OUTPUT_DIR = os.getenv("COMFYUI_DIR", "/path/to/ComfyUI")

# Pre-roll / tail-loss padding frames (also used by mv_audio / mv_clip).
PRE_ROLL_FRAMES = 4
"""Extra frames generated at clip start (trimmed after generation)."""

TAIL_LOSS_FRAMES = 4
"""Extra frames generated at clip end (trimmed after generation)."""

MAX_CONSECUTIVE_COMFYUI_FAILURES = 2  # abort after 2 consecutive failures


# ── Module-level delegating callables (BLOCKER 1: test interception) ──
# Every helper the test patches is kept as a module-level callable that
# delegates to the `comfyui_client` singleton, so mock.patch on
# mv_comfyui.<name> intercepts internal calls routed via mv_comfyui.<name>.


def _comfyui_post(url: str, payload: dict[str, Any], timeout: int = 30) -> dict:
    """POST JSON to ComfyUI API (delegate to ComfyUIClient)."""
    return comfyui_client.comfyui_post(url, payload, timeout)


def _comfyui_get(url: str, timeout: int = 30) -> dict:
    """GET JSON from ComfyUI API (delegate to ComfyUIClient)."""
    return comfyui_client.comfyui_get(url, timeout)


def _comfyui_is_ready() -> bool:
    """Check if ComfyUI is responding (delegate to ComfyUIClient)."""
    return comfyui_client.is_ready()


def _queue_workflow(workflow: dict[str, Any]) -> str:
    """Submit workflow to ComfyUI, return prompt_id (delegate)."""
    return comfyui_client.queue_workflow(workflow)


def _poll_completion(prompt_id: str, timeout: int = 600) -> dict:
    """Poll /history/{id} until prompt_id appears (delegate)."""
    return comfyui_client.poll_completion(prompt_id, timeout)


def _find_output_file(history_entry: dict, output_prefix: str, output_ext: str) -> Path:
    """Extract output file path from ComfyUI history entry (delegate)."""
    return comfyui_client.find_output_file(history_entry, output_prefix, output_ext)


def _start_comfyui_via_gpu_manager() -> bool:
    """Start ComfyUI via gpu-manager /comfyui/start (delegate)."""
    return comfyui_client.start_via_gpu_manager()


def _wait_for_comfyui_ready(timeout: int = 30) -> bool:
    """Wait for ComfyUI to become ready, polling every 2s (delegate)."""
    return comfyui_client.wait_for_ready(timeout)


def _check_vram_gate(min_free_mb: int | None = None) -> bool:
    """VRAM safety gate before each clip generation (delegate)."""
    return comfyui_client.check_vram_gate(min_free_mb)


def _reset_comfyui_state(timeout_s: int = 30) -> bool:
    """Clear ComfyUI internal state before a clip-gen retry (delegate)."""
    return comfyui_client.reset_state(timeout_s)


def _stop_comfyui_via_gpu_manager() -> bool:
    """Stop ComfyUI via gpu-manager /comfyui/stop endpoint.

    Used for abort cleanup — when the pipeline fails (no clips generated),
    ComfyUI is stopped before waking the LLM to avoid VRAM thrash.

    Returns True if the stop call succeeded, False otherwise.
    """
    import urllib.error
    import urllib.request

    try:
        url = f"{comfyui_client._gpu_manager_base}/comfyui/stop"
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info("ComfyUI: stop via gpu-manager (status=%d)", resp.status)
            return True
    except urllib.error.URLError as e:
        logger.warning("ComfyUI: stop via gpu-manager failed: %s", e)
        return False
    except Exception as e:
        logger.warning("ComfyUI: stop via gpu-manager unavailable: %s", e)
        return False


def _validate_and_migrate_ltx2_models() -> bool:
    """Validate LTX-2.3 model files are in correct ComfyUI `models/` subdirs.

    ComfyUI scans subdirectories under `models/` (e.g. `models/diffusion_models/`,
    `models/vae/`, `models/text_encoders/`, `models/checkpoints/`, `models/unet/`).
    Top-level directories (e.g. `ComfyUI/diffusion_models/`) are NOT scanned.

    If a model file is found in a wrong top-level directory, it is moved
    (not copied) to the correct location. Returns True if all required models
    are in place after migration, False otherwise.

    Mapping of model role -> (source top-level dir, correct dest subdir(s)):
      - GGUF DiT       -> diffusion_models/  -> models/diffusion_models/ + models/unet/
      - Video VAE      -> vae/               -> models/vae/
      - Audio VAE      -> vae/               -> models/vae/
      - Embeddings     -> text_encoders/     -> models/text_encoders/ + models/checkpoints/
      - Text encoder   -> text_encoders/     -> models/text_encoders/
    """
    try:
        from workflows.workflow_ltx2 import (
            LTX2_MODEL_FILE,
            LTX2_VAE_FILE,
            LTX2_AUDIO_VAE_FILE,
            LTX2_TEXT_PROJECTION_FILE,
            LTX2_TEXT_ENCODER_FILE,
        )
    except ImportError:
        logger.warning("Cannot import workflow_ltx2 — skipping model validation")
        return True

    comfyui_dir = Path(COMFYUI_OUTPUT_DIR)

    # (filename, wrong_top_level_subdir, list_of_correct_subdirs_under_models/)
    model_checks: list[tuple[str, str, list[str]]] = [
        (LTX2_MODEL_FILE, "diffusion_models", ["diffusion_models", "unet"]),
        (LTX2_VAE_FILE, "vae", ["vae"]),
        (LTX2_AUDIO_VAE_FILE, "vae", ["vae"]),
        (LTX2_TEXT_PROJECTION_FILE, "text_encoders", ["text_encoders", "checkpoints"]),
        (LTX2_TEXT_ENCODER_FILE, "text_encoders", ["text_encoders"]),
    ]

    all_ok = True
    for filename, wrong_dir, correct_dirs in model_checks:
        wrong_path = comfyui_dir / wrong_dir / filename
        dest_paths = [comfyui_dir / "models" / d / filename for d in correct_dirs]

        # Check if already in correct location
        if any(dp.exists() for dp in dest_paths):
            # Clean up misplaced file if it exists
            if wrong_path.exists():
                wrong_path.unlink()
                logger.info("Cleaned up misplaced model: %s", wrong_path.relative_to(comfyui_dir))
            continue

        # Check if file exists in wrong location
        if wrong_path.exists():
            logger.warning(
                "Model '%s' found in wrong location (%s). "
                "Moving to models/%s...",
                filename,
                wrong_path.relative_to(comfyui_dir),
                ", ".join(correct_dirs),
            )
            for dp in dest_paths:
                dp.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(wrong_path), str(dp))
                    logger.info("  -> moved to %s", dp.relative_to(comfyui_dir))
                except (shutil.Error, OSError) as e:
                    logger.error("  -> failed to move to %s: %s", dp.relative_to(comfyui_dir), e)
            continue

        # File not found anywhere
        logger.error("Required model not found: %s (checked %s and models/%s)",
                      filename, wrong_path.relative_to(comfyui_dir),
                      ", ".join(correct_dirs))
        all_ok = False

    if not all_ok:
        logger.error(
            "Some LTX-2.3 models are missing. Ensure all model files are in "
            "the correct ComfyUI/models/ subdirectories. See Phase 09.9 docs."
        )

    return all_ok


# ── Module-scope singleton (Manual DI; only module-level mutable object) ──

comfyui_client = ComfyUIClient(
    comfyui_base=COMFYUI_BASE,
    comfyui_output_dir=COMFYUI_OUTPUT_DIR,
    gpu_manager_base=GPU_MANAGER_BASE,
    ltx2_vram_mb=LTX2_VRAM_MB,
)
