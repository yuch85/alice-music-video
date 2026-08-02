"""HUMO model constants — filenames, precision, resolver dict.

Plan 09.9-24 — extracted from ``workflow_humo.py`` to keep it <=300 LOC
after adding the model-size axis. Import this module rather than duplicating
checkpoint strings.
"""

from __future__ import annotations

from typing import Any

# ── Checkpoint filenames ────────────────────────────────────────────────────
HUMO_MODEL_FILENAME: str = "Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors"
"""14B-fp8 checkpoint (Kijai conversion, scaled fp8 e4m3fn)."""

HUMO_MODEL_FILENAME_1_7B: str = "Wan2_1-HuMo-1.7B_bf16.pth"
"""1.7B bf16 checkpoint (ByteDance original ema.pth)."""

# ── Precision defaults ──────────────────────────────────────────────────────
HUMO_MODEL_1_7B_PRECISION: str = "bf16"
"""Base precision for the 1.7B checkpoint (native bf16).

Requires the WanHuMoCrossAttention dtype fix in
ComfyUI-WanVideoWrapper/wanvideo/modules/model.py (patched 2026-07-13).
The fix casts inputs to weight dtype BEFORE linear layers, matching
the safe `is_longcat` pattern in WanSelfAttention.qkv_fn.
Without the patch, float32 activations meet bf16 weights → RuntimeError.
"""


# ── Model roster ────────────────────────────────────────────────────────────
HUMO_MODELS: dict[str, dict[str, Any]] = {
    "14b": {
        "filename": HUMO_MODEL_FILENAME,
    },
    "1.7b": {
        "filename": HUMO_MODEL_FILENAME_1_7B,
        "precision": HUMO_MODEL_1_7B_PRECISION,
    },
}
"""Model-size roster keyed by alias (``14b``, ``1.7b``).

Each entry maps to ``filename`` (checkpoint basename in
``ComfyUI/models/diffusion_models/``) and optionally ``precision``
(base precision override; defaults to the builder's ``HUMO_BASE_PRECISION``).
"""
