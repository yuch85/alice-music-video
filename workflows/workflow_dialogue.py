"""Aggregator for dialogue-capable I2AV workflow builders.

Split into per-model modules to honor good-dev-practices' 300-LOC cap and
CONTEXT.md fallback rule 5 (per-builder 120-LOC ceiling). The per-model
modules own their filename constants + helpers; this module re-exports
both builders for the two consumers (`src/workflows.py` and
`src/server_dialogue.py`).
"""

from __future__ import annotations

from workflows.workflow_ltx2 import build_ltx2_i2av_workflow, build_ltx2_workflow
from workflows.workflow_ovi import build_ovi_i2av_workflow

__all__ = ["build_ltx2_i2av_workflow", "build_ltx2_workflow", "build_ovi_i2av_workflow"]
