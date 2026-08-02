#!/usr/bin/env python3
"""Post-processing LUT generation (Plan 09.9 Plan 07 / 09.9-12).

Module is kept <= 400 lines per STYLE.md (YC-approved 400 ceiling deviation
from the 300 default). Extracted from mv_post.py in 09.9-12 to recover
headroom (mv_post was 1 over the ceiling) and to isolate the stdlib-only
LUT logic so mv_post can import it without any internal cycle.

stdlib-only: logging, os, Path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Post-processing constants (Phase 09.9 Plan 07) ──

POST_PROCESS_LUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gpu-manager", "data", "luts",
)
DEFAULT_LUT_NAME = "Cine_Grade.cube"
DEFAULT_GRAIN_INTENSITY = 0.5  # subtle, 0.0-10.0 range
DEFAULT_SHARPEN_STRENGTH = 0.4  # mild, 0.0-1.5 range


def _download_default_lut() -> Path | None:
    """Generate the default Cine Grade 3D LUT file if it doesn't exist.

    Creates a 17x17x17 .cube file with a cinematic color grade:
    - Slight S-curve on luminance (contrast boost)
    - Warm shift in highlights (+R, +G slightly)
    - Cool shift in shadows (+B slightly)
    - Mild desaturation (~5%)

    Returns the path to the LUT file, or None if generation failed.
    Never raises — creates an identity LUT as last resort.
    """
    lut_dir = Path(POST_PROCESS_LUT_DIR).resolve()
    lut_path = lut_dir / DEFAULT_LUT_NAME

    if lut_path.exists():
        return lut_path

    try:
        lut_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.warning("Cannot create LUT directory %s: %s", lut_dir, e)
        return None

    try:
        # Generate Cine Grade 3D LUT (17x17x17)
        lines = ['TITLE "Cine Grade"', "LUT_3D_SIZE 17", ""]

        for i in range(17):
            for j in range(17):
                for k in range(17):
                    # Normalize to 0.0-1.0 range
                    r_in = i / 16.0
                    g_in = j / 16.0
                    b_in = k / 16.0

                    # Luminance S-curve (contrast boost)
                    lum = 0.299 * r_in + 0.587 * g_in + 0.114 * b_in
                    contrast = 1.15  # mild contrast boost
                    lum = lum ** (1.0 / contrast)

                    # Warm highlights: boost R and G slightly in upper range
                    warm_r = 0.03 if lum > 0.5 else 0.0
                    warm_g = 0.02 if lum > 0.5 else 0.0

                    # Cool shadows: boost B slightly in lower range
                    cool_b = 0.02 if lum < 0.3 else 0.0

                    # Mild desaturation (~5%)
                    desat = 0.95
                    gray = lum
                    r_out = (r_in * desat + gray * (1 - desat) + warm_r) * 1.04
                    g_out = (g_in * desat + gray * (1 - desat) + warm_g) * 1.02
                    b_out = (b_in * desat + gray * (1 - desat) + cool_b) * 1.06

                    # Clamp to [0, 1]
                    r_out = max(0.0, min(1.0, r_out))
                    g_out = max(0.0, min(1.0, g_out))
                    b_out = max(0.0, min(1.0, b_out))

                    lines.append(f"{r_out:.6f} {g_out:.6f} {b_out:.6f}")

        lut_path.write_text("\n".join(lines))
        logger.info("Generated Cine Grade LUT: %s", lut_path)
        return lut_path

    except Exception as e:
        logger.warning("Cine Grade LUT generation failed (%s). Creating identity LUT.", e)
        # Fallback: identity LUT (pass-through)
        try:
            lines = ['TITLE "Cine Grade (Identity Fallback)"', "LUT_3D_SIZE 17", ""]
            for i in range(17):
                for j in range(17):
                    for k in range(17):
                        lines.append(f"{i/16.0:.6f} {j/16.0:.6f} {k/16.0:.6f}")
            lut_path.write_text("\n".join(lines))
            logger.info("Created identity LUT fallback: %s", lut_path)
            return lut_path
        except Exception as e2:
            logger.error("Failed to create even identity LUT: %s", e2)
            return None
