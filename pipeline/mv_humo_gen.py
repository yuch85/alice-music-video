#!/usr/bin/env python3
"""Shared HuMo 14B talking-head clip generator (Plan 09.9-25-02).

`generate_humo_clip()` is the single entrypoint the plan-01 router calls for every
singer / talking-head segment. It converts the reference portrait to 16:9, builds
the VALIDATED VRGDG V9 HuMo 14B workflow (848x480), generates through ComfyUI,
upscales to 1920x1080 via Real-ESRGAN scale-to-fit (no crop → no zoom), and applies
the run-17 frame-0 contrast fix (copy frame 6 → positions 0-5).

CRITICAL — dispatch model: this module runs INSIDE a `/slingshot/exec` command
(CWD=gpu-manager), so it calls ComfyUI DIRECTLY here — it does NOT recursively
invoke slingshot/exec. The router POSTs to /slingshot/exec with the absolute path
of this script + TQDM_DISABLE=1. Torch / PIL / torchvision live in the ComfyUI venv,
so those imports are DEFERRED into the functions that need them (this file must
stay importable under the alice `uv` env for structural checks).

Generic + reusable (D-06): every song/portrait input is a parameter; no hardcoding.
STYLE.md: ≤300 LOC, keyword-only public params, full annotations, named constants,
no bare except / silent swallow.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import sysconfig
from pathlib import Path

# ── Environment setup BEFORE any GPU/CUDA import (gotcha, bug 260707) ────────
_platlib = sysconfig.get_path("platlib")
os.environ["LD_LIBRARY_PATH"] = (
    ":".join([
        str(Path(_platlib) / "nvidia" / "cublas" / "lib"),
        str(Path(_platlib) / "nvidia" / "cudnn" / "lib"),
        str(Path(_platlib) / "nvidia" / "cublasLt" / "lib"),
    ])
    + ":"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["TQDM_DISABLE"] = "1"  # tqdm+async stdout crashes gpu-manager NSFW classifier

_REPO_ROOT = Path(os.environ.get("MV_PROJECT_DIR", "/path/to/project"))
sys.path.insert(0, str(_REPO_ROOT / "gpu-manager"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("mv_humo_gen")

# Post-processing stages (Real-ESRGAN upscale + frame-0 fix) live in a sibling
# module (STYLE single-responsibility). Re-export their frame-0 constants so the
# plan-01 contract + acceptance greps can reference them from this module.
from mv_humo_postprocess import (  # noqa: E402
    FRAME0_FIX_COPY_COUNT,
    HUMO_FIRST_FRAME_FIX_SRC_INDEX,
    HUMO_TARGET_H,
    HUMO_TARGET_W,
    apply_frame0_fix,
    composite_bg_preserving,
    upscale_to_1080_realesrgan_fit,
)

# ── Named constants (STYLE.md: no magic numbers/strings) ────────────────────
HUMO_DEFAULT_STEPS = 4
HUMO_DEFAULT_CFG = 1.0
HUMO_DEFAULT_SHIFT = 10.0  # VRGDG V9 internal default (disconnected primitive showed 8.0)
HUMO_DEFAULT_AUDIO_SCALE = 1.5  # VRGDG V9 value (lower = better visual fidelity)
HUMO_PROD_WIDTH = 848
HUMO_PROD_HEIGHT = 480
HUMO_FPS = 25
HUMO_SEED = 42
HUMO_DEFAULT_DURATION_S = 16

HUMO_GENERATION_TIMEOUT_S = 1800  # 30 min
HUMO_COMFYUI_READY_TIMEOUT_S = 180
HUMO_REMBG_PADDING_COLOR = "white"

# ── S5 bg-preserving composite (approach b) ──────────────────────────────────
# Studio background plate used by composite_bg_preserving (postprocess). This is
# the source reference (full studio scene, RemBG-free) — the composite mats the
# HuMo subject and places it over this plate so the studio background is
# reconstructed after generation (HuMo cannot paint it, E7). Song-specific; the
# modotte-oide fixture's reference image. Callers may override via bg_plate_path.
HUMO_BG_PLATE_DEFAULT: Path | None = None  # Disabled — bg-composite rejected (YC: "ghosting worse")

# ── S5 studio-background directive (VRGDG-mirroring fix) ───────────────────────
# VRGDG's HuMo renders backgrounds PROMPT-DRIVEN: a "Background and location"
# context string feeds an LLM that generates rich scene prompts (e.g. "standing
# at the center of a circular stone chamber, candles line the perimeter").
# Their reference bg is ALSO stripped by RemBG before conditioning — so the bg
# never comes from the reference image; it comes from the positive prompt. Our
# pipeline hardcoded the positive prompt to "a person speaking clearly" with no
# scene → HuMo painted a default plain bg. This default description is appended
# to the HuMo positive prompt so the studio background is rendered (S5). It can
# be overridden per-call via ``scene_context`` so the studio scene is the
# default and existing behavior does not silently regress.
HUMO_STUDIO_SCENE_CONTEXT = ""  # empty — let reference image carry the scene (RemBG-off run)
# The studio scene_context is appended to the HuMo positive prompt unless it is
# already a substring of the incoming prompt (double-append guard, see
# _compose_humo_prompt). To suppress it entirely, call with scene_context="".

_ASPECT_W = 16
_ASPECT_H = 9


def _comfyui_input_dir() -> Path:
    """Return ComfyUI's ``input/`` directory (sibling of the output dir)."""
    import mv_comfyui  # deferred: pulls in the ComfyUI client layer

    return Path(mv_comfyui.COMFYUI_OUTPUT_DIR) / "input"


def _convert_reference_to_16x9(reference_path: Path, output_dir: Path) -> Path:
    """Center-crop the reference portrait to a 16:9 PNG (D-04, generic — D-06).

    HuMo's ResizeAndPadImage white-pads a mismatched aspect ratio → facial drift +
    letterbox bars; a ref already matching the 16:9 output aspect eliminates both.
    No specific portrait is hardcoded — the crop is centered on the image center.
    """
    from PIL import Image  # deferred: Pillow lives in the ComfyUI venv

    if not reference_path.exists():
        raise FileNotFoundError(f"reference portrait not found: {reference_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(reference_path) as raw:
        img = raw.convert("RGB")
        src_w, src_h = img.size
        target_ratio = _ASPECT_W / _ASPECT_H
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            # Too wide → crop width, keep height (center crop).
            new_w = int(round(src_h * target_ratio))
            left = (src_w - new_w) // 2
            box = (left, 0, left + new_w, src_h)
        else:
            # Too tall (or equal) → crop height, keep width (center crop).
            new_h = int(round(src_w / target_ratio))
            top = (src_h - new_h) // 2
            box = (0, top, src_w, top + new_h)
        cropped = img.crop(box)

    out_path = output_dir / f"{reference_path.stem}_16x9.png"
    cropped.save(out_path)
    logger.debug("reference converted to 16:9: %s (%dx%d)", out_path, *cropped.size)
    return out_path


def _build_workflow(*, comfy_ref_name: str, comfy_audio_name: str, prompt: str,
                    duration_s: int, seed: int,
                    scene_context: str = HUMO_STUDIO_SCENE_CONTEXT) -> dict:
    """Build the validated HuMo workflow and inject the caller's positive prompt.

    The builder's ``positive_prompt`` is otherwise hardcoded; we mutate it after
    build (minimal-change approach — the builder body stays unchanged). When the
    supplied ``prompt`` lacks explicit background direction, ``scene_context`` is
    appended so HuMo renders a studio background (S5 / VRGDG-mirroring fix).
    """
    from workflows.workflow_humo import (  # deferred import (gpu-manager on sys.path)
        HUMO_NODE_IDS,
        build_humo_talking_head_workflow,
        humo_production_overrides,
    )

    positive_prompt = _compose_humo_prompt(prompt, scene_context)
    overrides = humo_production_overrides(duration_s=duration_s, seed=seed)
    wf = build_humo_talking_head_workflow(
        portrait_path=comfy_ref_name,
        audio_path=comfy_audio_name,
        **overrides,
    )
    wf[HUMO_NODE_IDS["text_encode"]]["inputs"]["positive_prompt"] = positive_prompt
    return wf


def _compose_humo_prompt(prompt: str, scene_context: str) -> str:
    """Compose the HuMo positive prompt, appending the studio scene context (S5 fix).

    Mirrors VRGDG: background comes from the positive prompt, not the reference
    (which is RemBG-stripped before conditioning). ``scene_context`` is the default
    studio background; pass ``scene_context=""`` to suppress it (e.g. a caller
    supplying its own scene in ``prompt``). No auto-detection guard — a bare
    "studio" mention must NOT suppress the scene, and no caller passes the full
    constant as its own prompt, so a double-append cannot occur in practice.
    """
    if not scene_context:
        return prompt.rstrip(". ")
    return f"{prompt.rstrip('. ')}. {scene_context}"


def _run_generation(*, ref_16x9: Path, audio_segment_wav: Path, prompt: str,
                    duration_s: int, seed: int,
                    scene_context: str = HUMO_STUDIO_SCENE_CONTEXT) -> Path:
    """Stage the inputs into ComfyUI ``input/``, generate, return the base MP4."""
    import mv_comfyui  # deferred

    input_dir = _comfyui_input_dir()
    input_dir.mkdir(parents=True, exist_ok=True)
    comfy_ref = input_dir / ref_16x9.name
    comfy_audio = input_dir / audio_segment_wav.name
    # Defensive: skip the copy when the source already lives in ComfyUI's
    # input/ dir (copy-onto-self raises shutil.SameFileError). Pure no-op
    # when redundant — no other behavior changed.
    if ref_16x9.resolve() != comfy_ref.resolve():
        shutil.copy2(ref_16x9, comfy_ref)
    if audio_segment_wav.resolve() != comfy_audio.resolve():
        shutil.copy2(audio_segment_wav, comfy_audio)

    if not mv_comfyui._comfyui_is_ready():
        logger.info("starting ComfyUI via gpu-manager...")
        mv_comfyui._start_comfyui_via_gpu_manager()
        if not mv_comfyui._wait_for_comfyui_ready(timeout=HUMO_COMFYUI_READY_TIMEOUT_S):
            raise RuntimeError("ComfyUI failed to become ready")
    logger.info("ComfyUI ready")

    prefix = f"humo_clip_{ref_16x9.stem}"
    wf = _build_workflow(
        comfy_ref_name=comfy_ref.name,
        comfy_audio_name=comfy_audio.name,
        prompt=prompt,
        duration_s=duration_s,
        seed=seed,
        scene_context=scene_context,
    )
    wf_video_node = wf[_video_combine_id(wf)]
    wf_video_node["inputs"]["filename_prefix"] = prefix

    prompt_id = mv_comfyui._queue_workflow(wf)
    logger.info("queued prompt_id=%s (timeout %ds)", prompt_id, HUMO_GENERATION_TIMEOUT_S)
    history = mv_comfyui._poll_completion(prompt_id, timeout=HUMO_GENERATION_TIMEOUT_S)
    return mv_comfyui._find_output_file(history, prefix, ".mp4")


def _video_combine_id(wf: dict) -> str:
    """Return the VHS_VideoCombine node id from a built workflow."""
    for node_id, node in wf.items():
        if node.get("class_type") == "VHS_VideoCombine":
            return node_id
    raise RuntimeError("no VHS_VideoCombine node in workflow")


def _stage(clip_index: int, name: str, fn, *args, **kwargs):  # noqa: ANN001,ANN002,ANN003,ANN201
    """Run a pipeline stage; on failure log context (clip + stage) and re-raise.

    Never swallows — the exception propagates so the CLI exits non-zero (no silent
    proceed). ``fn`` may take positional or keyword args (both forwarded).
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.error("clip %d: %s stage failed", clip_index, name)
        raise


def generate_humo_clip(
    audio_segment_wav: Path,
    reference_path: Path,
    prompt: str,
    output_dir: Path,
    clip_index: int,
    *,
    duration_s: int = HUMO_DEFAULT_DURATION_S,
    width: int = HUMO_PROD_WIDTH,
    height: int = HUMO_PROD_HEIGHT,
    seed: int = HUMO_SEED,
    scene_context: str = HUMO_STUDIO_SCENE_CONTEXT,
    skip_upscale: bool = False,
    bg_plate_path: "Path | None" = None,
) -> Path:
    """Generate one HuMo 14B talking-head clip.

    Pipeline: convert ref → 16:9 (D-04) → build validated VRGDG V9 workflow (848x480)
    → generate via ComfyUI (runs inside slingshot/exec) → Real-ESRGAN fit-upscale to
    1080p (no crop) → copy-frame-6 frame-0 fix. All inputs are parameters (D-06).
    ``clip_index`` is logging/provenance only; ``duration_s`` defaults to 16s (18s
    VRAM ceiling).

    When ``skip_upscale`` is True the Real-ESRGAN upscale AND the frame-0 fix are
    skipped and the raw 848x480 base clip is returned (debug/verification runs that
    only need to inspect the generated content, not the final upscaled masters).
    ``bg_plate_path`` overrides the studio background plate used by the S5
    bg-preserving composite; None → :data:`HUMO_BG_PLATE_DEFAULT`. The composite
    always runs (it is the S5 fix), independent of ``skip_upscale``.
    Returns the final MP4 in ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not audio_segment_wav.exists():
        raise FileNotFoundError(f"audio segment not found: {audio_segment_wav}")
    if not reference_path.exists():
        raise FileNotFoundError(f"reference portrait not found: {reference_path}")

    logger.info("clip %d: ref=%s audio=%s %dx%d %ds seed=%d", clip_index,
                reference_path.name, audio_segment_wav.name, width, height,
                duration_s, seed)
    ref_16x9 = _stage(clip_index, "reference→16:9", _convert_reference_to_16x9,
                      reference_path, output_dir)
    base_mp4 = _stage(clip_index, "generation", _run_generation,
                      ref_16x9=ref_16x9, audio_segment_wav=audio_segment_wav,
                      prompt=prompt, duration_s=duration_s, seed=seed,
                      scene_context=scene_context)
    # S5 bg-preserving composite: matte the HuMo subject and composite over the
    # studio plate (approach b). Runs on the 848x480 base, before upscale.
    # GATED — composite is disabled by default (YC rejected: "ghosting worse").
    # Enable by passing an explicit bg_plate_path.
    if bg_plate_path is not None:
        composited = _stage(clip_index, "bg-composite", composite_bg_preserving,
                            base_mp4, bg_plate_path)
    elif HUMO_BG_PLATE_DEFAULT is not None:
        composited = _stage(clip_index, "bg-composite", composite_bg_preserving,
                            base_mp4, HUMO_BG_PLATE_DEFAULT)
    else:
        composited = base_mp4
        logger.info("clip %d: bg-composite skipped (no bg plate configured)", clip_index)
    if skip_upscale:
        logger.info("clip %d: upscale + frame-0 fix skipped (skip_upscale=True)",
                    clip_index)
        final = composited
    else:
        upscaled = _stage(clip_index, "upscale", upscale_to_1080_realesrgan_fit, composited)
        final = _stage(clip_index, "frame-0 fix", apply_frame0_fix, upscaled)

    dest = output_dir / final.name
    if final.resolve() != dest.resolve():
        shutil.copy2(final, dest)
        final = dest
    logger.info("clip %d complete → %s", clip_index, final)
    return final


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one HuMo 14B talking-head clip")
    parser.add_argument("--audio-segment", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--clip-index", required=True, type=int)
    parser.add_argument("--duration-s", type=int, default=HUMO_DEFAULT_DURATION_S)
    parser.add_argument("--seed", type=int, default=HUMO_SEED)
    parser.add_argument("--scene-context", type=str, default=HUMO_STUDIO_SCENE_CONTEXT)
    parser.add_argument("--skip-upscale", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """CLI entrypoint (invoked inside slingshot/exec)."""
    from mv_slingshot import SlingshotClient

    args = _parse_args(argv)
    slingshot = SlingshotClient()
    final_path = ""
    try:
        out = generate_humo_clip(
            args.audio_segment, args.reference, args.prompt, args.out_dir,
            args.clip_index,
            duration_s=args.duration_s,
            seed=args.seed,
            scene_context=args.scene_context,
            skip_upscale=args.skip_upscale,
        )
        final_path = str(out)
        print(final_path)  # noqa: T201 — surface path to the exec caller
        return 0
    except Exception as e:
        logger.error("generate_humo_clip failed: %s", e, exc_info=True)
        return 1
    finally:
        slingshot.ensure_wake(task_name="humo_clip", output_path=final_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
