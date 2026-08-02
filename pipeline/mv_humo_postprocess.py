#!/usr/bin/env python3
"""HuMo clip post-processing: Real-ESRGAN fit-upscale + frame-0 contrast fix.

Split out of ``mv_humo_gen`` (STYLE.md single-responsibility): this module owns the
two post-generation stages so the generator stays focused on producing the base clip.
Both stages reuse VALIDATED logic (benchmark upscale + run-17 frame-0 fix).

Torch / PIL / torchvision imports are DEFERRED into the functions (they live in the
ComfyUI venv, not the alice ``uv`` env), so this file imports cleanly for structural
checks.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("mv_humo_postprocess")

# ── Named constants (STYLE.md: no magic numbers/strings) ────────────────────
HUMO_FPS = 25
HUMO_TARGET_W = 1920
HUMO_TARGET_H = 1080

# Frame-0 contrast fix (VALIDATED run-17): copy frame 6 to positions 0-5.
HUMO_FIRST_FRAME_FIX_SRC_INDEX = 6
FRAME0_FIX_COPY_COUNT = 6
HUMO_FRAME0_FIX_MIN_FRAMES = 7  # need >= src_index+1 frames to apply the fix

# F2 (S4 softness, APPROVED): prefer 4x-UltraSharp — a sharper
# Real-ESRGAN-family RRDBNet(x4) model — over the general-purpose RealESRGAN_x4plus.
# 4x-UltraSharp shares the same RRDBNet arch (nf=64, nb=23, gc=32, scale=4) so the
# existing loader works unchanged. UltraSharp must be supplied by the operator
# (download into ComfyUI/models/upscale_models/); until present we fall back to
# x4plus with a loud warning so the pipeline still runs.
HUMO_REALESRGAN_MODEL_ULTRASHARP = os.environ.get("HUMO_REALESRGAN_MODEL", "/path/to/ComfyUI/models/upscale_models/4x-UltraSharp.pth")
HUMO_REALESRGAN_MODEL_FALLBACK = os.environ.get("HUMO_REALESRGAN_FALLBACK", "/path/to/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth")


def _resolve_realesrgan_model() -> str:
    """Resolve the active Real-ESRGAN model path (F2 UltraSharp swap).

    Returns the UltraSharp path when the asset is present (approved swap),
    otherwise the x4plus fallback with a warning. Keeps the pipeline runnable
    before the operator downloads 4x-UltraSharp.pth.
    """
    if Path(HUMO_REALESRGAN_MODEL_ULTRASHARP).exists():
        logger.info("F2: using 4x-UltraSharp for HuMo upscale (sharper than x4plus)")
        return HUMO_REALESRGAN_MODEL_ULTRASHARP
    logger.warning(
        "F2: 4x-UltraSharp.pth NOT found — falling back to RealESRGAN_x4plus "
        "(softer). Drop 4x-UltraSharp.pth into ComfyUI/models/upscale_models/ to apply."
    )
    return HUMO_REALESRGAN_MODEL_FALLBACK


_FFMPEG_CRF = "18"
_FFMPEG_AUDIO_BITRATE = "192k"

# ── S5 bg-preserving composite (approach b) ──────────────────────────────────
# HuMo can only paint a plain/default background: it uses an identity-only
# HuMoEmbeds lock, and production RemBG strips the reference bg BEFORE
# conditioning, so the studio scene never reaches the model (E7). The prompt
# driven reinforcement (approach a, HUMO_STUDIO_SCENE_CONTEXT) is weak on the
# single-ref path. So we reconstruct the background AFTER generation: matte the
# animated subject via RemBG, then composite it over a real studio BACKGROUND
# PLATE derived from the reference image (which holds the full studio scene).
# Runs on the 848x480 base clip; the downstream upscale then lifts it to 1080p.
HUMO_BG_COMPOSITE_MODEL = "u2net"
HUMO_BG_COMPOSITE_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
HUMO_BG_PLATE_BLUR_RADIUS = 100  # heavy blur: the plate subject dissolves into studio tone
HUMO_BG_SUBJECT_ALPHA_THRESH = 128  # rembg alpha above this = subject
# S5 ghost fix (2026-07-18, REVISED): the plate's subject hole must read as SOFT
# studio, never a person-shaped grey ghost. The first attempt (dilate-plate-hole
# 24px + dilate-subject 6px) was wrong in direction — growing the hole makes it
# HARDER to cover, and 6px can't span the upper-right offset observed. Correct:
#   * feather the hole edge (no hard outline) + fill it with a HEAVY blur so the
#     person dissolves into surrounding studio tone — an uncovered rim shows soft
#     studio light, not a blurred person;
#   * grow the per-frame HuMo subject alpha generously (covers the plate hole + any
#     positional offset between the generated subject and the reference plate) and
#     feather its edge so it sits on studio, never on a grey ghost.
HUMO_BG_PLATE_MASK_DILATE = 8   # px — keep hole >= plate subject (no sharp person leak)
HUMO_BG_PLATE_MASK_FEATHER = 8  # px — soften the hole edge
HUMO_SUBJECT_MASK_DILATE = 18   # px — grow HuMo subject to cover the plate hole (+ offset)
HUMO_SUBJECT_MASK_FEATHER = 6   # px — feather the composite edge


def _load_realesrgan(model_path: str):  # noqa: ANN201 — returns a torch.nn.Module
    """Load a Real-ESRGAN-family RRDBNet(x4) model onto CUDA in half precision.

    Works for both 4x-UltraSharp and RealESRGAN_x4plus (same arch).
    """
    from realesrgan_arch import RRDBNet  # local arch module (torch deferred)
    import torch

    net = RRDBNet(in_ch=3, out_ch=3, nf=64, nb=23, gc=32, scale=4)
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd["params_ema"])
    return net.eval().half().cuda()


def upscale_to_1080_realesrgan_fit(base_mp4: Path) -> Path:
    """Real-ESRGAN 4x per-frame → ffmpeg fit+pad to 1920x1080 (no stretch, no crop-zoom).

    Uses ``force_original_aspect_ratio=decrease`` + ``pad`` to letterbox-fit the
    upscaled frames into 1920x1080 without anisotropic stretch (the old
    ``scale=W:H`` with two ints was a hard stretch). ``setsar=1`` fixes anamorphic
    SAR. Reuses the validated benchmark path.
    """
    import torch
    from PIL import Image
    from torchvision import transforms  # type: ignore

    net = _load_realesrgan(_resolve_realesrgan_model())
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    tmp_dir = Path(tempfile.mkdtemp(prefix="humo_realesrgan_"))
    out_path = base_mp4.parent / f"{base_mp4.stem}_1080.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(base_mp4), "-pix_fmt", "rgb24",
             str(tmp_dir / "%04d.png")],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(base_mp4), "-vn", "-c:a", "aac",
             "-b:a", _FFMPEG_AUDIO_BITRATE, str(tmp_dir / "audio.aac")],
            check=True, capture_output=True,
        )
        frames = sorted(tmp_dir.glob("[0-9]*.png"))
        logger.info("upscaling %d frames via Real-ESRGAN...", len(frames))
        up_dir = tmp_dir / "up"
        up_dir.mkdir()
        for i, fp in enumerate(frames):
            img = to_tensor(Image.open(fp).convert("RGB")).half().unsqueeze(0).cuda()
            with torch.no_grad():
                out = torch.clamp(net(img), 0, 1)
            to_pil(out[0].cpu().float()).save(up_dir / f"{i:04d}.png")

        subprocess.run(
            ["ffmpeg", "-y", "-r", str(HUMO_FPS), "-i", str(up_dir / "%04d.png"),
             "-i", str(tmp_dir / "audio.aac"),
             "-vf", f"scale={HUMO_TARGET_W}:{HUMO_TARGET_H}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={HUMO_TARGET_W}:{HUMO_TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"setsar=1",  # true fit+pad (no stretch); setsar=1 fixes anamorphic SAR
             "-c:v", "libx264", "-crf", _FFMPEG_CRF, "-pix_fmt", "yuv420p",
             "-c:a", "copy", "-movflags", "+faststart", str(out_path)],
            check=True, capture_output=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("upscaled → %s", out_path)
    # Release the RealESRGAN model + CUDA reserved pool so the next clip's
    # ComfyUI WanVideoSampler generation is not starved of VRAM (09.9-25-05
    # clip-2 OOM: a competing ~17 GiB process left ComfyUI only ~30 GiB).
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return out_path


def apply_frame0_fix(upscaled_mp4: Path) -> Path:
    """Copy frame 6 → positions 0-5 to fix first-frame contrast (VALIDATED run-17).

    Guard: if fewer than HUMO_FRAME0_FIX_MIN_FRAMES frames, logs a warning and returns
    the input unchanged (no silent crash). Preserves the original audio.
    """
    out_path = upscaled_mp4.parent / f"{upscaled_mp4.stem}_frame0fix.mp4"
    tmp_dir = Path(tempfile.mkdtemp(prefix="humo_frame0fix_"))
    try:
        frames_dir = tmp_dir / "frames"
        frames_dir.mkdir()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(upscaled_mp4), "-q:v", "1",
             str(frames_dir / "%05d.png")],
            check=True, capture_output=True,
        )
        frame_files = sorted(frames_dir.glob("*.png"))
        if len(frame_files) < HUMO_FRAME0_FIX_MIN_FRAMES:
            logger.warning("frame-0 fix skipped: got %d frames, need >= %d",
                           len(frame_files), HUMO_FRAME0_FIX_MIN_FRAMES)
            return upscaled_mp4
        src_frame = frame_files[HUMO_FIRST_FRAME_FIX_SRC_INDEX]
        for i in range(FRAME0_FIX_COPY_COUNT):
            shutil.copy2(src_frame, frame_files[i])
        logger.info("copied frame %d → positions 0-%d",
                    HUMO_FIRST_FRAME_FIX_SRC_INDEX, FRAME0_FIX_COPY_COUNT - 1)

        audio_out = tmp_dir / "audio.aac"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(upscaled_mp4), "-vn", "-c:a", "aac",
             "-b:a", _FFMPEG_AUDIO_BITRATE, str(audio_out)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(HUMO_FPS),
             "-i", str(frames_dir / "%05d.png"), "-i", str(audio_out),
             "-c:v", "libx264", "-crf", _FFMPEG_CRF, "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", _FFMPEG_AUDIO_BITRATE, "-shortest",
             str(out_path)],
            check=True, capture_output=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("frame-0 fix applied → %s", out_path)
    return out_path


def _cover_resize(img: "Image.Image", size: tuple[int, int]) -> "Image.Image":
    """Resize ``img`` to ``size`` covering it (no empty bars), center-cropping overflow."""
    from PIL import Image

    tw, th = size
    sw, sh = img.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _build_bg_plate(plate_path: Path, frame_size: tuple[int, int]) -> "Image.Image":
    """Build a studio background plate (RGB) at ``frame_size`` from the reference.

    The reference holds the full studio scene BUT also the subject. We matte the
    subject out via RemBG and replace that hole with a heavily-blurred copy of
    the plate (soft studio colour); everywhere else the sharp studio background
    is kept. The result is cover-scaled to fill the frame (no black bars).
    """
    from PIL import Image, ImageFilter
    from rembg import new_session, remove

    plate = Image.open(plate_path).convert("RGB")
    sess = new_session(HUMO_BG_COMPOSITE_MODEL, providers=list(HUMO_BG_COMPOSITE_PROVIDERS))
    plate_rgba = remove(plate, session=sess)  # subject opaque (alpha=255)
    subj_mask = plate_rgba.split()[3].point(
        lambda a: 255 if a > HUMO_BG_SUBJECT_ALPHA_THRESH else 0
    )
    # Grow the hole slightly so it always fully covers the plate subject (avoids a
    # sharp person-edge leaking through as background), then FEATHER the edge so the
    # hole-to-studio boundary is soft (no hard grey outline — the S5 ghost).
    subj_mask = subj_mask.filter(
        ImageFilter.MaxFilter(HUMO_BG_PLATE_MASK_DILATE * 2 + 1)
    )
    subj_mask = subj_mask.filter(ImageFilter.GaussianBlur(HUMO_BG_PLATE_MASK_FEATHER))
    # HEAVY blur fill: the subject dissolves into the surrounding studio tone, so
    # even where the HuMo subject fails to cover the hole the rim reads as soft
    # studio light — never a blurred person.
    blur = plate.filter(ImageFilter.GaussianBlur(HUMO_BG_PLATE_BLUR_RADIUS))
    # where subj_mask is set → blurred fill (hole); else → sharp plate (studio bg)
    bg = Image.composite(blur, plate, subj_mask)
    return _cover_resize(bg, frame_size).convert("RGB")


def composite_bg_preserving(base_mp4: Path, bg_plate_path: Path) -> Path:
    """Matte the HuMo subject and composite it over the studio background plate (S5).

    Per-frame: extract frames, RemBG-matte each → subject RGBA, alpha-composite
    over the prebuilt background plate, re-encode with the original audio. The
    ``base_mp4`` is the 848x480 HuMo output; the downstream upscale then lifts
    the composited clip to 1080p. Returns the composited clip path.

    If ``bg_plate_path`` is missing, logs a warning and returns ``base_mp4``
    unchanged (degrades gracefully rather than crashing the pipeline).
    """
    base_mp4 = Path(base_mp4)
    bg_plate_path = Path(bg_plate_path)
    if not bg_plate_path.exists():
        logger.warning("bg plate missing (%s) — skipping composite", bg_plate_path)
        return base_mp4

    tmp_dir = Path(tempfile.mkdtemp(prefix="humo_bgcomp_"))
    out_path = base_mp4.parent / f"{base_mp4.stem}_bgcomp.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(base_mp4), "-pix_fmt", "rgb24",
             str(tmp_dir / "%04d.png")],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(base_mp4), "-vn", "-c:a", "aac",
             "-b:a", _FFMPEG_AUDIO_BITRATE, str(tmp_dir / "audio.aac")],
            check=True, capture_output=True,
        )
        frames = sorted(tmp_dir.glob("[0-9]*.png"))
        if not frames:
            logger.warning("no frames extracted — skipping composite")
            return base_mp4

        from PIL import Image, ImageFilter
        from rembg import new_session, remove

        frame_size = Image.open(frames[0]).size
        bg = _build_bg_plate(bg_plate_path, frame_size).convert("RGBA")
        sess = new_session(HUMO_BG_COMPOSITE_MODEL, providers=list(HUMO_BG_COMPOSITE_PROVIDERS))

        out_dir = tmp_dir / "comp"
        out_dir.mkdir()
        logger.info("compositing %d frames over studio plate...", len(frames))
        for i, fp in enumerate(frames):
            f = Image.open(fp).convert("RGB")
            subj = remove(f, session=sess)  # RGBA, subject opaque
            # S5 ghost fix (2026-07-18): dilate the subject alpha a few px and
            # feather it so the HuMo subject ALWAYS fully over-covers the plate's
            # (slightly larger) subject hole, and edges feather instead of ghost.
            a = subj.split()[3]
            a = a.filter(ImageFilter.MaxFilter(HUMO_SUBJECT_MASK_DILATE * 2 + 1))
            a = a.filter(ImageFilter.GaussianBlur(HUMO_SUBJECT_MASK_FEATHER))
            subj.putalpha(a)
            Image.alpha_composite(bg, subj).convert("RGB").save(out_dir / f"{i:04d}.png")

        subprocess.run(
            ["ffmpeg", "-y", "-r", str(HUMO_FPS), "-i", str(out_dir / "%04d.png"),
             "-i", str(tmp_dir / "audio.aac"),
             "-vf", f"scale={frame_size[0]}:{frame_size[1]},setsar=1",
             "-c:v", "libx264", "-crf", _FFMPEG_CRF, "-pix_fmt", "yuv420p",
             "-c:a", "copy", "-movflags", "+faststart", str(out_path)],
            check=True, capture_output=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("bg-preserving composite → %s", out_path)
    gc.collect()
    return out_path
