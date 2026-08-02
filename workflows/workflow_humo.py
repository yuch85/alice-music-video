"""ComfyUI workflow builder for a HUMO (Wan2_1-HuMo-14B) talking-head clip.

Plan 09.9-20 (Wave 2) — STYLE-compliant, single-responsibility builder that
emits a ComfyUI API-format workflow following the VRGDG V9 single-clip node
order (verified against
``ComfyUI-WanVideoWrapper/example_workflows/wanvideo_2_1_14B_HuMo_example_01.json``).

Identity lock = the reference portrait is fed into ``HuMoEmbeds`` as
``reference_images`` (with ``vae``). ``WANVIDIMAGE_EMBEDS`` is the RETURN TYPE
of ``HuMoEmbeds`` — NOT a node — so the sampler consumes the embeds directly
via ``image_embeds``. No separate ``WANVIDIMAGE_EMBEDS`` node is created.

This module imports no ComfyUI runtime; it only builds a plain dict, so it is
importable under ``uv`` without a live ComfyUI (the test exercises it directly).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Named constants (STYLE.md: no bare magic strings) ──────────────────────
# Re-export for backwards compatibility; roster lives in humo_models.py.
from workflows.humo_models import HUMO_MODEL_FILENAME, HUMO_MODEL_FILENAME_1_7B, HUMO_MODELS  # noqa: F401

WAN2_1_VAE_BF16 = "Wan2_1_VAE_bf16.safetensors"
WHISPER_FILENAME = "whisper_large_v3_encoder_fp16.safetensors"
UMT5_FILENAME = "umt5-xxl-enc-bf16.safetensors"
MEL_BAND_ROFORMER_FILENAME = "MelBandRoFormer/MelBandRoformer_fp16.safetensors"

HUMO_FRAME_RATE = 24
HUMO_DEFAULT_WIDTH = 832
HUMO_DEFAULT_HEIGHT = 480
# NOTE: production 16:9 generation uses 848x480 (see HUMO_PROD_* below); the
# existing 832x480 defaults are preserved because prior callers may rely on them.

# ── VALIDATED VRGDG V9 production config (Plan 09.9-25) ─────────────────────
# Single source of truth for the production HuMo 14B config. Runs 7-16 produced
# blurred/warped output from config drift; runs 8 + 17 locked this exact set.
# The wrapper (scripts/mv_humo_gen.py) consumes these via humo_production_overrides()
# so the validated config cannot silently drift.
HUMO_PROD_WIDTH = 848
"""Production 16:9 width (divisible by 8; 854 FAILS HuMoEmbeds per spike-001)."""
HUMO_PROD_HEIGHT = 480
"""Production 16:9 height (848x480 ≈ 16:9, both divisible by the VAE stride 8)."""
HUMO_FPS = 25
"""Production FPS (validated run-8/run-17). Note: builder default HUMO_FRAME_RATE=24."""

HUMO_VALIDATED_STEPS = 4
"""Sampler steps (YC reverted 8→4; run-17 confirmed 4 is the final value)."""
HUMO_VALIDATED_CFG = 1.0
"""Classifier-free guidance scale for the distilled LightX LoRA path (CFG=1.0)."""
HUMO_VALIDATED_SHIFT = 10.0
"""Sampler timestep shift (VRGDG V9 internal default; disconnected primitive showed 8.0)."""

HUMO_VALIDATED_AUDIO_SCALE = 1.5
"""HuMoEmbeds audio-conditioning strength (VRGDG V9 primitive override = 1.5; lower = better visual fidelity)."""
HUMO_VALIDATED_AUDIO_CFG_SCALE = 1.0
"""HuMoEmbeds audio CFG scale (VRGDG V9 = 1.0; no extra unconditional-audio pass)."""

HUMO_VALIDATED_SAMPLER = "dpm++_sde/beta"
"""Validated sampler/scheduler string. NEVER Kijai's lcm (run-7: blurred, warped)."""
HUMO_VALIDATED_SCHEDULER = "comfy"
"""Validated scheduler family label (documentary — the builder's `scheduler` kwarg
takes the sampler string HUMO_VALIDATED_SAMPLER; this records the comfy family)."""

HUMO_VALIDATED_BASE_PRECISION = "fp16_fast"
"""Model base precision (VRGDG V9 = fp16_fast on the 48GB card)."""
HUMO_VALIDATED_ATTENTION = "sageattn"
"""Attention implementation (VRGDG V9 = sageattn)."""

HUMO_DISTILL_LORA_NAME = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
"""LightX I2V step-distillation LoRA — REQUIRED for steps=4 (missing it → blur)."""
HUMO_DISTILL_LORA_STRENGTH = 1.0
"""LightX distillation LoRA strength (VRGDG V9 = 1.0)."""

HUMO_FASTWAN_LORA_NAME = "FastWan/FastWan_T2V_14B_480p_lora_rank_64_bf16.safetensors"
"""FastWan T2V acceleration LoRA — VRGDG V9 chain position 1 (base)."""
HUMO_FASTWAN_LORA_STRENGTH = 1.0
"""FastWan LoRA strength (VRGDG V9 = 1.0)."""

HUMO_FACE_DETAILER_LORA_NAME = "FaceDetailerV1.safetensors"
"""FaceDetailerV1 — VRGDG V9 chain position 2 (face sharpness). Custom VRGDG LoRA."""
HUMO_FACE_DETAILER_LORA_STRENGTH = 1.0
"""FaceDetailerV1 strength (VRGDG V9 = 1.0)."""

HUMO_DEFAULT_DURATION_S = 16
"""Default clip duration (D-01); within the 18s VRAM ceiling."""
HUMO_SAFETY_DURATION_S = 8
"""Internal safety fallback duration (drift-prone but VRAM-safe)."""

HUMO_SEED = 42
"""Deterministic production seed (VRGDG V9)."""

# Model-loader widget values (mirror the working example workflow).
HUMO_BASE_PRECISION = "bf16"
HUMO_QUANTIZATION = "disabled"
# main_device: load + merge LoRA directly on GPU. This 48GB card fits the
# 14B model + LoRA; offload_device forces a CPU LoRA-merge which segfaults in
# ComfyUI-WanVideoWrapper utils.py:apply_lora. Run-17 (LoRA, GPU) confirms this.
HUMO_LOAD_DEVICE = "main_device"
VAE_PRECISION = "bf16"
T5_PRECISION = "bf16"
WHISPER_BASE_PRECISION = "fp16"
WHISPER_LOAD_DEVICE = "main_device"

# HuMoEmbeds audio-conditioning defaults — match VRGDG V9 reference.
# VRGDG V9: audio_scale=1.5 (primitive override), audio_cfg_scale=1.0
# (no extra unconditional-audio pass). Kijai used audio_scale=1.0/2.5 which
# was inverted and caused identity drift in runs 1-5.
HUMO_AUDIO_SCALE = 1.5
HUMO_AUDIO_CFG_SCALE = 1.0
HUMO_AUDIO_START_PERCENT = 0.0
HUMO_AUDIO_END_PERCENT = 1.0

# Portrait preprocessing — resize reference image to output dimensions with
# padding before HuMoEmbeds (match VRGDG V9: RemBG → resize 720x480 pad white).
# VRGDG V9 uses PrimitiveInt width=720, height=480 fed into ImageResizeKJv2 —
# the SAME dimensions as the video output. This eliminates aspect ratio mismatch
# between reference embeds and generated canvas. See Run-9 post-mortem (2026-07-15).
# No longer hardcodes 512x512 — width/height are passed as kwargs to match output.
HUMO_PORTRAIT_RESIZE_INTERPOLATION = "lanczos"
HUMO_PORTRAIT_RESIZE_PADDING_COLOR = "white"

# RemBG (background removal) — VRGDG V9 removes background from reference
# image to isolate the person for stronger identity conditioning.
HUMO_REMBG_MODEL = "u2net: general purpose"
HUMO_REMBG_DEVICE = "CUDA"

# WanVideoSampler defaults (neutral, deterministic when seed supplied).
HUMO_SAMPLER_STEPS = 30
HUMO_SAMPLER_CFG = 6.0
HUMO_SAMPLER_SHIFT = 5.0
HUMO_SAMPLER_SCHEDULER = "unipc"
HUMO_SAMPLER_FORCE_OFFLOAD = True  # move model to offload device after sampling
HUMO_SAMPLER_RIFLEX_FREQ_INDEX = 0  # RIFLEX disabled at 0 (required input in WanVideoWrapper)

# WanVideoDecode tiling defaults (required inputs; mirror the verified example workflow).
HUMO_DECODE_ENABLE_VAE_TILING = True  # reduces VRAM for the 14B decode
HUMO_DECODE_TILE_X = 272
HUMO_DECODE_TILE_Y = 272
HUMO_DECODE_TILE_STRIDE_X = 144
HUMO_DECODE_TILE_STRIDE_Y = 128

# VHS_VideoCombine output settings.
HUMO_OUTPUT_PREFIX = "alice_humo"
HUMO_OUTPUT_FORMAT = "video/h264-mp4"
HUMO_OUTPUT_PIX_FMT = "yuv420p"
HUMO_OUTPUT_CRF = 19
HUMO_OUTPUT_LOOP_COUNT = 0  # required VHS input; 0 = no loop
HUMO_OUTPUT_PINGPONG = False  # required VHS input

# Role -> node id (single linear chain, VRGDG V9 order).
# Portrait pipeline: load_image → [rembg_session → rembg] → resize → [batch]
# Audio pipeline: load_audio → [mel_band_loader → mel_band_sampler → normalize]
HUMO_NODE_IDS = {
    "model_loader": "1",
    "vae_loader": "2",
    "whisper_loader": "3",
    "t5_loader": "4",
    "load_image": "5",
    "rembg_session": "5a",
    "rembg": "5b",
    "portrait_resize": "5c",
    "portrait_batch": "5d",
    "load_audio": "6",
    "mel_band_loader": "6a",
    "mel_band_sampler": "6b",
    "normalize_audio": "6c",
    "text_encode": "7",
    "cfg_schedule": "7b",
    "humo_embeds": "8",
    "sampler": "9",
    "decode": "10",
    "video_combine": "11",
}

# LoRA chain node ids (assigned after core nodes; chained via prev_lora).
# First available LoRA node id.
_HUMO_LORA_NODE_START = 13

# A/B speed-vs-quality presets (09.9-23). Each maps to builder override kwargs.
# Keys are the override names accepted by build_humo_talking_head_workflow.
# Identity lock (ref embeds) + audio conditioning are preserved across presets;
# only sampler steps / resolution / guidance change, so identity + lip-sync hold.
HUMO_PRESETS: dict[str, dict[str, Any]] = {
    "default": {"steps": 30, "width": 832, "height": 480, "cfg": 6.0, "shift": 5.0},
    "fast":    {"steps": 15, "width": 640, "height": 368, "cfg": 6.0, "shift": 5.0},
    "faster":  {"steps": 10, "width": 512, "height": 288, "cfg": 6.0, "shift": 5.0},
    "quality": {"steps": 40, "width": 832, "height": 480, "cfg": 7.0, "shift": 6.0},
}


def humo_preset_overrides(
    preset: str,
    model: str = "14b",
    **overrides: Any,
) -> dict[str, Any]:
    """Resolve builder override kwargs from a named preset, model size, and overrides.

    Explicit keyword overrides (non-None) win over the preset's values. Raises
    KeyError on an unknown preset name or model alias.

    Args:
        preset: Key into ``HUMO_PRESETS``.
        model: Model-size alias (``14b`` or ``1.7b``). Defaults to ``14b``.
        **overrides: Optional override kwargs (steps/width/height/cfg/shift/seed…).

    Returns:
        Merged builder override dict ready to splat into the workflow builder.
        Includes ``model`` key for downstream variant-tag resolution.
    """
    if preset not in HUMO_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(HUMO_PRESETS)}")
    if model not in HUMO_MODELS:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(HUMO_MODELS)}")
    merged: dict[str, Any] = dict(HUMO_PRESETS[preset])
    merged["model"] = model
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def humo_production_overrides(
    *,
    duration_s: float = HUMO_DEFAULT_DURATION_S,
    seed: int = HUMO_SEED,
) -> dict[str, Any]:
    """Return the VALIDATED VRGDG V9 builder kwargs to splat into the builder.

    This is the single source of truth for the production HuMo 14B config so the
    generation wrapper (scripts/mv_humo_gen.py) cannot drift from the config that
    runs 8 + 17 validated. The returned dict is splatted directly into
    ``build_humo_talking_head_workflow(portrait_path=…, audio_path=…, **overrides)``.

    NOTE on the ``scheduler`` key: the builder's ``scheduler`` kwarg takes the full
    sampler string (``dpm++_sde/beta``), so that string is passed here — NOT the
    ``comfy`` family label recorded in ``HUMO_VALIDATED_SCHEDULER``.

    Args:
        duration_s: Clip duration in seconds. Defaults to the 16s production value
            (D-01); bounded by the 18s VRAM ceiling. 8s is the safety fallback.
        seed: Deterministic seed. Defaults to the production seed 42.

    Returns:
        Builder override kwargs (width/height/fps/steps/cfg/shift/scheduler/audio/
        precision/attention/model/seed/rembg/lora_specs/duration_s).
    """
    return {
        "width": HUMO_PROD_WIDTH,
        "height": HUMO_PROD_HEIGHT,
        "fps": HUMO_FPS,
        "steps": HUMO_VALIDATED_STEPS,
        "cfg": HUMO_VALIDATED_CFG,
        "shift": HUMO_VALIDATED_SHIFT,
        "scheduler": HUMO_VALIDATED_SAMPLER,
        "audio_scale": HUMO_VALIDATED_AUDIO_SCALE,
        "audio_cfg_scale": HUMO_VALIDATED_AUDIO_CFG_SCALE,
        "base_precision": HUMO_VALIDATED_BASE_PRECISION,
        "attention_mode": HUMO_VALIDATED_ATTENTION,
        "model": "14b",
        "seed": seed,
        "use_rembg": False,  # RemBG off — full-reference generation (09.9-25-05)
        "use_mel_band": False,
        "build_cfg_schedule": False,
        "portrait_batch": False,
        "lora_specs": [
            (HUMO_FASTWAN_LORA_NAME, HUMO_FASTWAN_LORA_STRENGTH),
            (HUMO_FACE_DETAILER_LORA_NAME, HUMO_FACE_DETAILER_LORA_STRENGTH),
            (HUMO_DISTILL_LORA_NAME, HUMO_DISTILL_LORA_STRENGTH),
        ],
        "duration_s": duration_s,
    }


def _adjust_frames_for_humo(duration_s: float, fps: int = HUMO_FRAME_RATE) -> int:
    """Return the Wan 4n+1 frame count for a HUMO clip.

    ``HuMoEmbeds`` applies the SAME rounding internally, so passing
    ``num_frames=round(duration_s * fps)`` into ``HuMoEmbeds`` yields this exact
    count. The wrong ``4 * ((n + 2) // 4) + 1`` (a prior-agent bug) is NOT used.
    """
    n = round(duration_s * fps)
    return 4 * ((n - 1) // 4) + 1


def build_humo_talking_head_workflow(
    *,
    portrait_path: str,
    audio_path: str,
    duration_s: float,
    fps: int = HUMO_FRAME_RATE,
    seed: int | None = None,
    width: int = HUMO_DEFAULT_WIDTH,
    height: int = HUMO_DEFAULT_HEIGHT,
    steps: int = HUMO_SAMPLER_STEPS,
    cfg: float = HUMO_SAMPLER_CFG,
    shift: float = HUMO_SAMPLER_SHIFT,
    scheduler: str = HUMO_SAMPLER_SCHEDULER,
    audio_scale: float = HUMO_AUDIO_SCALE,
    audio_cfg_scale: float = HUMO_AUDIO_CFG_SCALE,
    output_prefix: str = HUMO_OUTPUT_PREFIX,
    model: str = "14b",
    base_precision: str | None = None,
    attention_mode: str | None = None,
    lora_specs: list[tuple[str, float]] | None = None,
    # VRGDG V9 features — RemBG portrait preprocessing.
    use_rembg: bool = False,
    # Path A (Kijai) features — MelBandRoFormer audio cleaning + dynamic CFG.
    use_mel_band: bool = False,
    build_cfg_schedule: bool = False,
    # Portrait batching — repeat reference image 2x for stronger identity lock.
    # VRGDG V9: no batching. Kijai: batch 2x.
    portrait_batch: bool = False,
) -> dict:
    """Build a ComfyUI API-format HUMO talking-head workflow dict.

    Topological order (VRGDG V9 single-clip graph): model loader -> VAE loader ->
    whisper loader -> T5 loader -> LoadImage -> [RemBG] -> [Resize+Pad] ->
    [Batch] -> LoadAudio -> [MelBandRoFormer] -> [NormalizeAudio] -> text encode ->
    [CFG schedule] -> HuMoEmbeds (identity lock) -> WanVideoSampler ->
    WanVideoDecode -> VHS_VideoCombine.

    When ``use_rembg=True``, the reference portrait goes through RemBG
    (background removal) before resizing, matching VRGDG V9's pipeline
    for stronger identity conditioning.

    Args:
        portrait_path: Portrait in ComfyUI ``input/`` (identity anchor).
        audio_path: Vocals WAV in ComfyUI ``input/`` (lip-sync source).
        duration_s: Target clip duration in seconds.
        fps: Frame rate (HUMO default 24).
        seed: Optional fixed seed; None = random.
        width, height: Output frame size (HUMO default 832x480).
        steps: Sampler denoising steps (fewer = faster).
        cfg: Classifier-free guidance scale.
        shift: Sampler timestep shift.
        scheduler: Sampler scheduler (e.g. ``unipc``).
        audio_scale: HuMoEmbeds audio-conditioning strength.
        audio_cfg_scale: HuMoEmbeds audio CFG scale.
        output_prefix: VHS filename prefix (variant-tag).
        model: Model-size alias (``14b`` or ``1.7b``) from ``HUMO_MODELS``.
        base_precision: Override model precision (e.g. ``fp16_fast``).
            Defaults to the roster value for the given model.
        attention_mode: Override attention implementation
            (e.g. ``sageattn``). Defaults to ``sdpa``.
        lora_specs: Optional list of ``(lora_name, strength)`` tuples.
            LoRAs are chained in order (first → … → last → model_loader).
            ``lora_name`` is the filename as listed in ``ComfyUI/models/loras/``
            (e.g. ``"FastWan/FastWan_T2V_14B_480p_lora_rank_64_bf16.safetensors"``).
        use_rembg: Enable RemBG background removal on the reference portrait
            (VRGDG V9 reference). When True, portrait goes through
            RemBGSession+ → ImageRemoveBackground+ before resizing.
        use_mel_band: Enable MelBandRoFormer audio cleaning + NormalizeAudioLoudness
            pipeline (Path A / Kijai reference). When True, audio flows through
            MelBandRoFormerSampler → NormalizeAudioLoudness before reaching
            HuMoEmbeds. The cleaned audio is also used in VHS_VideoCombine.
        build_cfg_schedule: Build a ``CreateCFGScheduleFloatList`` node and wire
            its output to the sampler's ``cfg`` input (Path A / Kijai reference).
            When False, ``cfg`` is passed as a static float.
        portrait_batch: Repeat the reference portrait 2x via ImageExpandBatch+
            for stronger identity lock (Kijai reference). VRGDG V9: disabled.

    Returns:
        ComfyUI API-format workflow dict (nodes keyed by id).
    """
    num_frames = round(duration_s * fps)
    actual_seed = seed if seed is not None else 0
    ids = HUMO_NODE_IDS

    # Resolve model checkpoint + precision from roster.
    _model_info = HUMO_MODELS.get(model)
    if _model_info is None:
        raise KeyError(f"unknown humo model {model!r}; choose from {sorted(HUMO_MODELS)}")
    _model_filename = _model_info["filename"]
    _model_precision = base_precision or _model_info.get("precision", HUMO_BASE_PRECISION)
    _attention_mode = attention_mode or "sdpa"
    wf: dict = {}

    wf[ids["model_loader"]] = {
        "class_type": "WanVideoModelLoader",
        "inputs": {
            "model": _model_filename,
            "base_precision": _model_precision,
            "quantization": HUMO_QUANTIZATION,
            "load_device": HUMO_LOAD_DEVICE,
            "attention_mode": _attention_mode,
        },
    }
    wf[ids["vae_loader"]] = {
        "class_type": "WanVideoVAELoader",
        "inputs": {"model_name": WAN2_1_VAE_BF16, "precision": VAE_PRECISION},
    }
    wf[ids["whisper_loader"]] = {
        "class_type": "WhisperModelLoader",
        "inputs": {
            "model": WHISPER_FILENAME,
            "base_precision": WHISPER_BASE_PRECISION,
            "load_device": WHISPER_LOAD_DEVICE,
        },
    }
    wf[ids["t5_loader"]] = {
        "class_type": "LoadWanVideoT5TextEncoder",
        "inputs": {"model_name": UMT5_FILENAME, "precision": T5_PRECISION},
    }
    wf[ids["load_image"]] = {
        "class_type": "LoadImage",
        "inputs": {"image": portrait_path},
    }
    # Portrait preprocessing pipeline: load_image → [rembg] → resize → [batch]
    # Determine the image source for the next stage.
    _portrait_source = ids["load_image"]

    # Optional: RemBG background removal (VRGDG V9 reference).
    # Removes background to isolate the person for stronger identity conditioning.
    if use_rembg:
        wf[ids["rembg_session"]] = {
            "class_type": "RemBGSession+",
            "inputs": {
                "model": HUMO_REMBG_MODEL,
                "providers": HUMO_REMBG_DEVICE,
            },
        }
        wf[ids["rembg"]] = {
            "class_type": "ImageRemoveBackground+",
            "inputs": {
                "image": [ids["load_image"], 0],
                "rembg_session": [ids["rembg_session"], 0],
            },
        }
        _portrait_source = ids["rembg"]

    # Resize reference to output dimensions with padding (VRGDG V9: 720x480).
    # Matching reference embeds to output canvas eliminates aspect ratio drift.
    # If reference aspect ratio differs from output, white padding is added.
    wf[ids["portrait_resize"]] = {
        "class_type": "ResizeAndPadImage",
        "inputs": {
            "image": [_portrait_source, 0],
            "target_width": width,
            "target_height": height,
            "interpolation": HUMO_PORTRAIT_RESIZE_INTERPOLATION,
            "padding_color": HUMO_PORTRAIT_RESIZE_PADDING_COLOR,
        },
    }
    _portrait_source = ids["portrait_resize"]

    # Optional: Reference image batching (Kijai reference).
    # Repeat portrait 2x for stronger identity lock. VRGDG V9: disabled.
    if portrait_batch:
        wf[ids["portrait_batch"]] = {
            "class_type": "ImageExpandBatch+",
            "inputs": {
                "image": [_portrait_source, 0],
                "size": 2,
                "method": "repeat first",
            },
        }
        _portrait_source = ids["portrait_batch"]
    wf[ids["load_audio"]] = {
        "class_type": "LoadAudio",
        "inputs": {"audio": audio_path},
    }
    # Determine the audio source for HuMoEmbeds and VHS_VideoCombine.
    # When use_mel_band is True, audio flows through MelBandRoFormer + loudnorm.
    _audio_source_node = ids["load_audio"]
    _audio_source_slot = 0
    if use_mel_band:
        wf[ids["mel_band_loader"]] = {
            "class_type": "MelBandRoFormerModelLoader",
            "inputs": {"model_name": MEL_BAND_ROFORMER_FILENAME},
        }
        wf[ids["mel_band_sampler"]] = {
            "class_type": "MelBandRoFormerSampler",
            "inputs": {
                "model": [ids["mel_band_loader"], 0],
                "audio": [ids["load_audio"], 0],
            },
        }
        wf[ids["normalize_audio"]] = {
            "class_type": "NormalizeAudioLoudness",
            "inputs": {
                "audio": [ids["mel_band_sampler"], 0],
                "lufs": -23.0,
            },
        }
        _audio_source_node = ids["normalize_audio"]
        _audio_source_slot = 0

    wf[ids["text_encode"]] = {
        "class_type": "WanVideoTextEncode",
        "inputs": {
            "positive_prompt": "a person speaking clearly, natural talking-head",
            "negative_prompt": "(static, blurred, distorted face, extra limbs)",
            "t5": [ids["t5_loader"], 0],
        },
    }
    # Determine the cfg source for the sampler.
    # When build_cfg_schedule is True, use CreateCFGScheduleFloatList (Path A).
    # Kijai reference: steps=8, cfg 2->2 (constant), linear, 0-0.01 percent.
    _cfg_input: list | float = [ids["cfg_schedule"], 0] if build_cfg_schedule else cfg
    if build_cfg_schedule:
        wf[ids["cfg_schedule"]] = {
            "class_type": "CreateCFGScheduleFloatList",
            "inputs": {
                "steps": steps,
                "cfg_scale_start": 8.0,
                "cfg_scale_end": 2.0,
                "interpolation": "linear",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        }
        _cfg_input = [ids["cfg_schedule"], 0]
    wf[ids["humo_embeds"]] = {
        "class_type": "HuMoEmbeds",
        "inputs": {
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "audio_scale": audio_scale,
            "audio_cfg_scale": audio_cfg_scale,
            "audio_start_percent": HUMO_AUDIO_START_PERCENT,
            "audio_end_percent": HUMO_AUDIO_END_PERCENT,
            "whisper_model": [ids["whisper_loader"], 0],
            "vae": [ids["vae_loader"], 0],
            "reference_images": [_portrait_source, 0],
            "audio": [_audio_source_node, _audio_source_slot],
        },
    }
    wf[ids["sampler"]] = {
        "class_type": "WanVideoSampler",
        "inputs": {
            "model": [ids["model_loader"], 0],
            "image_embeds": [ids["humo_embeds"], 0],
            "text_embeds": [ids["text_encode"], 0],
            "steps": steps,
            "cfg": _cfg_input,
            "shift": shift,
            "seed": actual_seed,
            "force_offload": HUMO_SAMPLER_FORCE_OFFLOAD,
            "scheduler": scheduler,
            "riflex_freq_index": HUMO_SAMPLER_RIFLEX_FREQ_INDEX,
        },
    }
    wf[ids["decode"]] = {
        "class_type": "WanVideoDecode",
        "inputs": {
            "vae": [ids["vae_loader"], 0],
            "samples": [ids["sampler"], 0],
            "enable_vae_tiling": HUMO_DECODE_ENABLE_VAE_TILING,
            "tile_x": HUMO_DECODE_TILE_X,
            "tile_y": HUMO_DECODE_TILE_Y,
            "tile_stride_x": HUMO_DECODE_TILE_STRIDE_X,
            "tile_stride_y": HUMO_DECODE_TILE_STRIDE_Y,
        },
    }
    wf[ids["video_combine"]] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": [ids["decode"], 0],
            "audio": [_audio_source_node, _audio_source_slot],
            "filename_prefix": output_prefix,
            "format": HUMO_OUTPUT_FORMAT,
            "frame_rate": float(fps),
            "pix_fmt": HUMO_OUTPUT_PIX_FMT,
            "crf": HUMO_OUTPUT_CRF,
            "loop_count": HUMO_OUTPUT_LOOP_COUNT,
            "pingpong": HUMO_OUTPUT_PINGPONG,
            "save_output": True,
        },
    }

    # ── LoRA chain ──────────────────────────────────────────────────────
    if lora_specs:
        lora_node_ids: list[str] = []
        for idx, (lora_name, strength) in enumerate(lora_specs):
            node_id = str(_HUMO_LORA_NODE_START + idx)
            lora_node_ids.append(node_id)
            lora_inputs: dict[str, Any] = {
                "lora_name": lora_name,
                "strength": strength,
            }
            # Chain prev_lora from previous LoRA node.
            if idx > 0:
                lora_inputs["prev_lora"] = [lora_node_ids[idx - 1], 0]
            wf[node_id] = {
                "class_type": "WanVideoLoraSelectByName",
                "inputs": lora_inputs,
            }
        # Connect final LoRA in chain to model loader.
        wf[ids["model_loader"]]["inputs"]["lora"] = [lora_node_ids[-1], 0]

    return wf
