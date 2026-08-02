"""ComfyUI workflow JSON builders for T2I, T2V, I2I, I2V.

Adapted from proven benchmark scripts at /home/tyc/ComfyUI/benchmark/.
Each builder returns a ComfyUI API-format dict ready for /prompt submission.

CRITICAL: Model filenames and CLIPLoader types MUST match the verified inventory.
Wrong values produce garbled output, not errors (Pitfall 1 in research).

LTX and Ovi dialogue builders (LTX-2 + Ovi I2AV) live in
`src/workflow_dialogue.py` and are re-exported from this module for
backward-compatible imports.
"""

from __future__ import annotations

import random

# ── Model Filenames (verified inventory from benchmarks) ─────────

FLUX2_DIFFUSION_MODEL = "flux2_dev_fp8mixed.safetensors"
FLUX2_TEXT_ENCODER = "mistral_3_small_flux2_fp8.safetensors"
FLUX2_VAE = "flux2-vae.safetensors"
FLUX2_GUIDANCE_CFG = 3.5

WAN_T2V_DIFFUSION_MODEL = "wan2.1_t2v_14B_bf16.safetensors"
WAN_I2V_DIFFUSION_MODEL = "wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
WAN_TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
WAN_CLIP_VISION = "clip_vision_h.safetensors"
CAUSVID_LORA = "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"

WAN_SHIFT = 8.0
WAN_VIDEO_FPS = 16

# Qwen Image Edit models (Aug 2025 — kept for build_qwen_t2i_workflow + parity-bench)
QWEN_EDIT_DIFFUSION_MODEL = "qwen_image_edit_fp8_e4m3fn.safetensors"
QWEN_EDIT_TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_EDIT_VAE = "qwen_image_vae.safetensors"
QWEN_EDIT_LIGHTNING_LORA = "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_AURAFLOW_SHIFT = 3

# Qwen Image Edit 2511 (Dec 2025 — multi-image fusion, up to 3 refs)
QWEN_2511_DIFFUSION_MODEL = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_2511_TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"   # reused
QWEN_2511_VAE = "qwen_image_vae.safetensors"                        # reused
QWEN_2511_LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_2511_ANIME_LORA = "Qwen-Image-Edit-2511-Anime-2000.safetensors"
QWEN_2511_AURAFLOW_SHIFT = 3.1

# Qwen Image Edit 2511 GGUF Q3_K_M (low-VRAM I2I, ~10 GB peak)
QWEN_GGUF_Q3_UNET = "qwen-image-edit-2511-Q3_K_M.gguf"
# TE and VAE reused from QWEN_2511 (safetensors — not downloading GGUF TE)
QWEN_GGUF_Q3_VRAM_MB = 9000  # live-measured 2026-06-25: delta 8650 MiB, rounded up

# Qwen Image 2512 (dedicated T2I — separate from edit models)
QWEN_2512_DIFFUSION_MODEL = "qwen_image_fp8_e4m3fn.safetensors"
QWEN_2512_TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"   # shared
QWEN_2512_VAE = "qwen_image_vae.safetensors"                        # shared
QWEN_2512_AURAFLOW_SHIFT = 3.0

# HiDream-I1 (native ComfyUI, quadruple text encoder)
HIDREAM_DEV_DIFFUSION_MODEL = "hidream_i1_dev_fp8.safetensors"
HIDREAM_FAST_DIFFUSION_MODEL = "hidream_i1_fast_fp8.safetensors"
HIDREAM_CLIP_L = "clip_l_hidream.safetensors"
HIDREAM_CLIP_G = "clip_g_hidream.safetensors"
HIDREAM_T5XXL = "t5xxl_fp8_e4m3fn_scaled.safetensors"
HIDREAM_LLAMA = "llama_3.1_8b_instruct_fp8_scaled.safetensors"
HIDREAM_VAE = "ae.safetensors"  # shared with Flux

# Flux.1-schnell GGUF Q4_K_S (low-VRAM T2I, ~7-9 GB peak)
FLUX_GGUF_Q4_UNET = "flux1-schnell-Q4_K_S.gguf"
FLUX_GGUF_Q4_CLIP_L = "clip_l.safetensors"
FLUX_GGUF_Q4_T5 = "t5xxl_fp8_e4m3fn_scaled.safetensors"
FLUX_GGUF_Q4_VAE = "ae.safetensors"

# Flux.2 Klein 4B GGUF Q4_K_M (low-VRAM T2I, ~5-6 GB peak)
FLUX2_KLEIN_GGUF_Q4_UNET = "flux-2-klein-4b-Q4_K_M.gguf"
FLUX2_KLEIN_GGUF_Q4_CLIP = "qwen_3_4b.safetensors"
FLUX2_KLEIN_GGUF_Q4_VAE = "flux2-vae.safetensors"

# Z-Image-Turbo GGUF Q4_K_M (low-VRAM T2I, ~8-10 GB peak)
ZIMAGE_GGUF_Q4_UNET = "z-image-turbo-Q4_K_M.gguf"
ZIMAGE_GGUF_Q4_CLIP = "qwen_3_4b.safetensors"
ZIMAGE_GGUF_Q4_VAE = "ae.safetensors"

# Sana 1.6B (low-VRAM T2I, ~6-8 GB peak)
SANA_1_6B_CHECKPOINT = "SANA1.5_1.6B_1024px.pth"
SANA_1_6B_MODEL_CONFIG = "SanaMS_1600M_P1_D20"
SANA_DCAE_VAE = "dc-ae-f32c32-sana-1.0.safetensors"
SANA_DCAE_VAE_TYPE = "dcae-f32c32-sana-1.0"
SANA_GEMMA_MODEL = "Efficient-Large-Model/gemma-2-2b-it"

SEED_MAX = 2**32 - 1


def _make_seed(seed: int | None) -> int:
    """Return provided seed or generate a random one."""
    if seed is not None:
        return seed
    return random.randint(0, SEED_MAX)


# ── T2I: FLUX.2 Dev fp8mixed ────────────────────────────────────


def build_t2i_workflow(
    prompt: str,
    negative: str = "",
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
) -> dict:
    """FLUX.2 Dev fp8mixed T2I workflow. Returns ComfyUI API-format dict.

    FLUX.2 uses FluxGuidance instead of negative prompts.
    KSampler cfg is set to 1.0 (guidance handled by FluxGuidance node).
    """
    actual_seed = _make_seed(seed)

    return {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": FLUX2_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": FLUX2_TEXT_ENCODER,
                "type": "flux2",
            },
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": FLUX2_VAE},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": prompt},
        },
        "13": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["6", 0],
                "guidance": FLUX2_GUIDANCE_CFG,
            },
        },
        "7": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "width": width,
                "height": height,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["13", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["12", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_t2i", "images": ["8", 0]},
        },
    }


# ── T2V: Wan 2.1 + CausVid LoRA ─────────────────────────────────


def build_t2v_workflow(
    prompt: str,
    negative: str = "",
    seed: int | None = None,
    *,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    steps: int = 8,
) -> dict:
    """Wan 2.1 + CausVid LoRA T2V workflow. Returns ComfyUI API-format dict.

    CausVid LoRA is a distillation LoRA allowing fast generation at 4-8 steps.
    Uses ModelSamplingSD3 with shift=8.0 for Wan 2.1 noise schedule.
    """
    actual_seed = _make_seed(seed)

    return {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": WAN_TEXT_ENCODER,
                "type": "wan",
            },
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": WAN_VAE},
        },
        "3": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": WAN_T2V_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 0]},
        },
        "6": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["3", 0], "shift": WAN_SHIFT},
        },
        "20": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["7", 0],
                "clip": ["1", 0],
                "lora_name": CAUSVID_LORA,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": actual_seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["20", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["2", 0]},
        },
        "11": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["9", 0],
                "fps": float(WAN_VIDEO_FPS),
            },
        },
        "10": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11", 0],
                "filename_prefix": "alice_t2v",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


# ── I2I: Qwen Image Edit (instruction-based) ─────────────────────


def build_i2i_workflow(
    prompt: str,
    image_paths: list[str],
    seed: int | None = None,
    *,
    steps: int = 4,
    anime_lora: bool = False,
) -> dict:
    """Qwen-Image-Edit-2511 multi-image-fusion I2I workflow (1..3 refs).

    Uses the native ComfyUI ``TextEncodeQwenImageEditPlus`` node — the Plus
    encoder packs each ref image into the conditioning's reference_latents
    list, so up to 3 identities fuse cleanly into one scene. The starting
    latent is image1 ONLY (refs 2/3 contribute purely as conditioning).

    Lightning coupling (single source of truth — owned here, not at MCP layer):
      - steps=4 → Lightning LoRA on, cfg=1.0  (Lightning trained operating point)
      - steps≠4 → Lightning LoRA off, cfg=4.0 (Comfy team recommendation; bumped
                  from Aug 2025's 2.5 per official 2511 template)

    Anime LoRA:
      - anime_lora=True → anime style LoRA loaded before Lightning in
        the model patch chain: UNET → anime → [Lightning] → CFGNorm.

    Args:
        prompt: Instruction describing what to change / target scene.
        image_paths: 1..3 reference image filenames (already in ComfyUI input/).
            For N=4..5, caller chains two passes — see scripts/qwen_multi_chain_n5.py.
        seed: Optional fixed seed; None = random.
        steps: 4 = Lightning fast path; 20 = non-Lightning higher quality.
        anime_lora: Load anime style LoRA for consistent anime outputs.

    Returns:
        ComfyUI API-format workflow dict.

    Raises:
        ValueError: ``len(image_paths)`` outside 1..3.
    """
    if not 1 <= len(image_paths) <= 3:
        raise ValueError(
            f"image_paths must have length 1..3, got {len(image_paths)}"
        )

    actual_seed = _make_seed(seed)
    use_lightning = (steps == 4)
    cfg = 1.0 if use_lightning else 4.0

    img1 = image_paths[0]
    img2 = image_paths[1] if len(image_paths) >= 2 else None
    img3 = image_paths[2] if len(image_paths) >= 3 else None

    wf: dict = {
        # ── Model loaders ──
        "37": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": QWEN_2511_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": QWEN_2511_TEXT_ENCODER,
                "type": "qwen_image",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": QWEN_2511_VAE},
        },
        # ── Reference images (image1 mandatory; image2/3 optional below) ──
        "14": {
            "class_type": "LoadImage",
            "inputs": {"image": img1},
        },
        # ── Conditioning: positive (TextEncodeQwenImageEditPlus, prompt + refs) ──
        "76": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["38", 0],
                "prompt": prompt,
                "vae": ["39", 0],
                "image1": ["14", 0],
            },
        },
        # ── Conditioning: negative (same Plus class, empty prompt, mirrored refs) ──
        "77": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["38", 0],
                "prompt": "",
                "vae": ["39", 0],
                "image1": ["14", 0],
            },
        },
        # ── Starting latent: image1 ONLY (refs 2/3 never fan into latent) ──
        "88": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["14", 0],
                "vae": ["39", 0],
            },
        },
        # ── Model patch chain: UNET → [anime LoRA] → [Lightning LoRA] → CFGNorm → ModelSamplingAuraFlow ──
        "75": {
            "class_type": "CFGNorm",
            "inputs": {
                "model": (
                    ["89", 0] if use_lightning
                    else ["90", 0] if anime_lora
                    else ["37", 0]
                ),
                "strength": 1,
            },
        },
        "66": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["75", 0],
                "shift": QWEN_2511_AURAFLOW_SHIFT,
            },
        },
        # ── Sampler ──
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["66", 0],
                "positive": ["76", 0],
                "negative": ["77", 0],
                "latent_image": ["88", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        # ── Decode + FluxKontextImageScale (template-trained resolution bucket) + Save ──
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        },
        "160": {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["8", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_i2i", "images": ["160", 0]},
        },
    }

    # Optional image2 / image3 — wired into BOTH positive and negative encoders
    if img2 is not None:
        wf["15"] = {"class_type": "LoadImage", "inputs": {"image": img2}}
        wf["76"]["inputs"]["image2"] = ["15", 0]
        wf["77"]["inputs"]["image2"] = ["15", 0]
    if img3 is not None:
        wf["16"] = {"class_type": "LoadImage", "inputs": {"image": img3}}
        wf["76"]["inputs"]["image3"] = ["16", 0]
        wf["77"]["inputs"]["image3"] = ["16", 0]

    # Lightning LoRA only on the steps=4 path
    if use_lightning:
        wf["89"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["90", 0] if anime_lora else ["37", 0],
                "lora_name": QWEN_2511_LIGHTNING_LORA,
                "strength_model": 1.0,
            },
        }

    # Anime style LoRA — loaded before Lightning in the patch chain
    if anime_lora:
        wf["90"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["37", 0],
                "lora_name": QWEN_2511_ANIME_LORA,
                "strength_model": 1.0,
            },
        }

    return wf


# ── I2I: Qwen Image Edit 2511 GGUF Q3_K_M (low-VRAM, ~10 GB) ─────


def build_qwen_gguf_i2i_workflow(
    prompt: str,
    image_paths: list[str],
    seed: int | None = None,
    *,
    steps: int = 4,
    anime_lora: bool = False,
) -> dict:
    """Qwen-Image-Edit-2511 GGUF Q3_K_M I2I workflow (low-VRAM, ~10GB).

    Uses UnetLoaderGGUF for the quantized diffusion model. Text encoder
    and VAE remain safetensors (per D-02).

    Lightning coupling (per D-03):
      - steps=4  → Lightning LoRA on, cfg=1.0
      - steps!=4 → Lightning LoRA off, cfg=2.5

    Anime LoRA:
      - anime_lora=True → anime style LoRA loaded before Lightning in
        the model patch chain: UNET → anime → [Lightning] → CFGNorm.

    Args:
        prompt: Instruction describing what to change / target scene.
        image_paths: 1..3 reference image filenames.
        seed: Optional fixed seed; None = random.
        steps: 4 = Lightning fast path; 20 = non-Lightning higher quality.
        anime_lora: Load anime style LoRA for consistent anime outputs.

    Returns:
        ComfyUI API-format workflow dict.

    Raises:
        ValueError: ``len(image_paths)`` outside 1..3.
    """
    if not 1 <= len(image_paths) <= 3:
        raise ValueError(
            f"image_paths must have length 1..3, got {len(image_paths)}"
        )

    actual_seed = _make_seed(seed)
    use_lightning = (steps == 4)
    cfg = 1.0 if use_lightning else 2.5

    img1 = image_paths[0]
    img2 = image_paths[1] if len(image_paths) >= 2 else None
    img3 = image_paths[2] if len(image_paths) >= 3 else None

    wf: dict = {
        # ── Model loaders ──
        "37": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": QWEN_GGUF_Q3_UNET},
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": QWEN_2511_TEXT_ENCODER,
                "type": "qwen_image",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": QWEN_2511_VAE},
        },
        # ── Reference images ──
        "14": {
            "class_type": "LoadImage",
            "inputs": {"image": img1},
        },
        # ── Conditioning: positive ──
        "76": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["38", 0],
                "prompt": prompt,
                "vae": ["39", 0],
                "image1": ["14", 0],
            },
        },
        # ── Conditioning: negative ──
        "77": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["38", 0],
                "prompt": "",
                "vae": ["39", 0],
                "image1": ["14", 0],
            },
        },
        # ── Starting latent: image1 only ──
        "88": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["14", 0],
                "vae": ["39", 0],
            },
        },
        # ── Model patch chain: UNET → [anime LoRA] → [Lightning LoRA] → CFGNorm → ModelSamplingAuraFlow ──
        "75": {
            "class_type": "CFGNorm",
            "inputs": {
                "model": (
                    ["89", 0] if use_lightning
                    else ["90", 0] if anime_lora
                    else ["37", 0]
                ),
                "strength": 1,
            },
        },
        "66": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["75", 0],
                "shift": QWEN_2511_AURAFLOW_SHIFT,
            },
        },
        # ── Sampler ──
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["66", 0],
                "positive": ["76", 0],
                "negative": ["77", 0],
                "latent_image": ["88", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        # ── Decode + scale + save ──
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        },
        "160": {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["8", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_i2i_gguf", "images": ["160", 0]},
        },
    }

    # Optional image2 / image3
    if img2 is not None:
        wf["15"] = {"class_type": "LoadImage", "inputs": {"image": img2}}
        wf["76"]["inputs"]["image2"] = ["15", 0]
        wf["77"]["inputs"]["image2"] = ["15", 0]
    if img3 is not None:
        wf["16"] = {"class_type": "LoadImage", "inputs": {"image": img3}}
        wf["76"]["inputs"]["image3"] = ["16", 0]
        wf["77"]["inputs"]["image3"] = ["16", 0]

    # Lightning LoRA only on the steps=4 path
    if use_lightning:
        wf["89"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["90", 0] if anime_lora else ["37", 0],
                "lora_name": QWEN_2511_LIGHTNING_LORA,
                "strength_model": 1.0,
            },
        }

    # Anime style LoRA — loaded before Lightning in the patch chain
    if anime_lora:
        wf["90"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["37", 0],
                "lora_name": QWEN_2511_ANIME_LORA,
                "strength_model": 1.0,
            },
        }

    return wf


def build_qwen_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    steps: int = 4,
) -> dict:
    """Qwen Image Edit T2I workflow (no input image).

    For A/B testing against FLUX.2 Dev. Same model but without reference image.
    """
    actual_seed = _make_seed(seed)

    return {
        "37": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": QWEN_EDIT_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": QWEN_EDIT_TEXT_ENCODER,
                "type": "qwen_image",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": QWEN_EDIT_VAE},
        },
        "89": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["37", 0],
                "lora_name": QWEN_EDIT_LIGHTNING_LORA,
                "strength_model": 1.0,
            },
        },
        "76": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "clip": ["38", 0],
                "prompt": prompt,
            },
        },
        "77": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "clip": ["38", 0],
                "prompt": "",
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1,
            },
        },
        "66": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["89", 0],
                "shift": QWEN_AURAFLOW_SHIFT,
            },
        },
        "75": {
            "class_type": "CFGNorm",
            "inputs": {
                "model": ["66", 0],
                "strength": 1,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["75", 0],
                "positive": ["76", 0],
                "negative": ["77", 0],
                "latent_image": ["5", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_qwen_t2i", "images": ["8", 0]},
        },
    }


# ── Flux I2I: FLUX.2 Dev img2img with denoise control ───────────

FLUX_I2I_DEFAULT_DENOISE = 0.6

# PuLID-Flux2 face-identity conditioning.
# Klein weights work for both Flux.2 Klein and Flux.2 Dev — no Dev-specific file on HuggingFace.
# v2 is preferred over v1 per Fayens/Pulid-Flux2 repo (both shipped together; v2 is the newer iteration).
PULID_FLUX2_WEIGHTS = "pulid_flux2_klein_v2.safetensors"
PULID_DEFAULT_STRENGTH = 1.0
FLUX_I2I_FACEID_DEFAULT_DENOISE = 0.7
INSIGHTFACE_PROVIDER = "CUDA"


def build_flux_i2i_workflow(
    prompt: str,
    image_filename: str,
    seed: int | None = None,
    *,
    denoise_strength: float = FLUX_I2I_DEFAULT_DENOISE,
    width: int = 1024,
    height: int = 1280,
    steps: int = 20,
) -> dict:
    """FLUX.2 Dev fp8mixed image-to-image workflow with denoise control.

    This is the "aggressive edit" counterpart to Qwen's instruction-based
    `build_i2i_workflow`. Use when the edit is a full-scene/composition
    shift (weapon swap, lighting change, pose shift) where identity drift
    on the subject is acceptable. For identity-preserving edits
    (instruction-based, preserve composition) use the Qwen builder.

    Note: `width`/`height` are accepted for API symmetry, but i2i derives
    the latent shape from VAEEncode of the input image — the output image
    aspect ratio follows the input. The params are forwarded for future
    use (e.g. adding a resize node) but do not affect current output.

    Args:
        prompt: Full-scene target description (not an edit instruction).
        image_filename: Filename (relative to ComfyUI input/) of the
            source image. Must already be copied into ComfyUI's input dir.
        seed: Optional fixed seed; None = random.
        denoise_strength: 0.0-1.0. Lower (0.3-0.5) preserves more of the
            input; higher (0.7-0.9) repaints more aggressively. Default
            0.6 is the recommended starting point.
        width: Accepted for API symmetry (see note above).
        height: Accepted for API symmetry (see note above).
        steps: KSampler steps (FLUX.2 default 20).

    Returns:
        ComfyUI API-format workflow dict.
    """
    actual_seed = _make_seed(seed)

    return {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": FLUX2_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": FLUX2_TEXT_ENCODER,
                "type": "flux2",
            },
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": FLUX2_VAE},
        },
        "14": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "88": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["14", 0],
                "vae": ["12", 0],
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": prompt},
        },
        "13": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["6", 0],
                "guidance": FLUX2_GUIDANCE_CFG,
            },
        },
        "7": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["13", 0],
                "negative": ["7", 0],
                "latent_image": ["88", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise_strength,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["12", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_flux_i2i", "images": ["8", 0]},
        },
    }


# ── Flux I2I + FaceID: PuLID-Flux2 face-identity-locked img2img ──


def build_flux_i2i_faceid_workflow(
    prompt: str,
    scene_image_filename: str,
    face_image_filename: str,
    seed: int | None = None,
    *,
    denoise_strength: float = FLUX_I2I_FACEID_DEFAULT_DENOISE,
    pulid_strength: float = PULID_DEFAULT_STRENGTH,
    width: int = 1024,
    height: int = 1280,
    steps: int = 20,
) -> dict:
    """FLUX.2 Dev img2img with PuLID-Flux2 face-identity conditioning.

    Preserves facial identity from ``face_image_filename`` across aggressive
    scene changes driven by ``prompt`` + ``denoise_strength``. The scene
    image (``scene_image_filename``) is the composition/pose reference that
    gets denoised; the face image supplies the identity that stays locked
    through the ApplyPuLIDFlux2 model patch feeding the KSampler.

    Use this builder when you'd otherwise reach for ``build_flux_i2i_workflow``
    but need identity preservation too. For instruction-only edits that
    preserve composition and identity automatically, use Qwen Image Edit
    (``build_i2i_workflow``).

    Args:
        prompt: Full target-scene description (not an edit instruction).
        scene_image_filename: Filename in ComfyUI ``input/`` of the
            scene/pose reference (gets denoised).
        face_image_filename: Filename in ComfyUI ``input/`` of the face
            identity reference (stays locked).
        seed: Optional fixed seed; None = random.
        denoise_strength: 0.0-1.0. Default 0.7 is higher than plain flux
            i2i because the identity lock makes aggressive denoise safe.
        pulid_strength: 0.0-2.0. 1.0 = normal identity lock. 1.4 =
            aggressive (per repo recommendation for stubborn drift).
        width: Accepted for API symmetry; output follows input aspect.
        height: Accepted for API symmetry; output follows input aspect.
        steps: KSampler steps (FLUX.2 default 20).

    Returns:
        ComfyUI API-format workflow dict. Node IDs are kept distinct from
        ``build_flux_i2i_workflow`` where they add new functionality to
        surface copy-paste bugs early: 15 (face LoadImage), 20/21/22
        (PuLID loaders), 23 (ApplyPuLIDFlux2). KSampler's ``model`` input
        routes from node 23, not node 10, so the patched model reaches
        the sampler.
    """
    actual_seed = _make_seed(seed)

    return {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": FLUX2_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": FLUX2_TEXT_ENCODER,
                "type": "flux2",
            },
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": FLUX2_VAE},
        },
        "14": {
            "class_type": "LoadImage",
            "inputs": {"image": scene_image_filename},
        },
        "15": {
            "class_type": "LoadImage",
            "inputs": {"image": face_image_filename},
        },
        "88": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["14", 0],
                "vae": ["12", 0],
            },
        },
        "20": {
            "class_type": "PuLIDInsightFaceLoader",
            "inputs": {"provider": INSIGHTFACE_PROVIDER},
        },
        "21": {
            "class_type": "PuLIDEVACLIPLoader",
            "inputs": {},
        },
        "22": {
            "class_type": "PuLIDModelLoader",
            "inputs": {"pulid_file": PULID_FLUX2_WEIGHTS},
        },
        "23": {
            "class_type": "ApplyPuLIDFlux2",
            "inputs": {
                "model": ["10", 0],
                "pulid_model": ["22", 0],
                "strength": pulid_strength,
                "eva_clip": ["21", 0],
                "face_analysis": ["20", 0],
                "image": ["15", 0],
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": prompt},
        },
        "13": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["6", 0],
                "guidance": FLUX2_GUIDANCE_CFG,
            },
        },
        "7": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["23", 0],
                "positive": ["13", 0],
                "negative": ["7", 0],
                "latent_image": ["88", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise_strength,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["12", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "alice_flux_faceid_i2i",
                "images": ["8", 0],
            },
        },
    }


# ── I2V: Wan 2.1 I2V 480p ───────────────────────────────────────


def build_i2v_workflow(
    prompt: str,
    image_filename: str,
    negative: str = "",
    seed: int | None = None,
    *,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    steps: int = 8,
) -> dict:
    """Wan 2.1 I2V 480p workflow.

    Uses CLIPVisionLoader + WanImageToVideo for image-conditioned video generation.
    KSampler uses cfg=6.0 and scheduler='normal' (different from T2V).
    """
    actual_seed = _make_seed(seed)

    return {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": WAN_TEXT_ENCODER,
                "type": "wan",
            },
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": WAN_VAE},
        },
        "3": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": WAN_I2V_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 0]},
        },
        "7": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["3", 0], "shift": WAN_SHIFT},
        },
        "32": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": WAN_CLIP_VISION},
        },
        "34": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "33": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["32", 0],
                "image": ["34", 0],
                "crop": "center",
            },
        },
        "30": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "vae": ["2", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "clip_vision_output": ["33", 0],
                "start_image": ["34", 0],
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": actual_seed,
                "steps": steps,
                "cfg": 6.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["7", 0],
                "positive": ["30", 0],
                "negative": ["30", 1],
                "latent_image": ["30", 2],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["2", 0]},
        },
        "11": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["9", 0],
                "fps": float(WAN_VIDEO_FPS),
            },
        },
        "10": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11", 0],
                "filename_prefix": "alice_i2v",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


# ── T2I: Qwen-Image-Edit-2511 (text-to-image, no reference) ────────


def build_qwen_2511_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
) -> dict:
    """Qwen-Image-Edit-2511 T2I workflow (no reference image).

    Uses the standard CLIPTextEncode + EmptySD3LatentImage pattern from the
    official ComfyUI Qwen-Image template (not the edit-specific
    TextEncodeQwenImageEdit node). Lightning LoRA at steps=4; disable LoRA
    for higher step counts.

    Same-family consistency: output can be fed directly to build_i2i_workflow
    (QEI 2511 I2I) with matched latent-space fidelity.
    """
    actual_seed = _make_seed(seed)
    use_lightning = steps == 4

    nodes: dict = {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": QWEN_2511_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": QWEN_2511_TEXT_ENCODER,
                "type": "qwen_image",
            },
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": QWEN_2511_VAE},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": prompt},
        },
        "7": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
    }

    model_ref = ["10", 0]

    if use_lightning:
        nodes["89"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": QWEN_2511_LIGHTNING_LORA,
                "strength_model": 1.0,
            },
        }
        model_ref = ["89", 0]

    nodes["66"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {
            "model": model_ref,
            "shift": QWEN_2511_AURAFLOW_SHIFT,
        },
    }

    nodes["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["66", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
            "seed": actual_seed,
            "steps": steps,
            "cfg": 1.0 if use_lightning else 4.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }

    nodes["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["12", 0]},
    }
    nodes["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "alice_qwen_2511_t2i", "images": ["8", 0]},
    }

    return nodes


# ── T2I: Qwen-Image-2512 (dedicated T2I model) ───────────────────


def build_qwen_2512_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
) -> dict:
    """Qwen-Image-2512 dedicated T2I workflow.

    Uses the purpose-built T2I diffusion model (not the edit model repurposed).
    Standard CLIPTextEncode + EmptySD3LatentImage + ConditioningZeroOut pattern.
    """
    actual_seed = _make_seed(seed)

    return {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": QWEN_2512_DIFFUSION_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": QWEN_2512_TEXT_ENCODER,
                "type": "qwen_image",
            },
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": QWEN_2512_VAE},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": prompt},
        },
        "7": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "66": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["10", 0],
                "shift": QWEN_2512_AURAFLOW_SHIFT,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["66", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["12", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_qwen_2512_t2i", "images": ["8", 0]},
        },
    }


# ── T2I: HiDream-I1 (quadruple text encoder) ─────────────────────

_HIDREAM_VARIANTS: dict[str, dict] = {
    "dev": {"model": HIDREAM_DEV_DIFFUSION_MODEL, "steps": 28, "shift": 6.0, "sampler": "lcm", "scheduler": "normal"},
    "fast": {"model": HIDREAM_FAST_DIFFUSION_MODEL, "steps": 16, "shift": 3.0, "sampler": "lcm", "scheduler": "normal"},
}


def build_hidream_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 0,
    variant: str = "dev",
) -> dict:
    """HiDream-I1 T2I workflow with quadruple text encoder.

    Uses QuadrupleCLIPLoader (CLIP-L + CLIP-G + T5-XXL + Llama 3.1 8B)
    with standard CLIPTextEncode for single-prompt dispatch.
    """
    actual_seed = _make_seed(seed)
    cfg = _HIDREAM_VARIANTS.get(variant, _HIDREAM_VARIANTS["dev"])
    effective_steps = steps if steps > 0 else cfg["steps"]

    return {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": cfg["model"],
                "weight_dtype": "default",
            },
        },
        "54": {
            "class_type": "QuadrupleCLIPLoader",
            "inputs": {
                "clip_name1": HIDREAM_CLIP_L,
                "clip_name2": HIDREAM_CLIP_G,
                "clip_name3": HIDREAM_T5XXL,
                "clip_name4": HIDREAM_LLAMA,
            },
        },
        "55": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": HIDREAM_VAE},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["54", 0], "text": prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["54", 0], "text": ""},
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "70": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["10", 0], "shift": cfg["shift"]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["70", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": actual_seed,
                "steps": effective_steps,
                "cfg": 1.0,
                "sampler_name": cfg["sampler"],
                "scheduler": cfg["scheduler"],
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["55", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": f"alice_hidream_{variant}_t2i", "images": ["8", 0]},
        },
    }


# ── T2I: Flux.1-schnell GGUF Q4 (low-VRAM, distillation) ──────────


def build_flux_gguf_q4_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
) -> dict:
    """Flux.1-schnell GGUF Q4_K_S T2I workflow (low-VRAM).

    Uses UnetLoaderGGUF for the GGUF UNet, standard DualCLIPLoader for
    safetensors text encoders (DualCLIPLoaderGGUF does not support FP8
    scaled safetensors). Distillation model: 4 steps, Euler, simple
    scheduler, cfg=1.0. Negative = positive (cfg-free distillation).
    """
    actual_seed = _make_seed(seed)

    return {
        "30": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": FLUX_GGUF_Q4_UNET},
        },
        "31": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": FLUX_GGUF_Q4_CLIP_L,
                "clip_name2": FLUX_GGUF_Q4_T5,
                "type": "flux",
            },
        },
        "32": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": FLUX_GGUF_Q4_VAE},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["31", 0], "text": prompt},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["30", 0],
                "positive": ["6", 0],
                "negative": ["6", 0],
                "latent_image": ["5", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["32", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_t2i_flux_gguf", "images": ["8", 0]},
        },
    }


# ── T2I: Flux.2 Klein 4B GGUF Q4 (low-VRAM, distillation) ─────────


def build_flux2_klein_gguf_q4_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
) -> dict:
    """Flux.2 Klein 4B GGUF Q4_K_M T2I workflow (low-VRAM).

    Uses UnetLoaderGGUF for the GGUF UNet + standard CLIPLoader for
    the safetensors text encoder (qwen_3_4b.safetensors).
    Distillation model: 4 steps, Euler, simple scheduler, cfg=1.0.
    """
    actual_seed = _make_seed(seed)

    return {
        "40": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": FLUX2_KLEIN_GGUF_Q4_UNET},
        },
        "41": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": FLUX2_KLEIN_GGUF_Q4_CLIP,
                "type": "flux2",
            },
        },
        "42": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": FLUX2_KLEIN_GGUF_Q4_VAE},
        },
        "43": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["41", 0], "text": prompt},
        },
        "44": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "45": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["40", 0],
                "positive": ["43", 0],
                "negative": ["43", 0],
                "latent_image": ["44", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "46": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["45", 0], "vae": ["42", 0]},
        },
        "47": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_t2i_flux2_klein", "images": ["46", 0]},
        },
    }


# ── T2I: Z-Image-Turbo GGUF Q4 (low-VRAM, distilled Turbo) ─────────


def build_zimage_gguf_q4_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
) -> dict:
    """Z-Image-Turbo GGUF Q4_K_M T2I workflow (low-VRAM).

    Uses UnetLoaderGGUF for the GGUF UNet, standard CLIPLoader with
    type="sd3" for the Qwen 3 4B text encoder (routes to z_image.te
    path with 2560-dim embeddings, not the flux2 path with 7680 dims).
    ConditioningZeroOut for negative (not cfg-free distillation).
    8 steps, Euler, simple scheduler, cfg=3.5.
    """
    actual_seed = _make_seed(seed)

    return {
        "50": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": ZIMAGE_GGUF_Q4_UNET},
        },
        "51": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": ZIMAGE_GGUF_Q4_CLIP,
                "type": "sd3",
            },
        },
        "52": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": ZIMAGE_GGUF_Q4_VAE},
        },
        "53": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["51", 0], "text": prompt},
        },
        "54": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["53", 0]},
        },
        "55": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "56": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["50", 0],
                "positive": ["53", 0],
                "negative": ["54", 0],
                "latent_image": ["55", 0],
                "seed": actual_seed,
                "steps": steps,
                "cfg": 3.5,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "57": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["56", 0], "vae": ["52", 0]},
        },
        "58": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alice_t2i_zimage", "images": ["57", 0]},
        },
    }


# ── T2I: Sana 1.6B (low-VRAM, DC-AE VAE, Gemma text encoder) ──────


def build_sana_t2i_workflow(
    prompt: str,
    seed: int | None = None,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
) -> dict:
    """Sana 1.6B T2I workflow (low-VRAM).

    Uses ComfyUI_ExtraModels nodes: SanaCheckpointLoader, GemmaLoader,
    SanaTextEncode (with built-in CHI prompt enhancement), ExtraVAELoader
    for DC-AE VAE, EmptySanaLatentImage (32x divisor).
    cfg=4.5, euler, normal scheduler.

    IMPORTANT: This function receives the ORIGINAL user prompt.
    SanaTextEncode has built-in CHI prompt enhancement — do NOT
    pre-enhance the prompt before passing it here.
    """
    actual_seed = _make_seed(seed)

    return {
        "200": {
            "class_type": "SanaCheckpointLoader",
            "inputs": {
                "ckpt_name": SANA_1_6B_CHECKPOINT,
                "model": SANA_1_6B_MODEL_CONFIG,
            },
        },
        "201": {
            "class_type": "GemmaLoader",
            "inputs": {
                "model_name": SANA_GEMMA_MODEL,
                "device": "cuda",
                "dtype": "BF16",
            },
        },
        "202": {
            "class_type": "SanaTextEncode",
            "inputs": {
                "text": prompt,
                "GEMMA": ["201", 0],
            },
        },
        "203": {
            "class_type": "GemmaTextEncode",
            "inputs": {
                "text": "",
                "GEMMA": ["201", 0],
            },
        },
        "204": {
            "class_type": "ExtraVAELoader",
            "inputs": {
                "vae_name": SANA_DCAE_VAE,
                "vae_type": SANA_DCAE_VAE_TYPE,
                "dtype": "BF16",
            },
        },
        "205": {
            "class_type": "EmptySanaLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "206": {
            "class_type": "KSampler",
            "inputs": {
                "seed": actual_seed,
                "steps": steps,
                "cfg": 4.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["200", 0],
                "positive": ["202", 0],
                "negative": ["203", 0],
                "latent_image": ["205", 0],
            },
        },
        "207": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["206", 0],
                "vae": ["204", 0],
            },
        },
        "208": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "alice_t2i_sana",
                "images": ["207", 0],
            },
        },
    }


# ── LTX and Ovi dialogue builders (re-exported from workflow_dialogue) ──
# Import placed at module tail to avoid circular import: workflow_dialogue
# imports `_make_seed` from this module.

from workflows.workflow_dialogue import (  # noqa: E402
    build_ltx2_i2av_workflow,
    build_ovi_i2av_workflow,
)

__all__ = [
    "build_t2i_workflow",
    "build_t2v_workflow",
    "build_i2i_workflow",
    "build_qwen_t2i_workflow",
    "build_qwen_2511_t2i_workflow",
    "build_qwen_2512_t2i_workflow",
    "build_hidream_t2i_workflow",
    "build_flux_i2i_workflow",
    "build_flux_i2i_faceid_workflow",
    "build_flux_gguf_q4_t2i_workflow",
    "build_flux2_klein_gguf_q4_t2i_workflow",
    "build_zimage_gguf_q4_t2i_workflow",
    "build_sana_t2i_workflow",
    "build_i2v_workflow",
    "build_ltx2_i2av_workflow",
    "build_ovi_i2av_workflow",
]
