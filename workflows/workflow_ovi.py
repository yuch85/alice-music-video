"""ComfyUI workflow builder for Ovi image-to-audio-video (I2AV).

Phase 09.5 Plan 03. Ovi (Character.AI) is an 11B I2AV model with native
`<S>..<E>` dialogue markers and `<AUDCAP>..<ENDAUDCAP>` ambient audio
caption syntax. Every literal filename / node class name is copied
verbatim from `.planning/phases/09.5-.../WAVE0_OVI.md`.
"""

from __future__ import annotations

from workflows.workflows import _make_seed

# ── Model filenames + loader constants (from WAVE0_OVI.md) ─────────

OVI_MODEL_PRECISION_FP8 = "Ovi-11B-fp8.safetensors"
"""fp8 variant — ~16-24 GB VRAM on 4090, MVP pick per WAVE0_OVI.md."""

OVI_ATTENTION_BACKEND = "sdpa"
"""Per WAVE0_OVI.md note: sdpa is stable on torch 2.6 + cu124."""

OVI_VAE_FILE = "Wan2.2_VAE.pth"
"""Ovi auto-downloads Wan2.2 VAE on first run via snapshot_download
(`Wan-AI/Wan2.2-TI2V-5B`). Symlinked into ComfyUI's models/vae/ by the
Plan 06 driver so `OviWanComponentLoader` can find it via the folder
listing. WAVE0_OVI.md earlier suggested `wan_2.1_vae.safetensors` but
Ovi's `OviWanComponentLoader.load` constructs a `Wan2_2_VAE` which
fails with a state_dict shape mismatch on the 2.1 checkpoint (decoder
middle layers 384 vs 1024 channels). Fix confirmed 2026-04-15 during
live run — VAE from Wan2.2 snapshot dir is the correct one."""

OVI_UMT5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
"""Ovi-specific umt5 XXL bf16 .pth file. WAVE0_OVI.md earlier suggested
the SwarmUI fp8-scaled .safetensors but Ovi's T5EncoderModel expects
the original `models_t5_umt5-xxl-enc-bf16.pth` shape (different
quantization layout). File is in the Wan2.2 snapshot dir; symlinked
into ComfyUI's models/text_encoders/ by the Plan 06 driver."""

OVI_DEFAULT_WIDTH = 512
OVI_DEFAULT_HEIGHT = 960  # vertical 9:16 for canonical sister portraits
OVI_DEFAULT_STEPS = 50
OVI_DEFAULT_SHIFT = 5.0
OVI_DEFAULT_VIDEO_GUIDANCE = 4.0
OVI_DEFAULT_AUDIO_GUIDANCE = 3.0
OVI_DEFAULT_SLG_LAYER = 11
OVI_FPS = 24
"""Ovi v1.0/v1.1 outputs at 24 fps (fixed by architecture)."""


def _ovi_combined_prompt(
    scene_prompt: str,
    dialogue_text: str,
    ambient_audio_hint: str,
) -> str:
    """Wrap Ovi prompt with required `<S>..<E>` speech markers + optional
    `<AUDCAP>..<ENDAUDCAP>` ambient-audio caption (Ovi model card format).

    Every Ovi I2AV prompt must include:
      - Scene description (plain prose)
      - `<S>dialogue here<E>` for synthesized speech
      - `<AUDCAP>ambient scene audio description<ENDAUDCAP>` for the
        non-speech audio track
    """
    parts = [scene_prompt.strip()]
    if dialogue_text:
        parts.append(f"<S>{dialogue_text}<E>")
    if ambient_audio_hint:
        parts.append(f"<AUDCAP>{ambient_audio_hint}<ENDAUDCAP>")
    return " ".join(parts)


def build_ovi_i2av_workflow(
    scene_prompt: str,
    dialogue_text: str,
    ambient_audio_hint: str,
    ref_image_filename: str,
    length_s: int = 10,
    seed: int | None = None,
    *,
    width: int = OVI_DEFAULT_WIDTH,
    height: int = OVI_DEFAULT_HEIGHT,
    steps: int = OVI_DEFAULT_STEPS,
) -> dict:
    """Ovi image-to-audio-video (I2AV) workflow. Returns ComfyUI API dict.

    Workflow shape per WAVE0_OVI.md "Minimal Ovi workflow skeleton":
      LoadImage -> IMAGE (first_frame_image)
      OviEngineLoader (model_precision=fp8) -> OVI_ENGINE
      OviAttentionSelector (sdpa) -> OVI_ENGINE
      OviWanComponentLoader (WAN VAE + umt5 fp8) -> OVI_ENGINE
      OviVideoGenerator (engine, prompt with <S>..<E> + <AUDCAP>..<ENDAUDCAP>,
                        first_frame_image) -> video_latents, audio_latents
      OviLatentDecoder -> IMAGE, AUDIO
      CreateVideo (images + audio) -> SaveVideo (mp4)

    length_s is informational only — Ovi clip duration is fixed by
    architecture (5s in v1.0, 10s in v1.1). The first run will confirm
    the resident version's behavior; builder does not pass a length param.

    Args:
        scene_prompt: Visual description of the scene.
        dialogue_text: Speech content — wrapped in <S>..<E> markers.
        ambient_audio_hint: Non-speech scene audio — wrapped in
            <AUDCAP>..<ENDAUDCAP>.
        ref_image_filename: Filename in ComfyUI input/ dir.
        length_s: Accepted for API symmetry; see note above.
        seed: Optional fixed seed; None = random.
        width: Output width (default 512 for vertical 9:16).
        height: Output height (default 960 for vertical 9:16).
        steps: OviVideoGenerator sample_steps (default 50 per WAVE0_OVI.md).

    Returns:
        ComfyUI API-format workflow dict.
    """
    actual_seed = _make_seed(seed)
    del length_s  # length is intrinsic to Ovi; explicit acknowledgement
    text_prompt = _ovi_combined_prompt(
        scene_prompt, dialogue_text, ambient_audio_hint
    )

    return {
        "10": {
            "class_type": "OviEngineLoader",
            "inputs": {
                "model_precision": OVI_MODEL_PRECISION_FP8,
                "cpu_offload": False,
                "device": 0,
            },
        },
        "11": {
            "class_type": "OviAttentionSelector",
            "inputs": {
                "components": ["10", 0],
                "attention_backend": OVI_ATTENTION_BACKEND,
            },
        },
        "12": {
            "class_type": "OviWanComponentLoader",
            "inputs": {
                "engine": ["11", 0],
                "vae_file": OVI_VAE_FILE,
                "umt5_file": OVI_UMT5_FILE,
            },
        },
        "14": {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_filename},
        },
        "16": {
            "class_type": "OviVideoGenerator",
            "inputs": {
                "components": ["12", 0],
                "text_prompt": text_prompt,
                "video_height": height,
                "video_width": width,
                "seed": actual_seed,
                "solver_name": "unipc",
                "sample_steps": steps,
                "shift": OVI_DEFAULT_SHIFT,
                "video_guidance_scale": OVI_DEFAULT_VIDEO_GUIDANCE,
                "audio_guidance_scale": OVI_DEFAULT_AUDIO_GUIDANCE,
                "slg_layer": OVI_DEFAULT_SLG_LAYER,
                "video_negative_prompt": "",
                "audio_negative_prompt": "",
                "first_frame_image": ["14", 0],
            },
        },
        "17": {
            "class_type": "OviLatentDecoder",
            "inputs": {
                "components": ["16", 2],
                "video_latents": ["16", 0],
                "audio_latents": ["16", 1],
            },
        },
        "18": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["17", 0],
                "audio": ["17", 1],
                "fps": float(OVI_FPS),
            },
        },
        "9": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["18", 0],
                "filename_prefix": "alice_ovi",
                "format": "auto",
                "codec": "auto",
            },
        },
    }
