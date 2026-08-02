"""ComfyUI workflow builder for LTX-2.3 image-to-audio-video (I2AV).

LTX-2.3 is Lightricks' joint audio-visual DiT, a single
checkpoint containing DiT transformer + video VAE + audio VAE + vocoder.
NATIVE_SPEECH=true per WAVE0_LTX2.md (architectural evidence + Lightricks
example workflow Gemma-enhancer speech guidance).

Every literal filename / node class name is copied verbatim from
`.planning/phases/09.5-.../WAVE0_LTX2.md` — mismatches produce garbled
output, not errors.

VRAM optimizations:
- Tiled VAE decode (VAEDecodeTiled) as default
- Two-stage sampling with AV latent separation (opt-in)
- GGUF loader support (opt-in)

 — LTX-2.3 upgrade:
- LTX-2.3 GGUF model (Q6_K) with separate Video VAE + Text Projection
- Audio VAE conditioning (LoadAudio + LTXVAudioVAEEncode branch)
- Text projection loader for LTX-2.3 DiT compatibility
"""

from __future__ import annotations

import os as _os
import random

# ── Seed utility ────────────────────────────────────────────────────
# Duplicated from workflows.workflows._make_seed to avoid circular import:
# workflow_ltx2 → workflows → workflow_dialogue → workflow_ltx2
# (workflows.py re-exports ltx2 builders via workflow_dialogue at module tail)

_SEED_MAX = 2**32 - 1


def _make_seed(seed: int | None) -> int:
    """Return provided seed or generate a random one."""
    if seed is not None:
        return seed
    return random.randint(0, _SEED_MAX)

# ── Model filenames + loader constants (from WAVE0_LTX2.md) ────────

LTX2_MODEL_FILE = "ltx-2.3-22b-distilled-1.1-Q6_K.gguf"
"""LTX-2.3 22B distilled DiT in Q6_K GGUF format (17GB).
Downloaded from unsloth/LTX-2.3-GGUF. Loaded via UnetLoaderGGUF."""

LTX2_TEXT_ENCODER_FILE = "gemma_3_12B_it_fp8_e4m3fn.safetensors"
"""Gemma 3 12B fp8 text encoder for LTX-2.3. Fed to LTXAVTextEncoderLoader.
Loaded from models/text_encoders/ (not checkpoints/)."""

LTX2_VAE_FILE = "ltx-2.3-22b-distilled_video_vae.safetensors"
"""Standalone video VAE for LTX-2.3 GGUF path (VAELoader). 1.4GB."""

LTX2_TEXT_PROJECTION_FILE = "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
"""LTX-2.3 text projection / embeddings connectors — maps CLIP embeddings to DiT latent space.
2.2GB. Loaded INTERNALLY by the model (not a separate ComfyUI node).
The LTX-2.3 model loaders handle the embeddings connector automatically.
No LTXAVTextProjectionLoader node exists in ComfyUI-LTXVideo."""
LTX2_AUDIO_VAE_FILE = "ltx-2.3-22b-distilled_audio_vae.safetensors"
"""LTX-2.3 Audio VAE (bf16, 348MB). Encodes audio for Audio VAE conditioning."""

LTX2_FRAME_RATE = 24
"""Frames per second for LTX-2 temporal attention + final mux."""

LTX2_FRAME_STRIDE = 8
"""LTXVImgToVideo `length` input has step=8, min=9; valid frame counts
are of the form 8k+1 (e.g. 9, 17, ..., 121, ..., 241). Builder rounds
length_s*fps to the nearest 8k+1."""

LTX2_DEFAULT_WIDTH = 768
LTX2_DEFAULT_HEIGHT = 768
LTX2_DEFAULT_STEPS = 8  # distilled operating point per LTX-2 README
LTX2_DEFAULT_CFG = 1.0

LTX2_NEGATIVE_PROMPT = (
    "low quality, blurry, static, distorted audio, silence, noise"
)

# ── VRDG sigma schedule () ────────────────────
# Custom sigma schedule for base sampling — replaces KSampler's internal
# sigma computation with the VRDG V5.1 9-step schedule. 9 entries = 9
# DiT passes. Proven from the VRDG upscale subgraph (workflow_ltx2_upscale.py).
LTX2_VRDG_SIGMA_SCHEDULE = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

# ── Lipdub IC-LoRA (Option C / BUG A fix) ──────────────────────────
# Lightricks' purpose-built audio-driven lip-sync LoRA. Downloaded from
# https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub into
# ComfyUI/models/loras/. Applied onto the DiT via LTXICLoRALoaderModelOnly and
# paired with LTXAddVideoICLoRAGuide + LTXVSetAudioRefTokens to replace the weak
# generic audio_adaln path (root cause of BUG A: inaccurate lip sync).
LTX2_LIPDUB_LORA_FILE = "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"

# Node-id namespace for the optional Lipdub IC-LoRA conditioning chain, kept in
# the 320s so it never collides with the base-gen subgraph (300s) or loaders.
_LIPDUB_GUIDE_ID = "320"
_LIPDUB_AUDREF_ID = "321"


def _comfyui_loras_dir() -> str:
    """Resolve ComfyUI models/loras directory (env COMFYUI_DIR or default)."""
    root = _os.environ.get("COMFYUI_DIR") or "/path/to/ComfyUI"
    return _os.path.join(root, "models", "loras")


def _lipdub_lora_present() -> bool:
    """Return True if the Lipdub IC-LoRA checkpoint is on disk.

    Never raises — returns False on missing file / any OSError so callers can
    fall back to the weak generic audio path instead of hard-crashing the job.
    """
    try:
        p = _os.path.join(_comfyui_loras_dir(), LTX2_LIPDUB_LORA_FILE)
        return _os.path.exists(p) and _os.path.getsize(p) > 0
    except OSError:
        return False


def _ltx2_num_frames(length_s: float) -> int:
    """Round length_s seconds (at 24 fps) to the nearest valid LTX-2 frame
    count of the form 8k+1. Accepts float for pre-roll/tail-loss padding.
    Examples: 5s→121, 6.33s→153, 10s→241, 7s→169.

    LTXVImgToVideo `length` input has step=8 min=9, so k>=1.
    """
    raw = length_s * LTX2_FRAME_RATE
    k = round((raw - 1) / LTX2_FRAME_STRIDE)
    if k < 1:
        k = 1
    return LTX2_FRAME_STRIDE * k + 1


def _ltx2_combined_prompt(
    scene_prompt: str,
    dialogue_text: str,
    speakers: list[dict[str, str]] | None = None,
) -> str:
    """Wrap dialogue in natural-language-quoted form for LTX-2 per
    Lightricks/ComfyUI-LTXVideo Gemma enhancer guidance.

    Single-speaker (dialogue_text only, no speakers list):
        "... the character says: 'dialogue here' with confident tone..."

    Multi-speaker (speakers list provided):
        "A tall man says: '...' then a woman replies: '...'"

    Each speaker dict: {"description": "A tall man", "line": "Hey watch out!"}

    When speakers is provided, dialogue_text is ignored.

    This is the NATIVE_SPEECH=true path (WAVE0_LTX2.md evidence chain).
    NOT Ovi's <S>..<E> tag syntax — LTX-2 was trained on natural prose.
    """
    if speakers:
        parts: list[str] = []
        for i, s in enumerate(speakers):
            desc = s["description"]
            line = s["line"]
            if i == 0:
                parts.append(f'{desc} says: "{line}"')
            else:
                parts.append(f'then {desc} replies: "{line}"')
        dialogue_clause = ", ".join(parts)
        return (
            f"{scene_prompt}. {dialogue_clause}, "
            "with clear diction and natural lip sync."
        )
    if not dialogue_text:
        return scene_prompt
    return (
        f"{scene_prompt}. The character speaks, saying: "
        f"\"{dialogue_text}\", with clear diction and natural lip sync."
    )


def _ltx2_loader_nodes(
    model_file: str | None = None,
    text_encoder_device: str = "default",
) -> tuple[dict, str]:
    """LTX-2.3 loader preamble. Returns (nodes_dict, model_node_id).

    The embeddings connector (text projection) is loaded INTERNALLY by the
    model loaders — there is no separate LTXAVTextProjectionLoader node in
    ComfyUI-LTXVideo. The model itself handles the CLIP→DiT mapping.

    Safetensors path (model_file is None or does not end with .gguf):
      CheckpointLoaderSimple (MODEL + video VAE), LTXAVTextEncoderLoader
      (Gemma CLIP overrides hollow default), LTXVAudioVAELoader
      (audio_vae.* keys from same checkpoint).

    GGUF path (model_file ends with .gguf):
      UnetLoaderGGUF for the DiT, LTXAVTextEncoderLoader for Gemma CLIP,
      LTXVAudioVAELoader for audio VAE. VAE loaded via VAELoader from a
      separate safetensors file.

    text_encoder_device: passed to LTXAVTextEncoderLoader node "11" (native
      ComfyUI node comfy_extras/nodes_lt_audio.py). "default" keeps the
      ~12GB Gemma-3-12B text encoder on GPU; "cpu" pins load_device AND
      offload_device to CPU so the encoder NEVER occupies GPU VRAM. Pinning
      to CPU is the deterministic fix for the 1920×1088 KSampler OOM — it
      guarantees the ~12GB text-encoder VRAM is free before the full-res
      sampling pass, rather than relying on ComfyUI smart-memory heuristics.
      Conditioning (CLIPTextEncode nodes 6/7) runs on CPU once per clip;
      the resulting conditioning tensors are moved to GPU for the KSampler.

    Extracted per CONTEXT.md fallback rule 5 to keep builder under 120 LOC.
    """
    effective_model = model_file if model_file is not None else LTX2_MODEL_FILE
    is_gguf = effective_model.endswith(".gguf")

    if is_gguf:
        return (
            {
                "30": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": effective_model},
                },
                "11": {
                    "class_type": "LTXAVTextEncoderLoader",
                    "inputs": {
                        "text_encoder": LTX2_TEXT_ENCODER_FILE,
                        "ckpt_name": LTX2_TEXT_PROJECTION_FILE,
                        "device": text_encoder_device,
                    },
                },
                "12": {
                    "class_type": "LTXVAudioVAELoader",
                    "inputs": {"ckpt_name": LTX2_AUDIO_VAE_FILE},
                },
                "31": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": LTX2_VAE_FILE},
                },
            },
            "30",
        )

    return (
        {
            "10": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": effective_model},
            },
            "11": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": LTX2_TEXT_ENCODER_FILE,
                    "ckpt_name": effective_model,
                    "device": text_encoder_device,
                },
            },
            "12": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": effective_model},
            },
        },
        "10",
    )


def _ltx2_muted_tail(
    video_frames_ref: list,
) -> dict:
    """Video-only tail (no audio decode). Caller muxes Orpheus-TTS.

    Used only when `mute_audio=True` — the mute_audio downgrade path exists
    for future-proofing in case an empirical speech-render test downgrades
    NATIVE_SPEECH from true. WAVE0_LTX2.md verdict is true on architectural
    evidence, so this branch is not the default.
    """
    return {
        "11b": {
            "class_type": "CreateVideo",
            "inputs": {"images": video_frames_ref, "fps": float(LTX2_FRAME_RATE)},
        },
        "9": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11b", 0],
                "filename_prefix": "alice_ltx2",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def _ltx2_audio_tail(
    video_frames_ref: list,
    audio_samples_ref: list,
) -> dict:
    """Audio+video tail: CreateVideo (with audio) → SaveVideo.
    Default NATIVE_SPEECH=true path.

    video_frames_ref: reference to the decoded video frames node output
    audio_samples_ref: reference to the decoded audio samples node output
    """
    return {
        "11b": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": video_frames_ref,
                "audio": audio_samples_ref,
                "fps": float(LTX2_FRAME_RATE),
            },
        },
        "9": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11b", 0],
                "filename_prefix": "alice_ltx2",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def _tiled_or_plain_vae_decode(
    use_tiled: bool,
    samples_ref: list,
    vae_ref: list,
) -> dict:
    """Return a VAEDecodeTiled or VAEDecode node dict.

    Tiled decode is the default (use_tiled=True) for VRAM savings.
    Plain VAEDecode is available via use_tiled_vae=False for compatibility.
    """
    if use_tiled:
        return {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": samples_ref,
                "vae": vae_ref,
                "tile_size": 256,
                "overlap": 64,
                "temporal_size": 16,
                "temporal_overlap": 8,
            },
        }
    return {
        "class_type": "VAEDecode",
        "inputs": {"samples": samples_ref, "vae": vae_ref},
    }


def _ltx2_audio_latent_nodes(
    audio_path: str | None,
    audio_vae_ref: list,
    num_frames: int,
    width: int = LTX2_DEFAULT_WIDTH,
    height: int = LTX2_DEFAULT_HEIGHT,
) -> tuple[dict, list]:
    """Build audio latent nodes for LTX-2.3 workflow.

    When audio_path is provided, creates LoadAudio + LTXVAudioVAEEncode
    nodes that feed real audio through the Audio VAE, followed by
    SolidMask(color=0) + SetLatentNoiseMask to preserve audio conditioning
    stability (VRDG pattern — prevents sampler from denoising audio latent).

    When audio_path is None, creates LTXVEmptyLatentAudio (backward
    compatible with existing workflows that don't use audio conditioning).

    Args:
        audio_path: Filename in ComfyUI input/ for audio conditioning,
            or None for empty latents.
        audio_vae_ref: Reference to the Audio VAE node output (e.g. ["12", 0]).
        num_frames: Frame count for LTXVEmptyLatentAudio (used when audio_path
            is None).
        width: Width for SolidMask (defaults to LTX2_DEFAULT_WIDTH).
        height: Height for SolidMask (defaults to LTX2_DEFAULT_HEIGHT).

    Returns:
        Tuple of (nodes_dict, audio_latent_ref) where audio_latent_ref is
        the node reference to feed into LTXVConcatAVLatent's audio_latent input.
    """
    if audio_path is not None:
        return (
            {
                "50": {
                    "class_type": "LoadAudio",
                    "inputs": {"audio": audio_path},
                },
                "51": {
                    "class_type": "LTXVAudioVAEEncode",
                    "inputs": {
                        "audio": ["50", 0],
                        "audio_vae": audio_vae_ref,
                    },
                },
                # Audio latent noise mask — preserves audio conditioning stability.
                # VRDG applies SolidMask=0 + SetLatentNoiseMask so sampler doesn't
                # denoise the audio latent across sampling steps.
                "52": {
                    "class_type": "SolidMask",
                    "inputs": {"value": 0, "width": width, "height": height},
                },
                "53": {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {"samples": ["51", 0], "mask": ["52", 0]},
                },
            },
            ["53", 0],
        )
    return (
        {
            "17": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": num_frames,
                    "frame_rate": LTX2_FRAME_RATE,
                    "batch_size": 1,
                    "audio_vae": audio_vae_ref,
                },
            },
        },
        ["17", 0],
    )


def _ltx2_lipdub_conditioning(
    positive_ref: list,
    negative_ref: list,
    video_latent_ref: list,
    audio_latent_ref: list,
    vae_ref: list,
    ref_image_ref: list | None,
    use_lipdub: bool,
    audio_path: str | None,
) -> tuple[dict, list, list, list, list]:
    """Optionally strengthen lip sync via the Lipdub IC-LoRA conditioning chain.

    When ``use_lipdub`` is True AND a real audio segment is present, insert:
      1. LTXAddVideoICLoRAGuide (positive, negative, vae, latent=video latent,
         image=ref portrait) — anchors the IC-LoRA to the character frame. Only
         inserted when ``ref_image_ref`` is provided (I2x path).
      2. LTXVSetAudioRefTokens (positive, negative, audio_latent) — attaches the
         encoded audio as reference tokens so the DiT aligns mouth to phonemes.

    The model-side LoRA (LTXICLoRALoaderModelOnly) is applied by the caller and
    arrives here as ``model_ref``; this helper only rewires the conditioning.

    Returns ``(extra_nodes, pos_ref, neg_ref, video_latent_for_concat,
    audio_latent_for_concat)``. When inactive, all refs are returned unchanged and
    ``extra_nodes`` is empty (so the caller's ConcatAVLatent/KSampler wiring is
    byte-identical to the pre-fix weak path).
    """
    extra: dict = {}
    pos, neg = positive_ref, negative_ref
    vid = video_latent_ref
    aud = audio_latent_ref
    if use_lipdub and audio_path is not None:
        if ref_image_ref is not None:
            extra[_LIPDUB_GUIDE_ID] = {
                "class_type": "LTXAddVideoICLoRAGuide",
                "inputs": {
                    "positive": pos,
                    "negative": neg,
                    "vae": vae_ref,
                    "latent": vid,
                    "image": ref_image_ref,
                    "frame_idx": 0,
                    "strength": 1.0,
                    "latent_downscale_factor": 1.0,
                    "crop": "disabled",
                    "use_tiled_encode": False,
                    "tile_size": 256,
                    "tile_overlap": 64,
                },
            }
            pos = [_LIPDUB_GUIDE_ID, 0]
            neg = [_LIPDUB_GUIDE_ID, 1]
            vid = [_LIPDUB_GUIDE_ID, 2]
        extra[_LIPDUB_AUDREF_ID] = {
            "class_type": "LTXVSetAudioRefTokens",
            "inputs": {"positive": pos, "negative": neg, "audio_latent": aud},
        }
        pos = [_LIPDUB_AUDREF_ID, 0]
        neg = [_LIPDUB_AUDREF_ID, 1]
        aud = [_LIPDUB_AUDREF_ID, 2]  # frozen_audio (noise_mask=0)
    return extra, pos, neg, vid, aud


# LTX-2 video VAE spatial downscale is 32, but the core ComfyUI `LatentUpscale`
# node sizes the latent as `width // 8` (its built-in convention for standard
# VAEs). Net decode factor = 32 / 8 = 4, so a `LatentUpscale` target of W
# actually decodes to 4W pixels. To make the Path A bislerp upscale (node 40)
# decode to the *requested* per-clip resolution, divide the target by 4.
# (Discovered during 09.9-13 Path A e2e: clip_001 decoded to 3840x2176 from a
# 960x544 target — exactly 4x. The LTX-2 KSampler itself is 1:1, so only the
# extra LatentUpscale node needs this correction.)
LTX2_VAE_SPATIAL_FACTOR = 32
_COMFYUI_LATENT_DIVISOR = 8


def _ltx2_latent_upscale_target(px: int) -> int:
    """Map a desired DECODED pixel dimension to the LatentUpscale `width`/`height`.

    LatentUpscale sizes the latent as `px // 8`; LTX-2 VAE decodes at x32, so the
    decoded pixels = (target // 8) * 32 = target * 4. Passing `px // 4` makes the
    decoded output equal `px`.
    """
    return px * _COMFYUI_LATENT_DIVISOR // LTX2_VAE_SPATIAL_FACTOR  # == px // 4


# ── VRDG sigma sampler helper () ──────────────
def _ltx2_vrdg_sampler_nodes(
    model_ref: list,
    pos_ref: list,
    neg_ref: list,
    latent_ref: list,
    seed: int,
) -> dict:
    """Return the VRDG V5.1 sigma sampler node chain (replaces KSampler).

    When ``use_vrdg_sigmas=True``, this replaces the single KSampler node "3"
    with: CFGGuider + RandomNoise + KSamplerSelect + ManualSigmas +
    SamplerCustomAdvanced. 9 sigmas = 9 DiT passes.
    """
    return {
        "3a": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": model_ref,
                "positive": pos_ref,
                "negative": neg_ref,
                "cfg": LTX2_DEFAULT_CFG,
            },
        },
        "3b": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
                "noise_index": 0,
                "generator": "fixed",
            },
        },
        "3c": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "3d": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": LTX2_VRDG_SIGMA_SCHEDULE},
        },
        "3e": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "guider": ["3a", 0],
                "noise": ["3b", 0],
                "sampler": ["3c", 0],
                "sigmas": ["3d", 0],
                "latent_image": latent_ref,
            },
        },
    }


# ── Base-gen latent sub-graph (Plan 09.9-16, combined graph) ────────
# Node-id namespace for the base-gen latent sub-graph consumed by the combined
# upscale builder. Deliberately in the 300s so it never collides with the VRGDG
# upscale sub-graph ids (187..234) or the shared loader ids (1/11/12/31/60).
_BASE_POS_ID = "301"
_BASE_NEG_ID = "302"
_BASE_COND_ID = "303"
_BASE_IMAGE_ID = "304"
_BASE_I2V_ID = "305"
_BASE_T2V_ID = "306"
_BASE_AUDIO_LOAD_ID = "307"
_BASE_AUDIO_ENC_ID = "308"
_BASE_AUDIO_EMPTY_ID = "309"
_BASE_CONCAT_ID = "310"
_BASE_KSAMPLER_ID = "311"


def _ltx2_base_latent_subgraph(
    scene_prompt: str,
    dialogue_text: str,
    ref_image_filename: str | None,
    num_frames: int,
    *,
    width: int,
    height: int,
    seed: int,
    model_ref: list,
    clip_ref: list,
    vae_ref: list,
    audio_vae_ref: list,
    audio_path: str | None = None,
    neg_suffix: str = "",
    speakers: list[dict[str, str]] | None = None,
    use_lipdub: bool = False,
    use_vrdg_sigmas: bool = False,
) -> tuple[dict, list]:
    """Build the base-gen nodes up to the joint AV latent, WITHOUT any decode tail.

    Plan 09.9-16 (Option B, latent→latent): the combined upscale builder chains
    base generation and the VRGDG upscale sub-graph inside ONE ComfyUI job. This
    helper emits only the base-gen nodes that produce the joint AV latent — the
    KSampler output — which the upscale sub-graph's LTXVSeparateAVLatent consumes
    directly (verified against the VRGDG reference: its 'LTX2' subgraph
    `denoised_output` = SamplerCustomAdvanced output feeds the 'Model upscale'
    `av_latent` input with NO decode/re-encode in between). There is deliberately
    NO LTXVSeparateAVLatent / VAE decode / MP4 tail here: decoding then
    re-encoding is exactly the ghosting source this plan removes.

    Loaders are NOT created here — the caller passes shared loader refs so the
    base DiT is loaded ONCE for the whole job (base sample + upscale refine share
    the resident DiT). This is the ONE-DiT-load invariant of Plan 09.9-16.

    Args:
        scene_prompt: Visual scene description (base-gen positive prompt).
        dialogue_text: Speech content, wrapped per NATIVE_SPEECH prose style.
        ref_image_filename: Portrait in ComfyUI input/ (I2x path), or None (T2x).
        num_frames: Frame count (8k+1), already rounded by the caller.
        width, height: LOW-RES base generation resolution (e.g. 960x544).
        seed: Fixed sampler seed (already resolved via _make_seed).
        model_ref: Shared DiT model output ref (e.g. ["1", 0]).
        clip_ref: Shared text-encoder CLIP output ref (e.g. ["11", 0]).
        vae_ref: Shared video VAE output ref (e.g. ["31", 0]).
        audio_vae_ref: Shared Audio VAE output ref (e.g. ["12", 0]).
        audio_path: Vocals stem filename in ComfyUI input/ for Audio VAE
            conditioning (D-09), or None for an empty audio latent.
        use_lipdub: When True AND audio_path is present, insert the Lipdub
            IC-LoRA conditioning chain (LTXAddVideoICLoRAGuide + LTXVSetAudioRefTokens)
            to strongly bind lip sync. The model-side LoRA is applied by the caller
            and arrives here as ``model_ref``. Gated by the caller (checkpoint present).
        neg_suffix: Appended to LTX2_NEGATIVE_PROMPT.
        speakers: Optional multi-speaker list (see _ltx2_combined_prompt).

    Returns:
        (nodes_dict, joint_av_latent_ref) where joint_av_latent_ref is the
        base-gen KSampler output — the joint AV latent (NestedTensor) fed to the
        upscale sub-graph's LTXVSeparateAVLatent.av_latent.
    """
    combined_prompt = _ltx2_combined_prompt(scene_prompt, dialogue_text, speakers=speakers)
    negative_prompt = LTX2_NEGATIVE_PROMPT + neg_suffix

    nodes: dict = {
        _BASE_POS_ID: {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_ref, "text": combined_prompt},
        },
        _BASE_NEG_ID: {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_ref, "text": negative_prompt},
        },
        _BASE_COND_ID: {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": [_BASE_POS_ID, 0],
                "negative": [_BASE_NEG_ID, 0],
                "frame_rate": LTX2_FRAME_RATE,
            },
        },
    }

    # Mode-specific latent seed: I2x (portrait re-condition) vs T2x (empty latent).
    if ref_image_filename is not None:
        nodes[_BASE_IMAGE_ID] = {
            "class_type": "LoadImage",
            "inputs": {"image": ref_image_filename},
        }
        # LTXVPreprocess — compress ref image through VP8 CRF 33 before I2V.
        # Matches training data distribution (LTX-2 trained on compressed frames).
        nodes["14b"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": [_BASE_IMAGE_ID, 0], "img_compression": 33},
        }
        nodes[_BASE_I2V_ID] = {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive": [_BASE_COND_ID, 0],
                "negative": [_BASE_COND_ID, 1],
                "vae": vae_ref,
                "image": ["14b", 0],
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
                "strength": 1.0,
            },
        }
        positive_ref = [_BASE_I2V_ID, 0]
        negative_ref = [_BASE_I2V_ID, 1]
        video_latent_ref = [_BASE_I2V_ID, 2]
    else:
        nodes[_BASE_T2V_ID] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        }
        positive_ref = [_BASE_COND_ID, 0]
        negative_ref = [_BASE_COND_ID, 1]
        video_latent_ref = [_BASE_T2V_ID, 0]

    # Audio latent (D-09): real vocals via Audio VAE, else empty audio latent.
    if audio_path is not None:
        nodes[_BASE_AUDIO_LOAD_ID] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": audio_path},
        }
        nodes[_BASE_AUDIO_ENC_ID] = {
            "class_type": "LTXVAudioVAEEncode",
            "inputs": {"audio": [_BASE_AUDIO_LOAD_ID, 0], "audio_vae": audio_vae_ref},
        }
        # Audio latent noise mask — preserves audio conditioning stability.
        # VRDG applies SolidMask=0 + SetLatentNoiseMask so sampler doesn't
        # denoise the audio latent across sampling steps.
        nodes["52"] = {
            "class_type": "SolidMask",
            "inputs": {"value": 0, "width": width, "height": height},
        }
        nodes["53"] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": [_BASE_AUDIO_ENC_ID, 0], "mask": ["52", 0]},
        }
        audio_latent_ref: list = ["53", 0]
    else:
        nodes[_BASE_AUDIO_EMPTY_ID] = {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": num_frames,
                "frame_rate": LTX2_FRAME_RATE,
                "batch_size": 1,
                "audio_vae": audio_vae_ref,
            },
        }
        audio_latent_ref = [_BASE_AUDIO_EMPTY_ID, 0]

    # Optionally strengthen lip sync via the Lipdub IC-LoRA conditioning chain.
    ref_image_ref = [_BASE_IMAGE_ID, 0] if ref_image_filename is not None else None
    lipdub_nodes, pos_ref, neg_ref, vid_for_concat, aud_for_concat = _ltx2_lipdub_conditioning(
        positive_ref, negative_ref, video_latent_ref, audio_latent_ref,
        vae_ref, ref_image_ref, use_lipdub, audio_path,
    )
    nodes.update(lipdub_nodes)

    # Concat AV latent → base KSampler. The KSampler output is the joint AV
    # latent carried directly into the upscale sub-graph (no decode).
    nodes[_BASE_CONCAT_ID] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {"video_latent": vid_for_concat, "audio_latent": aud_for_concat},
    }

    if use_vrdg_sigmas:
        # Replace KSampler with VRDG V5.1 sigma chain (9-step schedule)
        vrdg_nodes = _ltx2_vrdg_sampler_nodes(
            model_ref, pos_ref, neg_ref, [_BASE_CONCAT_ID, 0], seed,
        )
        nodes.update(vrdg_nodes)
        return nodes, ["3e", 0]
    else:
        nodes[_BASE_KSAMPLER_ID] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_ref,
                "positive": pos_ref,
                "negative": neg_ref,
                "latent_image": [_BASE_CONCAT_ID, 0],
                "seed": seed,
                "steps": LTX2_DEFAULT_STEPS,
                "cfg": LTX2_DEFAULT_CFG,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        }
        return nodes, [_BASE_KSAMPLER_ID, 0]


def build_ltx2_workflow(
    scene_prompt: str,
    dialogue_text: str,
    ref_image_filename: str | None,
    length_s: int = 10,
    seed: int | None = None,
    *,
    width: int = LTX2_DEFAULT_WIDTH,
    height: int = LTX2_DEFAULT_HEIGHT,
    steps: int = LTX2_DEFAULT_STEPS,
    mute_audio: bool = False,
    neg_suffix: str = "",
    speakers: list[dict[str, str]] | None = None,
    use_tiled_vae: bool = True,
    use_two_stage: bool = False,
    model_file: str | None = None,
    base_width: int = 512,
    base_height: int = 512,
    audio_path: str | None = None,
    text_encoder_device: str = "default",
    use_lipdub: bool = True,
    use_vrdg_sigmas: bool = False,
    output_audio_filename: str | None = None,
) -> dict:
    """LTX-2.3 general 4-mode workflow builder. Returns ComfyUI API dict.

    Supports I2AV (ref + dialogue), I2V (ref, no dialogue), T2AV (no ref +
    dialogue), T2V (no ref, no dialogue). Mode is determined by the caller
    (ref_image_filename presence controls I2x vs T2x path).

    VRAM optimizations:
      - use_tiled_vae (default True): VAEDecodeTiled instead of VAEDecode
      - use_two_stage (default False): low-res base sample → upscale → refine
      - model_file (default None): GGUF checkpoint path for UnetLoaderGGUF

    Audio VAE conditioning ():
      - audio_path (default None): path to audio file in ComfyUI input/ for
        Audio VAE conditioning. When provided, feeds real audio through the
        Audio VAE instead of empty latents.

    Args:
        scene_prompt: Visual description of the scene.
        dialogue_text: Speech content. Wrapped in natural-language-quoted
            form per LTX-2 prompt style (NATIVE_SPEECH=true, WAVE0_LTX2.md).
            Empty string → no dialogue wrapping in prompt.
        ref_image_filename: Filename in ComfyUI input/ dir (I2x path), or
            None for T2x path.
        length_s: Target duration in seconds; frame count rounded to 8k+1.
        seed: Optional fixed seed; None = random.
        width: Output width (step=32, default 768).
        height: Output height (step=32, default 768).
        steps: Sampler steps (default 8 for distilled checkpoint).
        mute_audio: If True, disconnect audio decoding — caller muxes VO
            externally or post-processes to add silent track.
        neg_suffix: String appended to LTX2_NEGATIVE_PROMPT in node "7".
            Pass NEG_SUFFIX_6TERM (from server_dialogue) when suppress_text=True.
            Default empty string leaves the base negative prompt unchanged.
        use_tiled_vae: Use VAEDecodeTiled (default True, saves VRAM).
        use_two_stage: Two-stage sampling for lower VRAM peak.
        model_file: GGUF model path for UnetLoaderGGUF (opt-in).
        base_width: Base resolution width for two-stage first pass.
        base_height: Base resolution height for two-stage first pass.
        audio_path: Path to audio file in ComfyUI input/ directory for Audio
            VAE conditioning. When provided, feeds real audio through Audio VAE
            instead of empty latents. None = use LTXVEmptyLatentAudio (current
            behavior, backward compatible).
        use_lipdub: When True AND a real audio segment is present AND the Lipdub
            IC-LoRA checkpoint is on disk, apply the strong lip-sync path
            (LTXICLoRALoaderModelOnly + LTXAddVideoICLoRAGuide + LTXVSetAudioRefTokens).
            Defaults to True; falls back to the weak generic audio path otherwise.
        text_encoder_device: "default" (Gemma text encoder on GPU) or "cpu"
            (encoder pinned to CPU, freeing ~12GB GPU VRAM for the full-res
            KSampler). Set "cpu" at high resolution (e.g. 1920x1088) to avoid
            the DiT KSampler OOM. See _ltx2_loader_nodes docstring.
        output_audio_filename: Filename in ComfyUI input/ to wire directly to
            the CreateVideo output, bypassing the AudioVAE decode. When provided,
            the original audio file (not the synthetic AudioVAE reconstruction) is
            muxed into the final video. The AudioVAE encode path still guides
            lip-sync visuals during generation. None = use AudioVAE decode (current
            behavior, backward compatible).

    Returns:
        ComfyUI API-format workflow dict.
    """
    actual_seed = _make_seed(seed)
    num_frames = _ltx2_num_frames(length_s)
    combined_prompt = _ltx2_combined_prompt(scene_prompt, dialogue_text, speakers=speakers)
    negative_prompt = LTX2_NEGATIVE_PROMPT + neg_suffix

    # ── Loader nodes (safetensors or GGUF) ──────────────────────────
    loader_nodes, model_node_id = _ltx2_loader_nodes(model_file, text_encoder_device)
    is_gguf = model_node_id == "30"

    # Effective model reference for KSampler: use the raw model node.
    # The embeddings connector is loaded internally by the model loaders
    # (no separate LTXAVTextProjectionLoader node).
    model_ref_for_sampler = [model_node_id, 0]

    # VAE node reference: GGUF path uses dedicated VAELoader ("31"),
    # safetensors path extracts VAE from CheckpointLoaderSimple ("10", output 2)
    vae_node_ref: list = ["31", 0] if is_gguf else ["10", 2]

    # ── Lipdub IC-LoRA (Option C / BUG A) ───────────────────────────
    # Apply the strong lip-sync LoRA onto the DiT when audio conditioning is
    # active and the checkpoint is present. The conditioning-side guide +
    # audio-ref-tokens are inserted later via _ltx2_lipdub_conditioning.
    lipdub_active = use_lipdub and audio_path is not None and _lipdub_lora_present()
    if lipdub_active:
        loader_nodes["30b"] = {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": [model_node_id, 0],
                "lora_name": LTX2_LIPDUB_LORA_FILE,
                "strength_model": 1.0,
            },
        }
        model_ref_for_sampler = ["30b", 0]

    # ── Conditioning nodes (shared I2x/T2x) ────────────────────────
    shared_nodes: dict = {
        **loader_nodes,
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": combined_prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": negative_prompt},
        },
        "15": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "frame_rate": LTX2_FRAME_RATE,
            },
        },
    }

    # ── Mode-specific nodes ─────────────────────────────────────────
    if ref_image_filename is not None:
        # I2x path: LoadImage (14) + LTXVPreprocess (14b) + LTXVImgToVideo (16)
        # KSampler/ConcatAV reference node "16" outputs
        first_pass_width = base_width if use_two_stage else width
        first_pass_height = base_height if use_two_stage else height

        mode_nodes: dict = {
            "14": {
                "class_type": "LoadImage",
                "inputs": {"image": ref_image_filename},
            },
            # LTXVPreprocess — compress ref image through VP8 CRF 33 before I2V.
            # Matches training data distribution (LTX-2 trained on compressed frames).
            # VRDG V5.1 uses this node; missing it causes suboptimal conditioning.
            "14b": {
                "class_type": "LTXVPreprocess",
                "inputs": {"image": ["14", 0], "img_compression": 33},
            },
            "16": {
                "class_type": "LTXVImgToVideo",
                "inputs": {
                    "positive": ["15", 0],
                    "negative": ["15", 1],
                    "vae": vae_node_ref,
                    "image": ["14b", 0],
                    "width": first_pass_width,
                    "height": first_pass_height,
                    "length": num_frames,
                    "batch_size": 1,
                    "strength": 1.0,
                },
            },
        }
        positive_ref = ["16", 0]
        negative_ref = ["16", 1]
        video_latent_ref = ["16", 2]
    else:
        # T2x path: EmptyLTXVLatentVideo (13)
        # KSampler/ConcatAV reference node "15" outputs
        first_pass_width = base_width if use_two_stage else width
        first_pass_height = base_height if use_two_stage else height

        mode_nodes = {
            "13": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": first_pass_width,
                    "height": first_pass_height,
                    "length": num_frames,
                    "batch_size": 1,
                },
            },
        }
        positive_ref = ["15", 0]
        negative_ref = ["15", 1]
        video_latent_ref = ["13", 0]

    # ── Common tail ─────────────────────────────────────────────────
    # Build audio latent nodes (real audio or empty, depending on audio_path)
    audio_latent_nodes, audio_latent_ref = _ltx2_audio_latent_nodes(
        audio_path, ["12", 0], num_frames, width=width, height=height
    )

    # Optionally strengthen lip sync via the Lipdub IC-LoRA conditioning chain.
    ref_image_ref_for_builder = ["14", 0] if ref_image_filename is not None else None
    lipdub_extra, pos_ref, neg_ref, vid_ref, aud_ref = _ltx2_lipdub_conditioning(
        positive_ref, negative_ref, video_latent_ref, audio_latent_ref,
        vae_node_ref, ref_image_ref_for_builder, lipdub_active, audio_path,
    )

    if use_two_stage:
        # Two-stage (Path A, Plan 09.9-13): sample at base res → separate AV →
        # bislerp upscale (node 40) → decode node 40 directly. The full-res
        # refine KSampler (former node 41) is REMOVED because it OOMs the 48GB
        # card at high resolution (high-res-oom: node 41
        # torch.OutOfMemoryError). Node 42 decodes the node-40 bislerp latent;
        # the final 1920×1080 frame is produced by the ffmpeg stitch scale
        # branch, so only GENERATION res drops (delivered res stays 1080p).
        common_tail: dict = {
            **audio_latent_nodes,
            "18": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": vid_ref,
                    "audio_latent": aud_ref,
                },
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": model_ref_for_sampler,
                    "positive": pos_ref,
                    "negative": neg_ref,
                    "latent_image": ["18", 0],
                    "seed": actual_seed,
                    "steps": steps,
                    "cfg": LTX2_DEFAULT_CFG,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            # Separate AV latents BEFORE upscaling (audio lane untouched)
            "19": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["3", 0]},
            },
            # Decode audio immediately from the base-resolution separation
            "20": {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {"samples": ["19", 1], "audio_vae": ["12", 0]},
            },
            # Upscale video latent only (audio bypasses this) via bislerp
            # (Path A, Plan 09.9-13): node 40's LatentUpscale is decoded
            # DIRECTLY by node 42. The full-res refine KSampler (former node 41)
            # is REMOVED — it OOMs the 48GB card at high resolution
            # (high-res-oom debug: node 41 torch.OutOfMemoryError).
            "40": {
                "class_type": "LatentUpscale",
                "inputs": {
                    "samples": ["19", 0],
                    # Path A bislerp upscale: decode to the *requested* per-clip
                    # resolution (width x height), not 4x of it. LTX-2 VAE decodes
                    # at x32 but LatentUpscale sizes the latent as width//8, so the
                    # target must be divided by 4 — see _ltx2_latent_upscale_target.
                    "width": _ltx2_latent_upscale_target(width),
                    "height": _ltx2_latent_upscale_target(height),
                    "upscale_method": "bislerp",
                    "crop": "disabled",
                },
            },
            # Decode the bislerp-upscaled latent (node 40) directly — no refine
            # KSampler. Final 1080p is produced by the ffmpeg stitch scale branch
            # (mv_vram.py:_resolution_filter), not by native 1080p generation.
            "42": _tiled_or_plain_vae_decode(use_tiled_vae, ["40", 0], vae_node_ref),
        }
        # When VRDG sigmas are enabled, replace KSampler (node "3") with the
        # CFGGuider+RandomNoise+KSamplerSelect+ManualSigmas+SamplerCustomAdvanced
        # chain (nodes "3a"-"3e"). Output latent comes from "3e" not "3".
        if use_vrdg_sigmas:
            vrdg_nodes = _ltx2_vrdg_sampler_nodes(
                model_ref_for_sampler, pos_ref, neg_ref, ["18", 0], actual_seed,
            )
            common_tail.update(vrdg_nodes)
            common_tail["19"]["inputs"]["av_latent"] = ["3e", 0]
            del common_tail["3"]
        video_frames_ref: list = ["42", 0]
        audio_samples_ref: list = ["20", 0]
    else:
        # Single-stage (default): sample at target resolution
        common_tail = {
            **audio_latent_nodes,
            "18": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": vid_ref,
                    "audio_latent": aud_ref,
                },
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": model_ref_for_sampler,
                    "positive": pos_ref,
                    "negative": neg_ref,
                    "latent_image": ["18", 0],
                    "seed": actual_seed,
                    "steps": steps,
                    "cfg": LTX2_DEFAULT_CFG,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "19": {
                "class_type": "LTXVSeparateAVLatent",
                # Input name is `av_latent` per nodes_lt.py:662 (io.Latent.Input("av_latent"))
                # NOT `concat` — WAVE0_LTX2.md's workflow blueprint was wrong on this.
                "inputs": {"av_latent": ["3", 0]},
            },
            "8": _tiled_or_plain_vae_decode(use_tiled_vae, ["19", 0], vae_node_ref),
            "20": {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {"samples": ["19", 1], "audio_vae": ["12", 0]},
            },
        }
        # When VRDG sigmas are enabled, replace KSampler (node "3") with the
        # CFGGuider+RandomNoise+KSamplerSelect+ManualSigmas+SamplerCustomAdvanced
        # chain (nodes "3a"-"3e"). Output latent comes from "3e" not "3".
        if use_vrdg_sigmas:
            vrdg_nodes = _ltx2_vrdg_sampler_nodes(
                model_ref_for_sampler, pos_ref, neg_ref, ["18", 0], actual_seed,
            )
            common_tail.update(vrdg_nodes)
            common_tail["19"]["inputs"]["av_latent"] = ["3e", 0]
            del common_tail["3"]
        video_frames_ref = ["8", 0]
        audio_samples_ref = ["20", 0]

    wf: dict = {**shared_nodes, **mode_nodes, **common_tail, **lipdub_extra}

    # ── Output tail ─────────────────────────────────────────────────
    if mute_audio:
        wf.update(_ltx2_muted_tail(video_frames_ref))
    elif output_audio_filename is not None:
        # Bypass AudioVAE decode — wire original audio directly to output.
        # AudioVAE encode path still guides lip-sync visuals during generation.
        # (Same pattern as build_ltx2_combined_workflow in workflow_ltx2_upscale.py)
        wf["335"] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": output_audio_filename},
        }
        wf["11b"] = {
            "class_type": "CreateVideo",
            "inputs": {
                "images": video_frames_ref,
                "audio": ["335", 0],
                "fps": float(LTX2_FRAME_RATE),
            },
        }
        wf["9"] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11b", 0],
                "filename_prefix": "alice_ltx2",
                "format": "auto",
                "codec": "auto",
            },
        }
    else:
        wf.update(_ltx2_audio_tail(video_frames_ref, audio_samples_ref))
    return wf


def build_ltx2_i2av_workflow(
    scene_prompt: str,
    dialogue_text: str,
    ref_image_filename: str,
    length_s: int = 10,
    seed: int | None = None,
    *,
    width: int = LTX2_DEFAULT_WIDTH,
    height: int = LTX2_DEFAULT_HEIGHT,
    steps: int = LTX2_DEFAULT_STEPS,
    mute_audio: bool = False,
    use_tiled_vae: bool = True,
    use_two_stage: bool = False,
    model_file: str | None = None,
    base_width: int = 512,
    base_height: int = 512,
) -> dict:
    """Backward-compat alias for I2AV callers. Use build_ltx2_workflow for new callers.

    Eight caller scripts (alice_parody_v5_gen.py, sy_animation_gen.py,
    sy_animation_pov_gen.py, etc.) import this by name — do NOT remove.
    Delegates to build_ltx2_workflow with ref_image_filename passed through.
    """
    return build_ltx2_workflow(
        scene_prompt=scene_prompt,
        dialogue_text=dialogue_text,
        ref_image_filename=ref_image_filename,
        length_s=length_s,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        mute_audio=mute_audio,
        use_tiled_vae=use_tiled_vae,
        use_two_stage=use_two_stage,
        model_file=model_file,
        base_width=base_width,
        base_height=base_height,
        audio_path=None,
    )
