"""ComfyUI workflow builder for the LTX-2.3 Path B generative 2x spatial upscaler.

Plan 09.9-16 — latent→latent handoff (Option B). The base-gen joint AV latent
flows DIRECTLY into the VRGDG 'Model upscale' sub-graph inside ONE chained
ComfyUI job. There is NO LoadVideo→VAEEncode MP4 round-trip: decoding then
re-encoding the base clip produced a latent the ~3-sigma refiner could not
cleanly re-condition → temporal ghosting (09.9-16-CONTEXT). VRGDG never decodes
— its 'LTX2' subgraph `denoised_output` (the SamplerCustomAdvanced output) feeds
the 'Model upscale' `av_latent` input directly, verified against the reference
174-node workflow. The base LTX-2.3 DiT is loaded ONCE and shared by the base
sample and the upscale refine (no inter-job model unload).

Plan 09.9-15 — adopt the VRGameDevGirl (VRGDG) reference upscale sub-graph
(D-01/D-02). We load the VRGDG UI-format workflow JSON at runtime, extract the
10-node 'Model upscale' sub-graph, re-home + re-wire it, and force the locked
params (defense-in-depth). All 10 node classes are standard ComfyUI core +
ComfyUI-LTXVideo — no VRGDG custom nodes (D-03). Locked params (D-04..D-09):
sigmas "0.909375, 0.725, 0.421875, 0.0" (full denoise, fixes pixelation); sampler
"euler" (not euler_cfg_pp); cfg 1.0; LTXVImgToVideoInplace strength=1.0
bypass=False (re-condition on portrait; never LTXVImgToVideoConditionOnly — the
haze bug); LTXVCropGuides present; audio latent carried through (D-09).
"""

from __future__ import annotations

import json as _json
import logging
import sys as _sys
from enum import Enum
from pathlib import Path as _Path

logger = logging.getLogger(__name__)

# Make gpu-manager `src` + project `scripts` importable (pipeline or test cwd).
_src_dir = _Path(__file__).resolve().parent
_gpu_mgr = _src_dir.parent
_proj_root = _gpu_mgr.parent
_scripts = _proj_root / "scripts"
for _p in (str(_gpu_mgr), str(_scripts)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from workflows.workflow_ltx2 import (
    LTX2_AUDIO_VAE_FILE,
    LTX2_DEFAULT_CFG,
    LTX2_FRAME_RATE,
    LTX2_LIPDUB_LORA_FILE,
    LTX2_MODEL_FILE,
    LTX2_NEGATIVE_PROMPT,
    LTX2_TEXT_ENCODER_FILE,
    LTX2_TEXT_PROJECTION_FILE,
    LTX2_VAE_FILE,
    _lipdub_lora_present,
    _ltx2_base_latent_subgraph,
    _ltx2_num_frames,
    _make_seed,
)
from pipeline.mv_mvconst import (  # noqa: E402
    LTX2_UPSCALE_REFINEMENT_SIGMAS,
    LTXV_TILED_VAE_OVERLAP,
    UPSCALE_MODEL_FILENAME,
)


class UpscaleSampler(str, Enum):
    """Sampler for the VRGDG upscale refine pass (D-05)."""

    EULER = "euler"


INPLACE_STRENGTH = 1.0
INPLACE_BYPASS = False
UPSCALE_FACTOR = 2

VRGDG_WORKFLOW_PATH = _src_dir / "vrgdg_upscale_workflow.json"
VRGDG_NODE_MAP_PATH = _src_dir / "vrgdg_upscale_node_map.json"

_SUBGRAPH_NAME = "Model upscale"
_SUBNODE_INPUT_ID = -10
_UPSCALE_CLASS_TYPES = frozenset(
    {
        "LTXVSeparateAVLatent",
        "LTXVCropGuides",
        "LTXVLatentUpsampler",
        "LTXVImgToVideoInplace",
        "LTXVConcatAVLatent",
        "SamplerCustomAdvanced",
        "KSamplerSelect",
        "CFGGuider",
        "RandomNoise",
        "ManualSigmas",
    }
)

_FORCE = object()

# VRGDG subgraph input slot -> our external node ref. Slot 6 (bypass): _FORCE.
def _slot_ref(slot: int, model_ref: list | None = None) -> object:
    # Slot 0 (model) is normally the raw DiT node "1"; when the Lipdub IC-LoRA
    # is active it points at the LoRA-loaded model node instead so BOTH the
    # base-gen KSampler and the VRGDG refine CFGGuider share the lip-sync LoRA.
    return {
        0: model_ref if model_ref is not None else ["1", 0],
        1: ["15", 0], 2: ["15", 1], 3: ["60", 0],
        4: ["31", 0], 5: ["14", 0], 6: _FORCE, 7: ["54", 0],
    }.get(slot)

# VRGDG -> our model filenames (D-10). Applied defensively during extraction.
_VRGDG_FILENAME_MAP = {
    "LTX-2.3-22B-distilled-1.1-Q6_K.gguf": LTX2_MODEL_FILE,
    "LTX23_video_vae_bf16.safetensors": LTX2_VAE_FILE,
    "LTX23_audio_vae_bf16.safetensors": LTX2_AUDIO_VAE_FILE,
    "gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors": LTX2_TEXT_ENCODER_FILE,
    "ltx-2.3_text_projection_bf16.safetensors": LTX2_TEXT_PROJECTION_FILE,
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": UPSCALE_MODEL_FILENAME,
}

LTX2_UPSCALE_POSITIVE_PROMPT = (
    "A high quality, detailed, sharp upscaled video, preserving the original "
    "scene, subject, and motion exactly."
)


def _load_vrgdg_workflow() -> dict:
    """Read the VRGDG UI-format workflow JSON. Raises on missing/parse failure."""
    if not VRGDG_WORKFLOW_PATH.exists():
        raise FileNotFoundError(f"VRGDG workflow JSON missing: {VRGDG_WORKFLOW_PATH}")
    with open(VRGDG_WORKFLOW_PATH, encoding="utf-8") as f:
        return _json.load(f)


def _find_upscale_subgraph(raw: dict) -> dict | None:
    """Return the 'Model upscale' subgraph dict, or a top-level fallback."""
    subs = (raw.get("definitions") or {}).get("subgraphs") or []
    for s in subs:
        if s.get("name") == _SUBGRAPH_NAME:
            return s
    nodes = raw.get("nodes")
    if nodes:
        return {"nodes": nodes, "links": raw.get("links", [])}
    return None


def _remap_filenames(inputs: dict) -> None:
    """Replace any VRGDG model filename with our constant (D-10, defense-in-depth)."""
    for k, v in list(inputs.items()):
        if isinstance(v, str) and v in _VRGDG_FILENAME_MAP:
            inputs[k] = _VRGDG_FILENAME_MAP[v]


def _extract_upscale_subgraph(raw: dict, model_ref: list | None = None) -> dict:
    """Extract + re-home the 10 upscale nodes from the VRGDG workflow.

    Converts UI-format subgraph nodes to a ComfyUI API dict, resolves internal
    links to new ids, and rewires subgraph-input links to our external nodes via
    ``_slot_ref``. Returns {node_id: {"class_type", "inputs"}}.

    ``model_ref`` is forwarded to ``_slot_ref`` so the VRGDG refine CFGGuider
    draws the (optionally LoRA-loaded) DiT when the Lipdub IC-LoRA is active.
    """
    sg = _find_upscale_subgraph(raw)
    if sg is None:
        raise RuntimeError("VRGDG workflow has no 'Model upscale' subgraph and no top-level nodes")
    nodes = sg.get("nodes", [])
    links = sg.get("links", [])

    lmap: dict = {}
    if links and isinstance(links[0], dict):
        for L in links:
            lmap[L["id"]] = (L["origin_id"], L["origin_slot"])
    elif links:
        for L in links:
            lmap[L[0]] = (L[1], L[2])

    up_nodes = [n for n in nodes if n.get("type") in _UPSCALE_CLASS_TYPES]
    if len(up_nodes) != 10:
        raise RuntimeError(f"VRGDG upscale subgraph must contain exactly 10 upscale nodes, found {len(up_nodes)}")

    subgraph: dict = {}
    for n in up_nodes:
        nid = str(n["id"])
        inputs: dict = {}
        for i in n.get("inputs", []):
            link = i.get("link")
            if link is None:
                continue  # widget input — forced by the builder
            origin = lmap.get(link)
            if origin is None:
                continue
            oid, oslot = origin
            if oid == _SUBNODE_INPUT_ID:
                ref = _slot_ref(oslot, model_ref=model_ref)
                if ref is _FORCE:
                    continue
                if ref is None:
                    raise RuntimeError(f"Unmapped VRGDG subgraph input slot {oslot} on {n['type']}")
                inputs[i["name"]] = ref
            else:
                inputs[i["name"]] = [str(oid), oslot]
        subgraph[nid] = {"class_type": n["type"], "inputs": inputs}
        _remap_filenames(inputs)

    return subgraph


def _force_locked_params(subgraph: dict, actual_seed: int, av_latent_ref: list) -> None:
    """Force the locked upscale params (defense-in-depth) + ban forbidden nodes.

    ``av_latent_ref`` is the base-gen joint AV latent output (the KSampler output
    of ``_ltx2_base_latent_subgraph``, e.g. ["311", 0]). It is wired into the
    upscale sub-graph's input-side LTXVSeparateAVLatent so the base latent flows
    latent→latent with no MP4 round-trip (Plan 09.9-16). This is the single link
    that removes the ghosting-causing re-encode.
    """
    by_ct: dict[str, list[str]] = {}
    for nid, node in subgraph.items():
        if node["class_type"] == "LTXVImgToVideoConditionOnly":
            raise RuntimeError("VRGDG subgraph contained forbidden LTXVImgToVideoConditionOnly")
        by_ct.setdefault(node["class_type"], []).append(nid)

    for nid in by_ct.get("ManualSigmas", []):
        subgraph[nid]["inputs"]["sigmas"] = LTX2_UPSCALE_REFINEMENT_SIGMAS
    for nid in by_ct.get("KSamplerSelect", []):
        subgraph[nid]["inputs"]["sampler_name"] = UpscaleSampler.EULER.value
    for nid in by_ct.get("CFGGuider", []):
        subgraph[nid]["inputs"]["cfg"] = LTX2_DEFAULT_CFG
    for nid in by_ct.get("LTXVImgToVideoInplace", []):
        subgraph[nid]["inputs"]["strength"] = INPLACE_STRENGTH
        subgraph[nid]["inputs"]["bypass"] = INPLACE_BYPASS
    for nid in by_ct.get("RandomNoise", []):
        subgraph[nid]["inputs"]["noise_seed"] = actual_seed
        subgraph[nid]["inputs"]["generator"] = "fixed"
    for nid in by_ct.get("LTXVSeparateAVLatent", []):
        # Input-side split of the base-gen joint AV latent (latent→latent, 09.9-16).
        subgraph[nid]["inputs"]["av_latent"] = av_latent_ref


def build_ltx2_combined_workflow(
    target_w: int,
    target_h: int,
    base_w: int,
    base_h: int,
    scene_prompt: str,
    dialogue_text: str,
    ref_image_filename: str | None,
    length_s: float,
    audio_filename: str | None,
    *,
    seed: int | None = None,
    use_tiled_vae: bool = True,
    mute_audio: bool = False,
    text_encoder_device: str | None = None,
    output_audio_filename: str | None = None,
    use_lipdub: bool = True,
    use_vrdg_sigmas: bool = False,
    comfyui_base: str | None = None,
) -> dict:
    """Build ONE ComfyUI job: base generation → VRGDG upscale (latent→latent).

    Option B (Plan 09.9-16, the only path). Base-gen samples at LOW res
    (``base_w`` x ``base_h``, e.g. 960x544); its joint AV latent flows DIRECTLY
    into the VRGDG 'Model upscale' sub-graph, which 2x spatial-upscales and
    refines at HIGH res (``target_w`` x ``target_h``, e.g. 1920x1088). There is
    NO LoadVideo / VAEEncode MP4 round-trip — the ghosting source removed here.
    The base LTX-2.3 DiT (node "1") is loaded ONCE and shared by the base
    KSampler and the upscale CFGGuider (no inter-job unload).

    Args:
        target_w, target_h: HIGH-res refine output (e.g. 1920x1088). The upsampler
            scales the low-res base 2x, so target should be 2x base.
        base_w, base_h: LOW-res base generation resolution (e.g. 960x544).
        scene_prompt: Base-gen visual scene description.
        dialogue_text: Base-gen speech content (NATIVE_SPEECH prose wrapping).
        ref_image_filename: Portrait in ComfyUI ``input/`` — seeds base gen AND
            re-conditions the refine via LTXVImgToVideoInplace (strength=1.0,
            bypass=False, D-07).
        length_s: Clip duration (s); frame count rounded to 8k+1.
        audio_filename: Vocals stem in ComfyUI ``input/`` for Audio VAE
            conditioning (D-09), carried inside the base-gen joint AV latent.
            None = empty audio latent.
        seed: Optional fixed seed; None = random (shared by base + refine).
        use_tiled_vae: Use LTXVTiledVAEDecode (default True, saves VRAM).
        mute_audio: Drop the audio track on the OUTPUT video (default False —
            D-09 still carries audio through the joint latent regardless).
        text_encoder_device: "default" (GPU) is standard for Path B — the ~12GB
            Gemma encoder runs on GPU, eliminating CPU-bound text encoding that
            caused ~60 min generation times. "cpu" (default when None) is the
            fallback for Path A or VRAM-constrained validation runs.
        output_audio_filename: Original audio file in ComfyUI ``input/`` for the
            output video track. When provided and ``mute_audio`` is False, a
            ``LoadAudio`` node wires this file directly to ``CreateVideo``'s audio
            input, bypassing the AudioVAE decode (Plan 09.9-17 Option B). The vocals
            latent path (``audio_filename`` → AudioVAE encode → conditioning) is
            unaffected — it still guides lip-sync visuals during generation.
        use_lipdub: When True AND ``audio_filename`` is present AND the Lipdub
            IC-LoRA checkpoint is on disk, apply the strong lip-sync path
            (LTXICLoRALoaderModelOnly + LTXAddVideoICLoRAGuide + LTXVSetAudioRefTokens).
            Defaults to True; falls back to the weak generic audio path otherwise.
        comfyui_base: When provided, run node compatibility checks against
            ComfyUI's /object_info endpoint before building the workflow.
            Catches param range changes (e.g. LTXVTiledVAEDecode overlap 64->8)
            before they cause silent generation failures.

    Returns:
        ComfyUI API-format workflow dict (single chained job).

    Raises:
        comfyui_node_compat.NodeCompatibilityError: If node params are
            incompatible with the live ComfyUI instance.
    """
    # ── Node compatibility check (C-1) ──────────────────────────────
    if comfyui_base is not None:
        from workflows.comfyui_node_compat import run_all_checks
        run_all_checks(comfyui_base)

    if target_h != base_h * UPSCALE_FACTOR:
        logger.warning(
            "Upscale geometry mismatch in build_ltx2_combined_workflow: "
            "target_h=%d but expected base_h * UPSCALE_FACTOR=%d (%dx%d -> %dx%d). "
            "Mismatched upscale geometry distorts aspect ratio (see mv-clip-quality debug).",
            target_h, base_h * UPSCALE_FACTOR, base_w, base_h, base_w * UPSCALE_FACTOR, base_h * UPSCALE_FACTOR,
        )

    actual_seed = _make_seed(seed)
    num_frames = _ltx2_num_frames(length_s)
    te_device = text_encoder_device or "cpu"
    wf: dict = {}

    # ── Shared loaders (ONE DiT load for base + refine, D-10) ────────
    wf["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": LTX2_MODEL_FILE}}
    wf["11"] = {
        "class_type": "LTXAVTextEncoderLoader",
        "inputs": {"text_encoder": LTX2_TEXT_ENCODER_FILE, "ckpt_name": LTX2_TEXT_PROJECTION_FILE, "device": te_device},
    }
    wf["12"] = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": LTX2_AUDIO_VAE_FILE}}
    wf["31"] = {"class_type": "VAELoader", "inputs": {"vae_name": LTX2_VAE_FILE}}
    wf["60"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL_FILENAME}}

    # ── Lipdub IC-LoRA (Option C / BUG A) ───────────────────────────
    # Apply the strong lip-sync LoRA onto the DiT when audio conditioning is
    # active and the checkpoint is present. The LoRA-loaded model is shared by
    # BOTH the base-gen KSampler and the VRGDG refine CFGGuider (base_model_ref
    # is forwarded to _ltx2_base_latent_subgraph and _extract_upscale_subgraph).
    lipdub_active = use_lipdub and audio_filename is not None and _lipdub_lora_present()
    base_model_ref: list = ["1", 0]
    if lipdub_active:
        wf["1a"] = {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": LTX2_LIPDUB_LORA_FILE,
                "strength_model": 1.0,
            },
        }
        base_model_ref = ["1a", 0]

    # ── Upscale-side conditioning (neutral upscale prompt) ───────────
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": LTX2_UPSCALE_POSITIVE_PROMPT}}
    wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": LTX2_NEGATIVE_PROMPT}}
    wf["15"] = {"class_type": "LTXVConditioning", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "frame_rate": LTX2_FRAME_RATE}}

    if ref_image_filename is not None:
        wf["14"] = {"class_type": "LoadImage", "inputs": {"image": ref_image_filename}}

    # ── Base-gen latent sub-graph (LOW res) → joint AV latent ────────
    base_nodes, joint_av_latent_ref = _ltx2_base_latent_subgraph(
        scene_prompt,
        dialogue_text,
        ref_image_filename,
        num_frames,
        width=base_w,
        height=base_h,
        seed=actual_seed,
        model_ref=base_model_ref,
        clip_ref=["11", 0],
        vae_ref=["31", 0],
        audio_vae_ref=["12", 0],
        audio_path=audio_filename,
        use_lipdub=lipdub_active,
        use_vrdg_sigmas=use_vrdg_sigmas,
    )
    wf.update(base_nodes)

    # ── VRGDG upscale sub-graph (HIGH res refine) ────────────────────
    # av_latent is force-wired to the base-gen joint AV latent (latent→latent).
    # base_model_ref is forwarded so the refine CFGGuider shares the LoRA-loaded DiT.
    subgraph = _extract_upscale_subgraph(_load_vrgdg_workflow(), model_ref=base_model_ref)
    _force_locked_params(subgraph, actual_seed, joint_av_latent_ref)
    wf.update(subgraph)

    wf["230"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["207", 0]}}
    h = 4 if use_tiled_vae else 1
    v = 4 if use_tiled_vae else 1
    wf["231"] = {
        "class_type": "LTXVTiledVAEDecode",
        "inputs": {
            "vae": ["31", 0], "latents": ["230", 0], "horizontal_tiles": h, "vertical_tiles": v,
            "overlap": LTXV_TILED_VAE_OVERLAP,  # M-4: centralised constant
            "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto",
        },
    }
    wf["232"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["230", 1], "audio_vae": ["12", 0]}}
    if mute_audio:
        wf["233"] = {"class_type": "CreateVideo", "inputs": {"images": ["231", 0], "fps": float(LTX2_FRAME_RATE)}}
    elif output_audio_filename is not None:
        # Option B (Plan 09.9-17): wire original audio directly to CreateVideo
        # output, bypassing AudioVAE decode. Vocals latent still guides visuals.
        wf["335"] = {"class_type": "LoadAudio", "inputs": {"audio": output_audio_filename}}
        wf["233"] = {"class_type": "CreateVideo", "inputs": {"images": ["231", 0], "audio": ["335", 0], "fps": float(LTX2_FRAME_RATE)}}
    else:
        wf["233"] = {"class_type": "CreateVideo", "inputs": {"images": ["231", 0], "audio": ["232", 0], "fps": float(LTX2_FRAME_RATE)}}
    wf["234"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["233", 0], "filename_prefix": "alice_ltx2_up", "format": "auto", "codec": "auto"},
    }
    return wf
