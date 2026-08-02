# ComfyUI Workflows for Music Video Pipeline

Pre-exported ComfyUI workflow JSONs used by the music video pipeline.
All workflows are in **UI-format** (human-readable, loadable via ComfyUI's
`Load` button), not API-format.

## Directory Layout

```
workflows/
  comfyui/                  # Workflow JSONs by category
    civitai/                # VRGDG CivitAI-published T2V workflows (v1, v2)
    ltx23_mv_creator/       # LTX 2.3 Music Video Creator V5.1 (T2V, I2V, Prompt)
    luts/                   # Color grading LUTs (11 .cube files)
    old/                    # Deprecated workflows (V6-V7, early Z-Image)
    *.json                  # Active workflows (see list below)
  *.py                      # Python workflow builders (programmatic JSON gen)
  *.json                    # Node maps and model configs
```

## Active Workflows

| File | Model | Mode | Description |
|------|-------|------|-------------|
| `wan_humo_mvc_v9.json` | Wan2.1-HuMo 14B | Full MVC | Primary HuMo music video creator, V9 |
| `wan_humo_mvc_v81.json` | Wan2.1-HuMo 14B | Full MVC | HuMo MVC V8.1 (previous stable) |
| `humo_manual_mode.json` | Wan2.1-HuMo 14B | Manual | Manual HuMo mode for per-clip control |
| `z_image_wan_humo_mvc.json` | Z-Image + Wan | Full MVC | Z-Image with Wan HuMo MVC |
| `prompt_creator.json` | N/A | Prompt only | Standalone prompt generation workflow |
| `lyric_extraction.json` | N/A | Audio only | Lyric extraction from audio |
| `image_edit.json` | Qwen Image Edit | I2I | Image-to-image editing |
| `vox_cpm_tts.json` | VoxCPM2 | TTS | Voice synthesis |
| `upscale_wan22.json` | Wan2.2 | Upscale | Wan2.2 upscaler |
| `upscale_generic.json` | Generic | Upscale | Model-agnostic upscale |
| `upscale_with_cn.json` | Generic+CN | Upscale | Upscale with control net |

## Required Custom Nodes

Install via ComfyUI Manager or clone manually into `custom_nodes/`.

### Required (core pipeline)

| Node Package | Install URL / Search | Nodes Used |
|--------------|----------------------|------------|
| **comfyui-vrgamedevgirl** | `vrgamedev` or `https://github.com/vrgamegirl19/comfyui-vrgamedevgirl` | All `VRGDG_*` nodes, `FastFilmGrain`, `ColorMatchToReference` |
| **ComfyUI-WanVideoWrapper** | `ComfyUI-WanVideoWrapper` | `WanVideoModelLoader`, `WanVideoVAELoader`, `WanVideoTextEncode`, `WanVideoBlockSwap`, `WanVideoLoraSelect` |
| **comfyui-videohelpersuite** | `ComfyUI-VideoHelperSuite` | `VHS_LoadAudioUpload`, `VHS_VideoCombine` |

### Required (utility nodes)

| Node Package | Install URL / Search | Nodes Used |
|--------------|----------------------|------------|
| **pysssss** | `ComfyUI_Custom_Nodes_Nygro` | `MathExpression`, `ShowText` |
| **rgthree** | `ComfyUI-rGthree` | `Fast Groups Muter`, `Image Comparer`, `MarkdownNote`, `Power Puter` |
| **ComfyUI-Essentials** | `ComfyUI_Essentials` | `CM_FloatToInt`, `CM_IntToFloat` |
| **seed-everywhere** | `seed-everywhere-comfyui` | `Seed Everywhere` |

### Optional (feature-gated)

| Node Package | Nodes Used | Used In |
|--------------|------------|---------|
| **comfyui-rembg** | `RemBGSession+`, `ImageRemoveBackground+` | Background removal workflows |
| **comfyui-kjnodes** | `ImageResizeKJv2`, `LoraLoaderModelOnly`, `LoadWanVideoT5TextEncoder`, `PathchSageAttentionKJ` | Upscale, LoRA workflows |
| **comfyui-frame-interpolation** | `RAMCleanup`, `VRAMCleanup` | VRAM management |
| **ComfyUI-Custom-Scripts** | `show_text_party` | Debug displays |

## Models

See `humo_models.py` for HuMo checkpoint constants and `workflows.py` for
the full model roster. Key models:

| Category | Filename | Location |
|----------|----------|----------|
| HuMo 14B | `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` | `models/diffusion_models/` |
| HuMo 1.7B | `Wan2_1-HuMo-1.7B_bf16.pth` | `models/diffusion_models/` |
| LTX 2.3 | `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | `models/unet/` (GGUF) |
| LTX VAE | `LTX23_video_vae_bf16.safetensors` | `models/vae/` |
| LTX Audio VAE | `LTX23_audio_vae_bf16.safetensors` | `models/vae/` |
| LTX Text Encoder | `gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors` | `models/checkpoints/` |
| LTX Text Projection | `ltx-2.3_text_projection_bf16.safetensors` | `models/clip/` |
| Upscale Model | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `models/upscale_models/` |

Full model inventory with exact filenames is in `workflows.py` (top-level constants).

## Python Workflow Builders

The pipeline does not load workflow JSONs directly at runtime. Instead, it
uses Python builders that generate API-format dicts programmatically:

| Module | Purpose |
|--------|---------|
| `workflows.py` | Core builders: Flux2, Wan T2V/I2V, Qwen Image Edit |
| `workflow_humo.py` | HuMo-specific workflow builder |
| `workflow_ltx2.py` | LTX 2.3 workflow builder |
| `workflow_ltx2_upscale.py` | LTX 2.3 upscale subgraph builder |
| `workflow_dialogue.py` | Dialogue-driven generation |
| `workflow_ovi.py` | Ovi I2AV workflow builder |
| `comfyui_node_compat.py` | Runtime node schema validation |

## Node Map

`vrgdg_upscale_node_map.json` documents the 10-node upscale subgraph
extracted from the VRGDG I2V workflow, including input slot wiring,
filename remaps, and locked runtime parameters.
