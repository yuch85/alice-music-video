# Installation Guide

## Prerequisites

- **Python 3.10+**
- **FFmpeg** (latest, available via package manager or ffmpeg.org)
- **NVIDIA GPU** with CUDA support (24GB+ VRAM minimum, 48GB+ recommended)
- **ComfyUI** (stable release; see [ComfyUI Setup](docs/COMFYUI_SETUP.md) for details)

## ComfyUI Setup

### 1. Install ComfyUI

Follow the official [ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI).

### 2. Install Required Custom Nodes

Install via ComfyUI Manager or clone manually into `ComfyUI/custom_nodes/`.

#### Core Pipeline Nodes

| Node Package | Search / Clone | Purpose |
|--------------|----------------|---------|
| **ComfyUI-LTXVideo** | `ComfyUI-LTXVideo` | LTX-2.3 video generation |
| **ComfyUI-VideoHelperSuite** | `ComfyUI-VideoHelperSuite` | Video helper nodes (load, combine) |
| **ComfyUI-WanVideoWrapper** | `ComfyUI-WanVideoWrapper` | Wan video model loaders and nodes |
| **ComfyUI-PuLID-Flux2** | `ComfyUI-PuLID-Flux2` | Face ID for Flux2 |
| **ComfyUI-GGUF** | `ComfyUI-GGUF` | GGUF model support |
| **ComfyUI_ExtraModels** | `ComfyUI_ExtraModels` | Extra model loaders |
| **ComfyUI-MelBandRoFormer** | `ComfyUI-MelBandRoFormer` | Audio processing |
| **ComfyUI-Ovi** | `ComfyUI-Ovi` | Ovi nodes |
| **comfyui_essentials** | `ComfyUI_Essentials` | Utility nodes (type conversions) |
| **comfyui-vrgamedevgirl** | `vrgamedev` or clone from [VRGameDevGirl](https://github.com/vrgamedevgirl) | VRGDG custom nodes (grain, color match, upscale subgraph) |

#### Utility Nodes

| Node Package | Search / Clone | Purpose |
|--------------|----------------|---------|
| **pysssss** | `ComfyUI_Custom_Nodes_Nygro` | MathExpression, ShowText |
| **rgthree** | `ComfyUI-rGthree` | Fast Groups Muter, Image Comparer, MarkdownNote |
| **seed-everywhere** | `seed-everywhere-comfyui` | Seed Everywhere |

#### Optional (Feature-Gated)

| Node Package | Purpose |
|--------------|---------|
| **comfyui-rembg** | Background removal |
| **comfyui-kjnodes** | Image resize, LoRA loading, Wan T5 encoder |
| **comfyui-frame-interpolation** | RAM/VRAM cleanup nodes |
| **ComfyUI-Custom-Scripts** | Debug displays |

See `workflows/README.md` for the full node inventory with specific node names used per package.

## Required Models

Download the following models and place them in the corresponding ComfyUI directories.

### Video Generation

| Model | ~Size | ComfyUI Directory |
|-------|-------|-------------------|
| LTX-2.3 (e.g., GGUF Q6 variant) | ~10GB | `models/unet/` |
| LTX-2.3 VAE | ~2GB | `models/vae/` |
| LTX-2.3 Text Projection | ~1GB | `models/clip/` |
| LTX-2.3 Spatial Upscaler x2 | ~1GB | `models/latent_upscale_models/` |

### Talking Head

| Model | ~Size | ComfyUI Directory |
|-------|-------|-------------------|
| HuMo 14B (FP8) | ~15GB | `models/diffusion_models/` |
| HuMo 1.7B (BF16, optional) | ~3GB | `models/diffusion_models/` |

### Reference Image Generation

| Model | ~Size | ComfyUI Directory |
|-------|-------|-------------------|
| Flux.1-dev or Flux.1-schnell | ~23GB | `models/checkpoints/` |

### Post-Processing

| Model | ~Size | ComfyUI Directory |
|-------|-------|-------------------|
| Real-ESRGAN | ~0.2GB | `models/upscale_models/` |

Exact filenames used by the pipeline are defined in `pipeline/mv_mvconst.py` and `workflows/workflows.py`. See [docs/COMFYUI_SETUP.md](docs/COMFYUI_SETUP.md) for download links.

## Environment Configuration

Copy the example environment file and adjust paths to match your installation:

```bash
cp config.example.env .env
```

Edit `.env` to set:
- ComfyUI server URL and port
- Model paths
- Working directories for outputs

## Verification

After installation, verify the pipeline can import correctly:

```bash
cd ~/alice/scripts/mv-public
python -c "from pipeline.mv_validation import run_validation; print('OK')"
```

Run the test suite to validate the installation:

```bash
python -m pytest tests/ -v
```

## Troubleshooting

- **CUDA errors**: Verify `nvidia-smi` shows your GPU and that the CUDA toolkit version matches your PyTorch build.
- **Missing custom nodes**: ComfyUI will log missing node classes on startup. Install the corresponding package via ComfyUI Manager.
- **VRAM OOM during generation**: Reduce clip duration or switch from two-stage to single-stage upscale. The pipeline attempts automatic fallback.
- **Node version mismatches**: Some custom nodes have breaking changes between versions. If a workflow fails to load, check `workflows/comfyui_node_compat.py` for known incompatibilities.
