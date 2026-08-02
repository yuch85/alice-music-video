# ComfyUI Setup

Detailed guide for installing ComfyUI, custom nodes, and models for the music video pipeline.

## ComfyUI Installation

### 1. Clone and Install

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
cd ~/ComfyUI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python main.py --listen 127.0.0.1
```

Open `http://127.0.0.1:8188` in your browser. The UI should load without errors.

### 3. Install ComfyUI Manager

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

Restart ComfyUI. The Manager node should appear in the UI.

## Custom Nodes

Install via ComfyUI Manager (click `Manager` -> `Install Custom Nodes`) or clone manually.

### Required for Core Pipeline

```bash
cd ~/ComfyUI/custom_nodes

# LTX-2.3 video generation
git clone https://github.com/Lightricks/ComfyUI-LTXVideo

# Video helper nodes (load, combine, save)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# Wan video wrapper (HuMo model loaders)
git clone https://github.com/AIFSH/ComfyUI-WanVideoWrapper

# Face ID for Flux2
git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-PuLID-Flux2

# GGUF model support
git clone https://github.com/city96/ComfyUI-GGUF

# Extra model loaders
git clone https://github.com/comfyanonymous/ComfyUI_ExtraModels

# Audio processing
git clone https://github.com/Ascension12/ComfyUI-MelBandRoFormer

# Ovi nodes
git clone https://github.com/AIFSH/ComfyUI-Ovi

# Utility nodes
git clone https://github.com/pomputer/ComfyUI_Essentials

# VRGDG custom nodes (grain, color match, upscale)
git clone https://github.com/vrgamegirl19/comfyui-vrgamedevgirl
```

### Required Utility Nodes

```bash
# pysssss nodes (math, text display)
git clone https://github.com/11cafe/comfyui-workspace-tools

# rgthree (groups, image comparer, notes)
git clone https://github.com/rgthree/rgthree-comfy

# seed-everywhere
git clone https://github.com/11cafe/seed-everywhere-comfyui
```

### Optional Nodes

```bash
# Background removal
git clone https://github.com/FGragon/ComfyUI-rembg

# KJ nodes (resize, LoRA, Wan T5)
git clone https://github.com/kijai/ComfyUI-KJNodes

# Frame interpolation / VRAM cleanup
git clone https://github.com/Fannovel16/comfyui-frame-interpolation
```

After installing nodes, install their Python dependencies:

```bash
cd ~/ComfyUI
python -m pip install -r custom_nodes/*/requirements.txt 2>/dev/null || true
```

## Model Downloads

Place models in the corresponding ComfyUI directories. Directories are relative to `~/ComfyUI/`.

### LTX-2.3 Video Generation

| Model | Download | Directory |
|-------|----------|-----------|
| LTX-2.3 (GGUF Q6) | [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3) | `models/unet/` |
| LTX-2.3 VAE | [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3) | `models/vae/` |
| LTX-2.3 Audio VAE | [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3) | `models/vae/` |
| LTX-2.3 Text Projection | [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3) | `models/clip/` |
| LTX-2.3 Text Encoder (Gemma) | [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3) | `models/checkpoints/` |
| LTX-2.3 Spatial Upscaler x2 | [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3) | `models/latent_upscale_models/` |

### HuMo Talking Head

| Model | Download | Directory |
|-------|----------|-----------|
| HuMo 14B FP8 | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-HuMo-14B) | `models/diffusion_models/` |
| HuMo 1.7B BF16 (optional) | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-HuMo-1.7B) | `models/diffusion_models/` |

### Reference Image Generation

| Model | Download | Directory |
|-------|----------|-----------|
| Flux.1-dev | [Flux website](https://www.bfl.ml/) | `models/checkpoints/` |
| Flux.1-schnell | [Flux website](https://www.bfl.ml/) | `models/checkpoints/` |

### Post-Processing

| Model | Download | Directory |
|-------|----------|-----------|
| Real-ESRGAN | [HuggingFace](https://huggingface.co/nickzha/Real-ESRGAN) | `models/upscale_models/` |

### Exact Filenames

The pipeline references specific filenames. The constants are defined in:
- `pipeline/mv_mvconst.py` (upscaler, VAE overlap)
- `workflows/workflows.py` (model roster)
- `workflows/humo_models.py` (HuMo checkpoints)

Verify your downloaded files match these names or update the configuration.

## VRAM Requirements

| Workflow | Minimum VRAM | Recommended VRAM | Notes |
|----------|--------------|------------------|-------|
| LTX-2.3 I2V (single-stage, 960x544) | 16GB | 24GB | Base generation only |
| LTX-2.3 I2V + upscale (two-stage) | 32GB | 48GB | 960x544 -> 1920x1088 |
| HuMo 14B (16s clips) | 24GB | 24GB | FP8 variant |
| HuMo 14B (18s clips) | 24GB | 48GB | Near VRAM ceiling |
| Flux.1 reference image gen | 12GB | 16GB | One-shot, not per-clip |

The pipeline includes automatic VRAM detection (`pipeline/mv_vram.py`) and will fall back to single-stage generation if two-stage upscale would exceed available memory.

## Testing the Setup

### 1. Load a Workflow

Open ComfyUI and load one of the workflow JSON files from `workflows/comfyui/`:

```
workflows/comfyui/wan_humo_mvc_v9.json    # HuMo talking-head
workflows/comfyui/image_edit.json          # Qwen Image Edit
workflows/comfyui/prompt_creator.json      # Prompt generation
```

### 2. Verify Node Registration

The workflow should load without red "missing node" errors. If nodes are missing, verify the corresponding custom node package is installed and ComfyUI has been restarted.

### 3. Run a Test Queue

Queue a single test generation with a short duration (6s) to verify the full pipeline works end-to-end before running the full music video generation.

## Environment Configuration

After setting up ComfyUI, configure the pipeline to connect to it:

```bash
cp config.example.env .env
```

Set the ComfyUI server address in `.env`:

```
COMFYUI_SERVER=http://127.0.0.1:8188
```

Adjust model paths if your ComfyUI installation is not at `~/ComfyUI`.
