# alice-music-video

AI-native pipeline for generating narrative music videos.

## Overview

A planning-driven pipeline for generating narrative music videos using open-source models. Creative decisions drive generation, not the other way around.

- **LTX-2.3** video generation with spatial upscaling and audio vocal stem conditioning for lip-sync
- **HuMo 14B** talking-head as fallback when vocal conditioning is unavailable
- **Planning-first** workflow: storyboard, shot list, and beat sheet precede every GPU pass
- **Resume-aware**: pipeline can resume from any stage after interruption
- **VRAM-aware**: automatic fallback from two-stage to single-stage upscale based on available GPU memory

## What This Does (and Does Not)

This repository focuses on **music video generation**. It does **not** generate music.

**Primary inputs:**
- A finished music track (audio file)
- Reference images (character portraits, scene compositions)
- Optional lyrics file

Reference images can be prepared using Qwen Image Edit (QVI), Flux, or other image generation tools. Music generation is outside the scope of this project (ACE-Step or other tools can be used upstream).

## Workflow

```
Music Track -> Audio Analysis -> Beat Sheet -> Storyboard -> Shot Plan
-> Keyframe Prompts -> Reference Images -> Motion Prompts -> Video Clips
-> Lip Sync -> Assembly -> Final Output
```

## Architecture

- **Claude Code skills** drive the pre-production planning phases (interview, treatment, storyboard, shot list, prompts)
- **Python pipeline** handles audio analysis, clip generation, and assembly
- **ComfyUI workflows** handle LTX-2.3 video generation, upscaling, and HuMo talking-head (fallback)
- **FFmpeg** handles final assembly, color grading, and subtitle burning

## Requirements

- Python 3.10+
- FFmpeg (latest)
- ComfyUI with custom nodes (see [INSTALL.md](INSTALL.md))
- NVIDIA GPU (24GB+ VRAM recommended; 48GB+ for two-stage upscale)
- ~50GB disk space for models

## Quick Start

1. Install prerequisites and ComfyUI custom nodes -- see [INSTALL.md](INSTALL.md)
2. Download the required models -- see [INSTALL.md](INSTALL.md) or [docs/COMFYUI_SETUP.md](docs/COMFYUI_SETUP.md)
3. Copy `config.example.env` to `.env` and adjust paths
4. Run the validation check:

```bash
python -c "from pipeline.mv_validation import run_validation; print('OK')"
```

5. Run the pipeline:

```bash
python pipeline/generate_music_video_pipeline.py \
    --input track.mp3 \
    --output ./output \
    --portrait portrait.png \
    --scene-prompt "singer on stage, concert lighting"
```

## Project Structure

```
pipeline/          Core pipeline modules (audio, beats, clips, validation)
assemble/          Assembly and post-processing scripts
workflows/         ComfyUI workflow builders (Python) and JSON files
skills/            Claude Code skills for pre-production planning
tests/             Test suite
docs/              Setup and operational documentation
```

## Documentation

- [INSTALL.md](INSTALL.md) -- Installation and setup
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Pipeline architecture and design principles
- [docs/COMFYUI_SETUP.md](docs/COMFYUI_SETUP.md) -- ComfyUI installation and model setup
- [docs/RELEASING.md](docs/RELEASING.md) -- Git subtree release workflow

## Acknowledgements

This project builds heavily upon the excellent work of the [VRGameDevGirl](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl) repository and community.
Many architectural ideas, ComfyUI workflows, and custom nodes originated from that project.
This repository is licensed under AGPLv3 in keeping with the upstream licensing philosophy.

## License

AGPLv3
