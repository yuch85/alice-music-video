# Automated Beat-Aligned Music Video Pipeline
**Technical Architecture & System Design**

## 1. Executive Summary
The Music Video Generation Pipeline is an enterprise-grade, multimodal AI orchestration system designed to autonomously plan, render, and composite beat-aligned music videos. It bridges the gap between Large Language Model (LLM) creative reasoning and compute-heavy diffusion models, emulating a professional film production crew. 

By utilizing a durable 12-stage Finite State Machine (FSM), the system safely navigates complex operations—including audio stem separation, time-stamped transcription, cinematic pre-production, visual rendering, and timeline compositing—while managing strict rigid constraints like VRAM arbitration and chronological drift.

## 2. Core Architectural Pillars

### 2.1 Durable Finite State Machine (FSM)
The pipeline is managed by a resumable, 12-stage FSM. State is serialized to an `index.md` / `index.json` file in the project folder. This enables the pipeline to crash, halt, or pause for human feedback without losing context.
*   **Rollback & Recovery:** Implements state-level bidirectional recovery. If validation checks fail or human reviewers reject artifacts, the FSM safely rolls back to the offending stage, resetting downstream states dynamically.

### 2.2 Dual-Engine Video Generation
Video segments are intelligently routed to one of two physical backend diffusion pipelines based on chronological and semantic needs:
*   **HuMo 14B (Audio-Conditioned):** Routed for "singer" clips requiring phoneme-accurate lip-syncing. Driven by the vocals stem; outputs at 848x480 native, upscaled to 1080p.
*   **LTX-2.3 (I2V Cinematic):** Routed for "B-roll" and narrative scenes. Utilizes a low-res base generation (960x544) combined with a VRGDG latent-to-latent upscaler (Path B) to produce pristine 1920x1080 cinematic video.

### 2.3 Slingshot VRAM Arbitration
Orchestrating local LLMs alongside heavy video diffusion models on a single 48GB GPU requires dynamic context swapping. When the FSM shifts from *Planning* to *Execution*, the `Slingshot` service hibernates the local LLM, clearing VRAM for ComfyUI. Upon completion, the LLM is restored into memory to evaluate the FSM output.

### 2.4 Syntactic & Schema Resilience
All inter-agent communication (JSON payloads, Markdown tables) is passed through an automatic repair layer. This compensates for LLM context truncation or hallucinated schema parameters by executing bracket balancing, trailing comma removal, and table column-padding.

---

## 3. The 12-Stage Pipeline

### Phase A: Pre-Initialization & Ingestion
*   **Stage 0: Project Initialisation:** Scans workspace directories (`songs/`) for pre-existing audio, raw lyrics, and reference images. Prompts the user for reuse confirmation, constructs the project directory schema, and instantiates the canonical `index.md` tracker.

### Phase B: Creative Direction (The Virtual Film Crew)
*   **Stage 1: Creative Interview:** Gathers high-level intent (Character Arcs, Theme, Tempo) through conversational interaction.
*   **Stage 2: Director's Treatment:** Generates a 13-point professional treatment outlining the philosophical and visual arc of the video.
*   **Stage 3: Continuity Bible:** Generates programmatic invariants (e.g., hair, makeup, wardrobe, lighting constraints) that all downstream prompts must obey to prevent temporal hallucinations.

### Phase C: Structural & Audio Processing
*   **Stage 4: Beat Sheet:** Invokes Demucs to separate the vocal stem, followed by Whisper to generate time-stamped lyrics (`transcript.json`). The song is chunked into logical cinematic beats. Includes gap-filling for instrumental sections and sub-18s splitting to accommodate LTX-2's hard maximum duration constraints.
*   **Stage 4.5: Visual Storyboard:** Merges the beat sheet with the Treatment. Assigns "Visual Weights", shot density, and coverage strategies to each time-slice.
*   **Stage 5: Shot List:** Translates the storyboard into camera-specific syntax (Action, Lens, Angle, Lighting) yielding multiple candidate options per beat.

### Phase D: Visual Look Dev & Pre-Vis
*   **Stage 6: Image Approval:** Wraps the QEI (Qwen Image Edit) pipeline to generate reference stills per beat. Synthesizes a self-contained HTML file (`approval_dashboard.html`) allowing humans to visually select candidate images and edit underlying prompts in a single DOM viewport, generating a unified JSON payload for the FSM.
*   **Stage 7: Prompts:** Finalizes the text prompt schema (Motion, Camera, Details, Negative) checking against Continuity Bible invariants.

### Phase E: Execution & Assembly
*   **Stage 8: Validation:** Runs pre-flight checks. Validates that the Storyboard is complete, timestamps are contiguous, and assets exist. Capable of programmatically rolling back the FSM if missing assets are detected.
*   **Stage 9: Generation:** The GPU batch job. Routes clips to ComfyUI, passing the respective references, prompts, and audio conditioning streams.
*   **Stage 10: Quality Control (QC):** Validates generated constraints, interrogating output duration and dimensions. 
*   **Stage 11: Complete:** Assembles intermediate mp4s. Applies ffmpeg canvas layering (timeline-aware compositing) padding generation to true audio track time, applying LUTs and film grain filters, and muxing the original high-fidelity audio track.

---

## 4. Key Artifacts & Directory Schema

```text
songs/music-videos/<Project_Name>/
├── index.md                    # Core canonical tracking and FSM State file
├── audio/                      # Source mp4/wav
├── stems/                      # Demucs-separated output (vocals/instrumentals)
├── lyrics/                     # Source untimed lyrics + transcript.json (timed)
├── refs/                       # Approved references & visual artifacts
├── approvals/                  # Pre-vis generation folders (Candidate images A/B/C)
├── approval_dashboard.html     # Human-in-the-loop candidate selection UI
├── clips/                      # Individual rendered scenes (.mp4)
├── final/                      # Post-processed complete timelines
└── * (Pre-prod artifacts)      # director_treatment.md, continuity_bible.md, etc.
```

## 5. Summary
This pipeline formalizes *cinematography* and *film grammar* within an autonomous AI loop. Rather than blindly mapping lyrics to latent noise, the architecture constructs boundaries, validates historical rules (Continuity), isolates processing workflows (Audio vs. Visuals), and arbitrates hardware restrictions gracefully. It is not merely a script, but a comprehensive virtual studio.
