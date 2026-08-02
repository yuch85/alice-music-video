---
name: ace-step-music
description: Generate original music, covers, and stem extraction via ACE-Step 1.5. Optimized for Turbo-2B model.
allowed-tools:
  - Read
  - Bash
  - mcp__gpu-manager__alice_ensure_service_ready
  - mcp__gpu-manager__alice_gpu_status
---

# /ace-step-music

ACE-Step 1.5 music generation via the `ace-step-mcp` MCP server. Optimized for the **Turbo-2B** model which is the only supported path for long-form (200s+) stability as of June 2026.

> [!IMPORTANT]
> **DEFAULT MODEL: Turbo-2B + 0.6B LM.**
> Stable, fast (~10s for 200s audio), and low VRAM (~12GB). Consolidated to `ace-step-mcp` server.

> [!CAUTION]
> **ROADBLOCK CHECKLIST (Check BEFORE calling tools):**
> 1. **Tool Prefix**: Tools are named `mcp__ace-step-mcp__ace_step_...`.
> 2. **Health Check**: Call `ace_step_health()` first. If `ready_for_inference: false`, call `ace_step_init_model(preset="turbo-2b")`.
> 3. **VRAM**: If you get "Insufficient VRAM", call `mcp__gpu-manager__alice_ensure_service_ready(service_name="ace-step")`.
> 4. **Hand Absolute Paths**: The wrapper auto-stages files for the server.

## When to use

- **New Song**: "write a song about X", "make an original tune"
- **Cover**: "cover this song in Japanese", "re-do with vocals"
- **Language Cover**: "translate this to Mandarin and re-sing"
- **Repaint**: "regenerate the chorus", "redo the bridge"
- **Mood Variant**: "sadder v2", "more aggressive"
- **Duet**: "male-female duet", "two singers trading verses"

## Service bringup

ACE-Step runs on `http://127.0.0.1:8015`.
gpu-manager name = `ace-step`, port 8015, ~12GB VRAM.

```python
# Ensure service is ready
mcp__gpu-manager__alice_ensure_service_ready(service_name="ace-step")

# Check health
health = mcp__ace-step-mcp__ace_step_health()
if not health["ready_for_inference"]:
    mcp__ace-step-mcp__ace_step_init_model(preset="turbo-2b")
```

## Recipe: original tune (text2music)

```python
result = mcp__ace-step-mcp__ace_step_generate_music(
    prompt="J-pop sad piano ballad, solo piano + sparse strings, exhausted vocal",
    lyrics=JP_LYRICS,        # multi-line, [Verse]/[Chorus] labels
    audio_duration=200,      # 10-600 seconds
    inference_steps=8,       # Turbo rec 8-16
    vocal_language="ja",     # ISO 639-1
    thinking=True,           # LM expansion enabled
    seed=11,
    use_random_seed=False,   # CRITICAL: must be False for seed to work
)
task_id = result["data"]["task_id"]

# Poll for result
final = mcp__ace-step-mcp__ace_step_query_result(
    task_ids=[task_id],
    wait=True,
    timeout_s=120.0,
)
```

## Recipe: duet (two-vocal)

Duets require explicit mention in **both** the `prompt` and `lyrics`.

1. **Caption (Prompt)**: Must mention "duet between male and female" or similar.
2. **Lyrics**: Use `[Verse 1 - Female]`, `[Verse 2 - Male]`, `[Chorus - Both]` tags.

```python
result = mcp__ace-step-mcp__ace_step_generate_music(
    prompt="Romantic duet ballad, interweaving male and female vocals.",
    lyrics="[Verse 1 - Female]\nLine 1...\n\n[Verse 2 - Male]\nLine 2...",
    vocal_language="en",
    # ... other params ...
)
```

## Core Gotchas

- **Absolute Paths**: The server only accepts paths in `/tmp/`. The MCP wrapper auto-copies files there.
- **Seed Breakdown**: Setting `seed` without `use_random_seed=False` results in random output.
- **Initial Warmup**: First call after a restart might fail with "Model not fully initialized". The wrapper probes `/v1/init` to mitigate this, but if it fails, simply retry.
- **XL Models**: **DEPRECATED**. XL-sft and XL-base are unstable at long durations (≥200s) and consume ~30GB VRAM. Use Turbo for all standard requests.

## Related Reference
- `ACE-STEP-MASTER.md` in `.planning/notes_alia_and_ishi` for historical context and corruption post-mortems.
