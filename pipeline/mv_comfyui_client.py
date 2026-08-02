#!/usr/bin/env python3
"""ComfyUIClient — stateful HTTP client for ComfyUI (Plan 09.9-10).

Holds the circuit-breaker counter (`self._consecutive_failures`) and all
ComfyUI HTTP helpers as instance methods. STYLE "Classes for Stateful
Services" + Manual DI: dependencies are constructor-injected.

Module is kept <= 400 lines per STYLE.md (YC-approved 400 ceiling deviation
from the 300 default). Logic is byte-for-byte identical to the original
`generate_music_video_pipeline.py` ComfyUI helpers.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Synchronous HTTP client for ComfyUI, owning circuit-breaker state."""

    def __init__(
        self,
        comfyui_base: str,
        comfyui_output_dir: str,
        gpu_manager_base: str,
        ltx2_vram_mb: int,
        max_consecutive_failures: int = 2,
        logger: logging.Logger = logger,
    ) -> None:
        self._comfyui_base = comfyui_base
        self._comfyui_output_dir = comfyui_output_dir
        self._gpu_manager_base = gpu_manager_base
        self._ltx2_vram_mb = ltx2_vram_mb
        self._max_consecutive_failures = max_consecutive_failures
        self._logger = logger
        self._consecutive_failures: int = 0

    # ── Low-level HTTP helpers ──

    def comfyui_post(self, url: str, payload: dict[str, Any], timeout: int = 30) -> dict:
        """POST JSON to ComfyUI API, return parsed JSON."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def comfyui_get(self, url: str, timeout: int = 30) -> dict:
        """GET JSON from ComfyUI API."""
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_ready(self) -> bool:
        """Check if ComfyUI is responding."""
        try:
            self.comfyui_get(f"{self._comfyui_base}/system_stats", timeout=5)
            return True
        except Exception:
            return False

    # ── Workflow queue / poll / output ──

    def queue_workflow(self, workflow: dict[str, Any]) -> str:
        """Submit workflow to ComfyUI, return prompt_id."""
        result = self.comfyui_post(f"{self._comfyui_base}/prompt", {"prompt": workflow})
        return result["prompt_id"]

    def poll_completion(self, prompt_id: str, timeout: int = 600) -> dict:
        """Poll /history/{id} until prompt_id appears. Returns history entry."""
        url = f"{self._comfyui_base}/history/{prompt_id}"
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Generation timed out after {timeout}s")

            data = self.comfyui_get(url)
            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution failed: {msgs}")
                return entry

            time.sleep(2.0)

    def find_output_file(self, history_entry: dict, output_prefix: str, output_ext: str) -> Path:
        """Extract output file path from ComfyUI history entry.

        Walks the output nodes looking for files matching the prefix.
        Returns the absolute path in ComfyUI's output directory.
        """
        outputs = history_entry.get("outputs", {})
        for node_id, node_output in outputs.items():
            for key, items in node_output.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict) and item.get("filename", "").startswith(output_prefix):
                        filename = item["filename"]
                        subfolder = item.get("subfolder", "")
                        file_type = item.get("type", "output")
                        base = Path(self._comfyui_output_dir) / file_type
                        if subfolder:
                            base = base / subfolder
                        return base / filename

        raise RuntimeError(
            f"No output file found with prefix '{output_prefix}' in ComfyUI history"
        )

    # ── GPU manager / readiness ──

    def start_via_gpu_manager(self) -> bool:
        """Start ComfyUI via gpu-manager /comfyui/start endpoint.

        The /ensure_ready endpoint rejects subprocess-managed services.
        /comfyui/start handles eviction + Slingshot coordination + startup.

        Returns True if ComfyUI becomes ready after the call, False otherwise.
        """
        url = f"{self._gpu_manager_base}/comfyui/start"
        try:
            payload = json.dumps({"vram_mb": self._ltx2_vram_mb}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok"):
                    pid = data.get("pid")
                    slingshot_used = data.get("slingshot_used", False)
                    self._logger.info("gpu-manager: ComfyUI started (pid=%s, slingshot=%s)", pid, slingshot_used)
                    return self.is_ready()
                else:
                    self._logger.warning("gpu-manager /comfyui/start failed: %s", data)
                    return False
        except Exception as e:
            self._logger.warning("gpu-manager /comfyui/start unavailable: %s", e)
            return False

    def wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for ComfyUI to become ready, polling every 2s."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.is_ready():
                return True
            time.sleep(2.0)
        return False

    # ── Circuit breaker + reset ──

    def reset_state(self, timeout_s: int = 30) -> bool:
        """Clear ComfyUI internal state before a clip-gen retry.

        Interrupts the prompt queue, waits for ComfyUI to become responsive
        again. If ComfyUI is unresponsive after the interrupt, restarts it
        via gpu-manager.

        Returns True if ComfyUI is ready after the reset, False otherwise.
        """
        import mv_comfyui  # lazy to avoid circular import (mv_comfyui imports this module)

        # Step 1: Interrupt the prompt queue (best-effort)
        try:
            url = f"{self._comfyui_base}/interrupt"
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._logger.info("ComfyUI: interrupt sent (status=%d)", resp.status)
        except Exception as e:
            self._logger.warning("ComfyUI: interrupt failed (best-effort): %s", e)

        # Step 2: Poll for ComfyUI to become ready
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if mv_comfyui._comfyui_is_ready():
                self._logger.info("ComfyUI: ready after interrupt (%.1fs)", time.monotonic() - start)
                return True
            time.sleep(2.0)

        # Step 3: ComfyUI unresponsive — restart via gpu-manager
        self._logger.warning("ComfyUI unresponsive after interrupt — restarting via gpu-manager")
        if mv_comfyui._start_comfyui_via_gpu_manager():
            return True
        self._logger.error("ComfyUI: restart via gpu-manager failed")
        return False

    def check_vram_gate(self, min_free_mb: int | None = None) -> bool:
        """VRAM safety gate before each clip generation (MTV-08).

        After Slingshot hibernates the LLM, this gate checks ComfyUI is
        responsive. VRAM is freed by Slingshot hibernate — no raw eviction
        call needed.

        Circuit breaker: after MAX_CONSECUTIVE_COMFYUI_FAILURES consecutive
        failures, returns False immediately without waiting.

        min_free_mb: when provided, a REAL per-clip free-VRAM re-check runs
            after ComfyUI is confirmed ready (fix for modotte-oide-1080p-oom:
            this gate was previously neutered — it only checked readiness +
            circuit breaker and NEVER looked at free VRAM). The value is the
            activation-transient headroom (mv_vram._activation_headroom_mb),
            NOT the full estimate — resident model weights are excluded so we
            don't double-count them on clips 2..N. Mainly catches a re-appeared
            orphan LLM squatter mid-batch. nvidia-smi being unavailable is
            treated as pass (guard already ran up front).

        Returns True if gate passes, False if blocked.
        """
        import mv_comfyui  # lazy to avoid circular import (mv_comfyui imports this module)

        # Circuit breaker — fail fast
        if self._consecutive_failures >= self._max_consecutive_failures:
            self._logger.error(
                "VRAM gate circuit OPEN — %d consecutive ComfyUI failures. Aborting pipeline.",
                self._consecutive_failures,
            )
            return False

        # Check ComfyUI is ready
        if not mv_comfyui._comfyui_is_ready():
            # Try to start ComfyUI via gpu-manager
            self._logger.warning("ComfyUI not ready — attempting start via gpu-manager...")
            if not mv_comfyui._start_comfyui_via_gpu_manager():
                self._logger.warning("ComfyUI start failed — waiting up to 30s for manual start...")
                if not mv_comfyui._wait_for_comfyui_ready(timeout=30):
                    self._consecutive_failures += 1
                    self._logger.warning(
                        "VRAM gate failed — ComfyUI not responding (failure %d/%d)",
                        self._consecutive_failures,
                        self._max_consecutive_failures,
                    )
                    return False

        # Real per-clip free-VRAM re-check (activation headroom only)
        if min_free_mb is not None:
            from mv_vram import _get_free_vram_mb  # lazy: avoid import cycle at module load
            free = _get_free_vram_mb()
            if free is not None and free < min_free_mb:
                # Proactively free VRAM held by music-GENERATION services (e.g.
                # ace-step, priority 1) that a music-VIDEO run does not need.
                # The gate was previously a passive check that aborted while
                # ace-step sat on ~10.6GB — now we evict blockers via gpu-manager
                # before declaring the gate failed (Finding 2).
                self._logger.warning(
                    "VRAM gate: %dMB free < %dMB needed — evicting music-generation "
                    "services via gpu-manager before failing",
                    free, min_free_mb,
                )
                if self._evict_for_vram(min_free_mb):
                    free = _get_free_vram_mb()
                if free is not None and free < min_free_mb:
                    self._consecutive_failures += 1
                    self._logger.warning(
                        "VRAM gate failed — only %dMB free, need >= %dMB activation "
                        "headroom after eviction (failure %d/%d)",
                        free, min_free_mb,
                        self._consecutive_failures, self._max_consecutive_failures,
                    )
                    return False

        # Success — reset circuit breaker
        if self._consecutive_failures > 0:
            self._logger.info("VRAM gate passed — resetting failure counter")
        self._consecutive_failures = 0
        return True

    def _evict_for_vram(self, required_mb: int) -> bool:
        """Ask gpu-manager to evict priority blockers until >= required_mb free.

        Uses POST /vram/ensure_for_service (priority-based eviction). For a
        music-VIDEO run the lowest-priority services (ace-step, embedding
        models, immich, mineru — all priority 1) are evicted first; comfyui
        (priority 3) is spared unless nothing else frees enough. Returns True
        if gpu-manager reported a successful eviction pass (Finding 2).
        """
        url = f"{self._gpu_manager_base}/vram/ensure_for_service"
        payload = json.dumps({"required_mb": required_mb}).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok"):
                    self._logger.info(
                        "gpu-manager evicted %s to free VRAM (needed %dMB)",
                        data.get("evicted"), required_mb,
                    )
                    return True
                self._logger.warning(
                    "gpu-manager evict returned ok=%s: %s",
                    data.get("ok"), data.get("error", data.get("detail")),
                )
                return False
        except Exception as e:
            self._logger.warning("gpu-manager evict call failed: %s", e)
            return False
