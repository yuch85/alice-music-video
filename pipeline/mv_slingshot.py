#!/usr/bin/env python3
"""SlingshotClient + Slingshot recovery wiring (Plan 09.9-08 + 09.9-09).

Module is kept <= 400 lines per STYLE.md (YC-approved 400 ceiling deviation
from the 300 default). Logic is byte-for-byte identical to the original
`generate_music_video_pipeline.py` block this module was split from.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

GPU_MANAGER_HOST = os.getenv("GPU_MANAGER_HOST", "127.0.0.1")
GPU_MANAGER_PORT = int(os.getenv("GPU_MANAGER_PORT", "8090"))
GPU_MANAGER_BASE = f"http://{GPU_MANAGER_HOST}:{GPU_MANAGER_PORT}"

# ── Slingshot recovery globals (Wave 2: wake on ALL exit paths) ──

_RECOVERY_SLINGSHOT: "SlingshotClient | None" = None
_recovery_registered: bool = False


def _atexit_wake() -> None:
    """Ensure slingshot.wake() runs on normal process exit."""
    if _RECOVERY_SLINGSHOT is not None:
        logger.info("atexit: ensuring Slingshot wake on process exit")
        _RECOVERY_SLINGSHOT.ensure_wake()


def _sigterm_handler(signum: int, frame: "object | None") -> None:
    """Ensure slingshot.wake() runs on SIGTERM, then exit cleanly."""
    if _RECOVERY_SLINGSHOT is not None:
        logger.info("SIGTERM: ensuring Slingshot wake before exit")
        _RECOVERY_SLINGSHOT.ensure_wake()
    raise SystemExit(128 + signum)  # intentional: clean exit after wake


def _register_slingshot_recovery(client: "SlingshotClient") -> None:
    """Register the SlingshotClient for atexit + SIGTERM recovery.

    Called once per pipeline run after the SlingshotClient is created.
    Idempotent — only registers once.
    """
    global _RECOVERY_SLINGSHOT, _recovery_registered
    _RECOVERY_SLINGSHOT = client
    # Keep the facade re-export (a static copy taken at import time) in sync so
    # `generate_music_video_pipeline._RECOVERY_SLINGSHOT` reflects the live value
    # (preserves the original monolith's observable behavior — the test reads it
    # from the facade module after registration).
    _sync_facade_recovery()
    if not _recovery_registered:
        atexit.register(_atexit_wake)
        signal.signal(signal.SIGTERM, _sigterm_handler)
        _recovery_registered = True
        logger.info("Slingshot recovery registered (atexit + SIGTERM)")


def _sync_facade_recovery() -> None:
    """Propagate the live _RECOVERY_SLINGSHOT to the facade module if loaded."""
    try:
        import sys

        gm = sys.modules.get("generate_music_video_pipeline")
        if gm is not None:
            gm._RECOVERY_SLINGSHOT = _RECOVERY_SLINGSHOT
    except Exception:  # pragma: no cover — defensive: facade may be absent
        pass


def is_local_llm_active() -> bool:
    """Return True iff a local LLM is currently provisioned/active.

    Queries gpu-manager's /slingshot/status (the same signal the rest of the
    pipeline uses). On a cloud-LLM session (alia-c / ishi-c) no local LLM is
    running, so this returns False. Used to skip ALL slingshot operations
    (hibernate + wake) when there is nothing to preserve (Finding 3).
    """
    try:
        client = SlingshotClient()
        status = client.status()
    except Exception:
        return False
    if not status:
        return False
    return bool(status.get("llm_active"))


class SlingshotClient:
    """Synchronous HTTP client for gpu-manager Slingshot endpoints.

    Wraps POST /slingshot/hibernate, POST /slingshot/wake, and
    GET /slingshot/status. Used by the music video pipeline to preserve
    local LLM context during GPU-intensive clip generation.
    """

    def __init__(self, base_url: str | None = None):
        self._base = base_url or GPU_MANAGER_BASE

    def status(self) -> dict | None:
        """GET /slingshot/status — return state dict or None on error."""
        url = f"{self._base}/slingshot/status"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("Slingshot status check failed: %s", e)
            return None

    def hibernate(self) -> bool:
        """POST /slingshot/hibernate — return True if hibernated.

        Saves KV cache (if supported), stops the local LLM service,
        and frees VRAM for GPU-intensive tasks.
        """
        url = f"{self._base}/slingshot/hibernate"
        payload = b""
        req = urllib.request.Request(url, data=payload, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok"):
                    logger.info("Slingshot: LLM hibernated (slot_saved=%s)", data.get("slot_saved"))
                    return True
                else:
                    logger.info("Slingshot: hibernate skipped — %s", data.get("error", data.get("detail", "unknown")))
                    return False
        except Exception as e:
            logger.warning("Slingshot hibernate failed: %s", e)
            return False

    def wake(self, task_name: str = "music_video", output_path: str = "") -> dict | None:
        """POST /slingshot/wake — return result dict or None on error.

        Restores KV cache (if saved), restarts the local LLM service,
        and returns a wake prompt summarizing the hibernation period.
        """
        url = f"{self._base}/slingshot/wake"
        payload_dict: dict[str, str] = {"gen_type": task_name}
        if output_path:
            payload_dict["output_path"] = output_path
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                logger.info("Slingshot: LLM woke (ok=%s)", data.get("ok"))
                return data
        except urllib.error.HTTPError as e:
            # 422 = endpoint rejected the request (e.g. duplicate wake,
            # state transition in progress). Not a failure — the LLM
            # is already waking or awake.
            if e.code == 422:
                logger.info("Slingshot: wake returned 422 (already waking)")
                return {"ok": True, "state": "waking"}
            logger.warning("Slingshot wake failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Slingshot wake failed: %s", e)
            return None

    def ensure_wake(self, task_name: str = "music_video", output_path: str = "") -> dict | None:
        """Idempotent wake — wakes if the Slingshot is hibernating or LLM is down.

        Safe to call unconditionally on all exit paths (finally, atexit,
        SIGTERM). If the LLM was never hibernated and is currently running,
        this is a no-op.

        Also wakes when state is "idle" but llm_active is false — this can
        happen when status() clears the HIBERNATING state after systemd
        auto-restarts the LLM during hibernation, but the restarted process
        later stops. See slingshot.py:status() lines 374-393.

        If wake fails (e.g. 422, network error), falls back to
        /slingshot/recover to clear stale hibernating state.
        """
        status = self.status()
        if not status:
            logger.warning("Slingshot: ensure_wake — status check failed")
            return None

        state = status.get("state", "unknown")
        llm_active = bool(status.get("llm_active"))

        if state == "hibernating" or (state == "idle" and not llm_active):
            logger.info(
                "Slingshot: ensure_wake — waking (state=%s, llm_active=%s)",
                state, llm_active,
            )
            result = self.wake(task_name=task_name, output_path=output_path)
            if result is None or not result.get("ok"):
                # Wake failed — try recover to clear stale state
                logger.warning("Slingshot: wake failed, attempting recover")
                return self._recover()
            return result
        logger.info(
            "Slingshot: ensure_wake — no-op (state=%s, llm_active=%s)",
            state, llm_active,
        )
        return None

    def _recover(self) -> dict | None:
        """POST /slingshot/recover — clear stale hibernating state.

        Used as fallback when wake() fails but the LLM may already be
        running (e.g. manual restart, systemd auto-restart).
        """
        url = f"{self._base}/slingshot/recover"
        req = urllib.request.Request(url, data=b"", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                logger.info("Slingshot: recover succeeded (health=%s)", data.get("health"))
                return data
        except Exception as e:
            logger.warning("Slingshot: recover failed: %s", e)
            return None

    def ensure_hibernate(self) -> bool:
        """Hibernate with stale-state recovery.

        If state is IDLE, hibernate normally. If HIBERNATING/HIBERNATED,
        run recover first (kills any orphan LLM still holding VRAM), then
        re-check and hibernate if now IDLE. Returns True if hibernated.
        """
        status = self.status()
        if not status:
            return False

        state = status.get("state")
        if state in ("hibernating", "hibernated"):
            logger.info("Slingshot: stale-state recovery (state=%s)", state)
            self._recover()
            status = self.status()
            if not status:
                return False
            state = status.get("state")

        if state == "idle":
            return self.hibernate()
        logger.info("Slingshot: not hibernating (state=%s)", state)
        return False
