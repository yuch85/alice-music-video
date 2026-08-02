#!/usr/bin/env python3
"""GPU manager facade — public repository (noop-only, no Slingshot dependency).

Slim facade for the alice-music-video public repository. Always loads
NoopProvider — there is no gpu-manager in the public repo. All original
mv_slingshot symbols are re-exported for backward compatibility with
pipeline consumers.

Provider: NoopProvider only (no env var dispatch, no Slingshot import).
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
from typing import TYPE_CHECKING

from mv_gpu_interface import GPUManagerProtocol
from mv_gpu_noop import NoopProvider

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

GPU_MANAGER_HOST = os.getenv("GPU_MANAGER_HOST", "127.0.0.1")
GPU_MANAGER_PORT = int(os.getenv("GPU_MANAGER_PORT", "8090"))
GPU_MANAGER_BASE = f"http://{GPU_MANAGER_HOST}:{GPU_MANAGER_PORT}"

# ── Active provider (always NoopProvider in public repo) ─────────────

_active_provider: GPUManagerProtocol = NoopProvider()
SlingshotClient = NoopProvider  # backward-compatible alias

# ── Recovery globals (noop-safe) ─────────────────────────────────────

_RECOVERY_SLINGSHOT: GPUManagerProtocol | None = None
_recovery_registered: bool = False


def _atexit_wake() -> None:
    """No-op atexit handler — noop provider has nothing to recover."""
    pass


def _sigterm_handler(signum: int, frame: "object | None") -> None:
    """No-op SIGTERM handler — noop provider has nothing to recover."""
    raise SystemExit(128 + signum)


def _register_slingshot_recovery(client: GPUManagerProtocol) -> None:
    """No-op recovery registration — noop provider needs no recovery.

    Args:
        client: GPU provider instance (stored but not used for recovery).
    """
    global _RECOVERY_SLINGSHOT
    _RECOVERY_SLINGSHOT = client


def _sync_facade_recovery() -> None:
    """No-op — no pipeline module to sync in public repo."""
    pass


# ── Module-level entry points ────────────────────────────────────────

def is_local_llm_active() -> bool:
    """Check if local LLM is active — always False in public repo."""
    return _active_provider.is_local_llm_active()
