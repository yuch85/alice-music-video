#!/usr/bin/env python3
"""No-op GPU provider — safe fallback for public repository and cloud sessions.

Implements GPUManagerProtocol with safe no-op defaults. Used when:
- The public alice-music-video repository runs (no gpu-manager available)
- Cloud-LLM sessions (alia-c / ishi-c) where no local LLM exists
- MV_GPU_PROVIDER=noop environment variable is set

All methods log INFO-level messages so operators can verify the provider
is active when GPU recovery is expected but no-op is configured.
"""

from __future__ import annotations

import logging
from typing import Any

from mv_gpu_interface import GPUManagerProtocol

logger = logging.getLogger(__name__)


class NoopProvider(GPUManagerProtocol):
    """GPU manager implementation that performs no GPU state changes.

    All methods return safe defaults and log INFO-level messages
    for operator visibility (threat mitigation T-09.9-34-01-R).
    """

    def status(self) -> dict[str, Any] | None:
        """Return idle status — no GPU manager available.

        Returns:
            Safe default status dict indicating idle state.
        """
        logger.info("GPU provider (noop): status — idle, no gpu-manager")
        return {"state": "idle", "llm_active": False}

    def hibernate(self) -> bool:
        """No-op hibernate — nothing to hibernate.

        Returns:
            False (nothing to hibernate).
        """
        logger.info("GPU provider (noop): hibernate — skipped, no gpu-manager")
        return False

    def wake(
        self,
        *,
        task_name: str = "music_video",
        output_path: str = "",
    ) -> dict[str, Any] | None:
        """No-op wake — nothing to wake.

        Args:
            task_name: Task identifier (unused by no-op provider).
            output_path: Output file path (unused by no-op provider).

        Returns:
            Safe default wake result dict.
        """
        logger.info(
            "GPU provider (noop): wake — skipped (task=%s), no gpu-manager",
            task_name,
        )
        return {"ok": True, "state": "idle"}

    def ensure_wake(
        self,
        *,
        task_name: str = "music_video",
        output_path: str = "",
    ) -> dict[str, Any] | None:
        """No-op ensure_wake — nothing to ensure.

        Args:
            task_name: Task identifier (unused by no-op provider).
            output_path: Output file path (unused by no-op provider).

        Returns:
            None (no-op, nothing to wake).
        """
        logger.info(
            "GPU provider (noop): ensure_wake — skipped (task=%s), no gpu-manager",
            task_name,
        )
        return None

    def ensure_hibernate(self) -> bool:
        """No-op ensure_hibernate — nothing to hibernate.

        Returns:
            False (nothing to hibernate).
        """
        logger.info("GPU provider (noop): ensure_hibernate — skipped, no gpu-manager")
        return False

    def is_local_llm_active(self) -> bool:
        """Check if local LLM is active — always False for no-op provider.

        Returns:
            False (no local LLM in no-op mode).
        """
        logger.info("GPU provider (noop): is_local_llm_active — False, no gpu-manager")
        return False
