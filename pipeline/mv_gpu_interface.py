#!/usr/bin/env python3
"""GPU manager protocol — abstraction over GPU management endpoints.

Defines the interface that all GPU provider implementations must satisfy.
Used by the music video pipeline to manage GPU state (hibernate/wake local
LLM) without hard dependencies on private infrastructure.

Provider selection is configuration-driven (MV_GPU_PROVIDER env var).
The pipeline only imports this protocol; concrete providers are resolved
through the mv_gpu_manager facade.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GPUManagerProtocol(Protocol):
    """GPU manager interface — hibernate/wake local LLM for VRAM management.

    Implementations:
    - SlingshotProvider: real gpu-manager HTTP client (private Alice repo)
    - NoopProvider: safe no-ops (public repo, cloud-LLM sessions)
    """

    def status(self) -> dict[str, Any] | None:
        """GET current GPU manager state.

        Returns:
            Dict with 'state', 'llm_active' keys, or None on error.
        """
        ...

    def hibernate(self) -> bool:
        """Hibernate the local LLM to free VRAM for GPU-intensive tasks.

        Returns:
            True if hibernated, False if already idle or error.
        """
        ...

    def wake(
        self,
        *,
        task_name: str = "music_video",
        output_path: str = "",
    ) -> dict[str, Any] | None:
        """Wake the local LLM after GPU-intensive tasks complete.

        Args:
            task_name: Task identifier for wake context (default "music_video").
            output_path: Optional output file path for wake prompt generation.

        Returns:
            Dict with 'ok' key on success, None on error.
        """
        ...

    def ensure_wake(
        self,
        *,
        task_name: str = "music_video",
        output_path: str = "",
    ) -> dict[str, Any] | None:
        """Idempotent wake — wakes only if LLM is hibernating or down.

        Safe to call unconditionally on all exit paths (finally, atexit,
        SIGTERM). No-op if LLM is already running.

        Args:
            task_name: Task identifier for wake context.
            output_path: Optional output file path for wake prompt generation.

        Returns:
            Dict with wake result, or None if already awake.
        """
        ...

    def ensure_hibernate(self) -> bool:
        """Hibernate with stale-state recovery.

        Recovers from stale hibernating/hibernated state before attempting
        to hibernate. Returns True if hibernated successfully.

        Returns:
            True if hibernated, False otherwise.
        """
        ...

    def is_local_llm_active(self) -> bool:
        """Check if a local LLM is currently provisioned and active.

        Returns:
            True if local LLM is running, False otherwise.
        """
        ...
