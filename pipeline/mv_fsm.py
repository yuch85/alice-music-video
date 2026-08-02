#!/usr/bin/env python3
"""Finite-state machine for the music video pre-production workflow.

Pure logic module — no file I/O. All state lives in-memory.
The project skill (Task 2) handles persistence to index.md.

13-stage canonical order:
INTERVIEW -> TREATMENT -> CONTINUITY -> AUDIO_ANALYSIS -> BEATS -> STORYBOARD -> SHOTS
-> IMAGE_APPROVAL -> PROMPTS -> VALIDATED -> GENERATING -> QC -> COMPLETE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MVStage(Enum):
    """Canonical 13-stage order for the music video pre-production workflow."""

    INTERVIEW = "INTERVIEW"
    TREATMENT = "TREATMENT"
    CONTINUITY = "CONTINUITY"
    AUDIO_ANALYSIS = "AUDIO_ANALYSIS"
    BEATS = "BEATS"
    STORYBOARD = "STORYBOARD"
    SHOTS = "SHOTS"
    IMAGE_APPROVAL = "IMAGE_APPROVAL"
    PROMPTS = "PROMPTS"
    VALIDATED = "VALIDATED"
    GENERATING = "GENERATING"
    QC = "QC"
    COMPLETE = "COMPLETE"


# Ordered list for index lookups
STAGE_ORDER: list[MVStage] = list(MVStage)

# Forward-transition map: each stage maps to its next stage
TRANSITIONS: dict[MVStage, MVStage] = {
    STAGE_ORDER[i]: STAGE_ORDER[i + 1]
    for i in range(len(STAGE_ORDER) - 1)
}

# Valid status values
_STATUS_APPROVED = "APPROVED"
_STATUS_COMPLETE = "COMPLETE"
_STATUS_REJECTED = "REJECTED"
_STATUS_IN_PROGRESS = "IN_PROGRESS"
_STATUS_NOT_STARTED = "NOT_STARTED"


class InvalidTransition(ValueError):
    """Raised when an FSM state transition is not allowed.

    Message includes current state, attempted state, and required prerequisite.
    """


@dataclass(frozen=True)
class _FSMState:
    """Immutable snapshot of FSM state for serialization."""

    current: str
    statuses: dict[str, str]


class MusicVideoFSM:
    """Finite-state machine for the music video pre-production workflow.

    Enforces stage gating: no downstream stage executes without approved
    prerequisites. Supports rollback for recovery from rejected stages.
    """

    def __init__(self) -> None:
        self._current: MVStage = MVStage.INTERVIEW
        self._statuses: dict[MVStage, str] = {
            stage: _STATUS_NOT_STARTED for stage in MVStage
        }

    def get_current(self) -> MVStage:
        """Return the current FSM stage."""
        return self._current

    def can_transition_to(self, target: MVStage) -> bool:
        """Check if *target* is the next state after current."""
        expected_next = TRANSITIONS.get(self._current)
        return expected_next == target

    def is_stage_complete(self, stage: MVStage) -> bool:
        """Return True if *stage* status is APPROVED or COMPLETE."""
        return self._statuses.get(stage) in (
            _STATUS_APPROVED,
            _STATUS_COMPLETE,
        )

    def get_status(self, stage: MVStage) -> str:
        """Return the status string for *stage* (default NOT_STARTED)."""
        return self._statuses.get(stage, _STATUS_NOT_STARTED)

    def set_status(self, stage: MVStage, status: str) -> None:
        """Set the status for *stage*."""
        self._statuses[stage] = status

    def transition(self, stage: MVStage, status: str) -> MVStage:
        """Advance state if *stage* matches current and *status* is APPROVED.

        If *status* is REJECTED, stay in current state.
        Raise ``InvalidTransition`` if *stage* doesn't match current state
        or if trying to skip stages.
        """
        if stage != self._current:
            required = TRANSITIONS.get(self._current)
            raise InvalidTransition(
                f"Cannot transition {stage.value}: current stage is "
                f"{self._current.value}, expected {required.value if required else 'none'}"
            )

        self._statuses[stage] = status

        if status != _STATUS_APPROVED:
            return self._current

        next_stage = TRANSITIONS.get(stage)
        if next_stage is None:
            # Already at the last stage (COMPLETE)
            return self._current

        self._current = next_stage
        self._statuses[next_stage] = _STATUS_IN_PROGRESS
        return next_stage

    def rollback(
        self, target_stage: MVStage | None = None
    ) -> MVStage:
        """Roll back to *target_stage* (or one stage back if None).

        All stages strictly after the target are reset to NOT_STARTED.
        The target stage is set to IN_PROGRESS.
        Raise ``InvalidTransition`` if target is ahead of current state.
        """
        if target_stage is None:
            # Roll back one stage from current
            current_idx = STAGE_ORDER.index(self._current)
            if current_idx == 0:
                raise InvalidTransition(
                    f"Cannot roll back from {self._current.value}: already at first stage"
                )
            target_stage = STAGE_ORDER[current_idx - 1]

        target_idx = STAGE_ORDER.index(target_stage)
        current_idx = STAGE_ORDER.index(self._current)

        if target_idx > current_idx:
            raise InvalidTransition(
                f"Cannot roll back to {target_stage.value}: current stage is "
                f"{self._current.value} (target is ahead)"
            )

        # Reset all stages strictly after target to NOT_STARTED
        for i in range(target_idx + 1, len(STAGE_ORDER)):
            self._statuses[STAGE_ORDER[i]] = _STATUS_NOT_STARTED

        # Set target to IN_PROGRESS
        self._statuses[target_stage] = _STATUS_IN_PROGRESS
        self._current = target_stage
        return target_stage

    def get_completed_stages(self) -> list[MVStage]:
        """Return list of stages with APPROVED or COMPLETE status."""
        return [
            stage
            for stage in STAGE_ORDER
            if self.is_stage_complete(stage)
        ]

    def get_next_pending_stage(self) -> MVStage | None:
        """Return first stage with NOT_STARTED or IN_PROGRESS status."""
        for stage in STAGE_ORDER:
            if self._statuses.get(stage) in (
                _STATUS_NOT_STARTED,
                _STATUS_IN_PROGRESS,
            ):
                return stage
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize state + statuses to a plain dict for persistence."""
        return {
            "current": self._current.value,
            "statuses": {s.value: st for s, st in self._statuses.items()},
        }

    def from_dict(self, data: dict[str, object]) -> None:
        """Deserialize from a dict produced by ``to_dict()``."""
        current_str = str(data.get("current", MVStage.INTERVIEW.value))
        self._current = MVStage(current_str)

        raw_statuses = data.get("statuses", {})
        if isinstance(raw_statuses, dict):
            self._statuses = {
                MVStage(k): str(v)
                for k, v in raw_statuses.items()
            }
        # Fill any missing stages
        for stage in MVStage:
            if stage not in self._statuses:
                self._statuses[stage] = _STATUS_NOT_STARTED
