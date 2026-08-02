#!/usr/bin/env python3
"""Tests for mv_fsm — FSM state machine module for music video pre-production."""

from __future__ import annotations

import pytest

from mv_fsm import InvalidTransition, MVStage, MusicVideoFSM


# ── Test 1: Initial state ────────────────────────────────────────────────

def test_initial_state_is_interview() -> None:
    """Creating FSM returns INTERVIEW as the initial state."""
    fsm = MusicVideoFSM()
    assert fsm.get_current() == MVStage.INTERVIEW


# ── Test 2: Forward transition ──────────────────────────────────────────

def test_transition_approved_moves_forward() -> None:
    """Transition from INTERVIEW with status=APPROVED moves to TREATMENT."""
    fsm = MusicVideoFSM()
    next_stage = fsm.transition(MVStage.INTERVIEW, "APPROVED")
    assert next_stage == MVStage.TREATMENT
    assert fsm.get_current() == MVStage.TREATMENT


# ── Test 3: Prerequisite enforcement ────────────────────────────────────

def test_transition_skipping_stages_raises() -> None:
    """Transition from BEATS without CONTINUITY=APPROVED raises InvalidTransition."""
    fsm = MusicVideoFSM()
    fsm.transition(MVStage.INTERVIEW, "APPROVED")
    with pytest.raises(InvalidTransition):
        fsm.transition(MVStage.BEATS, "APPROVED")


# ── Test 4: Rejected stays ──────────────────────────────────────────────

def test_transition_rejected_stays() -> None:
    """Transition with status=REJECTED stays in current state."""
    fsm = MusicVideoFSM()
    next_stage = fsm.transition(MVStage.INTERVIEW, "REJECTED")
    assert next_stage == MVStage.INTERVIEW
    assert fsm.get_current() == MVStage.INTERVIEW


# ── Test 5: is_stage_complete ───────────────────────────────────────────

def test_is_stage_complete() -> None:
    """is_stage_complete returns True only for APPROVED or COMPLETE statuses."""
    fsm = MusicVideoFSM()
    assert fsm.is_stage_complete(MVStage.INTERVIEW) is False
    fsm.set_status(MVStage.INTERVIEW, "IN_PROGRESS")
    assert fsm.is_stage_complete(MVStage.INTERVIEW) is False
    fsm.set_status(MVStage.INTERVIEW, "APPROVED")
    assert fsm.is_stage_complete(MVStage.INTERVIEW) is True
    fsm.set_status(MVStage.INTERVIEW, "COMPLETE")
    assert fsm.is_stage_complete(MVStage.INTERVIEW) is True
    fsm.set_status(MVStage.INTERVIEW, "REJECTED")
    assert fsm.is_stage_complete(MVStage.INTERVIEW) is False


# ── Test 6: Non-adjacent transition check ───────────────────────────────

def test_can_transition_to_non_adjacent() -> None:
    """can_transition_to returns False for non-adjacent states."""
    fsm = MusicVideoFSM()
    assert fsm.can_transition_to(MVStage.BEATS) is False
    assert fsm.can_transition_to(MVStage.TREATMENT) is True


# ── Test 7: Serialization round-trip ────────────────────────────────────

def test_serialization_roundtrip() -> None:
    """State serialization to/from dict round-trips correctly."""
    fsm = MusicVideoFSM()
    fsm.transition(MVStage.INTERVIEW, "APPROVED")
    fsm.transition(MVStage.TREATMENT, "APPROVED")
    fsm.set_status(MVStage.CONTINUITY, "IN_PROGRESS")

    data = fsm.to_dict()
    fsm2 = MusicVideoFSM()
    fsm2.from_dict(data)

    assert fsm2.get_current() == fsm.get_current()
    for stage in MVStage:
        assert fsm2.get_status(stage) == fsm.get_status(stage)


# ── Test 8: Rollback one stage ──────────────────────────────────────────

def test_rollback_one_stage() -> None:
    """Rollback from SHOTS resets SHOTS to IN_PROGRESS and downstream to NOT_STARTED."""
    fsm = MusicVideoFSM()
    for stage in [MVStage.INTERVIEW, MVStage.TREATMENT, MVStage.CONTINUITY,
                  MVStage.BEATS, MVStage.STORYBOARD]:
        fsm.transition(stage, "APPROVED")
    fsm.set_status(MVStage.SHOTS, "IN_PROGRESS")
    fsm.set_status(MVStage.IMAGE_APPROVAL, "IN_PROGRESS")
    fsm.set_status(MVStage.PROMPTS, "APPROVED")

    result = fsm.rollback()
    assert result == MVStage.STORYBOARD
    assert fsm.get_current() == MVStage.STORYBOARD
    assert fsm.get_status(MVStage.STORYBOARD) == "IN_PROGRESS"
    assert fsm.get_status(MVStage.SHOTS) == "NOT_STARTED"
    assert fsm.get_status(MVStage.IMAGE_APPROVAL) == "NOT_STARTED"
    assert fsm.get_status(MVStage.PROMPTS) == "NOT_STARTED"


# ── Test 9: Rollback to specific stage ──────────────────────────────────

def test_rollback_to_named_stage() -> None:
    """Rollback to BEATS resets all downstream stages from BEATS onward."""
    fsm = MusicVideoFSM()
    for stage in [MVStage.INTERVIEW, MVStage.TREATMENT, MVStage.CONTINUITY,
                  MVStage.BEATS, MVStage.STORYBOARD, MVStage.SHOTS]:
        fsm.transition(stage, "APPROVED")
    fsm.set_status(MVStage.IMAGE_APPROVAL, "IN_PROGRESS")
    fsm.set_status(MVStage.PROMPTS, "APPROVED")
    fsm.set_status(MVStage.VALIDATED, "APPROVED")

    result = fsm.rollback(MVStage.BEATS)
    assert result == MVStage.BEATS
    assert fsm.get_current() == MVStage.BEATS
    assert fsm.get_status(MVStage.BEATS) == "IN_PROGRESS"
    assert fsm.get_status(MVStage.STORYBOARD) == "NOT_STARTED"
    assert fsm.get_status(MVStage.SHOTS) == "NOT_STARTED"
    assert fsm.get_status(MVStage.IMAGE_APPROVAL) == "NOT_STARTED"
    assert fsm.get_status(MVStage.PROMPTS) == "NOT_STARTED"
    assert fsm.get_status(MVStage.VALIDATED) == "NOT_STARTED"
    assert fsm.get_status(MVStage.GENERATING) == "NOT_STARTED"


# ── Test 10: Rollback ahead raises ──────────────────────────────────────

def test_rollback_ahead_raises() -> None:
    """Rollback to a stage ahead of current raises InvalidTransition."""
    fsm = MusicVideoFSM()
    with pytest.raises(InvalidTransition):
        fsm.rollback(MVStage.BEATS)
