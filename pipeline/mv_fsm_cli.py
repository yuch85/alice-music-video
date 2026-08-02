#!/usr/bin/env python3
"""CLI wrapper around the FSM module for state transitions.

All subordinate skills call this CLI to change FSM states instead of
implementing fragile regex overrides on the markdown file directly.

Usage:
    mv_fsm_cli.py init <project_dir>
    mv_fsm_cli.py status <project_dir>
    mv_fsm_cli.py transition <project_dir> <stage> <status>
    mv_fsm_cli.py get <project_dir>
    mv_fsm_cli.py set-status <project_dir> <stage> <status>
    mv_fsm_cli.py rollback <project_dir> [--to <stage>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mv_fsm import InvalidTransition, MVStage
from mv_fsm_persist import get_template, load_fsm, save_fsm

_VALID_STATUSES = ("APPROVED", "REJECTED", "IN_PROGRESS", "NOT_STARTED", "COMPLETE")
_SUBDIRS = ("lyrics", "refs", "concepts", "prompts", "clips", "final", "approvals")


# ── Commands ──────────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize FSM state to INTERVIEW, create index.md from template."""
    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    index_path = project_dir / "index.md"
    if index_path.exists():
        print(
            f"Warning: {index_path} already exists. Use 'status' to check.",
            file=sys.stderr,
        )

    for subdir in _SUBDIRS:
        (project_dir / subdir).mkdir(exist_ok=True)

    project_name = project_dir.name.replace("-", " ").replace("_", " ").title()
    index_path.write_text(get_template(project_name))
    print(f"Initialized project '{project_name}' in {project_dir}")
    print("Current stage: INTERVIEW")


def cmd_status(args: argparse.Namespace) -> None:
    """Print current FSM stage and per-stage status table."""
    fsm, _, _ = load_fsm(Path(args.project_dir))
    current = fsm.get_current()
    print(f"Current stage: {current.value}\n")
    print(f"{'Stage':<20} {'Status':<15}")
    print("-" * 35)
    for stage in MVStage:
        marker = " <--" if stage == current else ""
        print(f"{stage.value:<20} {fsm.get_status(stage):<15}{marker}")


def cmd_transition(args: argparse.Namespace) -> None:
    """Transition a stage (APPROVED/REJECTED), update index.md."""
    status = args.status.upper()
    if status not in _VALID_STATUSES:
        print(f"Error: invalid status '{status}'.", file=sys.stderr)
        sys.exit(1)

    fsm, index_path, content = load_fsm(Path(args.project_dir))
    try:
        next_stage = fsm.transition(MVStage(args.stage), status)
        save_fsm(fsm, index_path, content)
        print(f"Transitioned {args.stage} -> {status}. Current: {next_stage.value}")
    except InvalidTransition as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_get(args: argparse.Namespace) -> None:
    """Return current stage as a single line."""
    fsm, _, _ = load_fsm(Path(args.project_dir))
    print(fsm.get_current().value)


def cmd_set_status(args: argparse.Namespace) -> None:
    """Update a stage's status without transitioning."""
    status = args.status.upper()
    if status not in _VALID_STATUSES:
        print(f"Error: invalid status '{status}'.", file=sys.stderr)
        sys.exit(1)

    fsm, index_path, content = load_fsm(Path(args.project_dir))
    fsm.set_status(MVStage(args.stage), status)
    save_fsm(fsm, index_path, content)
    print(f"Set {args.stage} -> {status}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """Roll back current stage, resetting downstream stages to NOT_STARTED."""
    fsm, index_path, content = load_fsm(Path(args.project_dir))
    target: MVStage | None = MVStage(args.to) if args.to else None
    try:
        result = fsm.rollback(target)
        save_fsm(fsm, index_path, content)
        msg = f"Rolled back to {result.value}."
        if target:
            msg += " All downstream stages reset to NOT_STARTED."
        print(msg)
    except InvalidTransition as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


# ── CLI entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="FSM state manager for music video pre-production"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("project_dir")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("status")
    p.add_argument("project_dir")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("transition")
    p.add_argument("project_dir")
    p.add_argument("stage")
    p.add_argument("status")
    p.set_defaults(func=cmd_transition)

    p = sub.add_parser("get")
    p.add_argument("project_dir")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("set-status")
    p.add_argument("project_dir")
    p.add_argument("stage")
    p.add_argument("status")
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("rollback")
    p.add_argument("project_dir")
    p.add_argument("--to", default=None)
    p.set_defaults(func=cmd_rollback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
