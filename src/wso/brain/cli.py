"""Command line interface for the autonomous Adapter Trust Gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import cast

from rich.console import Console

from wso.brain.benchmark import run_brain_demo
from wso.brain.runtime import (
    ACTION_LEDGER,
    DECISION_LEDGER,
    FEEDBACK_LEDGER,
    LOCK_FILENAME,
    PRIVATE_LEDGER,
    STATE_FILENAME,
    AutonomousWeightSpaceBrain,
)
from wso.brain.self_play import run_defensive_self_play
from wso.brain.types import ACTION_SEVERITY, BrainConfig, BrainFeedback, BrainMode
from wso.telemetry.schemas import TransientLoRATelemetry

console = Console()
error_console = Console(stderr=True)
MAX_BRAIN_TELEMETRY_RECORDS = 4096
MAX_BRAIN_TELEMETRY_FILE_BYTES = 32 * 1024 * 1024
MAX_BRAIN_TELEMETRY_LINE_BYTES = 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wso-brain",
        description="Learned autonomous control plane for weight-space telemetry.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="Initialize from a trusted benign baseline and labeled outcomes"
    )
    bootstrap.add_argument("--state-dir", type=Path, required=True)
    bootstrap.add_argument("--baseline", type=Path, required=True)
    bootstrap.add_argument("--labeled", type=Path, default=None)
    bootstrap.add_argument("--config", type=Path, default=None, help="Optional BrainConfig JSON")

    decide = subparsers.add_parser("decide", help="Score telemetry and choose bounded responses")
    decide.add_argument("--state-dir", type=Path, required=True)
    decide.add_argument("--telemetry", type=Path, required=True)
    decide.add_argument("--mode", choices=["shadow", "enforce"], default="shadow")
    decide.add_argument("--out", type=Path, default=None)

    feedback = subparsers.add_parser(
        "feedback", help="Apply a verified incident or benign disposition"
    )
    feedback.add_argument("--state-dir", type=Path, required=True)
    feedback.add_argument("--decision-id", required=True)
    feedback.add_argument(
        "--feedback-id",
        default=None,
        help="Stable transaction ID to retain and reuse for an exact crash retry",
    )
    outcome = feedback.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--incident", action="store_true")
    outcome.add_argument("--benign", action="store_true")
    feedback.add_argument("--impact", type=float, default=1.0)
    feedback.add_argument("--actual-action", choices=sorted(ACTION_SEVERITY), default=None)
    action_result = feedback.add_mutually_exclusive_group()
    action_result.add_argument("--action-succeeded", action="store_true")
    action_result.add_argument("--action-failed", action="store_true")
    containment = feedback.add_mutually_exclusive_group()
    containment.add_argument("--contained", action="store_true")
    containment.add_argument("--not-contained", action="store_true")
    feedback.add_argument("--realized-cost", type=float, default=0.0)
    feedback.add_argument("--disruption", type=float, default=0.0)
    feedback.add_argument("--latency-ms", type=float, default=0.0)
    feedback.add_argument("--note", default=None)
    feedback.add_argument(
        "--unverified",
        action="store_true",
        help="Log feedback without changing any learned state",
    )

    status = subparsers.add_parser("status", help="Show readiness and learning state")
    status.add_argument("--state-dir", type=Path, required=True)

    reconcile = subparsers.add_parser(
        "reconcile-action",
        help="Resolve an orphaned action intent using an operator attestation",
    )
    reconcile.add_argument("--state-dir", type=Path, required=True)
    reconcile.add_argument("--telemetry", type=Path, required=True)
    reconcile.add_argument("--mode", choices=["shadow", "enforce"], required=True)
    resolution = reconcile.add_mutually_exclusive_group(required=True)
    resolution.add_argument(
        "--attest-not-executed",
        action="store_true",
        help="Attest the external action did not occur, permitting one safe retry",
    )
    resolution.add_argument(
        "--keep-blocked",
        action="store_true",
        help="Keep the observation permanently blocked because execution is uncertain",
    )

    self_play = subparsers.add_parser(
        "self-play", help="Run sealed feature-space defensive hardening"
    )
    self_play.add_argument("--state-dir", type=Path, required=True)
    self_play.add_argument("--incidents", type=Path, required=True)
    self_play.add_argument("--benign-holdout", type=Path, required=True)
    self_play.add_argument("--candidates", type=int, default=48)
    self_play.add_argument("--epsilon", type=float, default=0.18)
    self_play.add_argument(
        "--incident-dataset-hash",
        default=None,
        help="Expected canonical SHA-256 of the frozen incident set (required to promote)",
    )
    self_play.add_argument(
        "--benign-holdout-hash",
        default=None,
        help="Expected canonical SHA-256 of the frozen benign holdout (required to promote)",
    )
    self_play.add_argument("--no-promote", action="store_true")
    self_play.add_argument("--out", type=Path, default=None)

    demo = subparsers.add_parser(
        "demo", help="Run an offline end-to-end fixture with honest claim boundaries"
    )
    demo.add_argument("--out-dir", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=20260801)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary must render all failures safely.
        error_console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "bootstrap":
        config = _load_config(args.config) if args.config else BrainConfig()
        brain = AutonomousWeightSpaceBrain(args.state_dir, config=config)
        baseline = _load_brain_telemetry(args.baseline)
        labeled = _load_brain_telemetry(args.labeled) if args.labeled else []
        bootstrap_result = brain.bootstrap(baseline=baseline, labeled=labeled)
        _print_json(bootstrap_result.model_dump())
        return 0

    if args.command == "decide":
        brain = AutonomousWeightSpaceBrain(args.state_dir)
        records = _load_brain_telemetry(args.telemetry)
        if not records:
            raise ValueError("decide requires at least one telemetry record")
        if args.out:
            _validate_output_path(
                args.out,
                args.state_dir,
                inputs=[args.telemetry],
            )
        decisions = [brain.decide(record, mode=args.mode) for record in records]
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            rendered = "".join(decision.model_dump_json() + "\n" for decision in decisions)
            _atomic_write_text(
                args.out,
                rendered,
                state_dir=args.state_dir,
                inputs=[args.telemetry],
            )
        _print_json(
            {
                "decision_count": len(decisions),
                "mode": args.mode,
                "decisions": [decision.model_dump() for decision in decisions],
            }
        )
        return 0

    if args.command == "feedback":
        brain = AutonomousWeightSpaceBrain(args.state_dir)
        feedback_payload: dict[str, object] = {
            "decision_id": args.decision_id,
            "incident": bool(args.incident),
            "verified": not args.unverified,
            "impact": args.impact,
            "actual_action": args.actual_action,
            "action_succeeded": _optional_flag(
                positive=args.action_succeeded,
                negative=args.action_failed,
            ),
            "containment_succeeded": _optional_flag(
                positive=args.contained,
                negative=args.not_contained,
            ),
            "realized_cost": args.realized_cost,
            "disruption": args.disruption,
            "latency_ms": args.latency_ms,
            "note": args.note,
        }
        if args.feedback_id is not None:
            feedback_payload["feedback_id"] = args.feedback_id
        feedback_result = brain.feedback(BrainFeedback.model_validate(feedback_payload))
        _print_json(feedback_result.model_dump())
        return 0

    if args.command == "status":
        brain = AutonomousWeightSpaceBrain(args.state_dir)
        _print_json(brain.status().model_dump())
        return 0

    if args.command == "reconcile-action":
        brain = AutonomousWeightSpaceBrain(args.state_dir)
        records = _load_brain_telemetry(args.telemetry)
        if len(records) != 1:
            raise ValueError("reconcile-action requires exactly one telemetry record")
        reconciliation = brain.reconcile_action_attempt(
            records[0],
            mode=cast(BrainMode, args.mode),
            attest_not_executed=bool(args.attest_not_executed),
        )
        _print_json(reconciliation)
        return 0

    if args.command == "self-play":
        brain = AutonomousWeightSpaceBrain(args.state_dir)
        if args.out:
            _validate_output_path(
                args.out,
                args.state_dir,
                inputs=[args.incidents, args.benign_holdout],
            )
        self_play_result = run_defensive_self_play(
            brain,
            incidents=_load_brain_telemetry(args.incidents),
            benign_holdout=_load_brain_telemetry(args.benign_holdout),
            candidates_per_incident=args.candidates,
            epsilon=args.epsilon,
            promote=not args.no_promote,
            expected_incident_dataset_hash=args.incident_dataset_hash,
            expected_benign_holdout_hash=args.benign_holdout_hash,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                args.out,
                self_play_result.model_dump_json(indent=2) + "\n",
                state_dir=args.state_dir,
                inputs=[args.incidents, args.benign_holdout],
            )
        _print_json(self_play_result.model_dump())
        return 0

    if args.command == "demo":
        demo_result = run_brain_demo(args.out_dir, seed=args.seed)
        _print_json(demo_result)
        return 0

    raise ValueError(f"unsupported command: {args.command}")


def _load_config(path: Path) -> BrainConfig:
    return BrainConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _print_json(payload: object) -> None:
    console.print_json(json.dumps(payload, sort_keys=True))


def _optional_flag(*, positive: bool, negative: bool) -> bool | None:
    if positive:
        return True
    if negative:
        return False
    return None


def _validate_output_path(
    path: Path,
    state_dir: Path,
    *,
    inputs: list[Path] | None = None,
) -> None:
    resolved_path = path.resolve()
    resolved_state_dir = state_dir.resolve()
    if resolved_path == resolved_state_dir or resolved_state_dir in resolved_path.parents:
        raise ValueError("--out cannot write anywhere inside the brain state directory")
    protected = [
        state_dir / filename
        for filename in (
            STATE_FILENAME,
            DECISION_LEDGER,
            PRIVATE_LEDGER,
            FEEDBACK_LEDGER,
            ACTION_LEDGER,
            LOCK_FILENAME,
        )
    ]
    if any(_paths_alias(path, candidate) for candidate in protected):
        raise ValueError("--out cannot overwrite brain state or an audit ledger")
    if any(_paths_alias(path, candidate) for candidate in inputs or []):
        raise ValueError("--out cannot overwrite an input telemetry file")


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def _atomic_write_text(
    path: Path,
    content: str,
    *,
    state_dir: Path,
    inputs: list[Path] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve()
    parent_descriptor = _open_directory(resolved_parent)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        _validate_pinned_output(
            parent_descriptor,
            path.name,
            state_dir=state_dir,
            inputs=inputs or [],
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_pinned_output(
            parent_descriptor,
            path.name,
            state_dir=state_dir,
            inputs=inputs or [],
        )
        os.replace(
            temporary,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _open_directory(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )


def _validate_pinned_output(
    parent_descriptor: int,
    filename: str,
    *,
    state_dir: Path,
    inputs: list[Path],
) -> None:
    if _directory_fd_is_within(parent_descriptor, state_dir.resolve()):
        raise ValueError("--out cannot write anywhere inside the brain state directory")
    for candidate in inputs:
        resolved = candidate.resolve()
        try:
            candidate_parent = _open_directory(resolved.parent)
        except FileNotFoundError:
            continue
        try:
            same_parent = _directory_identity(parent_descriptor) == _directory_identity(
                candidate_parent
            )
        finally:
            os.close(candidate_parent)
        if same_parent and filename == resolved.name:
            raise ValueError("--out cannot overwrite an input telemetry file")


def _directory_fd_is_within(child_descriptor: int, ancestor: Path) -> bool:
    try:
        ancestor_descriptor = _open_directory(ancestor)
    except FileNotFoundError:
        return False
    current = os.dup(child_descriptor)
    try:
        ancestor_identity = _directory_identity(ancestor_descriptor)
        for _ in range(256):
            current_identity = _directory_identity(current)
            if current_identity == ancestor_identity:
                return True
            parent = os.open(
                "..",
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
                dir_fd=current,
            )
            parent_identity = _directory_identity(parent)
            if parent_identity == current_identity:
                os.close(parent)
                return False
            os.close(current)
            current = parent
        return True
    finally:
        os.close(current)
        os.close(ancestor_descriptor)


def _directory_identity(descriptor: int) -> tuple[int, int]:
    status = os.fstat(descriptor)
    return status.st_dev, status.st_ino


def _load_brain_telemetry(path: Path) -> list[TransientLoRATelemetry]:
    if path.stat().st_size > MAX_BRAIN_TELEMETRY_FILE_BYTES:
        raise ValueError(f"brain telemetry input exceeds {MAX_BRAIN_TELEMETRY_FILE_BYTES} bytes")
    records: list[TransientLoRATelemetry] = []
    total_bytes = 0
    with path.open("rb") as handle:
        line_no = 0
        while True:
            line = handle.readline(MAX_BRAIN_TELEMETRY_LINE_BYTES + 1)
            if not line:
                break
            line_no += 1
            total_bytes += len(line)
            if len(line) > MAX_BRAIN_TELEMETRY_LINE_BYTES:
                raise ValueError(
                    f"brain telemetry line {line_no} exceeds {MAX_BRAIN_TELEMETRY_LINE_BYTES} bytes"
                )
            if total_bytes > MAX_BRAIN_TELEMETRY_FILE_BYTES:
                raise ValueError(
                    f"brain telemetry input exceeds {MAX_BRAIN_TELEMETRY_FILE_BYTES} bytes"
                )
            stripped = line.strip()
            if not stripped:
                continue
            if len(records) >= MAX_BRAIN_TELEMETRY_RECORDS:
                raise ValueError(
                    f"brain telemetry input exceeds {MAX_BRAIN_TELEMETRY_RECORDS} records"
                )
            try:
                records.append(TransientLoRATelemetry.model_validate_json(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid telemetry JSONL at line {line_no}: {exc}") from exc
    return records


if __name__ == "__main__":
    sys.exit(main())
