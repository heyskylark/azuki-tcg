#!/usr/bin/env python3
"""Read-only Discord observer for the registered +30M control continuation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import monitor_strategy_discovery as observer


SCHEMA_ID = "azuki.control_continuation"
SCHEMA_VERSION = 1
STAGES = (
    ("step10m", 10_000_000),
    ("step20m", 20_000_000),
    ("step30m", 30_000_000),
)
PANELS = tuple(stage for stage, _ in STAGES)
EXPECTED_GAMES_PER_PANEL = 2_700
EXPECTED_GAMES = 8_100
EXPECTED_OPPONENTS = {"p021000", "p060000", "control_start"}
COMPLETE = observer.COMPLETE

_base_process_identity = observer.process_identity
_base_message = observer.message


def process_identity(pid: int) -> str | None:
    """Bind persisted observer state to both the PID and its Linux start time."""
    start_time = _base_process_identity(pid)
    return None if start_time is None else f"{pid}:{start_time}"


def _json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return payload


def _validate_plan(plan: dict) -> None:
    if plan.get("schema_id") != SCHEMA_ID or plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported control continuation experiment schema")

    continuation = plan.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("Continuation registration is missing")
    registered_stages = continuation.get("stages")
    expected_stages = [
        {"id": stage, "additional_learner_steps": target}
        for stage, target in STAGES
    ]
    if (continuation.get("arm") != "control"
            or continuation.get("additional_learner_steps") != 30_000_000
            or registered_stages != expected_stages):
        raise ValueError("Unexpected control continuation stage registration")

    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Evaluation registration is missing")
    opponents = evaluation.get("opponents")
    if (evaluation.get("panels") != list(PANELS)
            or evaluation.get("games_per_checkpoint") != EXPECTED_GAMES_PER_PANEL
            or evaluation.get("expected_games") != EXPECTED_GAMES
            or not isinstance(opponents, dict)
            or set(opponents) != EXPECTED_OPPONENTS):
        raise ValueError("Unexpected continuation evaluation registration")
    for opponent_id, opponent in opponents.items():
        if (not isinstance(opponent, dict)
                or not isinstance(opponent.get("checkpoint"), str)
                or not Path(opponent["checkpoint"]).is_absolute()
                or not isinstance(opponent.get("checkpoint_sha256"), str)
                or len(opponent["checkpoint_sha256"]) != 64):
            raise ValueError(f"Invalid registered opponent {opponent_id}")


def snapshot(campaign: Path, diagnostic: Path | None, state: dict) -> dict:
    """Observe continuation artifacts without advancing or modifying the campaign."""
    del diagnostic
    campaign = campaign.resolve()
    plan_path = campaign / "experiment.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict):
        raise ValueError("Experiment registration must be a JSON object")
    _validate_plan(plan)

    registration_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    orchestration = _json_object(campaign / "orchestration_manifest.json")
    registered_hashes = orchestration.get("sha256")
    if (not isinstance(registered_hashes, dict)
            or registered_hashes.get(str(plan_path)) != registration_sha256):
        raise ValueError("Continuation experiment differs from its registered orchestration")
    if state.get("registration_sha256", registration_sha256) != registration_sha256:
        raise ValueError("Watcher belongs to another continuation registration")
    state["registration_sha256"] = registration_sha256

    parent = _json_object(campaign / "common_parent.json")
    common_step = int(parent["learner_global_step"])
    common_update = int(parent["update"])
    if common_step != 26_349_346 or common_update != 2_815:
        raise ValueError("Unexpected continuation parent state")

    activity: list[tuple[str, int, int]] = []

    def record_activity(path: Path) -> None:
        info = path.stat()
        activity.append((str(path), info.st_size, info.st_mtime_ns))

    status_path = campaign / "campaign_status.json"
    if status_path.is_file():
        status = _json_object(status_path)
        record_activity(status_path)
        phase = status.get("state")
        completed = status.get("completed", [])
        if not isinstance(phase, str) or not isinstance(completed, list):
            raise ValueError("Invalid continuation campaign status")
    else:
        # Registration can briefly precede the first atomic status write.
        status = {}
        phase = "pending_status"
        completed = []

    allowed_phases = {"pending_status", "pending", "registered", "initializing", "complete", "failed"}
    allowed_phases.update(f"training_control_{stage}" for stage in PANELS)
    allowed_phases.update(f"evaluating_{stage}" for stage in PANELS)
    if phase not in allowed_phases:
        raise ValueError("Unknown continuation campaign state")

    counters = state.setdefault("trace_counters", {})
    evaluation_panels: dict[str, int] = {}
    for panel in PANELS:
        panel_count = 0
        for trace in sorted((campaign / "evaluation" / panel).glob("*/*/*/shard*.jsonl")):
            panel_count += observer.count_records(trace, counters)
            record_activity(trace)
        evaluation_panels[panel] = panel_count
    evaluation_games = sum(evaluation_panels.values())

    latest_step = common_step
    latest_update = common_update
    completed_stages: list[str] = []
    cursors = state.setdefault("metric_cursors", {})
    run = campaign / "runs" / "control"
    for stage, _target in STAGES:
        result_path = run / "stages" / stage / "result.json"
        if result_path.is_file():
            result = _json_object(result_path)
            latest_step = max(latest_step, int(result["learner_global_step"]))
            latest_update = max(latest_update, int(result["update"]))
            completed_stages.append(stage)
            record_activity(result_path)
        metrics_path = run / "logs" / f"control_{stage}.jsonl"
        if metrics_path.is_file():
            latest = observer._latest_league_metrics(metrics_path, cursors)
            latest_step = max(latest_step, int(latest.get("global_step", common_step)))
            latest_update = max(latest_update, int(latest.get("epoch", common_update)))
            record_activity(metrics_path)
        console_path = run / "stages" / stage / "console.log"
        if console_path.is_file():
            record_activity(console_path)

    added_steps = latest_step - common_step
    if added_steps < 0:
        raise ValueError("Continuation progress precedes its registered parent")

    arm_state = "pending"
    if isinstance(phase, str) and phase.startswith("training_control_"):
        arm_state = "running"
    elif isinstance(phase, str) and phase.startswith("evaluating_"):
        arm_state = "evaluating"
    elif completed_stages:
        arm_state = f"{completed_stages[-1]}_complete"

    current_state = "running"
    if phase == "failed":
        current_state = "failed"
        arm_state = "failed"
    elif phase == "complete":
        result = _json_object(campaign / "result.json")
        if result.get("status") != "complete" or result.get("evaluation_games") != EXPECTED_GAMES:
            raise ValueError("Continuation completion lacks the full registered result")
        if evaluation_games != EXPECTED_GAMES or any(
                evaluation_panels[panel] != EXPECTED_GAMES_PER_PANEL for panel in PANELS):
            raise ValueError("Continuation completion lacks exact registered trace coverage")
        current_state = COMPLETE
        arm_state = "evaluated"

    fingerprint = hashlib.sha256(
        json.dumps(activity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "state": current_state,
        "phase": phase,
        "family": "bptt_league_experiment",
        "arms": [{
            "id": "control",
            "state": arm_state,
            "rows": added_steps,
            "target_rows": 30_000_000,
            "epoch": latest_update,
            "eval_games": evaluation_games,
        }],
        "diagnostic_complete": False,
        "expected_eval_games": EXPECTED_GAMES,
        "eval_games": evaluation_games,
        "evaluation_panels": evaluation_panels,
        "common_global_step": common_step,
        "activity_fingerprint": fingerprint,
    }


def message(current: dict, notices: dict[str, str]) -> str:
    """Adapt the established league notification without changing delivery behavior."""
    rendered = _base_message(current, notices)
    rendered = rendered.replace(
        "**Azuki personal-project C engine — BPTT league monitor**",
        "**Azuki control +30M continuation monitor**",
    ).replace(
        "Shared checkpoint learner step:",
        "Continuation parent learner step:",
    ).replace(
        "Completion requires review; old partial arms remain excluded.",
        "Completion requires exact 8,100-game trace coverage and review; no promotion is automatic.",
    )
    panel_counts = current.get("evaluation_panels", {})
    panel_line = "Evaluation panels: " + "; ".join(
        f"{panel} {int(panel_counts.get(panel, 0)):,}/{EXPECTED_GAMES_PER_PANEL:,}"
        for panel in PANELS
    ) + "."
    lines = rendered.splitlines()
    lines.insert(3 if len(lines) >= 3 else len(lines), panel_line)
    return "\n".join(lines)


def main() -> None:
    observer.__doc__ = __doc__
    observer.process_identity = process_identity
    observer.snapshot = snapshot
    observer.message = message
    observer.main()


if __name__ == "__main__":
    main()
