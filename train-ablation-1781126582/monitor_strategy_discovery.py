#!/usr/bin/env python3
"""Read-only campaign observer; notify Discord without changing or advancing training."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from monitor_local_production import _atomic_json, _discord_post

ROOT = Path(__file__).resolve().parents[1]
COMPLETE = "completed_observations_require_review"


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def count_records(path: Path, counters: dict) -> int:
    info = path.stat()
    key = str(path)
    entry = counters.get(key, {})
    if entry.get("inode") != info.st_ino or entry.get("offset", 0) > info.st_size:
        entry = {"inode": info.st_ino, "offset": 0, "count": 0}
    with path.open("rb") as handle:
        handle.seek(entry["offset"])
        while chunk := handle.read(1024 * 1024):
            entry["count"] += chunk.count(b"\n")
        entry["offset"] = handle.tell()
    counters[key] = entry
    return entry["count"]


def _latest_league_metrics(path: Path, cursors: dict) -> dict:
    info = path.stat()
    key = str(path)
    entry = cursors.get(key, {})
    if entry.get("inode") != info.st_ino or entry.get("offset", 0) > info.st_size:
        entry = {"inode": info.st_ino, "offset": 0, "latest": {}}
    with path.open("rb") as handle:
        handle.seek(entry["offset"])
        while line := handle.readline():
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if "epoch" in row:
                entry["latest"] = {"global_step": int(row["_step"]), "epoch": int(row["epoch"])}
            entry["offset"] = handle.tell()
    cursors[key] = entry
    return entry["latest"]


def _league_snapshot(campaign: Path, state: dict) -> dict:
    plan_path = campaign / "experiment.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema_id") != "azuki.bptt_league_experiment" or plan.get("schema_version") != 1:
        raise ValueError("Unsupported league experiment schema")
    digest = hashlib.sha256(plan_bytes).hexdigest()
    tooling = json.loads((campaign / "orchestration_manifest.json").read_text())["sha256"]
    if tooling.get(str(plan_path.resolve())) != digest:
        raise ValueError("League experiment differs from its registered orchestration")
    if state.get("registration_sha256", digest) != digest:
        raise ValueError("Watcher belongs to another registration")
    state["registration_sha256"] = digest
    status_path = campaign / "campaign_status.json"
    status = json.loads(status_path.read_text())
    parent = json.loads((campaign / "common_parent.json").read_text())
    common_step = int(parent["learner_global_step"])
    target = int(plan["comparison"]["additional_learner_steps_per_arm"])
    activity = []

    def record_activity(path: Path) -> None:
        info = path.stat()
        activity.append((str(path), info.st_size, info.st_mtime_ns))

    record_activity(status_path)
    counters = state.setdefault("trace_counters", {})
    evaluation_counts = {}
    for panel in ("canary", "midpoint", "final"):
        for trace in sorted((campaign / "evaluation" / panel).glob("*/*/*/shard*.jsonl")):
            role = trace.parent.name
            evaluation_counts[role] = evaluation_counts.get(role, 0) + count_records(trace, counters)
            record_activity(trace)
    expected_games = sum(plan["evaluation"][key] for key in (
        "initial_common_canary_checkpoint", "midpoint_two_arm_games", "final_two_arm_games"
    ))
    arms = []
    cursors = state.setdefault("metric_cursors", {})
    for arm in plan["comparison"]["arms"]:
        folder = campaign / "runs" / arm
        row = {"id": arm, "state": "pending", "rows": 0, "target_rows": target,
               "epoch": parent["update"], "eval_games": evaluation_counts.get(arm, 0)}
        latest_step = common_step
        for stage in ("midpoint", "final"):
            result_path = folder / "stages" / stage / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                latest_step = max(latest_step, int(result["learner_global_step"]))
                row.update(epoch=int(result["update"]),
                           state="evaluating" if stage == "final" else "midpoint_complete")
                record_activity(result_path)
            metrics_path = folder / "logs" / f"{arm}_{stage}.jsonl"
            if metrics_path.exists():
                latest = _latest_league_metrics(metrics_path, cursors)
                latest_step = max(latest_step, latest.get("global_step", common_step))
                row["epoch"] = max(row["epoch"], latest.get("epoch", row["epoch"]))
                record_activity(metrics_path)
            console_path = folder / "stages" / stage / "console.log"
            if console_path.exists():
                record_activity(console_path)
        row["rows"] = latest_step - common_step
        if status["state"].startswith(f"training_{arm}_"):
            row["state"] = "running"
        if "evaluation/final" in status["completed"]:
            row["state"] = "evaluated"
        arms.append(row)
    phase = status["state"]
    current_state = "failed" if phase == "failed" else "running"
    if phase == "complete":
        result = json.loads((campaign / "result.json").read_text())
        if result["status"] != "complete" or result["evaluation_games"] != expected_games:
            raise ValueError("League completion lacks the full registered result")
        if sum(evaluation_counts.values()) != expected_games:
            raise ValueError("League completion lacks the full registered trace coverage")
        current_state = COMPLETE
    return {"state": current_state, "phase": phase, "family": "bptt_league_experiment",
            "arms": arms, "diagnostic_complete": False, "expected_eval_games": expected_games,
            "eval_games": sum(evaluation_counts.values()), "common_global_step": common_step,
            "activity_fingerprint": hashlib.sha256(json.dumps(activity).encode()).hexdigest()}


def snapshot(campaign: Path, diagnostic: Path | None, state: dict) -> dict:
    if (campaign / "experiment.json").is_file():
        return _league_snapshot(campaign, state)
    registration_path = campaign / "registration.json"
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    recipe = registration.get("family") in ("strategy_recipe_v1", "strategy_retention_v1", "random50_continuation_v1", "leader_normal_penalty_v1")
    if not recipe and diagnostic is None:
        raise ValueError("Discovery monitoring requires --diagnostic")
    status = json.loads((campaign / "campaign_status.json").read_text())
    digest = hashlib.sha256(registration_bytes).hexdigest()
    if status["registration_sha256"] != digest:
        raise ValueError("Campaign status and registration hashes differ")
    if state.get("registration_sha256", digest) != digest:
        raise ValueError("Watcher belongs to another registration")
    state["registration_sha256"] = digest
    counters = state.setdefault("trace_counters", {})
    activity = []

    def record_activity(path: Path) -> None:
        info = path.stat()
        activity.append((str(path), info.st_size, info.st_mtime_ns))

    if not recipe:
        record_activity(campaign / "campaign_status.json")
    game_counts = {}
    for mode in ("sample", "argmax"):
        paths = [] if diagnostic is None else sorted(diagnostic.glob(f"trace_{mode}_*.jsonl"))
        game_counts[mode] = sum(count_records(path, counters) for path in paths)
        for path in paths:
            record_activity(path)
    diagnostic_result = None if diagnostic is None else diagnostic / "decision_packet.json"
    if diagnostic_result is not None and diagnostic_result.exists():
        record_activity(diagnostic_result)
    arms = []
    for arm in registration["arms"]:
        folder = ROOT / arm["result_root"]
        path = folder / "run_status.json"
        row = {"id": arm["id"], "state": "pending", "rows": 0,
               "target_rows": arm["total_timesteps"], "eval_games": 0,
               "target_updates": arm["total_updates"],
               "training_required": arm.get("initialization") != "saved_random50_checkpoints"}
        if path.exists():
            recorded = json.loads(path.read_text())
            if not recipe:
                record_activity(path)
            progress = recorded.get("progress", {})
            row.update(state=recorded["state"], rows=int(progress.get("agent_steps", 0)),
                       epoch=int(progress.get("epoch", 0)))
            if recorded["state"] == "completed":
                row["epoch"] = arm["total_updates"]
                row["state"] = "evaluating"
            for name in ("training_console.log", "logs/production.jsonl"):
                log = folder / name
                if log.exists():
                    record_activity(log)
        panels = ("paired", "heldout") if registration.get("family") in ("random50_continuation_v1", "leader_normal_penalty_v1") else ("paired",)
        evaluations = [campaign / panel / arm["name"] / "eval" for panel in panels] if recipe else [folder / "eval"]
        trace_pattern = "*/*/trace_*.jsonl" if recipe else "*/trace_*.jsonl"
        for evaluation in evaluations:
            for trace in sorted(evaluation.glob(trace_pattern)):
                row["eval_games"] += count_records(trace, counters)
                record_activity(trace)
            if not recipe:
                for artifact in sorted(evaluation.glob("*/*")):
                    if artifact.is_file() and artifact.suffix in (".json", ".log"):
                        record_activity(artifact)
        if ((recipe and any(item["id"] == arm["id"] and item["state"] == "evaluated" for item in status["arms"]))
                or (not recipe and (folder / "evaluation_index.json").exists())):
            row["state"] = "evaluated"
        elif (not row["training_required"] and status.get("active", {}).get("arm") == arm["id"]):
            row["state"] = "evaluating"
        arms.append(row)
    fingerprint = hashlib.sha256(json.dumps(activity, sort_keys=True).encode()).hexdigest()
    return {"state": status["state"], "phase": status.get("active", {}).get("phase", status.get("phase", "unknown")),
            "family": registration.get("family"), "recipe": recipe,
            "diagnostic_games": game_counts, "diagnostic_expected_games": 0 if recipe else 1536,
            "diagnostic_complete": diagnostic_result is not None and diagnostic_result.exists(),
            "expected_eval_games": registration.get("expected_evaluation_games"), "arms": arms,
            "activity_fingerprint": fingerprint}


def events(current: dict, state: dict, now: float, stall_seconds: float, alive: bool) -> dict[str, str]:
    notices = {}
    if "attached" not in state.get("sent_events", []):
        notices["attached"] = "Independent Discord watcher attached; monitoring progress, stalls, process exit and completion."
    if current.get("family") == "bptt_league_experiment" and current["state"] == "running":
        notices[f"phase:{current['phase']}"] = f"Campaign stage: {current['phase']}."
    for arm in current["arms"]:
        if arm["state"] != "pending":
            notices[f"arm:{arm['id']}:{arm['state']}"] = {
                "running": f"{arm['id']}: training started.",
                "evaluating": f"{arm['id']}: training finished; checkpoint evaluation is running." if arm.get("training_required", True) else f"{arm['id']}: saved-control evaluation is running; no training.",
                "evaluated": f"{arm['id']}: training and evaluation finished." if arm.get("training_required", True) else f"{arm['id']}: saved-control evaluation finished; no training.",
                "failed": f"{arm['id']}: FAILED; review the arm status and console log.",
                "interrupted": f"{arm['id']}: INTERRUPTED; review required.",
            }.get(arm["state"], f"{arm['id']}: {arm['state']}.")
    if current["diagnostic_complete"]:
        notices["diagnostic_complete"] = "Same-policy deck diagnostic finished; its result is available for review."
    if current["state"] == COMPLETE:
        notices["campaign_complete"] = "CAMPAIGN FINISHED — READY FOR REVIEW. All registered observations are complete."
    elif current["state"] != "running":
        notices[f"campaign:{current['state']}"] = f"CAMPAIGN {current['state'].upper()} — review required; do not treat this as success."
    elif not alive:
        notices["process_exited"] = "CAMPAIGN PROCESS EXITED without a terminal status — review required."
    else:
        changed = current["activity_fingerprint"] != state.get("activity_fingerprint")
        if changed:
            if state.get("stall_alerted"):
                notices[f"recovered:{state['last_activity_at']}"] = "Artifact progress resumed after the inactivity alert."
            state.update(activity_fingerprint=current["activity_fingerprint"], last_activity_at=now,
                         stall_alerted=False)
        elif now - state["last_activity_at"] >= stall_seconds:
            notices[f"stall:{state['last_activity_at']}"] = (
                f"POSSIBLE STALL: no observed artifact activity for {int((now - state['last_activity_at']) / 60)} minutes. "
                "The watcher has NOT stopped training."
            )
    pending = {**state.get("pending_events", {}), **notices}
    state["pending_events"] = {
        key: text for key, text in pending.items() if key not in state.get("sent_events", [])
    }
    return state["pending_events"]


def message(current: dict, notices: dict[str, str]) -> str:
    if current.get("family") == "bptt_league_experiment":
        lines = ["**Azuki personal-project C engine — BPTT league monitor**", *notices.values(),
                 f"Current stage: `{current['phase']}`.",
                 f"Evaluation traces: {current['eval_games']:,}/{current['expected_eval_games']:,} games.",
                 f"Shared checkpoint learner step: {current['common_global_step']:,}."]
        for arm in current["arms"]:
            lines.append(f"{arm['id']}: {arm['state']}; +{arm['rows']:,}/{arm['target_rows']:,} "
                         f"learner steps; update {arm['epoch']:,}; {arm['eval_games']:,} evaluation games.")
        lines.append("Read-only watcher; no training control or automatic model session. "
                     "Completion requires review; old partial arms remain excluded.")
        return "\n".join(lines)
    lines = ["**Azuki strategy campaign monitor**", *notices.values()]
    if current.get("recipe"):
        lines.append(f"Paired evaluation traces: {sum(arm['eval_games'] for arm in current['arms']):,}/{current['expected_eval_games']:,} games.")
    else:
        games = current["diagnostic_games"]
        lines.append(f"Deck diagnostic: {sum(games.values()):,}/{current['diagnostic_expected_games']:,} games "
                     f"(sample {games['sample']:,}; deterministic {games['argmax']:,}).")
    for requires_training, label in ((True, "Training/evaluation arms"), (False, "Saved-control evaluation arms")):
        group = [arm for arm in current["arms"] if arm.get("training_required", True) == requires_training]
        if group:
            done = sum(arm["state"] == "evaluated" for arm in group)
            lines.append(f"{label} finished: {done}/{len(group)}.")
    for arm in current["arms"]:
        if arm["state"] in ("running", "evaluating"):
            if not arm.get("training_required", True):
                lines.append(f"{arm['id']}: saved control; {arm['eval_games']:,} evaluation games; no new training.")
                continue
            lines.append(f"{arm['id']}: {arm['state']}; update {arm.get('epoch', 0):,}/{arm['target_updates']:,}; "
                         f"{arm['rows']:,} learner steps (allocation {arm['target_rows']:,} configured rows); "
                         f"{arm['eval_games']:,} evaluation trace games.")
    lines.append("No model session is spawned automatically. Resume/open an assistant session to review results and decide next steps. The 1B run remains prohibited.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, help="Required for discovery campaigns; omitted for paired recipe campaigns")
    parser.add_argument("--campaign-pid", type=int)
    parser.add_argument("--discord-webhook-file", type=Path,
                        default=Path.home() / ".config/azuki-tcg/discord-webhook-url")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--interval-seconds", type=float, default=1800)
    parser.add_argument("--stall-seconds", type=float, default=1800)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.poll_seconds, args.interval_seconds, args.stall_seconds) <= 0:
        raise ValueError("Monitoring intervals must be positive")
    state_path = args.state_file or args.campaign / "discord_monitor_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    identity = None
    if args.campaign_pid is not None:
        identity = state.get("process_identity") or process_identity(args.campaign_pid)
        if identity is None:
            raise ValueError("Campaign PID is no longer live; attach to the current supervised process")
        state["process_identity"] = identity
    if not args.dry_run:
        path = args.discord_webhook_file.expanduser()
        if path.stat().st_mode & 0o077:
            raise ValueError("Webhook file permissions must be owner-only")
        parsed = urlsplit(path.read_text().strip())
        if parsed.scheme != "https" or parsed.hostname not in ("discord.com", "discordapp.com", "canary.discord.com", "ptb.discord.com") or not parsed.path.startswith("/api/webhooks/"):
            raise ValueError("Webhook file must contain a Discord HTTPS webhook URL")
    print("[discord-monitor] watching campaign; training control and automatic model sessions disabled", flush=True)
    while True:
        now = time.time()
        try:
            current = snapshot(args.campaign, args.diagnostic, state)
            state["observation_errors"] = 0
            alive = args.campaign_pid is None or process_identity(args.campaign_pid) == identity
            pending = events(current, state, now, args.stall_seconds, alive)
            terminal = current["state"] != "running" or not alive
            due = bool(pending) or now - state.get("last_post_at", 0) >= args.interval_seconds
            if args.dry_run:
                print(message(current, pending), flush=True)
                return
            if due:
                # Reuse the existing notifier; never print exception text that could contain a token.
                with contextlib.redirect_stdout(io.StringIO()):
                    sent = _discord_post(args, message(current, pending))
                if sent:
                    state.setdefault("sent_events", []).extend(pending)
                    state["pending_events"] = {}
                    state["last_post_at"] = now
                    if any(key.startswith("stall:") for key in pending):
                        state["stall_alerted"] = True
                    print(f"[discord-monitor] delivered events={','.join(pending) or 'periodic_progress'}", flush=True)
                else:
                    print("[discord-monitor] delivery failed; retrying next poll (credential redacted)", flush=True)
            state.update(last_checked_at=now, snapshot=current)
            _atomic_json(state_path, state)
            if terminal and not pending.keys() - set(state.get("sent_events", [])):
                print("[discord-monitor] terminal notification delivered; review requires a model session opened by the user", flush=True)
                return
            if args.once:
                if due and not sent:
                    raise RuntimeError("Discord delivery failed")
                return
        except (OSError, ValueError, KeyError) as exc:
            # Read races and transient errors must not silently remove monitoring.
            print(f"[discord-monitor] observation error={type(exc).__name__}; retrying next poll", flush=True)
            state["observation_errors"] = state.get("observation_errors", 0) + 1
            if not args.dry_run and state["observation_errors"] >= 3 and now - state.get("last_error_post_at", 0) >= args.interval_seconds:
                with contextlib.redirect_stdout(io.StringIO()):
                    notified = _discord_post(
                        args, "**Azuki monitoring error**\nThe Discord watcher cannot read campaign progress reliably. "
                        "Review the local monitor log. Training has NOT been stopped; no model session was spawned."
                    )
                if notified:
                    state["last_error_post_at"] = now
                    print("[discord-monitor] delivered monitoring_error", flush=True)
            if not args.dry_run:
                _atomic_json(state_path, state)
            if args.once or args.dry_run:
                raise
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
