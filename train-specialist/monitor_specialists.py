#!/usr/bin/env python3
"""Read-only Discord observer for specialist training (run_queue.sh / run_continuation.sh).

Posts on stage transitions, failures, stalls, process exit and completion, plus a
periodic progress summary. It never controls or restarts training.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, "/home/skylark/git/azuki-tcg/train-ablation-1781126582")
from monitor_local_production import _atomic_json, _discord_post  # noqa: E402
from monitor_strategy_discovery import process_identity  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
# Defaults describe the original four-element queue; main() overrides them for continuations.
RUN_NAMES = ["water", "earth", "lightning", "fire"]
QUEUE_LOG = RUNS / "queue.log"
EVAL_RESULT = RUNS / "ensemble_eval_s30m.json"
TITLE = "**Azuki element-specialist fine-tunes (from u8223)**"


def run_info(root: Path) -> tuple[str, str, int, int]:
  """(element, stage prefix, target additional learner steps, base global step) of a run root."""
  if not (root / "experiment.json").is_file():
    return root.name, "s30m", 30_000_000, 0
  plan = json.loads((root / "experiment.json").read_text())
  continuation = plan.get("continuation", {})
  return (plan["element"], continuation.get("stage", "s30m"),
          int(continuation.get("additional_learner_steps", 30_000_000)), int(plan["parent"]["global_step"]))


def _last_metrics_row(path: Path) -> dict:
  if not path.is_file():
    return {}
  with path.open("rb") as handle:
    # Metric rows are ~1.2 MB each; read enough tail to hold two complete rows.
    handle.seek(max(0, path.stat().st_size - 4 * 1024 * 1024))
    lines = [line for line in handle.read().split(b"\n") if line.strip()]
  for line in reversed(lines):
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if "epoch" in row:
      return row
  return {}


def _latest_stage(root: Path, prefix: str) -> Path | None:
  stages = [path for path in (root / "stages").glob(f"{prefix}*") if (path / "launch.json").is_file()]
  return max(stages, key=lambda path: (path / "launch.json").stat().st_mtime_ns, default=None)


def snapshot() -> dict:
  activity = []
  elements = []
  for name in RUN_NAMES:
    root = RUNS / name
    element, prefix, target, base_step = run_info(root)
    row = {"id": name, "state": "pending", "steps": 0, "target": target, "sps": None, "element_fraction": None,
           "stage": None}
    stage = _latest_stage(root, prefix)
    if (root / "final_result.json").is_file():
      result = json.loads((root / "final_result.json").read_text())
      row.update(state="complete", steps=int(result["learner_global_step"]) - base_step,
                 sps=result.get("median_logged_sps"),
                 element_fraction=result.get("specialist", {}).get("learner_battle_element_fraction"))
    elif stage is not None:
      row["stage"] = stage.name
      if (stage / "failure.json").is_file():
        row["state"] = "failed"
      elif (stage / "result.json").is_file():
        row["state"] = "finalizing"
      else:
        row["state"] = "running"
      log = root / "logs" / f"specialist_{element}_{stage.name}.jsonl"
      metrics = _last_metrics_row(log)
      if metrics:
        row.update(steps=int(metrics["_step"]) - base_step, sps=metrics.get("SPS"),
                   element_fraction=metrics.get("environment/specialist/learner_battle_element_fraction"))
      for path in (log, stage / "console.log"):
        if path.is_file():
          info = path.stat()
          activity.append((str(path), info.st_size, info.st_mtime_ns))
    elements.append(row)
  queue_text = QUEUE_LOG.read_text(errors="replace") if QUEUE_LOG.is_file() else ""
  if QUEUE_LOG.is_file():
    info = QUEUE_LOG.stat()
    activity.append((str(QUEUE_LOG), info.st_size, info.st_mtime_ns))
  evaluating = "evaluating" in queue_text
  done = queue_text.rstrip().endswith("done") and (EVAL_RESULT is None or EVAL_RESULT.is_file())
  failed = any(row["state"] == "failed" for row in elements)
  state = "complete" if done else "failed" if failed else "running"
  summary = None
  if EVAL_RESULT is not None and EVAL_RESULT.is_file():
    summary = json.loads(EVAL_RESULT.read_text()).get("summary", {})
  return {"state": state, "evaluating": evaluating and not done, "elements": elements, "eval_summary": summary,
          "activity_fingerprint": hashlib.sha256(json.dumps(activity).encode()).hexdigest()}


def events(current: dict, state: dict, now: float, stall_seconds: float, alive: bool) -> dict[str, str]:
  notices = {}
  if "attached" not in state.get("sent_events", []):
    notices["attached"] = f"Watcher attached for {', '.join(RUN_NAMES)}; monitoring progress, stalls, exit and completion."
  for row in current["elements"]:
    if row["state"] != "pending":
      notices[f"{row['id']}:{row['stage']}:{row['state']}"] = {
        "running": f"{row['id']}: specialist training running (stage {row['stage']}).",
        "finalizing": f"{row['id']}: stage {row['stage']} finished; recording final result.",
        "complete": f"{row['id']}: specialist training finished (+{row['steps']:,} learner steps).",
        "failed": f"{row['id']}: FAILED — see train-specialist/runs/{row['id']}/stages/{row['stage']}/failure.json.",
      }[row["state"]]
  if current["evaluating"]:
    notices["evaluating"] = "Training finished; checkpoint evaluation is running."
  if current["state"] == "complete":
    overall = (current["eval_summary"] or {}).get("overall")
    score = (f" — ensemble score {overall.get('score', float('nan')):.3f} over {overall.get('games', 0):,} games"
             if overall else "")
    notices["complete"] = f"FINISHED{score}. Review results under train-specialist/runs/."
  elif current["state"] == "failed":
    notices["queue:failed"] = "STOPPED ON A FAILURE — review required."
  elif not alive:
    notices["process_exited"] = f"PROCESS EXITED without completing — review {QUEUE_LOG}."
  elif current["activity_fingerprint"] != state.get("activity_fingerprint"):
    if state.get("stall_alerted"):
      notices[f"recovered:{state['last_activity_at']}"] = "Training activity resumed after the stall alert."
    state.update(activity_fingerprint=current["activity_fingerprint"], last_activity_at=now, stall_alerted=False)
  elif now - state.get("last_activity_at", now) >= stall_seconds:
    notices[f"stall:{state['last_activity_at']}"] = (
      f"POSSIBLE STALL: no log activity for {int((now - state['last_activity_at']) / 60)} minutes. Training was NOT stopped.")
  state.setdefault("last_activity_at", now)
  pending = {**state.get("pending_events", {}), **notices}
  state["pending_events"] = {key: text for key, text in pending.items() if key not in state.get("sent_events", [])}
  return state["pending_events"]


def message(current: dict, notices: dict[str, str]) -> str:
  lines = [TITLE, *notices.values()]
  for row in current["elements"]:
    progress = f"{row['steps']:,}/{row['target']:,} learner steps"
    extra = ""
    if row["sps"] is not None:
      extra += f"; SPS {row['sps']:,.0f}"
    if row["element_fraction"] is not None:
      extra += f"; learner element fraction {row['element_fraction']:.3f}"
    lines.append(f"{row['id']}: {row['state']}; {progress}{extra}")
  lines.append("Read-only watcher; it does not control training.")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--queue-pid", type=int, required=True)
  parser.add_argument("--discord-webhook-file", type=Path, default=Path.home() / ".config/azuki-tcg/discord-webhook-url")
  parser.add_argument("--state-file", type=Path, default=RUNS / "discord_monitor_state.json")
  parser.add_argument("--poll-seconds", type=float, default=60)
  parser.add_argument("--interval-seconds", type=float, default=3600)
  parser.add_argument("--stall-seconds", type=float, default=1800)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--runs", default="", help="comma list of run dirs under train-specialist/runs (default: queue)")
  parser.add_argument("--queue-log", type=Path, help="driver log to watch (default runs/queue.log)")
  parser.add_argument("--eval-result", default="", help="ensemble eval JSON required for completion ('none' = not needed)")
  parser.add_argument("--title", default="")
  args = parser.parse_args()
  global RUN_NAMES, QUEUE_LOG, EVAL_RESULT, TITLE
  if args.runs:
    RUN_NAMES = [name.strip() for name in args.runs.split(",") if name.strip()]
  if args.queue_log is not None:
    QUEUE_LOG = args.queue_log.resolve()
  if args.eval_result == "none":
    EVAL_RESULT = None
  elif args.eval_result:
    EVAL_RESULT = Path(args.eval_result).resolve()
  if args.title:
    TITLE = f"**{args.title}**"
  state = json.loads(args.state_file.read_text()) if args.state_file.exists() else {}
  identity = state.get("process_identity") or process_identity(args.queue_pid)
  if identity is None:
    raise ValueError("Queue PID is not live")
  state["process_identity"] = identity
  print("[specialist-monitor] watching queue; read-only", flush=True)
  while True:
    now = time.time()
    try:
      current = snapshot()
      alive = process_identity(args.queue_pid) == identity
      pending = events(current, state, now, args.stall_seconds, alive)
      if args.dry_run:
        print(message(current, pending), flush=True)
        return
      if pending or now - state.get("last_post_at", 0) >= args.interval_seconds:
        with contextlib.redirect_stdout(io.StringIO()):
          sent = _discord_post(args, message(current, pending))
        if sent:
          state.setdefault("sent_events", []).extend(pending)
          state["pending_events"] = {}
          state["last_post_at"] = now
          if any(key.startswith("stall:") for key in pending):
            state["stall_alerted"] = True
          print(f"[specialist-monitor] delivered events={','.join(pending) or 'periodic_progress'}", flush=True)
        else:
          print("[specialist-monitor] delivery failed; retrying next poll", flush=True)
      state.update(last_checked_at=now, snapshot=current)
      _atomic_json(args.state_file, state)
      terminal = current["state"] != "running" or not alive
      if terminal and not state.get("pending_events"):
        print("[specialist-monitor] terminal notification delivered", flush=True)
        return
    except (OSError, ValueError, KeyError) as exc:
      print(f"[specialist-monitor] observation error={type(exc).__name__}; retrying", flush=True)
    time.sleep(args.poll_seconds)


if __name__ == "__main__":
  main()
