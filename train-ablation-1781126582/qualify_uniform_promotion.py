#!/usr/bin/env python3
"""Run and replay the uniform-context promotion evaluator in shadow mode."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

import azk_puffer.vector as azk_vector

from evaluate_checkpoint import _apply_checkpoint_resume_policy_config
from league_manager import LeagueManager, parse_league_manager_config
from league_promotion_store import checkpoint_sha256
from train import _load_model_weights
from training_utils import build_policy, build_vecenv, load_training_config


SEMANTIC_GAME_FIELDS = (
  "game_id",
  "block_id",
  "phase",
  "opponent_id",
  "seed",
  "candidate_seat",
  "candidate_gate",
  "opponent_gate",
  "candidate_leader",
  "opponent_leader",
  "winner_seat",
  "steps",
  "end_reason",
  "schedule_version",
  "reference_seat",
  "reference_deck_index",
  "world_seed",
  "starting_player",
  "evaluator_version",
)


def _json_write(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _link_checkpoint(source: Path, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  try:
    os.link(source, destination)
  except OSError:
    shutil.copy2(source, destination)
  source_meta = source.with_suffix(source.suffix + ".meta.json")
  if source_meta.exists():
    shutil.copy2(source_meta, destination.with_suffix(destination.suffix + ".meta.json"))


def _control_snapshot(manager: LeagueManager) -> dict:
  return {
    "league": {
      "champion_policy_id": manager.state.champion_policy_id,
      "learner_policy_id": manager.state.learner_policy_id,
      "current_candidate_policy_id": manager.state.current_candidate_policy_id,
      "next_policy_index": manager.state.next_policy_index,
      "active_pool": [
        {
          "policy_id": entry.policy_id,
          "checkpoint_path": entry.checkpoint_path,
          "bucket": entry.bucket,
          "active": entry.active,
        }
        for entry in sorted(manager.state.policies.values(), key=lambda item: item.policy_id)
        if entry.active
      ],
    },
    "promotion": {
      "production_anchor_policy_id": manager.archive_state.production_anchor_policy_id,
      "active_panel_version": manager.archive_state.active_panel_version,
      "quality_archive": [asdict(entry) for entry in manager.archive_state.quality_archive],
      "panels": [asdict(panel) for panel in manager.archive_state.panels],
    },
  }


def _semantic_games(payload: dict) -> list[dict]:
  games = payload.get("games")
  if not isinstance(games, list):
    raise ValueError("Promotion artifact does not contain a game list")
  return sorted(
    [{key: game.get(key) for key in SEMANTIC_GAME_FIELDS} for game in games],
    key=lambda game: str(game["game_id"]),
  )


def _context_checks(payload: dict) -> dict:
  games = payload["games"]
  panel = [game for game in games if game["phase"] != "reference"]
  screen = [game for game in panel if game["phase"] == "screen"]
  confirmation = [game for game in panel if game["phase"] == "confirmation"]
  references = [game for game in games if game["phase"] == "reference"]
  screen_contexts = {
    (int(game["candidate_gate"]), int(game["candidate_leader"])) for game in screen
  }
  screen_context_seats = {
    (
      int(game["candidate_gate"]),
      int(game["candidate_leader"]),
      int(game["candidate_seat"]),
    )
    for game in screen
  }
  reference_context_seats = {
    (
      int(game["candidate_gate"]),
      int(game["candidate_leader"]),
      int(game["candidate_seat"]),
    )
    for game in references
  }
  reference_manifest = payload.get("metadata", {}).get("reference_manifest", {})
  external_window = payload.get("metadata", {}).get("external_strength_window")
  checks = {
    "panel_games": len(panel),
    "screen_games": len(screen),
    "confirmation_games": len(confirmation),
    "reference_games": len(references),
    "screen_contexts": len(screen_contexts),
    "screen_context_seats": len(screen_context_seats),
    "reference_context_seats": len(reference_context_seats),
    "all_panel_leaders_assigned": all(
      int(game.get("candidate_leader", -1)) >= 0
      and int(game.get("opponent_leader", -1)) >= 0
      for game in panel
    ),
    "all_reference_candidate_leaders_assigned": all(
      int(game.get("candidate_leader", -1)) >= 0 for game in references
    ),
    "all_reference_leaders_assigned": all(
      int(game.get("candidate_leader", -1)) >= 0
      and int(game.get("opponent_leader", -1)) >= 0
      for game in references
    ),
    "all_games_completed": all(game.get("end_reason") == "gameover" for game in games),
    "uniform_reference_manifest": (
      reference_manifest.get("assignment_contract")
      == "uniform_gate_same_element_leader"
    ),
    "external_strength_window_recorded": (
      isinstance(external_window, dict)
      and int(external_window.get("count", 0)) >= 1
      and external_window.get("version") == "external-strength-window-v1"
    ),
  }
  checks["passed"] = bool(
    checks["panel_games"] == 192
    and checks["screen_games"] == 128
    and checks["confirmation_games"] == 64
    and checks["reference_games"] == 288
    and checks["screen_contexts"] == 16
    and checks["screen_context_seats"] == 32
    and checks["reference_context_seats"] == 32
    and checks["all_panel_leaders_assigned"]
    and checks["all_reference_candidate_leaders_assigned"]
    and checks["all_reference_leaders_assigned"]
    and checks["all_games_completed"]
    and checks["uniform_reference_manifest"]
    and checks["external_strength_window_recorded"]
  )
  return checks


def _build_candidate_policy(checkpoint: Path, trainer_args: dict, vecenv, device: str):
  policy_args = copy.deepcopy(trainer_args)
  _apply_checkpoint_resume_policy_config(policy_args, checkpoint)
  policy_args["train"]["device"] = device
  policy = build_policy(vecenv, policy_args)
  _load_model_weights(policy, checkpoint, device=device, strict=False)
  policy.eval()
  for parameter in policy.parameters():
    parameter.requires_grad_(False)
  return policy


def _run_replay(args: argparse.Namespace, replay_index: int) -> dict:
  replay_root = args.output_dir / f"replay{replay_index}"
  if replay_root.exists():
    raise FileExistsError(f"Replay output already exists: {replay_root}")
  replay_root.mkdir(parents=True)
  league_state = replay_root / "league_state.json"
  promotion_state = replay_root / "league_state_promotion.json"
  shutil.copy2(args.league_state, league_state)
  shutil.copy2(args.promotion_state, promotion_state)
  candidate = replay_root / "candidate.pt"
  _link_checkpoint(args.checkpoint, candidate)

  trainer_args = load_training_config(args.config, [])
  trainer_args["train"]["device"] = args.device
  trainer_args["env"]["draft_uniform_assignment"] = True
  trainer_args["env"]["native_envs_per_instance"] = 1
  league_cfg = trainer_args["league"]
  league_cfg["enable"] = True
  league_cfg["state_path"] = str(league_state)
  league_cfg["promotion_state_path"] = str(promotion_state)
  league_cfg["promotion_records_dir"] = str(replay_root / "promotion_records")
  league_cfg["promotion_shadow_mode"] = True
  league_cfg["promotion_archive_affects_training_pool"] = False
  league_cfg["quick_eval_interval"] = 1
  league_cfg["full_eval_interval"] = 1
  league_cfg["eval_interval"] = 1
  league_cfg["min_candidate_epoch_gap"] = 1
  league_cfg["promotion_panel_refresh_epochs"] = 1_000_000_000
  # Qualification must exercise all phases regardless of candidate strength.
  league_cfg["promotion_anchored_screen_pooled_floor"] = 0.0
  league_cfg["promotion_anchored_screen_opponent_floor"] = 0.0
  league_cfg["promotion_anchored_screen_seat_floor"] = 0.0
  league_cfg["promotion_anchored_screen_relative_pooled_min"] = -1.0
  league_cfg["promotion_anchored_screen_relative_opponent_floor"] = -1.0
  league_cfg["promotion_anchored_screen_relative_seat_floor"] = -1.0
  manager = LeagueManager(parse_league_manager_config(trainer_args))
  qualification_epoch = max(
    (int(entry.created_epoch) for entry in manager.state.policies.values()),
    default=0,
  ) + 1
  candidate_entry = manager.maybe_add_checkpoint(candidate, epoch=qualification_epoch)
  if candidate_entry is None:
    raise RuntimeError("Could not register the replay candidate")
  before = _control_snapshot(manager)

  vecenv = build_vecenv(
    trainer_args,
    backend=azk_vector.Serial,
    num_envs=1,
    seed=int(league_cfg["promotion_schedule_seed"]),
  )
  try:
    policy = _build_candidate_policy(candidate, trainer_args, vecenv, args.device)
    metrics = manager.maybe_evaluate_and_promote(
      epoch=qualification_epoch,
      trainer_args=trainer_args,
      vecenv=vecenv,
      build_policy_fn=build_policy,
      load_weights_fn=_load_model_weights,
      learner_policy=policy,
    )
  finally:
    vecenv.close()
  if not metrics:
    raise RuntimeError("Uniform promotion replay produced no evaluation metrics")
  after = _control_snapshot(manager)
  record_path = Path(str(metrics["league/eval_raw_record_path"]))
  artifact = json.loads(record_path.read_text(encoding="utf-8"))
  context_checks = _context_checks(artifact)
  timed_stages = (
    "league/eval_panel_anchor_seconds",
    "league/eval_screen_seconds",
    "league/eval_confirmation_seconds",
    "league/eval_reference_seconds",
    "league/eval_policy_load_seconds",
    "league/eval_serialization_seconds",
  )
  measured_gate_seconds = sum(float(metrics[key]) for key in timed_stages)
  timing_checks = {
    "screen_seconds": float(metrics["league/eval_screen_seconds"]),
    "measured_full_gate_seconds": measured_gate_seconds,
    "screen_under_180_seconds": float(metrics["league/eval_screen_seconds"]) <= 180.0,
    "full_gate_under_480_seconds": measured_gate_seconds <= 480.0,
  }
  timing_checks["passed"] = bool(
    timing_checks["screen_under_180_seconds"]
    and timing_checks["full_gate_under_480_seconds"]
  )
  result = {
    "replay": replay_index,
    "qualification_epoch": qualification_epoch,
    "candidate_policy_id": candidate_entry.policy_id,
    "candidate_checkpoint_sha256": checkpoint_sha256(candidate),
    "control_state_unchanged": before == after,
    "control_before": before,
    "control_after": after,
    "context_checks": context_checks,
    "timing_checks": timing_checks,
    "metrics": metrics,
    "record_path": str(record_path.resolve()),
    "semantic_games": _semantic_games(artifact),
  }
  _json_write(replay_root / "qualification.json", result)
  return result


def _markdown(payload: dict) -> str:
  first = payload["replays"][0]
  checks = first["context_checks"]
  metrics = first["metrics"]
  lines = [
    "# Uniform Promotion Shadow Qualification",
    "",
    f"Verdict: **{'PASS' if payload['passed'] else 'FAIL'}**",
    "",
    f"Candidate hash: `{payload['candidate_checkpoint_sha256']}`",
    "",
    "## Integrity",
    "",
    f"- Exact semantic replay: `{payload['semantic_replay_identical']}`",
    f"- Shadow control state unchanged: `{payload['shadow_control_state_unchanged']}`",
    f"- Context schedule passed: `{checks['passed']}`",
    f"- Screen/confirmation/reference games: `{checks['screen_games']}` / "
    f"`{checks['confirmation_games']}` / `{checks['reference_games']}`",
    f"- Screen contexts/context-seats: `{checks['screen_contexts']}` / "
    f"`{checks['screen_context_seats']}`",
    f"- Reference context-seats across two seeds: `{checks['reference_context_seats']}`",
    f"- Timing budgets passed: `{first['timing_checks']['passed']}`",
    f"- Windowed external strength recorded: "
    f"`{checks['external_strength_window_recorded']}`",
    "",
    "## Decision Telemetry",
    "",
    f"- Route: `{metrics['league/promotion_route']}`",
    f"- Rule admission: `{metrics['league/promotion_decision_admitted']}`; "
    f"live admission in shadow: `{metrics['league/promotion_accepted']}`",
    f"- Panel score/delta/LCB: `{metrics['league/promotion_panel_pooled_score']:.4f}` / "
    f"`{metrics['league/promotion_panel_relative_delta']:.4f}` / "
    f"`{metrics['league/promotion_panel_relative_lcb']:.4f}`",
    f"- Reference score/delta/LCB: "
    f"`{metrics.get('league/promotion_reference_candidate_score', 0.0):.4f}` / "
    f"`{metrics.get('league/promotion_reference_delta', 0.0):.4f}` / "
    f"`{metrics.get('league/promotion_reference_paired_lcb', 0.0):.4f}`",
    f"- Timeout rate: `{metrics['league/promotion_panel_timeout_rate']:.4f}`",
    "",
    "## Timing",
    "",
    f"- Anchor panel: `{metrics['league/eval_panel_anchor_seconds']:.2f}s`",
    f"- Screen: `{metrics['league/eval_screen_seconds']:.2f}s`",
    f"- Confirmation: `{metrics['league/eval_confirmation_seconds']:.2f}s`",
    f"- Reference: `{metrics['league/eval_reference_seconds']:.2f}s`",
    f"- Policy load: `{metrics['league/eval_policy_load_seconds']:.2f}s`",
    "",
  ]
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--league-state", type=Path, required=True)
  parser.add_argument("--promotion-state", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--replays", type=int, default=2)
  args = parser.parse_args()
  if args.replays != 2:
    raise ValueError("Qualification requires exactly two independent replays")
  for path in (args.config, args.checkpoint, args.league_state, args.promotion_state):
    if not path.exists():
      raise FileNotFoundError(path)
  if args.output_dir.exists():
    raise FileExistsError(f"Output directory already exists: {args.output_dir}")
  args.output_dir.mkdir(parents=True)

  replays = [_run_replay(args, replay_index) for replay_index in range(args.replays)]
  semantic_replay_identical = replays[0]["semantic_games"] == replays[1]["semantic_games"]
  shadow_unchanged = all(replay["control_state_unchanged"] for replay in replays)
  contexts_passed = all(replay["context_checks"]["passed"] for replay in replays)
  timing_passed = all(replay["timing_checks"]["passed"] for replay in replays)
  no_live_admission = all(
    float(replay["metrics"]["league/promotion_accepted"]) == 0.0 for replay in replays
  )
  candidate_hashes = {replay["candidate_checkpoint_sha256"] for replay in replays}
  payload = {
    "schema_version": 1,
    "candidate_checkpoint": str(args.checkpoint.resolve()),
    "candidate_checkpoint_sha256": next(iter(candidate_hashes)),
    "semantic_replay_identical": semantic_replay_identical,
    "shadow_control_state_unchanged": shadow_unchanged,
    "context_schedule_passed": contexts_passed,
    "timing_budgets_passed": timing_passed,
    "no_live_admission": no_live_admission,
    "passed": bool(
      len(candidate_hashes) == 1
      and semantic_replay_identical
      and shadow_unchanged
      and contexts_passed
      and timing_passed
      and no_live_admission
    ),
    "replays": [
      {key: value for key, value in replay.items() if key != "semantic_games"}
      for replay in replays
    ],
  }
  _json_write(args.output_dir / "summary.json", payload)
  (args.output_dir / "report.md").write_text(_markdown(payload), encoding="utf-8")
  print(
    "[uniform-promotion-qualification] "
    f"passed={payload['passed']} replay={semantic_replay_identical} "
    f"shadow_unchanged={shadow_unchanged} contexts={contexts_passed}",
    flush=True,
  )
  if not payload["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
