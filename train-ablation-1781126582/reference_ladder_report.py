from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("control", "ref05", "ref10")
PROBABILITIES = {"control": 0.0, "ref05": 0.05, "ref10": 0.10}
LEAGUE_PREFIX = "environment/league/"


def _load_json(path: Path) -> dict:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"Expected a JSON object: {path}")
  return payload


def _load_jsonl(path: Path) -> list[dict]:
  rows = []
  with path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
      if isinstance(row, dict) and isinstance(row.get("SPS"), (int, float)):
        rows.append(row)
  if not rows:
    raise ValueError(f"No metric rows found in {path}")
  return rows


def _mean_metric(rows: list[dict], key: str) -> float:
  values = [
    float(row[key])
    for row in rows
    if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
  ]
  return float(np.mean(values)) if values else 0.0


def _score_by_game(payload: dict) -> dict[str, float]:
  out = {}
  for game in payload.get("games", []):
    game_id = str(game["game_id"])
    score = game.get("score", game.get("candidate_score"))
    if not isinstance(score, (int, float)) or isinstance(score, bool):
      raise ValueError(f"Game {game_id} is missing a numeric score")
    if game_id in out:
      raise ValueError(f"Duplicate game id {game_id}")
    out[game_id] = float(score)
  return out


def paired_score_delta(
  candidate: dict,
  control: dict,
  *,
  seed: int,
  samples: int = 20_000,
) -> dict:
  candidate_scores = _score_by_game(candidate)
  control_scores = _score_by_game(control)
  if candidate_scores.keys() != control_scores.keys():
    missing = sorted(control_scores.keys() - candidate_scores.keys())[:3]
    extra = sorted(candidate_scores.keys() - control_scores.keys())[:3]
    raise ValueError(f"Paired schedules differ: missing={missing}, extra={extra}")
  game_ids = sorted(candidate_scores)
  deltas = np.asarray(
    [candidate_scores[game_id] - control_scores[game_id] for game_id in game_ids],
    dtype=np.float64,
  )
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
  boot = deltas[indices].mean(axis=1)
  return {
    "games": len(deltas),
    "mean": float(deltas.mean()),
    "lcb80": float(np.quantile(boot, 0.10)),
    "ucb80": float(np.quantile(boot, 0.90)),
    "lcb95": float(np.quantile(boot, 0.025)),
    "ucb95": float(np.quantile(boot, 0.975)),
  }


def _training_summary(rows: list[dict]) -> dict:
  steady = rows[20:] if len(rows) > 40 else rows[1:]
  if not steady:
    steady = rows
  sps = np.asarray([float(row["SPS"]) for row in steady], dtype=np.float64)
  completed_key = LEAGUE_PREFIX + "reference_completed_episodes"
  aligned_key = LEAGUE_PREFIX + "reference_learner_drafter_fraction"
  completed = np.asarray(
    [float(row.get(completed_key, 0.0) or 0.0) for row in rows], dtype=np.float64
  )
  aligned = np.asarray(
    [float(row.get(aligned_key, 1.0) or 0.0) for row in rows], dtype=np.float64
  )
  completed_total = float(completed.sum())
  aligned_fraction = (
    float((completed * aligned).sum() / completed_total) if completed_total else 1.0
  )
  return {
    "metric_rows": len(rows),
    "steady_rows": len(steady),
    "sps_median": float(np.median(sps)),
    "sps_p10": float(np.quantile(sps, 0.10)),
    "sps_last100_median": float(np.median(sps[-100:])),
    "frozen_matchup_fraction_mean": _mean_metric(
      steady, LEAGUE_PREFIX + "frozen_matchup_fraction"
    ),
    "reference_configured_probability": _mean_metric(
      steady, LEAGUE_PREFIX + "reference_configured_matchup_probability"
    ),
    "reference_completed_episodes": completed_total,
    "reference_completed_episode_fraction_mean": _mean_metric(
      steady, LEAGUE_PREFIX + "reference_completed_episode_fraction"
    ),
    "reference_learner_drafter_fraction_weighted": aligned_fraction,
    "reference_matchup_step_fraction_mean": _mean_metric(
      steady, LEAGUE_PREFIX + "reference_matchup_step_fraction"
    ),
    "reference_trainable_row_fraction_mean": _mean_metric(
      steady, LEAGUE_PREFIX + "reference_trainable_row_fraction"
    ),
    "reference_trainable_battle_row_fraction_mean": _mean_metric(
      steady, LEAGUE_PREFIX + "reference_trainable_battle_row_fraction"
    ),
  }


def _evaluation_summary(payload: dict) -> dict:
  summary = payload["summary"]
  return {
    "episodes": int(summary["episodes"]),
    "score": float(summary["score"]),
    "timeout_rate": float(summary["timeout_rate"]),
    "deck_metrics": dict(summary.get("deck_metrics", {})),
    "battle_metrics": dict(summary.get("battle_metrics", {})),
  }


def _build_arm_summary(root: Path, arm: str) -> tuple[dict, dict[str, dict]]:
  arm_root = root / arm
  train_ref = _load_json(arm_root / "draftref_train.json")
  holdout_ref = _load_json(arm_root / "draftref_holdout.json")
  h2h = _load_json(arm_root / "h2h_vs_parent.json")
  rows = _load_jsonl(arm_root / "train.jsonl")
  return (
    {
      "probability": PROBABILITIES[arm],
      "training": _training_summary(rows),
      "train_reference": _evaluation_summary(train_ref),
      "holdout_reference": _evaluation_summary(holdout_ref),
      "h2h_vs_parent": {
        "episodes": int(h2h["summary"]["episodes"]),
        "score": float(h2h["summary"]["score"]),
        "paired_lcb_80": float(h2h["summary"]["paired_lcb_80"]),
        "timeout_rate": float(h2h["summary"]["timeout_rate"]),
        "battle_metrics": dict(h2h["summary"].get("battle_metrics", {})),
      },
    },
    {"train": train_ref, "holdout": holdout_ref, "h2h": h2h},
  )


def _screen_arm(arm: dict, control: dict, deltas: dict) -> dict:
  training = arm["training"]
  control_training = control["training"]
  throughput_ratio = training["sps_median"] / max(control_training["sps_median"], 1.0)
  train_deck = arm["train_reference"]["deck_metrics"]
  control_train_deck = control["train_reference"]["deck_metrics"]
  holdout_delta = deltas["holdout_reference"]["mean"]
  train_delta = deltas["train_reference"]["mean"]
  h2h_delta = deltas["h2h_vs_parent"]["mean"]
  train_similarity_delta = float(
    train_deck.get("nearest_train_reference_jaccard_mean", 0.0)
    - control_train_deck.get("nearest_train_reference_jaccard_mean", 0.0)
  )
  holdout_similarity_delta = float(
    train_deck.get("nearest_holdout_reference_jaccard_mean", 0.0)
    - control_train_deck.get("nearest_holdout_reference_jaccard_mean", 0.0)
  )
  similarity_gap_delta = float(
    train_deck.get("train_holdout_similarity_gap", 0.0)
    - control_train_deck.get("train_holdout_similarity_gap", 0.0)
  )
  dominant_share_delta = float(
    train_deck.get("dominant_main_deck_share", 0.0)
    - control_train_deck.get("dominant_main_deck_share", 0.0)
  )
  integrity_pass = (
    training["reference_completed_episodes"] > 0
    and training["reference_learner_drafter_fraction_weighted"] >= 0.999
    and abs(
      training["reference_configured_probability"] - arm["probability"]
    ) < 1e-6
  )
  throughput_pass = throughput_ratio >= 0.90 and training["sps_median"] >= 1300.0
  timeout_pass = max(
    arm["train_reference"]["timeout_rate"],
    arm["holdout_reference"]["timeout_rate"],
    arm["h2h_vs_parent"]["timeout_rate"],
  ) == 0.0
  no_mode_collapse = dominant_share_delta <= 0.15
  imitation_only = (
    train_similarity_delta >= 0.03
    and holdout_delta <= 0.0
    and h2h_delta <= 0.0
  )
  evidence_count = sum(
    (
      train_delta >= 0.02,
      holdout_delta >= 0.02,
      h2h_delta >= 0.02,
    )
  )
  eligible = integrity_pass and throughput_pass and timeout_pass and no_mode_collapse
  advance = (
    eligible
    and not imitation_only
    and evidence_count >= 2
    and deltas["holdout_reference"]["lcb80"] >= -0.02
    and arm["h2h_vs_parent"]["score"] >= 0.50
  )
  borderline = (
    eligible
    and not imitation_only
    and holdout_delta >= 0.0
    and arm["h2h_vs_parent"]["score"] >= 0.49
  )
  return {
    "status": "advance_45m" if advance else "borderline" if borderline else "stop",
    "throughput_ratio_vs_control": throughput_ratio,
    "integrity_pass": integrity_pass,
    "throughput_pass": throughput_pass,
    "timeout_pass": timeout_pass,
    "no_mode_collapse": no_mode_collapse,
    "imitation_only_pattern": imitation_only,
    "evidence_count": int(evidence_count),
    "train_similarity_delta": train_similarity_delta,
    "holdout_similarity_delta": holdout_similarity_delta,
    "similarity_gap_delta": similarity_gap_delta,
    "dominant_main_deck_share_delta": dominant_share_delta,
  }


def _markdown(report: dict) -> str:
  lines = [
    "# Reference Ladder 15M Readout",
    "",
    "Scores are measured against the same p2930 reference pilot and paired schedules. "
    "Action-rate changes are descriptive only and are not optimization targets.",
    "",
    "| Arm | SPS median | Ref completed | Learner drafted | Train ref | Holdout ref | H2H parent | Screen |",
    "|---|---:|---:|---:|---:|---:|---:|---|",
  ]
  for arm in ARMS:
    value = report["arms"][arm]
    screen = value.get("screen", {}).get("status", "baseline")
    lines.append(
      f"| {arm} | {value['training']['sps_median']:.0f} | "
      f"{value['training']['reference_completed_episodes']:.0f} | "
      f"{value['training']['reference_learner_drafter_fraction_weighted']:.3f} | "
      f"{value['train_reference']['score']:.3f} | "
      f"{value['holdout_reference']['score']:.3f} | "
      f"{value['h2h_vs_parent']['score']:.3f} | {screen} |"
    )
  lines.extend(["", "## Paired Deltas", ""])
  for arm in ("ref05", "ref10"):
    deltas = report["arms"][arm]["paired_deltas_vs_control"]
    lines.append(
      f"- **{arm}:** train `{deltas['train_reference']['mean']:+.3f}`, "
      f"holdout `{deltas['holdout_reference']['mean']:+.3f}` "
      f"(80% CI `{deltas['holdout_reference']['lcb80']:+.3f}` to "
      f"`{deltas['holdout_reference']['ucb80']:+.3f}`), H2H "
      f"`{deltas['h2h_vs_parent']['mean']:+.3f}`."
    )
  lines.extend(
    [
      "",
      "## Decision Rule",
      "",
      "Advance only when throughput and seat integrity pass, timeouts remain zero, "
      "deck diversity does not collapse, heldout performance is not meaningfully worse, "
      "and gains appear in at least two of train-reference, holdout-reference, and native H2H. "
      "A rise in training-deck similarity without heldout or H2H gain is flagged as imitation-only.",
      "",
    ]
  )
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare completed 15M reference ladder arms.")
  parser.add_argument("root", type=Path)
  parser.add_argument("--json", type=Path, default=None)
  parser.add_argument("--markdown", type=Path, default=None)
  args = parser.parse_args()

  summaries = {}
  payloads = {}
  for arm in ARMS:
    summaries[arm], payloads[arm] = _build_arm_summary(args.root, arm)
  control_payloads = payloads["control"]
  for arm_index, arm in enumerate(("ref05", "ref10"), start=1):
    deltas = {
      split_name: paired_score_delta(
        payloads[arm][payload_name],
        control_payloads[payload_name],
        seed=42_700_000 + arm_index * 100 + split_index,
      )
      for split_index, (split_name, payload_name) in enumerate(
        (
          ("train_reference", "train"),
          ("holdout_reference", "holdout"),
          ("h2h_vs_parent", "h2h"),
        )
      )
    }
    summaries[arm]["paired_deltas_vs_control"] = deltas
    summaries[arm]["screen"] = _screen_arm(summaries[arm], summaries["control"], deltas)

  advance = [
    arm
    for arm in ("ref05", "ref10")
    if summaries[arm]["screen"]["status"] == "advance_45m"
  ]
  report = {
    "schema_version": 1,
    "parent": "qualified_p2930",
    "arms": summaries,
    "advance_candidates": advance,
  }
  json_path = args.json or args.root / "ladder_report.json"
  markdown_path = args.markdown or args.root / "ladder_report.md"
  json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  markdown_path.write_text(_markdown(report), encoding="utf-8")
  print(f"[reference-ladder-report] wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
  main()
