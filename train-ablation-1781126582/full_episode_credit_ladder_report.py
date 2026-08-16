#!/usr/bin/env python3
"""Synthesize the Stage 2 exact full-episode draft-credit ladder."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


WATER_GATES = frozenset(("STT02-002", "AZK01-126"))


def _read(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
  return float(sum(values) / len(values)) if values else 0.0


def _weighted_gate_score(payload: dict, gates: frozenset[str]) -> float:
  selected = [
    metrics
    for gate, metrics in payload["summary"]["by_candidate_gate"].items()
    if gate in gates
  ]
  games = sum(int(metrics["games"]) for metrics in selected)
  return (
    sum(int(metrics["games"]) * float(metrics["score"]) for metrics in selected)
    / max(games, 1)
  )


def _context_deck_summary(payload: dict) -> dict[str, float]:
  contexts = payload["contexts"]
  water_decks = [
    deck
    for context in contexts
    if context["element"] == "WATER"
    for deck in context["stochastic"]["decks"]
  ]
  return {
    "greedy_unique_mean": _mean([
      float(context["greedy"]["summary"]["main_unique"]) for context in contexts
    ]),
    "stochastic_unique_mean": _mean([
      float(context["stochastic"]["distribution"]["main_unique_mean"])
      for context in contexts
    ]),
    "stochastic_quad_slot_share_mean": _mean([
      float(context["stochastic"]["distribution"]["quad_slot_share_mean"])
      for context in contexts
    ]),
    "water_spell_slots_mean": _mean([
      float(deck["summary"]["type_slots"].get("SPELL", 0.0))
      for deck in water_decks
    ]),
  }


def _context_kl_summary(payload: dict) -> dict[str, float]:
  return {
    "gate_given_leader": float(
      payload["aggregate"]["gate_given_leader"]["symmetric_kl_mean"]
    ),
    "leader_given_gate": float(
      payload["aggregate"]["leader_given_gate"]["symmetric_kl_mean"]
    ),
    "determinism_max": max(
      float(value["determinism_kl_max"])
      for value in payload["aggregate"].values()
    ),
  }


def _hybrid_summary(payload: dict) -> dict[str, dict]:
  return {
    arm: dict(metrics["overall"])
    for arm, metrics in payload["matched_comparisons"].items()
  }


def _battle_delta(candidate: dict, control: dict) -> dict[str, float]:
  left = candidate["summary"]["policy_a"]["battle_metrics"]
  right = control["summary"]["policy_a"]["battle_metrics"]
  return {
    key: float(left[key]) - float(right.get(key, 0.0))
    for key in left
    if isinstance(left[key], (int, float))
  }


def _markdown(report: dict) -> str:
  decision = report["decision"]
  candidate_arm = report["candidate_arm"]
  lines = [
    f"# Stage 2 Draft-Credit Ladder: {candidate_arm}",
    "",
    f"Decision: **{decision['verdict']}**.",
    "",
    decision["reason"],
    "",
    "| Epoch | Credit vs control | Parent-panel delta | Heldout delta |",
    "| ---: | ---: | ---: | ---: |",
  ]
  for epoch in report["evaluation_epochs"]:
    window = report["windows"][str(epoch)]
    lines.append(
      f"| {epoch} | {window['credit_vs_control_score']:.3f} | "
      f"{window['parent_panel_delta']:+.3f} | {window['heldout_delta']:+.3f} |"
    )
  lines.extend([
    "",
    "## Registered Gates",
    "",
    f"- Integrity: `{decision['integrity_pass']}`",
    f"- Throughput diagnostic: `{decision['throughput_pass']}` "
    f"(`{report['throughput']['candidate_over_control']:.4f}` of control; "
    f"raw relative pass `{report['throughput']['relative_pass']}`; "
    f"relative waiver `{report['throughput']['relative_waived']}`; "
    f"raw absolute pass `{report['throughput']['absolute_pass']}`; "
    f"absolute waiver `{report['throughput']['absolute_waived']}`)",
    f"- External nonregression: `{decision['external_safety_pass']}`",
    f"- Mechanics safety: `{decision['mechanics_pass']}`",
    f"- Strength improvement: `{decision['strength_improvement_pass']}`",
    f"- Sustained causal context fit: `{decision['causal_improvement_pass']}`",
    "",
    "## Causal Context Fit",
    "",
    "| Epoch | Credit matched advantage | Control matched advantage | Delta | Pass |",
    "| ---: | ---: | ---: | ---: | --- |",
  ])
  for item in report["causal_windows"]:
    lines.append(
      f"| {item['epoch']} | {item['candidate_matched_advantage']:+.3f} | "
      f"{item['control_matched_advantage']:+.3f} | "
      f"{item['candidate_minus_control']:+.3f} | `{item['passed']}` |"
    )
  lines.extend([
    "",
    "The causal score is the matched main's advantage over the main drafted for "
    "both the sibling gate and sibling leader while gate, leader, opponent, seed, "
    "seat, and battle policy remain fixed. KL is diagnostic only.",
    "",
  ])
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--md", type=Path, required=True)
  parser.add_argument("--candidate-arm", default="full_episode_credit")
  parser.add_argument(
    "--confirmation",
    action="store_true",
    help="Emit the final Stage 2 acceptance verdict for a fresh 45M run.",
  )
  args = parser.parse_args()
  candidate_arm = args.candidate_arm
  if candidate_arm not in ("full_episode_credit", "prefix_outcome"):
    raise ValueError(f"unsupported candidate arm: {candidate_arm}")
  arms_tuple = ("control", candidate_arm)
  evaluation = args.root / "evaluation"
  manifest = _read(evaluation / "evaluation_manifest.json")
  epochs = [int(value) for value in manifest["evaluation_epochs"]]
  hybrid_epochs = [int(value) for value in manifest["causal_hybrid_epochs"]]
  training = {
    arm: _read(args.root / arm / "training_summary.json") for arm in arms_tuple
  }
  training_gate = _read(args.root / "training_gate.json")

  windows: dict[str, dict] = {}
  for epoch in epochs:
    root = evaluation / f"p{epoch}"
    direct = _read(root / "credit_vs_control.json")
    candidate_parent = _read(root / "credit_vs_parent.json")
    control_parent = _read(root / "control_vs_parent.json")
    candidate_heldout = _read(root / candidate_arm / "heldout.json")
    control_heldout = _read(root / "control" / "heldout.json")
    arms = {}
    for arm in arms_tuple:
      arm_root = root / arm
      arms[arm] = {
        "decks": _context_deck_summary(_read(arm_root / "context_decks.json")),
        "conditioning": _context_kl_summary(_read(arm_root / "context_kl.json")),
      }
      hybrid_path = arm_root / "context_hybrid.json"
      if hybrid_path.exists():
        arms[arm]["hybrids"] = _hybrid_summary(_read(hybrid_path))
    windows[str(epoch)] = {
      "credit_vs_control_score": float(direct["summary"]["score"]),
      "credit_vs_control_lcb80": float(direct["summary"]["paired_lcb_80"]),
      "credit_vs_parent_score": float(candidate_parent["summary"]["score"]),
      "control_vs_parent_score": float(control_parent["summary"]["score"]),
      "parent_panel_delta": float(candidate_parent["summary"]["score"])
      - float(control_parent["summary"]["score"]),
      "credit_heldout_score": float(candidate_heldout["summary"]["score"]),
      "control_heldout_score": float(control_heldout["summary"]["score"]),
      "heldout_delta": float(candidate_heldout["summary"]["score"])
      - float(control_heldout["summary"]["score"]),
      "water_heldout_delta": _weighted_gate_score(candidate_heldout, WATER_GATES)
      - _weighted_gate_score(control_heldout, WATER_GATES),
      "timeout_rate_max": max(
        float(direct["summary"]["timeout_rate"]),
        float(candidate_parent["summary"]["timeout_rate"]),
        float(control_parent["summary"]["timeout_rate"]),
        float(candidate_heldout["summary"]["timeout_rate"]),
        float(control_heldout["summary"]["timeout_rate"]),
      ),
      "battle_delta": _battle_delta(candidate_parent, control_parent),
      **arms,
    }

  candidate_training = training[candidate_arm]
  control_training = training["control"]
  sps_ratio = float(candidate_training["median_sps"]) / max(
    float(control_training["median_sps"]), 1e-9
  )
  credit_mode = candidate_training.get("credit_mode", "retained_rows")
  if credit_mode == "prefix_outcome":
    prefix = candidate_training["prefix_outcome"]
    credit_integrity = (
      prefix["telemetry_rows"] == candidate_training["epochs"]
      and prefix["delta_count"] > 0
      and prefix["residual_count"] > 0
      and prefix["completed"] > 0
      and prefix["truncated"] == 0
      and prefix["unsynchronized"] == 0
      and prefix["telescope_abs_max"] <= 1e-5
      and prefix["delta_abs_max"] > 0
      and prefix["prediction_std_mean"] > 0
      and prefix["inference_seconds"] > 0
      and all(value > 0 for value in prefix["quartile_deltas"].values())
      and training["control"]["prefix_outcome"]["telemetry_rows"] == 0
    )
  else:
    finite_signal = all(
      value is not None and math.isfinite(float(value))
      for value in (
        candidate_training["importance_mean_median"],
        candidate_training["clipfrac_max"],
      )
    )
    credit_integrity = (
      candidate_training["captured_records"] > 0
      and candidate_training["labeled_records"] > 0
      and candidate_training["completed_episodes"] > 0
      and candidate_training["decisive_episodes"] > 0
      and candidate_training["win_episodes"] > 0
      and candidate_training["loss_episodes"] > 0
      and (
        candidate_training["draw_episodes"]
        < candidate_training["completed_episodes"]
      )
      and candidate_training["trained_examples"] > 0
      and candidate_training["gradient_norm_max"] > 0
      and candidate_training["standard_actor_rows"] > 0
      and candidate_training["standard_masked_rows"] > 0
      and all(value > 0 for value in candidate_training["quartile_examples"].values())
      and candidate_training["incomplete_episodes"] == 0
      and finite_signal
    )
  integrity_pass = (
    credit_integrity
    and max(window["timeout_rate_max"] for window in windows.values()) == 0.0
    and all(
      arm["conditioning"]["determinism_max"] < 1e-7
      for window in windows.values()
      for arm in (window["control"], window[candidate_arm])
    )
  )
  relative_sps_pass = sps_ratio >= 0.95
  absolute_sps_pass = float(candidate_training["median_sps"]) >= 1235.0
  relative_sps_waived = bool(training_gate.get("relative_sps_waived", False))
  absolute_sps_waived = bool(training_gate.get("absolute_sps_waived", False))
  throughput_pass = (
    absolute_sps_pass or absolute_sps_waived
  ) and (
    relative_sps_pass or relative_sps_waived
  )
  if not bool(training_gate.get("throughput_diagnostic_only", False)):
    raise RuntimeError("training gate did not mark throughput as diagnostic-only")

  late = [windows[str(epoch)] for epoch in epochs[-3:]]
  direct_late_mean = _mean([window["credit_vs_control_score"] for window in late])
  parent_delta_late_mean = _mean([window["parent_panel_delta"] for window in late])
  heldout_delta_late_mean = _mean([window["heldout_delta"] for window in late])
  external_safety_pass = (
    direct_late_mean >= 0.47
    and min(window["parent_panel_delta"] for window in late) >= -0.05
    and min(window["heldout_delta"] for window in late) >= -0.05
  )
  endpoint = windows[str(epochs[-1])]
  battle = endpoint["battle_delta"]
  control_portal = float(
    _read(evaluation / f"p{epochs[-1]}" / "control_vs_parent.json")["summary"]
    ["policy_a"]["battle_metrics"]["portal_rate_mean"]
  )
  candidate_portal = control_portal + float(battle.get("portal_rate_mean", 0.0))
  portal_ratio = candidate_portal / max(control_portal, 1e-9)
  water_spell_delta = (
    endpoint[candidate_arm]["decks"]["water_spell_slots_mean"]
    - endpoint["control"]["decks"]["water_spell_slots_mean"]
  )
  mechanics_pass = (
    endpoint["water_heldout_delta"] >= -0.05
    and water_spell_delta >= -2.0
    and portal_ratio >= 0.90
    and float(battle.get("attack_rate_mean", 0.0)) >= -0.03
    and float(battle.get("spell_rate_mean", 0.0)) >= -0.02
    and float(battle.get("play_entity_to_garden_rate_mean", 0.0)) >= -0.01
    and float(battle.get("garden_or_leader_ability_rate_mean", 0.0)) >= -0.02
  )
  strength_improvement = (
    endpoint["credit_vs_control_score"] >= 0.53
    and endpoint["credit_vs_control_lcb80"] >= 0.45
    and (parent_delta_late_mean >= 0.02 or heldout_delta_late_mean >= 0.02)
  )

  causal_windows = []
  for epoch in hybrid_epochs:
    window = windows[str(epoch)]
    candidate = window[candidate_arm]["hybrids"]["sibling_both_main"]
    control = window["control"]["hybrids"]["sibling_both_main"]
    candidate_advantage = float(candidate["matched_advantage"])
    control_advantage = float(control["matched_advantage"])
    delta = candidate_advantage - control_advantage
    gate_advantage = float(
      window[candidate_arm]["hybrids"]["sibling_gate_main"]
      ["matched_advantage"]
    )
    leader_advantage = float(
      window[candidate_arm]["hybrids"]["sibling_leader_main"]
      ["matched_advantage"]
    )
    passed = (
      candidate_advantage >= 0.025
      and float(candidate["matched_advantage_ci80"][0]) >= -0.025
      and delta >= 0.02
      and min(gate_advantage, leader_advantage) >= -0.025
    )
    causal_windows.append({
      "epoch": epoch,
      "candidate_matched_advantage": candidate_advantage,
      "control_matched_advantage": control_advantage,
      "candidate_minus_control": delta,
      "gate_matched_advantage": gate_advantage,
      "leader_matched_advantage": leader_advantage,
      "passed": passed,
    })
  causal_improvement = (
    sum(bool(item["passed"]) for item in causal_windows) >= 2
    and bool(causal_windows[-1]["passed"])
  )
  efficacy_pass = strength_improvement or causal_improvement
  advance = all((
    integrity_pass,
    external_safety_pass,
    mechanics_pass,
    efficacy_pass,
  ))
  if advance:
    mechanism = (
      "Frozen prefix-outcome redistribution"
      if credit_mode == "prefix_outcome"
      else "Exact full-episode credit"
    )
    if args.confirmation:
      verdict = f"accept_{candidate_arm}"
      reason = (
        f"The fresh 45M {mechanism} confirmation clears integrity, external, and "
        "mechanics gates and retains a registered strength or sustained causal gain."
      )
    else:
      verdict = "advance_to_45m_confirmation"
      reason = (
        f"{mechanism} clears integrity, external, and "
        "mechanics gates and shows a registered strength or sustained causal gain."
      )
  elif not integrity_pass:
    verdict = "reject_integrity"
    reason = "The draft-credit path failed a label, gradient, determinism, or timeout invariant."
  elif not external_safety_pass:
    verdict = "reject_strength_regression"
    reason = "The candidate materially regressed on direct, parent-panel, or heldout strength."
  elif not mechanics_pass:
    verdict = "diagnose_mechanics_regression"
    reason = "Strength is not enough to waive the Water, portal, attack, spell, or Garden safety checks."
  else:
    verdict = "stop_after_neutral_15m"
    reason = (
      "The candidate is safe but lacks a greater-than-noise strength gain or a "
      "sustained causal gate-leader deck-fit improvement."
    )

  report = {
    "schema_version": 1,
    "campaign": args.root.name,
    "candidate_arm": candidate_arm,
    "credit_mode": credit_mode,
    "parent_epoch": int(manifest["parent_epoch"]),
    "target_epoch": int(manifest["target_epoch"]),
    "evaluation_epochs": epochs,
    "hybrid_epochs": hybrid_epochs,
    "promotion_used_as_signal": False,
    "confirmation": bool(args.confirmation),
    "training": training,
    "throughput": {
      "candidate_over_control": sps_ratio,
      "relative_floor": 0.95,
      "absolute_floor": 1235.0,
      "relative_pass": relative_sps_pass,
      "absolute_pass": absolute_sps_pass,
      "relative_waived": relative_sps_waived,
      "absolute_waived": absolute_sps_waived,
      "relative_waiver_reason": training_gate.get(
        "relative_sps_waiver_reason", ""
      ),
      "absolute_waiver_reason": training_gate.get(
        "absolute_sps_waiver_reason", ""
      ),
      "diagnostic_only": True,
    },
    "windows": windows,
    "late_window": {
      "credit_vs_control_mean": direct_late_mean,
      "parent_panel_delta_mean": parent_delta_late_mean,
      "heldout_delta_mean": heldout_delta_late_mean,
    },
    "endpoint_mechanics": {
      "water_heldout_delta": endpoint["water_heldout_delta"],
      "water_spell_slots_delta": water_spell_delta,
      "portal_rate_ratio": portal_ratio,
      "battle_delta": battle,
    },
    "causal_windows": causal_windows,
    "decision": {
      "verdict": verdict,
      "reason": reason,
      "advance": advance,
      "integrity_pass": integrity_pass,
      "throughput_pass": throughput_pass,
      "external_safety_pass": external_safety_pass,
      "mechanics_pass": mechanics_pass,
      "strength_improvement_pass": strength_improvement,
      "causal_improvement_pass": causal_improvement,
    },
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(_markdown(report), encoding="utf-8")
  print(
    f"[full-credit-report] verdict={verdict} direct_late={direct_late_mean:.4f} "
    f"parent_delta={parent_delta_late_mean:+.4f} "
    f"heldout_delta={heldout_delta_late_mean:+.4f} sps_ratio={sps_ratio:.4f}",
    flush=True,
  )


if __name__ == "__main__":
  main()
