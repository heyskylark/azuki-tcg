#!/usr/bin/env python3
"""Synthesize the matched Stage 1 uniform-assignment ladder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


DEFAULT_EPOCHS = (5000, 5200, 5500, 5840)
DEFAULT_HYBRID_EPOCHS = (5200, 5500, 5840)
ARMS = ("control", "uniform_assignment")


def _read(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
  return float(sum(values) / len(values)) if values else 0.0


def _parse_epochs(raw: str) -> tuple[int, ...]:
  try:
    epochs = tuple(int(value) for value in raw.replace(",", " ").split())
  except ValueError as exc:
    raise argparse.ArgumentTypeError("epochs must be integers") from exc
  if len(epochs) != len(set(epochs)) or any(epoch < 1 for epoch in epochs):
    raise argparse.ArgumentTypeError("epochs must be unique positive integers")
  return epochs


def _context_deck_summary(path: Path) -> dict[str, float]:
  payload = _read(path)
  contexts = payload["contexts"]
  return {
    "greedy_unique_mean": _mean(
      [float(context["greedy"]["summary"]["main_unique"]) for context in contexts]
    ),
    "greedy_ikz_mean": _mean(
      [float(context["greedy"]["summary"]["average_ikz_cost"]) for context in contexts]
    ),
    "stochastic_unique_mean": _mean(
      [
        float(context["stochastic"]["distribution"]["main_unique_mean"])
        for context in contexts
      ]
    ),
    "stochastic_quad_slot_share_mean": _mean(
      [
        float(context["stochastic"]["distribution"]["quad_slot_share_mean"])
        for context in contexts
      ]
    ),
    "stochastic_pairwise_jaccard_mean": _mean(
      [
        float(
          context["stochastic"]["distribution"][
            "pairwise_multiset_jaccard_mean"
          ]
        )
        for context in contexts
      ]
    ),
  }


def _training_summary(path: Path) -> dict[str, float | int | None]:
  rows: list[dict] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(row.get("epoch"), (int, float)):
      rows.append(row)
  interval_sps: list[float] = []
  for left, right in zip(rows, rows[1:]):
    elapsed = float(right["uptime"]) - float(left["uptime"])
    steps = float(right["agent_steps"]) - float(left["agent_steps"])
    if elapsed > 0.0 and steps > 0.0:
      interval_sps.append(steps / elapsed)
  steady = interval_sps[20:] if len(interval_sps) > 40 else interval_sps
  return {
    "metric_rows": len(rows),
    "epoch_first": int(rows[0]["epoch"]) if rows else None,
    "epoch_final": int(rows[-1]["epoch"]) if rows else None,
    "interval_sps_median": statistics.median(steady) if steady else 0.0,
    "tail100_interval_sps_median": (
      statistics.median(steady[-100:]) if steady else 0.0
    ),
    "timeout_rate_max": max(
      (float(row.get("environment/timeout_truncation_rate", 0.0)) for row in rows),
      default=0.0,
    ),
    "native_shaping_scale_final": (
      float(rows[-1].get("environment/effective_reward_shaping_scale", 0.0))
      if rows
      else None
    ),
  }


def _metric_deltas(left: dict, right: dict, section: str) -> dict[str, float]:
  left_metrics = left["summary"]["policy_a"][section]
  right_metrics = right["summary"]["policy_a"][section]
  return {
    key: float(left_metrics[key]) - float(right_metrics.get(key, 0.0))
    for key in left_metrics
    if isinstance(left_metrics[key], (int, float))
  }


def _hybrid_summary(path: Path) -> dict[str, dict]:
  payload = _read(path)
  return {
    arm: dict(metrics["overall"])
    for arm, metrics in payload["matched_comparisons"].items()
  }


def _markdown(report: dict) -> str:
  lines = [
    "# Stage 1 Uniform-Assignment Ladder",
    "",
    f"Decision: **{report['decision']['verdict']}**.",
    "",
    report["decision"]["reason"],
    "",
    "| Epoch | Uniform vs control | Uniform vs parent | Control vs parent | "
    "Uniform heldout | Control heldout |",
    "| ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for epoch in report["evaluation_epochs"]:
    row = report["windows"][str(epoch)]
    lines.append(
      f"| {epoch} | {row['uniform_vs_control_score']:.3f} | "
      f"{row['uniform_vs_parent_score']:.3f} | {row['control_vs_parent_score']:.3f} | "
      f"{row['uniform_heldout_score']:.3f} | {row['control_heldout_score']:.3f} |"
    )
  lines.extend(
    [
      "",
      "## Throughput",
      "",
      f"Control median interval SPS: `{report['training']['control']['interval_sps_median']:.2f}`.",
      "",
      f"Uniform median interval SPS: "
      f"`{report['training']['uniform_assignment']['interval_sps_median']:.2f}` "
      f"(`{report['sps']['candidate_over_control']:.4f}` of control).",
      "",
      "## Registered Gates",
      "",
      f"- Throughput: `{report['decision']['throughput_pass']}`",
      f"- Terminal/timeout integrity: `{report['decision']['integrity_pass']}`",
      f"- External strength safety: `{report['decision']['external_safety_pass']}`",
      f"- Mechanics/deck safety: `{report['decision']['mechanics_safety_pass']}`",
      f"- Causal context-fit safety: `{report['decision']['causal_safety_pass']}`",
      "",
      "## Endpoint Diagnostics",
      "",
      "| Arm | Gate KL given leader | Leader KL given gate | Stochastic unique | "
      "Four-copy slots |",
      "| --- | ---: | ---: | ---: | ---: |",
    ]
  )
  endpoint_key = str(report["target_epoch"])
  endpoint = report["windows"][endpoint_key]
  for arm in ARMS:
    values = endpoint[arm]
    lines.append(
      f"| {arm} | {values['gate_kl_given_leader']:.8f} | "
      f"{values['leader_kl_given_gate']:.8f} | "
      f"{values['decks']['stochastic_unique_mean']:.2f} | "
      f"{values['decks']['stochastic_quad_slot_share_mean']:.3f} |"
    )
  lines.extend(
    [
      "",
      "Endpoint uniform-minus-control mechanics deltas are retained in "
      f"`ladder_report.json` under `windows.{endpoint_key}.mechanics_delta`.",
      "",
    ]
  )
  lines.extend([
    "## Causal Context Fit",
    "",
    "| Epoch | Uniform matched advantage | Control matched advantage | Delta |",
    "| ---: | ---: | ---: | ---: |",
  ])
  for item in report["causal_windows"]:
    lines.append(
      f"| {item['epoch']} | {item['uniform_matched_advantage']:+.3f} | "
      f"{item['control_matched_advantage']:+.3f} | "
      f"{item['uniform_minus_control']:+.3f} |"
    )
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--md", type=Path, required=True)
  parser.add_argument(
    "--epochs", type=_parse_epochs, default=DEFAULT_EPOCHS,
    help="Comma- or space-separated full evaluation epochs",
  )
  parser.add_argument(
    "--hybrid-epochs", type=_parse_epochs, default=DEFAULT_HYBRID_EPOCHS,
    help="Comma- or space-separated causal-hybrid epochs",
  )
  args = parser.parse_args()
  if len(args.epochs) < 2:
    parser.error("--epochs requires at least two windows")
  if not args.hybrid_epochs:
    parser.error("--hybrid-epochs requires at least one window")
  if any(epoch not in args.epochs for epoch in args.hybrid_epochs):
    parser.error("every hybrid epoch must also be a full evaluation epoch")
  epochs = args.epochs
  hybrid_epochs = args.hybrid_epochs
  evaluation = args.root / "evaluation"

  training = {
    arm: _training_summary(args.root / arm / "train.jsonl") for arm in ARMS
  }
  sps_ratio = (
    float(training["uniform_assignment"]["interval_sps_median"])
    / max(float(training["control"]["interval_sps_median"]), 1e-9)
  )
  windows: dict[str, dict] = {}
  for epoch in epochs:
    root = evaluation / f"p{epoch}"
    direct = _read(root / "uniform_vs_control.json")
    uniform_parent = _read(root / "uniform_vs_parent.json")
    control_parent = _read(root / "control_vs_parent.json")
    uniform_heldout = _read(root / "uniform_assignment" / "heldout.json")
    control_heldout = _read(root / "control" / "heldout.json")
    arm_payload: dict[str, dict] = {}
    for arm in ARMS:
      kl_payload = _read(root / arm / "context_kl.json")
      arm_payload[arm] = {
        "decks": _context_deck_summary(root / arm / "context_decks.json"),
        "gate_kl_given_leader": float(
          kl_payload["aggregate"]["gate_given_leader"]["symmetric_kl_mean"]
        ),
        "leader_kl_given_gate": float(
          kl_payload["aggregate"]["leader_given_gate"]["symmetric_kl_mean"]
        ),
        "determinism_kl_max": max(
          float(value["determinism_kl_max"])
          for value in kl_payload["aggregate"].values()
        ),
      }
      hybrid_path = root / arm / "context_hybrid.json"
      if hybrid_path.exists():
        arm_payload[arm]["hybrids"] = _hybrid_summary(hybrid_path)
    windows[str(epoch)] = {
      "uniform_vs_control_score": float(direct["summary"]["score"]),
      "uniform_vs_control_paired_lcb_80": float(
        direct["summary"]["paired_lcb_80"]
      ),
      "uniform_vs_parent_score": float(uniform_parent["summary"]["score"]),
      "control_vs_parent_score": float(control_parent["summary"]["score"]),
      "parent_panel_delta": float(uniform_parent["summary"]["score"])
      - float(control_parent["summary"]["score"]),
      "uniform_heldout_score": float(uniform_heldout["summary"]["score"]),
      "control_heldout_score": float(control_heldout["summary"]["score"]),
      "heldout_delta": float(uniform_heldout["summary"]["score"])
      - float(control_heldout["summary"]["score"]),
      "timeout_rate_max": max(
        float(direct["summary"]["timeout_rate"]),
        float(uniform_parent["summary"]["timeout_rate"]),
        float(control_parent["summary"]["timeout_rate"]),
        float(uniform_heldout["summary"]["timeout_rate"]),
        float(control_heldout["summary"]["timeout_rate"]),
      ),
      "mechanics_delta": _metric_deltas(
        uniform_parent, control_parent, "battle_metrics"
      ),
      "uniform_battle_metrics": {
        key: float(value)
        for key, value in uniform_parent["summary"]["policy_a"]["battle_metrics"].items()
        if isinstance(value, (int, float))
      },
      "control_battle_metrics": {
        key: float(value)
        for key, value in control_parent["summary"]["policy_a"]["battle_metrics"].items()
        if isinstance(value, (int, float))
      },
      "panel_deck_delta": _metric_deltas(
        uniform_parent, control_parent, "deck_metrics"
      ),
      **arm_payload,
    }

  late = [windows[str(epoch)] for epoch in epochs[1:]]
  direct_late_mean = _mean(
    [float(window["uniform_vs_control_score"]) for window in late]
  )
  parent_delta_late_mean = _mean(
    [float(window["parent_panel_delta"]) for window in late]
  )
  heldout_delta_late_mean = _mean(
    [float(window["heldout_delta"]) for window in late]
  )
  timeouts_clean = max(
    float(window["timeout_rate_max"]) for window in windows.values()
  ) == 0.0
  sps_pass = sps_ratio >= 0.95 and float(
    training["uniform_assignment"]["interval_sps_median"]
  ) >= 1235.0
  safety_pass = (
    direct_late_mean >= 0.47
    and parent_delta_late_mean >= -0.05
    and heldout_delta_late_mean >= -0.05
    and float(windows[str(epochs[-1])]["uniform_vs_control_score"]) >= 0.47
    and float(windows[str(epochs[-1])]["parent_panel_delta"]) >= -0.05
    and float(windows[str(epochs[-1])]["heldout_delta"]) >= -0.05
  )
  endpoint = windows[str(epochs[-1])]
  mechanics_delta = endpoint["mechanics_delta"]
  control_portal = float(endpoint["control_battle_metrics"].get("portal_rate_mean", 0.0))
  uniform_portal = float(endpoint["uniform_battle_metrics"].get("portal_rate_mean", 0.0))
  portal_ratio = uniform_portal / max(control_portal, 1e-9)
  unique_delta = (
    endpoint["uniform_assignment"]["decks"]["stochastic_unique_mean"]
    - endpoint["control"]["decks"]["stochastic_unique_mean"]
  )
  quad_delta = (
    endpoint["uniform_assignment"]["decks"]["stochastic_quad_slot_share_mean"]
    - endpoint["control"]["decks"]["stochastic_quad_slot_share_mean"]
  )
  mechanics_safety_pass = (
    portal_ratio >= 0.90
    and float(mechanics_delta.get("attack_rate_mean", 0.0)) >= -0.03
    and float(mechanics_delta.get("spell_rate_mean", 0.0)) >= -0.02
    and float(mechanics_delta.get("play_entity_to_garden_rate_mean", 0.0)) >= -0.01
    and float(mechanics_delta.get("garden_or_leader_ability_rate_mean", 0.0)) >= -0.02
    and unique_delta >= -3.0
    and quad_delta <= 0.10
  )
  causal_windows = []
  for epoch in hybrid_epochs:
    window = windows[str(epoch)]
    uniform = window["uniform_assignment"]["hybrids"]["sibling_both_main"]
    control = window["control"]["hybrids"]["sibling_both_main"]
    uniform_advantage = float(uniform["matched_advantage"])
    control_advantage = float(control["matched_advantage"])
    causal_windows.append({
      "epoch": epoch,
      "uniform_matched_advantage": uniform_advantage,
      "control_matched_advantage": control_advantage,
      "uniform_minus_control": uniform_advantage - control_advantage,
      "uniform_gate_matched_advantage": float(
        window["uniform_assignment"]["hybrids"]["sibling_gate_main"]
        ["matched_advantage"]
      ),
      "uniform_leader_matched_advantage": float(
        window["uniform_assignment"]["hybrids"]["sibling_leader_main"]
        ["matched_advantage"]
      ),
    })
  causal_delta_mean = _mean([
    float(item["uniform_minus_control"]) for item in causal_windows
  ])
  causal_safety_pass = (
    causal_delta_mean >= -0.05
    and float(causal_windows[-1]["uniform_minus_control"]) >= -0.05
  )
  if (
    sps_pass
    and timeouts_clean
    and safety_pass
    and mechanics_safety_pass
    and causal_safety_pass
  ):
    verdict = "adopt uniform assignment"
    reason = (
      "The permanent 50-pick lifecycle clears throughput and integrity, and its "
      "late-window strength and causal deck-fit results show no material regression."
    )
  elif not sps_pass:
    verdict = "stop on throughput"
    reason = "The candidate failed the preregistered 95% relative or 1,235 absolute SPS guard."
  elif not mechanics_safety_pass:
    verdict = "diagnose before adoption"
    reason = (
      "The permanent lifecycle failed an endpoint portal, attack, spell, Garden, "
      "unique-card, or copy-concentration safety check."
    )
  else:
    verdict = "diagnose before adoption"
    reason = (
      "The environment contract is not rejected outright, but at least one late-window "
      "strength, integrity, or causal-fit guard requires diagnosis before Stage 2."
    )

  report = {
    "schema_version": 1,
    "campaign": args.root.name,
    "parent_epoch": min(
      int(training[arm]["epoch_first"] or 1) - 1 for arm in ARMS
    ),
    "target_epoch": epochs[-1],
    "evaluation_epochs": list(epochs),
    "hybrid_epochs": list(hybrid_epochs),
    "promotion_used_as_signal": False,
    "training": training,
    "sps": {
      "candidate_over_control": sps_ratio,
      "relative_guard": 0.95,
      "absolute_guard": 1235.0,
      "passed": sps_pass,
    },
    "windows": windows,
    "late_window": {
      "uniform_vs_control_score_mean": direct_late_mean,
      "parent_panel_delta_mean": parent_delta_late_mean,
      "heldout_delta_mean": heldout_delta_late_mean,
    },
    "causal_windows": causal_windows,
    "causal_safety": {
      "uniform_minus_control_mean": causal_delta_mean,
      "material_regression_floor": -0.05,
      "passed": causal_safety_pass,
    },
    "endpoint_mechanics": {
      "portal_rate_ratio": portal_ratio,
      "stochastic_unique_delta": unique_delta,
      "stochastic_quad_slot_share_delta": quad_delta,
      "battle_delta": mechanics_delta,
    },
    "decision": {
      "verdict": verdict,
      "reason": reason,
      "throughput_pass": sps_pass,
      "integrity_pass": timeouts_clean,
      "external_safety_pass": safety_pass,
      "mechanics_safety_pass": mechanics_safety_pass,
      "causal_safety_pass": causal_safety_pass,
    },
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(_markdown(report), encoding="utf-8")
  print(
    f"[uniform-report] verdict={verdict} direct_late={direct_late_mean:.4f} "
    f"parent_delta={parent_delta_late_mean:+.4f} "
    f"heldout_delta={heldout_delta_late_mean:+.4f} sps_ratio={sps_ratio:.4f}",
    flush=True,
  )


if __name__ == "__main__":
  main()
