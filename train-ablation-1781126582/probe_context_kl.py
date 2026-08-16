#!/usr/bin/env python3
"""Interventional main-draft KL for gate and leader context separately."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from probe_gate_kl import EpisodeRunner, GATE_CODE_PAIRS, kl, tv


DEFAULT_CONFIG = Path("python/config/azuki_deckbuild_native_3090.ini")
QUARTILES = ((1, 13), (14, 25), (26, 38), (39, 50))


def _quartile(pick: int) -> str:
  for index, (low, high) in enumerate(QUARTILES, start=1):
    if low <= pick <= high:
      return f"q{index}"
  raise ValueError(f"Main pick outside 1..50: {pick}")


def _summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
  if not rows:
    return {
      "pick_rows": 0,
      "kl_source_to_target_mean": 0.0,
      "kl_target_to_source_mean": 0.0,
      "symmetric_kl_mean": 0.0,
      "symmetric_kl_p90": 0.0,
      "total_variation_mean": 0.0,
      "determinism_kl_mean": 0.0,
      "determinism_kl_max": 0.0,
    }
  values = lambda key: np.asarray([float(row[key]) for row in rows])
  sym = values("symmetric_kl")
  control = values("determinism_kl")
  return {
    "pick_rows": len(rows),
    "kl_source_to_target_mean": float(values("kl_source_to_target").mean()),
    "kl_target_to_source_mean": float(values("kl_target_to_source").mean()),
    "symmetric_kl_mean": float(sym.mean()),
    "symmetric_kl_p90": float(np.quantile(sym, 0.9)),
    "total_variation_mean": float(values("total_variation").mean()),
    "determinism_kl_mean": float(control.mean()),
    "determinism_kl_max": float(control.max()),
  }


def _group(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, float | int]]:
  groups: dict[str, list[dict[str, object]]] = defaultdict(list)
  for row in rows:
    groups[str(row[key])].append(row)
  return {name: _summary(selected) for name, selected in sorted(groups.items())}


def _markdown(payload: dict[str, object]) -> str:
  lines = [
    "# Uniform Context Draft KL",
    "",
    f"Checkpoint: `{payload['checkpoint']}`",
    "",
    f"Histories per direction: `{payload['histories_per_direction']}`.",
    "",
    "| Intervention | Pick rows | Symmetric KL | P90 | TV | Control KL |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  aggregate = payload["aggregate"]
  for intervention, summary in aggregate.items():
    lines.append(
      f"| {intervention} | {summary['pick_rows']} | "
      f"{summary['symmetric_kl_mean']:.8f} | {summary['symmetric_kl_p90']:.8f} | "
      f"{summary['total_variation_mean']:.8f} | "
      f"{summary['determinism_kl_mean']:.3e} |"
    )
  for intervention in ("gate_given_leader", "leader_given_gate"):
    lines.extend(
      [
        "",
        f"## {intervention}",
        "",
        "| Element | Pick rows | Symmetric KL | P90 | TV |",
        "| --- | ---: | ---: | ---: | ---: |",
      ]
    )
    for element, summary in payload["by_element"][intervention].items():
      lines.append(
        f"| {element} | {summary['pick_rows']} | "
        f"{summary['symmetric_kl_mean']:.8f} | {summary['symmetric_kl_p90']:.8f} | "
        f"{summary['total_variation_mean']:.8f} |"
      )
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--histories", type=int, default=4)
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--smoothing-eps", type=float, default=0.05)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--md", type=Path, required=True)
  args = parser.parse_args()
  if args.histories < 1:
    raise ValueError("--histories must be positive")

  from policy.v2 import tcg_sampler

  tcg_sampler.set_sampling_params(
    subaction_temperature=args.temperature,
    smoothing_eps=args.smoothing_eps,
  )
  runner = EpisodeRunner(
    args.config,
    args.checkpoint,
    args.device,
    uniform_assignment=True,
  )
  catalog = runner.catalog
  code_by_id = {
    int(card_id): record.card_code
    for card_id, record in catalog.records_by_def_id.items()
  }
  leaders_by_element = {
    element: tuple(
      code_by_id[int(leader_id)]
      for leader_id in catalog.leader_def_ids_by_element[element]
    )
    for element in GATE_CODE_PAIRS
  }
  opponent_leader = code_by_id[int(catalog.leader_def_ids_by_element["WATER"][0])]
  rows: list[dict[str, object]] = []

  def compare(
    *,
    intervention: str,
    element: str,
    fixed_context: str,
    source_gate: str,
    target_gate: str,
    source_leader: str,
    target_leader: str,
    history_index: int,
  ) -> None:
    seed = 83_000_021 + 100_003 * history_index
    actions, source_probs, source_candidates = runner.run_episode(
      seed,
      source_gate,
      None,
      p0_leader_code=source_leader,
      p1_leader_code=opponent_leader,
    )
    _, target_probs, target_candidates = runner.run_episode(
      seed,
      target_gate,
      actions,
      p0_leader_code=target_leader,
      p1_leader_code=opponent_leader,
    )
    _, control_probs, control_candidates = runner.run_episode(
      seed,
      source_gate,
      actions,
      p0_leader_code=source_leader,
      p1_leader_code=opponent_leader,
    )
    if not (
      len(source_probs)
      == len(target_probs)
      == len(control_probs)
      == len(source_candidates)
      == len(target_candidates)
      == len(control_candidates)
      == 50
    ):
      raise RuntimeError("Uniform context KL probe expected exactly 50 main picks")
    for index, (source, target, control) in enumerate(
      zip(source_probs, target_probs, control_probs),
      start=1,
    ):
      if not np.array_equal(source_candidates[index - 1], target_candidates[index - 1]):
        raise RuntimeError("Intervention changed the main-card candidate set")
      if not np.array_equal(source_candidates[index - 1], control_candidates[index - 1]):
        raise RuntimeError("Determinism replay changed the candidate set")
      rows.append(
        {
          "intervention": intervention,
          "element": element,
          "fixed_context": fixed_context,
          "source_gate": source_gate,
          "target_gate": target_gate,
          "source_leader": source_leader,
          "target_leader": target_leader,
          "direction": (
            f"{source_gate}:{source_leader}->{target_gate}:{target_leader}"
          ),
          "history_index": history_index,
          "seed": seed,
          "main_pick": index,
          "quartile": _quartile(index),
          "kl_source_to_target": kl(source, target),
          "kl_target_to_source": kl(target, source),
          "symmetric_kl": 0.5 * (kl(source, target) + kl(target, source)),
          "total_variation": tv(source, target),
          "determinism_kl": kl(source, control),
        }
      )

  try:
    for element, (gate_a, gate_b) in GATE_CODE_PAIRS.items():
      leader_a, leader_b = leaders_by_element[element]
      for leader in (leader_a, leader_b):
        for source_gate, target_gate in ((gate_a, gate_b), (gate_b, gate_a)):
          for history_index in range(args.histories):
            compare(
              intervention="gate_given_leader",
              element=element,
              fixed_context=leader,
              source_gate=source_gate,
              target_gate=target_gate,
              source_leader=leader,
              target_leader=leader,
              history_index=history_index,
            )
      for gate in (gate_a, gate_b):
        for source_leader, target_leader in (
          (leader_a, leader_b),
          (leader_b, leader_a),
        ):
          for history_index in range(args.histories):
            compare(
              intervention="leader_given_gate",
              element=element,
              fixed_context=gate,
              source_gate=gate,
              target_gate=gate,
              source_leader=source_leader,
              target_leader=target_leader,
              history_index=history_index,
            )
  finally:
    runner.vecenv.close()

  by_intervention = {
    intervention: [row for row in rows if row["intervention"] == intervention]
    for intervention in ("gate_given_leader", "leader_given_gate")
  }
  payload: dict[str, object] = {
    "schema_version": 1,
    "checkpoint": str(args.checkpoint.resolve()),
    "lifecycle": "uniform_gate_uniform_compatible_leader_50_main_picks",
    "histories_per_direction": args.histories,
    "sampling": {
      "temperature": args.temperature,
      "smoothing_eps": args.smoothing_eps,
    },
    "aggregate": {
      intervention: _summary(selected)
      for intervention, selected in by_intervention.items()
    },
    "by_element": {
      intervention: _group(selected, "element")
      for intervention, selected in by_intervention.items()
    },
    "by_direction": {
      intervention: _group(selected, "direction")
      for intervention, selected in by_intervention.items()
    },
    "by_quartile": {
      intervention: _group(selected, "quartile")
      for intervention, selected in by_intervention.items()
    },
    "rows": rows,
  }
  control_max = max(
    summary["determinism_kl_max"]
    for summary in payload["aggregate"].values()
  )
  if float(control_max) > 1e-7:
    raise RuntimeError(f"Determinism control KL exceeded tolerance: {control_max}")
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(_markdown(payload), encoding="utf-8")
  print(
    "[context-kl] "
    + " ".join(
      f"{name}={summary['symmetric_kl_mean']:.8f}"
      for name, summary in payload["aggregate"].items()
    ),
    flush=True,
  )


if __name__ == "__main__":
  main()
