from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_ladder_report import (
  _evaluation_summary,
  _load_json,
  _load_jsonl,
  _training_summary,
  paired_score_delta,
)


ARMS = ("control", "ref05")
PROBABILITIES = {"control": 0.0, "ref05": 0.05}


def _arm_summary(root: Path, arm: str) -> tuple[dict, dict[str, dict]]:
  arm_root = root / arm
  train_reference = _load_json(arm_root / "train_reference.json")
  holdout_reference = _load_json(arm_root / "holdout_reference.json")
  h2h = _load_json(arm_root / "h2h_vs_parent.json")
  training = _training_summary(_load_jsonl(arm_root / "train.jsonl"))
  return (
    {
      "probability": PROBABILITIES[arm],
      "training": training,
      "train_reference": _evaluation_summary(train_reference),
      "holdout_reference": _evaluation_summary(holdout_reference),
      "h2h_vs_parent": {
        "episodes": int(h2h["summary"]["episodes"]),
        "score": float(h2h["summary"]["score"]),
        "paired_lcb_80": float(h2h["summary"]["paired_lcb_80"]),
        "timeout_rate": float(h2h["summary"]["timeout_rate"]),
        "battle_metrics": dict(h2h["summary"].get("battle_metrics", {})),
      },
    },
    {
      "train_reference": train_reference,
      "holdout_reference": holdout_reference,
      "h2h_vs_parent": h2h,
    },
  )


def _screen(candidate: dict, control: dict, deltas: dict) -> dict:
  candidate_training = candidate["training"]
  control_training = control["training"]
  candidate_decks = candidate["train_reference"]["deck_metrics"]
  control_decks = control["train_reference"]["deck_metrics"]
  dominant_share_delta = float(
    candidate_decks.get("dominant_main_deck_share", 0.0)
    - control_decks.get("dominant_main_deck_share", 0.0)
  )
  integrity_pass = (
    control_training["reference_completed_episodes"] == 0
    and abs(control_training["reference_configured_probability"]) < 1e-9
    and candidate_training["reference_completed_episodes"] > 0
    and candidate_training["reference_learner_drafter_fraction_weighted"] >= 0.999
    and abs(candidate_training["reference_configured_probability"] - 0.05) < 1e-6
  )
  timeout_pass = max(
    candidate["train_reference"]["timeout_rate"],
    candidate["holdout_reference"]["timeout_rate"],
    candidate["h2h_vs_parent"]["timeout_rate"],
    control["train_reference"]["timeout_rate"],
    control["holdout_reference"]["timeout_rate"],
    control["h2h_vs_parent"]["timeout_rate"],
  ) == 0.0
  no_mode_collapse = dominant_share_delta <= 0.15
  evidence_count = sum(
    deltas[name]["mean"] >= 0.02
    for name in ("train_reference", "holdout_reference", "h2h_vs_parent")
  )
  advance = (
    integrity_pass
    and timeout_pass
    and no_mode_collapse
    and evidence_count >= 2
    and deltas["holdout_reference"]["lcb80"] >= -0.02
    and candidate["h2h_vs_parent"]["score"] >= 0.50
  )
  return {
    "status": "continue_to_45m" if advance else "stop_after_15m",
    "integrity_pass": integrity_pass,
    "timeout_pass": timeout_pass,
    "no_mode_collapse": no_mode_collapse,
    "evidence_count": int(evidence_count),
    "dominant_main_deck_share_delta": dominant_share_delta,
    "sps_observational_only": {
      "control_median": control_training["sps_median"],
      "ref05_median": candidate_training["sps_median"],
      "ratio": candidate_training["sps_median"]
      / max(control_training["sps_median"], 1.0),
    },
  }


def _markdown(report: dict) -> str:
  lines = [
    "# Regional Reference-Seat 15M Readout",
    "",
    "SPS is reported but is not a rejection criterion. Scores use identical paired schedules.",
    "",
    "| Arm | Sampled rows | SPS median | Fixed-deck games | Train refs | Holdout refs | H2H parent |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for arm in ARMS:
    value = report["arms"][arm]
    lines.append(
      f"| {arm} | {value['sampled_rows']:,} | {value['training']['sps_median']:.0f} | "
      f"{value['training']['reference_completed_episodes']:.0f} | "
      f"{value['train_reference']['score']:.3f} | "
      f"{value['holdout_reference']['score']:.3f} | "
      f"{value['h2h_vs_parent']['score']:.3f} |"
    )
  deltas = report["arms"]["ref05"]["paired_deltas_vs_control"]
  lines.extend(["", "## Paired ref05 minus control", ""])
  for name, value in deltas.items():
    lines.append(
      f"- **{name}:** `{value['mean']:+.4f}`; 80% CI "
      f"`{value['lcb80']:+.4f}` to `{value['ucb80']:+.4f}`; 95% CI "
      f"`{value['lcb95']:+.4f}` to `{value['ucb95']:+.4f}`."
    )
  screen = report["arms"]["ref05"]["screen"]
  lines.extend(
    [
      "",
      "## Decision",
      "",
      f"**{screen['status']}**",
      "",
      "Continue only when reference-seat integrity and zero-timeout checks pass, deck diversity "
      "does not collapse, at least two of train-reference, signature-heldout, and H2H improve "
      "by 2 points, the heldout 80% lower bound is at least -2 points, and H2H is at least 50%.",
      "",
    ]
  )
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("root", type=Path)
  parser.add_argument("--json", type=Path, default=None)
  parser.add_argument("--markdown", type=Path, default=None)
  args = parser.parse_args()

  summaries: dict[str, dict] = {}
  payloads: dict[str, dict[str, dict]] = {}
  for arm in ARMS:
    summaries[arm], payloads[arm] = _arm_summary(args.root, arm)
    summaries[arm]["sampled_rows"] = 14_899_200

  deltas = {
    name: paired_score_delta(
      payloads["ref05"][name],
      payloads["control"][name],
      seed=42_700_100 + index,
      samples=100_000,
    )
    for index, name in enumerate(
      ("train_reference", "holdout_reference", "h2h_vs_parent")
    )
  }
  summaries["ref05"]["paired_deltas_vs_control"] = deltas
  summaries["ref05"]["screen"] = _screen(summaries["ref05"], summaries["control"], deltas)
  report = {
    "schema_version": 1,
    "parent_update": 11030,
    "target_update": 12000,
    "sampled_rows_per_arm": 14_899_200,
    "sps_is_rejection_criterion": False,
    "arms": summaries,
    "decision": summaries["ref05"]["screen"]["status"],
  }
  json_path = args.json or args.root / "ladder_report.json"
  markdown_path = args.markdown or args.root / "ladder_report.md"
  json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  markdown_path.write_text(_markdown(report), encoding="utf-8")
  print(f"[regional-reference-ladder] wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
  main()
