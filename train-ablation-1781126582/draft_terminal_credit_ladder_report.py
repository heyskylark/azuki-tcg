#!/usr/bin/env python3
"""Build the preregistered Step 5b trajectory and advancement report."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from strategy_recovery_ladder_report import (
    _evaluation_summary,
    _h2h_summary,
    _load_json,
    _opportunity_summary,
    _paired_delta,
)


EPOCHS = (5200, 5500, 5847)
ARMS = ("control", "draft_terminal_long")


def _deck_counter(deck: dict) -> Counter[str]:
    return Counter({str(card["code"]): int(card["copies"]) for card in deck["cards"]})


def _multiset_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left).union(right)
    intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return float(intersection / union) if union else 1.0


def _deck_summary(payload: dict) -> dict[str, object]:
    gates = {}
    for gate_code, gate in payload["gates"].items():
        sampled = gate["sampled"]
        gates[gate_code] = {
            "leader_split": dict(sampled["leader_split"]),
            "average_cost": float(sampled["summary"]["avg_cost"]),
            "type_share": dict(sampled["summary"]["type_share"]),
            "distribution": dict(sampled["distribution"]),
            "by_leader": dict(sampled["by_leader"]),
            "argmax_leader": str(gate["argmax_deck"]["leader"]["code"]),
            "argmax_cards": {
                str(card["code"]): int(card["copies"])
                for card in gate["argmax_deck"]["cards"]
            },
        }
    distributions = [gate["distribution"] for gate in gates.values()]
    costs = [float(gate["average_cost"]) for gate in gates.values()]
    type_names = sorted(
        {name for gate in gates.values() for name in gate["type_share"]}
    )
    return {
        "main_unique_mean": float(
            np.mean([item["main_unique_mean"] for item in distributions])
        ),
        "singleton_slot_share_mean": float(
            np.mean([item["singleton_slot_share_mean"] for item in distributions])
        ),
        "quad_slot_share_mean": float(
            np.mean([item["quad_slot_share_mean"] for item in distributions])
        ),
        "within_gate_pairwise_jaccard_mean": float(
            np.mean(
                [
                    item["within_gate_pairwise_multiset_jaccard_mean"]
                    for item in distributions
                ]
            )
        ),
        "average_cost": float(np.mean(costs)),
        "type_share": {
            name: float(
                np.mean([float(gate["type_share"].get(name, 0.0)) for gate in gates.values()])
            )
            for name in type_names
        },
        "gates": gates,
    }


def _paired_deck_summary(control: dict, candidate: dict) -> dict[str, object]:
    by_gate = {}
    all_jaccards = []
    leader_matches = []
    for gate in control["gates"]:
        control_decks = {
            int(deck["seed"]): deck for deck in control["gates"][gate]["sampled"]["decks"]
        }
        candidate_decks = {
            int(deck["seed"]): deck for deck in candidate["gates"][gate]["sampled"]["decks"]
        }
        if control_decks.keys() != candidate_decks.keys():
            raise ValueError(f"Paired sampled-deck seeds differ for {gate}")
        jaccards = []
        gate_leader_matches = []
        for seed in sorted(control_decks):
            left = control_decks[seed]
            right = candidate_decks[seed]
            jaccards.append(
                _multiset_jaccard(_deck_counter(left), _deck_counter(right))
            )
            gate_leader_matches.append(str(left["leader"]) == str(right["leader"]))
        all_jaccards.extend(jaccards)
        leader_matches.extend(gate_leader_matches)
        by_gate[gate] = {
            "paired_decks": len(jaccards),
            "main_multiset_jaccard_mean": float(np.mean(jaccards)),
            "leader_agreement": float(np.mean(gate_leader_matches)),
        }
    return {
        "paired_decks": len(all_jaccards),
        "main_multiset_jaccard_mean": float(np.mean(all_jaccards)),
        "main_multiset_jaccard_p10": float(np.quantile(all_jaccards, 0.10)),
        "leader_agreement": float(np.mean(leader_matches)),
        "by_gate": by_gate,
    }


def _conditioning_summary(payload: dict) -> dict[str, object]:
    sibling = {
        element: float(metrics["mean_symmetric_kl"])
        for element, metrics in payload["sibling_gate"].items()
    }
    leader_by_gate = {
        gate: float(metrics["mean_symmetric_kl"])
        for gate, metrics in payload["leader_conditioned"].items()
    }
    control_max = max(
        float(metrics["control_max_kl"])
        for metrics in payload["sibling_gate"].values()
    )
    return {
        "sibling_gate_main_pick_kl": float(np.mean(list(sibling.values()))),
        "sibling_gate_by_element": sibling,
        "leader_conditioned_main_pick_kl": float(
            np.mean(list(leader_by_gate.values()))
        ),
        "leader_conditioned_by_gate": leader_by_gate,
        "determinism_control_max_kl": control_max,
    }


def _hybrid_summary(payload: dict) -> dict[str, object]:
    comparisons = payload["matched_comparisons"]
    return {
        arm: {
            "overall": dict(metrics["overall"]),
            "by_element": dict(metrics["by_element"]),
            "by_gate": dict(metrics["by_gate"]),
        }
        for arm, metrics in comparisons.items()
    }


def _load_window(
    root: Path, label: str, trajectory_dir: str = "trajectory"
) -> tuple[dict, dict]:
    path = root / trajectory_dir / label
    holdout_raw = _load_json(path / "draftref_holdout.json")
    decks_raw = _load_json(path / "decks.json")
    window = {
        "label": label,
        "checkpoint": (path / "checkpoint.txt").read_text(encoding="utf-8").strip(),
        "holdout_reference": _evaluation_summary(holdout_raw),
        "decks": _deck_summary(decks_raw),
        "conditioning": _conditioning_summary(
            _load_json(path / "draft_conditioning.json")
        ),
        "opportunity": _opportunity_summary(_load_json(path / "opportunity.json")),
        "hybrids": _hybrid_summary(_load_json(path / "hybrid.json")),
    }
    train_path = path / "draftref_train.json"
    if train_path.exists():
        window["train_reference"] = _evaluation_summary(_load_json(train_path))
    parent_path = path / "h2h_vs_parent.json"
    if parent_path.exists():
        window["h2h_vs_parent"] = _h2h_summary(_load_json(parent_path))
    direct_path = path / "h2h_vs_control.json"
    if direct_path.exists():
        window["h2h_vs_control"] = _h2h_summary(_load_json(direct_path))
    return window, {
        "holdout": holdout_raw,
        "decks": decks_raw,
        "h2h_parent": _load_json(parent_path) if parent_path.exists() else None,
        "h2h_control": _load_json(direct_path) if direct_path.exists() else None,
    }


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _build_screen(report: dict) -> dict[str, object]:
    control_training = report["training"]["control"]
    candidate_training = report["training"]["draft_terminal_long"]
    final = report["trajectory"]["5847"]
    control = final["control"]
    candidate = final["draft_terminal_long"]
    paired = final["paired"]
    direct = candidate["h2h_vs_control"]
    throughput_ratio = (
        candidate_training["steady_sps_median"]
        / control_training["steady_sps_median"]
    )

    position_integrity = all(
        float(metrics["examples"]) > 0
        and all(
            metrics.get(key) is not None and _finite(metrics[key])
            for key in (
                "advantage_mean",
                "advantage_abs_mean",
                "advantage_std",
                "baseline_bce",
                "baseline_brier",
            )
        )
        for metrics in candidate_training["positions"].values()
    )
    signal_integrity = (
        candidate_training["captured_records"] > 0
        and candidate_training["labeled_records"] > 0
        and candidate_training["trained_examples"] > 0
        and candidate_training["trained_examples"] % 5 == 0
        and candidate_training["fixed_batch_rows"]
        >= candidate_training["trained_examples"]
        and candidate_training["incomplete_episodes"] == 0
        and candidate_training["incomplete_records"] == 0
        and candidate_training["clipfrac_max"] is not None
        and candidate_training["clipfrac_max"] <= 0.25
        and _finite(candidate_training["importance_mean_median"])
        and _finite(candidate_training["gradient_norm_median"])
        and position_integrity
    )
    schedule_integrity = all(
        training["epoch_final"] == 5847
        and training["learning_rate_final"] <= 1e-8
        and training["trainer_shaped_reward_multiplier_min"] == 1.0
        and training["trainer_shaped_reward_multiplier_max"] == 1.0
        and abs(training["native_reward_shaping_scale_final"] - 0.15) <= 1e-3
        for training in (control_training, candidate_training)
    )
    determinism_integrity = all(
        arm["conditioning"]["determinism_control_max_kl"] < 1e-6
        for epoch in report["trajectory"].values()
        for arm in (epoch["control"], epoch["draft_terminal_long"])
    )
    throughput_pass = (
        throughput_ratio >= 0.95
        and candidate_training["steady_sps_median"] >= 1235.0
    )

    parent_means = [
        report["trajectory"][str(epoch)]["paired"]["h2h_vs_parent"]["mean"]
        for epoch in EPOCHS
    ]
    holdout_means = [
        report["trajectory"][str(epoch)]["paired"]["holdout_reference"]["mean"]
        for epoch in EPOCHS
    ]
    strength_nonregression = (
        direct["score"] >= 0.47
        and direct["paired_lcb80"] >= 0.43
        and paired["h2h_vs_parent"]["lcb80"] >= -0.05
        and paired["holdout_reference"]["lcb80"] >= -0.05
        and min(parent_means) >= -0.05
        and min(holdout_means) >= -0.05
    )
    water_delta = (
        candidate["holdout_reference"]["water_score"]
        - control["holdout_reference"]["water_score"]
    )
    portal_ratio = candidate["opportunity"]["portals_per_game"] / max(
        control["opportunity"]["portals_per_game"], 1e-9
    )
    behavior_guards = water_delta >= -0.05 and portal_ratio >= 0.90

    strength_improvement = (
        direct["score"] >= 0.53
        and paired["h2h_vs_parent"]["lcb80"] >= 0.0
    )
    causal_windows = []
    for epoch in EPOCHS:
        window = report["trajectory"][str(epoch)]
        candidate_causal = window["draft_terminal_long"]["hybrids"]["sibling_main"][
            "overall"
        ]
        control_causal = window["control"]["hybrids"]["sibling_main"]["overall"]
        delta = (
            candidate_causal["matched_advantage"]
            - control_causal["matched_advantage"]
        )
        passed = (
            candidate_causal["matched_advantage"] >= 0.025
            and candidate_causal["matched_advantage_ci80"][0] >= -0.025
            and delta >= 0.02
        )
        causal_windows.append(
            {
                "epoch": epoch,
                "candidate_matched_advantage": candidate_causal["matched_advantage"],
                "control_matched_advantage": control_causal["matched_advantage"],
                "candidate_minus_control": delta,
                "passed": passed,
            }
        )
    sustained_causal_improvement = (
        sum(bool(item["passed"]) for item in causal_windows) >= 2
        and bool(causal_windows[-1]["passed"])
    )
    efficacy_pass = strength_improvement or sustained_causal_improvement
    integrity_pass = signal_integrity and schedule_integrity and determinism_integrity
    advance = all(
        (
            integrity_pass,
            throughput_pass,
            strength_nonregression,
            behavior_guards,
            efficacy_pass,
        )
    )
    if advance:
        status = "advance"
    elif not integrity_pass:
        status = "stop_integrity"
    elif not throughput_pass:
        status = "stop_throughput"
    elif not strength_nonregression:
        status = "stop_strength_regression"
    elif not behavior_guards:
        status = "stop_behavior_guard"
    else:
        status = "neutral_no_causal_or_strength_gain"
    return {
        "status": status,
        "advance": advance,
        "integrity_pass": integrity_pass,
        "signal_integrity_pass": signal_integrity,
        "schedule_integrity_pass": schedule_integrity,
        "determinism_integrity_pass": determinism_integrity,
        "throughput_pass": throughput_pass,
        "throughput_ratio": throughput_ratio,
        "strength_nonregression_pass": strength_nonregression,
        "behavior_guards_pass": behavior_guards,
        "water_holdout_score_delta": water_delta,
        "portal_rate_ratio": portal_ratio,
        "strength_improvement_pass": strength_improvement,
        "sustained_causal_improvement_pass": sustained_causal_improvement,
        "causal_windows": causal_windows,
        "efficacy_pass": efficacy_pass,
    }


def _fmt(value: object, digits: int = 3) -> str:
    return "n/a" if not _finite(value) else f"{float(value):.{digits}f}"


def _markdown(report: dict) -> str:
    screen = report["screen"]
    training = report["training"]
    lines = [
        "# Whole-Draft Terminal-Credit 15M Ladder",
        "",
        "The control and candidate resume the exact accepted p4870 model, optimizer, "
        "episode progression, league state, reward stack, and schedule. The candidate "
        "adds only unannealed delayed true-terminal actor credit for the leader and "
        "one sampled main pick per quartile. Promotion is not an evaluation signal.",
        "",
        "## Outcome",
        "",
        f"Decision: **{screen['status']}**.",
        "",
        "| Arm | SPS | Parent final | Holdout final | Unique | Quads | Spell share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    final = report["trajectory"]["5847"]
    for arm in ARMS:
        item = final[arm]
        lines.append(
            f"| {arm} | {training[arm]['steady_sps_median']:.0f} | "
            f"{item['h2h_vs_parent']['score']:.3f} | "
            f"{item['holdout_reference']['score']:.3f} | "
            f"{item['decks']['main_unique_mean']:.2f} | "
            f"{item['decks']['quad_slot_share_mean']:.3f} | "
            f"{item['decks']['type_share'].get('SPELL', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory",
            "",
            "| Epoch | Direct score | Parent delta | Holdout delta | "
            "Sibling main KL C->T | Paired deck Jaccard | Matched-main causal delta |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for epoch in EPOCHS:
        item = report["trajectory"][str(epoch)]
        candidate = item["draft_terminal_long"]
        control = item["control"]
        causal = next(
            row for row in screen["causal_windows"] if row["epoch"] == epoch
        )
        lines.append(
            f"| {epoch} | {candidate['h2h_vs_control']['score']:.3f} | "
            f"{item['paired']['h2h_vs_parent']['mean']:+.3f} | "
            f"{item['paired']['holdout_reference']['mean']:+.3f} | "
            f"{control['conditioning']['sibling_gate_main_pick_kl']:.4f}->"
            f"{candidate['conditioning']['sibling_gate_main_pick_kl']:.4f} | "
            f"{item['paired_decks']['main_multiset_jaccard_mean']:.3f} | "
            f"{causal['candidate_minus_control']:+.3f} |"
        )
    candidate_training = training["draft_terminal_long"]
    lines.extend(
        [
            "",
            "## Registered Screen",
            "",
            f"- Integrity: `{screen['integrity_pass']}`; throughput: "
            f"`{screen['throughput_ratio']:.3f}` of control; strength nonregression: "
            f"`{screen['strength_nonregression_pass']}`.",
            f"- Final direct candidate-control score: "
            f"`{final['draft_terminal_long']['h2h_vs_control']['score']:.3f}`; "
            f"paired parent delta: `{final['paired']['h2h_vs_parent']['mean']:+.3f}`; "
            f"paired holdout delta: `{final['paired']['holdout_reference']['mean']:+.3f}`.",
            f"- Strength improvement: `{screen['strength_improvement_pass']}`; "
            f"sustained causal drafting improvement: "
            f"`{screen['sustained_causal_improvement_pass']}`. KL alone is not a pass.",
            f"- Candidate credit examples: `{candidate_training['trained_examples']:.0f}`; "
            f"clip fraction max: `{_fmt(candidate_training['clipfrac_max'], 4)}`; "
            f"auxiliary GPU time: `{candidate_training['aux_gpu_seconds']:.1f}s`.",
            f"- Heldout Water delta: `{screen['water_holdout_score_delta']:+.3f}`; "
            f"portal-rate ratio: `{screen['portal_rate_ratio']:.3f}`.",
            "",
            "Detailed per-gate/per-leader sampled compositions, position-specific "
            "credit calibration, action ratios, conditioning KL, and hybrid outcomes "
            "are retained in `ladder_report.json` and the trajectory subdirectories.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(root: Path) -> dict:
    training = {
        arm: _load_json(root / arm / "training_summary.json") for arm in ARMS
    }
    parent, _ = _load_window(root, "parent_p4870")
    trajectory = {}
    for epoch in EPOCHS:
        control, control_raw = _load_window(root, f"control_p{epoch}")
        candidate, candidate_raw = _load_window(
            root, f"draft_terminal_long_p{epoch}"
        )
        paired = {
            "h2h_vs_parent": _paired_delta(
                candidate_raw["h2h_parent"],
                control_raw["h2h_parent"],
                seed=42_908_000 + epoch,
                group_blocks=True,
            ),
            "holdout_reference": _paired_delta(
                candidate_raw["holdout"],
                control_raw["holdout"],
                seed=42_909_000 + epoch,
            ),
        }
        trajectory[str(epoch)] = {
            "control": control,
            "draft_terminal_long": candidate,
            "paired": paired,
            "paired_decks": _paired_deck_summary(
                control_raw["decks"], candidate_raw["decks"]
            ),
        }
    report = {
        "schema_version": 1,
        "campaign": root.name,
        "parent_epoch": 4870,
        "target_epoch": 5847,
        "promotion_used_as_signal": False,
        "training": training,
        "parent": parent,
        "trajectory": trajectory,
    }
    report["screen"] = _build_screen(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["screen"], sort_keys=True))


if __name__ == "__main__":
    main()
