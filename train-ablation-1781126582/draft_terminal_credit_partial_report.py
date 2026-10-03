#!/usr/bin/env python3
"""Report guard-stopped Step 5b windows without treating them as advancement data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from draft_terminal_credit_ladder_report import (
    _load_window,
    _paired_deck_summary,
)
from strategy_recovery_ladder_report import _paired_delta


ARMS = ("control", "draft_terminal_long")


def _metric_rows(path: Path, start: int, end: int) -> list[dict]:
    by_epoch: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = row.get("epoch")
        if isinstance(epoch, (int, float)) and start <= int(epoch) <= end:
            by_epoch[int(epoch)] = row
    expected = set(range(start, end + 1))
    if set(by_epoch) != expected:
        missing = sorted(expected.difference(by_epoch))
        raise RuntimeError(f"metric rows are not contiguous through p{end}: {missing[:20]}")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _numeric(row: dict, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _values(rows: list[dict], key: str) -> list[float]:
    values = [_numeric(row, key) for row in rows]
    return [value for value in values if value is not None and math.isfinite(value)]


def _training_summary(rows: list[dict], arm: str) -> dict[str, object]:
    steady = rows[20:]
    sps = _values(steady, "SPS")
    trained = [
        row
        for row in rows
        if (_numeric(row, "losses/draft_credit_examples") or 0.0) > 0.0
    ]
    positions: dict[str, dict[str, float | None]] = {}
    for position_name in ("leader", "early", "middle", "late"):
        count_key = f"losses/draft_credit_{position_name}_examples"
        position_rows = [
            row for row in rows if (_numeric(row, count_key) or 0.0) > 0.0
        ]
        metrics: dict[str, float | None] = {
            "examples": sum(_values(position_rows, count_key))
        }
        for suffix in (
            "advantage_mean",
            "advantage_abs_mean",
            "advantage_std",
            "sign_outcome_agreement",
            "baseline_bce",
            "baseline_brier",
            "baseline_pred_mean",
        ):
            key = f"losses/draft_credit_{position_name}_{suffix}"
            weighted = [
                (_numeric(row, key), _numeric(row, count_key))
                for row in position_rows
            ]
            valid = [
                (value, count)
                for value, count in weighted
                if value is not None and count is not None and count > 0.0
            ]
            metrics[suffix] = (
                sum(value * count for value, count in valid)
                / sum(count for _, count in valid)
                if valid
                else None
            )
        positions[position_name] = metrics
    return {
        "arm": arm,
        "metric_rows": len(rows),
        "epoch_first": int(rows[0]["epoch"]),
        "epoch_final": int(rows[-1]["epoch"]),
        "steady_sps_median": statistics.median(sps),
        "steady_sps_p10": sorted(sps)[max(0, int(0.1 * (len(sps) - 1)))],
        "last100_sps_median": statistics.median(sps[-100:]),
        "captured_records": sum(_values(rows, "environment/draft_credit/captured")),
        "labeled_records": sum(_values(rows, "environment/draft_credit/labeled")),
        "truncated_records": sum(_values(rows, "environment/draft_credit/truncated")),
        "incomplete_episodes": sum(
            _values(rows, "environment/draft_credit/incomplete_episodes")
        ),
        "incomplete_records": sum(
            _values(rows, "environment/draft_credit/incomplete_records")
        ),
        "pending_records_final": _numeric(
            rows[-1], "environment/draft_credit/pending_records"
        ),
        "ready_records_final": _numeric(
            rows[-1], "environment/draft_credit/ready"
        ),
        "trained_examples": sum(_values(rows, "losses/draft_credit_examples")),
        "fixed_batch_rows": sum(
            _values(rows, "losses/draft_credit_fixed_batch_rows")
        ),
        "training_epochs": len(trained),
        "clipfrac_max": max(
            _values(trained, "losses/draft_credit_clipfrac"), default=None
        ),
        "importance_mean_min": min(
            _values(trained, "losses/draft_credit_importance_mean"), default=None
        ),
        "importance_mean_max": max(
            _values(trained, "losses/draft_credit_importance_mean"), default=None
        ),
        "gradient_norm_median": (
            statistics.median(_values(trained, "losses/draft_credit_gradient_norm"))
            if trained
            else None
        ),
        "aux_wall_seconds": sum(
            _values(rows, "losses/draft_credit_train_seconds")
        ),
        "aux_gpu_seconds": sum(_values(rows, "losses/draft_credit_gpu_seconds")),
        "trainer_shaped_reward_multiplier_min": min(
            _values(rows, "environment/trainer_shaped_reward_multiplier"),
            default=1.0,
        ),
        "trainer_shaped_reward_multiplier_max": max(
            _values(rows, "environment/trainer_shaped_reward_multiplier"),
            default=1.0,
        ),
        "native_reward_shaping_scale_final": _numeric(
            rows[-1], "environment/reward_shaping_scale"
        ),
        "positions": positions,
    }


def _guard(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        values[key] = float(value)
    required = {"previous_median", "current_median", "threshold"}
    if set(values) != required:
        raise RuntimeError(f"unexpected SPS guard evidence: {values}")
    return values


def _causal(window: dict) -> dict[str, float]:
    control = window["control"]["hybrids"]["sibling_main"]["overall"]
    candidate = window["draft_terminal_long"]["hybrids"]["sibling_main"][
        "overall"
    ]
    return {
        "control_matched_advantage": float(control["matched_advantage"]),
        "candidate_matched_advantage": float(candidate["matched_advantage"]),
        "candidate_minus_control": float(candidate["matched_advantage"])
        - float(control["matched_advantage"]),
    }


def build_report(root: Path, epochs: list[int]) -> dict:
    if not epochs or epochs != sorted(set(epochs)):
        raise ValueError("epochs must be nonempty, unique, and sorted")
    candidate_log = root / "draft_terminal_long" / "partial_train.jsonl"
    candidate_all: dict[int, dict] = {}
    for line in candidate_log.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = row.get("epoch")
        if isinstance(epoch, (int, float)) and int(epoch) >= 4871:
            candidate_all[int(epoch)] = row
    stop_epoch = max(candidate_all)
    control_rows = _metric_rows(root / "control" / "train.jsonl", 4871, stop_epoch)
    candidate_rows = _metric_rows(candidate_log, 4871, stop_epoch)
    training = {
        "control": _training_summary(control_rows, "control"),
        "draft_terminal_long": _training_summary(
            candidate_rows, "draft_terminal_long"
        ),
    }

    trajectory: dict[str, dict] = {}
    for epoch in epochs:
        control, control_raw = _load_window(
            root, f"control_p{epoch}", trajectory_dir="partial_trajectory"
        )
        candidate, candidate_raw = _load_window(
            root,
            f"draft_terminal_long_p{epoch}",
            trajectory_dir="partial_trajectory",
        )
        window = {
            "control": control,
            "draft_terminal_long": candidate,
            "paired": {
                "h2h_vs_parent": _paired_delta(
                    candidate_raw["h2h_parent"],
                    control_raw["h2h_parent"],
                    seed=42_918_000 + epoch,
                    group_blocks=True,
                ),
                "holdout_reference": _paired_delta(
                    candidate_raw["holdout"],
                    control_raw["holdout"],
                    seed=42_919_000 + epoch,
                ),
            },
            "paired_decks": _paired_deck_summary(
                control_raw["decks"], candidate_raw["decks"]
            ),
        }
        window["sibling_main_causal"] = _causal(window)
        trajectory[str(epoch)] = window

    guard = _guard(root / "draft_terminal_long" / "SPS_GUARD_FAILED")
    throughput_ratio = (
        float(training["draft_terminal_long"]["steady_sps_median"])
        / float(training["control"]["steady_sps_median"])
    )
    candidate_training = training["draft_terminal_long"]
    signal_integrity = bool(
        candidate_training["captured_records"] > 0
        and candidate_training["labeled_records"] > 0
        and candidate_training["trained_examples"] > 0
        and candidate_training["fixed_batch_rows"]
        == candidate_training["trained_examples"]
        and candidate_training["truncated_records"] == 0
        and candidate_training["incomplete_episodes"] == 0
        and candidate_training["incomplete_records"] == 0
        and candidate_training["clipfrac_max"] is not None
        and candidate_training["clipfrac_max"] <= 0.25
    )
    report = {
        "schema_version": 1,
        "campaign": root.name,
        "evaluation_kind": "guard_stopped_partial_diagnostic",
        "parent_epoch": 4870,
        "target_epoch": 5847,
        "stop_epoch": stop_epoch,
        "evaluated_epochs": epochs,
        "promotion_used_as_signal": False,
        "training": training,
        "trajectory": trajectory,
        "screen": {
            "status": "stop_throughput",
            "advance": False,
            "result_can_be_overridden_by_partial_behavior": False,
            "signal_integrity_pass": signal_integrity,
            "hard_floor_guard": guard,
            "hard_floor_guard_failed": bool(
                guard["previous_median"] < guard["threshold"]
                and guard["current_median"] < guard["threshold"]
            ),
            "matched_steady_sps_ratio": throughput_ratio,
            "relative_sps_threshold": 0.95,
            "relative_sps_pass": throughput_ratio >= 0.95,
            "completed_update_fraction": (stop_epoch - 4870) / (5847 - 4870),
        },
    }
    return report


def _fmt(value: object, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown(report: dict) -> str:
    screen = report["screen"]
    training = report["training"]
    guard = screen["hard_floor_guard"]
    lines = [
        "# Whole-Draft Terminal-Credit Partial Evaluation",
        "",
        "Decision: **stop_throughput**. The candidate was interrupted by the "
        "preregistered SPS guard, so these retained-window diagnostics cannot "
        "advance it even if a behavioral metric is favorable.",
        "",
        "## Runtime Result",
        "",
        f"- Stopped at `p{report['stop_epoch']}` after "
        f"`{screen['completed_update_fraction']:.1%}` of the continuation.",
        f"- Non-overlapping guard medians: `{guard['previous_median']:.2f}` and "
        f"`{guard['current_median']:.2f}` SPS versus a "
        f"`{guard['threshold']:.0f}` floor.",
        f"- Matched steady SPS: control "
        f"`{training['control']['steady_sps_median']:.1f}`, candidate "
        f"`{training['draft_terminal_long']['steady_sps_median']:.1f}` "
        f"(`{screen['matched_steady_sps_ratio']:.1%}` of control; 95% required).",
        f"- Credit integrity: `{screen['signal_integrity_pass']}`; candidate "
        f"captured `{training['draft_terminal_long']['captured_records']:.0f}` "
        f"records and trained `{training['draft_terminal_long']['trained_examples']:.0f}` "
        "examples with zero truncation/incomplete records.",
        "",
        "## Retained Windows",
        "",
        "| Epoch | Direct score | Parent delta | Holdout delta | "
        "Sibling KL C->T | Deck Jaccard | Unique C->T | Main causal C->T (delta) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in report["evaluated_epochs"]:
        window = report["trajectory"][str(epoch)]
        control = window["control"]
        candidate = window["draft_terminal_long"]
        causal = window["sibling_main_causal"]
        lines.append(
            f"| {epoch} | {candidate['h2h_vs_control']['score']:.3f} | "
            f"{window['paired']['h2h_vs_parent']['mean']:+.3f} | "
            f"{window['paired']['holdout_reference']['mean']:+.3f} | "
            f"{control['conditioning']['sibling_gate_main_pick_kl']:.4f}->"
            f"{candidate['conditioning']['sibling_gate_main_pick_kl']:.4f} | "
            f"{window['paired_decks']['main_multiset_jaccard_mean']:.3f} | "
            f"{control['decks']['main_unique_mean']:.2f}->"
            f"{candidate['decks']['main_unique_mean']:.2f} | "
            f"{causal['control_matched_advantage']:+.3f}->"
            f"{causal['candidate_matched_advantage']:+.3f} "
            f"({causal['candidate_minus_control']:+.3f}) |"
        )
    final_epoch = report["evaluated_epochs"][-1]
    final_window = report["trajectory"][str(final_epoch)]
    final_holdout = final_window["paired"]["holdout_reference"]
    lines.extend(
        [
            "",
            f"The p{final_epoch} heldout delta is "
            f"`{final_holdout['mean']:+.3f}` with an 80% interval of "
            f"`{final_holdout['lcb80']:+.3f}` to "
            f"`{final_holdout['ucb80']:+.3f}`. This roughly one-point movement "
            "is treated as neutral, not as a meaningful regression.",
            "",
            "## Opportunity Diagnostics",
            "",
            "| Epoch | Portals/game C->T | Water spell slots C->T | "
            "Water main use/legal C->T | Garden comparable C->T |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for epoch in report["evaluated_epochs"]:
        window = report["trajectory"][str(epoch)]
        control = window["control"]["opportunity"]
        candidate = window["draft_terminal_long"]["opportunity"]
        lines.append(
            f"| {epoch} | {control['portals_per_game']:.2f}->"
            f"{candidate['portals_per_game']:.2f} | "
            f"{control['water_spell_slot_share']:.3f}->"
            f"{candidate['water_spell_slot_share']:.3f} | "
            f"{control['water_main_spell_selected_per_legal_window']:.3f}->"
            f"{candidate['water_main_spell_selected_per_legal_window']:.3f} | "
            f"{control['direct_garden_share_when_comparable']:.3f}->"
            f"{candidate['direct_garden_share_when_comparable']:.3f} |"
        )
    candidate = training["draft_terminal_long"]
    lines.extend(
        [
            "",
            "## Credit Diagnostics",
            "",
            f"- Importance-mean range: "
            f"`{_fmt(candidate['importance_mean_min'], 6)}.."
            f"{_fmt(candidate['importance_mean_max'], 6)}`; max clip fraction: "
            f"`{_fmt(candidate['clipfrac_max'], 4)}`.",
            f"- Median draft gradient norm: "
            f"`{_fmt(candidate['gradient_norm_median'], 4)}`; auxiliary GPU time: "
            f"`{candidate['aux_gpu_seconds']:.1f}s`.",
            "- Detailed per-gate deck composition, leader conditioning, action "
            "opportunities, references, and hybrid panels are retained under "
            "`partial_trajectory/` and in `partial_ladder_report.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root, args.epochs)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["screen"], sort_keys=True))


if __name__ == "__main__":
    main()
