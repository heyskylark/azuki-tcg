from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strategy_recovery_ladder_report import (
    _build_arm,
    _load_json,
    _load_metric_rows,
    _paired_delta,
    _weighted_gate_score,
)


ARMS = ("control", "leader_credit")
FIRE_GATES = ("AZK01-122", "STT04-002")
FIRE_KAGORO = "AZK01-121"
FIRE_ZERO = "STT04-001"
TARGET_EPOCH = 4870


def _numeric(rows: list[dict], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]


def _credit_training_summary(rows: list[dict]) -> dict:
    examples = _numeric(rows, "losses/leader_credit_examples")
    labeled = _numeric(rows, "losses/win_prob_aux_labeled_rows")
    captures = _numeric(rows, "environment/leader_credit/captured")
    importance = []
    clipfrac = []
    baseline_losses = []
    train_seconds = []
    for row in rows:
        count = row.get("losses/leader_credit_examples")
        if not isinstance(count, (int, float)) or float(count) <= 0.0:
            continue
        for key, target in (
            ("losses/leader_credit_importance_mean", importance),
            ("losses/leader_credit_clipfrac", clipfrac),
            ("losses/leader_credit_baseline_loss", baseline_losses),
            ("losses/leader_credit_train_seconds", train_seconds),
        ):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target.append(float(value))
    return {
        "max_labeled_rows": max(labeled, default=0.0),
        "labeled_epochs": sum(value > 0.0 for value in labeled),
        "captured_records": float(sum(captures)),
        "trained_examples": float(sum(examples)),
        "training_epochs": sum(value > 0.0 for value in examples),
        "importance_mean_median": float(np.median(importance)) if importance else None,
        "clipfrac_max": max(clipfrac, default=None),
        "baseline_loss_median": (
            float(np.median(baseline_losses)) if baseline_losses else None
        ),
        "train_seconds_total": float(sum(train_seconds)),
        "train_seconds_median": (
            float(np.median(train_seconds)) if train_seconds else None
        ),
    }


def _fire_leader_summary(decks: dict) -> dict:
    kagoro = 0
    zero = 0
    other = 0
    argmax_by_gate = {}
    for gate_code in FIRE_GATES:
        gate = decks["gates"][gate_code]
        leader_split = gate["sampled"]["leader_split"]
        kagoro += int(leader_split.get(FIRE_KAGORO, 0))
        zero += int(leader_split.get(FIRE_ZERO, 0))
        other += sum(
            int(count)
            for code, count in leader_split.items()
            if code not in {FIRE_KAGORO, FIRE_ZERO}
        )
        argmax_by_gate[gate_code] = str(gate["argmax_deck"]["leader"]["code"])
    total = kagoro + zero + other
    return {
        "sampled_total": total,
        "sampled_kagoro": kagoro,
        "sampled_zero": zero,
        "sampled_other": other,
        "sampled_kagoro_share": kagoro / total if total else None,
        "argmax_by_gate": argmax_by_gate,
        "argmax_kagoro_gates": sum(
            leader == FIRE_KAGORO for leader in argmax_by_gate.values()
        ),
    }


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _screen(candidate: dict, control: dict, paired: dict) -> dict:
    direct = candidate["h2h_vs_control"]
    throughput_ratio = (
        candidate["training"]["sps_median"] / control["training"]["sps_median"]
    )
    portal_ratio = candidate["opportunity"]["portals_per_game"] / max(
        control["opportunity"]["portals_per_game"], 1e-9
    )
    water_delta = (
        candidate["holdout_reference"]["water_score"]
        - control["holdout_reference"]["water_score"]
    )
    sampled_kagoro_delta = _delta(
        candidate["fire_leaders"]["sampled_kagoro_share"],
        control["fire_leaders"]["sampled_kagoro_share"],
    )
    argmax_kagoro_delta = (
        candidate["fire_leaders"]["argmax_kagoro_gates"]
        - control["fire_leaders"]["argmax_kagoro_gates"]
    )
    distribution_shift = (
        sampled_kagoro_delta is not None
        and sampled_kagoro_delta >= 0.10
    ) or argmax_kagoro_delta >= 1
    credit = candidate["credit_training"]
    control_credit = control["credit_training"]
    label_integrity = (
        control_credit["max_labeled_rows"] > 0
        and credit["max_labeled_rows"] > 0
        and credit["trained_examples"] > 0
        and credit["captured_records"] > 0
        and credit["clipfrac_max"] is not None
        and credit["clipfrac_max"] <= 0.25
    )
    integrity = all(
        arm["training"]["epoch_final"] == TARGET_EPOCH
        and arm["training"]["learning_rate_final"] <= 1e-8
        and arm["training"]["reward_shaping_scale_final"] is not None
        and abs(arm["training"]["reward_shaping_scale_final"] - 0.15) <= 1e-3
        for arm in (control, candidate)
    )
    strength = (
        direct["score"] >= 0.47
        and direct["paired_lcb80"] >= 0.43
        and paired["h2h_vs_parent"]["lcb80"] >= -0.05
    )
    fire_nonloss = candidate["fire_h2h_vs_control"] >= 0.50
    behavior_guards = portal_ratio >= 0.90 and water_delta >= -0.05
    throughput = throughput_ratio >= 0.95
    advance = all(
        (
            distribution_shift,
            label_integrity,
            integrity,
            strength,
            fire_nonloss,
            behavior_guards,
            throughput,
        )
    )
    if advance:
        status = "advance"
    elif not label_integrity or not integrity:
        status = "stop_integrity"
    elif not throughput:
        status = "stop_throughput"
    elif not strength or not fire_nonloss:
        status = "stop_strength"
    elif not behavior_guards:
        status = "stop_behavior_guard"
    else:
        status = "neutral_no_leader_recovery"
    return {
        "status": status,
        "advance": advance,
        "distribution_shift_pass": distribution_shift,
        "label_integrity_pass": label_integrity,
        "schedule_integrity_pass": integrity,
        "strength_pass": strength,
        "fire_nonloss_pass": fire_nonloss,
        "behavior_guards_pass": behavior_guards,
        "throughput_pass": throughput,
        "throughput_ratio": throughput_ratio,
        "portal_ratio": portal_ratio,
        "holdout_water_score_delta": water_delta,
        "sampled_kagoro_share_delta": sampled_kagoro_delta,
        "argmax_kagoro_gate_delta": argmax_kagoro_delta,
        "fire_h2h_vs_control": candidate["fire_h2h_vs_control"],
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(report: dict) -> str:
    control = report["arms"]["control"]
    candidate = report["arms"]["leader_credit"]
    screen = candidate["screen"]
    lines = [
        "# Fire Leader Terminal-Credit 15M Ladder",
        "",
        "Both arms resume the exact p3900 tempo-dedup model, optimizer, episode "
        "progression, league state, and reward stack. The only training-arm "
        "difference is Fire leader-decision terminal actor credit; promotion is "
        "not a signal.",
        "",
        "## Outcome",
        "",
        f"Decision: **{screen['status']}**.",
        "",
        "| Arm | SPS | H2H parent | H2H control | Holdout ref | Holdout Water | Kagoro sampled | Kagoro argmax gates | Labels max | Credit examples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        arm = report["arms"][name]
        lines.append(
            f"| {name} | {arm['training']['sps_median']:.0f} | "
            f"{arm['h2h_vs_parent']['score']:.3f} | "
            f"{_fmt(arm.get('h2h_vs_control', {}).get('score'))} | "
            f"{arm['holdout_reference']['score']:.3f} | "
            f"{arm['holdout_reference']['water_score']:.3f} | "
            f"{_fmt(arm['fire_leaders']['sampled_kagoro_share'])} | "
            f"{arm['fire_leaders']['argmax_kagoro_gates']}/2 | "
            f"{arm['credit_training']['max_labeled_rows']:.0f} | "
            f"{arm['credit_training']['trained_examples']:.0f} |"
        )
    paired = report["paired_deltas"]
    lines.extend(
        [
            "",
            "## Registered Screen",
            "",
            f"- Direct candidate-control score: `{candidate['h2h_vs_control']['score']:.3f}`; "
            f"paired lower 80% bound `{candidate['h2h_vs_control']['paired_lcb80']:.3f}`.",
            f"- Paired parent-panel delta: `{paired['h2h_vs_parent']['mean']:+.3f}` "
            f"(80% `{paired['h2h_vs_parent']['lcb80']:+.3f}` to "
            f"`{paired['h2h_vs_parent']['ucb80']:+.3f}`).",
            f"- Fire score against control: `{candidate['fire_h2h_vs_control']:.3f}`.",
            f"- Kagoro sampled-share delta: `{_fmt(screen['sampled_kagoro_share_delta'])}`; "
            f"argmax-gate delta: `{screen['argmax_kagoro_gate_delta']:+d}`.",
            f"- SPS ratio: `{screen['throughput_ratio']:.3f}`; portal ratio: "
            f"`{screen['portal_ratio']:.3f}`; heldout Water delta: "
            f"`{screen['holdout_water_score_delta']:+.3f}`.",
            f"- Delayed-credit importance median: "
            f"`{_fmt(candidate['credit_training']['importance_mean_median'], 4)}`; "
            f"maximum clip fraction: `{_fmt(candidate['credit_training']['clipfrac_max'], 4)}`; "
            f"auxiliary GPU time: `{candidate['credit_training']['train_seconds_total']:.1f}s`.",
            "",
            "Advance requires all registered strength, Fire nonloss, leader-shift, "
            "Water/portal, label-integrity, schedule-integrity, and 95% SPS gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(root: Path) -> dict:
    arms = {}
    raw = {}
    for name in ARMS:
        arm, arm_raw = _build_arm(root, name)
        rows = _load_metric_rows(root / name / "train.jsonl")
        arm["credit_training"] = _credit_training_summary(rows)
        arm["fire_leaders"] = _fire_leader_summary(
            _load_json(root / name / "decks.json")
        )
        arm["fire_h2h_vs_parent"] = _weighted_gate_score(
            arm_raw["h2h_parent"], FIRE_GATES
        )
        if name != "control":
            arm["fire_h2h_vs_control"] = _weighted_gate_score(
                arm_raw["h2h_control"], FIRE_GATES
            )
        arms[name] = arm
        raw[name] = arm_raw
    paired = {
        "h2h_vs_parent": _paired_delta(
            raw["leader_credit"]["h2h_parent"],
            raw["control"]["h2h_parent"],
            seed=42_905_101,
            group_blocks=True,
        ),
        "holdout_reference": _paired_delta(
            raw["leader_credit"]["holdout_ref"],
            raw["control"]["holdout_ref"],
            seed=42_905_102,
        ),
    }
    arms["leader_credit"]["screen"] = _screen(
        arms["leader_credit"], arms["control"], paired
    )
    return {
        "campaign": root.name,
        "parent_epoch": 3900,
        "target_epoch": TARGET_EPOCH,
        "arms": arms,
        "paired_deltas": paired,
        "advance_candidates": (
            ["leader_credit"]
            if arms["leader_credit"]["screen"]["advance"]
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["arms"]["leader_credit"]["screen"], sort_keys=True))


if __name__ == "__main__":
    main()
