from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strategy_recovery_ladder_report import (
    _deck_summary,
    _gate_kl_summary,
    _h2h_summary,
    _load_json,
    _opportunity_summary,
    _paired_delta,
    _weighted_gate_score,
)


ARMS = ("fixed_floor", "zero_tail")
EPOCHS = (6000, 6900, 7400, 7800)
WATER_GATES = ("STT02-002", "AZK01-126")
GATE_ORDER = (
    "AZK01-120",
    "STT01-002",
    "AZK01-126",
    "STT02-002",
    "AZK01-122",
    "STT04-002",
    "AZK01-124",
    "STT03-002",
)


def _evaluation_summary(payload: dict) -> dict:
    summary = payload["summary"]
    return {
        "episodes": int(summary["episodes"]),
        "score": float(summary["score"]),
        "timeout_rate": float(summary["timeout_rate"]),
        "water_score": _weighted_gate_score(payload, WATER_GATES),
        "deck_metrics": dict(summary.get("deck_metrics", {})),
        "battle_metrics": dict(summary.get("battle_metrics", {})),
    }


def _point(root: Path, arm: str, epoch: int) -> tuple[dict, dict]:
    point_root = root / "eval" / f"p{epoch}" / arm
    parent = _load_json(point_root / "h2h_vs_parent.json")
    holdout = _load_json(point_root / "draftref_holdout.json")
    decks = _load_json(point_root / "decks.json")
    opportunity = _load_json(point_root / "opportunity" / "opportunity.json")
    deterministic = holdout["summary"]["deck_metrics"]
    point = {
        "h2h_vs_parent": _h2h_summary(parent),
        "holdout_reference": _evaluation_summary(holdout),
        "decks": _deck_summary(decks, deterministic),
        "sibling_gate_kl": _gate_kl_summary(_load_json(point_root / "gate_kl.json")),
        "opportunity": _opportunity_summary(opportunity),
    }
    if epoch == 7800:
        point["train_reference"] = _evaluation_summary(
            _load_json(point_root / "draftref_train.json")
        )
    return point, {"parent": parent, "holdout": holdout, "decks": decks}


def _difference(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control is None:
        return None
    return float(candidate) - float(control)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _screen(report: dict) -> dict:
    endpoint = report["trajectory"]["7800"]
    floor = endpoint["arms"]["fixed_floor"]
    tail = endpoint["arms"]["zero_tail"]
    direct = endpoint["tail_vs_floor"]
    paired = endpoint["paired_deltas"]

    water_main_delta = _difference(
        tail["opportunity"]["water_main_spell_selected_per_legal_window"],
        floor["opportunity"]["water_main_spell_selected_per_legal_window"],
    )
    water_response_delta = _difference(
        tail["opportunity"]["water_response_spell_selected_per_legal_window"],
        floor["opportunity"]["water_response_spell_selected_per_legal_window"],
    )
    garden_delta = _difference(
        tail["opportunity"]["direct_garden_share_when_comparable"],
        floor["opportunity"]["direct_garden_share_when_comparable"],
    )
    portal_ratio = _safe_ratio(
        tail["opportunity"]["portals_per_game"],
        floor["opportunity"]["portals_per_game"],
    )
    water_holdout_delta = (
        tail["holdout_reference"]["water_score"]
        - floor["holdout_reference"]["water_score"]
    )
    unique_delta = (
        tail["decks"]["deterministic_main_unique_mean"]
        - floor["decks"]["deterministic_main_unique_mean"]
    )
    quad_delta = (
        tail["decks"]["deterministic_quad_slot_share_mean"]
        - floor["decks"]["deterministic_quad_slot_share_mean"]
    )

    late_direct_scores = [
        report["trajectory"][str(epoch)]["tail_vs_floor"]["score"]
        for epoch in (6900, 7400, 7800)
    ]
    clear_strength = (
        direct["score"] >= 0.52
        and direct["paired_lcb80"] >= 0.48
        and float(np.mean(late_direct_scores)) >= 0.51
    )
    strength_safety = (
        paired["h2h_vs_parent"]["lcb80"] >= -0.03
        and paired["holdout_reference"]["lcb80"] >= -0.03
        and min(late_direct_scores) >= 0.47
    )
    mechanics_safety = (
        water_main_delta is not None
        and water_main_delta >= -0.03
        and water_response_delta is not None
        and water_response_delta >= -0.05
        and portal_ratio is not None
        and portal_ratio >= 0.90
        and water_holdout_delta >= -0.05
        and unique_delta >= -2.0
        and quad_delta <= 0.05
    )
    throughput = report["training_report"]["status"] == "pass"
    integrity = (
        report["training_report"]["zero_tail_checkpoint_multipliers"]
        == {"6000": 1.0, "6900": 0.0, "7400": 0.0, "7800": 0.0}
        and report["training_report"]["zero_tail_terminal_labels_max"] > 0.0
    )

    if not integrity:
        status = "reject_integrity"
    elif not throughput:
        status = "reject_throughput"
    elif not strength_safety:
        status = "reject_strength"
    elif not mechanics_safety:
        status = "reject_mechanics"
    elif clear_strength:
        status = "adopt_zero_tail"
    else:
        status = "neutral_keep_floor"
    return {
        "status": status,
        "adopt_zero_tail": status == "adopt_zero_tail",
        "clear_strength_pass": clear_strength,
        "strength_safety_pass": strength_safety,
        "mechanics_safety_pass": mechanics_safety,
        "throughput_pass": throughput,
        "integrity_pass": integrity,
        "late_direct_scores": late_direct_scores,
        "late_direct_mean": float(np.mean(late_direct_scores)),
        "endpoint_direct_score": direct["score"],
        "endpoint_direct_lcb80": direct["paired_lcb80"],
        "endpoint_parent_delta": paired["h2h_vs_parent"],
        "endpoint_holdout_delta": paired["holdout_reference"],
        "water_main_selection_delta": water_main_delta,
        "water_response_selection_delta": water_response_delta,
        "direct_garden_selection_delta": garden_delta,
        "portal_ratio": portal_ratio,
        "holdout_water_delta": water_holdout_delta,
        "deterministic_unique_delta": unique_delta,
        "deterministic_quad_slot_delta": quad_delta,
    }


def build_report(root: Path) -> dict:
    trajectory = {}
    raw = {}
    for epoch in EPOCHS:
        epoch_key = str(epoch)
        arms = {}
        raw[epoch_key] = {}
        for arm in ARMS:
            arms[arm], raw[epoch_key][arm] = _point(root, arm, epoch)
        direct_payload = _load_json(root / "eval" / f"p{epoch}" / "h2h_tail_vs_floor.json")
        trajectory[epoch_key] = {
            "arms": arms,
            "tail_vs_floor": _h2h_summary(direct_payload),
            "paired_deltas": {
                "h2h_vs_parent": _paired_delta(
                    raw[epoch_key]["zero_tail"]["parent"],
                    raw[epoch_key]["fixed_floor"]["parent"],
                    seed=43_905_100 + epoch,
                    group_blocks=True,
                ),
                "holdout_reference": _paired_delta(
                    raw[epoch_key]["zero_tail"]["holdout"],
                    raw[epoch_key]["fixed_floor"]["holdout"],
                    seed=43_906_100 + epoch,
                ),
            },
        }

    for arm in ARMS:
        base_parent = raw["6000"][arm]["parent"]
        base_holdout = raw["6000"][arm]["holdout"]
        for epoch in EPOCHS:
            trajectory[str(epoch)]["arms"][arm]["delta_vs_p6000"] = {
                "h2h_vs_parent": _paired_delta(
                    raw[str(epoch)][arm]["parent"],
                    base_parent,
                    seed=43_907_100 + epoch + (0 if arm == "fixed_floor" else 10_000),
                    group_blocks=True,
                ),
                "holdout_reference": _paired_delta(
                    raw[str(epoch)][arm]["holdout"],
                    base_holdout,
                    seed=43_908_100 + epoch + (0 if arm == "fixed_floor" else 10_000),
                ),
            }

    report = {
        "training_report": _load_json(root / "training_report.json"),
        "training": {
            arm: _load_json(root / arm / "summary.json") for arm in ARMS
        },
        "trajectory": trajectory,
    }
    report["screen"] = _screen(report)
    return report


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(report: dict) -> str:
    screen = report["screen"]
    lines = [
        "# Shaped-Reward Zero-Tail Validation",
        "",
        "Both arms resume the exact p4870 tempo-dedup control. Fixed-floor keeps "
        "the mature native shaping scale at 0.15; zero-tail reaches trainer "
        "multiplier 0 at p6900 and remains terminal-only through p7800.",
        "",
        "## Outcome",
        "",
        f"Decision: **{screen['status']}**.",
        "",
        "| Epoch | Tail vs floor | LCB80 | Parent paired delta (80%) | Holdout paired delta (80%) | Floor Water | Tail Water | Floor portals | Tail portals |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in EPOCHS:
        point = report["trajectory"][str(epoch)]
        floor = point["arms"]["fixed_floor"]
        tail = point["arms"]["zero_tail"]
        parent_delta = point["paired_deltas"]["h2h_vs_parent"]
        holdout_delta = point["paired_deltas"]["holdout_reference"]
        lines.append(
            f"| {epoch} | {point['tail_vs_floor']['score']:.3f} | "
            f"{point['tail_vs_floor']['paired_lcb80']:.3f} | "
            f"{parent_delta['mean']:+.3f} ({parent_delta['lcb80']:+.3f}, {parent_delta['ucb80']:+.3f}) | "
            f"{holdout_delta['mean']:+.3f} ({holdout_delta['lcb80']:+.3f}, {holdout_delta['ucb80']:+.3f}) | "
            f"{floor['holdout_reference']['water_score']:.3f} | "
            f"{tail['holdout_reference']['water_score']:.3f} | "
            f"{floor['opportunity']['portals_per_game']:.2f} | "
            f"{tail['opportunity']['portals_per_game']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Conditional Mechanics",
            "",
            "| Epoch | Arm | Water spell slots | Proactive use/legal | Response use/legal | Garden comparable share | Unique cards | Four-of slots | Sibling KL |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for epoch in EPOCHS:
        for arm in ARMS:
            point = report["trajectory"][str(epoch)]["arms"][arm]
            opp = point["opportunity"]
            deck = point["decks"]
            lines.append(
                f"| {epoch} | {arm} | {_fmt(opp['water_spell_slot_share'])} | "
                f"{_fmt(opp['water_main_spell_selected_per_legal_window'])} | "
                f"{_fmt(opp['water_response_spell_selected_per_legal_window'])} | "
                f"{_fmt(opp['direct_garden_share_when_comparable'])} | "
                f"{deck['deterministic_main_unique_mean']:.2f} | "
                f"{deck['deterministic_quad_slot_share_mean']:.3f} | "
                f"{point['sibling_gate_kl']['mean']:.3g} |"
            )

    endpoint = report["trajectory"]["7800"]
    lines.extend(
        [
            "",
            "## Endpoint Play Style",
            "",
            "Battle action ratios come from the same fixed-seed p4870 parent panel.",
            "",
            "| Arm | Attack | Spell | Weapon | Portal | Entity | No-op | Ability | Length |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        metrics = endpoint["arms"][arm]["h2h_vs_parent"]["battle_metrics"]
        lines.append(
            f"| {arm} | {metrics['attack_rate_mean']:.3f} | "
            f"{metrics['spell_rate_mean']:.3f} | {metrics['weapon_rate_mean']:.3f} | "
            f"{metrics['portal_rate_mean']:.3f} | {metrics['play_entity_rate_mean']:.3f} | "
            f"{metrics['noop_rate_mean']:.3f} | {metrics['ability_rate_mean']:.3f} | "
            f"{metrics['episode_length_mean']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Registered Gates",
            "",
            f"- Endpoint direct score: `{screen['endpoint_direct_score']:.3f}` "
            f"(LCB80 `{screen['endpoint_direct_lcb80']:.3f}`); late-checkpoint mean "
            f"`{screen['late_direct_mean']:.3f}`.",
            f"- Parent-panel paired endpoint delta: `{screen['endpoint_parent_delta']['mean']:+.3f}` "
            f"(80% `{screen['endpoint_parent_delta']['lcb80']:+.3f}` to "
            f"`{screen['endpoint_parent_delta']['ucb80']:+.3f}`).",
            f"- Heldout paired endpoint delta: `{screen['endpoint_holdout_delta']['mean']:+.3f}` "
            f"(80% `{screen['endpoint_holdout_delta']['lcb80']:+.3f}` to "
            f"`{screen['endpoint_holdout_delta']['ucb80']:+.3f}`).",
            f"- Endpoint Water proactive/response deltas: "
            f"`{_fmt(screen['water_main_selection_delta'])}` / "
            f"`{_fmt(screen['water_response_selection_delta'])}`; holdout Water "
            f"delta `{screen['holdout_water_delta']:+.3f}`.",
            f"- Endpoint portal ratio: `{_fmt(screen['portal_ratio'])}`; unique-card "
            f"delta `{screen['deterministic_unique_delta']:+.2f}`; four-of-slot delta "
            f"`{screen['deterministic_quad_slot_delta']:+.3f}`.",
            f"- Integrity `{screen['integrity_pass']}`, throughput "
            f"`{screen['throughput_pass']}`, strength safety "
            f"`{screen['strength_safety_pass']}`, mechanics safety "
            f"`{screen['mechanics_safety_pass']}`, clear strength "
            f"`{screen['clear_strength_pass']}`.",
            "",
            "Garden selection remains diagnostic rather than an adoption gate because "
            "the exact Garden counterfactual was neutral-negative. Promotion events "
            "remain secondary telemetry and are not used here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _deck_markdown(report: dict) -> str:
    lines = [
        "# Endpoint Deck Compositions by Portal",
        "",
        "Greedy p7800 decks are listed separately for the fixed-floor and "
        "zero-tail arms. Sampled summaries show whether the stochastic policy "
        "uses a broader shell than its argmax deck.",
    ]
    for arm in ARMS:
        payload = _load_json(
            Path(report["_root"]) / "eval" / "p7800" / arm / "decks.json"
        )
        lines.extend(["", f"## {arm}", ""])
        for gate_code in GATE_ORDER:
            gate = payload["gates"][gate_code]
            deck = gate["argmax_deck"]
            sampled = gate["sampled"]
            leader = deck["leader"]
            summary = deck["summary"]
            sampled_summary = sampled["summary"]
            leader_split = ", ".join(
                f"{code}: {count}" for code, count in sorted(sampled["leader_split"].items())
            )
            lines.extend(
                [
                    f"### {gate_code}: {gate['gate_name']}",
                    "",
                    f"Leader: **{leader['name']}** (`{leader['code']}`). Greedy average "
                    f"cost `{summary['avg_cost']:.2f}`; sampled average cost "
                    f"`{sampled_summary['avg_cost']:.2f}`. Sampled leader split: "
                    f"{leader_split}.",
                    "",
                    "| Copies | Card | Code | Cost | Type | Element |",
                    "|---:|---|---|---:|---|---|",
                ]
            )
            for card in deck["cards"]:
                lines.append(
                    f"| {int(card['copies'])} | {card['name']} | `{card['code']}` | "
                    f"{int(card['cost'])} | {card['type']} | {card['element']} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--decks-markdown", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.root)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    report["_root"] = str(args.root)
    args.decks_markdown.write_text(_deck_markdown(report), encoding="utf-8")
    print(json.dumps(report["screen"], indent=2))


if __name__ == "__main__":
    main()
