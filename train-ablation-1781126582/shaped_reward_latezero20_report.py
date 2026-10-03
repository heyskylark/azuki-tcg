from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strategy_recovery_ladder_report import (
    _deck_summary,
    _h2h_summary,
    _load_json,
    _opportunity_summary,
    _paired_delta,
    _weighted_gate_score,
)


ARMS = ("fixed_floor", "late_zero")
TRAJECTORY_EPOCHS = (7200, 7300, 7400, 7500, 7600, 7700, 7800)
ANCHOR_EPOCHS = (7200, 7500, 7800)
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


def _extended_deck_summary(payload: dict, deterministic_metrics: dict) -> dict:
    result = _deck_summary(payload, deterministic_metrics)
    summaries = [gate["argmax_deck"]["summary"] for gate in payload["gates"].values()]
    result["argmax_avg_cost"] = float(
        np.mean([float(summary["avg_cost"]) for summary in summaries])
    )
    for card_type in ("ENTITY", "SPELL", "WEAPON"):
        result[f"argmax_{card_type.lower()}_slot_share"] = float(
            np.mean(
                [
                    float(summary.get("type_share", {}).get(card_type, 0.0))
                    for summary in summaries
                ]
            )
        )
    return result


def _anchor_point(root: Path, arm: str, epoch: int) -> tuple[dict, dict]:
    point_root = root / "eval" / f"p{epoch}" / arm
    parent = _load_json(point_root / "h2h_vs_parent.json")
    holdout = _load_json(point_root / "draftref_holdout.json")
    decks = _load_json(point_root / "decks.json")
    opportunity = _load_json(point_root / "opportunity" / "opportunity.json")
    deterministic = holdout["summary"]["deck_metrics"]
    point = {
        "h2h_vs_parent": _h2h_summary(parent),
        "holdout_reference": _evaluation_summary(holdout),
        "decks": _extended_deck_summary(decks, deterministic),
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


def _trajectory_trend(scores: list[float]) -> dict:
    x = np.asarray(TRAJECTORY_EPOCHS, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(x))
        for j in range(i + 1, len(x))
    ]
    late = y[-3:]
    return {
        "scores": [float(value) for value in y],
        "ols_slope_per_100_updates": float(np.polyfit(x, y, 1)[0] * 100.0),
        "theil_sen_slope_per_100_updates": float(np.median(slopes) * 100.0),
        "late_three_mean": float(np.mean(late)),
        "late_three_min": float(np.min(late)),
        "score_min": float(np.min(y)),
        "score_max": float(np.max(y)),
        "score_range": float(np.max(y) - np.min(y)),
        "mean_absolute_step": float(np.mean(np.abs(np.diff(y)))),
    }


def _screen(report: dict) -> dict:
    endpoint = report["trajectory"]["7800"]
    floor = endpoint["arms"]["fixed_floor"]
    late = endpoint["arms"]["late_zero"]
    direct = endpoint["late_vs_floor"]
    paired = endpoint["paired_deltas"]
    trend = report["direct_trend"]

    water_main_delta = _difference(
        late["opportunity"]["water_main_spell_selected_per_legal_window"],
        floor["opportunity"]["water_main_spell_selected_per_legal_window"],
    )
    water_response_delta = _difference(
        late["opportunity"]["water_response_spell_selected_per_legal_window"],
        floor["opportunity"]["water_response_spell_selected_per_legal_window"],
    )
    garden_delta = _difference(
        late["opportunity"]["direct_garden_share_when_comparable"],
        floor["opportunity"]["direct_garden_share_when_comparable"],
    )
    portal_ratio = _safe_ratio(
        late["opportunity"]["portals_per_game"],
        floor["opportunity"]["portals_per_game"],
    )
    water_holdout_delta = (
        late["holdout_reference"]["water_score"]
        - floor["holdout_reference"]["water_score"]
    )
    unique_delta = (
        late["decks"]["deterministic_main_unique_mean"]
        - floor["decks"]["deterministic_main_unique_mean"]
    )
    quad_delta = (
        late["decks"]["deterministic_quad_slot_share_mean"]
        - floor["decks"]["deterministic_quad_slot_share_mean"]
    )
    floor_battle = floor["h2h_vs_parent"]["battle_metrics"]
    late_battle = late["h2h_vs_parent"]["battle_metrics"]
    attack_delta = late_battle["attack_rate_mean"] - floor_battle["attack_rate_mean"]
    spell_delta = late_battle["spell_rate_mean"] - floor_battle["spell_rate_mean"]
    game_length_ratio = _safe_ratio(
        late_battle["episode_length_mean"], floor_battle["episode_length_mean"]
    )
    heldout_delta = paired["holdout_reference"]

    training = report["training_report"]
    expected_multipliers = {
        "6900": 1.0,
        "7000": 2.0 / 3.0,
        "7100": 1.0 / 3.0,
        "7200": 0.0,
        "7300": 0.0,
        "7400": 0.0,
        "7500": 0.0,
        "7600": 0.0,
        "7700": 0.0,
        "7800": 0.0,
    }
    observed_multipliers = training.get("late_zero_checkpoint_multipliers", {})
    multiplier_integrity = all(
        abs(float(observed_multipliers.get(epoch, -1.0)) - expected) <= 1e-8
        for epoch, expected in expected_multipliers.items()
    )
    integrity = (
        bool(training.get("fork_identical"))
        and multiplier_integrity
        and int(training.get("late_zero_zero_update_count", -1)) == 601
        and float(training.get("late_zero_terminal_labels_max", 0.0)) > 0.0
    )
    throughput = training.get("status") == "pass"
    strength_safety = (
        direct["score"] >= 0.47
        and trend["late_three_mean"] >= 0.47
        and paired["h2h_vs_parent"]["lcb80"] >= -0.03
        and heldout_delta["lcb80"] >= -0.03
    )
    mechanics_safety = (
        water_main_delta is not None
        and water_main_delta >= -0.03
        and water_response_delta is not None
        and water_response_delta >= -0.05
        and portal_ratio is not None
        and portal_ratio >= 0.90
        and water_holdout_delta >= -0.05
        and attack_delta >= -0.04
        and game_length_ratio is not None
        and game_length_ratio <= 1.15
        and unique_delta >= -2.0
        and quad_delta <= 0.05
    )
    clear_strength = (
        direct["score"] >= 0.52
        and direct["paired_lcb80"] >= 0.48
        and trend["late_three_mean"] >= 0.51
        and trend["theil_sen_slope_per_100_updates"] >= 0.0
    )
    viable_neutral = (
        direct["score"] >= 0.48
        and trend["late_three_mean"] >= 0.48
        and trend["theil_sen_slope_per_100_updates"] >= -0.005
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
        status = "adopt_zero"
    elif viable_neutral:
        status = "zero_viable_neutral"
    else:
        status = "inconclusive_keep_floor"

    return {
        "status": status,
        "adopt_zero": status == "adopt_zero",
        "zero_viable": status in {"adopt_zero", "zero_viable_neutral"},
        "integrity_pass": integrity,
        "throughput_pass": throughput,
        "strength_safety_pass": strength_safety,
        "mechanics_safety_pass": mechanics_safety,
        "clear_strength_pass": clear_strength,
        "endpoint_direct_score": direct["score"],
        "endpoint_direct_lcb80": direct["paired_lcb80"],
        "late_three_direct_mean": trend["late_three_mean"],
        "theil_sen_slope_per_100_updates": trend[
            "theil_sen_slope_per_100_updates"
        ],
        "endpoint_parent_delta": paired["h2h_vs_parent"],
        "endpoint_holdout_delta": heldout_delta,
        "heldout_approximately_neutral": abs(heldout_delta["mean"]) <= 0.015,
        "water_main_selection_delta": water_main_delta,
        "water_response_selection_delta": water_response_delta,
        "direct_garden_selection_delta": garden_delta,
        "portal_ratio": portal_ratio,
        "holdout_water_delta": water_holdout_delta,
        "attack_rate_delta": attack_delta,
        "spell_rate_delta": spell_delta,
        "game_length_ratio": game_length_ratio,
        "deterministic_unique_delta": unique_delta,
        "deterministic_quad_slot_delta": quad_delta,
    }


def build_report(root: Path) -> dict:
    trajectory: dict[str, dict] = {}
    raw: dict[str, dict[str, dict]] = {}
    direct_scores = []
    for epoch in TRAJECTORY_EPOCHS:
        epoch_key = str(epoch)
        direct_payload = _load_json(
            root / "eval" / f"p{epoch}" / "h2h_late_vs_floor.json"
        )
        direct = _h2h_summary(direct_payload)
        direct_scores.append(direct["score"])
        trajectory[epoch_key] = {"late_vs_floor": direct}
        if epoch not in ANCHOR_EPOCHS:
            continue
        raw[epoch_key] = {}
        arms = {}
        for arm in ARMS:
            arms[arm], raw[epoch_key][arm] = _anchor_point(root, arm, epoch)
        trajectory[epoch_key]["arms"] = arms
        trajectory[epoch_key]["paired_deltas"] = {
            "h2h_vs_parent": _paired_delta(
                raw[epoch_key]["late_zero"]["parent"],
                raw[epoch_key]["fixed_floor"]["parent"],
                seed=44_905_100 + epoch,
                group_blocks=True,
            ),
            "holdout_reference": _paired_delta(
                raw[epoch_key]["late_zero"]["holdout"],
                raw[epoch_key]["fixed_floor"]["holdout"],
                seed=44_906_100 + epoch,
            ),
        }

    for arm in ARMS:
        baseline_parent = raw["7200"][arm]["parent"]
        baseline_holdout = raw["7200"][arm]["holdout"]
        for epoch in ANCHOR_EPOCHS:
            trajectory[str(epoch)]["arms"][arm]["delta_vs_p7200"] = {
                "h2h_vs_parent": _paired_delta(
                    raw[str(epoch)][arm]["parent"],
                    baseline_parent,
                    seed=44_907_100 + epoch + (0 if arm == "fixed_floor" else 10_000),
                    group_blocks=True,
                ),
                "holdout_reference": _paired_delta(
                    raw[str(epoch)][arm]["holdout"],
                    baseline_holdout,
                    seed=44_908_100 + epoch + (0 if arm == "fixed_floor" else 10_000),
                ),
            }

    report = {
        "training_report": _load_json(root / "training_report.json"),
        "shared_training": _load_json(root / "shared" / "summary.json"),
        "fork_qualification": _h2h_summary(
            _load_json(root / "shared" / "fork_qualification.json")
        ),
        "training": {
            arm: _load_json(root / arm / "summary.json") for arm in ARMS
        },
        "trajectory": trajectory,
        "direct_trend": _trajectory_trend(direct_scores),
    }
    report["screen"] = _screen(report)
    return report


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(report: dict) -> str:
    screen = report["screen"]
    trend = report["direct_trend"]
    lines = [
        "# Late Shaped-Reward Removal Validation",
        "",
        "Both arms share the exact p4870-p6900 floor-trained trunk. The control "
        "keeps effective shaping at 0.15; late-zero ramps to zero at p7200 and "
        "then trains for 601 logged zero-multiplier updates through p7800.",
        "",
        "## Outcome",
        "",
        f"Decision: **{screen['status']}**.",
        "",
        "| Epoch | Late-zero vs floor | LCB80 |",
        "|---:|---:|---:|",
    ]
    for epoch in TRAJECTORY_EPOCHS:
        direct = report["trajectory"][str(epoch)]["late_vs_floor"]
        lines.append(
            f"| {epoch} | {direct['score']:.3f} | {direct['paired_lcb80']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Zero-Interval Trend",
            "",
            f"The final-three mean is `{trend['late_three_mean']:.3f}`. The robust "
            f"Theil-Sen slope is `{trend['theil_sen_slope_per_100_updates']:+.4f}` "
            f"score per 100 updates (OLS `{trend['ols_slope_per_100_updates']:+.4f}`). "
            f"Scores ranged from `{trend['score_min']:.3f}` to "
            f"`{trend['score_max']:.3f}`, with mean absolute adjacent movement "
            f"`{trend['mean_absolute_step']:.3f}`.",
            "",
            "Training episode return is intentionally excluded from this trend: its "
            "units change when shaped reward is removed. These fixed terminal panels "
            "are comparable across checkpoints.",
            "",
            "## External Anchors",
            "",
            "| Epoch | Arm | vs p4870 | Heldout | Heldout Water | Portals/game | Unique | Four-of |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for epoch in ANCHOR_EPOCHS:
        for arm in ARMS:
            point = report["trajectory"][str(epoch)]["arms"][arm]
            lines.append(
                f"| {epoch} | {arm} | {point['h2h_vs_parent']['score']:.3f} | "
                f"{point['holdout_reference']['score']:.3f} | "
                f"{point['holdout_reference']['water_score']:.3f} | "
                f"{point['opportunity']['portals_per_game']:.2f} | "
                f"{point['decks']['deterministic_main_unique_mean']:.2f} | "
                f"{point['decks']['deterministic_quad_slot_share_mean']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Conditional Mechanics",
            "",
            "| Epoch | Arm | Water spell slots | Proactive use/legal | Response use/legal | Garden comparable | Cost | Entity | Spell | Weapon |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for epoch in ANCHOR_EPOCHS:
        for arm in ARMS:
            point = report["trajectory"][str(epoch)]["arms"][arm]
            opp = point["opportunity"]
            deck = point["decks"]
            lines.append(
                f"| {epoch} | {arm} | {_fmt(opp['water_spell_slot_share'])} | "
                f"{_fmt(opp['water_main_spell_selected_per_legal_window'])} | "
                f"{_fmt(opp['water_response_spell_selected_per_legal_window'])} | "
                f"{_fmt(opp['direct_garden_share_when_comparable'])} | "
                f"{deck['argmax_avg_cost']:.2f} | "
                f"{deck['argmax_entity_slot_share']:.3f} | "
                f"{deck['argmax_spell_slot_share']:.3f} | "
                f"{deck['argmax_weapon_slot_share']:.3f} |"
            )

    endpoint = report["trajectory"]["7800"]
    lines.extend(
        [
            "",
            "## Endpoint Play Style",
            "",
            "Battle ratios come from the same fixed-seed p4870 panel.",
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
            f"{metrics['portal_rate_mean']:.3f} | "
            f"{metrics['play_entity_rate_mean']:.3f} | "
            f"{metrics['noop_rate_mean']:.3f} | {metrics['ability_rate_mean']:.3f} | "
            f"{metrics['episode_length_mean']:.1f} |"
        )

    neutral_note = (
        "falls inside"
        if screen["heldout_approximately_neutral"]
        else "falls outside"
    )
    lines.extend(
        [
            "",
            "## Registered Gates",
            "",
            f"- Endpoint direct score: `{screen['endpoint_direct_score']:.3f}` "
            f"(LCB80 `{screen['endpoint_direct_lcb80']:.3f}`); final-three mean "
            f"`{screen['late_three_direct_mean']:.3f}`.",
            f"- Parent paired endpoint delta: `{screen['endpoint_parent_delta']['mean']:+.3f}` "
            f"(80% `{screen['endpoint_parent_delta']['lcb80']:+.3f}` to "
            f"`{screen['endpoint_parent_delta']['ucb80']:+.3f}`).",
            f"- Heldout paired endpoint delta: `{screen['endpoint_holdout_delta']['mean']:+.3f}` "
            f"(80% `{screen['endpoint_holdout_delta']['lcb80']:+.3f}` to "
            f"`{screen['endpoint_holdout_delta']['ucb80']:+.3f}`); this "
            f"{neutral_note} the registered approximately-one-point neutral band.",
            f"- Water proactive/response deltas: "
            f"`{_fmt(screen['water_main_selection_delta'])}` / "
            f"`{_fmt(screen['water_response_selection_delta'])}`; heldout Water "
            f"`{screen['holdout_water_delta']:+.3f}`.",
            f"- Portal ratio `{_fmt(screen['portal_ratio'])}`; attack delta "
            f"`{screen['attack_rate_delta']:+.3f}`; game-length ratio "
            f"`{_fmt(screen['game_length_ratio'])}`.",
            f"- Unique-card delta `{screen['deterministic_unique_delta']:+.2f}`; "
            f"four-of delta `{screen['deterministic_quad_slot_delta']:+.3f}`.",
            f"- Integrity `{screen['integrity_pass']}`, throughput "
            f"`{screen['throughput_pass']}`, strength safety "
            f"`{screen['strength_safety_pass']}`, mechanics safety "
            f"`{screen['mechanics_safety_pass']}`, clear strength "
            f"`{screen['clear_strength_pass']}`.",
            "",
            "Promotion is not a decision signal. Garden remains diagnostic because "
            "its exact placement counterfactual was neutral-negative. A passing zero "
            "endpoint still requires a separate matched stability extension before "
            "zero shaping becomes the default.",
        ]
    )
    return "\n".join(lines) + "\n"


def _deck_markdown(report: dict, root: Path) -> str:
    lines = [
        "# Endpoint Deck Compositions by Portal",
        "",
        "Greedy p7800 decks are listed separately for the fixed-floor and "
        "late-zero arms. Sampled summaries use 16 drafts per gate.",
    ]
    for arm in ARMS:
        payload = _load_json(root / "eval" / "p7800" / arm / "decks.json")
        lines.extend(["", f"## {arm}", ""])
        for gate_code in GATE_ORDER:
            gate = payload["gates"][gate_code]
            deck = gate["argmax_deck"]
            sampled = gate["sampled"]
            leader = deck["leader"]
            summary = deck["summary"]
            sampled_summary = sampled["summary"]
            leader_split = ", ".join(
                f"{code}: {count}"
                for code, count in sorted(sampled["leader_split"].items())
            )
            lines.extend(
                [
                    f"### {gate_code}: {gate['gate_name']}",
                    "",
                    f"Leader: **{leader['name']}** (`{leader['code']}`). Greedy "
                    f"average cost `{summary['avg_cost']:.2f}`; sampled average "
                    f"cost `{sampled_summary['avg_cost']:.2f}`. Sampled leader "
                    f"split: {leader_split}.",
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
    args.decks_markdown.write_text(
        _deck_markdown(report, args.root), encoding="utf-8"
    )
    print(json.dumps(report["screen"], indent=2))


if __name__ == "__main__":
    main()
