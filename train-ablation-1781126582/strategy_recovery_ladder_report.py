from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Callable

import numpy as np


ARMS = ("control", "tempo_dedup", "portal_gp_tail")
FIRE_KAGORO = "AZK01-121"
FIRE_ZERO = "STT04-001"
WATER_GATES = ("STT02-002", "AZK01-126")
SIBLING_GATES = (
    ("STT01-002", "AZK01-120"),
    ("STT02-002", "AZK01-126"),
    ("STT04-002", "AZK01-122"),
    ("STT03-002", "AZK01-124"),
)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_metric_rows(path: Path) -> list[dict]:
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
        raise ValueError(f"No metric rows in {path}")
    return rows


def _numeric_values(rows: list[dict], key: str) -> np.ndarray:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]
    return np.asarray(values, dtype=np.float64)


def _training_summary(rows: list[dict]) -> dict:
    steady = rows[20:] if len(rows) > 40 else rows[1:]
    if not steady:
        steady = rows
    sps = _numeric_values(steady, "SPS")
    kl = _numeric_values(steady, "losses/approx_kl")
    shaping = _numeric_values(rows, "environment/reward_shaping_scale")
    final = rows[-1]
    return {
        "metric_rows": len(rows),
        "epoch_first": int(rows[0]["epoch"]),
        "epoch_final": int(final["epoch"]),
        "learning_rate_first": float(rows[0]["learning_rate"]),
        "learning_rate_final": float(final["learning_rate"]),
        "sps_median": float(np.median(sps)),
        "sps_p10": float(np.quantile(sps, 0.10)),
        "sps_last100_median": float(np.median(sps[-100:])),
        "approx_kl_median": float(np.median(kl)) if kl.size else None,
        "approx_kl_p90": float(np.quantile(kl, 0.90)) if kl.size else None,
        "reward_shaping_scale_first": float(shaping[0]) if shaping.size else None,
        "reward_shaping_scale_final": float(shaping[-1]) if shaping.size else None,
    }


def _game_score(game: dict) -> float:
    value = game.get("score", game.get("candidate_score"))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Game lacks numeric candidate score: {game.get('game_id')}")
    return float(value)


def _score_map(
    payload: dict,
    key: Callable[[dict], str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for game in payload["games"]:
        grouped[key(game)].append(_game_score(game))
    return {group: float(np.mean(scores)) for group, scores in grouped.items()}


def _paired_delta(
    candidate: dict,
    control: dict,
    *,
    seed: int,
    group_blocks: bool = False,
) -> dict:
    key = (
        (lambda game: str(game["block_id"]))
        if group_blocks
        else (lambda game: str(game["game_id"]))
    )
    candidate_scores = _score_map(candidate, key)
    control_scores = _score_map(control, key)
    if candidate_scores.keys() != control_scores.keys():
        raise ValueError("Candidate and control evaluation schedules differ")
    ids = sorted(candidate_scores)
    deltas = np.asarray(
        [candidate_scores[item] - control_scores[item] for item in ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = deltas[
        rng.integers(0, len(deltas), size=(30_000, len(deltas)))
    ].mean(axis=1)
    return {
        "paired_units": len(deltas),
        "mean": float(deltas.mean()),
        "lcb80": float(np.quantile(samples, 0.10)),
        "ucb80": float(np.quantile(samples, 0.90)),
        "lcb95": float(np.quantile(samples, 0.025)),
        "ucb95": float(np.quantile(samples, 0.975)),
    }


def _weighted_gate_score(payload: dict, gates: tuple[str, ...]) -> float:
    numerator = 0.0
    denominator = 0
    by_gate = payload["summary"]["by_candidate_gate"]
    for gate in gates:
        games = int(by_gate[gate]["games"])
        numerator += games * float(by_gate[gate]["score"])
        denominator += games
    return numerator / denominator


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


def _h2h_summary(payload: dict) -> dict:
    summary = payload["summary"]
    return {
        "episodes": int(summary["episodes"]),
        "score": float(summary["score"]),
        "paired_lcb80": float(summary["paired_lcb_80"]),
        "timeout_rate": float(summary["timeout_rate"]),
        "battle_metrics": dict(summary.get("battle_metrics", {})),
    }


def _deck_signature(deck: dict) -> tuple:
    cards = tuple(sorted((str(card["code"]), int(card["copies"])) for card in deck["cards"]))
    return str(deck["leader"]["code"]), cards


def _deck_summary(payload: dict, deterministic_metrics: dict) -> dict:
    gates = payload["gates"]
    sampled_cost = []
    sampled_spell_share = []
    identical_sibling_pairs = 0
    for gate in gates.values():
        sampled = gate["sampled"]["summary"]
        sampled_cost.append(float(sampled["avg_cost"]))
        sampled_spell_share.append(float(sampled["type_share"].get("SPELL", 0.0)))
    for gate_a, gate_b in SIBLING_GATES:
        if _deck_signature(gates[gate_a]["argmax_deck"]) == _deck_signature(
            gates[gate_b]["argmax_deck"]
        ):
            identical_sibling_pairs += 1
    return {
        "sampled_avg_cost": float(np.mean(sampled_cost)),
        "sampled_spell_slot_share": float(np.mean(sampled_spell_share)),
        "argmax_identical_sibling_pairs": identical_sibling_pairs,
        "deterministic_main_unique_mean": float(deterministic_metrics["main_unique_mean"]),
        "deterministic_quad_slot_share_mean": float(
            deterministic_metrics["quad_slot_share_mean"]
        ),
        "deterministic_singleton_slot_share_mean": float(
            deterministic_metrics["singleton_slot_share_mean"]
        ),
    }


def _gate_kl_summary(payload: dict) -> dict:
    means = {element: float(value["mean_kl"]) for element, value in payload.items()}
    values = np.asarray(list(means.values()), dtype=np.float64)
    return {
        "by_element": means,
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _opportunity_summary(payload: dict) -> dict:
    overall = payload["groups"]["ALL"]
    water = payload["groups"]["ELEMENT/WATER"]
    garden_cards = payload["garden_conditional_on_play"]
    fire_leaders = payload.get("leaders_by_element", {}).get("FIRE", {})
    fire_total = sum(int(value) for value in fire_leaders.values())
    garden_legal = sum(int(value.get("legal_garden_windows", 0)) for value in garden_cards.values())
    garden_plays = sum(int(value.get("garden_plays", 0)) for value in garden_cards.values())
    effect_offers = sum(
        int(value.get("interactive_effect_offers", 0)) for value in garden_cards.values()
    )
    return {
        "games": int(payload["n_games"]),
        "water_spell_slot_share": water.get("spell_slot_share"),
        "water_main_spell_selected_per_legal_window": water.get(
            "main_spell_selected_per_legal_window"
        ),
        "water_response_spell_selected_per_legal_window": water.get(
            "response_spell_selected_per_legal_window"
        ),
        "direct_garden_share_when_comparable": overall.get(
            "garden_share_when_comparable_entity_selected"
        ),
        "special_garden_legal_windows": garden_legal,
        "special_garden_plays": garden_plays,
        "special_garden_effect_offers": effect_offers,
        "portals_per_game": float(overall.get("selected/GATE_PORTAL", 0))
        / max(int(payload["n_games"]), 1),
        "portal_decision_share": float(overall.get("selected/GATE_PORTAL", 0))
        / max(int(overall["decision_windows"]), 1),
        "fire_leader_counts": dict(fire_leaders),
        "fire_kagoro_share": _safe_ratio(
            float(fire_leaders.get(FIRE_KAGORO, 0)), float(fire_total)
        ),
        "fire_zero_share": _safe_ratio(
            float(fire_leaders.get(FIRE_ZERO, 0)), float(fire_total)
        ),
        "fire_leader_outcomes": payload.get("leader_outcomes_by_element", {}).get(
            "FIRE", {}
        ),
        "actual_overlap_reward_per_game_at_floor": float(
            payload["reward_overlap"]["combined_at_mature_floor_per_game"]
        ),
        "reward_config": dict(payload["reward_overlap"]["config"]),
    }


def _build_arm(root: Path, arm: str) -> tuple[dict, dict]:
    arm_root = root / arm
    h2h_parent = _load_json(arm_root / "h2h_vs_parent.json")
    train_ref = _load_json(arm_root / "draftref_train.json")
    holdout_ref = _load_json(arm_root / "draftref_holdout.json")
    opportunity = _load_json(arm_root / "opportunity.json")
    deterministic_metrics = holdout_ref["summary"]["deck_metrics"]
    raw = {
        "h2h_parent": h2h_parent,
        "train_ref": train_ref,
        "holdout_ref": holdout_ref,
    }
    result = {
        "training": _training_summary(_load_metric_rows(arm_root / "train.jsonl")),
        "h2h_vs_parent": _h2h_summary(h2h_parent),
        "train_reference": _evaluation_summary(train_ref),
        "holdout_reference": _evaluation_summary(holdout_ref),
        "decks": _deck_summary(_load_json(arm_root / "decks.json"), deterministic_metrics),
        "sibling_gate_kl": _gate_kl_summary(_load_json(arm_root / "gate_kl.json")),
        "opportunity": _opportunity_summary(opportunity),
    }
    if arm != "control":
        direct = _load_json(arm_root / "h2h_vs_control.json")
        raw["h2h_control"] = direct
        result["h2h_vs_control"] = _h2h_summary(direct)
    return result, raw


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _screen(candidate: dict, control: dict, paired: dict) -> dict:
    throughput_ratio = (
        candidate["training"]["sps_median"]
        / control["training"]["sps_median"]
    )
    portal_ratio = (
        candidate["opportunity"]["portals_per_game"]
        / max(control["opportunity"]["portals_per_game"], 1e-9)
    )
    water_spell_delta = _delta(
        candidate["opportunity"]["water_main_spell_selected_per_legal_window"],
        control["opportunity"]["water_main_spell_selected_per_legal_window"],
    )
    garden_delta = _delta(
        candidate["opportunity"]["direct_garden_share_when_comparable"],
        control["opportunity"]["direct_garden_share_when_comparable"],
    )
    kagoro_delta = _delta(
        candidate["opportunity"]["fire_kagoro_share"],
        control["opportunity"]["fire_kagoro_share"],
    )
    recovered = []
    if water_spell_delta is not None and water_spell_delta >= 0.03:
        recovered.append("water_proactive_spell_use")
    if garden_delta is not None and garden_delta >= 0.02:
        recovered.append("direct_garden_choice")
    if kagoro_delta is not None and kagoro_delta >= 0.10:
        recovered.append("fire_leader_selection")

    direct = candidate["h2h_vs_control"]
    strength_pass = (
        direct["score"] >= 0.47
        and direct["paired_lcb80"] >= 0.43
        and paired["h2h_vs_parent"]["lcb80"] >= -0.05
    )
    throughput_pass = throughput_ratio >= 0.95
    portal_pass = portal_ratio >= 0.90
    water_guard_pass = (
        candidate["holdout_reference"]["water_score"]
        - control["holdout_reference"]["water_score"]
        >= -0.05
    )
    integrity_pass = (
        candidate["training"]["epoch_final"] == 3900
        and candidate["training"]["learning_rate_final"] <= 1e-8
        and candidate["training"]["reward_shaping_scale_final"] is not None
        and abs(candidate["training"]["reward_shaping_scale_final"] - 0.15) <= 1e-3
    )
    eligible = (
        strength_pass
        and throughput_pass
        and portal_pass
        and water_guard_pass
        and integrity_pass
    )
    if eligible and recovered:
        status = "advance"
    elif eligible:
        status = "neutral_no_conditional_recovery"
    elif not strength_pass:
        status = "stop_strength"
    elif not throughput_pass:
        status = "stop_throughput"
    elif not portal_pass:
        status = "stop_portal_backbone"
    elif not water_guard_pass:
        status = "stop_water_guard"
    else:
        status = "stop_integrity"
    return {
        "status": status,
        "recovered_signals": recovered,
        "strength_pass": strength_pass,
        "throughput_pass": throughput_pass,
        "portal_backbone_pass": portal_pass,
        "water_guard_pass": water_guard_pass,
        "integrity_pass": integrity_pass,
        "throughput_ratio": throughput_ratio,
        "portal_rate_ratio": portal_ratio,
        "water_main_spell_selection_delta": water_spell_delta,
        "direct_garden_selection_delta": garden_delta,
        "fire_kagoro_share_delta": kagoro_delta,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(report: dict) -> str:
    lines = [
        "# Strategy Recovery 15M Ladder",
        "",
        "All arms resume the exact qualified p2930 model/trainer/league state, restore "
        "the episode-driven shaping scale at `0.15`, and finish a true p2930-p3900 "
        "cosine schedule. Promotion is excluded from the decision rule.",
        "",
        "## Strength And Runtime",
        "",
        "| Arm | SPS | Ratio | KL median | Final LR | H2H parent | H2H control | Holdout ref | Holdout Water | Screen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    control_sps = report["arms"]["control"]["training"]["sps_median"]
    for arm in ARMS:
        value = report["arms"][arm]
        h2h_control = value.get("h2h_vs_control", {}).get("score")
        screen = value.get("screen", {}).get("status", "control_gate_passed")
        lines.append(
            f"| {arm} | {value['training']['sps_median']:.0f} | "
            f"{value['training']['sps_median'] / control_sps:.3f} | "
            f"{_fmt(value['training']['approx_kl_median'], 5)} | "
            f"{value['training']['learning_rate_final']:.2e} | "
            f"{value['h2h_vs_parent']['score']:.3f} | {_fmt(h2h_control)} | "
            f"{value['holdout_reference']['score']:.3f} | "
            f"{value['holdout_reference']['water_score']:.3f} | {screen} |"
        )

    lines.extend(
        [
            "",
            "## Conditional Mechanics",
            "",
            "| Arm | Water spell slots | Water main use/legal | Water response use/legal | Garden choice | Special Garden play/legal | Portals/game | Kagoro share | Zero share | Overlap reward/game |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        value = report["arms"][arm]["opportunity"]
        lines.append(
            f"| {arm} | {_fmt(value['water_spell_slot_share'])} | "
            f"{_fmt(value['water_main_spell_selected_per_legal_window'])} | "
            f"{_fmt(value['water_response_spell_selected_per_legal_window'])} | "
            f"{_fmt(value['direct_garden_share_when_comparable'])} | "
            f"{value['special_garden_plays']}/{value['special_garden_legal_windows']} | "
            f"{value['portals_per_game']:.2f} | {_fmt(value['fire_kagoro_share'])} | "
            f"{_fmt(value['fire_zero_share'])} | "
            f"{value['actual_overlap_reward_per_game_at_floor']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Deck Policy",
            "",
            "| Arm | Sampled avg IKZ | Sampled spell share | Deterministic unique | Four-of slots | Identical sibling pairs | Mean sibling KL |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        value = report["arms"][arm]
        deck = value["decks"]
        lines.append(
            f"| {arm} | {deck['sampled_avg_cost']:.2f} | "
            f"{deck['sampled_spell_slot_share']:.3f} | "
            f"{deck['deterministic_main_unique_mean']:.2f} | "
            f"{deck['deterministic_quad_slot_share_mean']:.3f} | "
            f"{deck['argmax_identical_sibling_pairs']}/4 | "
            f"{value['sibling_gate_kl']['mean']:.3e} |"
        )

    lines.extend(["", "## Paired Changes Versus Control", ""])
    for arm in ARMS[1:]:
        value = report["arms"][arm]
        paired = value["paired_deltas_vs_control"]
        screen = value["screen"]
        lines.append(
            f"- **{arm}:** parent-panel delta `{paired['h2h_vs_parent']['mean']:+.3f}` "
            f"(80% CI `{paired['h2h_vs_parent']['lcb80']:+.3f}` to "
            f"`{paired['h2h_vs_parent']['ucb80']:+.3f}`), holdout-reference delta "
            f"`{paired['holdout_reference']['mean']:+.3f}`, direct H2H-control score "
            f"`{value['h2h_vs_control']['score']:.3f}`, recovered "
            f"`{', '.join(screen['recovered_signals']) or 'none'}`."
        )

    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "A candidate advances only if direct H2H and paired parent-panel strength pass, "
            "median SPS is at least 95% of control, portals/game retain at least 90% of "
            "control, heldout Water loses no more than five points, the schedule finishes "
            "at p3900 with LR near zero and shaping scale 0.15, and at least one conditional "
            "blind spot recovers. Recovery screens are +3 points Water proactive spell use, "
            "+2 points direct Garden choice, or +10 points Kagoro share. These thresholds "
            "screen hypotheses; terminal strength remains primary.",
            "",
            f"Control strength gate: `{report['control_strength_gate']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    arms = {}
    raw = {}
    for arm in ARMS:
        arms[arm], raw[arm] = _build_arm(args.root, arm)

    for arm_index, arm in enumerate(ARMS[1:], start=1):
        paired = {
            "h2h_vs_parent": _paired_delta(
                raw[arm]["h2h_parent"],
                raw["control"]["h2h_parent"],
                seed=42_950_000 + arm_index,
                group_blocks=True,
            ),
            "train_reference": _paired_delta(
                raw[arm]["train_ref"],
                raw["control"]["train_ref"],
                seed=42_951_000 + arm_index,
            ),
            "holdout_reference": _paired_delta(
                raw[arm]["holdout_ref"],
                raw["control"]["holdout_ref"],
                seed=42_952_000 + arm_index,
            ),
        }
        arms[arm]["paired_deltas_vs_control"] = paired
        arms[arm]["screen"] = _screen(arms[arm], arms["control"], paired)

    control_gate = _load_json(args.root / "control" / "control_strength_gate.json")
    report = {
        "schema_version": 1,
        "parent": "qualified_promotion_v2_p2930",
        "control_strength_gate": control_gate,
        "arms": arms,
        "advance_candidates": [
            arm
            for arm in ARMS[1:]
            if arms[arm]["screen"]["status"] == "advance"
        ],
    }
    json_path = args.json or args.root / "ladder_report.json"
    markdown_path = args.markdown or args.root / "ladder_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(f"[strategy-recovery-report] wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
