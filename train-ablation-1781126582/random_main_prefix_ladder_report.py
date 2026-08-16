#!/usr/bin/env python3
"""Synthesize the Stage 3 random main-card prefix ladder."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("control_no_prefix", "random_main_prefix")
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
    return sum(
        int(metrics["games"]) * float(metrics["score"]) for metrics in selected
    ) / max(games, 1)


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
            float(context["greedy"]["summary"]["main_unique"])
            for context in contexts
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


def _opportunity_summary(payload: dict) -> dict[str, float | None]:
    overall = payload["groups"]["ALL"]
    water = payload["groups"].get("ELEMENT/WATER", {})
    reward = payload["reward_overlap"]
    return {
        "portal_per_game": float(reward.get("portals", 0.0))
        / max(int(payload["n_games"]), 1),
        "spell_selected_per_legal": overall.get(
            "spell_selected_per_legal_window"
        ),
        "water_main_spell_selected_per_legal": water.get(
            "main_spell_selected_per_legal_window"
        ),
        "garden_share_when_comparable": overall.get(
            "garden_share_when_comparable_entity_selected"
        ),
        "leader_ability_selected_per_legal": overall.get(
            "leader_ability_selected_per_legal_window"
        ),
    }


def _card_funnel_comparison(control: dict, candidate: dict) -> dict:
    control_cards = control["cards"]
    candidate_cards = candidate["cards"]
    codes = sorted(set(control_cards) | set(candidate_cards))
    fields = (
        "deck_seats",
        "observed_in_hand_seats",
        "drawn_after_opening_seats",
        "legal_play_windows",
        "selected_plays",
        "realized_effect_seats",
    )

    def value(cards: dict, code: str, field: str) -> int:
        return int(cards.get(code, {}).get(field, 0))

    newly_reached = {}
    stage_thresholds = {
        "drafted": ("deck_seats", 2),
        "observed": ("observed_in_hand_seats", 2),
        "drawn_after_opening": ("drawn_after_opening_seats", 2),
        "legal": ("legal_play_windows", 2),
        "selected": ("selected_plays", 2),
        "realized": ("realized_effect_seats", 2),
    }
    for name, (field, threshold) in stage_thresholds.items():
        newly_reached[name] = [
            code
            for code in codes
            if value(control_cards, code, field) == 0
            and value(candidate_cards, code, field) >= threshold
        ]

    useful_rare = []
    for code in codes:
        control_decks = value(control_cards, code, "deck_seats")
        candidate_decks = value(candidate_cards, code, "deck_seats")
        candidate_selected = value(candidate_cards, code, "selected_plays")
        candidate_realized = value(candidate_cards, code, "realized_effect_seats")
        if (
            control_decks <= 2
            and candidate_decks >= control_decks + 4
            and candidate_selected >= 2
            and candidate_realized >= 2
        ):
            record = candidate_cards[code]
            useful_rare.append({
                "code": code,
                "name": record.get("name", code),
                "control_deck_seats": control_decks,
                "candidate_deck_seats": candidate_decks,
                "candidate_selected_plays": candidate_selected,
                "candidate_realized_seats": candidate_realized,
                "candidate_deck_win_rate": record.get("deck_win_rate"),
            })
    useful_rare.sort(
        key=lambda item: (
            -int(item["candidate_realized_seats"]),
            -int(item["candidate_deck_seats"]),
            str(item["code"]),
        )
    )
    return {
        "control_coverage": dict(control["coverage"]),
        "candidate_coverage": dict(candidate["coverage"]),
        "coverage_delta": {
            key: int(candidate["coverage"].get(key, 0))
            - int(control["coverage"].get(key, 0))
            for key in sorted(set(control["coverage"]) | set(candidate["coverage"]))
        },
        "newly_reached": newly_reached,
        "newly_reached_counts": {
            key: len(values) for key, values in newly_reached.items()
        },
        "useful_rare_cards": useful_rare,
        "field_totals": {
            field: {
                "control": sum(value(control_cards, code, field) for code in codes),
                "candidate": sum(value(candidate_cards, code, field) for code in codes),
            }
            for field in fields
        },
    }


def _markdown(report: dict) -> str:
    decision = report["decision"]
    lines = [
        "# Stage 3 Random Main-Prefix Ladder",
        "",
        f"Decision: **{decision['verdict']}**.",
        "",
        decision["reason"],
        "",
        "| Epoch | Prefix vs control | Parent-panel delta | Heldout delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for epoch in report["evaluation_epochs"]:
        window = report["windows"][str(epoch)]
        lines.append(
            f"| {epoch} | {window['prefix_vs_control_score']:.3f} | "
            f"{window['parent_panel_delta']:+.3f} | "
            f"{window['heldout_delta']:+.3f} |"
        )
    lines.extend([
        "",
        "## Registered Gates",
        "",
        f"- Integrity: `{decision['integrity_pass']}`",
        f"- Throughput (diagnostic only): raw pass "
        f"`{report['throughput']['raw_pass']}` "
        f"(`{report['throughput']['candidate_over_control']:.4f}` of control; "
        f"relative `{report['throughput']['relative_pass']}`; "
        f"absolute `{report['throughput']['absolute_pass']}`)",
        f"- External nonregression: `{decision['external_safety_pass']}`",
        f"- Mechanics/deck safety: `{decision['mechanics_pass']}`",
        f"- Strength improvement: `{decision['strength_improvement_pass']}`",
        f"- Sustained causal context fit: `{decision['causal_improvement_pass']}`",
        "",
        "## Exposure Funnel",
        "",
        "| Stage | Control cards | Prefix cards | Delta | Newly reached |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    funnels = report["card_funnels"]
    for stage in (
        "drafted",
        "opening",
        "drawn_after_opening",
        "observed",
        "legal",
        "selected",
        "realized",
    ):
        lines.append(
            f"| {stage} | {funnels['control_coverage'].get(stage, 0)} | "
            f"{funnels['candidate_coverage'].get(stage, 0)} | "
            f"{funnels['coverage_delta'].get(stage, 0):+d} | "
            f"{funnels['newly_reached_counts'].get(stage, 0)} |"
        )
    lines.extend([
        "",
        f"Useful rare-card lines: `{len(funnels['useful_rare_cards'])}`. A line "
        "must move from at most two control decks to at least four additional "
        "candidate decks and be both selected and realized in at least two seats.",
        "",
        "## Causal Context Fit",
        "",
        "| Epoch | Prefix matched advantage | Control matched advantage | Delta | Pass |",
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
        "Prefixing was disabled throughout evaluation. Exposure is diagnostic; "
        "advancement still requires a greater-than-noise strength gain or sustained "
        "causal deck/context improvement.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.root / "evaluation"
    manifest = _read(evaluation / "evaluation_manifest.json")
    epochs = [int(value) for value in manifest["evaluation_epochs"]]
    hybrid_epochs = [int(value) for value in manifest["causal_hybrid_epochs"]]
    training = {
        arm: _read(args.root / arm / "training_summary.json") for arm in ARMS
    }
    training_gate = _read(args.root / "training_gate.json")
    if (
        not bool(training_gate.get("throughput_diagnostic_only", False))
        or not bool(training_gate.get("continuation_pass", False))
    ):
        raise RuntimeError(
            "Stage 3 training gate does not record the approved diagnostic-only "
            "throughput policy and a passing integrity continuation"
        )

    windows: dict[str, dict] = {}
    for epoch in epochs:
        root = evaluation / f"p{epoch}"
        direct = _read(root / "prefix_vs_control_no_prefix.json")
        candidate_parent = _read(root / "prefix_vs_parent.json")
        control_parent = _read(root / "control_no_prefix_vs_parent.json")
        candidate_heldout = _read(root / "random_main_prefix" / "heldout.json")
        control_heldout = _read(root / "control_no_prefix" / "heldout.json")
        arms = {}
        for arm in ARMS:
            arm_root = root / arm
            arms[arm] = {
                "decks": _context_deck_summary(_read(arm_root / "context_decks.json")),
                "conditioning": _context_kl_summary(
                    _read(arm_root / "context_kl.json")
                ),
            }
            hybrid_path = arm_root / "context_hybrid.json"
            if hybrid_path.exists():
                arms[arm]["hybrids"] = _hybrid_summary(_read(hybrid_path))
        windows[str(epoch)] = {
            "prefix_vs_control_score": float(direct["summary"]["score"]),
            "prefix_vs_control_lcb80": float(direct["summary"]["paired_lcb_80"]),
            "prefix_vs_parent_score": float(candidate_parent["summary"]["score"]),
            "control_vs_parent_score": float(control_parent["summary"]["score"]),
            "parent_panel_delta": float(candidate_parent["summary"]["score"])
            - float(control_parent["summary"]["score"]),
            "prefix_heldout_score": float(candidate_heldout["summary"]["score"]),
            "control_heldout_score": float(control_heldout["summary"]["score"]),
            "heldout_delta": float(candidate_heldout["summary"]["score"])
            - float(control_heldout["summary"]["score"]),
            "water_heldout_delta": _weighted_gate_score(
                candidate_heldout, WATER_GATES
            ) - _weighted_gate_score(control_heldout, WATER_GATES),
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

    endpoint_root = evaluation / f"p{epochs[-1]}"
    opportunity = {
        arm: _opportunity_summary(_read(endpoint_root / arm / "opportunity.json"))
        for arm in ARMS
    }
    funnels = _card_funnel_comparison(
        _read(endpoint_root / "control_no_prefix" / "card_funnels.json"),
        _read(endpoint_root / "random_main_prefix" / "card_funnels.json"),
    )

    candidate_training = training["random_main_prefix"]
    control_training = training["control_no_prefix"]
    sps_ratio = float(candidate_training["median_sps"]) / max(
        float(control_training["median_sps"]), 1e-9
    )
    finite_signal = all(
        value is not None and math.isfinite(float(value))
        for value in (
            candidate_training["importance_mean_median"],
            candidate_training["clipfrac_max"],
        )
    )
    integrity_pass = (
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
        and candidate_training["prefix_episodes"] > 0
        and candidate_training["prefix_forced_rows"] > 0
        and candidate_training["delayed_forced_rows"] > 0
        and 0.7 <= candidate_training["prefix_mean_length_mean"] <= 1.3
        and all(
            value > 0
            for value in candidate_training["prefix_length_episodes"].values()
        )
        and control_training["prefix_forced_rows"] == 0
        and control_training["delayed_forced_rows"] == 0
        and candidate_training["incomplete_episodes"] == 0
        and finite_signal
        and max(window["timeout_rate_max"] for window in windows.values()) == 0.0
        and all(
            arm["conditioning"]["determinism_max"] < 1e-7
            for window in windows.values()
            for arm in (window["control_no_prefix"], window["random_main_prefix"])
        )
    )
    relative_sps_pass = sps_ratio >= 0.95
    absolute_sps_pass = float(candidate_training["median_sps"]) >= 1235.0
    throughput_raw_pass = relative_sps_pass and absolute_sps_pass
    if (
        bool(training_gate.get("relative_sps_pass")) != relative_sps_pass
        or bool(training_gate.get("absolute_sps_pass")) != absolute_sps_pass
    ):
        raise RuntimeError("Stage 3 raw SPS gate fields disagree with training data")

    late = [windows[str(epoch)] for epoch in epochs[-3:]]
    direct_late_mean = _mean([window["prefix_vs_control_score"] for window in late])
    parent_delta_late_mean = _mean([window["parent_panel_delta"] for window in late])
    heldout_delta_late_mean = _mean([window["heldout_delta"] for window in late])
    external_safety_pass = (
        direct_late_mean >= 0.47
        and min(window["parent_panel_delta"] for window in late) >= -0.05
        and min(window["heldout_delta"] for window in late) >= -0.05
    )
    endpoint = windows[str(epochs[-1])]
    battle = endpoint["battle_delta"]
    control_portal = opportunity["control_no_prefix"]["portal_per_game"]
    candidate_portal = opportunity["random_main_prefix"]["portal_per_game"]
    control_garden = opportunity["control_no_prefix"][
        "garden_share_when_comparable"
    ]
    candidate_garden = opportunity["random_main_prefix"][
        "garden_share_when_comparable"
    ]
    garden_delta = (
        float(candidate_garden) - float(control_garden)
        if candidate_garden is not None and control_garden is not None
        else 0.0
    )
    water_spell_delta = (
        endpoint["random_main_prefix"]["decks"]["water_spell_slots_mean"]
        - endpoint["control_no_prefix"]["decks"]["water_spell_slots_mean"]
    )
    deck_unique_delta = (
        endpoint["random_main_prefix"]["decks"]["stochastic_unique_mean"]
        - endpoint["control_no_prefix"]["decks"]["stochastic_unique_mean"]
    )
    quad_delta = (
        endpoint["random_main_prefix"]["decks"]["stochastic_quad_slot_share_mean"]
        - endpoint["control_no_prefix"]["decks"]["stochastic_quad_slot_share_mean"]
    )
    coverage_noncollapse = (
        funnels["coverage_delta"]["selected"] >= -5
        and funnels["coverage_delta"]["realized"] >= -5
    )
    mechanics_pass = (
        endpoint["water_heldout_delta"] >= -0.05
        and water_spell_delta >= -2.0
        and candidate_portal / max(control_portal, 1e-9) >= 0.90
        and garden_delta >= -0.05
        and deck_unique_delta >= -3.0
        and quad_delta <= 0.10
        and coverage_noncollapse
        and float(battle.get("attack_rate_mean", 0.0)) >= -0.03
        and float(battle.get("spell_rate_mean", 0.0)) >= -0.02
        and float(battle.get("play_entity_to_garden_rate_mean", 0.0)) >= -0.01
        and float(battle.get("garden_or_leader_ability_rate_mean", 0.0)) >= -0.02
    )
    strength_improvement = (
        endpoint["prefix_vs_control_score"] >= 0.53
        and endpoint["prefix_vs_control_lcb80"] >= 0.45
        and (parent_delta_late_mean >= 0.02 or heldout_delta_late_mean >= 0.02)
    )

    causal_windows = []
    for epoch in hybrid_epochs:
        window = windows[str(epoch)]
        candidate = window["random_main_prefix"]["hybrids"]["sibling_both_main"]
        control = window["control_no_prefix"]["hybrids"]["sibling_both_main"]
        candidate_advantage = float(candidate["matched_advantage"])
        control_advantage = float(control["matched_advantage"])
        delta = candidate_advantage - control_advantage
        gate_advantage = float(
            window["random_main_prefix"]["hybrids"]["sibling_gate_main"]
            ["matched_advantage"]
        )
        leader_advantage = float(
            window["random_main_prefix"]["hybrids"]["sibling_leader_main"]
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
        verdict = "accept_random_main_prefix"
        reason = (
            "Random prefixes clear integrity, external, and mechanics gates and "
            "produce a registered strength or sustained causal deck-fit gain. "
            "Throughput remains a separately reported engineering tradeoff."
        )
    elif not integrity_pass:
        verdict = "reject_integrity"
        reason = "The forced-prefix lifecycle or delayed-credit masking failed an invariant."
    elif not external_safety_pass:
        verdict = "reject_strength_regression"
        reason = "The prefix arm materially regressed on direct, parent, or heldout strength."
    elif not mechanics_pass:
        verdict = "diagnose_mechanics_or_deck_regression"
        reason = "The prefix arm failed a registered battle, Water, Garden, or deck safety gate."
    else:
        verdict = "reject_neutral_prefix"
        reason = (
            "The prefix arm is safe but lacks a greater-than-noise strength gain or "
            "repeatable causal deck/context improvement; exposure alone is not efficacy."
        )

    report = {
        "schema_version": 1,
        "campaign": args.root.name,
        "parent_epoch": int(manifest["parent_epoch"]),
        "target_epoch": int(manifest["target_epoch"]),
        "evaluation_epochs": epochs,
        "hybrid_epochs": hybrid_epochs,
        "promotion_used_as_signal": False,
        "prefix_disabled_during_evaluation": True,
        "training": training,
        "throughput": {
            "candidate_over_control": sps_ratio,
            "relative_floor": 0.95,
            "absolute_floor": 1235.0,
            "diagnostic_only": True,
            "raw_pass": throughput_raw_pass,
            "relative_pass": relative_sps_pass,
            "absolute_pass": absolute_sps_pass,
        },
        "windows": windows,
        "late_window": {
            "direct_mean": direct_late_mean,
            "parent_panel_delta_mean": parent_delta_late_mean,
            "heldout_delta_mean": heldout_delta_late_mean,
        },
        "opportunity": opportunity,
        "card_funnels": funnels,
        "endpoint_mechanics": {
            "water_spell_slot_delta": water_spell_delta,
            "portal_per_game_ratio": candidate_portal / max(control_portal, 1e-9),
            "garden_selection_delta": garden_delta,
            "stochastic_unique_delta": deck_unique_delta,
            "quad_slot_share_delta": quad_delta,
            "coverage_noncollapse": coverage_noncollapse,
        },
        "causal_windows": causal_windows,
        "decision": {
            "verdict": verdict,
            "reason": reason,
            "advance": advance,
            "integrity_pass": integrity_pass,
            "throughput_pass": True,
            "throughput_raw_pass": throughput_raw_pass,
            "external_safety_pass": external_safety_pass,
            "mechanics_pass": mechanics_pass,
            "strength_improvement_pass": strength_improvement,
            "causal_improvement_pass": causal_improvement,
            "efficacy_pass": efficacy_pass,
        },
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
