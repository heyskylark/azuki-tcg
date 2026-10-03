#!/usr/bin/env python3
"""Synthesize the Stage 4 fixed-floor versus exact late-zero ladder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from random_main_prefix_ladder_report import (
    WATER_GATES,
    _card_funnel_comparison,
    _context_deck_summary,
    _context_kl_summary,
    _hybrid_summary,
    _mean,
    _opportunity_summary,
    _read,
    _weighted_gate_score,
)


ARMS = ("fixed_floor", "late_zero")


def _battle_metrics(payload: dict, policy: str) -> dict[str, float]:
    raw = payload["summary"][policy]["battle_metrics"]
    return {
        key: float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float))
    }


def _battle_delta(candidate: dict[str, float], control: dict[str, float]) -> dict:
    return {
        key: float(candidate[key]) - float(control.get(key, 0.0))
        for key in candidate
    }


def _trend(epochs: list[int], values: list[float]) -> dict[str, float | list[float]]:
    if len(epochs) != len(values) or len(values) < 2:
        raise ValueError("trajectory trend requires at least two aligned values")
    x = np.asarray(epochs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    slopes = [
        (y[right] - y[left]) / (x[right] - x[left])
        for left in range(len(x))
        for right in range(left + 1, len(x))
    ]
    return {
        "epochs": epochs,
        "scores": [float(value) for value in y],
        "ols_slope_per_100_updates": float(np.polyfit(x, y, 1)[0] * 100.0),
        "theil_sen_slope_per_100_updates": float(np.median(slopes) * 100.0),
        "first_to_last_delta": float(y[-1] - y[0]),
        "mean": float(y.mean()),
        "minimum": float(y.min()),
        "maximum": float(y.max()),
        "mean_absolute_step": float(np.abs(np.diff(y)).mean()),
    }


def _markdown(report: dict) -> str:
    decision = report["decision"]
    lines = [
        "# Stage 4 Late Shaping Removal",
        "",
        f"Decision: **{decision['verdict']}**. Selected arm: "
        f"**{decision['selected_arm']}**.",
        "",
        decision["reason"],
        "",
        "True terminal draft credit remained enabled in both arms and was not "
        "multiplied by the shaping schedule.",
        "",
        "| Epoch | Multiplier | Late-zero vs floor | LCB80 | Phase |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for epoch in report["evaluation_epochs"]:
        point = report["windows"][str(epoch)]
        lines.append(
            f"| {epoch} | {point['late_zero_multiplier']:.3f} | "
            f"{point['direct_score']:.3f} | {point['direct_lcb80']:.3f} | "
            f"{point['phase']} |"
        )
    trend = report["zero_interval_trend"]
    lines.extend([
        "",
        "## Exact-Zero Interval",
        "",
        f"The exact-zero mean is `{trend['mean']:.3f}` and endpoint-minus-first "
        f"zero score is `{trend['first_to_last_delta']:+.3f}`. The robust slope "
        f"is `{trend['theil_sen_slope_per_100_updates']:+.4f}` score per 100 "
        f"updates (OLS `{trend['ols_slope_per_100_updates']:+.4f}`).",
        "",
        "| Epoch | Parent delta | Heldout delta | Water delta | Causal delta |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for epoch in report["anchor_epochs"]:
        point = report["windows"][str(epoch)]
        lines.append(
            f"| {epoch} | {point['parent_panel_delta']:+.3f} | "
            f"{point['heldout_delta']:+.3f} | "
            f"{point['water_heldout_delta']:+.3f} | "
            f"{point['causal']['candidate_minus_control']:+.3f} |"
        )
    lines.extend([
        "",
        "## Registered Gates",
        "",
        f"- Schedule and terminal-credit integrity: `{decision['integrity_pass']}`",
        f"- Throughput: `{decision['throughput_pass']}` "
        f"(`{report['throughput']['candidate_over_control']:.4f}` of control)",
        f"- External nonregression: `{decision['external_safety_pass']}`",
        f"- Exact-zero stability: `{decision['stability_pass']}`",
        f"- Mechanics and deck safety: `{decision['mechanics_pass']}`",
        f"- Causal context-fit safety: `{decision['causal_safety_pass']}`",
        f"- Clear zero-tail strength: `{decision['clear_strength_pass']}`",
        "",
        "A heldout movement within approximately one point is treated as neutral. "
        "A viable but neutral zero tail therefore keeps the `0.15` floor.",
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
    anchor_epochs = [int(value) for value in manifest["anchor_epochs"]]
    anneal_end = int(manifest["anneal_end_epoch"])
    training = {
        arm: _read(args.root / arm / "training_summary.json") for arm in ARMS
    }
    training_gate = _read(args.root / "training_gate.json")

    multiplier_by_epoch = {
        int(item["epoch"]): float(item["multiplier"])
        for item in training["late_zero"]["schedule"]
    }
    windows: dict[str, dict] = {}
    for epoch in epochs:
        root = evaluation / f"p{epoch}"
        direct = _read(root / "late_zero_vs_fixed_floor.json")
        candidate_battle = _battle_metrics(direct, "policy_a")
        control_battle = _battle_metrics(direct, "policy_b")
        point = {
            "phase": "exact_zero" if epoch >= anneal_end else "ramp",
            "late_zero_multiplier": multiplier_by_epoch[epoch],
            "direct_score": float(direct["summary"]["score"]),
            "direct_lcb80": float(direct["summary"]["paired_lcb_80"]),
            "timeout_rate": float(direct["summary"]["timeout_rate"]),
            "candidate_battle": candidate_battle,
            "control_battle": control_battle,
            "battle_delta": _battle_delta(candidate_battle, control_battle),
        }
        if epoch in anchor_epochs:
            candidate_parent = _read(root / "late_zero_vs_parent.json")
            control_parent = _read(root / "fixed_floor_vs_parent.json")
            candidate_heldout = _read(root / "late_zero" / "heldout.json")
            control_heldout = _read(root / "fixed_floor" / "heldout.json")
            point.update({
                "late_zero_vs_parent_score": float(
                    candidate_parent["summary"]["score"]
                ),
                "fixed_floor_vs_parent_score": float(
                    control_parent["summary"]["score"]
                ),
                "parent_panel_delta": float(candidate_parent["summary"]["score"])
                - float(control_parent["summary"]["score"]),
                "late_zero_heldout_score": float(
                    candidate_heldout["summary"]["score"]
                ),
                "fixed_floor_heldout_score": float(
                    control_heldout["summary"]["score"]
                ),
                "heldout_delta": float(candidate_heldout["summary"]["score"])
                - float(control_heldout["summary"]["score"]),
                "water_heldout_delta": _weighted_gate_score(
                    candidate_heldout, WATER_GATES
                ) - _weighted_gate_score(control_heldout, WATER_GATES),
                "late_zero": {
                    "decks": _context_deck_summary(
                        _read(root / "late_zero" / "context_decks.json")
                    ),
                    "conditioning": _context_kl_summary(
                        _read(root / "late_zero" / "context_kl.json")
                    ),
                    "hybrids": _hybrid_summary(
                        _read(root / "late_zero" / "context_hybrid.json")
                    ),
                },
                "fixed_floor": {
                    "decks": _context_deck_summary(
                        _read(root / "fixed_floor" / "context_decks.json")
                    ),
                    "conditioning": _context_kl_summary(
                        _read(root / "fixed_floor" / "context_kl.json")
                    ),
                    "hybrids": _hybrid_summary(
                        _read(root / "fixed_floor" / "context_hybrid.json")
                    ),
                },
            })
            candidate_hybrid = point["late_zero"]["hybrids"]["sibling_both_main"]
            control_hybrid = point["fixed_floor"]["hybrids"]["sibling_both_main"]
            candidate_advantage = float(candidate_hybrid["matched_advantage"])
            control_advantage = float(control_hybrid["matched_advantage"])
            point["causal"] = {
                "candidate_matched_advantage": candidate_advantage,
                "control_matched_advantage": control_advantage,
                "candidate_minus_control": candidate_advantage - control_advantage,
                "candidate_ci80": candidate_hybrid["matched_advantage_ci80"],
            }
        windows[str(epoch)] = point

    endpoint_root = evaluation / f"p{epochs[-1]}"
    opportunity = {
        arm: _opportunity_summary(_read(endpoint_root / arm / "opportunity.json"))
        for arm in ARMS
    }
    funnels = _card_funnel_comparison(
        _read(endpoint_root / "fixed_floor" / "card_funnels.json"),
        _read(endpoint_root / "late_zero" / "card_funnels.json"),
    )

    zero_scores = [windows[str(epoch)]["direct_score"] for epoch in anchor_epochs]
    zero_trend = _trend(anchor_epochs, zero_scores)
    parent_delta_trend = _trend(
        anchor_epochs,
        [windows[str(epoch)]["parent_panel_delta"] for epoch in anchor_epochs],
    )
    heldout_delta_trend = _trend(
        anchor_epochs,
        [windows[str(epoch)]["heldout_delta"] for epoch in anchor_epochs],
    )
    sps_ratio = float(training["late_zero"]["median_sps"]) / max(
        float(training["fixed_floor"]["median_sps"]), 1e-9
    )
    integrity_pass = (
        bool(training_gate["integrity_pass"])
        and bool(training["fixed_floor"]["schedule_integrity"])
        and bool(training["late_zero"]["schedule_integrity"])
        and int(training["late_zero"]["zero_update_count"])
        == int(args.root.joinpath("campaign_config.txt").read_text().split(
            "zero_updates=", 1
        )[1].splitlines()[0])
        and float(training["late_zero"]["zero_tail_credit_examples"]) > 0.0
        and float(training["late_zero"]["zero_tail_credit_labels"]) > 0.0
        and all(windows[str(epoch)]["timeout_rate"] == 0.0 for epoch in epochs)
        and all(
            windows[str(epoch)][arm]["conditioning"]["determinism_max"] < 1e-7
            for epoch in anchor_epochs
            for arm in ARMS
        )
    )
    throughput_pass = (
        bool(training_gate["performance_pass"])
        and sps_ratio >= 0.95
        and float(training["late_zero"]["median_sps"]) >= 1235.0
    )
    external_safety_pass = (
        zero_trend["mean"] >= 0.47
        and zero_trend["minimum"] >= 0.44
        and windows[str(anchor_epochs[-1])]["direct_score"] >= 0.47
        and min(
            windows[str(epoch)]["parent_panel_delta"] for epoch in anchor_epochs
        ) >= -0.05
        and min(
            windows[str(epoch)]["heldout_delta"] for epoch in anchor_epochs
        ) >= -0.05
    )
    stability_pass = (
        zero_trend["theil_sen_slope_per_100_updates"] >= -0.005
        and zero_trend["first_to_last_delta"] >= -0.03
        and parent_delta_trend["first_to_last_delta"] >= -0.04
        and heldout_delta_trend["first_to_last_delta"] >= -0.04
    )

    mechanics_windows = []
    for epoch in anchor_epochs:
        point = windows[str(epoch)]
        candidate_decks = point["late_zero"]["decks"]
        control_decks = point["fixed_floor"]["decks"]
        candidate_battle = point["candidate_battle"]
        control_battle = point["control_battle"]
        mechanics_windows.append({
            "epoch": epoch,
            "water_spell_slot_delta": candidate_decks["water_spell_slots_mean"]
            - control_decks["water_spell_slots_mean"],
            "unique_delta": candidate_decks["stochastic_unique_mean"]
            - control_decks["stochastic_unique_mean"],
            "quad_slot_delta": candidate_decks["stochastic_quad_slot_share_mean"]
            - control_decks["stochastic_quad_slot_share_mean"],
            "portal_ratio": candidate_battle.get("portal_rate_mean", 0.0)
            / max(control_battle.get("portal_rate_mean", 0.0), 1e-9),
            "game_length_ratio": candidate_battle.get("episode_length_mean", 0.0)
            / max(control_battle.get("episode_length_mean", 0.0), 1e-9),
            "attack_delta": point["battle_delta"].get("attack_rate_mean", 0.0),
            "spell_delta": point["battle_delta"].get("spell_rate_mean", 0.0),
            "garden_play_delta": point["battle_delta"].get(
                "play_entity_to_garden_rate_mean", 0.0
            ),
            "ability_delta": point["battle_delta"].get(
                "garden_or_leader_ability_rate_mean", 0.0
            ),
        })
    control_portal = opportunity["fixed_floor"]["portal_per_game"]
    candidate_portal = opportunity["late_zero"]["portal_per_game"]
    control_garden = opportunity["fixed_floor"]["garden_share_when_comparable"]
    candidate_garden = opportunity["late_zero"]["garden_share_when_comparable"]
    garden_delta = (
        float(candidate_garden) - float(control_garden)
        if candidate_garden is not None and control_garden is not None
        else 0.0
    )
    mechanics_pass = (
        all(windows[str(epoch)]["water_heldout_delta"] >= -0.05 for epoch in anchor_epochs)
        and all(item["water_spell_slot_delta"] >= -2.0 for item in mechanics_windows)
        and all(item["unique_delta"] >= -2.0 for item in mechanics_windows)
        and all(item["quad_slot_delta"] <= 0.05 for item in mechanics_windows)
        and all(item["portal_ratio"] >= 0.90 for item in mechanics_windows)
        and all(item["game_length_ratio"] <= 1.15 for item in mechanics_windows)
        and all(item["attack_delta"] >= -0.04 for item in mechanics_windows)
        and all(item["spell_delta"] >= -0.03 for item in mechanics_windows)
        and all(item["garden_play_delta"] >= -0.015 for item in mechanics_windows)
        and all(item["ability_delta"] >= -0.025 for item in mechanics_windows)
        and candidate_portal / max(control_portal, 1e-9) >= 0.90
        and garden_delta >= -0.05
        and funnels["coverage_delta"]["selected"] >= -5
        and funnels["coverage_delta"]["realized"] >= -5
    )
    causal_safety_pass = all(
        windows[str(epoch)]["causal"]["candidate_minus_control"] >= -0.05
        and windows[str(epoch)]["causal"]["candidate_matched_advantage"] >= -0.05
        for epoch in anchor_epochs
    )
    endpoint = windows[str(anchor_epochs[-1])]
    parent_delta_mean = _mean([
        windows[str(epoch)]["parent_panel_delta"] for epoch in anchor_epochs
    ])
    heldout_delta_mean = _mean([
        windows[str(epoch)]["heldout_delta"] for epoch in anchor_epochs
    ])
    clear_strength_pass = (
        zero_trend["mean"] >= 0.52
        and endpoint["direct_score"] >= 0.52
        and endpoint["direct_lcb80"] >= 0.48
        and zero_trend["theil_sen_slope_per_100_updates"] >= 0.0
        and (parent_delta_mean >= 0.015 or heldout_delta_mean >= 0.015)
    )
    zero_viable = all((
        integrity_pass,
        throughput_pass,
        external_safety_pass,
        stability_pass,
        mechanics_pass,
        causal_safety_pass,
    ))
    if zero_viable and clear_strength_pass:
        verdict = "adopt_late_zero"
        selected_arm = "late_zero"
        reason = (
            "The exact-zero tail is stable and safe and shows a clear registered "
            "strength gain, so late zero becomes the selected recipe."
        )
    elif zero_viable:
        verdict = "zero_viable_neutral_keep_floor"
        selected_arm = "fixed_floor"
        reason = (
            "The exact-zero tail is stable and nonregressive but does not show a "
            "clear gain; the preregistered neutral-result rule retains the 0.15 floor."
        )
    elif not integrity_pass:
        verdict = "reject_integrity"
        selected_arm = "fixed_floor"
        reason = "Schedule, exact-zero, terminal-credit, or determinism integrity failed."
    elif not throughput_pass:
        verdict = "reject_throughput"
        selected_arm = "fixed_floor"
        reason = "Late zero failed the 95% relative or 1,235 absolute SPS guard."
    elif not external_safety_pass or not stability_pass:
        verdict = "reject_unstable_or_regressive"
        selected_arm = "fixed_floor"
        reason = "The exact-zero trajectory was unstable or materially regressive."
    else:
        verdict = "reject_mechanics_or_context_regression"
        selected_arm = "fixed_floor"
        reason = "Late zero failed a mechanics, deck, or causal context-fit safety gate."

    report = {
        "schema_version": 1,
        "campaign": args.root.name,
        "parent_epoch": int(manifest["parent_epoch"]),
        "target_epoch": int(manifest["target_epoch"]),
        "anneal_end_epoch": anneal_end,
        "evaluation_epochs": epochs,
        "anchor_epochs": anchor_epochs,
        "promotion_used_as_signal": False,
        "terminal_credit_annealed": False,
        "training": training,
        "training_gate": training_gate,
        "throughput": {
            "candidate_over_control": sps_ratio,
            "relative_floor": 0.95,
            "absolute_floor": 1235.0,
        },
        "windows": windows,
        "zero_interval_trend": zero_trend,
        "parent_delta_trend": parent_delta_trend,
        "heldout_delta_trend": heldout_delta_trend,
        "opportunity": opportunity,
        "card_funnels": funnels,
        "mechanics_windows": mechanics_windows,
        "decision": {
            "verdict": verdict,
            "selected_arm": selected_arm,
            "reason": reason,
            "zero_viable": zero_viable,
            "integrity_pass": integrity_pass,
            "throughput_pass": throughput_pass,
            "external_safety_pass": external_safety_pass,
            "stability_pass": stability_pass,
            "mechanics_pass": mechanics_pass,
            "causal_safety_pass": causal_safety_pass,
            "clear_strength_pass": clear_strength_pass,
        },
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
