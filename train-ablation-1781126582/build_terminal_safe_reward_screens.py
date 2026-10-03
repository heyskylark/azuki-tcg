#!/usr/bin/env python3
"""Build terminal-safe reward screens, follow-ups, and selected 45M qualifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_shaping_screens import ROOT, set_ini_value, sha256
from strategy_descriptor import SCHEMA_VERSION


def _registration_decision(path: Path) -> dict[str, object]:
    packet = json.loads(path.read_text())
    decision = packet.get("decision", {})
    state = f"{decision.get('status', '')} {decision.get('disposition', '')}".lower()
    if (
        packet.get("strategy_descriptor_schema_version") != SCHEMA_VERSION
        or "selected_arm" not in decision
        or decision.get("reward_foundation_qualified") is False
        or (decision.get("selected_arm") is None and decision.get("status") != "rejected")
        or any(word in state for word in ("superseded", "unqualified", "requalification", "hold", "blocked"))
    ):
        raise ValueError(f"parent decision is not current authorization: {path}")
    return packet


TEMPLATE = ROOT / "python/config/azuki_reward_screen_r1_no_tempo_15m.ini"
RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_screens"
FOLLOWUP_RESULTS = ROOT / "train-ablation-1781126582/results/terminal_safe_reward_followup"
LATE_FOLLOWUP_RESULTS = (
    ROOT / "train-ablation-1781126582/results/terminal_safe_reward_tail_followup"
)
EXPLORATION_FOLLOWUP_RESULTS = (
    ROOT / "train-ablation-1781126582/results/terminal_safe_reward_exploration_followup"
)
HORIZON_FOLLOWUP_RESULTS = (
    ROOT / "train-ablation-1781126582/results/terminal_safe_reward_horizon_followup"
)
SCREEN_ROWS = 15_006_720
QUALIFIER_ROWS = 45_004_800
SCREEN_EVALUATION_UPDATES = (325, 650, 975)
QUALIFIER_EVALUATION_UPDATES = (325, 975, 1950, 2925)

ARM_SPECS = {
    "R5": {
        "slug": "r5_resource_only_closed",
        "description": "Proper terminal-closed PBRS with readiness/resource Phi only.",
        "potential_weights": {
            "leader_health": 0.0,
            "garden_attack": 0.0,
            "untapped_garden": 0.15,
            "untapped_ikz": 0.15,
        },
    },
    "R6": {
        "slug": "r6_no_state_potential",
        "description": "Proper terminal-closed PBRS with Phi identically zero.",
        "potential_weights": {
            "leader_health": 0.0,
            "garden_attack": 0.0,
            "untapped_garden": 0.0,
            "untapped_ikz": 0.0,
        },
    },
}

FOLLOWUP_ARM_SPECS = {
    "R7": {
        "slug": "r7_half_combat_potential_closed",
        "description": "Terminal-closed PBRS with half-strength combat Phi and full readiness/resource Phi.",
        "potential_weights": {
            "leader_health": 2.0,
            "garden_attack": 0.35,
            "untapped_garden": 0.15,
            "untapped_ikz": 0.15,
        },
    },
    "R8": {
        "slug": "r8_full_combat_potential_closed",
        "description": "Terminal-closed PBRS with full combat and readiness/resource Phi.",
        "potential_weights": {
            "leader_health": 4.0,
            "garden_attack": 0.7,
            "untapped_garden": 0.15,
            "untapped_ikz": 0.15,
        },
    },
}
LATE_FOLLOWUP_ARM_SPECS = {
    "R9": {
        "slug": "r9_half_combat_potential_tail_030",
        "description": "R7 terminal-closed Phi with a 0.30 sustained potential tail.",
        "potential_tail": 0.30,
        "potential_weights": FOLLOWUP_ARM_SPECS["R7"]["potential_weights"],
    },
    "R10": {
        "slug": "r10_half_combat_potential_tail_050",
        "description": "R7 terminal-closed Phi with a 0.50 sustained potential tail.",
        "potential_tail": 0.50,
        "potential_weights": FOLLOWUP_ARM_SPECS["R7"]["potential_weights"],
    },
}
EXPLORATION_FOLLOWUP_ARM_SPECS = {
    "R11": {
        "slug": "r11_potential_030_exploration_tail_030",
        "description": "R9 terminal-closed Phi with a 0.30 sustained exploration tail.",
        "potential_tail": 0.30,
        "exploration_tail": 0.30,
        "potential_weights": FOLLOWUP_ARM_SPECS["R7"]["potential_weights"],
    },
    "R12": {
        "slug": "r12_potential_030_exploration_tail_050",
        "description": "R9 terminal-closed Phi with a 0.50 sustained exploration tail.",
        "potential_tail": 0.30,
        "exploration_tail": 0.50,
        "potential_weights": FOLLOWUP_ARM_SPECS["R7"]["potential_weights"],
    },
}
HORIZON_FOLLOWUP_ARM_SPECS = {
    "R13": {
        "slug": "r13_tail_030_anneal_075",
        "description": "R9 terminal-closed Phi annealed over 75% of sampled rows.",
        "potential_tail": 0.30,
        "anneal_fraction": 0.75,
        "potential_weights": FOLLOWUP_ARM_SPECS["R7"]["potential_weights"],
    },
    "R14": {
        "slug": "r14_tail_030_anneal_100",
        "description": "R9 terminal-closed Phi annealed over all sampled rows.",
        "potential_tail": 0.30,
        "anneal_fraction": 1.0,
        "potential_weights": FOLLOWUP_ARM_SPECS["R7"]["potential_weights"],
    },
}










def _build_arm(
    template: str,
    arm_id: str,
    *,
    sampled_rows: int,
    phase: str,
    arm_specs: dict[str, dict[str, object]] = ARM_SPECS,
    results: Path = RESULTS,
) -> dict[str, object]:
    spec = arm_specs[arm_id]
    slug = str(spec["slug"])
    suffix = "15m" if phase == "screen" else "45m"
    run_name = f"{slug}_{suffix}"
    result_root = results / phase / run_name
    runtime = result_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "league_state.json"
    config_path = ROOT / f"python/config/azuki_reward_{run_name}.ini"
    anneal_fraction = float(spec.get("anneal_fraction", 0.5))
    anneal_end_rows = int(sampled_rows * anneal_fraction)
    weights = spec["potential_weights"]
    potential_tail = float(spec.get("potential_tail", 0.15))
    exploration_tail = float(spec.get("exploration_tail", 0.15))
    replacements = {
        ("base", "jsonl_log"): (result_root / "logs/production.jsonl").relative_to(ROOT),
        ("base", "tag"): f"reward_{run_name}",
        ("env", "pbrs_mode"): "discounted",
        ("env", "pbrs_gamma"): 0.99,
        ("env", "pbrs_terminal_closure"): "true",
        ("env", "reward_decomposed_schedule"): "true",
        ("league", "state_path"): state_path.relative_to(ROOT),
        ("league", "promotion_state_path"): (runtime / "league_state_promotion.json").relative_to(ROOT),
        ("league", "promotion_records_dir"): (runtime / "promotion_records").relative_to(ROOT),
        ("league", "opponent_dir"): (runtime / "opponents").relative_to(ROOT),
        ("train", "total_timesteps"): sampled_rows,
        ("process_env", "azk_reward_leader_health_weight"): weights["leader_health"],
        ("process_env", "azk_reward_garden_attack_weight"): weights["garden_attack"],
        ("process_env", "azk_reward_untapped_garden_weight"): weights["untapped_garden"],
        ("process_env", "azk_reward_untapped_ikz_weight"): weights["untapped_ikz"],
        ("process_env", "azk_reward_leader_delta_weight"): 0,
        ("process_env", "azk_reward_board_delta_weight"): 0,
        ("process_env", "azk_early_tempo_bonus"): 0,
        ("process_env", "azk_reward_shaping_anneal"): 0,
        ("process_env", "azk_reward_decomposed_schedule"): 1,
        ("process_env", "azk_potential_scale_initial"): 1.0,
        ("process_env", "azk_potential_scale_final"): potential_tail,
        ("process_env", "azk_potential_anneal_start_rows"): 0,
        ("process_env", "azk_exploration_scale_initial"): 1.0,
        ("process_env", "azk_exploration_scale_final"): exploration_tail,
        ("process_env", "azk_exploration_anneal_start_rows"): 0,
        ("process_env", "azk_potential_anneal_end_rows"): anneal_end_rows,
        ("process_env", "azk_exploration_anneal_end_rows"): anneal_end_rows,
    }
    text = template
    for (section, key), value in replacements.items():
        text = set_ini_value(text, section, key, value)
    text = text.replace(
        "; 977 complete updates x 15,360 sampled rows = 15,006,720.",
        f"; {sampled_rows // 15_360} complete updates x 15,360 sampled rows = {sampled_rows:,}.",
    )
    config_path.write_text(text)
    return {
        "id": arm_id,
        "name": run_name,
        "description": spec["description"],
        "phase": phase,
        "tag": f"reward_{run_name}",
        "result_root": str(result_root.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "pbrs_mode": "discounted",
        "pbrs_gamma": 0.99,
        "pbrs_terminal_closure": True,
        "potential_weights": weights,
        "direct_leader_weight": 0.0,
        "direct_board_weight": 0.0,
        "early_tempo_bonus": 0.0,
        "potential_tail": potential_tail,
        "exploration_tail": exploration_tail,
        "anneal_end_rows": anneal_end_rows,
        "anneal_fraction": anneal_fraction,
        "initial_policy_count": 0,
    }


def _shared_contract(sampled_rows: int, evaluation_updates: tuple[int, ...]) -> dict[str, object]:
    return {
        "fresh_initialization": True,
        "matched_empty_initial_league": True,
        "r1_nonreward_config_unchanged": True,
        "draft_credit_schedule": "unchanged_r1",
        "sampled_rows": sampled_rows,
        "evaluation_updates": list(evaluation_updates),
        "pbrs_mode": "discounted",
        "pbrs_gamma": 0.99,
        "pbrs_terminal_closure": True,
        "direct_leader_weight": 0.0,
        "direct_board_weight": 0.0,
        "early_tempo_bonus": 0.0,
        "terminal_and_draft_credit_unscaled": True,
        "reward_component_reconstruction_required": True,
        "no_entity_or_leader_attack_reward_added": True,
    }

def _write_registration(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument(
        "--qualifier",
        choices=tuple(
            ARM_SPECS
            | FOLLOWUP_ARM_SPECS
            | LATE_FOLLOWUP_ARM_SPECS
            | EXPLORATION_FOLLOWUP_ARM_SPECS
            | HORIZON_FOLLOWUP_ARM_SPECS
        ),
        help="After the 15M decision, register a fresh 45M qualifier for the selected arm.",
    )
    choices.add_argument(
        "--followup",
        action="store_true",
        help="Register the matched R7/R8 combat-potential attenuation follow-up.",
    )
    choices.add_argument(
        "--late-followup",
        action="store_true",
        help="Register the matched R9/R10 sustained-potential-tail follow-up.",
    )
    choices.add_argument(
        "--exploration-followup",
        action="store_true",
        help="Register the matched R11/R12 sustained-exploration-tail follow-up.",
    )
    choices.add_argument(
        "--horizon-followup",
        action="store_true",
        help="Register the matched R13/R14 anneal-horizon follow-up.",
    )
    choices.add_argument(
        "--horizon-fallback",
        action="store_true",
        help="Register the preregistered R14 45M fallback after R13 rejection.",
    )
    args = parser.parse_args()

    template = TEMPLATE.read_text()
    if args.horizon_fallback:
        decision_path = HORIZON_FOLLOWUP_RESULTS / "qualifier_decision_packet.json"
        decision = _registration_decision(decision_path)
        if decision.get("decision", {}).get("selected_arm") != "R14":
            raise ValueError("R14 fallback requires selection by the R13 decision")
        fallback_arm = _build_arm(
            template,
            "R14",
            sampled_rows=QUALIFIER_ROWS,
            phase="fallback_qualifier",
            arm_specs=HORIZON_FOLLOWUP_ARM_SPECS,
            results=HORIZON_FOLLOWUP_RESULTS,
        )
        fallback_registration = {
            "schema_id": "azuki.ablation_registration",
            "schema_version": 1,
            "family": "terminal_safe_reward_r14_45m_fallback_qualifier",
            "control_arm_id": "R14",
            "sampled_rows_per_arm": QUALIFIER_ROWS,
            "seed": 42,
            "selected_from": str(decision_path.relative_to(ROOT)),
            "selected_from_sha256": sha256(decision_path),
            "single_changed_boundary": "duration_only_from_r14_15m_recipe",
            "shared_contract": _shared_contract(
                QUALIFIER_ROWS,
                QUALIFIER_EVALUATION_UPDATES,
            ),
            "qualification_rules": [
                "All integrity and exact reward-contract checks must pass at every retained window.",
                "Strength must persist through the final window without post-midpoint forgetting.",
                "Reject reduced corrected sampled element-sequence breadth, worse conditional targeting, collapsed sampled sibling differentiation, or incomplete curated evaluation.",
                "This is the final reward-schedule fallback; failure leaves reward finalization blocked rather than opening another unregistered recipe dimension.",
            ],
            "arms": [fallback_arm],
        }
        _write_registration(
            HORIZON_FOLLOWUP_RESULTS / "fallback_qualifier_registration.json",
            fallback_registration,
        )
        return

    if args.horizon_followup:
        decision_path = EXPLORATION_FOLLOWUP_RESULTS / "decision_packet.json"
        decision = _registration_decision(decision_path)
        if decision.get("decision", {}).get("selected_arm") is not None:
            raise ValueError("R13/R14 follow-up requires rejected R11/R12")
        r9_qualifier_decision_path = (
            LATE_FOLLOWUP_RESULTS / "qualifier_decision_packet.json"
        )
        r9_qualifier_decision = _registration_decision(r9_qualifier_decision_path)
        r9_endpoint = r9_qualifier_decision["arm"]["windows"][-1]
        horizon_arms = [
            _build_arm(
                template,
                arm_id,
                sampled_rows=SCREEN_ROWS,
                phase="screen",
                arm_specs=HORIZON_FOLLOWUP_ARM_SPECS,
                results=HORIZON_FOLLOWUP_RESULTS,
            )
            for arm_id in HORIZON_FOLLOWUP_ARM_SPECS
        ]
        shared_contract = _shared_contract(
            SCREEN_ROWS,
            SCREEN_EVALUATION_UPDATES,
        )
        shared_contract.update(
            {
                "combat_and_resource_potential_weights": FOLLOWUP_ARM_SPECS["R7"][
                    "potential_weights"
                ],
                "potential_tail": 0.30,
                "exploration_tail": 0.15,
                "anneal_fraction_is_only_treatment": True,
            }
        )
        horizon_registration = {
            "schema_id": "azuki.ablation_registration",
            "schema_version": 1,
            "family": "terminal_safe_reward_r13_r14_horizon_followup",
            "control_arm_id": "R9",
            "sampled_rows_per_arm": SCREEN_ROWS,
            "seed": 42,
            "parent_arm": "R9",
            "parent_recipe": (
                "python/config/"
                "azuki_reward_r9_half_combat_potential_tail_030_45m.ini"
            ),
            "parent_decision": str(decision_path.relative_to(ROOT)),
            "parent_decision_sha256": sha256(decision_path),
            "parent_qualifier_decision": str(
                r9_qualifier_decision_path.relative_to(ROOT)
            ),
            "parent_qualifier_decision_sha256": sha256(
                r9_qualifier_decision_path
            ),
            "reused_control": {
                "id": "R9",
                "checkpoint": r9_endpoint["checkpoint"],
                "checkpoint_sha256": r9_endpoint["checkpoint_sha256"],
                "evaluation_index": str(
                    (
                        LATE_FOLLOWUP_RESULTS / "qualifier_evaluation_index.json"
                    ).relative_to(ROOT)
                ),
            },
            "single_changed_boundary": "potential_and_exploration_anneal_horizon",
            "shared_contract": shared_contract,
            "selection_rules": [
                "Reject any arm with integrity, reconstruction, terminal-closure, timeout, or incomplete-credit failure.",
                "R13 and R14 differ only in the shared potential/exploration anneal endpoint, 75% versus 100% of sampled rows; R9 weights, tails, direct deltas, draft credit, and initialization remain matched.",
                "Require external strength in the established R1/R4/S4 range and complete corrected sampled sequence breadth across all four elements at the final window.",
                "Require conditional targeting and sampled sibling differentiation to remain at least as healthy as the R9 45M endpoint.",
                "Prefer R13 if the shorter horizon matches R14; advance only one arm to a fresh 45M qualifier.",
            ],
            "arms": horizon_arms,
        }
        _write_registration(
            HORIZON_FOLLOWUP_RESULTS / "registration.json",
            horizon_registration,
        )
        return

    if args.exploration_followup:
        decision_path = LATE_FOLLOWUP_RESULTS / "qualifier_decision_packet.json"
        decision = _registration_decision(decision_path)
        if decision.get("decision", {}).get("selected_arm") is not None:
            raise ValueError("R11/R12 follow-up requires rejected R9 qualification")
        r9_endpoint = decision["arm"]["windows"][-1]
        exploration_arms = [
            _build_arm(
                template,
                arm_id,
                sampled_rows=SCREEN_ROWS,
                phase="screen",
                arm_specs=EXPLORATION_FOLLOWUP_ARM_SPECS,
                results=EXPLORATION_FOLLOWUP_RESULTS,
            )
            for arm_id in EXPLORATION_FOLLOWUP_ARM_SPECS
        ]
        shared_contract = _shared_contract(
            SCREEN_ROWS,
            SCREEN_EVALUATION_UPDATES,
        )
        shared_contract.update(
            {
                "combat_and_resource_potential_weights": FOLLOWUP_ARM_SPECS["R7"][
                    "potential_weights"
                ],
                "potential_tail": 0.30,
                "exploration_tail_is_only_treatment": True,
            }
        )
        exploration_registration = {
            "schema_id": "azuki.ablation_registration",
            "schema_version": 1,
            "family": "terminal_safe_reward_r11_r12_exploration_followup",
            "control_arm_id": "R9",
            "sampled_rows_per_arm": SCREEN_ROWS,
            "seed": 42,
            "parent_arm": "R9",
            "parent_recipe": str(
                (
                    ROOT
                    / "python/config/azuki_reward_r9_half_combat_potential_tail_030_45m.ini"
                ).relative_to(ROOT)
            ),
            "parent_decision": str(decision_path.relative_to(ROOT)),
            "parent_decision_sha256": sha256(decision_path),
            "reused_control": {
                "id": "R9",
                "checkpoint": r9_endpoint["checkpoint"],
                "checkpoint_sha256": r9_endpoint["checkpoint_sha256"],
                "evaluation_index": str(
                    (
                        LATE_FOLLOWUP_RESULTS / "qualifier_evaluation_index.json"
                    ).relative_to(ROOT)
                ),
            },
            "single_changed_boundary": "terminal_exploration_scale_tail",
            "shared_contract": shared_contract,
            "selection_rules": [
                "Reject any arm with integrity, reconstruction, terminal-closure, timeout, or incomplete-credit failure.",
                "R11 and R12 differ only in the terminal exploration tail, 0.30 versus 0.50; R9 potential weights, 0.30 potential tail, direct deltas, draft credit, initialization, and anneal horizon remain matched.",
                "Require external strength at least in the established R1/R4/S4 range and complete corrected sampled sequence breadth across all four elements after annealing.",
                "Require conditional targeting and sampled sibling differentiation to remain at least as healthy as the R9 45M endpoint.",
                "Prefer R11 if the 0.30 exploration tail matches R12; advance only one arm to a fresh 45M qualifier.",
            ],
            "arms": exploration_arms,
        }
        _write_registration(
            EXPLORATION_FOLLOWUP_RESULTS / "registration.json",
            exploration_registration,
        )
        return

    if args.late_followup:
        decision_path = FOLLOWUP_RESULTS / "qualifier_decision_packet.json"
        decision = _registration_decision(decision_path)
        if decision.get("decision", {}).get("selected_arm") is not None:
            raise ValueError("R9/R10 follow-up requires rejected R7 qualification")
        r7_endpoint = decision["arm"]["windows"][-1]
        late_arms = [
            _build_arm(
                template,
                arm_id,
                sampled_rows=SCREEN_ROWS,
                phase="screen",
                arm_specs=LATE_FOLLOWUP_ARM_SPECS,
                results=LATE_FOLLOWUP_RESULTS,
            )
            for arm_id in LATE_FOLLOWUP_ARM_SPECS
        ]
        shared_contract = _shared_contract(
            SCREEN_ROWS,
            SCREEN_EVALUATION_UPDATES,
        )
        shared_contract.update(
            {
                "combat_and_resource_potential_weights": FOLLOWUP_ARM_SPECS["R7"][
                    "potential_weights"
                ],
                "exploration_tail": 0.15,
                "potential_tail_is_only_treatment": True,
            }
        )
        late_registration = {
            "schema_id": "azuki.ablation_registration",
            "schema_version": 1,
            "family": "terminal_safe_reward_r9_r10_tail_followup",
            "control_arm_id": "R7",
            "sampled_rows_per_arm": SCREEN_ROWS,
            "seed": 42,
            "parent_arm": "R7",
            "parent_recipe": str(
                (
                    ROOT
                    / "python/config/azuki_reward_r7_half_combat_potential_closed_45m.ini"
                ).relative_to(ROOT)
            ),
            "parent_decision": str(decision_path.relative_to(ROOT)),
            "parent_decision_sha256": sha256(decision_path),
            "reused_control": {
                "id": "R7",
                "checkpoint": r7_endpoint["checkpoint"],
                "checkpoint_sha256": r7_endpoint["checkpoint_sha256"],
                "evaluation_index": str(
                    (FOLLOWUP_RESULTS / "qualifier_evaluation_index.json").relative_to(
                        ROOT
                    )
                ),
            },
            "single_changed_boundary": "terminal_potential_scale_tail",
            "shared_contract": shared_contract,
            "selection_rules": [
                "Reject any arm with integrity, reconstruction, terminal-closure, timeout, or incomplete-credit failure.",
                "R9 and R10 differ only in the terminal potential tail, 0.30 versus 0.50; R7 weights, exploration tail, direct deltas, draft credit, initialization, and anneal horizon remain matched.",
                "Require external strength at least in the established R1/R4/S4 range and no post-anneal reduction in corrected sampled four-element sequence breadth.",
                "Require conditional face-versus-entity targeting and sampled sibling differentiation to remain at least as healthy as the R7 45M endpoint.",
                "Prefer R9 if the 0.30 tail matches R10; advance only one arm to a fresh 45M qualifier.",
            ],
            "arms": late_arms,
        }
        _write_registration(
            LATE_FOLLOWUP_RESULTS / "registration.json",
            late_registration,
        )
        return

    if args.followup:
        decision_path = RESULTS / "decision_packet.json"
        decision = _registration_decision(decision_path)
        if decision.get("decision", {}).get("selected_arm") is not None:
            raise ValueError("R7/R8 follow-up requires no selected R5/R6 arm")
        r5_endpoint = decision["arms"]["R5"]["windows"][-1]
        followup_arms = [
            _build_arm(
                template,
                arm_id,
                sampled_rows=SCREEN_ROWS,
                phase="screen",
                arm_specs=FOLLOWUP_ARM_SPECS,
                results=FOLLOWUP_RESULTS,
            )
            for arm_id in FOLLOWUP_ARM_SPECS
        ]
        followup_registration = {
            "schema_id": "azuki.ablation_registration",
            "schema_version": 1,
            "family": "terminal_safe_reward_r7_r8_followup",
            "control_arm_id": "R5",
            "sampled_rows_per_arm": SCREEN_ROWS,
            "seed": 42,
            "parent_arm": "R5",
            "parent_recipe": str(TEMPLATE.relative_to(ROOT)),
            "parent_recipe_sha256": sha256(TEMPLATE),
            "parent_decision": str(decision_path.relative_to(ROOT)),
            "parent_decision_sha256": sha256(decision_path),
            "reused_control": {
                "id": "R5",
                "checkpoint": r5_endpoint["checkpoint"],
                "checkpoint_sha256": r5_endpoint["checkpoint_sha256"],
                "evaluation_index": str(
                    (RESULTS / "evaluation_index.json").relative_to(ROOT)
                ),
            },
            "single_changed_boundary": "combat_state_phi_amplitude",
            "shared_contract": _shared_contract(
                SCREEN_ROWS,
                SCREEN_EVALUATION_UPDATES,
            ),
            "selection_rules": [
                "Reject any arm with integrity, reconstruction, terminal-closure, timeout, or incomplete-credit failure.",
                "R7 and R8 differ only in combat-state Phi amplitude; readiness/resource Phi, direct deltas, exploration aids, draft credit, initialization, and schedules remain matched.",
                "Require a material recovery from R5 toward the R1/R4/S4 external-strength range without losing sampled four-element completion or worsening conditional targeting.",
                "Prefer R7 if half combat Phi matches R8's strength and strategy trajectory; otherwise retain R8 only if full Phi passes the strategy-v2 and late-forgetting gates.",
                "Advance only one sustained arm to a fresh 45M qualifier; a 15M result cannot qualify production.",
            ],
            "arms": followup_arms,
        }
        _write_registration(
            FOLLOWUP_RESULTS / "registration.json",
            followup_registration,
        )
        return
    if args.qualifier is None:
        _registration_decision(
            ROOT / "train-ablation-1781126582/results/reward_screens/decision_packet.json"
        )
        screen_arms = [
            _build_arm(
                template,
                arm_id,
                sampled_rows=SCREEN_ROWS,
                phase="screen",
            )
            for arm_id in ARM_SPECS
        ]
        screen_registration = {
            "schema_id": "azuki.ablation_registration",
            "schema_version": 1,
            "family": "terminal_safe_reward_r5_r6_screen",
            "control_arm_id": "R5",
            "sampled_rows_per_arm": SCREEN_ROWS,
            "seed": 42,
            "parent_arm": "R1",
            "parent_decision": "train-ablation-1781126582/results/reward_screens/decision_packet.json",
            "parent_recipe": str(TEMPLATE.relative_to(ROOT)),
            "parent_recipe_sha256": sha256(TEMPLATE),
            "single_changed_boundary": "r5_vs_r6_state_potential_composition_on_shared_terminal_safe_r1_base",
            "shared_contract": _shared_contract(SCREEN_ROWS, SCREEN_EVALUATION_UPDATES),
            "selection_rules": [
                "Reject any arm with non-finite metrics, timeout truncation, incomplete episodes, reward reconstruction error, or non-telescoping discounted potential.",
                "R5 must emit zero leader-health and Garden-attack potential components while retaining readiness/resource potential; R6 must emit zero potential components.",
                "Compare every retained window under sampled and deterministic policies, the complete-context strength schedule, current strategy funnels, sibling differentiation, and conditional face-versus-entity choice.",
                "Select one arm only if it avoids S4's broad strength regression and late forgetting without reducing strategic breadth.",
                "Advance only the selected arm to a fresh matched 45M qualifier; the 15M screen is not production qualification.",
            ],
            "arms": screen_arms,
        }
        _write_registration(RESULTS / "registration.json", screen_registration)
        return

    if args.qualifier in ARM_SPECS:
        qualifier_results = RESULTS
        qualifier_specs = ARM_SPECS
        decision_path = RESULTS / "decision_packet.json"
    elif args.qualifier in FOLLOWUP_ARM_SPECS:
        qualifier_results = FOLLOWUP_RESULTS
        qualifier_specs = FOLLOWUP_ARM_SPECS
        decision_path = FOLLOWUP_RESULTS / "decision_packet.json"
    elif args.qualifier in LATE_FOLLOWUP_ARM_SPECS:
        qualifier_results = LATE_FOLLOWUP_RESULTS
        qualifier_specs = LATE_FOLLOWUP_ARM_SPECS
        decision_path = LATE_FOLLOWUP_RESULTS / "decision_packet.json"
    elif args.qualifier in EXPLORATION_FOLLOWUP_ARM_SPECS:
        qualifier_results = EXPLORATION_FOLLOWUP_RESULTS
        qualifier_specs = EXPLORATION_FOLLOWUP_ARM_SPECS
        decision_path = EXPLORATION_FOLLOWUP_RESULTS / "decision_packet.json"
    else:
        qualifier_results = HORIZON_FOLLOWUP_RESULTS
        qualifier_specs = HORIZON_FOLLOWUP_ARM_SPECS
        decision_path = HORIZON_FOLLOWUP_RESULTS / "decision_packet.json"
    decision = _registration_decision(decision_path)
    selected_arm = decision.get("decision", {}).get("selected_arm")
    if selected_arm != args.qualifier:
        raise ValueError(
            f"qualifier {args.qualifier} does not match selected arm {selected_arm}"
        )

    qualifier_arm = _build_arm(
        template,
        args.qualifier,
        sampled_rows=QUALIFIER_ROWS,
        phase="qualifier",
        arm_specs=qualifier_specs,
        results=qualifier_results,
    )
    qualifier_registration = {
        "schema_id": "azuki.ablation_registration",
        "schema_version": 1,
        "family": f"terminal_safe_reward_{args.qualifier.lower()}_45m_qualifier",
        "control_arm_id": args.qualifier,
        "sampled_rows_per_arm": QUALIFIER_ROWS,
        "seed": 42,
        "selected_from": str(decision_path.relative_to(ROOT)),
        "selected_from_sha256": sha256(decision_path),
        "single_changed_boundary": "duration_only_from_selected_15m_recipe",
        "shared_contract": _shared_contract(
            QUALIFIER_ROWS,
            QUALIFIER_EVALUATION_UPDATES,
        ),
        "qualification_rules": [
            "All integrity and exact reward-contract checks must pass at every retained window.",
            "Strength and strategy gains must persist after the midpoint anneal and through the final window.",
            "Reject late forgetting, reduced corrected element-sequence breadth, worse conditional face-versus-entity choice, or collapsed sibling differentiation.",
            "Qualification fixes the reward foundation only; D2/D3, strategic exposure, league performance, R3, and S1/S3 remain separate recipe boundaries.",
        ],
        "arms": [qualifier_arm],
    }
    _write_registration(
        qualifier_results / "qualifier_registration.json",
        qualifier_registration,
    )


if __name__ == "__main__":
    main()
