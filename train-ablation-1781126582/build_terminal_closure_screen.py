#!/usr/bin/env python3
"""Build the missing final-recipe terminal-closure qualification arm."""

from __future__ import annotations

import json
from pathlib import Path

from build_shaping_screens import ROOT, seed_pool, set_ini_value, sha256


TEMPLATE = ROOT / "python/config/azuki_shaping_s0_p015_e015_15m.ini"
RESULTS = ROOT / "train-ablation-1781126582/results/terminal_closure_screen"
CONFIG = ROOT / "python/config/azuki_terminal_closure_s4_discounted_closed_15m.ini"
EVALUATION_CONFIG = ROOT / "python/config/azuki_draft_screen_d2_credit_floor_01_15m.ini"


def main() -> None:
    source_state, active = seed_pool()
    arm_dir = RESULTS / "s4_discounted_closed"
    runtime = arm_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "league_state.json"
    text = TEMPLATE.read_text()
    replacements = {
        ("base", "jsonl_log"): "train-ablation-1781126582/results/terminal_closure_screen/s4_discounted_closed/logs/production.jsonl",
        ("base", "tag"): "terminal_closure_s4_discounted_closed_15m",
        ("env", "pbrs_mode"): "discounted",
        ("env", "pbrs_terminal_closure"): "true",
        ("league", "state_path"): state_path.relative_to(ROOT),
        ("league", "promotion_state_path"): (runtime / "league_state_promotion.json").relative_to(ROOT),
        ("league", "promotion_records_dir"): (runtime / "promotion_records").relative_to(ROOT),
        ("league", "opponent_dir"): (runtime / "opponents").relative_to(ROOT),
    }
    for (section, key), value in replacements.items():
        text = set_ini_value(text, section, key, value)
    CONFIG.write_text(text)

    state = {
        "version": 1,
        "next_policy_index": int(source_state["next_policy_index"]),
        "learner_policy_id": None,
        "champion_policy_id": None,
        "current_candidate_policy_id": None,
        "policies": {str(entry["policy_id"]): entry for entry in active},
        "history": [],
    }
    if not state_path.exists():
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    arm = {
        "id": "S4",
        "name": "s4_discounted_closed",
        "tag": "terminal_closure_s4_discounted_closed_15m",
        "result_root": "train-ablation-1781126582/results/terminal_closure_screen/s4_discounted_closed",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "evaluation_config": str(EVALUATION_CONFIG.relative_to(ROOT)),
        "evaluation_config_sha256": sha256(EVALUATION_CONFIG),
        "seed_state": str(state_path.relative_to(ROOT)),
        "pbrs_mode": "discounted",
        "pbrs_terminal_closure": True,
        "direct_leader_weight": 1.25,
        "direct_board_weight": 0.35,
        "potential_tail": 0.15,
        "exploration_tail": 0.15,
        "initial_policy_count": len(active),
    }
    registration = {
        "schema_id": "azuki.ablation_registration",
        "schema_version": 1,
        "family": "final_recipe_terminal_closure_qualification",
        "control_arm_id": "S0",
        "sampled_rows_per_arm": 15_006_720,
        "seed": 42,
        "parent_arm": "S0",
        "parent_decision": "train-ablation-1781126582/results/shaping_screens/decision_packet.json",
        "reused_control": {
            "id": "S0",
            "config": "python/config/azuki_shaping_s0_p015_e015_15m.ini",
            "checkpoint": "experiments/azuki_local_shaping_s0_p015_e015_15m_178833128483/model_azuki_local_000975.pt",
            "checkpoint_sha256": "2bbdc1d68041d007d28fcc19df1ca5b8d29717a18e6843698e80fae5ce0d5bbf",
            "evaluation_index": "train-ablation-1781126582/results/shaping_screens/evaluation_index.json"
        },
        "single_changed_boundary": "legacy_unclosed_to_discounted_terminal_closed_pbrs",
        "shared_contract": {
            "fresh_initialization": True,
            "matched_seed_and_initial_opponent_pool": True,
            "all_non_pbrs_parameters_identical_to_s0": True,
            "evaluation_updates": [325, 650, 975],
            "terminal_and_draft_credit_unscaled": True,
            "reward_component_reconstruction_required": True
        },
        "decision_rule": [
            "Terminal closure and reward reconstruction telemetry must pass exactly.",
            "Compare S4 against S0 at every retained window under both policy modes and the complete-context strength schedule.",
            "Reject S4 for broad strength regression, late forgetting, reduced strategic breadth, or worse conditional face-versus-entity choices.",
            "A passing terminal-safe reward recipe is required before the staged 1B production candidate; S4 did not qualify, so test R5/R6 next."
        ],
        "arms": [arm],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
