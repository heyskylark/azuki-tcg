#!/usr/bin/env python3
"""Freeze and build the selected fresh 100M combined canary."""

from __future__ import annotations

import json
from pathlib import Path

from build_shaping_screens import ROOT, seed_pool, set_ini_value, sha256


TEMPLATE = ROOT / "python/config/azuki_shaping_s0_p015_e015_15m.ini"
RESULTS = ROOT / "train-ablation-1781126582/results/canary_100m"
CONFIG = ROOT / "python/config/azuki_canary_combined_100m.ini"
EVALUATION_CONFIG = ROOT / "python/config/azuki_draft_screen_d2_credit_floor_01_15m.ini"
DECISION_UPDATES = (325, 975, 1950, 3900, 6500, 6511)


def main() -> None:
    source_state, active = seed_pool()
    runtime = RESULTS / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "league_state.json"
    text = TEMPLATE.read_text()
    replacements = {
        ("base", "jsonl_log"): "train-ablation-1781126582/results/canary_100m/logs/production.jsonl",
        ("base", "tag"): "canary_combined_100m",
        ("train", "total_timesteps"): 100_000_000,
        ("league", "state_path"): state_path.relative_to(ROOT),
        ("league", "promotion_state_path"): (runtime / "league_state_promotion.json").relative_to(ROOT),
        ("league", "promotion_records_dir"): (runtime / "promotion_records").relative_to(ROOT),
        ("league", "opponent_dir"): (runtime / "opponents").relative_to(ROOT),
        ("artifacts", "evaluation_interval_updates"): 325,
        ("artifacts", "milestone_interval_updates"): 325,
        ("artifacts", "evaluation_keep"): 12,
    }
    for (section, key), value in replacements.items():
        text = set_ini_value(text, section, key, value)
    text = text.replace(
        "; 977 complete updates x 15,360 sampled rows = 15,006,720.",
        "; 6,511 complete updates x 15,360 sampled rows = 100,008,960.",
    )
    CONFIG.write_text(text)

    seed_state = {
        "version": 1,
        "next_policy_index": int(source_state["next_policy_index"]),
        "learner_policy_id": None,
        "champion_policy_id": None,
        "current_candidate_policy_id": None,
        "policies": {str(entry["policy_id"]): entry for entry in active},
        "history": [],
    }
    if not state_path.exists():
        state_path.write_text(json.dumps(seed_state, indent=2, sort_keys=True) + "\n")

    contract = {
        "schema_id": "azuki.canary_validation_contract",
        "schema_version": 1,
        "recipe_frozen_from": "train-ablation-1781126582/results/shaping_screens/decision_packet.json",
        "status": "blocked_strategy_coverage",
        "strategy_first_reassessment": (
            "train-ablation-1781126582/results/strategy_first_reassessment.json"
        ),
        "terminal_closure_decision": (
            "train-ablation-1781126582/results/terminal_closure_screen/decision_packet.json"
        ),
        "fresh_random_learner_initialization": True,
        "configured_rows": 100_000_000,
        "effective_rows": 100_008_960,
        "batch_rows": 15_360,
        "decision_updates": list(DECISION_UPDATES),
        "decision_rows": {str(update): update * 15_360 for update in DECISION_UPDATES},
        "evaluation_config": str(EVALUATION_CONFIG.relative_to(ROOT)),
        "evaluation_config_sha256": sha256(EVALUATION_CONFIG),
        "policy_modes": ["sample_temperature_1_no_smoothing", "legal_argmax_stable_first"],
        "strength_panel": {
            "complete_context_schedule": True,
            "anchors": {
                "p021000": "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p19500_178757505266/model_azuki_local_021000.pt",
                "p044000": "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p29300_178765385220/model_azuki_local_044000.pt",
                "p060000": "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p52750_178783658456/model_azuki_local_060000.pt"
            },
            "curated_panel": "train-ablation-1781126582/run_curated_strategy_panel.py",
            "report": ["pooled_score", "worst_element", "worst_gate_leader", "worst_opponent_family", "paired_deltas"]
        },
        "strategy_panel": {
            "trace_games_per_mode": 200,
            "descriptor": "train-ablation-1781126582/strategy_descriptor.py",
            "required": [
                "sibling_deck_distances",
                "element_and_type_slot_shares",
                "mechanic_funnels",
                "ordered_sequence_opportunities_completions_and_conversions",
                "conditional_face_vs_entity_choices",
                "reward_component_raw_and_scaled_magnitudes",
                "opponent_role_and_temporal_exposure"
            ]
        },
        "decision_semantics": {
            "p000325": "mechanics and opportunity discovery only; no strength promotion",
            "p000975": "element coverage plus reward, draft, and league telemetry integrity",
            "p001950": "strategy persistence after early pressure falls",
            "p003900": "first skill-ceiling check against cheap-policy ancestors",
            "p006500_to_p006511": "late strategy, strength conversion, and forgetting decision"
        },
        "hard_stop": [
            "integrity failure",
            "structural collapse across repeated windows",
            "no strategy acquisition despite real opportunities",
            "broad strength regression with no improving strategy trajectory"
        ],
        "acceptance": [
            "Lightning weapons and Water spells are drafted and converted in opportunity-normalized sampled and deterministic traces",
            "Earth and Fire retain multiple valid lines",
            "sibling decks and actions differ where interaction probes establish different value",
            "curated value-labeled interaction probes show conditional face attacks do not dominate stronger legal setup or control alternatives",
            "strategy support persists after reward, draft, and curriculum anneals",
            "the 60M-100M window improves against cheap ancestors or has a credible positive conversion slope while respecting the strength floor",
            "actual rollout logs show opponent role and temporal diversity",
            "quality-adjusted SPS and every integrity guard pass"
        ],
        "advance_on_pass": "fresh 450M confirmation with an unchanged recipe"
    }
    for checkpoint in contract["strength_panel"]["anchors"].values():
        if not (ROOT / checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    RESULTS.mkdir(parents=True, exist_ok=True)
    contract_path = RESULTS / "validation_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    arm = {
        "id": "CANARY-100M",
        "name": "canary_combined_100m",
        "tag": "canary_combined_100m",
        "result_root": "train-ablation-1781126582/results/canary_100m",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "evaluation_config": str(EVALUATION_CONFIG.relative_to(ROOT)),
        "evaluation_config_sha256": sha256(EVALUATION_CONFIG),
        "seed_state": str(state_path.relative_to(ROOT)),
        "initial_policy_count": len(active),
    }
    registration = {
        "schema_id": "azuki.ablation_registration",
        "schema_version": 1,
        "family": "fresh_combined_canary_100m",
        "control_arm_id": None,
        "sampled_rows_per_arm": 100_008_960,
        "configured_rows_per_arm": 100_000_000,
        "seed": 42,
        "parent_arm": "S0",
        "parent_decision": "train-ablation-1781126582/results/shaping_screens/decision_packet.json",
        "validation_contract": str(contract_path.relative_to(ROOT)),
        "validation_contract_sha256": sha256(contract_path),
        "fresh_random_learner_initialization": True,
        "strategy_first_reassessment": (
            "train-ablation-1781126582/results/strategy_first_reassessment.json"
        ),
        "terminal_closure_decision": (
            "train-ablation-1781126582/results/terminal_closure_screen/decision_packet.json"
        ),
        "status": "blocked_strategy_coverage",
        "blocker": (
            "No 15M candidate spans all four elements: Lightning sequence completion is "
            "zero in every ablation window and policy mode. Hold is strategy-based, not "
            "a short-run win-rate decision."
        ),
        "arms": [arm],
    }
    (RESULTS / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
