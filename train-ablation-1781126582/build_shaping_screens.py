#!/usr/bin/env python3
"""Build preregistered global-row shaping-tail screen configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "python/config/azuki_league_l5_window8_k1_frozen025_15m.ini"
SOURCE_STATE = (
    ROOT
    / "experiments/azuki_local_curriculum_c1_strategic_15m_178819296582/league_state_000977.json"
)
C1_EXPERIMENT = ROOT / "experiments/azuki_local_curriculum_c1_strategic_15m_178819296582"
POOL_ANCHOR_CHECKPOINT = (
    ROOT
    / "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p19500_178757505266/model_azuki_local_021000.pt"
)
C1_POOL_UPDATES = (50, 100, 150, 200, 325, 650, 800, 850, 900, 950, 975, 977)
EVALUATION_CONFIG = ROOT / "python/config/azuki_draft_screen_d2_credit_floor_01_15m.ini"
RESULTS = ROOT / "train-ablation-1781126582/results/shaping_screens"
ANNEAL_END_ROWS = 7_503_360


BASE_ARMS = (
    ("S0", "s0_p015_e015", 0.15, 0.15),
    ("S1", "s1_p015_e005", 0.15, 0.05),
    ("S2", "s2_p015_e000", 0.15, 0.00),
)


def set_ini_value(text: str, section: str, key: str, value: object) -> str:
    lines = text.splitlines()
    section_header = f"[{section}]"
    try:
        start = lines.index(section_header)
    except ValueError as exc:
        raise ValueError(f"missing INI section {section_header}") from exc
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("[")),
        len(lines),
    )
    prefix = f"{key} ="
    for index in range(start + 1, end):
        if lines[index].strip().startswith(prefix):
            lines[index] = f"{key} = {value}"
            return "\n".join(lines) + "\n"
    lines.insert(end, f"{key} = {value}")
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_pool() -> tuple[dict[str, object], list[dict[str, object]]]:
    source_state = json.loads(SOURCE_STATE.read_text())
    policies_by_path = {
        str(Path(entry["checkpoint_path"]).resolve()): entry
        for entry in source_state["policies"].values()
    }
    active: list[dict[str, object]] = []
    for update in C1_POOL_UPDATES:
        checkpoint = (C1_EXPERIMENT / f"model_azuki_local_{update:06d}.pt").resolve()
        source_entry = policies_by_path.get(str(checkpoint))
        if source_entry is None or not checkpoint.is_file():
            raise ValueError(f"qualified C1 seed checkpoint unavailable: {checkpoint}")
        entry = dict(source_entry)
        entry["rating"] = dict(source_entry["rating"])
        entry["active"] = True
        active.append(entry)
    if not POOL_ANCHOR_CHECKPOINT.is_file():
        raise FileNotFoundError(POOL_ANCHOR_CHECKPOINT)
    active.append(
        {
            "active": True,
            "bucket": "old",
            "checkpoint_path": str(POOL_ANCHOR_CHECKPOINT.resolve()),
            "created_by_learner_id": None,
            "created_epoch": 0,
            "created_ts": 0.0,
            "policy_id": "production_anchor_p021000",
            "rating": {
                "draws": 0,
                "elo": 1000.0,
                "games": 0,
                "losses": 0,
                "policy_id": "production_anchor_p021000",
                "wins": 0,
            },
            "source": "production_anchor",
        }
    )
    active.sort(key=lambda entry: (int(entry["created_epoch"]), str(entry["policy_id"])))
    if len(active) != 13:
        raise ValueError("shaping seed pool must contain exactly 13 policies")
    return source_state, active


def build_arm(
    template: str,
    source_state: dict[str, object],
    active: list[dict[str, object]],
    arm_id: str,
    slug: str,
    potential_tail: float,
    exploration_tail: float,
) -> dict[str, object]:
    arm_dir = RESULTS / slug
    runtime_dir = arm_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / f"python/config/azuki_shaping_{slug}_15m.ini"
    state_path = runtime_dir / "league_state.json"
    promotion_state_path = runtime_dir / "league_state_promotion.json"
    replacements = {
        ("base", "jsonl_log"): f"train-ablation-1781126582/results/shaping_screens/{slug}/logs/production.jsonl",
        ("base", "tag"): f"shaping_{slug}_15m",
        ("env", "reward_decomposed_schedule"): "true",
        ("league", "state_path"): state_path.relative_to(ROOT),
        ("league", "promotion_state_path"): promotion_state_path.relative_to(ROOT),
        ("league", "promotion_records_dir"): (runtime_dir / "promotion_records").relative_to(ROOT),
        ("league", "opponent_dir"): (runtime_dir / "opponents").relative_to(ROOT),
        ("process_env", "azk_reward_shaping_anneal"): 0,
        ("process_env", "azk_reward_decomposed_schedule"): 1,
        ("process_env", "azk_potential_scale_initial"): 1.0,
        ("process_env", "azk_potential_scale_final"): potential_tail,
        ("process_env", "azk_potential_anneal_start_rows"): 0,
        ("process_env", "azk_potential_anneal_end_rows"): ANNEAL_END_ROWS,
        ("process_env", "azk_exploration_scale_initial"): 1.0,
        ("process_env", "azk_exploration_scale_final"): exploration_tail,
        ("process_env", "azk_exploration_anneal_start_rows"): 0,
        ("process_env", "azk_exploration_anneal_end_rows"): ANNEAL_END_ROWS,
    }
    text = template
    for (section, key), value in replacements.items():
        text = set_ini_value(text, section, key, value)
    config_path.write_text(text)

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
    return {
        "id": arm_id,
        "name": slug,
        "tag": f"shaping_{slug}_15m",
        "result_root": f"train-ablation-1781126582/results/shaping_screens/{slug}",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "evaluation_config": str(EVALUATION_CONFIG.relative_to(ROOT)),
        "evaluation_config_sha256": sha256(EVALUATION_CONFIG),
        "seed_state": str(state_path.relative_to(ROOT)),
        "potential_tail": potential_tail,
        "exploration_tail": exploration_tail,
        "anneal_end_rows": ANNEAL_END_ROWS,
        "initial_policy_count": len(active),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3-exploration-tail",
        type=float,
        choices=(0.0, 0.05, 0.15),
        help="Register S3 after selecting the exploration tail from S0-S2.",
    )
    args = parser.parse_args()
    source_state, active = seed_pool()
    arm_specs = list(BASE_ARMS)
    if args.s3_exploration_tail is not None:
        code = {0.0: "000", 0.05: "005", 0.15: "015"}[args.s3_exploration_tail]
        arm_specs.append(("S3", f"s3_p000_e{code}", 0.0, args.s3_exploration_tail))
    template = TEMPLATE.read_text()
    arms = [build_arm(template, source_state, active, *spec) for spec in arm_specs]
    registration = {
        "schema_id": "azuki.ablation_registration",
        "schema_version": 1,
        "family": "global_row_shaping_tail",
        "control_arm_id": "S0",
        "sampled_rows_per_arm": 15_006_720,
        "seed": 42,
        "parent_arm": "L5",
        "parent_decision": "train-ablation-1781126582/results/league_screens/decision_packet.json",
        "source_pool_state": str(SOURCE_STATE.relative_to(ROOT)),
        "source_pool_state_sha256": sha256(SOURCE_STATE),
        "shared_contract": {
            "fresh_initialization": True,
            "matched_initial_opponent_pool": True,
            "global_sampled_row_schedule": True,
            "potential_components": [4, 5, 6, 7, 8, 9],
            "exploration_components": list(range(10, 20)),
            "terminal_components_unscaled": [0, 1, 2, 3],
            "anneal_start_rows": 0,
            "anneal_end_rows": ANNEAL_END_ROWS,
            "retained_checkpoint_interval_updates": 50,
        },
        "selection_rules": {
            "S0_S2": "Choose the lowest exploration tail that preserves strategic and structural readouts.",
            "S3": "Test zero potential only after selecting the exploration tail; reject on late forgetting.",
        },
        "arms": arms,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
