#!/usr/bin/env python3
"""Build preregistered league-screen configs from the qualified C1 recipe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "python/config/azuki_curriculum_c1_strategic_15m.ini"
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
RESULTS = ROOT / "train-ablation-1781126582/results/league_screens"
ANCHOR = (
    "experiments/azuki_local_corrected_production_1b_lr1500_fresh_resume_p19500_"
    "178757505266/model_azuki_local_021000.pt"
)
ROLE_FLOORS = "anchor:0.15,recent:0.25,hard:0.25,distinct:0.15,history:0.20"
ROLE_QUOTAS = "anchor:1,distinct:1,recent:6,history:5"

ARMS = {
    "L0": {"slug": "l0_window8_k1", "window": 8, "k": 1, "roles": False},
    "L1": {"slug": "l1_window1_k1", "window": 1, "k": 1, "roles": False},
    "L2": {"slug": "l2_roles_k1", "window": 1, "k": 1, "roles": True},
    "L3": {"slug": "l3_roles_k2", "window": 1, "k": 2, "roles": True},
    "L4": {"slug": "l4_roles_k4", "window": 1, "k": 4, "roles": True},
    "L5": {
        "slug": "l5_window8_k1_frozen025",
        "window": 8,
        "k": 1,
        "roles": False,
        "frozen_ratio": 0.25,
    },
}


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


def main() -> None:
    template = TEMPLATE.read_text()
    source_state = json.loads(SOURCE_STATE.read_text())
    policies_by_path = {
        str(Path(entry["checkpoint_path"]).resolve()): entry
        for entry in source_state["policies"].values()
    }
    active = []
    for update in C1_POOL_UPDATES:
        checkpoint = (C1_EXPERIMENT / f"model_azuki_local_{update:06d}.pt").resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        source_entry = policies_by_path.get(str(checkpoint))
        if source_entry is None:
            raise ValueError(f"qualified C1 state does not register {checkpoint}")
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
    active.sort(key=lambda entry: (int(entry["created_epoch"]), entry["policy_id"]))
    if len(active) != 13 or not all(Path(entry["checkpoint_path"]).is_file() for entry in active):
        raise ValueError("league seed pool must contain 13 existing checkpoints")
    opponent_checkpoints = ",".join(entry["checkpoint_path"] for entry in active)
    registrations = []

    for arm_id, arm in ARMS.items():
        slug = str(arm["slug"])
        arm_dir = RESULTS / slug
        runtime_dir = arm_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        config_path = ROOT / f"python/config/azuki_league_{slug}_15m.ini"
        state_path = runtime_dir / "league_state.json"
        promotion_state_path = runtime_dir / "league_state_promotion.json"

        text = template
        replacements = {
            ("base", "jsonl_log"): f"train-ablation-1781126582/results/league_screens/{slug}/logs/production.jsonl",
            ("base", "tag"): f"league_{slug}_15m",
            ("league", "frozen_window_epochs"): arm["window"],
            ("league", "max_distinct_frozen"): arm["k"],
            ("league", "opponent_checkpoints"): opponent_checkpoints,
            ("league", "state_path"): state_path.relative_to(ROOT),
            ("league", "promotion_state_path"): promotion_state_path.relative_to(ROOT),
            ("league", "promotion_records_dir"): (runtime_dir / "promotion_records").relative_to(ROOT),
            ("league", "opponent_dir"): (runtime_dir / "opponents").relative_to(ROOT),
            ("league", "frozen_ratio"): arm.get("frozen_ratio", 0.40),
            ("league", "role_sampling_floors"): ROLE_FLOORS if arm["roles"] else "",
            ("league", "sampling_policy_budget"): 13 if arm["roles"] else 0,
            ("league", "sampling_role_quotas"): ROLE_QUOTAS if arm["roles"] else "",
        }
        if arm["roles"]:
            replacements.update(
                {
                    ("league", "eval_interval"): 50,
                    ("league", "quick_eval_interval"): 300,
                    ("league", "full_eval_interval"): 600,
                    ("league", "promotion_shadow_mode"): "false",
                    ("league", "promotion_archive_affects_training_pool"): "true",
                    ("league", "promotion_panel_refresh_epochs"): 600,
                    ("league", "promotion_bootstrap_panel_path"): "",
                    ("league", "production_anchor_checkpoint"): ANCHOR,
                    ("league", "promotion_anchor_in_training_pool"): "true",
                }
            )
        for (section, key), value in replacements.items():
            text = set_ini_value(text, section, key, value)
        config_path.write_text(text)

        seed_state = {
            "version": 1,
            "next_policy_index": int(source_state["next_policy_index"]),
            "learner_policy_id": None,
            "champion_policy_id": None,
            "current_candidate_policy_id": None,
            "policies": {entry["policy_id"]: entry for entry in active},
            "history": [],
        }
        if not state_path.exists():
            state_path.write_text(json.dumps(seed_state, indent=2, sort_keys=True) + "\n")
        registrations.append(
            {
                "id": arm_id,
                "name": slug,
                "tag": f"league_{slug}_15m",
                "result_root": f"train-ablation-1781126582/results/league_screens/{slug}",
                **arm,
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256(config_path),
                "evaluation_config": str(EVALUATION_CONFIG.relative_to(ROOT)),
                "evaluation_config_sha256": sha256(EVALUATION_CONFIG),
                "seed_state": str(state_path.relative_to(ROOT)),
                "initial_policy_count": len(active),
            }
        )

    registration = {
        "schema_id": "azuki.ablation_registration",
        "schema_version": 1,
        "family": "league_retention_and_window_breadth",
        "control_arm_id": "L0",
        "sampled_rows_per_arm": 15_006_720,
        "seed": 42,
        "parent_arm": "C1",
        "parent_decision": "train-ablation-1781126582/results/curriculum_screens/decision_packet.json",
        "source_pool_state": str(SOURCE_STATE.relative_to(ROOT)),
        "source_pool_state_sha256": sha256(SOURCE_STATE),
        "preflight": "train-ablation-1781126582/results/league_screens/preflight.json",
        "retention_selection": "train-ablation-1781126582/results/league_screens/retention_selection.json",
        "initial_policy_epochs": [int(entry["created_epoch"]) for entry in active],
        "shared_contract": {
            "fresh_initialization": True,
            "matched_initial_opponent_pool": True,
            "frozen_row_ratio_control": 0.40,
            "strategic_exposure_parent": "C1",
            "promotion_anchor": ANCHOR,
            "role_floors": ROLE_FLOORS,
            "role_policy_budget": 13,
            "panel_refresh_epochs": 600,
            "quick_validation_epochs": 300,
            "full_validation_epochs": 600,
            "minimum_panel_incubation_validations": 2,
        },
        "selection_rules": {
            "L1": "Prefer over L0 only if strategic breadth improves without throughput regression.",
            "L2_L4": "Select the smallest k with nonzero exposure to every role, acceptable SPS, and improved strategic evaluation.",
            "L5": "Run only after selecting the retention/breadth parent; compare frozen row ratios 0.25 and 0.40.",
        },
        "arms": registrations,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
