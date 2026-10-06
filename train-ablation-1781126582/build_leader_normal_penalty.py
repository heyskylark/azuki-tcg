#!/usr/bin/env python3
"""Register one fresh S43 Normal-penalty run against saved RANDOM50 checkpoints."""
from __future__ import annotations

import argparse
import configparser
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys

from build_strategy_discovery import BINARY_GLOBS, ROOT, ROWS_PER_UPDATE, SOURCE_FILES, SOURCE_GLOBS, json_bytes, sha256
from build_strategy_recipe import smoke_tasks
from evaluate_reward_screens import _checkpoint_from_manifest
from run_prefix_paired_eval import DISABLED_PROCESS_ENV

FAMILY = "leader_normal_penalty_v1"
DESIGN = "fresh_single_seed43_vs_saved_random50"
SEED = 43
FULL_UPDATES = 3256
FULL_ROWS = FULL_UPDATES * ROWS_PER_UPDATE
PROBES = [975, 1950, FULL_UPDATES]
ANNEAL_ROWS = 15_006_720
SMOKE_UPDATES = 20
DEFAULT_BASELINE = ROOT / "train-ablation-1781126582/results/strategy_retention_v1"
EXPECTED_LEADERS = {
    "STT01-001": 0.56, "AZK01-119": 0.56,
    "STT02-001": 0.36, "AZK01-125": 0.36,
    "STT03-001": 0.26, "AZK01-123": 0.64,
    "STT04-001": 0.22, "AZK01-121": 0.36,
}
FILES = (
    "train-ablation-1781126582/build_leader_normal_penalty.py",
    "train-ablation-1781126582/run_leader_normal_penalty.py",
    "train-ablation-1781126582/report_leader_normal_penalty.py",
    "train-ablation-1781126582/build_strategy_recipe.py",
    "train-ablation-1781126582/run_strategy_recipe.py",
    "train-ablation-1781126582/report_strategy_recipe.py",
    "train-ablation-1781126582/run_prefix_paired_eval.py",
    "train-ablation-1781126582/run_prefix_followup.py",
    "train-ablation-1781126582/monitor_local_production.py",
    "train-ablation-1781126582/monitor_strategy_discovery.py",
    "python/src/draft_normal_penalty.py",
    "python/config/policy_card_metadata_v1.npz",
    "scripts/azuki-card-defs.jsonl",
)


def portable(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve_registered(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def read_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    with path.open() as handle:
        config.read_file(handle)
    return config


def config_bytes(config: configparser.ConfigParser) -> bytes:
    buffer = io.StringIO()
    config.write(buffer)
    return buffer.getvalue().encode()


def build(output: Path, baseline_root: Path, penalty_config: Path, smoke: bool = False) -> dict:
    output, baseline_root, penalty_config = output.resolve(), baseline_root.resolve(), penalty_config.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"Immutable output already exists: {output}")
    frozen = {}

    def freeze(name: str | Path, expected: str | None = None) -> Path:
        path = resolve_registered(str(name)).resolve()
        key = portable(path)
        actual = frozen.get(key)
        if actual is None:
            actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen input drift: {path}")
        frozen[key] = actual
        return path

    def freeze_model(name: str | Path, expected: str | None = None) -> Path:
        checkpoint = freeze(name, expected)
        metadata = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
        if metadata.exists():
            freeze(metadata)
        return checkpoint

    baseline_registration_path = freeze(baseline_root / "registration.json")
    baseline_registration = json.loads(baseline_registration_path.read_text())
    if baseline_registration.get("family") != "strategy_retention_v1" or baseline_registration.get("smoke") is not False:
        raise ValueError("Expected the completed fresh 50M strategy_retention_v1 baseline")
    baseline_arm = next(arm for arm in baseline_registration["arms"] if arm["id"] == "RANDOM50_S43")
    if (baseline_arm["initialization"] != "fresh_matched_seed43_no_checkpoint_resume"
            or baseline_arm["total_updates"] != FULL_UPDATES or baseline_arm["total_timesteps"] != FULL_ROWS):
        raise ValueError("Baseline is not the fresh S43 50M run")
    base_path = freeze(baseline_root / "configs/random50_s43.ini", baseline_arm["config_sha256"])
    eval_path = freeze(baseline_root / "configs/random50_s43_evaluation.ini", baseline_arm["evaluation_config_sha256"])
    status_path = freeze(baseline_root / "runs/random50_s43/run_status.json")
    status = json.loads(status_path.read_text())
    if (status.get("state") != "completed" or status.get("verified_final_update") != FULL_UPDATES
            or status.get("config_sha256") != baseline_arm["config_sha256"]
            or status.get("registration_sha256") != sha256(baseline_registration_path)):
        raise ValueError("Saved RANDOM50_S43 completion/config provenance is invalid")
    experiment = resolve_registered(status["latest_experiment_dir"]).resolve()
    if not experiment.is_relative_to((baseline_root / "runs/random50_s43").resolve()):
        raise ValueError("Baseline checkpoint directory escapes its registered run")
    baseline_updates = [PROBES[0]] if smoke else PROBES
    checkpoints = {}
    for update in baseline_updates:
        model, digest = _checkpoint_from_manifest(experiment, update)
        freeze(experiment / f"checkpoint_{update:06d}.manifest.json")
        checkpoint = freeze_model(model, digest)
        checkpoints[str(update)] = {"checkpoint": portable(checkpoint), "checkpoint_sha256": digest}

    base, evaluation_base = read_config(base_path), read_config(eval_path)
    for config in (base, evaluation_base):
        for section, key in (("env", "deck_pool_path"), ("league", "promotion_bootstrap_panel_path"),
                             ("league", "production_anchor_checkpoint"), ("policy", "card_metadata_path")):
            value = config.get(section, key, fallback="").strip()
            if not value:
                continue
            path = freeze(value)
            if key == "promotion_bootstrap_panel_path":
                for member in json.loads(path.read_text())["members"]:
                    freeze_model(member["checkpoint_path"], member["checkpoint_hash"])
            elif path.suffix == ".pt":
                freeze_model(path)
    paired = copy.deepcopy(baseline_registration["paired_template"])
    for spec in paired["opponents"].values():
        freeze_model(spec["checkpoint"], spec["checkpoint_sha256"])
        expected = spec.get("config_sha256") or paired["source_sha256"].get(spec["config"])
        if expected is None:
            raise ValueError("Frozen opponent config lacks source provenance")
        freeze(spec["config"], expected)
    freeze(penalty_config)
    penalty = json.loads(penalty_config.read_text())
    if (penalty.get("schema_id") != "azuki.leader_normal_penalty" or penalty.get("schema_version") != 1
            or penalty.get("main_deck_size") != 50
            or {code: spec["max_normal_fraction"] for code, spec in penalty["leaders"].items()} != EXPECTED_LEADERS
            or any(spec["weight"] != 1.0 for spec in penalty["leaders"].values())):
        raise ValueError("Unexpected leader penalty configuration")
    catalog = freeze("python/config/policy_card_metadata_v1.json")
    if penalty["provenance"]["metadata_sha256"] != sha256(catalog):
        raise ValueError("Corrected catalog required before registering this experiment")
    freeze(penalty["provenance"]["regional_source"], penalty["provenance"]["regional_sha256"])

    updates = SMOKE_UPDATES if smoke else FULL_UPDATES
    rows = updates * ROWS_PER_UPDATE
    penalty_updates = [updates] if smoke else PROBES
    anneal_rows = rows if smoke else ANNEAL_ROWS
    artifacts, arms = {}, []
    for treatment in ("PENALTY", "CONTROL"):
        control = treatment == "CONTROL"
        arm_id, name = f"{treatment}_S43", f"{treatment.lower()}_s43"
        folder = output / "runs" / name
        tag = f"leader_normal_penalty_{name}"
        config = copy.deepcopy(base)
        values = {
            "base": {"tag": tag, "jsonl_log": portable(folder / "logs/production.jsonl")},
            "train": {"total_timesteps": FULL_ROWS if control else rows, "data_dir": portable(folder / "artifacts")},
            "league": {"state_path": portable(folder / "league/league_state.json"),
                       "promotion_state_path": portable(folder / "league/league_state_promotion.json"),
                       "promotion_records_dir": portable(folder / "league/promotion_records"),
                       "opponent_dir": portable(folder / "league/opponents")},
            "process_env": {"azk_draft_normal_penalty_config": str(penalty_config),
                            "azk_draft_normal_penalty_coef_initial": 0.0 if control else 0.5,
                            "azk_draft_normal_penalty_coef_final": 0.0 if control else 0.075,
                            "azk_draft_normal_penalty_anneal_start_rows": 0,
                            "azk_draft_normal_penalty_anneal_end_rows": anneal_rows},
            "logging": {"max_patterns": base.get("logging", "max_patterns") + ",losses/draft_episode_credit_normal_penalty_max"},
        }
        if smoke and not control:
            values["train"]["checkpoint_interval"] = SMOKE_UPDATES
            values["artifacts"] = {"recovery_interval_updates": SMOKE_UPDATES,
                                   "evaluation_interval_updates": SMOKE_UPDATES, "milestone_interval_updates": SMOKE_UPDATES}
            values["process_env"].update(azk_potential_anneal_end_rows=anneal_rows,
                                         azk_exploration_anneal_end_rows=anneal_rows)
        for section, fields in values.items():
            for key, value in fields.items():
                config.set(section, key, str(value))
        config_path = output / "configs" / f"{name}.ini"
        artifacts[config_path] = config_bytes(config)
        evaluation = copy.deepcopy(evaluation_base)
        for section in ("base", "train", "league"):
            for key, value in values[section].items():
                evaluation.set(section, key, str(value))
        for key, value in DISABLED_PROCESS_ENV.items():
            evaluation.set("process_env", key.lower(), value)
        evaluation_path = output / "configs" / f"{name}_evaluation.ini"
        artifacts[evaluation_path] = config_bytes(evaluation)
        arms.append({
            "id": arm_id, "name": name, "recipe": treatment, "treatment": treatment, "seed": SEED, "tag": tag,
            "initialization": "saved_random50_checkpoints" if control else "fresh_matched_seed43_no_checkpoint_resume",
            "config": portable(config_path), "config_sha256": hashlib.sha256(artifacts[config_path]).hexdigest(),
            "evaluation_config": portable(evaluation_path), "evaluation_config_sha256": hashlib.sha256(artifacts[evaluation_path]).hexdigest(),
            "run_root": portable(folder), "result_root": portable(folder), "production_qualified": False,
            "total_timesteps": FULL_ROWS if control else rows, "total_updates": FULL_UPDATES if control else updates,
            "new_training_rows": 0 if control else rows,
            "evaluation_updates": baseline_updates if control else penalty_updates,
            "saved_checkpoints": checkpoints if control else {},
            "baseline_updates": {} if control else {str(update): baseline_updates[index] for index, update in enumerate(penalty_updates)},
            "process_env": {key.upper(): value for key, value in config["process_env"].items()},
            "paired_registration": portable(output / "paired" / name / "registration.json"),
            "heldout_registration": portable(output / "heldout" / name / "registration.json"),
        })
    expected = set(SOURCE_FILES) | set(FILES)
    membership = {}
    for pattern in (*SOURCE_GLOBS, *BINARY_GLOBS):
        matches = sorted(str(path.relative_to(ROOT)) for path in ROOT.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"Missing registered source/binary glob: {pattern}")
        membership[pattern] = matches
        expected.update(matches)
    for name in sorted(expected):
        freeze(name)
    artifact_hashes = {portable(path): hashlib.sha256(content).hexdigest() for path, content in artifacts.items()}
    fingerprints = {**frozen, **artifact_hashes}
    if smoke:
        paired["tasks"] = smoke_tasks(paired["tasks"], paired["opponents"])
    if len(paired["tasks"]) != (4 if smoke else 512):
        raise ValueError("Wrong registered task allocation")
    paired["checkpoints"], paired["source_sha256"] = {}, fingerprints
    heldout = copy.deepcopy(paired)
    for task in heldout["tasks"]:
        task["task_id"] = "heldout_" + task["task_id"]
        task["seed"] += 1_000_000_000
    games_per_panel = sum(len(arm["evaluation_updates"]) for arm in arms) * 2 * len(paired["tasks"])
    registration = {
        "schema_id": "azuki.ablation_registration", "schema_version": 1, "family": FAMILY, "design": DESIGN,
        "status": "registered", "scope": "diagnostic_only", "production_qualified": False, "smoke": smoke,
        "arms": arms, "seeds": [SEED], "training_arms": ["PENALTY_S43"], "saved_control_arms": ["CONTROL_S43"],
        "total_timesteps": rows, "total_updates": updates, "total_new_training_rows": rows,
        "evaluation_modes": ["sample", "argmax"], "evaluation_panels": ["paired", "heldout"],
        "paired_template": paired, "heldout_template": heldout,
        "expected_evaluation_games": 2 * games_per_panel,
        "expected_evaluation_games_by_panel": {"paired": games_per_panel, "heldout": games_per_panel},
        "baseline": {"registration": portable(baseline_registration_path), "sha256": sha256(baseline_registration_path),
                     "config": portable(base_path), "config_sha256": sha256(base_path),
                     "status": portable(status_path), "arm_id": "RANDOM50_S43"},
        "penalty_config": {"path": portable(penalty_config), "sha256": sha256(penalty_config), **penalty},
        "catalog_runtime": {"migration": "Sundering Strike Normal correction; PR50 c8444da", "same_evaluation_runtime_both_arms": True,
                            "historical_control_training_catalog": "Original uncorrected catalog; historical control is not retrained."},
        "schedule": {"learning_rate_initial": 0.003, "learning_rate_final": 0.0, "kind": "fresh_cosine",
                     "normal_initial": 0.5, "normal_final": 0.075, "normal_anneal_start_rows": 0,
                     "normal_anneal_end_rows": anneal_rows, "potential_final": 0.30, "exploration_final": 0.15},
        "comparison_contract": {"primary": "Consequential elemental sequences and supporting Normal cards, not low Normal share or activation volume alone.",
                                "seed_selection": "S43 selected from the user's completed short evaluation: both 5/5 games, Fire #7 and Earth #14. Selection is pragmatic, not independent evidence of seed superiority.",
                                "training": "One fresh S43 learner; no model, optimizer, RNG or league-state checkpoint resume. Existing bootstrap opponent recipe retained.",
                                "controls": "Reuse saved RANDOM50_S43 checkpoints; evaluate with corrected catalog. No new control training.",
                                "matched_progress": not smoke,
                                "smoke": "Infrastructure only; fresh p20 versus saved p975 is deliberately not a learning-quality comparison." if smoke else False,
                                "limitations": "One selected seed; historical control trained before catalog correction. No clean causal attribution to the penalty alone."},
        "selection_contract": {"automatic_selection": False, "automatic_promotion": False, "automatic_1b_launch": False, "winrate_gate": False},
        "runner_contract": {"require_discord_monitor": True, "train_saved_controls": False, "resume_checkpoint": False,
                            "no_overwrite": True, "serial_only": True, "ready_banner": "[leader-normal-penalty] ready"},
        "expected_source_files": sorted(fingerprints), "expected_source_globs": list(SOURCE_GLOBS),
        "expected_binary_globs": list(BINARY_GLOBS), "expected_glob_membership": membership,
        "source_sha256": fingerprints, "artifact_sha256": artifact_hashes,
        "runtime": {"python": sys.version, "python_executable": sys.executable,
                    "python_executable_sha256": sha256(Path(sys.executable)),
                    "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")}},
    }
    artifacts[output / "registration.json"] = json_bytes(registration)
    output.mkdir(parents=True, exist_ok=False)
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    print(portable(output / "registration.json"), flush=True)
    print(f"Registered one fresh S43 run: {updates} updates / {rows:,} rows; saved controls only.", flush=True)
    return registration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--penalty-config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    build(args.output, args.baseline_root, args.penalty_config, smoke=args.smoke)


if __name__ == "__main__":
    main()
