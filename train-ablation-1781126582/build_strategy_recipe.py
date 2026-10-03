#!/usr/bin/env python3
"""Register fresh strategy-first recipe diagnostics without launching training."""
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

from build_strategy_discovery import (
    ROOT, SOURCE_GLOBS, SOURCE_FILES, BINARY_GLOBS, ROWS_PER_UPDATE,
    SCREEN_ROWS, json_bytes, portable, sha256,
)
from run_prefix_paired_eval import DISABLED_PROCESS_ENV, validate_config

RESULTS = ROOT / "train-ablation-1781126582/results/strategy_recipe_v1"
DISCOVERY = ROOT / "train-ablation-1781126582/results/strategy_discovery_v1/registration.json"
PAIRED = ROOT / "train-ablation-1781126582/results/prefix_paired_followup_v1/registration.json"
RECIPES = ("CONTROL", "RANDOM20", "STRATEGIC20", "STRATEGIC50", "STRATEGIC20_HOLD_EXPLORATION")
RETENTION_ROWS = 50_012_160
RETENTION_UPDATES = [325, 650, 975, 1625, 2275, 3250]
RETENTION_RECIPES = ("STRATEGIC50", "RANDOM50")
CONTINUATION_PARENT = ROOT / "train-ablation-1781126582/results/strategy_retention_v1/registration.json"
CONTINUATION_PARENT_UPDATE = 3256
CONTINUATION_UPDATES = 13021
FILES = (
    "train-ablation-1781126582/build_strategy_recipe.py",
    "train-ablation-1781126582/run_strategy_recipe.py",
    "train-ablation-1781126582/report_strategy_recipe.py",
    "train-ablation-1781126582/run_prefix_paired_eval.py",
    "train-ablation-1781126582/run_prefix_followup.py",
    "train-ablation-1781126582/monitor_local_production.py",
    "train-ablation-1781126582/monitor_strategy_discovery.py",
)


def read_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    with path.open() as handle:
        config.read_file(handle)
    return config


def config_bytes(config: configparser.ConfigParser) -> bytes:
    buffer = io.StringIO()
    config.write(buffer)
    return buffer.getvalue().encode()


def seed_config(config: configparser.ConfigParser, seed: int) -> None:
    """Move explicit stochastic side channels together, retaining their offsets."""
    old_seed = config.getint("train", "seed")
    shift = (seed - old_seed) * 10_000
    for section in config.sections():
        for key, value in list(config[section].items()):
            if key == "seed_process_rngs" or not value.strip():
                continue
            if key == "seed" or key.endswith("_seed") or key.endswith("_seeds"):
                values = [int(part.strip()) for part in value.split(",")]
                config.set(section, key, ",".join(str(v + shift) for v in values))
    config.set("train", "seed", str(seed))
    config.set("train", "seed_process_rngs", "true")
    # League sampling and terminal-credit fallbacks normally derive from train.seed.
    config.set("league", "seed", str(seed))
    config.set("process_env", "azk_draft_episode_credit_seed", str(seed * 10_000 + 52))
    config.set("process_env", "azk_draft_prefix_seed", str(seed * 10_000 + 53))


def smoke_tasks(tasks: list[dict], opponents: dict) -> list[dict]:
    selected = []
    for opponent in opponents:
        for seat in (0, 1):
            task = next((t for t in tasks if t["opponent_id"] == opponent and t["candidate_seat"] == seat), None)
            if task is None:
                raise ValueError(f"Missing paired smoke coverage: {opponent}, seat {seat}")
            selected.append(copy.deepcopy(task))
    return selected


def continuation_parent(arm: dict, run_root: Path, freeze, registration_hash: str) -> tuple[configparser.ConfigParser, dict, dict]:
    """Freeze manifest-backed full-state inputs, including every live league member."""
    config_path = freeze(arm["config"], arm["config_sha256"])
    status_path = freeze(str(ROOT / arm["run_root"] / "run_status.json"))
    status = json.loads(status_path.read_text())
    if (status.get("state") != "completed" or status.get("verified_final_update") != CONTINUATION_PARENT_UPDATE
            or status.get("arm") != arm["id"] or status.get("config_sha256") != arm["config_sha256"]
            or status.get("registration_sha256") != registration_hash):
        raise ValueError(f"Unverified continuation parent: {arm['id']}")
    experiment = ROOT / status["latest_experiment_dir"]
    experiment.resolve().relative_to((ROOT / arm["run_root"]).resolve())
    manifest_path = freeze(str(experiment / "checkpoint_003256.manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("update") != CONTINUATION_PARENT_UPDATE or manifest.get("config") != {
            "path": arm["config"], "sha256": arm["config_sha256"]}:
        raise ValueError(f"Parent manifest/config mismatch: {manifest_path}")
    members = {item["path"]: item for item in manifest["artifacts"]}
    paths = {}
    for name in ("model_azuki_local_003256.pt", "model_azuki_local_003256.pt.meta.json",
                 "trainer_state_003256.pt", "league_state_003256.json", "promotion_state_003256.json"):
        item = members[name]
        paths[name] = freeze(str(experiment / name), item["sha256"])
    resume = {"checkpoint": portable(paths["model_azuki_local_003256.pt"]),
              "checkpoint_sha256": members["model_azuki_local_003256.pt"]["sha256"],
              "trainer_state": portable(paths["trainer_state_003256.pt"]),
              "trainer_state_sha256": members["trainer_state_003256.pt"]["sha256"],
              "metadata": portable(paths["model_azuki_local_003256.pt.meta.json"]),
              "metadata_sha256": members["model_azuki_local_003256.pt.meta.json"]["sha256"],
              "experiment_dir": portable(experiment), "update": CONTINUATION_PARENT_UPDATE,
              "manifest": portable(manifest_path), "manifest_sha256": sha256(manifest_path),
              "run_status": portable(status_path), "run_status_sha256": sha256(status_path)}
    restore = {"checkpoints": [], "path_rewrites": {}}
    snapshots = {}
    for key, name, destination in (
            ("state", "league_state_003256.json", "league_state.json"),
            ("promotion_state", "promotion_state_003256.json", "league_state_promotion.json")):
        restore[key] = {"source": portable(paths[name]), "sha256": members[name]["sha256"],
                        "destination": portable(run_root / "league" / destination)}
        snapshots[key] = json.loads(paths[name].read_text())
    consumed = {}
    for policy in snapshots["state"]["policies"].values():
        path = ROOT / policy["checkpoint_path"]
        if policy.get("active") or path.exists():
            consumed[str(path.resolve())] = policy.get("checkpoint_hash")
    for panel in snapshots["promotion_state"]["panels"]:
        for member in panel["members"]:
            consumed[str((ROOT / member["checkpoint_path"]).resolve())] = member["checkpoint_hash"]
    for source, expected in sorted(consumed.items()):
        checkpoint = freeze(source, expected)
        digest = sha256(checkpoint)
        destination = run_root / "league/opponents" / digest / checkpoint.name
        item = {"source": portable(checkpoint), "sha256": digest, "destination": portable(destination)}
        restore["path_rewrites"][source] = str(destination)
        restore["path_rewrites"][portable(checkpoint)] = str(destination)
        metadata = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
        if metadata.exists():
            freeze(str(metadata))
            item.update(metadata_source=portable(metadata), metadata_sha256=sha256(metadata),
                        metadata_destination=portable(destination.with_suffix(destination.suffix + ".meta.json")))
        restore["checkpoints"].append(item)
    return read_config(config_path), resume, restore


def build(output: Path, smoke: bool = False, retention: bool = False, continuation: bool = False) -> dict:
    if continuation and retention:
        raise ValueError("Continuation and retention are mutually exclusive")
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"Immutable output already exists: {output}")
    discovery = json.loads(DISCOVERY.read_text())
    paired = json.loads(PAIRED.read_text())
    if discovery.get("family") != "strategy_discovery_v1" or discovery.get("production_qualified") is not False:
        raise ValueError("Expected the diagnostic discovery parent")
    if (paired.get("schema_id") != "azuki.prefix_paired_eval_registration"
            or paired.get("schema_version") != 1 or paired.get("production_qualified") is not False
            or paired.get("step_cap") != 1200 or len(paired["tasks"]) != 512
            or len(paired["opponents"]) != 2):
        raise ValueError("Expected the frozen 512-task, two-opponent paired protocol")
    parent = next(arm for arm in discovery["arms"] if arm["id"] == "CONTROL")
    parent_path = ROOT / parent["config"]
    evaluation_path = PAIRED.parent / "control_evaluation.ini"
    frozen = {portable(DISCOVERY): sha256(DISCOVERY), portable(PAIRED): sha256(PAIRED)}

    def freeze(name: str, expected: str | None = None) -> Path:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        key = portable(path)
        actual = frozen.get(key)
        if actual is None:
            actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen input drift: {name}")
        frozen[key] = actual
        return path

    freeze(parent["config"], parent["config_sha256"])
    # The historical evaluator's source set is not the current runtime source set.
    # Only frozen inputs are checked against that registration; code is re-fingerprinted.
    expected_eval_hash = paired["source_sha256"].get(portable(evaluation_path))
    if expected_eval_hash is None:
        expected_eval_hash = paired["source_sha256"].get(str(evaluation_path))
    if expected_eval_hash is None:
        raise ValueError("Paired control evaluation config lacks registered provenance")
    freeze(str(evaluation_path), expected_eval_hash)
    validate_config(evaluation_path)
    for collection in ("opponents", "checkpoints"):
        for spec in paired[collection].values():
            checkpoint = freeze(spec["checkpoint"], spec["checkpoint_sha256"])
            config_path = ROOT / spec["config"]
            expected = spec.get("config_sha256") or paired["source_sha256"].get(spec["config"])
            if expected is None:
                expected = paired["source_sha256"].get(str(config_path))
            if expected is None:
                raise ValueError(f"Unregistered paired config: {config_path}")
            freeze(str(config_path), expected)
            validate_config(config_path)
            metadata = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
            if metadata.exists():
                expected = paired["source_sha256"].get(portable(metadata)) or paired["source_sha256"].get(str(metadata))
                if expected is None:
                    raise ValueError(f"Unregistered checkpoint metadata: {metadata}")
                freeze(str(metadata), expected)
    for key in ("pool", "prefix_pool"):
        freeze(discovery[key]["path"], discovery[key]["sha256"])
    base = read_config(parent_path)
    evaluation_base = read_config(evaluation_path)
    if base.get("train", "device") != "cuda" or base.getint("train", "total_timesteps") != SCREEN_ROWS:
        raise ValueError("Parent must be the full CUDA 15M CONTROL diagnostic")
    # Freeze consumed paths, including league bootstrap panel/checkpoint and deck metadata.
    for config in (base, evaluation_base):
        for section, key in (("env", "deck_pool_path"), ("league", "promotion_bootstrap_panel_path"),
                             ("league", "production_anchor_checkpoint"), ("policy", "card_metadata_path")):
            value = config.get(section, key, fallback="").strip()
            if value:
                path = freeze(value)
                if key == "promotion_bootstrap_panel_path":
                    panel = json.loads(path.read_text())
                    for member in panel["members"]:
                        checkpoint = freeze(member["checkpoint_path"], member["checkpoint_hash"])
                        metadata = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
                        if metadata.exists():
                            freeze(str(metadata))
                if path.suffix == ".pt":
                    metadata = path.with_suffix(path.suffix + ".meta.json")
                    if metadata.exists():
                        freeze(str(metadata))
    continuation_registration = None
    if continuation:
        freeze(str(CONTINUATION_PARENT))
        continuation_registration = json.loads(CONTINUATION_PARENT.read_text())
        if continuation_registration.get("family") != "strategy_retention_v1" or continuation_registration.get("smoke"):
            raise ValueError("Expected full strategy_retention_v1 parent")
    ceiling = CONTINUATION_UPDATES * ROWS_PER_UPDATE if continuation else RETENTION_ROWS if retention else SCREEN_ROWS
    recipes = ("RANDOM50",) if continuation else RETENTION_RECIPES if retention else RECIPES
    rows = (3266 * ROWS_PER_UPDATE if continuation else 153_600) if smoke else ceiling
    updates = rows // ROWS_PER_UPDATE
    evaluation_updates = ([3256, 3266] if smoke else [3256, 6511, 13021]) if continuation else (
        [10] if smoke else RETENTION_UPDATES if retention else [325, 650, 975])
    specs = [(seed, "RANDOM50") for seed in (43, 44)] if continuation else (
        ([(43, recipe) for recipe in RETENTION_RECIPES] if retention else [(43, "STRATEGIC50")]) if smoke else [
            (seed, recipe) for seed in (43, 44) for recipe in recipes])
    artifacts = {}
    arms = []
    for seed, recipe in specs:
        arm_id = f"{recipe}_S{seed}"
        name = arm_id.lower()
        run_root = output / "runs" / name
        resume = restore = None
        if continuation:
            parent_arm = next(arm for arm in continuation_registration["arms"] if arm["id"] == arm_id)
            config, resume, restore = continuation_parent(parent_arm, run_root, freeze, frozen[portable(CONTINUATION_PARENT)])
            if config.getint("train", "seed") != seed or config.getint("league", "seed") != seed:
                raise ValueError(f"Continuation seed mismatch: {arm_id}")
        else:
            config = copy.deepcopy(base)
            seed_config(config, seed)
        tag = f"random50_continuation_{name}" if continuation else f"strategy_{'retention' if retention else 'recipe'}_{name}"
        values = {
            "base": {"tag": tag, "jsonl_log": portable(run_root / "logs/production.jsonl")},
            "train": {"total_timesteps": rows, "data_dir": portable(run_root / "artifacts")},
            "league": {"state_path": portable(run_root / "league/league_state.json"),
                       "promotion_state_path": portable(run_root / "league/league_state_promotion.json"),
                       "promotion_records_dir": portable(run_root / "league/promotion_records"),
                       "opponent_dir": portable(run_root / "league/opponents")},
            "process_env": {"azk_draft_prefix_lengths": "0,4",
                            "azk_draft_prefix_probs": "" if recipe == "CONTROL" else "0.5,0.5" if recipe in ("STRATEGIC50", "RANDOM50") else "0.8,0.2",
                            "azk_draft_prefix_pool_path": discovery["prefix_pool"]["path"] if recipe.startswith("STRATEGIC") else ""},
        }
        if continuation:
            del values["process_env"]
            values["train"]["learning_rate"] = "0.00003"
            values["resume"] = {"load_optimizer": "true", "restart_lr_schedule": "true",
                                "strict": "true", "auto_reset_critic": "false"}
            values["artifacts"] = {"evaluation_interval_updates": 6511, "milestone_interval_updates": 6511}
        if recipe == "STRATEGIC20_HOLD_EXPLORATION":
            values["process_env"]["azk_exploration_scale_final"] = "1.0"
        if smoke:
            values["train"]["checkpoint_interval"] = 10
            values["artifacts"] = {"recovery_interval_updates": 10, "evaluation_interval_updates": 10, "milestone_interval_updates": 10}
        for section, fields in values.items():
            if not config.has_section(section):
                config.add_section(section)
            for key, value in fields.items():
                config.set(section, key, str(value))
        config_path = output / "configs" / f"{name}.ini"
        artifacts[config_path] = config_bytes(config)
        evaluation = copy.deepcopy(evaluation_base)
        seed_config(evaluation, seed)
        for section in ("base", "train", "league"):
            for key, value in values[section].items():
                evaluation.set(section, key, str(value))
        for key, value in DISABLED_PROCESS_ENV.items():
            evaluation.set("process_env", key.lower(), value)
        for key in ("draft_cross_gate_replay_prob", "draft_same_element_matchup_prob"):
            evaluation.set("env", key, "0")
        evaluation_config = output / "configs" / f"{name}_evaluation.ini"
        artifacts[evaluation_config] = config_bytes(evaluation)
        arms.append({"id": arm_id, "name": name, "recipe": recipe, "seed": seed,
                     "config": portable(config_path), "config_sha256": hashlib.sha256(artifacts[config_path]).hexdigest(),
                     "evaluation_config": portable(evaluation_config), "evaluation_config_sha256": hashlib.sha256(artifacts[evaluation_config]).hexdigest(),
                     "run_root": portable(run_root), "result_root": portable(run_root), "tag": tag,
                     "total_timesteps": rows, "total_updates": updates, "evaluation_updates": evaluation_updates,
                     "initialization": f"fresh_matched_seed{seed}_no_checkpoint_resume", "production_qualified": False,
                     "process_env": {key.upper(): value for key, value in config["process_env"].items()},
                     "paired_registration": portable(output / "paired" / name / "registration.json"),
                     "treatment": {"prefix": "none" if recipe == "CONTROL" else "random" if recipe.startswith("RANDOM") else "strategic",
                                   "prefix_probability": 0 if recipe == "CONTROL" else 0.5 if recipe in ("STRATEGIC50", "RANDOM50") else 0.2,
                                   "prefix_lengths": [0, 4], "hold_exploration": recipe == "STRATEGIC20_HOLD_EXPLORATION"}})
        if continuation:
            arms[-1].update(
                initialization="full_state_continuation", parent_update=CONTINUATION_PARENT_UPDATE,
                resume_checkpoint=resume, league_restore=restore,
                heldout_registration=portable(output / "heldout" / name / "registration.json"),
                parent_config=parent_arm["config"], parent_config_sha256=parent_arm["config_sha256"],
                exact_trajectory_resume=False,
                schedule={"kind": "new_cosine", "peak_learning_rate": 0.00003, "final_learning_rate": 0.0,
                          "start_update": CONTINUATION_PARENT_UPDATE, "remaining_updates": updates - CONTINUATION_PARENT_UPDATE,
                          "production_remaining_updates": CONTINUATION_UPDATES - CONTINUATION_PARENT_UPDATE,
                          "smoke_short_cosine": smoke, "production_lr_parity": not smoke,
                          "parent_fresh_peak_learning_rate": 0.003,
                          "peak_rationale": "Deliberate conservative 100x reduction from parent fresh peak; not a quality guarantee.",
                          "learner_step_schedules": "Entropy, temperature and smoothing retain parent 150M->225M learner-step endpoints and restored global_step.",
                          "reward_anneal_end_rows": 15_006_720, "reward_progression": "restored_sampled_rows"})
    if any(arm["total_timesteps"] > ceiling or arm["total_updates"] > ceiling // ROWS_PER_UPDATE for arm in arms):
        raise ValueError("Diagnostic production ceiling exceeded")
    expected = set(SOURCE_FILES) | set(FILES)
    membership = {}
    for pattern in (*SOURCE_GLOBS, *BINARY_GLOBS):
        matches = sorted(portable(path) for path in ROOT.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"Required source/runtime glob has no matches: {pattern}")
        membership[pattern] = matches
        expected.update(matches)
    for name in sorted(expected):
        freeze(name)
    artifact_hashes = {portable(path): hashlib.sha256(content).hexdigest() for path, content in artifacts.items()}
    fingerprints = {**frozen, **artifact_hashes}
    tasks = smoke_tasks(paired["tasks"], paired["opponents"]) if smoke else copy.deepcopy(paired["tasks"])
    template = {"schema_id": "azuki.prefix_paired_eval_registration", "schema_version": 1,
                "production_qualified": False, "checkpoints": {},
                **{key: copy.deepcopy(paired[key]) for key in ("opponents", "step_cap", "process_env")},
                "tasks": tasks, "source_sha256": fingerprints}
    registration = {
        "schema_id": "azuki.ablation_registration", "schema_version": 1,
        "family": "strategy_retention_v1" if retention else "strategy_recipe_v1",
        "status": "registered", "scope": "diagnostic_only", "production_qualified": False, "smoke": smoke,
        "purpose": "strategy_first_emergence_retention_and_diversity", "arms": arms,
        "seeds": [43] if smoke else [43, 44], "recipes": list(dict.fromkeys(arm["recipe"] for arm in arms)),
        "evaluation_modes": ["sample", "argmax"], "expected_evaluation_games": len(arms) * len(evaluation_updates) * 2 * len(tasks),
        "parent": {"config": parent["config"], "sha256": parent["config_sha256"], "registration": portable(DISCOVERY),
                   "status": "R14_shaped_unqualified_diagnostic_control"},
        "pool": copy.deepcopy(discovery["pool"]), "prefix_pool": copy.deepcopy(discovery["prefix_pool"]),
        "paired_template": template, "paired_parent": {"path": portable(PAIRED), "sha256": frozen[portable(PAIRED)]},
        "expected_source_files": sorted(fingerprints), "expected_source_globs": list(SOURCE_GLOBS),
        "expected_binary_globs": list(BINARY_GLOBS), "expected_glob_membership": membership,
        "source_sha256": fingerprints, "artifact_sha256": artifact_hashes,
        "comparison_contract": {
            "primary": "Unique strategy emergence, retention and diversity; candidate-only opportunity-normalized realized effects and draft diversity, not raw action volume.",
            "retention": "Compare early/middle/late checkpoints separately in both modes; expose gains that disappear by the endpoint.",
            "seed_consistency": "Report each seed separately, matched against RANDOM50." if retention else "Report each seed separately, matched against its CONTROL and RANDOM20; cross-seed direction is required before claiming consistency.",
            "seed_side_channels": "Explicit RNG seed fields retain offsets under a seed-based shift; credit/prefix seeds are train_seed*10000+52/+53; process RNGs enabled; league seed equals train seed. Matched recipes share seeds, not guaranteed identical trajectories.",
            "treatments": "STRATEGIC50 versus RANDOM50 isolates prefix content at matched 50% dosage. Reward anneal endpoints remain at 15,006,720 configured rows; potential and all other parent knobs are unchanged. Fresh 50M training extends the horizon-dependent learning-rate schedule; it is not an exact continuation of the 15M checkpoints." if retention else "RANDOM20 versus STRATEGIC20 isolates prefix content; STRATEGIC50 tests dosage; HOLD_EXPLORATION only raises exploration final scale .15 to 1.0 relative to STRATEGIC20. Potential and all remaining parent knobs are unchanged.",
            "supporting_only": "Winrate is optional supporting evidence, never an ordering or qualification gate.",
            "evaluation": "Both seats freely draft; prefixes, supplied decks, replay and same-element oversampling disabled; identical frozen task schedule in both policy modes.",
        },
        "selection_contract": {"objective": "strategy_first", "automatic_selection": False,
                               "automatic_1b_launch": False, "automatic_promotion": False,
                               "longer_confirmation_required": True, "winrate_gate": False,
                               "decision": "Human strategy-first review of emergence, retention, diversity and independent-seed consistency; neither diagnostics nor smoke qualify production."},
        "limitations": ["Frozen p21000/p60000 opponents have unproven historical correction provenance; their use is not current-source training or production qualification.",
                        "Deferred effects remain incompletely observed. Opportunity counts and observed effects are not causal action quality or complete strategy measurement.",
                        "One world seed per exact paired context is not independent repeated context evidence; training seeds remain separate.",
                        "Smoke covers only two seats by two frozen opponents, not the full context panel; partial campaign reports must be provisional."],
        "runner_contract": {"launch": "serial_only", "fresh_initialization": True, "working_directory": "repository root",
                            "pythonpath": "build/python/src:python/src", "max_arm_rows": ceiling,
                            "source_guard": "Verify all hashes and exact glob membership before each launch; reject existing run roots and production scope.",
                            "paired_checkpoints": "Add verified checkpoint/metadata hashes dynamically; checkpoint key is <arm.id>_p<update:06d>; source_sha256 derives from this registration.",
                            "paired_results": "paired/<arm.name>/eval/<arm.id>_p<update:06d>/<sample|argmax>/result.json",
                            "report": "Run report_strategy_recipe.py --registration <path> after every arm and at campaign end."},
        "runtime": {"python": sys.version, "python_executable": sys.executable,
                    "python_executable_sha256": sha256(Path(sys.executable)),
                    "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
                    "source_hashes_authoritative_for_working_tree": True},
    }
    if continuation:
        heldout = copy.deepcopy(template)
        for task in heldout["tasks"]:
            task["task_id"] = "heldout_" + task["task_id"]
            task["seed"] += 1_000_000_000
        games_per_panel = len(arms) * len(evaluation_updates) * 2 * len(tasks)
        registration.update(
            family="random50_continuation_v1", purpose="random50_full_state_strategy_retention_continuation",
            seeds=[43, 44], parent_update=CONTINUATION_PARENT_UPDATE,
            total_updates=updates, total_timesteps=rows, evaluation_updates=evaluation_updates,
            parent={"registration": portable(CONTINUATION_PARENT), "sha256": frozen[portable(CONTINUATION_PARENT)],
                    "status": "completed_random50_p3256_full_state"},
            heldout_template=heldout, expected_evaluation_games=2 * games_per_panel,
            expected_evaluation_games_by_panel={"paired": games_per_panel, "heldout": games_per_panel},
            evaluation_panels=["paired", "heldout"])
        registration["comparison_contract"].update(
            seed_consistency="Report RANDOM50 seeds43/44 separately; compare each against its own p3256 parent.",
            seed_side_channels="All parent training seed fields preserved unchanged; optimizer and saved RNG/coordinator state restored.",
            treatments="Strict full-state RANDOM50 continuation; no critic reset; deliberate new cosine peak3e-5. Reward/prefix/architecture/batch settings unchanged.",
            heldout="Same contexts and frozen opponents, every task seed +1000000000 and task_id prefixed heldout_; not new-opponent generalization.")
        registration["runner_contract"].update(
            fresh_initialization=False, full_state_resume=True, require_discord_monitor=True,
            parent_evaluation_before_training=True,
            resume="CLI --resume-checkpoint plus config strict/load_optimizer/restart_lr_schedule=true, auto_reset_critic=false; no compatibility bypass.",
            league_restoration="Hash guard every source, copy checkpoint/metadata to child-only destinations, rewrite snapshot paths using path_rewrites, write child active state files; never mutate parent.",
            parent_results="Evaluate immutable <panel>/<arm.name>/parent_registration.json first; retain its provenance. Final registration includes parent+future checkpoints; never relabel parent traces.",
            heldout_results="heldout/<arm.name>/eval/<arm.id>_p<update:06d>/<sample|argmax>/result.json",
            report="Separate strategy_report.json and strategy_report_heldout.json; never pool panel counts.",
            state_restoration="Model, optimizer, epoch/global_step, trainer RNG/coordinator state and epoch*15360 sampled-row reward clock restored (potential .3/exploration .15); worker env/RNG/live episodes/rollouts reset, so exact trajectory resume=false.",
            lr_schedule="Smoke restarts a short 10-update cosine, not production LR parity; full continuation restarts one 9765-update cosine.")
    artifacts[output / "registration.json"] = json_bytes(registration)
    output.mkdir(parents=True, exist_ok=False)
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    print(portable(output / "registration.json"))
    print(f"Strategy-first diagnostic: {len(arms)} {'full-state continuation' if continuation else 'fresh'} CUDA arms, {rows} cumulative rows/arm, {registration['expected_evaluation_games']} evaluation games; no production promotion or 1B launch.")
    return registration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--retention", action="store_true", help="Register the authorized 50M STRATEGIC50/RANDOM50 confirmation")
    modes.add_argument("--continuation", action="store_true", help="Register strict RANDOM50 p3256 to p13021 full-state continuations")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.retention and output == RESULTS:
        output = RESULTS.with_name("strategy_retention_v1")
    if args.continuation and output == RESULTS:
        output = RESULTS.with_name("random50_continuation_v1")
    build(output, smoke=args.smoke, retention=args.retention, continuation=args.continuation)


if __name__ == "__main__":
    main()
