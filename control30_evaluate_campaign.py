#!/usr/bin/env python3
"""Launch or strictly reanalyze a registered control-continuation evaluation panel."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


SHARDS = 8
SUITES = ("fixed", "free_draft")
MODES = ("sample", "argmax")
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 4_315_015
DECK_POOL_SHA256 = "4df6ca9109167ef7c32729a82dace4a7a5b4ac50afad45f08fbce3d3e8f34b90"
ORIGINAL_TASKS_CANONICAL_SHA256 = "f8cd84259bf3acd5c6826a4b1eac598a126d7dedede8e279ff121aa6b72a0fac"
OPPONENT_IDS = ("p021000", "p060000", "control_start")
HISTORICAL_OPPONENT_HASHES = {
    "p021000": "c04062e8b2f529c518422e8595b22184baf62be11f78ad69b21b525744336e1b",
    "p060000": "3ba834ffc9cce5a9827ee0df38bc5eaecd6b8393ac814cebedd5fb7e22cb3237",
}
PANEL_IDS = ("step10m", "step20m", "step30m")
TASKS_PER_CHECKPOINT = 1_350
GAMES_PER_CHECKPOINT = TASKS_PER_CHECKPOINT * len(MODES)
EXPECTED_CAMPAIGN_GAMES = GAMES_PER_CHECKPOINT * len(PANEL_IDS)
BEHAVIOR_FIELDS = (
    "battle_steps",
    "spell_cards",
    "weapon_cards",
    "normal_cards",
    "unique_main_cards",
    "mean_main_cost",
    "sprout_cards",
    "spell_plays",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def checked_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    return path


def add_hash(hashes: dict[str, str], path: Path, expected: str | None = None) -> str:
    path = checked_file(path, "registered input")
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Hash mismatch for {path}: expected {expected}, got {actual}")
    previous = hashes.setdefault(str(path), actual)
    if previous != actual:
        raise ValueError(f"Conflicting registered hash for {path}")
    return actual


def verify_hashes(hashes: dict[str, str]) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Registration has no source hashes")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute() or digest(checked_file(path, "frozen source")) != expected:
            raise ValueError(f"Frozen input drift: {name}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def clone_for_control_start(task: dict[str, Any]) -> dict[str, Any]:
    clone = dict(task)
    if task["opponent_id"] != "p021000":
        raise ValueError("Only p021000 tasks may seed control_start clones")
    if task["block_id"].count(":p021000") != 1 or not task["block_id"].endswith(":p021000"):
        raise ValueError(f"p021000 block id has unexpected shape: {task['block_id']}")
    if task["task_id"].count(":p021000:") != 1:
        raise ValueError(f"p021000 task id has unexpected shape: {task['task_id']}")
    clone["opponent_id"] = "control_start"
    clone["block_id"] = task["block_id"][: -len("p021000")] + "control_start"
    clone["task_id"] = task["task_id"].replace(":p021000:", ":control_start:")
    return clone


def validate_tasks(tasks: Any) -> None:
    if not isinstance(tasks, list) or len(tasks) != TASKS_PER_CHECKPOINT:
        raise ValueError("The full 1350-task control-continuation matrix is required")
    required = {
        "task_id", "block_id", "suite", "seed", "candidate_seat", "candidate_gate",
        "candidate_leader", "candidate_deck_index", "reference_deck_index", "opponent_id",
    }
    ids: set[str] = set()
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    suite_counts = Counter()
    opponent_counts = Counter()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != required:
            raise ValueError("Evaluation task object schema drift")
        if task["task_id"] in ids:
            raise ValueError(f"Duplicate task id: {task['task_id']}")
        ids.add(task["task_id"])
        opponent_id = task["opponent_id"]
        if task["suite"] not in SUITES or opponent_id not in OPPONENT_IDS:
            raise ValueError(f"Invalid task suite/opponent: {task['task_id']}")
        if task["candidate_seat"] not in (0, 1):
            raise ValueError(f"Invalid task seat: {task['task_id']}")
        expected_task_id = f"{task['block_id']}:seat{task['candidate_seat']}"
        if task["task_id"] != expected_task_id or task["block_id"].split(":")[-1] != opponent_id:
            raise ValueError(f"Task/block opponent token drift: {task['task_id']}")
        blocks[task["block_id"]].append(task)
        suite_counts[task["suite"]] += 1
        opponent_counts[opponent_id] += 1
    if suite_counts != Counter({"free_draft": 864, "fixed": 486}):
        raise ValueError(f"Task matrix suite counts drifted: {dict(suite_counts)}")
    if opponent_counts != Counter({opponent_id: 450 for opponent_id in OPPONENT_IDS}):
        raise ValueError(f"Task matrix opponent counts drifted: {dict(opponent_counts)}")
    for block_id, rows in blocks.items():
        if len(rows) != 2 or {row["candidate_seat"] for row in rows} != {0, 1}:
            raise ValueError(f"Block is not seat-paired: {block_id}")
        normalized = [{key: value for key, value in row.items() if key not in ("task_id", "candidate_seat")} for row in rows]
        if normalized[0] != normalized[1]:
            raise ValueError(f"Seat-paired task drift: {block_id}")
    original = [task for task in tasks if task["opponent_id"] != "control_start"]
    if canonical_sha256(original) != ORIGINAL_TASKS_CANONICAL_SHA256:
        raise ValueError("The original 900 registered tasks changed or were reordered")
    expected_clones = {task["task_id"]: task for task in map(clone_for_control_start, (
        task for task in original if task["opponent_id"] == "p021000"
    ))}
    actual_clones = {
        task["task_id"]: task for task in tasks if task["opponent_id"] == "control_start"
    }
    if actual_clones != expected_clones:
        raise ValueError("control_start tasks are not exact p021000 clones")


def load_experiment_contract(
    experiment_root: Path,
    runtime: Path,
    panel_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    experiment_path = checked_file(experiment_root / "experiment.json", "experiment plan")
    common_parent_path = checked_file(experiment_root / "common_parent.json", "common parent")
    experiment = load_json(experiment_path)
    if (
        not isinstance(experiment, dict)
        or experiment.get("schema_id") != "azuki.control_continuation"
        or experiment.get("schema_version") != 1
    ):
        raise ValueError("Unsupported control-continuation experiment schema")
    registered_runtime = experiment.get("runtime")
    if registered_runtime is not None and Path(str(registered_runtime)).expanduser().resolve() != runtime:
        raise ValueError("Experiment runtime does not match --runtime")
    continuation = experiment.get("continuation")
    expected_stages = [
        {"id": "step10m", "additional_learner_steps": 10_000_000},
        {"id": "step20m", "additional_learner_steps": 20_000_000},
        {"id": "step30m", "additional_learner_steps": 30_000_000},
    ]
    if continuation != {
        "arm": "control",
        "additional_learner_steps": 30_000_000,
        "stages": expected_stages,
    }:
        raise ValueError("Experiment continuation contract drift")
    evaluation = experiment.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Experiment evaluation contract is missing")
    if (
        evaluation.get("panels") != list(PANEL_IDS)
        or evaluation.get("games_per_checkpoint") != GAMES_PER_CHECKPOINT
        or evaluation.get("expected_games") != EXPECTED_CAMPAIGN_GAMES
        or panel_id not in PANEL_IDS
    ):
        raise ValueError("Experiment panel/game-count contract drift")
    tasks_sha256 = evaluation.get("tasks_sha256")
    if not isinstance(tasks_sha256, str) or len(tasks_sha256) != 64:
        raise ValueError("Experiment must pin evaluation.tasks_sha256")
    raw_opponents = evaluation.get("opponents")
    if not isinstance(raw_opponents, dict) or set(raw_opponents) != set(OPPONENT_IDS):
        raise ValueError(f"Experiment opponents must be exactly {OPPONENT_IDS}")
    common_parent = load_json(common_parent_path)
    for opponent_id, spec in raw_opponents.items():
        if not isinstance(spec, dict) or set(spec) != {"checkpoint", "checkpoint_sha256"}:
            raise ValueError(f"Malformed experiment opponent: {opponent_id}")
        checkpoint = Path(str(spec["checkpoint"])).expanduser()
        expected_hash = spec["checkpoint_sha256"]
        if not checkpoint.is_absolute() or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Opponent path/hash is not strictly pinned: {opponent_id}")
        historical_hash = HISTORICAL_OPPONENT_HASHES.get(opponent_id)
        if historical_hash is not None and expected_hash != historical_hash:
            raise ValueError(f"Historical opponent hash drift: {opponent_id}")
    start_spec = raw_opponents["control_start"]
    if (
        start_spec["checkpoint"] != common_parent.get("checkpoint")
        or start_spec["checkpoint_sha256"] != common_parent.get("checkpoint_sha256")
        or common_parent.get("learner_global_step") != 26_349_346
        or common_parent.get("update") != 2_815
    ):
        raise ValueError("control_start is not the registered final-v3 control parent")
    return experiment, evaluation, experiment_path, common_parent_path


def normalize_manifest_sources(runtime: Path, manifest: dict[str, Any]) -> dict[str, str]:
    raw = manifest.get("source_sha256")
    if (
        manifest.get("schema_version") != 1
        or Path(str(manifest.get("runtime", ""))).expanduser().resolve() != runtime
    ):
        raise ValueError("runtime_manifest.json does not pin the selected frozen runtime")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("runtime_manifest.json must contain a nonempty source_sha256 map")
    normalized: dict[str, str] = {}
    for name, expected in raw.items():
        path = Path(name)
        if not path.is_absolute():
            path = runtime / path
        add_hash(normalized, path, expected)
    return normalized


def verify_atomic_manifest(manifest_path: Path, hashes: dict[str, str]) -> None:
    manifest = load_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"Atomic checkpoint manifest has no artifacts: {manifest_path}")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or not isinstance(artifact.get("sha256"), str):
            raise ValueError(f"Malformed atomic checkpoint artifact in {manifest_path}")
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        add_hash(hashes, artifact_path, artifact["sha256"])


def normalize_checkpoints(raw: Any, hashes: dict[str, str]) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    if not isinstance(raw, dict) or tuple(raw) != ("control",):
        raise ValueError("Checkpoint map must have the sole key 'control'")
    keys = ("control",)
    result: dict[str, dict[str, Any]] = {}
    for key in keys:
        incoming = raw[key]
        if not isinstance(incoming, dict):
            raise ValueError(f"Checkpoint spec must be an object: {key}")
        for field in ("checkpoint", "config", "manifest"):
            if not isinstance(incoming.get(field), str):
                raise ValueError(f"Checkpoint {key} is missing {field}")
        spec = dict(incoming)
        checkpoint = checked_file(Path(spec["checkpoint"]), f"{key} checkpoint")
        config = checked_file(Path(spec["config"]), f"{key} evaluation config")
        manifest = checked_file(Path(spec["manifest"]), f"{key} atomic manifest")
        metadata = checked_file(Path(str(checkpoint) + ".meta.json"), f"{key} checkpoint metadata")
        spec.update(
            checkpoint=str(checkpoint),
            checkpoint_sha256=add_hash(hashes, checkpoint, spec.get("checkpoint_sha256")),
            config=str(config),
            config_sha256=add_hash(hashes, config, spec.get("config_sha256")),
            manifest=str(manifest),
            manifest_sha256=add_hash(hashes, manifest, spec.get("manifest_sha256")),
            metadata=str(metadata),
            metadata_sha256=add_hash(hashes, metadata, spec.get("metadata_sha256")),
        )
        verify_atomic_manifest(manifest, hashes)
        result[key] = spec
    return result, keys


def build_registration(
    experiment_root: Path,
    runtime: Path,
    panel_id: str,
    checkpoint_map_path: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    experiment, evaluation, experiment_path, common_parent_path = load_experiment_contract(
        experiment_root, runtime, panel_id
    )
    runtime_manifest_path = checked_file(experiment_root / "runtime_manifest.json", "runtime manifest")
    runtime_manifest = load_json(runtime_manifest_path)
    hashes = normalize_manifest_sources(runtime, runtime_manifest)
    tasks_path = checked_file(experiment_root / "evaluation_tasks.json", "evaluation tasks")
    deck_pool = checked_file(experiment_root / "deck_pool.json", "deck pool")
    add_hash(hashes, tasks_path, evaluation["tasks_sha256"])
    add_hash(hashes, deck_pool, DECK_POOL_SHA256)
    add_hash(hashes, experiment_path)
    add_hash(hashes, common_parent_path)
    add_hash(hashes, runtime_manifest_path)
    add_hash(hashes, checkpoint_map_path)
    add_hash(hashes, Path(__file__))
    evaluator = checked_file(runtime / "train-ablation-1781126582/evaluate_prebuilt_mix.py", "frozen evaluator")
    dispatcher = checked_file(runtime / "train-ablation-1781126582/run_prefix_paired_eval.py", "frozen dispatcher")
    catalog = checked_file(runtime / "python/config/policy_card_metadata_v1.json", "card metadata")
    add_hash(hashes, evaluator)
    add_hash(hashes, dispatcher)
    add_hash(hashes, catalog)
    tasks = load_json(tasks_path)
    validate_tasks(tasks)
    checkpoints, checkpoint_keys = normalize_checkpoints(load_json(checkpoint_map_path), hashes)
    opponent_config = checked_file(
        experiment_root / "configs/control_evaluation.ini", "control evaluation config"
    )
    add_hash(hashes, opponent_config)
    if Path(checkpoints["control"]["config"]) != opponent_config:
        raise ValueError("Control checkpoint and every opponent must use configs/control_evaluation.ini")
    opponents: dict[str, dict[str, Any]] = {}
    for opponent_id in OPPONENT_IDS:
        planned = evaluation["opponents"][opponent_id]
        checkpoint = checked_file(Path(planned["checkpoint"]), f"{opponent_id} checkpoint")
        metadata = checked_file(Path(str(checkpoint) + ".meta.json"), f"{opponent_id} metadata")
        opponents[opponent_id] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": add_hash(hashes, checkpoint, planned["checkpoint_sha256"]),
            "config": str(opponent_config),
            "metadata": str(metadata),
            "metadata_sha256": add_hash(hashes, metadata),
            "provenance": (
                "unchanged retained frozen league opponent"
                if opponent_id in HISTORICAL_OPPONENT_HASHES
                else "frozen final-v3 control continuation start"
            ),
        }
    registration = {
        "schema_id": "azuki.prebuilt_mix_evaluation",
        "schema_version": 1,
        "step_cap": 1200,
        "panel_id": panel_id,
        "runtime": str(runtime),
        "deck_pool": str(deck_pool),
        "source_sha256": hashes,
        "checkpoints": checkpoints,
        "opponents": opponents,
        "tasks": tasks,
        "campaign_provenance": {
            "experiment": str(experiment_path),
            "experiment_sha256": digest(experiment_path),
            "common_parent": str(common_parent_path),
            "common_parent_sha256": digest(common_parent_path),
            "runtime_manifest": str(runtime_manifest_path),
            "runtime_manifest_sha256": digest(runtime_manifest_path),
            "checkpoint_map": str(checkpoint_map_path),
            "checkpoint_map_sha256": digest(checkpoint_map_path),
            "evaluation_tasks": str(tasks_path),
            "evaluation_tasks_sha256": evaluation["tasks_sha256"],
            "deck_pool_sha256": DECK_POOL_SHA256,
            "evaluator": str(evaluator),
            "evaluator_sha256": digest(evaluator),
            "dispatcher": str(dispatcher),
            "dispatcher_sha256": digest(dispatcher),
            "shards": SHARDS,
            "modes": list(MODES),
            "suites": list(SUITES),
            "opponent_ids": list(OPPONENT_IDS),
            "tasks_per_checkpoint": TASKS_PER_CHECKPOINT,
            "games_per_checkpoint": GAMES_PER_CHECKPOINT,
            "campaign_expected_games": EXPECTED_CAMPAIGN_GAMES,
        },
    }
    verify_hashes(hashes)
    return registration, checkpoint_keys


def validate_registration(
    registration: Any,
    experiment_root: Path,
    runtime: Path,
    panel_id: str,
    checkpoint_map_path: Path,
) -> tuple[str, ...]:
    if not isinstance(registration, dict) or registration.get("schema_id") != "azuki.prebuilt_mix_evaluation" or registration.get("schema_version") != 1:
        raise ValueError("Unsupported registration schema")
    experiment, evaluation, experiment_path, common_parent_path = load_experiment_contract(
        experiment_root, runtime, panel_id
    )
    if registration.get("step_cap") != 1200 or registration.get("panel_id") != panel_id:
        raise ValueError("Registration panel or step cap mismatch")
    if Path(registration.get("runtime", "")).resolve() != runtime:
        raise ValueError("Registration runtime mismatch")
    if Path(registration.get("deck_pool", "")).resolve() != (experiment_root / "deck_pool.json").resolve():
        raise ValueError("Registration deck pool mismatch")
    validate_tasks(registration.get("tasks"))
    provenance = registration.get("campaign_provenance", {})
    required_provenance = {
        "experiment_sha256": digest(experiment_path),
        "common_parent_sha256": digest(common_parent_path),
        "checkpoint_map_sha256": digest(checkpoint_map_path),
        "evaluation_tasks_sha256": evaluation["tasks_sha256"],
        "deck_pool_sha256": DECK_POOL_SHA256,
        "tasks_per_checkpoint": TASKS_PER_CHECKPOINT,
        "games_per_checkpoint": GAMES_PER_CHECKPOINT,
        "campaign_expected_games": EXPECTED_CAMPAIGN_GAMES,
    }
    if any(provenance.get(key) != value for key, value in required_provenance.items()):
        raise ValueError("Registered immutable input or static coverage contract mismatch")
    keys = tuple(registration.get("checkpoints", {}))
    if keys != ("control",):
        raise ValueError("Registration checkpoint role must be solely 'control'")
    opponents = registration.get("opponents")
    if not isinstance(opponents, dict) or tuple(opponents) != OPPONENT_IDS:
        raise ValueError("Registration opponent roles/order are invalid")
    expected_config = (experiment_root / "configs/control_evaluation.ini").resolve()
    verify_hashes(registration.get("source_sha256"))
    for collection in ("checkpoints", "opponents"):
        for key, spec in registration[collection].items():
            if digest(Path(spec["checkpoint"])) != spec["checkpoint_sha256"]:
                raise ValueError(f"Registered {collection} checkpoint drift: {key}")
            if Path(spec["config"]).resolve() != expected_config:
                raise ValueError(f"Wrong control evaluation config: {collection}.{key}")
            if str(expected_config) not in registration["source_sha256"]:
                raise ValueError(f"Unpinned evaluation config: {key}")
            metadata = Path(str(spec["checkpoint"]) + ".meta.json").resolve()
            if str(metadata) not in registration["source_sha256"]:
                raise ValueError(f"Unpinned checkpoint metadata: {key}")
    for opponent_id, planned in evaluation["opponents"].items():
        spec = opponents[opponent_id]
        if spec["checkpoint"] != str(Path(planned["checkpoint"]).resolve()) or spec["checkpoint_sha256"] != planned["checkpoint_sha256"]:
            raise ValueError(f"Registered opponent differs from experiment plan: {opponent_id}")
    return keys


def expected_jobs(panel_root: Path, checkpoint_keys: tuple[str, ...]) -> list[tuple[str, str, str, int, Path]]:
    return [
        (suite, mode, key, shard, panel_root / suite / mode / key / f"shard{shard:02d}.jsonl")
        for suite in SUITES for mode in MODES for key in checkpoint_keys for shard in range(SHARDS)
    ]


def run_logged(command: list[str], output: Path, runtime: Path, environment: dict[str, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output.with_suffix(".stdout.log")
    stderr_path = output.with_suffix(".stderr.log")
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=runtime, env=environment, stdout=stdout, stderr=stderr, check=False)
    if result.returncode:
        raise RuntimeError(
            f"Evaluator failed with status {result.returncode}; retained {stdout_path}, {stderr_path}, and any partial {output}"
        )


def evaluation_environment(runtime: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AZK_")}
    environment.update(
        PYTHONPATH=os.pathsep.join(
            (str(runtime / "build/python/src"), str(runtime / "python/src"), str(runtime / "train-ablation-1781126582"))
        ),
        LD_LIBRARY_PATH=str(runtime / "build/_deps/flecs_src-build"),
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="43",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        TORCH_NUM_THREADS="1",
    )
    return environment


def launch_evaluations(
    panel_root: Path,
    runtime: Path,
    registration_path: Path,
    checkpoint_keys: tuple[str, ...],
    workers: int,
) -> list[Path]:
    jobs = expected_jobs(panel_root, checkpoint_keys)
    expected_artifacts = {
        artifact
        for _, _, _, _, output in jobs
        for artifact in (output, output.with_suffix(".stdout.log"), output.with_suffix(".stderr.log"))
    }
    collisions = [path for path in expected_artifacts if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to reuse partial evaluation artifact: {collisions[0]}")
    # Resolving the executable symlink would select the system Python, not this venv.
    python = runtime / ".venv/bin/python"
    checked_file(python, "runtime Python")
    evaluator = checked_file(runtime / "train-ablation-1781126582/evaluate_prebuilt_mix.py", "frozen evaluator")
    environment = evaluation_environment(runtime)
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for suite, mode, key, shard, output in jobs:
            command = [
                str(python), str(evaluator), "--registration", str(registration_path),
                "--checkpoint-key", key, "--suite", suite, "--mode", mode,
                "--shards", str(SHARDS), "--shard-index", str(shard), "--out", str(output),
            ]
            futures[pool.submit(run_logged, command, output, runtime, environment)] = output
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [output for _, _, _, _, output in jobs]


def arm_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation slice")
    values = [row["score"] for row in rows]
    stats = {
        "games": len(rows),
        "wins": values.count(1.0),
        "draws": values.count(0.5),
        "losses": values.count(0.0),
        "score": float(np.mean(values)),
    }
    stats.update({f"{field}_mean": float(np.mean([row[field] for row in rows])) for field in BEHAVIOR_FIELDS})
    stats["gate_effects_per_game"] = float(np.mean([row["ability_outcomes"]["gate"] for row in rows]))
    stats["leader_effects_per_game"] = float(np.mean([row["ability_outcomes"]["leader"] for row in rows]))
    return stats


def panel_statistics(selected: list[dict[str, Any]], checkpoint_keys: tuple[str, ...]) -> dict[str, Any]:
    scores = {key: {row["task_id"]: row for row in selected if row["arm"] == key} for key in checkpoint_keys}
    result: dict[str, Any] = {"arms": {key: arm_statistics(list(scores[key].values())) for key in checkpoint_keys}}
    if len(checkpoint_keys) == 1:
        result["absolute_only"] = True
        return result
    control, candidate = checkpoint_keys
    if set(scores[control]) != set(scores[candidate]):
        raise ValueError("Unpaired control/candidate comparison")
    blocks: dict[str, list[float]] = defaultdict(list)
    for task_id, control_row in scores[control].items():
        blocks[control_row["block_id"]].append(scores[candidate][task_id]["score"] - control_row["score"])
    differences = np.asarray([np.mean(values) for _, values in sorted(blocks.items())], dtype=np.float64)
    if not len(differences):
        raise ValueError("No paired worlds in comparison")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.choice(differences, size=(BOOTSTRAP_RESAMPLES, len(differences)), replace=True).mean(axis=1)
    result.update(
        paired_worlds=len(blocks),
        treatment_minus_control=float(differences.mean()),
        paired_world_bootstrap_95_ci=np.quantile(resamples, [0.025, 0.975]).tolist(),
    )
    return result


def selected_task_ids(registration: dict[str, Any], suite: str, shard: int) -> set[str]:
    tasks = [task for task in registration["tasks"] if task["suite"] == suite]
    blocks = list(dict.fromkeys(task["block_id"] for task in tasks))
    selected_blocks = set(blocks[shard::SHARDS])
    return {task["task_id"] for task in tasks if task["block_id"] in selected_blocks}


def read_trace(
    path: Path,
    registration: dict[str, Any],
    registration_sha256: str,
    suite: str,
    mode: str,
    checkpoint_key: str,
    shard: int,
    task_by_id: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    if not data or not data.endswith(b"\n"):
        raise ValueError(f"Empty or truncated trace: {path}")
    initial_hash = hashlib.sha256(data).hexdigest()
    rows: list[dict[str, Any]] = []
    file_task_ids: set[str] = set()
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Malformed trace {path}:{line_number}: {error}") from error
        if record.get("trace_schema_version") != 2:
            raise ValueError(f"Wrong trace schema in {path}:{line_number}")
        pair = record.get("paired_eval")
        if not isinstance(pair, dict):
            raise ValueError(f"Missing paired_eval in {path}:{line_number}")
        task_id = pair.get("task_id")
        task = task_by_id.get(task_id)
        if task is None or task_id in file_task_ids:
            raise ValueError(f"Unknown/duplicate task in {path}: {task_id}")
        file_task_ids.add(task_id)
        if any(pair.get(key) != value for key, value in task.items()):
            raise ValueError(f"Evaluation task object drift: {task_id}")
        if (
            pair.get("checkpoint_key") != checkpoint_key
            or pair.get("mode") != mode
            or pair.get("shards") != SHARDS
            or pair.get("shard_index") != shard
            or pair.get("registration_sha256") != registration_sha256
        ):
            raise ValueError(f"Trace registration/job identity drift: {path}:{line_number}")
        if pair.get("candidate") != registration["checkpoints"][checkpoint_key] or pair.get("opponent") != registration["opponents"][task["opponent_id"]]:
            raise ValueError(f"Trace model provenance drift: {task_id}")
        outcome = record.get("outcome", {})
        if (
            not pair.get("complete")
            or pair.get("validation_errors") != []
            or pair.get("smoke_task_limit") is not None
            or not outcome.get("terminated")
            or outcome.get("truncated")
            or pair.get("candidate_score") not in (0, 0.5, 1)
        ):
            raise ValueError(f"Incomplete, invalid, smoke, or truncated result: {task_id}")
        seat = task["candidate_seat"]
        decks = record.get("decks")
        steps = record.get("steps")
        if not isinstance(decks, list) or len(decks) != 2 or not isinstance(steps, list):
            raise ValueError(f"Incomplete trace payload: {task_id}")
        main = decks[seat].get("main")
        if not isinstance(main, list) or len(main) != 50 or any(code not in catalog for code in main):
            raise ValueError(f"Invalid candidate deck payload: {task_id}")
        own_steps = [step for step in steps if step.get("p") == seat]
        abilities = outcome.get("ability_outcomes")
        if not isinstance(abilities, list) or len(abilities) != 2:
            raise ValueError(f"Missing ability outcomes: {task_id}")
        rows.append(
            {
                **task,
                "arm": checkpoint_key,
                "mode": mode,
                "score": float(pair["candidate_score"]),
                "element": catalog[task["candidate_gate"]]["element"],
                "battle_steps": record["battle_steps"],
                "draft_steps": record["draft_steps"],
                "spell_cards": sum(catalog[code]["card_type"] == "SPELL" for code in main),
                "weapon_cards": sum(catalog[code]["card_type"] == "WEAPON" for code in main),
                "normal_cards": sum(catalog[code]["element"] == "NORMAL" for code in main),
                "unique_main_cards": len(set(main)),
                "mean_main_cost": sum(catalog[code]["ikz_cost"] for code in main) / 50,
                "sprout_cards": main.count("STT03-017"),
                "spell_plays": sum(step.get("a", [None])[0] == 8 for step in own_steps),
                "ability_outcomes": abilities[seat],
            }
        )
    if file_task_ids != selected_task_ids(registration, suite, shard):
        raise ValueError(f"Exact shard coverage mismatch: {path}")
    if digest(path) != initial_hash:
        raise ValueError(f"Trace changed during analysis: {path}")
    return rows, initial_hash


def summarize(
    panel_root: Path,
    registration_path: Path,
    registration: dict[str, Any],
    checkpoint_keys: tuple[str, ...],
) -> dict[str, Any]:
    registration_sha256 = digest(registration_path)
    catalog_document = load_json(Path(registration["runtime"]) / "python/config/policy_card_metadata_v1.json")
    catalog = {record["card_code"]: record for record in catalog_document["records"]}
    task_by_id = {task["task_id"]: task for task in registration["tasks"]}
    jobs = expected_jobs(panel_root, checkpoint_keys)
    expected_paths = {output.resolve() for _, _, _, _, output in jobs}
    actual_paths = {path.resolve() for path in panel_root.rglob("*.jsonl")}
    if actual_paths != expected_paths:
        raise ValueError(f"Trace file set mismatch: missing={len(expected_paths-actual_paths)}, extra={len(actual_paths-expected_paths)}")
    games: list[dict[str, Any]] = []
    trace_hashes: dict[str, str] = {}
    seen: set[tuple[str, str, str]] = set()
    for suite, mode, checkpoint_key, shard, path in jobs:
        rows, trace_hash = read_trace(
            path, registration, registration_sha256, suite, mode, checkpoint_key, shard, task_by_id, catalog
        )
        trace_hashes[str(path.resolve())] = trace_hash
        for row in rows:
            identity = (checkpoint_key, mode, row["task_id"])
            if identity in seen:
                raise ValueError(f"Duplicate evaluation result: {identity}")
            seen.add(identity)
        games.extend(rows)
    expected = {
        (key, mode, task["task_id"])
        for key in checkpoint_keys for mode in MODES for task in registration["tasks"]
    }
    if seen != expected:
        raise ValueError(f"Evaluation coverage mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}")
    if len(games) != GAMES_PER_CHECKPOINT:
        raise ValueError(
            f"Panel must contain exactly {GAMES_PER_CHECKPOINT} registered games, got {len(games)}"
        )
    panels: dict[str, Any] = {}
    for suite in SUITES:
        for mode in MODES:
            selected = [row for row in games if row["suite"] == suite and row["mode"] == mode]
            panel = panel_statistics(selected, checkpoint_keys)
            panel["by_opponent"] = {
                value: panel_statistics([row for row in selected if row["opponent_id"] == value], checkpoint_keys)
                for value in registration["opponents"]
            }
            historical = [
                row for row in selected if row["opponent_id"] in HISTORICAL_OPPONENT_HASHES
            ]
            frozen_start = [
                row for row in selected if row["opponent_id"] == "control_start"
            ]
            panel["historical_reference_panel"] = panel_statistics(
                historical, checkpoint_keys
            )
            panel["direct_frozen_start_panel"] = panel_statistics(
                frozen_start, checkpoint_keys
            )
            panel["by_element"] = {
                value: panel_statistics([row for row in selected if row["element"] == value], checkpoint_keys)
                for value in sorted({row["element"] for row in selected})
            }
            panels[f"{suite}_{mode}"] = panel
    provenance = {
        **registration["campaign_provenance"],
        "runtime": registration["runtime"],
        "source_sha256": registration["source_sha256"],
        "checkpoints": registration["checkpoints"],
        "opponents": registration["opponents"],
    }
    result = {
        "schema_version": 1,
        "panel_id": registration["panel_id"],
        "checkpoint_roles": list(checkpoint_keys),
        "completed_games": len(games),
        "incomplete_games": 0,
        "registered_games_per_checkpoint": GAMES_PER_CHECKPOINT,
        "registered_campaign_games": EXPECTED_CAMPAIGN_GAMES,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "uncertainty_scope": "evaluation worlds conditional on registered checkpoints; one training seed; exploratory slices uncorrected",
        "registration": str(registration_path),
        "registration_sha256": registration_sha256,
        "trace_sha256": trace_hashes,
        "provenance": provenance,
        "panels": panels,
    }
    save_json(panel_root / "paired_scores.json", games)
    save_json(panel_root / "comparison.json", result)
    if digest(registration_path) != registration_sha256:
        raise ValueError("Registration changed during analysis")
    for path, expected_hash in trace_hashes.items():
        if digest(Path(path)) != expected_hash:
            raise ValueError(f"Trace changed after analysis: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--checkpoint-map", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.panel_id or args.panel_id in (".", "..") or Path(args.panel_id).name != args.panel_id:
        raise ValueError("panel-id must be one nonempty path component")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    experiment_root = args.experiment_root.expanduser().resolve()
    runtime = args.runtime.expanduser().resolve()
    checkpoint_map_path = checked_file(args.checkpoint_map, "checkpoint map")
    if not experiment_root.is_dir() or not runtime.is_dir():
        raise ValueError("experiment-root and runtime must be existing directories")
    panel_root = experiment_root / "evaluation" / args.panel_id
    registration_path = panel_root / "registration.json"
    if args.analyze_only:
        registration = load_json(checked_file(registration_path, "panel registration"))
        checkpoint_keys = validate_registration(
            registration, experiment_root, runtime, args.panel_id, checkpoint_map_path
        )
    else:
        if registration_path.exists():
            raise FileExistsError(f"Panel is already registered; refusing partial reuse: {registration_path}")
        panel_root.mkdir(parents=True, exist_ok=True)
        if any(panel_root.iterdir()):
            raise FileExistsError(f"Panel directory is not empty; refusing partial reuse: {panel_root}")
        registration, checkpoint_keys = build_registration(
            experiment_root, runtime, args.panel_id, checkpoint_map_path
        )
        save_json(registration_path, registration, exclusive=True)
        validate_registration(registration, experiment_root, runtime, args.panel_id, checkpoint_map_path)
        launch_evaluations(panel_root, runtime, registration_path, checkpoint_keys, args.workers)
    result = summarize(panel_root, registration_path, registration, checkpoint_keys)
    print(
        json.dumps(
            {
                "panel_id": args.panel_id,
                "completed_games": result["completed_games"],
                "registration_sha256": result["registration_sha256"],
                "comparison": str(panel_root / "comparison.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
