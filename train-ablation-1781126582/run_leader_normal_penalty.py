#!/usr/bin/env python3
"""Run one fresh S43 penalty learner; existing RANDOM50 controls are read-only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from build_leader_normal_penalty import (
    ANNEAL_ROWS, DESIGN, EXPECTED_LEADERS, FAMILY, FULL_ROWS, FULL_UPDATES,
    PROBES, ROOT, ROWS_PER_UPDATE, SMOKE_UPDATES, portable, read_config, resolve_registered,
)
from evaluate_reward_screens import _checkpoint_from_manifest
from monitor_local_production import _atomic_json
from report_leader_normal_penalty import report
from run_prefix_followup import summarize
from run_prefix_paired_eval import validate_config as validate_evaluation_config
from run_strategy_discovery import read_metrics, sha256, verify_registration_files
from run_strategy_recipe import Campaign, stop

PYTHON = ROOT / ".venv/bin/python"


def validate(registration: dict) -> None:
    if (registration.get("schema_id") != "azuki.ablation_registration" or registration.get("schema_version") != 1
            or registration.get("family") != FAMILY or registration.get("design") != DESIGN
            or registration.get("status") != "registered" or registration.get("scope") != "diagnostic_only"
            or registration.get("production_qualified") is not False):
        raise ValueError("Expected the fresh single-seed S43 Normal-penalty registration")
    verify_registration_files(registration)
    for pattern, expected in registration["expected_glob_membership"].items():
        actual = sorted(str(path.relative_to(ROOT)) for path in ROOT.glob(pattern) if path.is_file())
        if actual != expected:
            raise ValueError(f"Source membership drift: {pattern}")
    smoke = registration["smoke"] is True
    updates = SMOKE_UPDATES if smoke else FULL_UPDATES
    rows = updates * ROWS_PER_UPDATE
    baseline_updates = [PROBES[0]] if smoke else PROBES
    penalty_updates = [updates] if smoke else PROBES
    anneal_rows = rows if smoke else ANNEAL_ROWS
    if (registration.get("seeds") != [43] or registration.get("training_arms") != ["PENALTY_S43"]
            or registration.get("saved_control_arms") != ["CONTROL_S43"]
            or registration.get("total_new_training_rows") != rows
            or registration.get("total_timesteps") != rows or registration.get("total_updates") != updates
            or registration.get("runner_contract", {}).get("train_saved_controls") is not False
            or registration.get("runner_contract", {}).get("resume_checkpoint") is not False):
        raise ValueError("Only one fresh S43 learner is authorized; no control training or checkpoint resume")
    if any(registration["selection_contract"].get(key) is not False for key in
           ("automatic_selection", "automatic_promotion", "automatic_1b_launch", "winrate_gate")):
        raise ValueError("Automatic selection/promotion/1B and winrate gates are forbidden")
    arms = registration["arms"]
    if len(arms) != 2 or {arm["id"] for arm in arms} != {"CONTROL_S43", "PENALTY_S43"}:
        raise ValueError("Expected one saved control and one penalty learner")
    penalty = registration["penalty_config"]
    if (sha256(resolve_registered(penalty["path"])) != penalty["sha256"]
            or {code: spec["max_normal_fraction"] for code, spec in penalty["leaders"].items()} != EXPECTED_LEADERS
            or any(spec["weight"] != 1.0 for spec in penalty["leaders"].values())):
        raise ValueError("Leader penalty configuration drift")
    baseline = registration["baseline"]
    if sha256(resolve_registered(baseline["registration"])) != baseline["sha256"]:
        raise ValueError("Historical baseline registration changed")
    original = read_config(resolve_registered(baseline["config"]))
    allowed = {
        "base": {"tag", "jsonl_log"}, "train": {"total_timesteps", "data_dir"},
        "league": {"state_path", "promotion_state_path", "promotion_records_dir", "opponent_dir"},
        "process_env": {"azk_draft_normal_penalty_config", "azk_draft_normal_penalty_coef_initial",
                        "azk_draft_normal_penalty_coef_final", "azk_draft_normal_penalty_anneal_start_rows",
                        "azk_draft_normal_penalty_anneal_end_rows"},
        "logging": {"max_patterns"},
    }
    for arm in arms:
        control = arm["id"] == "CONTROL_S43"
        expected_updates = baseline_updates if control else penalty_updates
        expected_init = "saved_random50_checkpoints" if control else "fresh_matched_seed43_no_checkpoint_resume"
        if (arm["seed"] != 43 or arm["initialization"] != expected_init
                or arm["new_training_rows"] != (0 if control else rows)
                or arm["evaluation_updates"] != expected_updates
                or arm["total_updates"] != (FULL_UPDATES if control else updates)
                or arm["total_timesteps"] != (FULL_ROWS if control else rows)):
            raise ValueError(f"Arm initialization or budget drift: {arm['id']}")
        config = read_config(resolve_registered(arm["config"]))
        allowed_arm = {section: set(keys) for section, keys in allowed.items()}
        if smoke and not control:
            allowed_arm["train"].add("checkpoint_interval")
            allowed_arm["artifacts"] = {"recovery_interval_updates", "evaluation_interval_updates", "milestone_interval_updates"}
            allowed_arm["process_env"].update(("azk_potential_anneal_end_rows", "azk_exploration_anneal_end_rows"))
        for section in set(original.sections()) | set(config.sections()):
            before = dict(original[section]) if original.has_section(section) else {}
            after = dict(config[section]) if config.has_section(section) else {}
            for key in set(before) | set(after):
                if key not in allowed_arm.get(section, set()) and before.get(key) != after.get(key):
                    raise ValueError(f"Unauthorized RANDOM50 recipe change: {section}.{key}")
        if (config.has_section("resume") or config.getint("train", "seed") != 43
                or not config.getboolean("train", "seed_process_rngs")
                or config.getfloat("train", "learning_rate") != 0.003
                or dict(config["process_env"]) != {key.lower(): str(value) for key, value in arm["process_env"].items()}):
            raise ValueError("Fresh initialization, learning rate or registered process environment changed")
        if any(key.startswith("azk_resume_") for key in config["process_env"]):
            raise ValueError("Resume overrides are forbidden in a fresh run")
        expected_normal = {"coef_initial": 0.0 if control else 0.5, "coef_final": 0.0 if control else 0.075,
                           "anneal_start_rows": 0, "anneal_end_rows": anneal_rows}
        if any(config.getfloat("process_env", "azk_draft_normal_penalty_" + key) != value
               for key, value in expected_normal.items()):
            raise ValueError("Normal-penalty schedule drift")
        if resolve_registered(config.get("process_env", "azk_draft_normal_penalty_config")).resolve() != resolve_registered(penalty["path"]).resolve():
            raise ValueError("Arm uses a different leader configuration")
        if not control and any(config.getint("process_env", key) != anneal_rows for key in
                               ("azk_potential_anneal_end_rows", "azk_exploration_anneal_end_rows")):
            raise ValueError("All nonterminal schedules must share the registered annealing horizon")
        validate_evaluation_config(resolve_registered(arm["evaluation_config"]))
        folder = resolve_registered(arm["result_root"]).resolve()
        for section, key in (("base", "jsonl_log"), ("train", "data_dir"), ("league", "state_path"),
                             ("league", "promotion_state_path"), ("league", "promotion_records_dir"), ("league", "opponent_dir")):
            if not resolve_registered(config.get(section, key)).resolve().is_relative_to(folder):
                raise ValueError(f"Writable path escapes fresh output: {section}.{key}")
        if control:
            if set(arm["saved_checkpoints"]) != {str(update) for update in baseline_updates}:
                raise ValueError("Saved control checkpoint coverage differs")
            for spec in arm["saved_checkpoints"].values():
                if sha256(resolve_registered(spec["checkpoint"])) != spec["checkpoint_sha256"]:
                    raise ValueError("Saved RANDOM50 checkpoint changed")
        elif (arm["saved_checkpoints"] or arm["baseline_updates"] !=
              {str(update): baseline_updates[index] for index, update in enumerate(penalty_updates)}):
            raise ValueError("Fresh learner must not consume saved weights; baseline pairing drift")
    paired, heldout = registration["paired_template"], registration["heldout_template"]
    tasks = paired["tasks"]
    if len(tasks) != (4 if smoke else 512):
        raise ValueError("Unexpected paired evaluation task allocation")
    if heldout["tasks"] != [{**task, "task_id": "heldout_" + task["task_id"], "seed": task["seed"] + 1_000_000_000} for task in tasks]:
        raise ValueError("Heldout panel must change only IDs and world seeds")
    if any(heldout[key] != paired[key] for key in paired if key != "tasks"):
        raise ValueError("Heldout contexts, opponents or evaluator settings changed")
    games = sum(len(arm["evaluation_updates"]) for arm in arms) * 2 * len(tasks)
    if (registration["expected_evaluation_games_by_panel"] != {"paired": games, "heldout": games}
            or registration["expected_evaluation_games"] != games * 2):
        raise ValueError("Evaluation allocation differs from registered probes")


class NormalPenaltyCampaign(Campaign):
    def __init__(self, args: argparse.Namespace, registration: dict):
        super().__init__(args, registration)
        self.continuation = False
        self.panels = ("paired", "heldout")
        self.state.update(schema_id="azuki.leader_normal_penalty_status", objective="one fresh S43 penalty learner; saved controls only")

    def train(self, arm: dict) -> dict:
        if arm["id"] != "PENALTY_S43" or arm["initialization"] != "fresh_matched_seed43_no_checkpoint_resume":
            raise ValueError("Saved RANDOM50 controls must never be trained")
        validate(self.registration)
        state = super().train(arm)
        folder = resolve_registered(arm["result_root"])
        try:
            metrics = read_metrics(folder / "logs/production.jsonl")
            if not 0 <= metrics[0]["epoch"] <= 10:
                raise RuntimeError("Fresh training did not start at the beginning of the update clock")
            keys = sorted({key for row in metrics for key in row if key.startswith("losses/draft_episode_credit_normal_")})
            state["normal_penalty_telemetry"] = {
                key: {"first": values[0], "last": values[-1], "minimum": min(values), "maximum": max(values), "metric_windows": len(values)}
                for key in keys if (values := [row[key] for row in metrics if isinstance(row.get(key), (int, float))])
            }
            required = ("coefficient", "fraction_mean", "penalty_mean", "penalty_max")
            if any("losses/draft_episode_credit_normal_" + key not in state["normal_penalty_telemetry"] for key in required):
                raise RuntimeError("Required applied draft-penalty telemetry was not observed")
            state["initialization_verified"] = "fresh; no checkpoint resume; update clock starts at zero"
        except BaseException as exc:
            state.update(state="failed", error=str(exc), finished_at=time.time())
            _atomic_json(folder / "run_status.json", state)
            raise
        _atomic_json(folder / "run_status.json", state)
        return state

    def evaluate(self, arm: dict, training: dict | None, panel: str = "paired") -> None:
        verify_registration_files(self.registration)
        control = arm["initialization"] == "saved_random50_checkpoints"
        if control != (training is None):
            raise ValueError("Saved controls are evaluated directly; fresh learners require completed training")
        folder = self.folder / panel / arm["name"]
        folder.mkdir(parents=True)
        evaluation = json.loads(json.dumps(self.registration[f"{panel}_template"]))
        evaluation["source_sha256"] = dict(self.registration["source_sha256"])
        evaluation["checkpoints"] = {}
        evaluation["parent_registration_sha256"] = self.state["registration_sha256"]
        for update in arm["evaluation_updates"]:
            if control:
                spec = arm["saved_checkpoints"][str(update)]
                checkpoint, digest = resolve_registered(spec["checkpoint"]), spec["checkpoint_sha256"]
            else:
                checkpoint, digest = _checkpoint_from_manifest(resolve_registered(training["latest_experiment_dir"]), update)
            evaluation["checkpoints"][f"{arm['id']}_p{update:06d}"] = {
                "config": arm["evaluation_config"], "checkpoint": portable(checkpoint), "checkpoint_sha256": digest,
            }
            metadata = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
            evaluation["source_sha256"][portable(metadata)] = sha256(metadata)
        registration_path = folder / "registration.json"
        with registration_path.open("x") as handle:
            json.dump(evaluation, handle, indent=2)
            handle.write("\n")
        self.notify(f"**Normal-penalty {panel} evaluation started**\n{arm['id']}; {'saved control, no training' if control else 'fresh S43 learner'}; free draft, both modes.")
        for key in evaluation["checkpoints"]:
            for mode in ("sample", "argmax"):
                batch = folder / "eval" / key / mode
                batch.mkdir(parents=True)
                self.state["active"] = {"phase": f"{panel}_evaluation", "arm": arm["id"], "panel": panel,
                                        "checkpoint": key, "mode": mode, "expected_games": len(evaluation["tasks"])}
                live, logs, traces = [], [], []
                try:
                    workers = min(self.args.workers, len(evaluation["tasks"]))
                    for shard in range(workers):
                        trace = batch / f"trace_{shard:02d}.jsonl"
                        log = (batch / f"worker_{shard:02d}.log").open("x")
                        traces.append(trace)
                        logs.append(log)
                        env = {name: value for name, value in os.environ.items() if not name.startswith("AZK_")}
                        env.update(evaluation["process_env"])
                        command = [str(PYTHON), "train-ablation-1781126582/run_prefix_paired_eval.py", "--registration", str(registration_path),
                                   "--checkpoint-key", key, "--mode", mode, "--shards", str(workers), "--shard-index", str(shard), "--out", str(trace)]
                        live.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                    while any(process.poll() is None for process in live):
                        if any(process.poll() not in (None, 0) for process in live):
                            raise RuntimeError(f"Paired evaluation worker failed: {key}/{mode}")
                        self.state["active"]["written_games"] = sum(sum(1 for _ in path.open("rb")) for path in traces if path.exists())
                        self.progress()
                        time.sleep(10)
                    if any(process.returncode != 0 for process in live):
                        raise RuntimeError(f"Paired evaluation worker failed: {key}/{mode}")
                finally:
                    for process in live:
                        stop(process)
                    for log in logs:
                        log.close()
                result = summarize(traces, evaluation, key, mode, batch / "result.json")
                if sum(cell["player_games"] for cell in result["candidate_only_context_counts"].values()) != result["games"]:
                    raise RuntimeError("Candidate-only denominator mismatch")
                print(f"[leader-normal-penalty] evaluated {panel}/{key}/{mode}: {result['games']} games", flush=True)
        report(self.args.registration, panel=panel)

    def run(self) -> None:
        try:
            print("[leader-normal-penalty] ready: one fresh S43 training run; saved controls only; watcher required", flush=True)
            self.save()
            self.await_monitor()
            for arm in self.registration["arms"]:
                training = None if arm["initialization"] == "saved_random50_checkpoints" else self.train(arm)
                for panel in self.panels:
                    self.evaluate(arm, training, panel)
                self.state["arms"].append({"id": arm["id"], "state": "evaluated", "trained": training is not None})
                self.save()
            for panel in self.panels:
                if not report(self.args.registration, panel=panel)["complete"]:
                    raise RuntimeError(f"{panel} evidence remains provisional")
            self.state.update(state="completed_observations_require_review", finished_at=time.time())
            self.state.pop("active", None)
            self.notify("**FRESH S43 NORMAL-PENALTY DIAGNOSTIC FINISHED — READY FOR REVIEW**\nSaved controls were not retrained. Review consequential strategy, not Normal fraction alone. No automatic promotion or 1B.")
        except BaseException as exc:
            self.state.update(state="stopped", error=str(exc), finished_at=time.time())
            self.notify("**Fresh S43 penalty campaign stopped**\n" + str(exc) + "\nEvidence retained; no fallback or automatic extension.")
            raise
        finally:
            self.save()
            while self.state["pending_notifications"]:
                time.sleep(60)
                self.deliver()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--discord-webhook-file", type=Path, default=Path.home() / ".config/azuki-tcg/discord-webhook-url")
    parser.set_defaults(require_discord_monitor=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    registration = json.loads(args.registration.read_text())
    validate(registration)
    print("[leader-normal-penalty] verified one fresh learner and saved S43 controls", flush=True)
    if not args.validate_only:
        NormalPenaltyCampaign(args, registration).run()


if __name__ == "__main__":
    main()
