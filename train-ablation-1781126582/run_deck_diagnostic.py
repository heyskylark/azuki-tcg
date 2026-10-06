#!/usr/bin/env python3
"""Execute registered paired decks and report only the candidate seat's mechanics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

from strategy_descriptor import build_strategy_descriptor

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report-only", action="store_true",
                        help="Recover reports from registration-hashed completed traces; run no games")
    args = parser.parse_args()
    if not 1 <= args.workers <= 6 or (args.device == "cuda" and args.workers != 1):
        raise ValueError("Use 1..6 CPU workers or exactly one CUDA worker")
    folder = args.registration.resolve().parent
    registration = json.loads(args.registration.read_text())
    if registration.get("schema_id") != "azuki.deck_diagnostic_registration":
        raise ValueError("Invalid diagnostic registration")
    for mapping in (registration["source_sha256"], registration["trace_sha256"]):
        for name, expected in mapping.items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Diagnostic source changed: {name}")
    for name, expected in registration["deck_artifact_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Diagnostic deck changed: {name}")
    if sha256(ROOT / registration["checkpoint"]) != registration["checkpoint_sha256"]:
        raise ValueError("Diagnostic checkpoint changed")
    jobs = [(mode, shard) for mode in ("sample", "argmax") for shard in range(args.workers)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("AZK_")}
    env.update(PYTHONPATH=f"{ROOT / 'build/python/src'}:{ROOT / 'python/src'}:{ROOT / 'train-ablation-1781126582'}",
               OMP_NUM_THREADS="1")

    def run(job: tuple[str, int]) -> Path:
        mode, shard = job
        output = folder / f"trace_{mode}_{shard}.jsonl"
        if output.exists():
            raise ValueError(f"Refusing to overwrite diagnostic trace: {output}")
        command = [str(ROOT / ".venv/bin/python"), "train-ablation-1781126582/run_fixed_deck_counterfactual.py",
                   "--config", registration["config"], "--checkpoint", registration["checkpoint"],
                   "--deck-arms", str(folder / f"arms_{mode}.json"),
                   "--opponent-deck-arms", str(folder / "opponents.json"), "--opponent-arm", "holdout",
                   "--action-mode", mode, "--device", args.device,
                   "--shards", str(args.workers), "--shard-index", str(shard), "--out", str(output)]
        print(f"[deck-diagnostic] starting {mode} shard {shard}", flush=True)
        with output.with_suffix(".log").open("w") as console:
            subprocess.run(command, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT, check=True)
        print(f"[deck-diagnostic] completed {mode} shard {shard}", flush=True)
        return output

    print(f"[deck-diagnostic] registered 1536 paired games, {args.workers} {args.device} workers", flush=True)
    if args.report_only:
        outputs = [folder / f"trace_{mode}_{shard}.jsonl" for mode, shard in jobs]
        expected = registration.get("report_input_trace_sha256", {})
        if set(expected) != {path.name for path in outputs}:
            raise ValueError("Report recovery requires hashes for exactly the expected trace shards")
        for path in outputs:
            if sha256(path) != expected[path.name]:
                raise ValueError(f"Diagnostic trace changed: {path}")
        print("[deck-diagnostic] recovering reports from verified traces; no games launched", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            outputs = list(workers.map(run, jobs))
    grouped = defaultdict(list)
    seen = set()
    for path in outputs:
        for line in path.open():
            game = json.loads(line)
            task = game["counterfactual"]
            identity = (game["policy_action_mode"], task["global_task_index"])
            if identity in seen:
                raise ValueError(f"Duplicate diagnostic task {identity}")
            seen.add(identity)
            grouped[(game["policy_action_mode"], task["arm"], task["candidate_seat"])].append(game)
    expected_tasks = {(mode, index) for mode in ("sample", "argmax") for index in range(768)}
    if seen != expected_tasks:
        raise ValueError(f"Incomplete or unexpected diagnostic tasks: {len(seen)}/1536")
    profiles = defaultdict(Counter)
    scores = defaultdict(list)
    descriptors = []
    incomplete = 0
    for (mode, arm, seat), games in sorted(grouped.items()):
        descriptor = build_strategy_descriptor(games, label=f"deck_diagnostic_{mode}_{arm}_seat{seat}",
                                              checkpoint_sha256=registration["checkpoint_sha256"])
        path = folder / f"descriptor_{mode}_{arm}_seat{seat}.json"
        path.write_text(json.dumps(descriptor, indent=2) + "\n")
        descriptors.append({"path": str(path.relative_to(ROOT)), "descriptor_id": descriptor["descriptor_id"]})
        for context in descriptor["elemental_strategy"]["contexts"].values():
            if context["seat"] == seat:
                key = f"{mode}|{arm}|{context['gate']}|{context['leader']}"
                profiles[key].update(context["counts"])
        for game in games:
            candidate = game["decks"][seat]
            key = f"{mode}|{arm}|{candidate['gate']}|{candidate['leader']}"
            if game["outcome"].get("truncated") or not game["outcome"].get("terminated"):
                incomplete += 1
            else:
                scores[key].append(game["counterfactual"]["candidate_score"])
    result = {"schema_id": "azuki.deck_diagnostic_result", "schema_version": 1,
              "registration_sha256": sha256(args.registration), "games": len(seen),
              "incomplete_games": incomplete, "production_qualified": False,
              "status": "observations_complete" if not incomplete else "incomplete_games_require_review",
              "candidate_only_elemental_counts": {key: dict(value) for key, value in sorted(profiles.items())},
              "supporting_completed_game_scores": {key: {"games": len(values), "mean": sum(values)/len(values)}
                                                  for key, values in sorted(scores.items())},
              "descriptors": descriptors,
              "interpretation": "Opportunity/effect evidence under supplied decks; not expert play, causal action value, or production qualification."}
    if args.report_only:
        result["report_input_trace_sha256"] = registration["report_input_trace_sha256"]
    (folder / "decision_packet.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[deck-diagnostic] complete games={len(seen)} incomplete={incomplete}; no production promotion", flush=True)


if __name__ == "__main__":
    main()
