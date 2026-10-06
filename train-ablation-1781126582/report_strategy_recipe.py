#!/usr/bin/env python3
"""Report strategy-recipe evidence without ranking, qualification, or promotion."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from monitor_local_production import _atomic_json
from strategy_descriptor import (
    CARD_METADATA_PATH, SEQUENCE_REGISTRY, SEQUENCE_SEMANTICS_VERSION,
    _metadata, _spell_effect_observed,
)

MODES = ("sample", "argmax")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sequence(counts: dict, sequence: str) -> dict:
    fields = ("eligible", "completed", "converted")
    values = {field: counts.get(f"{sequence}/{field}") for field in fields}
    games = counts.get("player_games")
    if games is None or any(value is None for value in values.values()):
        return {**values, "player_games": games, "state": "unmeasured",
                "completion_per_eligible": None, "converted_per_eligible": None, "conversion_per_completed": None,
                "converted_per_player_game": None}
    eligible, completed, converted = (values[field] for field in fields)
    return {**values, "player_games": games,
            "state": "absent_opportunity" if not eligible else "no_execution" if not completed else "completed_without_observed_conversion" if not converted else "converted",
            "completion_per_eligible": _ratio(completed, eligible),
            "converted_per_eligible": _ratio(converted, eligible),
            "conversion_per_completed": _ratio(converted, completed),
            "converted_per_player_game": _ratio(converted, games)}


def _metrics(counts: dict, elements: set[str]) -> dict:
    return {"counts": dict(sorted(counts.items())),
            "sequences": {sequence: _sequence(counts, sequence)
                          for sequence, element in SEQUENCE_REGISTRY if element in elements}}


def _aggregate(payload: dict, metadata: dict) -> dict:
    groups = {name: defaultdict(Counter) for name in
              ("by_element", "by_gate_leader", "by_opponent", "by_seat", "by_context")}
    elements = {name: defaultdict(set) for name in groups}
    for key, counts in payload["candidate_only_context_counts"].items():
        # Existing summarize(): opponent=<id>|gate|leader|opp=g:l|seat=N|starter=N.
        opponent, gate, leader, opponent_context, seat, starter = key.split("|")
        if not opponent.startswith("opponent=") or not seat.startswith("seat="):
            raise ValueError(f"Unexpected candidate context: {key}")
        element = metadata[gate]["element"]
        # Keep denominators element-specific even in opponent/seat projections.
        coordinates = {"by_element": element, "by_gate_leader": f"{gate}|{leader}",
                       "by_opponent": f"{element}|{opponent}|{opponent_context}",
                       "by_seat": f"{element}|{seat}", "by_context": key}
        for group, coordinate in coordinates.items():
            groups[group][coordinate].update(counts)
            elements[group][coordinate].add(element)
    return {group: {coordinate: _metrics(counts, elements[group][coordinate])
                    for coordinate, counts in sorted(cells.items())}
            for group, cells in groups.items()}


def _trace_spells(payload: dict, arm: dict, mode: str, tasks: dict, metadata: dict) -> dict:
    """Hash and decode each trace in one pass, retaining counters, not games."""
    contexts = {}
    for task in tasks.values():
        if metadata[task["candidate_gate"]]["element"] == "WATER":
            context = f"{task['candidate_gate']}|{task['candidate_leader']}"
            contexts.setdefault(context, {"player_games": 0, "cards": defaultdict(Counter)})
    seen = set()
    provenance = {}
    registrations = set()
    candidates = {}
    for name, expected_hash in sorted(payload["trace_sha256"].items()):
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                game = json.loads(line)
                task = game["paired_eval"]
                identity = task["task_id"]
                if identity in seen or identity not in tasks:
                    raise ValueError(f"Duplicate or unexpected task {identity} in {path}")
                if any(task.get(field) != value for field, value in tasks[identity].items()):
                    raise ValueError(f"Task assignment drift: {identity}")
                if (task["checkpoint_key"] != payload["checkpoint_key"] or task["mode"] != mode
                        or game["policy_action_mode"] != mode or not task["complete"]
                        or task["validation_errors"] or not game["outcome"]["terminated"]
                        or game["outcome"]["truncated"]):
                    raise ValueError(f"Invalid paired trace {identity} in {path}")
                seen.add(identity)
                registrations.add(task["registration_sha256"])
                candidate = task["candidate"]
                candidates[candidate["checkpoint_sha256"]] = candidate
                seat = task["candidate_seat"]
                if metadata[task["candidate_gate"]]["element"] != "WATER":
                    continue
                context = contexts[f"{task['candidate_gate']}|{task['candidate_leader']}"]
                context["player_games"] += 1
                selected_cards, positive_cards = set(), set()
                for step in game["steps"]:
                    if step["p"] != seat or step.get("d", {}).get("t") != "PLAY_SPELL_FROM_HAND":
                        continue
                    code = step["d"]["card"]
                    record = metadata.get(code)
                    if record is None or record["card_type"] != "SPELL":
                        raise ValueError(f"Missing or non-spell metadata for selection {code}")
                    card = context["cards"][code]
                    card["selections"] += 1
                    selected_cards.add(code)
                    effect = _spell_effect_observed(step)
                    card["effect_observed"] += int(effect is not None)
                    card["effect_positive"] += int(bool(effect))
                    if effect:
                        positive_cards.add(code)
                for code in selected_cards:
                    context["cards"][code]["selected_player_games"] += 1
                for code in positive_cards:
                    context["cards"][code]["positive_effect_player_games"] += 1
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Trace SHA256 drift: {path}")
        provenance[name] = actual_hash
    if seen != set(tasks) or len(seen) != payload["games"]:
        raise ValueError(f"Trace task coverage mismatch: {len(seen)}/{len(tasks)}")
    if len(candidates) != 1 or len(registrations) != 1:
        raise ValueError("Mixed checkpoint or registration provenance within batch")
    finished = {}
    for key, context in sorted(contexts.items()):
        cards = {}
        for code, counts in sorted(context["cards"].items()):
            cards[code] = {"card_code": code, "name": metadata[code]["name"],
                           "element": metadata[code]["element"], "card_type": "SPELL",
                           **{field: counts[field] for field in ("selections", "selected_player_games", "effect_observed", "effect_positive", "positive_effect_player_games")},
                           "selected_per_player_game": _ratio(counts["selected_player_games"], context["player_games"]),
                           "effect_observation_coverage": _ratio(counts["effect_observed"], counts["selections"])}
        finished[key] = {"player_games": context["player_games"], "cards": cards,
                         "selected_card_identities": list(cards), "distinct_selected_spells": len(cards),
                         "effect_coverage": {"distinct_spells_with_positive_observed_effect": sum(card["effect_positive"] > 0 for card in cards.values()),
                                             "observed": sum(card["effect_observed"] for card in cards.values()),
                                             "positive": sum(card["effect_positive"] for card in cards.values()),
                                             "selected": sum(card["selections"] for card in cards.values())}}
    return {"contexts": finished, "trace_sha256": provenance,
            "paired_registration_sha256": sorted(registrations), "candidate_checkpoints": candidates,
            "semantics": "All spell identities, including Normal spells, selected via PLAY_SPELL_FROM_HAND by the candidate in Water contexts. Card elements are retained separately; a Normal spell can support Water-specific mechanics. Selection is not an effect or strategy success. Effect coverage is separate; no action-count ranking."}


def _trajectory(arm: dict, mode: str, sequence: str, element: str, batches: dict) -> dict:
    points = []
    for update in arm["evaluation_updates"]:
        batch = batches.get((arm["id"], update, mode))
        metric = None if batch is None else batch["aggregates"]["by_element"].get(element, {}).get("sequences", {}).get(sequence)
        points.append({"update": update, "metrics": metric,
                       "available": metric is not None and metric["state"] != "unmeasured"})
    observed = [point for point in points if point["available"]]
    acquired = [point["update"] for point in observed if (point["metrics"]["converted"] or 0) > 0]
    complete = all(point["available"] for point in points)
    late = points[-2:]
    persistence = (all((point["metrics"]["converted"] or 0) > 0 for point in late)
                   if len(late) == 2 and all(point["available"] for point in late) else None)
    return {"seed": arm["seed"], "arm_id": arm["id"], "points": points,
            "observed_acquisition": True if acquired else False if complete else None,
            "first_observed_conversion_update": min(acquired) if acquired else None,
            "late_persistence": persistence,
            "observed_emergence_after_first_checkpoint":
                (points[0]["metrics"]["converted"] == 0 and bool(acquired)) if complete else None}


def _comparisons(arms: list[dict], batches: dict) -> list[dict]:
    matched = {(arm["recipe"], arm["seed"]): arm for arm in arms}
    evidence = []
    for arm in arms:
        for baseline in (("RANDOM50",) if "RANDOM50" in {a["recipe"] for a in arms} else ("CONTROL", "RANDOM20")):
            if arm["recipe"] == baseline:
                continue
            other = matched.get((baseline, arm["seed"]))
            for mode in MODES:
                for update in arm["evaluation_updates"]:
                    left = batches.get((arm["id"], update, mode))
                    baseline_update = arm.get("baseline_updates", {}).get(str(update), update)
                    right = None if other is None else batches.get((other["id"], baseline_update, mode))
                    item = {"recipe": arm["recipe"], "baseline": baseline, "seed": arm["seed"],
                            "mode": mode, "update": update, "baseline_update": baseline_update,
                            "matched_training_progress": update == baseline_update,
                            "available": left is not None and right is not None}
                    if not item["available"]:
                        item["evidence"] = None
                    else:
                        cells = {}
                        for context in sorted(set(left["aggregates"]["by_context"]) | set(right["aggregates"]["by_context"])):
                            a = left["aggregates"]["by_context"].get(context)
                            b = right["aggregates"]["by_context"].get(context)
                            sequences = {}
                            for sequence in sorted(set(a["sequences"] if a else {}) | set(b["sequences"] if b else {})):
                                x = a["sequences"].get(sequence) if a else None
                                y = b["sequences"].get(sequence) if b else None
                                sequences[sequence] = {"recipe": x, "baseline": y,
                                    "converted_per_player_game_delta":
                                        x["converted_per_player_game"] - y["converted_per_player_game"]
                                        if x and y and x["converted_per_player_game"] is not None and y["converted_per_player_game"] is not None else None}
                            cells[context] = sequences
                        spell_evidence = {}
                        for context, value in left["water_spell_diversity"]["contexts"].items():
                            reference = right["water_spell_diversity"]["contexts"].get(context)
                            spell_evidence[context] = {"recipe_identities": value["selected_card_identities"],
                                "baseline_identities": reference["selected_card_identities"] if reference else None,
                                "recipe_only_observed_identities": sorted(set(value["cards"]) - set(reference["cards"])) if reference else None,
                                "recipe_effect_coverage": value["effect_coverage"],
                                "baseline_effect_coverage": reference["effect_coverage"] if reference else None}
                        item["evidence"] = {"candidate_context_sequences": cells, "water_spell_diversity": spell_evidence}
                    evidence.append(item)
    return evidence


def report(registration_path: Path, panel: str = "paired") -> dict:
    registration_path = Path(registration_path).resolve()
    raw = registration_path.read_bytes()
    registration = json.loads(raw)
    if (registration["schema_id"] != "azuki.ablation_registration"
            or registration["schema_version"] != 1 or registration["family"] not in ("strategy_recipe_v1", "strategy_retention_v1", "random50_continuation_v1", "leader_normal_penalty_v1")
            or registration["production_qualified"] is not False):
        raise ValueError("Expected diagnostic strategy recipe or retention registration")
    if panel not in ("paired", "heldout") or (panel == "heldout" and registration["family"] not in ("random50_continuation_v1", "leader_normal_penalty_v1")):
        raise ValueError("Panel is not registered for this campaign")
    root = registration_path.parent
    metadata = _metadata()
    arms = registration["arms"]
    template = registration[f"{panel}_template"]
    tasks = {task["task_id"]: task for task in template["tasks"]}
    if len(tasks) != len(template["tasks"]):
        raise ValueError("Duplicate registered task IDs")
    batches, missing = {}, []
    for arm in arms:
        for update in arm["evaluation_updates"]:
            for mode in MODES:
                path = root / panel / arm["name"] / "eval" / f"{arm['id']}_p{update:06d}" / mode / "result.json"
                identity = {"arm_id": arm["id"], "recipe": arm["recipe"], "seed": arm["seed"],
                            "update": update, "mode": mode, "result_path": str(path)}
                if not path.is_file():
                    missing.append({**identity, "reason": "missing_result"})
                    continue
                result_bytes = path.read_bytes()
                payload = json.loads(result_bytes)
                if (payload["schema_id"] != "azuki.prefix_paired_eval_result" or payload["schema_version"] != 1
                        or payload["checkpoint_key"] != f"{arm['id']}_p{update:06d}" or payload["mode"] != mode
                        or payload["games"] != len(tasks) or payload["incomplete"] != 0):
                    raise ValueError(f"Invalid completed result: {path}")
                if not payload["trace_sha256"]:
                    raise ValueError(f"No retained trace provenance: {path}")
                absent = [name for name in payload["trace_sha256"] if not (ROOT / name).is_file()]
                if absent:
                    missing.append({**identity, "reason": "missing_traces", "paths": absent})
                    continue
                aggregates = _aggregate(payload, metadata)
                if sum(cell["counts"]["player_games"] for cell in aggregates["by_element"].values()) != len(tasks):
                    raise ValueError(f"Candidate-only count coverage mismatch: {path}")
                spells = _trace_spells(payload, arm, mode, tasks, metadata)
                if registration["family"] in ("random50_continuation_v1", "leader_normal_penalty_v1"):
                    manifest_name = "parent_registration.json" if registration["family"] == "random50_continuation_v1" and update == arm["parent_update"] else "registration.json"
                    manifest_path = root / panel / arm["name"] / manifest_name
                    manifest = json.loads(manifest_path.read_text())
                    candidate = manifest["checkpoints"][payload["checkpoint_key"]]
                    if (spells["paired_registration_sha256"] != [_digest(manifest_path)]
                            or spells["candidate_checkpoints"] != {candidate["checkpoint_sha256"]: candidate}
                            or manifest["parent_registration_sha256"] != hashlib.sha256(raw).hexdigest()
                            or manifest["tasks"] != template["tasks"]):
                        raise ValueError(f"Continuation panel provenance mismatch: {path}")
                batches[arm["id"], update, mode] = {**identity, "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                    "games": payload["games"], "aggregates": aggregates, "water_spell_diversity": spells}
    recipes = []
    for recipe in dict.fromkeys(arm["recipe"] for arm in arms):
        recipe_arms = [arm for arm in arms if arm["recipe"] == recipe]
        modes = {}
        for mode in MODES:
            sequences = {}
            for sequence, element in SEQUENCE_REGISTRY:
                trajectories = [_trajectory(arm, mode, sequence, element, batches) for arm in recipe_arms]
                def replicated(field: str) -> bool | None:
                    values = [trajectory[field] for trajectory in trajectories]
                    if len(values) < 2 or any(value is None for value in values):
                        return None
                    return all(values)
                sequences[sequence] = {"element": element, "per_seed": trajectories,
                    "descriptive_replicated_acquisition": replicated("observed_acquisition"),
                    "descriptive_replicated_late_persistence": replicated("late_persistence")}
            modes[mode] = {"sequences": sequences}
        recipes.append({"recipe": recipe, "modes": modes})
    water_contexts = sorted({f"{task['candidate_gate']}|{task['candidate_leader']}" for task in tasks.values()
                             if metadata[task["candidate_gate"]]["element"] == "WATER"})
    payload = {"schema_id": "azuki.strategy_recipe_report", "schema_version": 1,
        "family": registration["family"], "scope": "diagnostic_only", "production_qualified": False,
        "panel": panel, "panel_semantics": "Held-out task seeds; same frozen contexts and opponents, not new-opponent generalization." if panel == "heldout" else "Original frozen paired task panel.",
        "complete": not missing, "provisional": bool(missing), "state": "provisional" if missing else "complete_diagnostic",
        "registration": str(registration_path), "registration_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": registration["source_sha256"], "metadata_sha256": _digest(CARD_METADATA_PATH),
        "sequence_semantics_version": SEQUENCE_SEMANTICS_VERSION,
        "expected_batches": sum(len(arm["evaluation_updates"]) * len(MODES) for arm in arms),
        "available_batches": len(batches), "expected_games": sum(len(arm["evaluation_updates"]) * len(MODES) * len(tasks) for arm in arms),
        "available_games": sum(batch["games"] for batch in batches.values()), "missing_batches": missing,
        "water_contexts": water_contexts, "batches": list(batches.values()), "recipes": recipes,
        "matched_comparisons": _comparisons(arms, batches),
        "frozen_opponents": template["opponents"], "evaluation_process_env": template["process_env"],
        "interpretation": {
            "objective": "Unique strategy emergence, retention and diversity; no scalar winner, raw action-count ranking, winrate gate, or production qualification.",
            "flags": "Acquisition means at least one observed sequence conversion; late persistence means nonzero conversion at both final two registered checkpoints. Replicated means observed separately in every registered seed (at least two), separately by action mode. These are descriptive, not statistical significance, causal treatment effects, or qualification criteria.",
            "emergence": "First observation is not proof of first learning; emergence-after-first requires zero conversions at the first checkpoint and later nonzero conversion. Missing evidence remains null, never zero.",
            "opportunity_guard": "Eligible, completed, converted and converted/player_games remain together. No opportunity is distinct from failing to execute; conditional conversion alone can hide opportunity collapse.",
            "comparison": "Baseline recipes are paired by training seed, mode and task schedule at explicitly recorded updates. Smoke probes may deliberately differ in training progress and are not quality comparisons. Context projections include realized starter, which can diverge; these are descriptive, not independent paired significance tests.",
            "water": "All registered Water gate+leader contexts retained, not only Echoed Waves. Spell identity diversity and spell effect coverage are separate.",
            "horizon": "Longer-horizon acquisition and retention evidence is needed before considering 1B. This report never launches, promotes, or authorizes 1B.",
        },
        "limitations": ["Frozen-opponent checkpoint/config provenance is retained; this panel does not establish generalization beyond those opponents.",
            "Deferred effects and physical card-copy/effect identities are incompletely observed. Absent observed conversion is not proof of no eventual effect.",
            "Observed spell effects are state-change proxies, not causal spell value. Hydromancy resource-to-spell attribution is a same-turn lower bound.",
            "Named sequences are diagnostic probes, not reward targets or an exhaustive strategy vocabulary; unregistered coherent lines require trace review.",
            "A provisional campaign cannot establish cross-seed retention or treatment superiority; no score is used for recipe ordering."]}
    _atomic_json(root / ("strategy_report.json" if panel == "paired" else "strategy_report_heldout.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--panel", choices=("paired", "heldout"), default="paired")
    args = parser.parse_args()
    result = report(args.registration, panel=args.panel)
    print(json.dumps({key: result[key] for key in ("state", "available_batches", "expected_batches", "available_games", "expected_games")}))


if __name__ == "__main__":
    main()
