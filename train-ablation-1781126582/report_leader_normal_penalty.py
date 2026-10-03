#!/usr/bin/env python3
"""Report leader-conditioned Normal-composition evidence without selection gates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitor_local_production import _atomic_json
from report_strategy_recipe import _metrics, report as report_strategy_recipe
from strategy_descriptor import PLAY_ACTION_NAMES, SEQUENCE_SEMANTICS_VERSION, _selected_resolved, _spell_effect_observed

MODES = ("sample", "argmax")
MAIN_DECK_SIZE = 50
MAIN_CARD_TYPES = frozenset(("ENTITY", "SPELL", "WEAPON"))
ABILITY_ACTIONS = frozenset(("ACTIVATE_GARDEN_OR_LEADER_ABILITY", "ACTIVATE_ALLEY_ABILITY"))


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _runtime_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_catalog(registration: dict) -> tuple[dict[str, dict], Path, str]:
    path = ROOT / "python/config/policy_card_metadata_v1.json"
    expected = registration.get("source_sha256", {}).get("python/config/policy_card_metadata_v1.json")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Runtime card catalog is not frozen in source_sha256")
    actual = _digest(path)
    if actual != expected:
        raise ValueError(f"Runtime card catalog SHA256 drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Runtime card catalog has no records")
    by_code = {str(record["card_code"]): record for record in records}
    if len(by_code) != len(records):
        raise ValueError("Runtime card catalog has duplicate card codes")
    return by_code, path, actual


def _load_penalty_config(registration: dict, catalog: dict[str, dict]) -> tuple[dict, Path, str]:
    spec = registration.get("penalty_config")
    if (not isinstance(spec, dict) or not isinstance(spec.get("path"), str)
            or not isinstance(spec.get("sha256"), str) or len(spec["sha256"]) != 64):
        raise ValueError("Registration lacks frozen penalty_config path and SHA256")
    path = _runtime_path(spec["path"])
    actual = _digest(path)
    if actual != spec["sha256"]:
        raise ValueError(f"Normal-penalty configuration SHA256 drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"schema_id", "schema_version", "main_deck_size", "leaders", "provenance"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Invalid leader Normal-penalty configuration fields")
    if any(key not in spec or spec[key] != value for key, value in payload.items()):
        raise ValueError("Embedded penalty_config registration fields differ from frozen file")
    if (payload.get("schema_id") != "azuki.leader_normal_penalty" or payload.get("schema_version") != 1
            or payload.get("main_deck_size") != MAIN_DECK_SIZE):
        raise ValueError("Invalid leader Normal-penalty configuration schema")
    leader_codes = {code for code, record in catalog.items() if record.get("card_type") == "LEADER"}
    leaders = payload.get("leaders")
    if not isinstance(leaders, dict) or set(leaders) != leader_codes:
        raise ValueError("Normal-penalty configuration does not cover the full runtime leader catalog")
    for code, setting in leaders.items():
        if not isinstance(setting, dict) or set(setting) != {"max_normal_fraction", "weight"}:
            raise ValueError(f"Invalid Normal-penalty setting for {code}")
        threshold, weight = setting["max_normal_fraction"], setting["weight"]
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 <= threshold < 1
                or isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError(f"Invalid Normal-penalty threshold/weight for {code}")
    for arm in registration["arms"]:
        env_path = arm.get("process_env", {}).get("AZK_DRAFT_NORMAL_PENALTY_CONFIG")
        if env_path is None or _runtime_path(env_path).resolve() != path.resolve():
            raise ValueError(f"Arm {arm.get('id')} does not use the frozen Normal-penalty configuration")
    return payload, path, actual


def _histogram(values: list[float], digits: int = 6) -> dict[str, int]:
    return dict(sorted(Counter(f"{value:.{digits}f}" for value in values).items()))


def _distribution(rows: list[dict], setting: dict) -> dict:
    fractions = [float(row["normal_fraction"]) for row in rows]
    excesses = [float(row["threshold_excess_fraction"]) for row in rows]
    normalized = [float(row["normalized_excess"]) for row in rows]
    costs = [float(row["weighted_cost"]) for row in rows]

    def summary(values: list[float]) -> dict:
        return {
            "mean": sum(values) / len(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "histogram": _histogram(values),
        }

    return {
        "sample_count": len(rows),
        "max_normal_fraction": float(setting["max_normal_fraction"]),
        "weight": float(setting["weight"]),
        "normal_copy_count_histogram": dict(sorted(Counter(str(row["normal_count"]) for row in rows).items(), key=lambda item: int(item[0]))),
        "normal_fraction": summary(fractions),
        "threshold_excess_fraction": summary(excesses),
        "normalized_excess": summary(normalized),
        "weighted_quadratic_cost": summary(costs),
        "decks_above_threshold": sum(value > 0 for value in excesses),
    }


def _finish_support(raw: dict) -> dict:
    leaders = {}
    for leader, value in sorted(raw.items()):
        cards = {}
        for code, counts in sorted(value["cards"].items()):
            cards[code] = {
                "card_code": code,
                "name": counts["name"],
                "element": counts["element"],
                "card_type": counts["card_type"],
                "selections": counts["selections"],
                "resolved_observed": counts["resolved_observed"],
                "resolved": counts["resolved"],
                "spell_effect_observed": counts["spell_effect_observed"],
                "spell_effect_positive": counts["spell_effect_positive"],
                "selected_player_games": len(value["card_games"][code]),
            }
        abilities = {}
        for key, counts in sorted(value["abilities"].items()):
            abilities[key] = {
                "activations": counts["activations"],
                "resolution_observed": counts["resolution_observed"],
                "resolved": counts["resolved"],
            }
        leaders[leader] = {
            "player_games": value["player_games"],
            "cards": cards,
            "ability_activations": abilities,
        }
    return leaders


def _trace_batch(payload: dict, arm: dict, mode: str, tasks: dict[str, dict], catalog: dict[str, dict], settings: dict) -> tuple[dict, dict[str, dict]]:
    seen: set[str] = set()
    rows: dict[str, dict] = {}
    provenance = {}
    support = defaultdict(lambda: {
        "player_games": 0,
        "cards": {},
        "card_games": defaultdict(set),
        "abilities": defaultdict(Counter),
    })
    candidate_hashes = set()
    registration_hashes = set()
    for name, expected_hash in sorted(payload["trace_sha256"].items()):
        path = _runtime_path(name)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                game = json.loads(line)
                task = game.get("paired_eval", {})
                task_id = str(task.get("task_id", ""))
                if task_id in seen or task_id not in tasks:
                    raise ValueError(f"Duplicate or unexpected task {task_id} in {path}")
                if any(task.get(field) != value for field, value in tasks[task_id].items()):
                    raise ValueError(f"Registered task assignment drift: {task_id}")
                if (task.get("checkpoint_key") != payload["checkpoint_key"] or task.get("mode") != mode
                        or game.get("policy_action_mode") != mode or not task.get("complete")
                        or task.get("validation_errors") or not game.get("outcome", {}).get("terminated")
                        or game.get("outcome", {}).get("truncated")):
                    raise ValueError(f"Incomplete or mismatched candidate trace: {task_id}")
                seat = int(task["candidate_seat"])
                decks = game.get("decks")
                if not isinstance(decks, list) or len(decks) != 2:
                    raise ValueError(f"Trace has no complete two-seat decks: {task_id}")
                deck = decks[seat]
                if deck.get("gate") != task["candidate_gate"] or deck.get("leader") != task["candidate_leader"]:
                    raise ValueError(f"Candidate deck assignment drift: {task_id}")
                main = deck.get("main")
                if not isinstance(main, list) or len(main) != MAIN_DECK_SIZE:
                    raise ValueError(f"Candidate main deck is not exactly 50 cards: {task_id}")
                unknown = sorted({str(code) for code in main if str(code) not in catalog})
                if unknown:
                    raise ValueError(f"Candidate main deck lacks full runtime catalog coverage: {task_id}: {unknown}")
                forbidden = sorted({str(code) for code in main if catalog[str(code)].get("card_type") not in MAIN_CARD_TYPES})
                if forbidden:
                    raise ValueError(f"Candidate main deck includes leader/gate/IKZ/non-main cards: {task_id}: {forbidden}")
                draft = game.get("draft")
                if not isinstance(draft, list):
                    raise ValueError(f"Missing actual draft evidence: {task_id}")
                picks = [str(pick.get("selected")) for pick in draft if int(pick.get("p", -1)) == seat]
                if len(picks) != MAIN_DECK_SIZE or Counter(picks) != Counter(str(code) for code in main):
                    raise ValueError(f"Actual 50-pick draft (including final pick) does not match candidate deck: {task_id}")
                leader = str(deck["leader"])
                setting = settings.get(leader)
                if setting is None:
                    raise ValueError(f"Candidate leader absent from penalty configuration: {leader}")
                normal_count = sum(catalog[str(code)].get("element") == "NORMAL" for code in main)
                fraction = normal_count / MAIN_DECK_SIZE
                threshold = float(setting["max_normal_fraction"])
                excess = max(0.0, fraction - threshold)
                normalized_excess = excess / (1.0 - threshold)
                rows[task_id] = {
                    "task_id": task_id,
                    "leader": leader,
                    "gate": str(deck["gate"]),
                    "normal_count": normal_count,
                    "normal_fraction": fraction,
                    "threshold_excess_fraction": excess,
                    "normalized_excess": normalized_excess,
                    "weighted_cost": float(setting["weight"]) * normalized_excess * normalized_excess,
                }
                gate_element = str(catalog[str(deck["gate"])]["element"])
                leader_support = support[leader]
                leader_support["player_games"] += 1
                for step in game.get("steps", []):
                    if int(step.get("p", -1)) != seat:
                        continue
                    detail = step.get("d", {})
                    action = str(detail.get("t", ""))
                    code = detail.get("card")
                    if action in PLAY_ACTION_NAMES and isinstance(code, str):
                        record = catalog.get(code)
                        if record is None:
                            raise ValueError(f"Played card lacks runtime catalog coverage: {task_id}: {code}")
                        if record.get("element") not in ("NORMAL", gate_element):
                            continue
                        counts = leader_support["cards"].setdefault(code, {
                            "name": str(record.get("name", code)),
                            "element": str(record.get("element")),
                            "card_type": str(record.get("card_type")),
                            "selections": 0,
                            "resolved_observed": 0,
                            "resolved": 0,
                            "spell_effect_observed": 0,
                            "spell_effect_positive": 0,
                        })
                        counts["selections"] += 1
                        leader_support["card_games"][code].add(task_id)
                        resolved = _selected_resolved(step, winner=-1)
                        counts["resolved_observed"] += int(resolved is not None)
                        counts["resolved"] += int(bool(resolved))
                        if record.get("card_type") == "SPELL":
                            effect = _spell_effect_observed(step)
                            counts["spell_effect_observed"] += int(effect is not None)
                            counts["spell_effect_positive"] += int(bool(effect))
                    if action in ABILITY_ACTIONS:
                        source = str(detail.get("src") or detail.get("card") or "unknown")
                        key = f"{action}|source={source}"
                        resolved = _selected_resolved(step, winner=-1)
                        leader_support["abilities"][key]["activations"] += 1
                        leader_support["abilities"][key]["resolution_observed"] += int(resolved is not None)
                        leader_support["abilities"][key]["resolved"] += int(bool(resolved))
                seen.add(task_id)
                candidate = task.get("candidate", {})
                if isinstance(candidate, dict) and isinstance(candidate.get("checkpoint_sha256"), str):
                    candidate_hashes.add(candidate["checkpoint_sha256"])
                registration_hashes.add(str(task.get("registration_sha256", "")))
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Trace SHA256 drift: {path}")
        provenance[name] = actual_hash
    if seen != set(tasks) or len(seen) != payload["games"]:
        raise ValueError(f"Candidate-only trace coverage mismatch: {len(seen)}/{len(tasks)}")
    if len(candidate_hashes) != 1 or len(registration_hashes) != 1 or "" in registration_hashes:
        raise ValueError("Mixed or missing checkpoint/registration trace provenance")
    by_leader_rows = defaultdict(list)
    for row in rows.values():
        by_leader_rows[row["leader"]].append(row)
    composition = {
        leader: _distribution(leader_rows, settings[leader])
        for leader, leader_rows in sorted(by_leader_rows.items())
    }
    return {
        "composition_by_leader": composition,
        "played_elemental_and_normal_support_by_leader": _finish_support(support),
        "trace_sha256": provenance,
        "candidate_checkpoint_sha256": next(iter(candidate_hashes)),
        "paired_eval_registration_sha256": next(iter(registration_hashes)),
        "candidate_games": len(rows),
    }, rows


def _leader_sequences(strategy_batch: dict, catalog: dict[str, dict]) -> dict:
    counts = defaultdict(Counter)
    elements = defaultdict(set)
    for gate_leader, metrics in strategy_batch["aggregates"]["by_gate_leader"].items():
        gate, leader = gate_leader.split("|", 1)
        counts[leader].update(metrics["counts"])
        elements[leader].add(str(catalog[gate]["element"]))
    return {leader: _metrics(value, elements[leader])["sequences"] for leader, value in sorted(counts.items())}


def _paired_delta(rows: list[dict], field: str) -> dict:
    values = [float(row[field]) for row in rows]
    return {
        "sample_count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "histogram": _histogram(values),
    }


def _paired_comparisons(registration: dict, batches: dict, strategy_batches: dict) -> list[dict]:
    by_treatment_seed = {(arm["treatment"], int(arm["seed"])): arm for arm in registration["arms"]}
    comparisons = []
    for seed in sorted({int(arm["seed"]) for arm in registration["arms"]}):
        control = by_treatment_seed.get(("CONTROL", seed))
        penalty = by_treatment_seed.get(("PENALTY", seed))
        if control is None or penalty is None:
            raise ValueError(f"Seed {seed} lacks a CONTROL/PENALTY pair")
        updates = penalty["evaluation_updates"]
        baseline_updates = penalty["baseline_updates"]
        if (set(baseline_updates) != {str(update) for update in updates}
                or any(value not in control["evaluation_updates"] for value in baseline_updates.values())):
            raise ValueError(f"Seed {seed} has an unregistered baseline pairing")
        for update in updates:
            for mode in MODES:
                left = batches.get((penalty["id"], update, mode))
                control_update = baseline_updates[str(update)]
                right = batches.get((control["id"], control_update, mode))
                item = {"seed": seed, "update": update, "control_update": control_update,
                        "matched_training_progress": update == control_update,
                        "mode": mode, "penalty_arm_id": penalty["id"], "control_arm_id": control["id"],
                        "available": left is not None and right is not None}
                if not item["available"]:
                    item["evidence"] = None
                    comparisons.append(item)
                    continue
                penalty_rows, control_rows = left["rows"], right["rows"]
                if set(penalty_rows) != set(control_rows):
                    raise ValueError(f"Pairing task coverage differs for seed={seed}, update={update}, mode={mode}")
                by_leader = defaultdict(list)
                for task_id in sorted(penalty_rows):
                    a, b = penalty_rows[task_id], control_rows[task_id]
                    if (a["leader"], a["gate"]) != (b["leader"], b["gate"]):
                        raise ValueError(f"Paired candidate context drift: {task_id}")
                    by_leader[a["leader"]].append({
                        "normal_fraction_delta": a["normal_fraction"] - b["normal_fraction"],
                        "threshold_excess_fraction_delta": a["threshold_excess_fraction"] - b["threshold_excess_fraction"],
                        "weighted_cost_delta": a["weighted_cost"] - b["weighted_cost"],
                    })
                composition = {}
                for leader, deltas in sorted(by_leader.items()):
                    composition[leader] = {
                        field: _paired_delta(deltas, field)
                        for field in ("normal_fraction_delta", "threshold_excess_fraction_delta", "weighted_cost_delta")
                    }
                penalty_strategy = strategy_batches[(penalty["id"], update, mode)]
                control_strategy = strategy_batches[(control["id"], control_update, mode)]
                p_sequences = penalty_strategy["sequences_by_leader"]
                c_sequences = control_strategy["sequences_by_leader"]
                sequence_comparison = {}
                for leader in sorted(set(p_sequences) | set(c_sequences)):
                    sequence_comparison[leader] = {}
                    for sequence in sorted(set(p_sequences.get(leader, {})) | set(c_sequences.get(leader, {}))):
                        a = p_sequences.get(leader, {}).get(sequence)
                        b = c_sequences.get(leader, {}).get(sequence)
                        sequence_comparison[leader][sequence] = {
                            "penalty": a,
                            "control": b,
                            "converted_per_player_game_delta": (
                                a["converted_per_player_game"] - b["converted_per_player_game"]
                                if a and b and a["converted_per_player_game"] is not None and b["converted_per_player_game"] is not None else None
                            ),
                        }
                item["evidence"] = {
                    "paired_task_count": len(penalty_rows),
                    "composition_delta_penalty_minus_control_by_leader": composition,
                    "opportunity_denominated_sequences_by_leader": sequence_comparison,
                    "played_support": {
                        "penalty": left["public"]["played_elemental_and_normal_support_by_leader"],
                        "control": right["public"]["played_elemental_and_normal_support_by_leader"],
                    },
                }
                comparisons.append(item)
    return comparisons


def report(registration_path: Path, panel: str = "paired") -> dict:
    registration_path = Path(registration_path).resolve()
    raw = registration_path.read_bytes()
    registration = json.loads(raw)
    if (registration.get("schema_id") != "azuki.ablation_registration" or registration.get("schema_version") != 1
            or registration.get("family") != "leader_normal_penalty_v1" or registration.get("production_qualified") is not False):
        raise ValueError("Expected diagnostic leader_normal_penalty_v1 registration")
    if panel not in ("paired", "heldout"):
        raise ValueError("Panel must be paired or heldout")
    catalog, catalog_path, catalog_hash = _load_catalog(registration)
    penalty_config, config_path, config_hash = _load_penalty_config(registration, catalog)
    strategy_report = report_strategy_recipe(registration_path, panel=panel)
    template = registration[f"{panel}_template"]
    tasks = {str(task["task_id"]): task for task in template["tasks"]}
    if len(tasks) != len(template["tasks"]):
        raise ValueError("Duplicate registered task IDs")
    strategy_batches = {}
    for batch in strategy_report["batches"]:
        key = (batch["arm_id"], int(batch["update"]), batch["mode"])
        strategy_batches[key] = {
            "result_sha256": batch["result_sha256"],
            "sequences_by_leader": _leader_sequences(batch, catalog),
        }
    root = registration_path.parent
    batches = {}
    public_batches = []
    for arm in registration["arms"]:
        for update in arm["evaluation_updates"]:
            for mode in MODES:
                key = (arm["id"], int(update), mode)
                strategy_batch = strategy_batches.get(key)
                if strategy_batch is None:
                    continue
                result_path = root / panel / arm["name"] / "eval" / f"{arm['id']}_p{update:06d}" / mode / "result.json"
                result_bytes = result_path.read_bytes()
                payload = json.loads(result_bytes)
                if hashlib.sha256(result_bytes).hexdigest() != strategy_batch["result_sha256"]:
                    raise ValueError(f"Strategy/composition result provenance differs: {result_path}")
                public, rows = _trace_batch(payload, arm, mode, tasks, catalog, penalty_config["leaders"])
                public.update({
                    "arm_id": arm["id"],
                    "treatment": arm["treatment"],
                    "seed": arm["seed"],
                    "update": update,
                    "mode": mode,
                    "result_path": str(result_path),
                    "result_sha256": strategy_batch["result_sha256"],
                    "opportunity_denominated_sequences_by_leader": strategy_batch["sequences_by_leader"],
                })
                batches[key] = {"public": public, "rows": rows}
                public_batches.append(public)
    comparisons = _paired_comparisons(registration, batches, strategy_batches)
    payload = {
        "schema_id": "azuki.leader_normal_penalty_report",
        "schema_version": 1,
        "family": registration["family"],
        "scope": "diagnostic_only",
        "production_qualified": False,
        "panel": panel,
        "panel_semantics": strategy_report["panel_semantics"],
        "complete": strategy_report["complete"],
        "provisional": strategy_report["provisional"],
        "state": strategy_report["state"],
        "registration": str(registration_path),
        "registration_sha256": hashlib.sha256(raw).hexdigest(),
        "strategy_report": str(root / ("strategy_report.json" if panel == "paired" else "strategy_report_heldout.json")),
        "strategy_report_registration_sha256": strategy_report["registration_sha256"],
        "sequence_semantics_version": SEQUENCE_SEMANTICS_VERSION,
        "runtime_catalog": {"path": str(catalog_path), "sha256": catalog_hash, "migration": registration.get("catalog_runtime")},
        "comparison_contract": registration["comparison_contract"],
        "training_arms": registration["training_arms"],
        "penalty_config": {"path": str(config_path), "sha256": config_hash, **penalty_config},
        "expected_batches": strategy_report["expected_batches"],
        "available_batches": len(public_batches),
        "expected_games": strategy_report["expected_games"],
        "available_games": sum(batch["candidate_games"] for batch in public_batches),
        "missing_batches": strategy_report["missing_batches"],
        "batches": public_batches,
        "paired_control_penalty_comparisons": comparisons,
        "interpretation": {
            "composition": "Normal fractions count copies in the candidate's actual final 50-card main deck, including the final pick. Gate, leader, IKZ/non-main cards and the opponent deck are excluded. Threshold excess is descriptive treatment-mechanism evidence, not strategy success.",
            "pairing": "A fresh S43 penalty learner is compared with saved RANDOM50_S43 checkpoints on registered tasks, modes and panels. Explicit control_update and matched_training_progress fields distinguish matched full probes from unmatched infrastructure smoke probes.",
            "strategy": "Eligible, completed and converted sequence evidence retains legal-opportunity denominators. Converted consequential elemental sequences are distinct from raw ability activations, low Normal share, card-play volume, win rate, or a balanced-element rubric.",
            "support": "Played-support tables contain candidate actions only and retain Normal spells/entities/weapons plus cards matching the candidate gate element. A selection, activation, resolution or observed spell effect is not by itself a meaningful strategy or proof of success.",
            "modes": "sample and argmax are reported separately and are never pooled.",
            "decision": "No arbitrary success gate, automatic treatment selection, promotion, human-match access, future evaluation access, or 1B authorization is produced by this report.",
        },
        "limitations": [
            "Only one seed, selected from the user's short evaluation, is trained. Historical controls trained before catalog correction; this is not a clean causal estimate of the penalty alone.",
            "Trace observations do not identify counterfactual card value; played support and ability activations are descriptive counts only.",
            "Named sequences are diagnostic evidence, not an exhaustive strategy vocabulary or reward targets.",
            "Provisional coverage means missing batches remain missing, never zero; no incomplete or partial-deck trace contributes composition metrics.",
            "Spell-effect and deferred-effect coverage inherit the registered strategy descriptor's observational limitations.",
        ],
    }
    output = root / ("normal_penalty_report.json" if panel == "paired" else "normal_penalty_report_heldout.json")
    _atomic_json(output, payload)
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
