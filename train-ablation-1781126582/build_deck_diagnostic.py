#!/usr/bin/env python3
"""Freeze same-policy learned/entity-only/strategic decks from the shared proposal pool."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Diagnostic registration already exists; do not overwrite")
    index = json.loads(args.index.read_text())
    window = index["arms"][0]["windows"][-1]
    if sha256(ROOT / window["checkpoint"]) != window["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from retained evaluation index")
    payload = json.loads(args.pool.read_text())
    catalog = build_deck_build_catalog(load_training_deck_pool(args.pool))
    learned = {mode: {} for mode in ("sample", "argmax")}
    trace_hashes = {}
    for mode in learned:
        path = ROOT / window[f"{mode}_trace"]
        trace_hashes[str(path.relative_to(ROOT))] = sha256(path)
        with path.open() as handle:
            for line in handle:
                for deck in json.loads(line)["decks"]:
                    key = (deck["gate"], deck["leader"])
                    signature = tuple(sorted(Counter(deck["main"]).items()))
                    if signature not in learned[mode].setdefault(key, []):
                        learned[mode][key].append(signature)
    arms_by_mode = {mode: {arm: {} for arm in ("learned", "entity_only", "strategic")}
                    for mode in learned}
    provenance = []
    strategic_indices = payload["summary"]["strategic_exposure_deck_indices"]
    entity_indices = payload["summary"]["entity_only_control_deck_indices"]
    if len(strategic_indices) != 32 or len(entity_indices) != 32:
        raise ValueError("Expected two paired variants per sixteen contexts")
    for strategic_index, entity_index in zip(strategic_indices, entity_indices):
        strategic, entity = (payload["decks"][i] for i in (strategic_index, entity_index))
        gate, leader, variant = (strategic[k] for k in ("target_gate", "target_leader", "variant"))
        if (gate, leader, variant) != tuple(entity[k] for k in ("target_gate", "target_leader", "variant")):
            raise ValueError("Unpaired strategic/entity contexts")
        key = f"{gate}:{leader}:v{variant}"
        provenance.append({"context": key, "strategic_pool_index": strategic_index,
                           "entity_pool_index": entity_index, "provenance": strategic["provenance"]})
        for mode, arms in arms_by_mode.items():
            candidates = learned[mode][(gate, leader)]
            drawn = candidates[(variant - 1) % len(candidates)]
            if sum(n for _, n in drawn) != 50 or any(not 0 < n <= 4 for _, n in drawn):
                raise ValueError("Illegal retained learned deck")
            element = catalog.records_by_code[gate].element
            if any(catalog.records_by_code[c].element not in (element, "NORMAL") for c, _ in drawn):
                raise ValueError("Retained learned deck incompatible with assigned element")
            arms["learned"][key] = {"native_deck": [[gate, 1], [leader, 1], ["IKZ-001", 10], *drawn]}
            for arm, deck in (("entity_only", entity), ("strategic", strategic)):
                arms[arm][key] = {"native_deck": [[c["card_id"], c["quantity"]] for c in deck["cards"]]}
    opponents = {}
    for element in ("LIGHTNING", "WATER", "FIRE", "EARTH"):
        deck = next(d for d in payload["decks"]
                    if d.get("reference_role") == "holdout" and d["element"] == element)
        opponents[element] = {"native_deck": [[c["card_id"], c["quantity"]] for c in deck["cards"]],
                              "source": deck["deck_slug"]}
    source_paths = [Path(__file__).resolve(), args.config.resolve(), args.pool.resolve()]
    source_paths.extend(ROOT / "train-ablation-1781126582" / name for name in (
        "run_fixed_deck_counterfactual.py", "run_deck_diagnostic.py", "probe_gate_kl.py",
        "play_selfplay_games.py", "strategy_descriptor.py", "build_strategy_discovery.py"))
    source_paths.extend((ROOT / "python/src/policy/v2").glob("*.py"))
    output = {
        "schema_id": "azuki.deck_diagnostic_registration", "schema_version": 1,
        "status": "registered", "production_qualified": False,
        "config": str(args.config.resolve().relative_to(ROOT)),
        "checkpoint": window["checkpoint"], "checkpoint_sha256": window["checkpoint_sha256"],
        "checkpoint_training_probability_provenance": "unproven; observational diagnostic only",
        "index_sha256": sha256(args.index), "pool_sha256": sha256(args.pool),
        "trace_sha256": trace_hashes, "variants": provenance,
        "contract": {"contexts": 16, "variants_per_context": 2, "arms": 3,
                     "opponents": 4, "candidate_seats": [0, 1], "modes": ["sample", "argmax"],
                     "games_per_mode": 768, "recurrent_start": "zero for all fixed-deck arms",
                     "primary": "candidate-seat elemental opportunities, effects and conversions by gate and leader",
                     "strength": "supporting only; no winrate promotion",
                     "limits": ["regional transplantation is legal exposure, not proven gate/leader fit",
                                "sparse Water variants share a compatible source and are not independent submissions",
                                "entity replacement changes card identity and interactions, not only card type",
                                "retained learned decks are replayed with reset recurrent state, not end-to-end drafting",
                                "two variants are a diagnostic, not statistical qualification",
                                "historical policy weights cannot be repaired by corrected inference"]},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
    }
    args.out.mkdir(parents=True)
    for mode, arms in arms_by_mode.items():
        (args.out / f"arms_{mode}.json").write_text(json.dumps({"arms": arms}, indent=2) + "\n")
    (args.out / "opponents.json").write_text(json.dumps({"arms": {"holdout": opponents}}, indent=2) + "\n")
    output["deck_artifact_sha256"] = {p.name: sha256(p) for p in args.out.glob("*.json")}
    (args.out / "registration.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"registered {args.out}: 1536 games; observational only", flush=True)


if __name__ == "__main__":
    main()
