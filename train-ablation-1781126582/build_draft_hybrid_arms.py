#!/usr/bin/env python3
"""Build gate/leader/main factorial decks from one checkpoint's greedy drafts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_frozen_deck_arms import _argmax_main, _deck_payload, _read_dump
from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool


SIBLING_PAIRS = (
    ("STT01-002", "AZK01-120"),
    ("STT02-002", "AZK01-126"),
    ("AZK01-122", "STT04-002"),
    ("AZK01-124", "STT03-002"),
)
SIBLING_BY_GATE = {
    left: right
    for left, right in SIBLING_PAIRS
} | {
    right: left
    for left, right in SIBLING_PAIRS
}


def build_arms(deck_dump: dict) -> dict:
    catalog = build_deck_build_catalog(load_training_deck_pool())
    records_by_code = catalog.records_by_code
    arms = {
        "matched": {},
        "sibling_main": {},
        "sibling_leader": {},
        "sibling_both": {},
    }
    for gate, sibling in SIBLING_BY_GATE.items():
        target_payload = deck_dump["gates"][gate]
        sibling_payload = deck_dump["gates"][sibling]
        target_main = _argmax_main(target_payload)
        sibling_main = _argmax_main(sibling_payload)
        target_leader = str(target_payload["argmax_deck"]["leader"]["code"])
        sibling_leader = str(sibling_payload["argmax_deck"]["leader"]["code"])
        definitions = {
            "matched": (target_leader, target_main),
            "sibling_main": (target_leader, sibling_main),
            "sibling_leader": (sibling_leader, target_main),
            "sibling_both": (sibling_leader, sibling_main),
        }
        for arm, (leader, main) in definitions.items():
            arms[arm][gate] = {
                **_deck_payload(
                    gate=gate,
                    leader=leader,
                    main=main,
                    native_main=target_main,
                    records_by_code=records_by_code,
                ),
                "target_gate": gate,
                "component_source_gate": (
                    gate if arm == "matched" else sibling
                ),
                "leader_source_gate": (
                    sibling if arm in {"sibling_leader", "sibling_both"} else gate
                ),
                "main_source_gate": (
                    sibling if arm in {"sibling_main", "sibling_both"} else gate
                ),
            }
    return {
        "schema_version": 1,
        "source_checkpoint": deck_dump.get("checkpoint"),
        "arm_definitions": {
            "matched": "Target gate with the leader and main learned for that gate.",
            "sibling_main": "Target gate/leader with the sibling gate's learned main.",
            "sibling_leader": "Target gate/main with the sibling gate's learned leader.",
            "sibling_both": "Target gate with the sibling gate's learned leader and main.",
        },
        "sibling_pairs": [list(pair) for pair in SIBLING_PAIRS],
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decks", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    payload = build_arms(_read_dump(args.decks))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for arm, gates in payload["arms"].items():
        leaders = sorted({str(deck["leader"]) for deck in gates.values()})
        print(f"{arm}: gates={len(gates)} leaders={','.join(leaders)}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
