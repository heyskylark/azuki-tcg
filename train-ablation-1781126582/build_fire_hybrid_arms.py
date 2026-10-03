#!/usr/bin/env python3
"""Build Fire-only leader/main factorial arms for the frozen p2930 policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_frozen_deck_arms import (
    DEFAULT_P2930,
    DEFAULT_S14,
    _argmax_main,
    _deck_payload,
    _read_dump,
)
from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool


FIRE_GATES = ("AZK01-122", "STT04-002")


def build_arms(p2930: dict, s14: dict) -> dict:
    catalog = build_deck_build_catalog(load_training_deck_pool())
    records_by_code = catalog.records_by_code
    arms = {
        "native_p2930": {},
        "s14_main_p2930_leader": {},
        "p2930_main_s14_leader": {},
        "s14_greedy": {},
    }
    for gate in FIRE_GATES:
        p_gate = p2930["gates"][gate]
        s_gate = s14["gates"][gate]
        p_main = _argmax_main(p_gate)
        s_main = _argmax_main(s_gate)
        p_leader = str(p_gate["argmax_deck"]["leader"]["code"])
        s_leader = str(s_gate["argmax_deck"]["leader"]["code"])
        definitions = {
            "native_p2930": (p_leader, p_main),
            "s14_main_p2930_leader": (p_leader, s_main),
            "p2930_main_s14_leader": (s_leader, p_main),
            "s14_greedy": (s_leader, s_main),
        }
        for arm, (leader, main) in definitions.items():
            arms[arm][gate] = _deck_payload(
                gate=gate,
                leader=leader,
                main=main,
                native_main=p_main,
                records_by_code=records_by_code,
            )
    return {
        "schema_version": 1,
        "sources": {
            "p2930_checkpoint": p2930.get("checkpoint"),
            "s14_checkpoint": s14.get("checkpoint"),
        },
        "arm_definitions": {
            "native_p2930": "Exact p2930 Fire leader and main.",
            "s14_main_p2930_leader": "Exact S14 Fire main with p2930's Fire leader.",
            "p2930_main_s14_leader": "Exact p2930 Fire main with S14's Fire leader.",
            "s14_greedy": "Exact S14 Fire leader and main.",
        },
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2930", type=Path, default=DEFAULT_P2930)
    parser.add_argument("--s14", type=Path, default=DEFAULT_S14)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    payload = build_arms(_read_dump(args.p2930), _read_dump(args.s14))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for arm, gates in payload["arms"].items():
        leaders = sorted({deck["leader"] for deck in gates.values()})
        costs = [float(deck["summary"]["average_cost"]) for deck in gates.values()]
        print(f"{arm}: leaders={','.join(leaders)} mean_cost={sum(costs) / len(costs):.3f}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
