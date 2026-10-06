#!/usr/bin/env python3
"""Rebuild retained ablation descriptors after evaluator schema changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_selfplay_games import load_games
from strategy_descriptor import SCHEMA_VERSION, build_strategy_descriptor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "train-ablation-1781126582" / "results"
FAMILIES = (
    "reward_screens",
    "draft_screens",
    "curriculum_screens",
    "league_screens",
    "shaping_screens",
    "terminal_closure_screen",
)
REWARD_FAMILIES = (
    "terminal_safe_reward_screens",
    "terminal_safe_reward_followup",
    "terminal_safe_reward_tail_followup",
    "terminal_safe_reward_exploration_followup",
    "terminal_safe_reward_horizon_followup",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve(path: str) -> Path:
    direct = ROOT / path
    if direct.is_file():
        return direct
    nested = ROOT / "train-ablation-1781126582" / path
    if nested.is_file():
        return nested
    raise FileNotFoundError(path)


def rebuild_descriptor(trace_path: Path, descriptor_path: Path) -> dict[str, Any]:
    previous = read_json(descriptor_path)
    descriptor = build_strategy_descriptor(
        load_games([trace_path]),
        label=str(previous["label"]),
        checkpoint_sha256=previous.get("checkpoint_sha256"),
        payoff_cells=previous.get("payoff_vector", []),
    )
    if descriptor["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected descriptor schema: {descriptor_path}")
    temporary = descriptor_path.with_suffix(descriptor_path.suffix + ".tmp")
    temporary.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    temporary.replace(descriptor_path)
    return descriptor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=FAMILIES + REWARD_FAMILIES,
        help="Rebuild only this family; may be repeated. Defaults to all families.",
    )
    args = parser.parse_args()
    families = tuple(args.family or FAMILIES + REWARD_FAMILIES)

    rebuilt = 0
    indexes = [
        path
        for family in families
        for path in sorted((RESULTS / family).glob("*evaluation_index.json"))
    ]
    for index_path in indexes:
        family = index_path.parent.name
        index = read_json(index_path)
        for arm in index["arms"]:
            for window in arm["windows"]:
                for mode in ("sample", "argmax"):
                    trace_path = resolve(str(window[f"{mode}_trace"]))
                    descriptor_path = resolve(str(window[f"{mode}_descriptor"]))
                    descriptor = rebuild_descriptor(trace_path, descriptor_path)
                    lightning = {
                        name: {
                            "eligible": sequence["eligible"],
                            "completed": sequence["completed"],
                            "converted": sequence["converted"],
                        }
                        for name, sequence in descriptor["sequences"].items()
                        if name.startswith("lightning.")
                    }
                    rebuilt += 1
                    print(
                        f"{family} {arm['id']} p{int(window['update']):06d} "
                        f"{mode}: {lightning}"
                    )
    print(f"rebuilt {rebuilt} descriptors at schema v{SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
