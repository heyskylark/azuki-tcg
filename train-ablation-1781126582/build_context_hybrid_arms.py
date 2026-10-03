#!/usr/bin/env python3
"""Build gate-leader/main factorial fixed decks from uniform context drafts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from build_frozen_deck_arms import _deck_payload
from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool


SIBLING_BY_GATE = {
  "STT01-002": "AZK01-120",
  "AZK01-120": "STT01-002",
  "STT02-002": "AZK01-126",
  "AZK01-126": "STT02-002",
  "AZK01-122": "STT04-002",
  "STT04-002": "AZK01-122",
  "AZK01-124": "STT03-002",
  "STT03-002": "AZK01-124",
}
ARMS = ("matched", "sibling_gate_main", "sibling_leader_main", "sibling_both_main")


def _greedy_main(context: dict) -> Counter[str]:
  main = Counter(
    {
      str(card["code"]): int(card["copies"])
      for card in context["greedy"]["cards"]
    }
  )
  if sum(main.values()) != 50:
    raise ValueError(
      f"Context {context.get('context_id')} has {sum(main.values())} main cards"
    )
  return main


def build_arms(payload: dict) -> dict:
  raw_contexts = payload.get("contexts")
  if not isinstance(raw_contexts, list) or len(raw_contexts) != 16:
    raise ValueError("Uniform deck dump must contain exactly 16 contexts")
  contexts: dict[tuple[str, str], dict] = {}
  leaders_by_gate: dict[str, list[str]] = {}
  for context in raw_contexts:
    gate = str(context["gate_code"])
    leader = str(context["leader_code"])
    key = (gate, leader)
    if key in contexts:
      raise ValueError(f"Duplicate uniform context {key}")
    contexts[key] = context
    leaders_by_gate.setdefault(gate, []).append(leader)
  if set(leaders_by_gate) != set(SIBLING_BY_GATE):
    raise ValueError("Uniform deck dump does not cover the registered eight gates")
  if any(len(set(leaders)) != 2 for leaders in leaders_by_gate.values()):
    raise ValueError("Every gate must have exactly two leader contexts")

  catalog = build_deck_build_catalog(load_training_deck_pool())
  records_by_code = catalog.records_by_code
  arms: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
  for context in raw_contexts:
    element = str(context["element"])
    gate = str(context["gate_code"])
    leader = str(context["leader_code"])
    context_id = str(context["context_id"])
    sibling_gate = SIBLING_BY_GATE[gate]
    other_leaders = sorted(set(leaders_by_gate[gate]) - {leader})
    if len(other_leaders) != 1:
      raise ValueError(f"Context {context_id} has no unique sibling leader")
    sibling_leader = other_leaders[0]
    sources = {
      "matched": (gate, leader),
      "sibling_gate_main": (sibling_gate, leader),
      "sibling_leader_main": (gate, sibling_leader),
      "sibling_both_main": (sibling_gate, sibling_leader),
    }
    target_main = _greedy_main(context)
    for arm, source_key in sources.items():
      source = contexts.get(source_key)
      if source is None:
        raise ValueError(f"Missing source context {source_key} for {context_id}")
      source_main = _greedy_main(source)
      arms[arm][context_id] = {
        **_deck_payload(
          gate=gate,
          leader=leader,
          main=source_main,
          native_main=target_main,
          records_by_code=records_by_code,
        ),
        "target_context": context_id,
        "element": element,
        "target_gate": gate,
        "target_leader": leader,
        "main_source_context": str(source["context_id"]),
      }
  return {
    "schema_version": 1,
    "source_checkpoint": payload.get("checkpoint"),
    "arm_order": list(ARMS),
    "arm_definitions": {
      "matched": "Target gate and leader with the main drafted for that context.",
      "sibling_gate_main": "Target gate and leader with its sibling gate's main for the same leader.",
      "sibling_leader_main": "Target gate and leader with the same gate's main for the sibling leader.",
      "sibling_both_main": "Target gate and leader with the sibling gate and sibling leader's main.",
    },
    "arms": arms,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--decks", type=Path, required=True)
  parser.add_argument("--json", type=Path, required=True)
  args = parser.parse_args()
  payload = build_arms(json.loads(args.decks.read_text(encoding="utf-8")))
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(
    f"[context-hybrids] contexts={len(payload['arms']['matched'])} "
    f"arms={len(payload['arms'])} output={args.json}",
    flush=True,
  )


if __name__ == "__main__":
  main()
