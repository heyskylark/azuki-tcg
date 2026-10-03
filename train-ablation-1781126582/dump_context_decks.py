#!/usr/bin/env python3
"""Dump greedy and stochastic decks for every uniform gate-leader context."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from analyze_decks import CARD_META_PATH, GATE_NAMES
from probe_gate_kl import EpisodeRunner, GATE_CODE_PAIRS, OPPONENT_GATE


DEFAULT_CONFIG = Path("python/config/azuki_deckbuild_native_3090.ini")


def _card_metadata() -> dict[str, dict[str, object]]:
  payload = json.loads(CARD_META_PATH.read_text(encoding="utf-8"))
  return {
    str(record["card_code"]): {
      "name": str(record.get("name", "?")),
      "type": str(record.get("card_type", "?")),
      "element": str(record.get("element", "?")),
      "cost": int(record.get("ikz_cost", 0) or 0),
    }
    for record in payload["records"]
  }


def _distribution_summary(decks: list[Counter[str]]) -> dict[str, float]:
  if not decks:
    raise ValueError("At least one deck is required")
  jaccards: list[float] = []
  for left, right in combinations(decks, 2):
    cards = set(left) | set(right)
    intersection = sum(min(left.get(card, 0), right.get(card, 0)) for card in cards)
    union = sum(max(left.get(card, 0), right.get(card, 0)) for card in cards)
    jaccards.append(intersection / union if union else 1.0)
  return {
    "main_unique_mean": sum(len(deck) for deck in decks) / len(decks),
    "singleton_slot_share_mean": sum(
      sum(count for count in deck.values() if count == 1) / 50.0 for deck in decks
    ) / len(decks),
    "pair_slot_share_mean": sum(
      sum(count for count in deck.values() if count == 2) / 50.0 for deck in decks
    ) / len(decks),
    "triplet_slot_share_mean": sum(
      sum(count for count in deck.values() if count == 3) / 50.0 for deck in decks
    ) / len(decks),
    "quad_slot_share_mean": sum(
      sum(count for count in deck.values() if count == 4) / 50.0 for deck in decks
    ) / len(decks),
    "pairwise_multiset_jaccard_mean": (
      sum(jaccards) / len(jaccards) if jaccards else 1.0
    ),
  }


def _deck_summary(
  deck: Counter[str],
  metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
  total = sum(deck.values())
  type_counts: Counter[str] = Counter()
  cost_total = 0
  for code, count in deck.items():
    card = metadata[code]
    type_counts[str(card["type"])] += count
    cost_total += int(card["cost"]) * count
  return {
    "main_total": total,
    "main_unique": len(deck),
    "average_ikz_cost": cost_total / max(total, 1),
    "type_slots": dict(sorted(type_counts.items())),
    "singleton_slot_share": sum(count for count in deck.values() if count == 1)
    / max(total, 1),
    "quad_slot_share": sum(count for count in deck.values() if count == 4)
    / max(total, 1),
  }


def _card_entries(
  deck: Counter[str],
  metadata: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
  return [
    {
      "code": code,
      **metadata[code],
      "copies": count,
    }
    for code, count in sorted(
      deck.items(), key=lambda item: (int(metadata[item[0]]["cost"]), item[0])
    )
  ]


def _markdown(payload: dict[str, object]) -> str:
  lines = [
    "# Uniform Context Deck Compositions",
    "",
    f"Checkpoint: `{payload['checkpoint']}`",
    "",
    f"Stochastic drafts per context: `{payload['stochastic_drafts_per_context']}` ",
    f"at temperature `{payload['sampling']['temperature']}` and smoothing ",
    f"`{payload['sampling']['smoothing_eps']}`.",
    "",
  ]
  contexts = payload["contexts"]
  if not isinstance(contexts, list):
    raise TypeError("contexts must be a list")
  for context in contexts:
    lines.extend(
      [
        f"## {context['element']} | {context['gate_name']} | {context['leader_name']}",
        "",
        "### Greedy",
        "",
        f"Unique `{context['greedy']['summary']['main_unique']}`, average IKZ ",
        f"`{context['greedy']['summary']['average_ikz_cost']:.2f}`, type slots ",
        f"`{context['greedy']['summary']['type_slots']}`.",
        "",
        "| Copies | Cost | Type | Card |",
        "| ---: | ---: | --- | --- |",
      ]
    )
    for card in context["greedy"]["cards"]:
      lines.append(
        f"| {card['copies']} | {card['cost']} | {card['type']} | "
        f"{card['name']} (`{card['code']}`) |"
      )
    distribution = context["stochastic"]["distribution"]
    lines.extend(
      [
        "",
        "### Stochastic",
        "",
        f"Mean unique `{distribution['main_unique_mean']:.2f}`, four-copy slot share ",
        f"`{distribution['quad_slot_share_mean']:.3f}`, within-context multiset ",
        f"Jaccard `{distribution['pairwise_multiset_jaccard_mean']:.3f}`.",
        "",
        "| Mean copies | Cost | Type | Card |",
        "| ---: | ---: | --- | --- |",
      ]
    )
    for card in context["stochastic"]["mean_cards"]:
      lines.append(
        f"| {card['copies']:.3f} | {card['cost']} | {card['type']} | "
        f"{card['name']} (`{card['code']}`) |"
      )
    lines.append("")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--drafts-per-context", type=int, default=24)
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--smoothing-eps", type=float, default=0.05)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--md", type=Path, required=True)
  args = parser.parse_args()
  if args.drafts_per_context < 2:
    raise ValueError("--drafts-per-context must be at least 2")

  from policy.v2 import tcg_sampler

  metadata = _card_metadata()
  runner = EpisodeRunner(
    args.config,
    args.checkpoint,
    args.device,
    uniform_assignment=True,
  )
  catalog = runner.catalog
  code_by_id = {
    int(card_id): record.card_code
    for card_id, record in catalog.records_by_def_id.items()
  }
  opponent_leader_id = catalog.leader_def_ids_by_element["WATER"][0]
  opponent_leader_code = code_by_id[int(opponent_leader_id)]

  def drafted_main() -> Counter[str]:
    state = runner.base_env._states[0]
    return Counter(
      code_by_id[int(state.main_card_def_ids[index])]
      for index in range(int(state.main_count))
    )

  contexts: list[dict[str, object]] = []
  try:
    for element, gate_codes in GATE_CODE_PAIRS.items():
      leader_codes = [
        code_by_id[int(leader_id)]
        for leader_id in catalog.leader_def_ids_by_element[element]
      ]
      for gate_code in gate_codes:
        for leader_code in leader_codes:
          tcg_sampler.set_sampling_params(
            subaction_temperature=1e-6,
            smoothing_eps=0.0,
          )
          runner.run_episode(
            71_000_003,
            gate_code,
            None,
            p0_leader_code=leader_code,
            p1_leader_code=opponent_leader_code,
          )
          greedy = drafted_main()

          tcg_sampler.set_sampling_params(
            subaction_temperature=args.temperature,
            smoothing_eps=args.smoothing_eps,
          )
          stochastic_decks: list[Counter[str]] = []
          raw_decks: list[dict[str, object]] = []
          aggregate: Counter[str] = Counter()
          for draft_index in range(args.drafts_per_context):
            seed = 73_000_019 + 100_003 * draft_index
            runner.run_episode(
              seed,
              gate_code,
              None,
              p0_leader_code=leader_code,
              p1_leader_code=opponent_leader_code,
            )
            deck = drafted_main()
            if sum(deck.values()) != 50:
              raise RuntimeError("Uniform context draft did not contain 50 main cards")
            stochastic_decks.append(deck)
            aggregate.update(deck)
            raw_decks.append(
              {
                "draft_index": draft_index,
                "seed": seed,
                "summary": _deck_summary(deck, metadata),
                "cards": _card_entries(deck, metadata),
              }
            )
          mean_deck = Counter(
            {
              code: count / args.drafts_per_context
              for code, count in aggregate.items()
            }
          )
          contexts.append(
            {
              "context_id": f"{element}:{gate_code}:{leader_code}",
              "element": element,
              "gate_code": gate_code,
              "gate_name": GATE_NAMES.get(gate_code, gate_code),
              "leader_code": leader_code,
              "leader_name": str(metadata[leader_code]["name"]),
              "greedy": {
                "summary": _deck_summary(greedy, metadata),
                "cards": _card_entries(greedy, metadata),
              },
              "stochastic": {
                "distribution": _distribution_summary(stochastic_decks),
                "mean_cards": _card_entries(mean_deck, metadata),
                "decks": raw_decks,
              },
            }
          )
          print(
            f"[context-decks] {element} {gate_code} {leader_code} "
            f"greedy_unique={len(greedy)} stochastic_unique_mean="
            f"{_distribution_summary(stochastic_decks)['main_unique_mean']:.2f}",
            flush=True,
          )
  finally:
    runner.vecenv.close()

  payload: dict[str, object] = {
    "schema_version": 1,
    "checkpoint": str(args.checkpoint.resolve()),
    "lifecycle": "uniform_gate_uniform_compatible_leader_50_main_picks",
    "stochastic_drafts_per_context": args.drafts_per_context,
    "sampling": {
      "temperature": args.temperature,
      "smoothing_eps": args.smoothing_eps,
    },
    "contexts": contexts,
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(_markdown(payload), encoding="utf-8")
  print(f"[context-decks] wrote {args.json} and {args.md}", flush=True)


if __name__ == "__main__":
  main()
