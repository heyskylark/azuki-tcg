#!/usr/bin/env python3
"""Build deterministic supplied-deck arms for the p2930 counterfactual."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from deck_building import MAIN_CARD_TYPES, build_deck_build_catalog
from training_deck_pool import load_training_deck_pool


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_P2930 = (
    REPO_ROOT
    / "train-ablation-1781126582/results/promotionv2_p2930_analysis/decks.json"
)
DEFAULT_S14 = REPO_ROOT / "train-ablation-1781126582/results/s14prod45_ep2000_decks.json"
GATES = (
    "STT01-002",
    "AZK01-120",
    "STT02-002",
    "AZK01-126",
    "AZK01-122",
    "STT04-002",
    "AZK01-124",
    "STT03-002",
)
WATER_GATES = frozenset(("STT02-002", "AZK01-126"))


def _read_dump(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise ValueError(f"Deck dump has no gates object: {path}")
    missing = set(GATES).difference(gates)
    if missing:
        raise ValueError(f"Deck dump {path} is missing gates: {sorted(missing)}")
    return payload


def _argmax_main(gate_payload: dict) -> Counter[str]:
    cards = gate_payload["argmax_deck"]["cards"]
    main = Counter({str(card["code"]): int(card["copies"]) for card in cards})
    if sum(main.values()) != 50:
        raise ValueError(f"Argmax deck has {sum(main.values())} main cards, expected 50")
    return main


def _sampled_frequency(gate_payload: dict) -> dict[str, float]:
    return {
        str(card["code"]): float(card["copies"])
        for card in gate_payload["sampled"]["mean_copies"]
    }


def _multiset_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left).union(right)
    intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return float(intersection / union) if union else 0.0


def _curve_balanced(native: Counter[str], records_by_code) -> Counter[str]:
    """Reweight 15 native cores to 10 four-ofs plus five two-ofs by cost."""
    ordered = sorted(
        native,
        key=lambda code: (
            int(records_by_code[code].ikz_cost),
            -int(native[code]),
            code,
        ),
    )
    if len(ordered) < 15:
        raise ValueError(f"Curve arm needs at least 15 native cards, got {len(ordered)}")
    selected = ordered[:15]
    four_ofs = set(
        sorted(
            selected,
            key=lambda code: (
                int(native[code]) != 4,
                int(records_by_code[code].ikz_cost),
                code,
            ),
        )[:10]
    )
    return Counter({code: 4 if code in four_ofs else 2 for code in selected})


def _water_spell_restored(
    p2930_gate: dict,
    s14_gate: dict,
    records_by_code,
) -> Counter[str]:
    """Keep eight p2930 non-spell cores at four copies plus S14's 18 spells."""
    native = _argmax_main(p2930_gate)
    frequency = _sampled_frequency(p2930_gate)
    nonspells = [
        code for code in native if records_by_code[code].card_type != "SPELL"
    ]
    nonspells.sort(
        key=lambda code: (
            -frequency.get(code, 0.0),
            -native[code],
            int(records_by_code[code].ikz_cost),
            code,
        )
    )
    if len(nonspells) < 8:
        raise ValueError("Water arm cannot retain eight p2930 non-spell cores")
    restored = Counter({code: 4 for code in nonspells[:8]})

    s14_main = _argmax_main(s14_gate)
    spell_shell = Counter(
        {
            code: quantity
            for code, quantity in s14_main.items()
            if records_by_code[code].card_type == "SPELL"
        }
    )
    if sum(spell_shell.values()) != 18:
        raise ValueError(
            f"Expected the S14 Water spell shell to contain 18 slots, got "
            f"{sum(spell_shell.values())}"
        )
    restored.update(spell_shell)
    if sum(restored.values()) != 50:
        raise AssertionError("Water spell-restored main must contain 50 cards")
    return restored


def _deck_payload(
    *,
    gate: str,
    leader: str,
    main: Counter[str],
    native_main: Counter[str],
    records_by_code,
) -> dict:
    if sum(main.values()) != 50:
        raise ValueError(f"{gate} main has {sum(main.values())} cards")
    if any(quantity < 1 or quantity > 4 for quantity in main.values()):
        raise ValueError(f"{gate} main violates the one-to-four copy limit")
    gate_element = records_by_code[gate].element
    for code in main:
        record = records_by_code[code]
        if record.card_type not in MAIN_CARD_TYPES:
            raise ValueError(f"{gate} main includes non-main card {code}")
        if record.element not in ("NORMAL", gate_element):
            raise ValueError(f"{gate} main includes off-element card {code}")
    if records_by_code[leader].element != gate_element:
        raise ValueError(f"{gate} leader {leader} has the wrong element")

    type_slots = Counter()
    cost_total = 0
    for code, quantity in main.items():
        record = records_by_code[code]
        type_slots[record.card_type] += quantity
        cost_total += int(record.ikz_cost) * quantity
    cards = [
        {
            "code": code,
            "name": code,
            "type": records_by_code[code].card_type,
            "cost": int(records_by_code[code].ikz_cost),
            "copies": quantity,
        }
        for code, quantity in sorted(
            main.items(), key=lambda item: (records_by_code[item[0]].ikz_cost, item[0])
        )
    ]
    native_deck = [
        [leader, 1],
        [gate, 1],
        *[[card["code"], card["copies"]] for card in cards],
        ["IKZ-001", 10],
    ]
    return {
        "gate": gate,
        "leader": leader,
        "main": cards,
        "native_deck": native_deck,
        "summary": {
            "main_slots": 50,
            "unique_cards": len(main),
            "average_cost": round(cost_total / 50.0, 4),
            "spell_slots": int(type_slots["SPELL"]),
            "entity_slots": int(type_slots["ENTITY"]),
            "weapon_slots": int(type_slots["WEAPON"]),
            "four_of_slots": 4 * sum(quantity == 4 for quantity in main.values()),
            "two_of_slots": 2 * sum(quantity == 2 for quantity in main.values()),
            "native_multiset_jaccard": round(_multiset_jaccard(main, native_main), 6),
        },
    }


def build_arms(p2930: dict, s14: dict) -> dict:
    pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(pool)
    records_by_code = catalog.records_by_code
    arms: dict[str, dict] = {
        "native_p2930": {},
        "curve_balanced": {},
        "water_spell_restored": {},
        "s14_greedy": {},
    }
    for gate in GATES:
        p_gate = p2930["gates"][gate]
        s_gate = s14["gates"][gate]
        native = _argmax_main(p_gate)
        s14_main = _argmax_main(s_gate)
        p_leader = str(p_gate["argmax_deck"]["leader"]["code"])
        s14_leader = str(s_gate["argmax_deck"]["leader"]["code"])
        mains = {
            "native_p2930": native,
            "curve_balanced": _curve_balanced(native, records_by_code),
            "water_spell_restored": (
                _water_spell_restored(p_gate, s_gate, records_by_code)
                if gate in WATER_GATES
                else native
            ),
            "s14_greedy": s14_main,
        }
        leaders = {
            "native_p2930": p_leader,
            "curve_balanced": p_leader,
            "water_spell_restored": p_leader,
            "s14_greedy": s14_leader,
        }
        for arm, main in mains.items():
            arms[arm][gate] = _deck_payload(
                gate=gate,
                leader=leaders[arm],
                main=main,
                native_main=native,
                records_by_code=records_by_code,
            )
    return {
        "schema_version": 1,
        "sources": {
            "p2930_checkpoint": p2930.get("checkpoint"),
            "s14_checkpoint": s14.get("checkpoint"),
        },
        "arm_definitions": {
            "native_p2930": "Exact p2930 stable-argmax leader and 50-card main per gate.",
            "curve_balanced": (
                "The 15 cheapest p2930 argmax cores: ten as four-ofs and five as two-ofs."
            ),
            "water_spell_restored": (
                "Native p2930 for non-Water gates; Water keeps eight high-frequency p2930 "
                "non-spell cores as four-ofs plus the exact 18-slot S14 Water spell shell."
            ),
            "s14_greedy": "Exact S14 stable-argmax leader and 50-card main per gate.",
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
        costs = [float(deck["summary"]["average_cost"]) for deck in gates.values()]
        spells = [int(deck["summary"]["spell_slots"]) for deck in gates.values()]
        print(
            f"{arm}: mean_cost={sum(costs) / len(costs):.3f} "
            f"spell_slots={min(spells)}-{max(spells)}"
        )
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
