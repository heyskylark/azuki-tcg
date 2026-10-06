#!/usr/bin/env python3
"""Build matched strategic and entity-only fixed-deck curriculum panels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from deck_building import build_deck_build_catalog
from probe_deck_gate_interaction import ARCHETYPES, build_archetype_decks
from probe_gate_kl import GATE_CODE_PAIRS
from training_deck_pool import load_training_deck_pool


SOURCE = ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_decks.json"
OUTPUT = ROOT / "train-ablation-1781126582" / "strategic_exposure_decks.json"
REFERENCE_PANEL_SIZE = 18


def _deck_entry(
    *,
    catalog,
    element: str,
    gate: str,
    leader_id: int,
    main: list[tuple[str, int]],
    treatment: str,
) -> dict[str, object]:
    leader = catalog.records_by_def_id[int(leader_id)].card_code
    cards = [
        {"card_id": leader, "quantity": 1},
        {"card_id": gate, "quantity": 1},
        *(
            {"card_id": card_id, "quantity": int(quantity)}
            for card_id, quantity in main
        ),
        {"card_id": "IKZ-001", "quantity": 10},
    ]
    canonical = json.dumps(cards, sort_keys=True, separators=(",", ":")).encode()
    slug = f"curriculum-{treatment}-{element.lower()}-{gate.lower()}-{leader.lower()}"
    return {
        "deck_name": slug,
        "deck_slug": slug,
        "source_submission_number": None,
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
        "element": element,
        "leader_card_id": leader,
        "gate_card_id": gate,
        "cards": cards,
        "reference_role": treatment,
    }


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    pool = load_training_deck_pool(SOURCE)
    catalog = build_deck_build_catalog(pool)
    retained = list(source["decks"][:REFERENCE_PANEL_SIZE])
    panels: dict[str, list[dict[str, object]]] = {
        "strategic_exposure": [],
        "entity_only_control": [],
    }

    for treatment in panels:
        for element, gates in GATE_CODE_PAIRS.items():
            archetypes = build_archetype_decks(catalog, element)
            archetype = ARCHETYPES[element][0] if treatment == "strategic_exposure" else "entity_only"
            main = archetypes[archetype]
            for gate in gates:
                for leader_id in catalog.leader_def_ids_by_element[element]:
                    panels[treatment].append(
                        _deck_entry(
                            catalog=catalog,
                            element=element,
                            gate=gate,
                            leader_id=int(leader_id),
                            main=main,
                            treatment=treatment,
                        )
                    )

    strategic_start = len(retained)
    retained.extend(panels["strategic_exposure"])
    control_start = len(retained)
    retained.extend(panels["entity_only_control"])
    strategic_indices = list(range(strategic_start, control_start))
    control_indices = list(range(control_start, len(retained)))

    payload = {
        "schema_version": 1,
        "source": {
            "builder": str(Path(__file__).relative_to(ROOT)),
            "base_path": str(SOURCE.relative_to(ROOT)),
            "base_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        },
        "normalization_notes": [
            "Preserved the Arena promotion/holdout reference panel at indices 0-17.",
            "Appended matched legal strategic and entity-only panels over every element, gate, and compatible leader.",
            "Each treatment contains one deck for every element x gate x leader stratum.",
        ],
        "summary": {
            "reference_panel_size": REFERENCE_PANEL_SIZE,
            "promotion_reference_deck_indices": source["summary"]["promotion_reference_deck_indices"],
            "holdout_reference_deck_indices": source["summary"]["holdout_reference_deck_indices"],
            "strategic_exposure_deck_indices": strategic_indices,
            "entity_only_control_deck_indices": control_indices,
            "strategic_exposure_deck_count": len(strategic_indices),
            "entity_only_control_deck_count": len(control_indices),
        },
        "decks": retained,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    loaded = load_training_deck_pool(OUTPUT)
    if len(loaded) != len(retained):
        raise RuntimeError(f"Expected {len(retained)} legal decks, loaded {len(loaded)}")
    print(
        f"wrote {OUTPUT.relative_to(ROOT)}: "
        f"strategic={strategic_indices[0]}..{strategic_indices[-1]}, "
        f"control={control_indices[0]}..{control_indices[-1]}"
    )


if __name__ == "__main__":
    main()
