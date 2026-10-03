#!/usr/bin/env python3
"""Build heldout-safe, tournament-derived four-card ability exposure prefixes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".codex/docs/azuki_garden_arena_2026-08-15_decks.json"
METADATA = ROOT / "python/config/policy_card_metadata_v1.json"
REFERENCE_PANEL_SIZE = 18
PACKAGE_COUNT = 2
MISSING_TOURNAMENT_CONTEXTS = {
    "AZK01-120:AZK01-119",  # Stormchain / Piko
    "STT01-002:AZK01-119",  # Surge / Piko
    "STT04-002:AZK01-121",  # Ragefire / Kagoro
    "AZK01-126:STT02-001",  # Echoed Waves / Shao
    "AZK01-124:STT03-001",  # Devotion / Bobu
}

# Each recipe is an ordered set of mechanical roles. Card order within a role is
# an intentional preference, not tournament frequency. Every selected card must
# occur in the single tournament deck recorded as the package source.
RECIPES = {
    "LIGHTNING": (
        (
            ("Gate Power body", ("STT01-005", "STT01-010", "STT01-011", "AZK01-077", "AZK01-015")),
            ("cheap weapon", ("AZK01-094", "STT01-012", "STT01-014", "STT01-013")),
            ("weapon discard setup", ("STT01-003", "AZK01-097", "STT01-004")),
            ("equipped payoff or recovery", ("AZK01-100", "STT01-008", "STT01-009", "AZK01-041")),
        ),
        (
            ("Gate Power body", ("STT01-005", "STT01-010", "STT01-011", "AZK01-077", "AZK01-015")),
            ("cheap weapon", ("AZK01-094", "STT01-012", "STT01-014", "STT01-013")),
            ("equipped attacker", ("STT01-008", "AZK01-039", "AZK01-098")),
            ("weapon reuse or discard setup", ("AZK01-100", "AZK01-041", "STT01-003", "AZK01-097")),
        ),
    ),
    "WATER": (
        (
            ("Gate Power body", ("STT02-013", "STT02-006", "STT02-003", "AZK01-022", "AZK01-015")),
            ("interaction spell", ("STT02-016", "STT02-015", "AZK01-087", "AZK01-127")),
            ("discard enabler", ("AZK01-022", "STT02-016", "AZK01-029", "AZK01-016")),
            ("card access or return payoff", ("STT02-007", "AZK01-031", "STT02-010", "AZK01-024")),
        ),
        (
            ("Gate Power body", ("STT02-013", "STT02-006", "STT02-003", "AZK01-022", "AZK01-015")),
            ("IKZ recovery or interaction spell", ("AZK01-030", "STT02-016", "STT02-015", "AZK01-087")),
            ("discard enabler", ("AZK01-022", "STT02-016", "AZK01-029", "AZK01-016")),
            ("return or multi-play support", ("AZK01-024", "STT02-010", "STT02-007", "STT02-005")),
        ),
    ),
    "EARTH": (
        (
            ("high Gate Power body", ("STT03-014", "STT03-012", "STT03-011", "AZK01-049", "AZK01-048")),
            ("sacrifice body or payoff", ("AZK01-105", "STT03-004", "STT03-005", "STT03-006")),
            ("healing support", ("AZK01-050", "STT03-015", "AZK01-047", "AZK01-002")),
            ("health conversion or defense", ("AZK01-108", "AZK01-103", "STT03-009", "AZK01-128")),
        ),
        (
            ("high Gate Power body", ("STT03-014", "STT03-012", "STT03-011", "AZK01-049", "AZK01-048")),
            ("Defender or durable body", ("STT03-009", "AZK01-049", "AZK01-048", "STT03-004")),
            ("healing support", ("AZK01-050", "STT03-015", "AZK01-047", "AZK01-002")),
            ("destroyed or sacrificed value", ("STT03-005", "STT03-006", "AZK01-105", "AZK01-128")),
        ),
    ),
    "FIRE": (
        (
            ("Gate Power body", ("AZK01-116", "AZK01-118", "AZK01-062", "AZK01-060", "AZK01-058")),
            ("cheap extra play", ("STT04-003", "STT04-004", "STT04-005", "AZK01-056")),
            ("multi-play or sacrifice payoff", ("AZK01-113", "AZK01-118", "AZK01-058", "AZK01-060")),
            ("self-damage or sacrifice enabler", ("STT04-016", "STT04-015", "AZK01-117", "AZK01-115")),
        ),
        (
            ("Gate Power body", ("AZK01-116", "AZK01-118", "AZK01-062", "AZK01-060", "AZK01-058")),
            ("self-damage enabler", ("STT04-016", "STT04-015", "AZK01-065", "STT04-003")),
            ("damaged-entity payoff", ("STT04-009", "STT04-007", "AZK01-059", "AZK01-129")),
            ("sacrifice support", ("STT04-004", "AZK01-115", "AZK01-058", "STT04-005")),
        ),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _card_counts(deck: dict) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in deck["cards"]:
        counts[item["card_id"]] += int(item["quantity"])
    return counts


def canonical_deck_signature(deck: dict) -> str:
    pairs = sorted((code, quantity) for code, quantity in _card_counts(deck).items() if quantity)
    encoded = json.dumps(pairs, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _choose_package(counts: Counter[str], recipe: tuple) -> tuple[list[str], list[str]] | None:
    cards: list[str] = []
    roles: list[str] = []
    for role, preferences in recipe:
        code = next((candidate for candidate in preferences if counts[candidate] and candidate not in cards), None)
        if code is None:
            return None
        cards.append(code)
        roles.append(role)
    return cards, roles


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def build_payloads() -> tuple[dict, dict]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    metadata_payload = json.loads(METADATA.read_text(encoding="utf-8"))
    records = {record["card_code"]: dict(record) for record in metadata_payload["records"]}

    holdout_indices = list(source["summary"]["holdout_reference_deck_indices"])
    holdout_signatures = {canonical_deck_signature(source["decks"][index]) for index in holdout_indices}
    heldout_excluded_rows = []
    duplicate_excluded_rows = []
    eligible = []
    seen_eligible_signatures: set[str] = set()
    all_context_counts: Counter[str] = Counter()
    eligible_context_counts: Counter[str] = Counter()
    for index, deck in enumerate(source["decks"]):
        context = f'{deck["gate_card_id"]}:{deck["leader_card_id"]}'
        all_context_counts[context] += 1
        signature = canonical_deck_signature(deck)
        row_provenance = {
            "source_index": index,
            "source_submission_number": deck.get("source_submission_number"),
            "source_content_sha256": deck.get("content_sha256"),
            "canonical_deck_signature": signature,
        }
        if signature in holdout_signatures:
            heldout_excluded_rows.append({
                **row_provenance,
                "reason": "canonical_full_deck_signature_matches_holdout",
            })
        elif signature in seen_eligible_signatures:
            duplicate_excluded_rows.append({
                **row_provenance,
                "reason": "duplicate_canonical_full_deck_signature",
            })
        else:
            seen_eligible_signatures.add(signature)
            eligible.append((index, deck, _card_counts(deck), signature))
            eligible_context_counts[context] += 1

    gates = sorted((record for record in records.values() if record["card_type"] == "GATE"), key=lambda r: r["card_code"])
    leaders_by_element = {
        element: sorted(
            (record for record in records.values() if record["card_type"] == "LEADER" and record["element"] == element),
            key=lambda r: r["card_code"],
        )
        for element in RECIPES
    }
    expected_contexts = [
        (gate, leader)
        for gate in gates
        for leader in leaders_by_element[gate["element"]]
    ]
    if len(expected_contexts) != 16:
        raise ValueError(f"Expected sixteen gate/leader contexts, found {len(expected_contexts)}")
    observed_missing = {context for context in (f'{g["card_code"]}:{l["card_code"]}' for g, l in expected_contexts) if not all_context_counts[context]}
    if observed_missing != MISSING_TOURNAMENT_CONTEXTS:
        raise ValueError(f"Tournament context coverage changed: {sorted(observed_missing)}")

    contexts: dict[str, list[dict]] = {}
    coverage = []
    for gate, leader in expected_contexts:
        element = gate["element"]
        context = f'{gate["card_code"]}:{leader["card_code"]}'
        packages = []
        used_card_sets: set[tuple[str, ...]] = set()
        used_source_signatures: set[str] = set()
        for recipe_number, recipe in enumerate(RECIPES[element], start=1):
            choices = []
            for index, deck, counts, signature in eligible:
                if deck["element"] != element:
                    continue
                chosen = _choose_package(counts, recipe)
                if chosen is None:
                    continue
                cards, roles = chosen
                card_set = tuple(sorted(cards))
                if card_set in used_card_sets:
                    continue
                exact = deck["gate_card_id"] == gate["card_code"] and deck["leader_card_id"] == leader["card_code"]
                rank = (
                    not exact,
                    deck["gate_card_id"] != gate["card_code"],
                    deck["leader_card_id"] != leader["card_code"],
                    signature in used_source_signatures,
                    index,
                )
                choices.append((rank, index, deck, signature, cards, roles, card_set))
            if not choices:
                raise ValueError(f"No heldout-safe tournament package for {context} recipe {recipe_number}")
            _, index, deck, signature, cards, roles, card_set = min(choices, key=lambda item: item[0])
            assignment = "exact_tournament_context" if (
                deck["gate_card_id"] == gate["card_code"] and deck["leader_card_id"] == leader["card_code"]
            ) else "constructed_adapted_same_element_tournament_package"
            package_id = f"{context}:mechanic-{recipe_number}"
            package = {"id": package_id, "cards": cards}
            packages.append(package)
            used_card_sets.add(card_set)
            used_source_signatures.add(signature)
            package["_provenance"] = {
                "package_id": package_id,
                "target_context": context,
                "target_gate_name": gate["name"],
                "target_leader_name": leader["name"],
                "element": element,
                "context_assignment": assignment,
                "tournament_context_missing": context in MISSING_TOURNAMENT_CONTEXTS,
                "source_index": index,
                "source_submission_number": deck.get("source_submission_number"),
                "source_deck_slug": deck.get("deck_slug"),
                "source_content_sha256": deck.get("content_sha256"),
                "source_canonical_deck_signature": signature,
                "source_gate": deck["gate_card_id"],
                "source_leader": deck["leader_card_id"],
                "intended_mechanics": [
                    {
                        "role": role,
                        "card": code,
                        "name": records[code]["name"],
                        "card_type": records[code]["card_type"],
                        "element": records[code]["element"],
                        "ikz_cost": records[code]["ikz_cost"],
                        "gate_points": records[code]["gate_points"],
                        "effect_text": records[code]["effect_text"],
                    }
                    for role, code in zip(roles, cards)
                ],
                "context_mechanics": {
                    "gate_effect": gate["effect_text"],
                    "leader_effect": leader["effect_text"],
                    "intent": "Expose coherent Gate Power plus activation-supporting pieces; this is opportunity exposure, not a claim that an outcome occurs.",
                },
            }
        contexts[context] = [{"id": package["id"], "cards": package["cards"]} for package in packages]
        coverage.append({
            "context": context,
            "element": element,
            "tournament_source_row_count": all_context_counts[context],
            "eligible_exact_source_row_count": eligible_context_counts[context],
            "tournament_context_missing": context in MISSING_TOURNAMENT_CONTEXTS,
            "packages": [package["_provenance"] for package in packages],
        })

    prefix_pool = {
        "schema_id": "azuki.strategic_prefix_pool",
        "schema_version": 1,
        "contexts": contexts,
        "metadata_sha256": _sha256(METADATA),
    }
    manifest = {
        "schema_id": "azuki.ability_outcome_exposure_manifest",
        "schema_version": 1,
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": _sha256(SOURCE),
            "row_count": len(source["decks"]),
            "unique_canonical_deck_signature_count": len({canonical_deck_signature(deck) for deck in source["decks"]}),
            "reference_panel_size": REFERENCE_PANEL_SIZE,
        },
        "metadata": {
            "path": str(METADATA.relative_to(ROOT)),
            "sha256": _sha256(METADATA),
            "usage": "Consumed unchanged; run from the experiment's frozen catalog snapshot.",
        },
        "heldout_exclusion": {
            "reference_indices": holdout_indices,
            "canonical_signature_definition": "sha256(JSON of sorted aggregated full-deck [card_code,quantity] pairs)",
            "signature_count": len(holdout_signatures),
            "signatures": sorted(holdout_signatures),
            "excluded_source_row_count": len(heldout_excluded_rows),
            "excluded_source_rows": heldout_excluded_rows,
        },
        "source_signature_deduplication": {
            "eligible_unique_signature_count": len(eligible),
            "excluded_duplicate_row_count": len(duplicate_excluded_rows),
            "excluded_duplicate_rows": duplicate_excluded_rows,
        },
        "design": {
            "context_assignment": "uniform over exactly sixteen gate/leader contexts",
            "packages_per_context": PACKAGE_COUNT,
            "forced_main_picks": 4,
            "free_main_picks_after_prefix": 46,
            "prefix_length_probabilities": {"0": 0.5, "4": 0.5},
            "fully_free_episode_probability": 0.5,
            "forced_rows_count_as_policy_decisions": False,
            "selection": "two explicit mechanical recipes per element; source rows are ranked by context match, never field frequency",
        },
        "context_coverage": coverage,
        "element_package_counts": dict(sorted(Counter(item["element"] for item in coverage for _ in item["packages"]).items())),
    }
    selected_signatures = {
        package["source_canonical_deck_signature"]
        for item in coverage
        for package in item["packages"]
    }
    if selected_signatures & holdout_signatures:
        raise ValueError("Selected package source overlaps a holdout full-deck signature")
    if set(manifest["element_package_counts"].values()) != {8}:
        raise ValueError(f"Unbalanced element coverage: {manifest['element_package_counts']}")
    return prefix_pool, manifest


def build(output_dir: Path) -> tuple[Path, Path]:
    prefix_pool, manifest = build_payloads()
    output_dir = output_dir.resolve()
    prefix_path = output_dir / "strategic_prefix_pool.json"
    manifest_path = output_dir / "exposure_manifest.json"
    if prefix_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to replace exposure output in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path.write_bytes(_json_bytes(prefix_pool))
    manifest_path.write_bytes(_json_bytes(manifest))
    return prefix_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prefix_path, manifest_path = build(args.output_dir)
    print(f"wrote {prefix_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
