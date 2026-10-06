#!/usr/bin/env python3
"""Build immutable current-source strategy diagnostics; never launch training."""
from __future__ import annotations

import argparse
from collections import Counter
import configparser
import hashlib
import io
import json
import importlib.metadata
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/src"))
from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool

RESULTS = ROOT / "train-ablation-1781126582/results/strategy_discovery_v1"
SOURCE = ROOT / ".codex/docs/azuki_garden_arena_2026-08-15_decks.json"
METADATA = ROOT / "python/config/policy_card_metadata_v1.json"
BASE = ROOT / "python/config/azuki_reward_r14_tail_030_anneal_100_45m.ini"
SCREEN_ROWS = 15_006_720
CONFIRM_ROWS = 45_004_800
ROWS_PER_UPDATE = 15_360
SOURCE_GLOBS = (
    "python/src/policy/v2/**/*.py", "python/src/azk_puffer/**/*.py",
    "python/src/**/*.py",
    "src/**/*.c", "include/**/*.h", "python/src/*.h",
)
SOURCE_FILES = (
    "python/src/league_training.py", "python/src/train.py",
    "python/src/training_utils.py", "python/src/training_deck_pool.py",
    "python/src/deck_building.py", "python/src/deckbuild_metrics.py",
    "python/src/azk_native.py", "python/src/binding.c",
    "python/src/azk_puffer/config/default.ini", "python/config/policy_card_metadata_v1.json",
    "CMakeLists.txt", "train-ablation-1781126582/build_strategy_discovery.py",
    "train-ablation-1781126582/run_strategy_discovery.py",
    "train-ablation-1781126582/evaluate_reward_screens.py",
    "train-ablation-1781126582/play_selfplay_games.py",
    "train-ablation-1781126582/probe_gate_kl.py",
    "train-ablation-1781126582/strategy_descriptor.py",
)
BINARY_GLOBS = ("build/python/src/binding*.so", "build/**/libazuki_lib.a")
HISTORY = {
    "train-ablation-1781126582/results/draft_screens/registration.json": "historical_R1_parent_not_R14; sampler_fix_status_unproven",
    "train-ablation-1781126582/results/draft_screens/d2_credit_floor_01/run_status.json": "historical_diagnostic; sampler_fix_status_unproven",
    "train-ablation-1781126582/results/draft_screens/d3_credit_zero_tail/run_status.json": "historical_diagnostic; sampler_fix_status_unproven",
    "train-ablation-1781126582/results/terminal_safe_reward_horizon_followup/final_decision_packet.json": "R14_unqualified_diagnostic_control; strategy_qualification_withdrawn",
    "train-ablation-1781126582/results/terminal_safe_reward_horizon_followup/fallback_qualifier/r14_tail_030_anneal_100_45m/run_status.json": "historical_reward_integrity_evidence; sampler_fix_status_unproven",
    "train-ablation-1781126582/results/next_ablation_v1/stage3/random_main_prefix_ladder15_v1/ladder_report.json": "historical_rejected_prefix; sampler_fix_status_unproven",
}


def portable(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def signature(cards: Counter[str]) -> str:
    return hashlib.sha256(json.dumps(sorted((c, n) for c, n in cards.items() if n), separators=(",", ":")).encode()).hexdigest()


def counts(deck: dict) -> Counter[str]:
    result: Counter[str] = Counter()
    for card in deck["cards"]:
        result[card["card_id"]] += int(card["quantity"])
    return result


def entity_control(main: Counter[str], legal: set[str], records: dict) -> tuple[Counter[str], list[dict]]:
    result = Counter({c: n for c, n in main.items() if records[c]["card_type"] == "ENTITY"})
    replacements = []
    for code, number in sorted(main.items()):
        if records[code]["card_type"] == "ENTITY":
            continue
        for _ in range(number):
            candidates = [c for c in legal if records[c]["card_type"] == "ENTITY" and result[c] < 4]
            if not candidates:
                raise ValueError("Insufficient legal entity capacity")
            chosen = min(candidates, key=lambda c: (abs(records[c]["ikz_cost"] - records[code]["ikz_cost"]), c))
            result[chosen] += 1
            replacements.append({"from": code, "to": chosen, "cost_delta": records[chosen]["ikz_cost"] - records[code]["ikz_cost"]})
    return result, replacements


def prefix_package(main: Counter[str], element: str, records: dict, gate: str, leader: str) -> dict:
    """Select enabling roles from supported metadata, never a cost-sorted quartet."""
    def entity(r):
        return r["card_type"] == "ENTITY"

    def text(r, *terms):
        return any(term in r["effect_text"].lower() for term in terms)

    roles = {
        "WATER": [
            ("spell interaction", lambda r: r["card_type"] == "SPELL" and r["element"] == "WATER"),
            ("discard or resource enabler", lambda r: text(r, "discard", "untap", "ikz") and r["element"] == "WATER"),
            ("card access / replay", lambda r: entity(r) and text(r, "draw", "hand", "deck")),
            ("portal body", lambda r: entity(r) and r["gate_points"] > 0),
        ],
        "EARTH": [
            ("defensive body", lambda r: entity(r) and text(r, "defender", "carapace")),
            ("heal setup", lambda r: text(r, "heal")),
            ("defensive support", lambda r: text(r, "health", "defender", "carapace", "heal")),
            ("portal body", lambda r: entity(r) and r["gate_points"] > 0),
        ],
        "FIRE": [
            ("multi-play starter", lambda r: entity(r) and r["ikz_cost"] <= 2),
            ("self-damage enabler", lambda r: text(r, "damage to an entity in your", "damage to this", "damage to your leader", "damage to all leaders")),
            ("enabled attacker / damage payoff", lambda r: entity(r) and text(r, "takes damage", "charge", "takes or deals damage", "after this card attacks")),
            ("second play / portal body", lambda r: entity(r) and r["gate_points"] > 0 and r["ikz_cost"] <= 3),
        ],
        "LIGHTNING": [
            ("weapon", lambda r: r["card_type"] == "WEAPON"),
            ("weapon interaction entity", lambda r: entity(r) and text(r, "weapon", "equipped")),
            ("weapon fuel or recovery", lambda r: text(r, "discard pile", "equipped", "weapon")),
            ("portal body", lambda r: entity(r) and r["gate_points"] > 0),
        ],
    }
    selected: Counter[str] = Counter()
    cards, evidence = [], []
    for role, predicate in roles[element]:
        candidates = [c for c in main if selected[c] < main[c] and predicate(records[c])]
        if not candidates:
            raise ValueError(f"No supported {element} package role: {role}")
        # Prefer distinct cards, then regional abundance; cost is NOT the selection objective.
        code = min(candidates, key=lambda c: (selected[c] > 0, -main[c], c))
        selected[code] += 1
        cards.append(code)
        evidence.append({"role": role, "card": code, "name": records[code]["name"], "effect_text": records[code]["effect_text"], "gate_points": records[code]["gate_points"], "regional_variant_copies": main[code]})
    return {"cards": cards, "rationale": {"selection": "supported enabling roles from regional variant; not cheapest-first; effects are hypotheses, not observed conversion", "roles": evidence, "gate_effect": records[gate]["effect_text"], "leader_effect": records[leader]["effect_text"], "limitation": "Four picks enable opportunities, not guaranteed hand draws or realized effects; gate/leader-specific effects must be measured."}}


def make_entry(main: Counter[str], gate: str, leader: str, element: str, variant: int, treatment: str, provenance: dict) -> dict:
    full = main + Counter({gate: 1, leader: 1, "IKZ-001": 10})
    slug = f"{treatment}-{gate}-{leader}-v{variant}"
    return {"deck_name": slug, "deck_slug": slug, "element": element,
            "gate_card_id": gate, "leader_card_id": leader, "target_gate": gate,
            "target_leader": leader, "variant": variant, "treatment": treatment,
            "reference_role": treatment, "content_sha256": signature(full),
            "cards": [{"card_id": c, "quantity": n} for c, n in sorted(full.items())],
            "provenance": provenance}


def build_pools() -> tuple[dict, dict]:
    source = json.loads(SOURCE.read_text())
    records = {r["card_code"]: r for r in json.loads(METADATA.read_text())["records"]}
    catalog = build_deck_build_catalog(load_training_deck_pool(SOURCE))
    references = source["decks"][:18]
    forbidden = {signature(counts(d)) for d in references}
    legal = {el: {catalog.records_by_def_id[i].card_code for i in ids} for el, ids in catalog.main_def_ids_by_element.items()}
    candidates, excluded, seen = [], [], set()
    for index, deck in enumerate(source["decks"]):
        sig = signature(counts(deck))
        main = Counter({c: n for c, n in counts(deck).items() if records[c]["card_type"] in ("ENTITY", "SPELL", "WEAPON")})
        incompatible = sorted(set(main) - legal[deck["element"]])
        reason = "reference_signature_including_duplicates" if sig in forbidden else "duplicate_source_signature" if sig in seen else "unsupported_draft_element" if incompatible else None
        if reason:
            excluded.append({"source_index": index, "reason": reason, "incompatible_cards": incompatible, "source_signature": sig})
            continue
        seen.add(sig)
        candidates.append((index, deck, main, sig))
    strategic, controls, contexts, coverage = [], [], {}, []
    for element in sorted(legal):
        gates = sorted(r["card_code"] for r in records.values() if r["card_type"] == "GATE" and r["element"] == element)
        leaders = sorted(catalog.records_by_def_id[i].card_code for i in catalog.leader_def_ids_by_element[element])
        for gate in gates:
            for leader in leaders:
                context = f"{gate}:{leader}"
                ordered = sorted((x for x in candidates if x[1]["element"] == element), key=lambda x: (x[1]["gate_card_id"] != gate or x[1]["leader_card_id"] != leader, x[1]["gate_card_id"] != gate, x[1]["leader_card_id"] != leader, x[0]))
                accepted, main_seen, control_seen = [], set(), set()
                # First use exact source mains. Sparse strata may use a documented
                # one-copy redistribution among cards ALREADY in a legal source.
                for perturb in (False, True):
                    for index, deck, original, source_sig in ordered:
                        variants = [(original, None)] if not perturb else [
                            (original + Counter({add: 1}) - Counter({remove: 1}), {"removed": remove, "added": add, "copies": 1})
                            for remove in sorted(original) if original[remove] > 1
                            for add in sorted(original) if add != remove and original[add] < 4
                        ]
                        for main, delta in variants:
                            if signature(main) in main_seen:
                                continue
                            try:
                                package = prefix_package(main, element, records, gate, leader)
                            except ValueError:
                                continue
                            control, replacements = entity_control(main, legal[element], records)
                            if signature(control) in control_seen:
                                continue
                            provenance = {"source_path": portable(SOURCE), "source_index": index, "source_submission_number": deck.get("source_submission_number"), "source_signature": source_sig, "source_gate": deck["gate_card_id"], "source_leader": deck["leader_card_id"], "context_assignment": "exact_gate_leader" if (gate, leader) == (deck["gate_card_id"], deck["leader_card_id"]) else "legal_same_element_gate_leader_transplant", "main_derivation": "one_copy_redistribution_of_existing_legal_regional_cards" if delta else "exact_regional_main", "main_redistribution": delta}
                            variant = len(accepted) + 1
                            s = make_entry(main, gate, leader, element, variant, "strategic_exposure", provenance)
                            e = make_entry(control, gate, leader, element, variant, "entity_only_control", {**provenance, "paired_strategic_signature": s["content_sha256"], "nonentity_replacements": replacements, "confound": "Entity density and abilities change; nearest-cost replacement does not preserve the exact curve or strategic effects. All original entities are retained."})
                            if s["content_sha256"] in forbidden or e["content_sha256"] in forbidden:
                                continue
                            package.update(id=f"{context}:v{variant}", source_signature=source_sig, variant_signature=s["content_sha256"], provenance=provenance)
                            accepted.append((s, e, package))
                            main_seen.add(signature(main))
                            control_seen.add(signature(control))
                            if len(accepted) == 2:
                                break
                        if len(accepted) == 2:
                            break
                    if len(accepted) == 2:
                        break
                if len(accepted) != 2:
                    raise ValueError(f"Cannot derive two distinct supported heldout-safe regional variants for {context}")
                contexts[context] = [p for _, _, p in accepted]
                strategic.extend(s for s, _, _ in accepted)
                controls.extend(e for _, e, _ in accepted)
                coverage.append({"context": context, "element": element, "variants": [s["provenance"] for s, _, _ in accepted]})
    if len(contexts) != 16 or len(strategic) != 32 or len(controls) != 32:
        raise ValueError("Expected sixteen contexts with two matched variants each")
    if {d["content_sha256"] for d in strategic + controls} & forbidden:
        raise ValueError("Supplied deck overlaps promotion/holdout")
    pool = {"schema_version": 1, "decks": references + strategic + controls,
            "source": {"path": portable(SOURCE), "sha256": sha256(SOURCE)},
            "summary": {"reference_panel_size": 18, "promotion_reference_deck_indices": source["summary"]["promotion_reference_deck_indices"], "holdout_reference_deck_indices": source["summary"]["holdout_reference_deck_indices"], "strategic_exposure_deck_indices": list(range(18, 50)), "entity_only_control_deck_indices": list(range(50, 82))},
            "provenance": {"canonical_signature": "sha256(sorted aggregated full-deck card/count pairs), including gate and leader", "excluded_reference_signatures": sorted(forbidden), "excluded_source_rows": excluded, "excluded_reason_counts": dict(Counter(e["reason"] for e in excluded)), "incompatible_source_row_count": sum(bool(e["incompatible_cards"]) for e in excluded), "coverage": coverage, "sparse_coverage_limitation": "One-copy derived variants are not independent regional submissions; provenance identifies all reuse and legal transplants."}}
    prefix = {"schema_id": "azuki.strategic_prefix_pool", "schema_version": 1, "contexts": contexts, "metadata_sha256": sha256(METADATA)}
    return pool, prefix


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def build(output: Path) -> dict:
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"Immutable output already exists: {output}")
    pool, prefix = build_pools()
    pool_path, prefix_path = output / "strategic_exposure_decks.json", output / "strategic_prefix_pool.json"
    artifacts = {pool_path: json_bytes(pool), prefix_path: json_bytes(prefix)}
    specs = [
        ("CONTROL", "control_15m", SCREEN_ROWS, {}),
        ("ENTITY", "entity_exposure_15m", SCREEN_ROWS, {"exposure": "entity_only_control"}),
        ("STRATEGIC", "strategic_exposure_15m", SCREEN_ROWS, {"exposure": "strategic_exposure"}),
        ("PREFIX_RANDOM", "random_prefix_15m", SCREEN_ROWS, {"prefix": "random"}),
        ("PREFIX_STRATEGIC", "strategic_prefix_15m", SCREEN_ROWS, {"prefix": "strategic"}),
        ("REPLAY_OFF", "replay_off_15m", SCREEN_ROWS, {"replay": 0}),
        ("GAMMA1", "learner_gamma1_15m", SCREEN_ROWS, {"gamma": 1.0}),
        ("D2", "d2_credit_floor_01_45m", CONFIRM_ROWS, {"credit": 0.1}),
        ("D3", "d3_credit_zero_tail_45m", CONFIRM_ROWS, {"credit": 0.0}),
    ]
    arms = []
    for arm_id, name, rows, changes in specs:
        run_root = output / "runs" / name
        config_path = output / "configs" / f"{name}.ini"
        config = configparser.ConfigParser(interpolation=None)
        config.read(BASE)
        values = {
            "base": {"tag": f"strategy_discovery_{name}", "jsonl_log": portable(run_root / "logs/production.jsonl")},
            "env": {"deck_pool_path": portable(pool_path), "draft_cross_gate_replay_prob": changes.get("replay", 0.15), "pbrs_gamma": 0.99},
            "train": {"total_timesteps": rows, "seed": 42, "seed_process_rngs": "true", "data_dir": portable(run_root / "artifacts"), "gamma": changes.get("gamma", 0.99)},
            "league": {"state_path": portable(run_root / "league/league_state.json"), "promotion_state_path": portable(run_root / "league/league_state_promotion.json"), "promotion_records_dir": portable(run_root / "league/promotion_records"), "opponent_dir": portable(run_root / "league/opponents")},
            "process_env": {"azk_ppo_diagnostics": 1, "azk_draft_episode_credit_exclude_xgate": 1, "azk_draft_episode_credit_coef": 1.0, "azk_draft_episode_credit_final_coef": changes.get("credit", 1.0), "azk_draft_episode_credit_anneal_start_rows": 0, "azk_draft_episode_credit_anneal_end_rows": rows // 2 if "credit" in changes else 0, "azk_potential_anneal_end_rows": rows, "azk_exploration_anneal_end_rows": rows, "azk_draft_ref_seat_prob": 0.2 if "exposure" in changes else 0, "azk_draft_ref_learner_fixed": int("exposure" in changes), "azk_draft_ref_opponent_only": 0, "azk_draft_ref_deck_indices": ",".join(map(str, pool["summary"][changes["exposure"] + "_deck_indices"])) if "exposure" in changes else "", "azk_draft_prefix_lengths": "0,4", "azk_draft_prefix_probs": "0.8,0.2" if "prefix" in changes else "", "azk_draft_prefix_seed": 420053, "azk_draft_prefix_pool_path": portable(prefix_path) if changes.get("prefix") == "strategic" else ""},
        }
        for section, section_values in values.items():
            for key, value in section_values.items():
                config.set(section, key, str(value))
        config.set("logging", "metric_patterns", config["logging"]["metric_patterns"] + ",environment/draft_prefix/*,environment/strategic_exposure/*,league/strategic_exposure/*")
        buffer = io.StringIO()
        config.write(buffer)
        content = buffer.getvalue().encode()
        artifacts[config_path] = content
        arms.append({"id": arm_id, "name": name, "config": portable(config_path), "config_sha256": hashlib.sha256(content).hexdigest(), "run_root": portable(run_root), "total_timesteps": rows, "evaluation_updates": [325, 650, 975] if rows == SCREEN_ROWS else [325, 975, 1950, 2925], "total_updates": rows // ROWS_PER_UPDATE, "seed": 42, "initialization": "fresh_matched_seed42_no_checkpoint_resume", "production_qualified": False, "treatment": changes, "process_env": {k.upper(): v for k, v in config["process_env"].items()}})
        evaluation_config_path = output / "configs" / f"{name}_evaluation.ini"
        config.set("env", "deck_pool_path", portable(SOURCE))
        config.set("env", "draft_cross_gate_replay_prob", "0")
        for key, value in {
            "azk_draft_ref_seat_prob": "0", "azk_draft_ref_learner_fixed": "0",
            "azk_draft_ref_opponent_only": "0", "azk_draft_ref_deck_indices": "",
            "azk_draft_prefix_probs": "", "azk_draft_prefix_pool_path": "",
        }.items():
            config.set("process_env", key, value)
        evaluation_buffer = io.StringIO()
        config.write(evaluation_buffer)
        evaluation_content = evaluation_buffer.getvalue().encode()
        artifacts[evaluation_config_path] = evaluation_content
        arms[-1].update(
            tag=f"strategy_discovery_{name}", result_root=portable(run_root),
            evaluation_config=portable(evaluation_config_path),
            evaluation_config_sha256=hashlib.sha256(evaluation_content).hexdigest(),
        )
    expected = set(SOURCE_FILES) | {portable(BASE), portable(SOURCE)}
    for pattern in SOURCE_GLOBS:
        matches = [p for p in ROOT.glob(pattern) if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"Required source glob has no matches: {pattern}")
        expected.update(portable(p) for p in matches)
    for pattern in BINARY_GLOBS:
        matches = [p for p in ROOT.glob(pattern) if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"Required runtime binary missing: {pattern}; build current source before registration")
        expected.update(portable(p) for p in matches)
    fingerprints = {p: sha256(ROOT / p) for p in sorted(expected)}
    artifact_fingerprints = {portable(p): hashlib.sha256(data).hexdigest() for p, data in artifacts.items()}
    fingerprints.update({portable(p): hashlib.sha256(data).hexdigest() for p, data in artifacts.items()})
    registration = {"schema_id": "azuki.ablation_registration", "schema_version": 1, "purpose": "current_source_strategy_discovery_diagnostic", "scope": "diagnostic_only", "production_qualified": False, "seed": 42, "arms": arms,
        "parent": {"config": portable(BASE), "sha256": sha256(BASE), "status": "R14_unqualified_diagnostic_control_not_production_winner"},
        "history": [{"path": p, "sha256": sha256(ROOT / p), "status": status} for p, status in HISTORY.items()],
        "expected_source_files": sorted(fingerprints), "expected_source_globs": list(SOURCE_GLOBS), "expected_binary_globs": list(BINARY_GLOBS), "source_sha256": fingerprints,
        "pool": {"path": portable(pool_path), "sha256": fingerprints[portable(pool_path)], **pool["summary"]}, "prefix_pool": {"path": portable(prefix_path), "sha256": fingerprints[portable(prefix_path)]},
        "comparison_groups": [{"id": "exposure_15m", "arms": ["CONTROL", "ENTITY", "STRATEGIC"], "total_timesteps": SCREEN_ROWS}, {"id": "prefix_15m", "arms": ["PREFIX_RANDOM", "PREFIX_STRATEGIC"], "total_timesteps": SCREEN_ROWS}, {"id": "replay_15m", "arms": ["CONTROL", "REPLAY_OFF"], "total_timesteps": SCREEN_ROWS}, {"id": "learner_discount_15m", "arms": ["CONTROL", "GAMMA1"], "total_timesteps": SCREEN_ROWS}, {"id": "credit_confirmation_45m", "arms": ["D2", "D3"], "total_timesteps": CONFIRM_ROWS}],
        "comparison_contract": {"pool_15m_and_45m": False, "primary": "elemental/gate/leader legal opportunities and realized effects; strategic acquisition", "supporting_only": "early winrate", "potential_and_exploration_anneals": "0 to configured horizon; original R14 scales unchanged; common within every matched group", "learner_gamma_limitation": "GAMMA1 changes learner discount only; PBRS remains gamma=.99. Discount mismatch breaks the usual shared-discount policy-invariance premise; this is not a clean PBRS-invariance test.", "exposure": "0.2 episode-level lottery; learner assigned supplied seat. NOT 20 percent of all player rows. Report actual supplied learner battle rows and denominator.", "prefix": "Both arms lengths0,4 probabilities.8,.2 seed420053 throughout screen; forced actor rows excluded; evaluation disabled", "excluded_future_work": ["S1", "S3", "league_ablation", "1B_run"], "historical_action_head_fix": "Unproven where sampler/policy hashes absent; distinct from old evaluator text-cache defect."},
        "evaluation_process_env": {"AZK_DRAFT_REF_SEAT_PROB": "0", "AZK_DRAFT_REF_LEARNER_FIXED": "0", "AZK_DRAFT_REF_OPPONENT_ONLY": "0", "AZK_DRAFT_PREFIX_PROBS": "", "AZK_DRAFT_PREFIX_POOL_PATH": ""},
        "runner_contract": {"entrypoint": "python/src/train.py --config <arm.config>", "working_directory": "repository root", "pythonpath": "build/python/src:python/src", "launch": "serial_only_diagnostic_runner", "fresh_initialization": True, "source_guard": "Verify every source_sha256 entry and exact glob membership before every launch; reject changes, missing binaries, existing run roots, or production scope.", "evaluation": "Out-of-process at each arm.evaluation_updates; prefixes/exposure disabled; never pool horizons."}}
    registration.update(family="strategy_discovery_v1", status="registered", artifact_sha256=artifact_fingerprints)
    registration["runtime"] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_executable_sha256": sha256(Path(sys.executable)),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes_authoritative_for_working_tree": True,
    }
    artifacts[output / "registration.json"] = json_bytes(registration)
    output.mkdir(parents=True, exist_ok=False)
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
        print(portable(path))
    return registration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
