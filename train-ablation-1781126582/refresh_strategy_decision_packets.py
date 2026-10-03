#!/usr/bin/env python3
"""Refresh strategy-derived fields and conclusions in retained decision packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strategy_descriptor import SCHEMA_VERSION
from build_terminal_safe_reward_decision import _descriptor_summary, superseded_decision
from regenerate_strategy_descriptors import REWARD_FAMILIES


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "train-ablation-1781126582"
RESULTS = CAMPAIGN / "results"
ELEMENTS = ("EARTH", "FIRE", "LIGHTNING", "WATER")
FAMILIES = ("reward_screens", "draft_screens", "curriculum_screens")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve(path: str) -> Path:
    direct = ROOT / path
    if direct.is_file():
        return direct
    nested = CAMPAIGN / path
    if nested.is_file():
        return nested
    raise FileNotFoundError(path)


def descriptor(path: str) -> dict[str, Any]:
    payload = read_json(resolve(path))
    if payload.get("schema_id") != "azuki.strategy_descriptor":
        raise ValueError(f"wrong descriptor schema: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"stale strategy descriptor: {path}")
    return payload


def completed_by_element(payload: dict[str, Any]) -> dict[str, int]:
    return {
        element: sum(
            int(sequence["completed"])
            for sequence in payload["sequences"].values()
            if sequence["element"] == element and sequence.get("valid", True)
        )
        for element in ELEMENTS
    }


def sequence_summary(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields = (
        "element",
        "eligible",
        "completed",
        "converted",
        "completion_per_eligible",
        "conversion_per_completed",
        "stages",
        "valid",
        "invalid_reason",
    )
    return {
        name: {field: sequence.get(field) for field in fields}
        for name, sequence in payload["sequences"].items()
    }


def sibling_mean(payload: dict[str, Any]) -> float | None:
    pairs = payload["deck"]["relationships"]["sibling_pairs"]
    if not pairs:
        return None
    return sum(float(pair["multiset_distance"]) for pair in pairs) / len(pairs)


def mode_summary(path: str, previous: dict[str, Any]) -> dict[str, Any]:
    payload = descriptor(path)
    result = dict(previous)
    result.update(
        {
            "descriptor": path,
            "descriptor_id": payload["descriptor_id"],
            "elemental_strategy": payload["elemental_strategy"],
            "games": int(payload["games"]),
            "sibling_exact_collision_pairs": int(
                payload["deck"]["relationships"]["sibling_exact_collision_pairs"]
            ),
            "sibling_mean_multiset_distance": sibling_mean(payload),
            "face_share_when_both_legal": payload["funnels"]
            ["attack.both_legal_face_choice"]["selected_per_opportunity"],
            "completed_sequences_by_element": completed_by_element(payload),
            "sequences": sequence_summary(payload),
        }
    )
    return result


def endpoint_status(modes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mode, summary in modes.items():
        counts: dict[str, dict[str, int]] = {}
        unavailable: list[str] = []
        no_opportunity: list[str] = []
        stalled_with_opportunity: list[str] = []
        for element in ELEMENTS:
            sequences = [
                sequence
                for sequence in summary["sequences"].values()
                if sequence["element"] == element and sequence.get("valid", True)
            ]
            if not sequences:
                unavailable.append(element)
                continue
            eligible = sum(int(sequence["eligible"]) for sequence in sequences)
            completed = sum(int(sequence["completed"]) for sequence in sequences)
            converted = sum(int(sequence["converted"]) for sequence in sequences)
            counts[element] = {
                "eligible": eligible,
                "completed": completed,
                "converted": converted,
            }
            if eligible == 0:
                no_opportunity.append(element)
            elif completed == 0:
                stalled_with_opportunity.append(element)
        result[mode] = {
            "by_element": counts,
            "unavailable": unavailable,
            "no_opportunity": no_opportunity,
            "stalled_with_opportunity": stalled_with_opportunity,
        }
    return result


def refresh_window_families() -> None:
    for family in FAMILIES:
        packet_path = RESULTS / family / "decision_packet.json"
        index = read_json(RESULTS / family / "evaluation_index.json")
        packet = read_json(packet_path)
        indexed_arms = {str(arm["id"]): arm for arm in index["arms"]}
        for arm_id, arm_payload in packet["arms"].items():
            arm_index = indexed_arms[arm_id]
            indexed_windows = {
                int(window["update"]): window for window in arm_index["windows"]
            }
            for window in arm_payload["windows"]:
                window_index = indexed_windows[int(window["update"])]
                for mode in ("sample", "argmax"):
                    path = str(window_index[f"{mode}_descriptor"])
                    window["modes"][mode] = mode_summary(
                        path, window["modes"][mode]
                    )
            arm_payload.pop("endpoint_structural_absence", None)
            arm_payload["endpoint_sequence_status"] = endpoint_status(
                arm_payload["windows"][-1]["modes"]
            )
        packet["schema_version"] = 2
        packet["strategy_descriptor_schema_version"] = SCHEMA_VERSION
        write_packet(packet_path, packet)


def refresh_endpoint(
    target: dict[str, Any], sample_path: str, argmax_path: str
) -> None:
    sample = descriptor(sample_path)
    argmax = descriptor(argmax_path)
    for legacy_key in (
        "sample_completed_sequences_by_element",
        "argmax_completed_sequences_by_element",
        "sample_face_share_when_both_legal",
        "argmax_face_share_when_both_legal",
        "sample_sibling_mean_multiset_distance",
        "argmax_sibling_mean_multiset_distance",
        "endpoint_entity_share_when_both_legal",
    ):
        target.pop(legacy_key, None)
    target.update(
        {
            "endpoint_sample_descriptor": sample_path,
            "endpoint_sample_descriptor_id": sample["descriptor_id"],
            "endpoint_argmax_descriptor": argmax_path,
            "endpoint_argmax_descriptor_id": argmax["descriptor_id"],
            "endpoint_sample_sequences": completed_by_element(sample),
            "endpoint_argmax_sequences": completed_by_element(argmax),
            "endpoint_sample_face_share_when_both_legal": sample["funnels"]
            ["attack.both_legal_face_choice"]["selected_per_opportunity"],
            "endpoint_argmax_face_share_when_both_legal": argmax["funnels"]
            ["attack.both_legal_face_choice"]["selected_per_opportunity"],
            "endpoint_sample_entity_share_when_both_legal": sample["funnels"]
            ["attack.both_legal_entity_choice"]["selected_per_opportunity"],
            "endpoint_argmax_entity_share_when_both_legal": argmax["funnels"]
            ["attack.both_legal_entity_choice"]["selected_per_opportunity"],
            "endpoint_sample_sibling_mean_multiset_distance": sibling_mean(sample),
            "endpoint_argmax_sibling_mean_multiset_distance": sibling_mean(argmax),
        }
    )


def endpoint_paths(index: dict[str, Any], arm_id: str) -> tuple[str, str]:
    arm = next(arm for arm in index["arms"] if arm["id"] == arm_id)
    endpoint = arm["windows"][-1]
    return str(endpoint["sample_descriptor"]), str(endpoint["argmax_descriptor"])


def refresh_inherited_endpoints() -> None:
    reward_index = read_json(RESULTS / "reward_screens/evaluation_index.json")
    draft_index = read_json(RESULTS / "draft_screens/evaluation_index.json")
    draft_path = RESULTS / "draft_screens/decision_packet.json"
    curriculum_path = RESULTS / "curriculum_screens/decision_packet.json"
    draft_packet = read_json(draft_path)
    curriculum_packet = read_json(curriculum_path)

    refresh_endpoint(
        draft_packet["control"]["endpoint"], *endpoint_paths(reward_index, "R1")
    )
    refresh_endpoint(
        curriculum_packet["parent"]["endpoint"], *endpoint_paths(draft_index, "D2")
    )
    write_packet(draft_path, draft_packet)
    write_packet(curriculum_path, curriculum_packet)


def refresh_shaping() -> None:
    packet_path = RESULTS / "shaping_screens/decision_packet.json"
    packet = read_json(packet_path)
    index = read_json(RESULTS / "shaping_screens/evaluation_index.json")
    for arm_id, arm in packet["arms"].items():
        refresh_endpoint(arm, *endpoint_paths(index, arm_id))
    packet["schema_version"] = 2
    packet["strategy_descriptor_schema_version"] = SCHEMA_VERSION
    write_packet(packet_path, packet)


def refresh_terminal() -> None:
    packet_path = RESULTS / "terminal_closure_screen/decision_packet.json"
    packet = read_json(packet_path)
    shaping_index = read_json(RESULTS / "shaping_screens/evaluation_index.json")
    terminal_index = read_json(RESULTS / "terminal_closure_screen/evaluation_index.json")
    refresh_endpoint(packet["control"], *endpoint_paths(shaping_index, "S0"))
    refresh_endpoint(packet["treatment"], *endpoint_paths(terminal_index, "S4"))
    packet["schema_version"] = 2
    packet["strategy_descriptor_schema_version"] = SCHEMA_VERSION
    write_packet(packet_path, packet)


def refresh_conclusions() -> None:
    # Old prose encoded numeric v2 results and selected winners unconditionally.
    # Rebuilt observations cannot validate those historical causal conclusions.
    for family in (*FAMILIES, "shaping_screens", "terminal_closure_screen"):
        path = RESULTS / family / "decision_packet.json"
        packet = read_json(path)
        decision = packet["decision"]
        for key in (
            "strategy_first_retained", "longer_confirmation_candidates",
            "interaction_candidates", "selected_arm", "selected_closure_screen_control",
        ):
            decision.pop(key, None)
        decision.update(
            production_qualified=False,
            canary_eligible=False,
            status="superseded_strategy_semantics_require_requalification",
            reason=[
                f"Embedded strategy observations have been recomputed at descriptor schema {SCHEMA_VERSION}.",
                "Historical v2 rankings are not valid decisions under corrected effect, recovery, and timing semantics.",
                "Use per-context elemental evidence and matched counterfactual probes; neither mechanic volume nor early win rate selects a successor.",
            ],
            next_action="Requalify the R14 foundation before downstream recipe comparisons; R3 removal is redundant with R14.",
        )
        if isinstance(decision.get("disposition"), dict):
            decision["disposition"] = {
                arm: "requires_strategy_requalification" for arm in decision["disposition"]
            }
        else:
            decision["disposition"] = "requires_strategy_requalification"
        write_packet(path, packet)


def refresh_terminal_safe() -> None:
    def refresh(value: object) -> None:
        if isinstance(value, list):
            for child in value:
                refresh(child)
        elif isinstance(value, dict):
            for child in list(value.values()):
                refresh(child)
            path = value.get("descriptor")
            if isinstance(path, str) and "descriptor_id" in value:
                value.update(_descriptor_summary(resolve(path)))

    for family in REWARD_FAMILIES:
        for path in sorted((RESULTS / family).glob("*decision_packet.json")):
            if path.name == "final_decision_packet.json":
                continue
            packet = read_json(path)
            # These cached aggregates lack their own descriptor provenance.
            # Retain refreshed windows, not inherited v2 semantic conclusions.
            removed = [
                key for key in packet
                if key == "derived" or key == "references"
                or key.endswith("_reference") or key.endswith("_references")
            ]
            for key in removed:
                del packet[key]
            refresh(packet)
            packet["strategy_descriptor_schema_version"] = SCHEMA_VERSION
            packet["semantic_refresh"] = {
                "removed_inherited_fields": removed,
                "reason": "Inherited derived summaries are omitted; current observations are in descriptor-backed windows.",
                "historical_archive": "train-ablation-1781126582/results/strategy_semantics_v3/prior_decision_manifest.json",
            }
            packet["decision"] = superseded_decision()
            write_packet(path, packet)

    # The final packet is a computed qualification, not a historical verdict.
    # Rebuild it only after its observation parents have been refreshed.
    from build_terminal_safe_reward_final_decision import main as build_final_decision

    build_final_decision()


def write_packet(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    refresh_window_families()
    refresh_inherited_endpoints()
    refresh_shaping()
    refresh_terminal()
    refresh_conclusions()
    refresh_terminal_safe()
    print(f"refreshed retained decision packets from schema-v{SCHEMA_VERSION} strategy descriptors")


if __name__ == "__main__":
    main()
