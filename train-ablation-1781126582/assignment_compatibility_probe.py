#!/usr/bin/env python3
"""Stage 0 recurrent compatibility probe for assigned draft contexts.

The same frozen checkpoint is run through the legacy forced-leader row and the
new prefilled/no-row lifecycle. Main-pick histories are replayed in both
directions so KL and hidden-state differences isolate the missing recurrent
transition rather than divergent sampled decks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from probe_gate_kl import EpisodeRunner, GATE_CODE_PAIRS, OPPONENT_GATE, kl, tv


MAIN_QUARTILES = ((1, 13), (14, 25), (26, 38), (39, 50))


def _leaders_by_element(runner: EpisodeRunner) -> dict[str, tuple[str, ...]]:
    records = runner.catalog.records_by_def_id
    return {
        element: tuple(records[int(def_id)].card_code for def_id in def_ids)
        for element, def_ids in runner.catalog.leader_def_ids_by_element.items()
    }


def _deck_summary(main_ids: list[int]) -> dict[str, float | int]:
    counts = Counter(int(card_id) for card_id in main_ids if int(card_id) >= 0)
    total = sum(counts.values())
    return {
        "main_count": int(total),
        "unique_cards": int(len(counts)),
        "singletons": int(sum(value == 1 for value in counts.values())),
        "pairs": int(sum(value == 2 for value in counts.values())),
        "triplets": int(sum(value == 3 for value in counts.values())),
        "quads": int(sum(value == 4 for value in counts.values())),
        "max_copies": int(max(counts.values(), default=0)),
    }


def _hidden_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    denom = max(left_norm * right_norm, 1e-12)
    cosine = float(np.dot(left, right) / denom)
    return float(np.linalg.norm(left - right)), cosine, right_norm - left_norm


def _assert_main_replay(
    old_candidates: list[np.ndarray],
    prefilled_candidates: list[np.ndarray],
) -> None:
    if len(old_candidates) != 51 or len(prefilled_candidates) != 50:
        raise RuntimeError(
            "Lifecycle row mismatch: expected forced-row=51 and prefilled=50, "
            f"got {len(old_candidates)} and {len(prefilled_candidates)}"
        )
    for pick, (old, prefilled) in enumerate(
        zip(old_candidates[1:], prefilled_candidates), start=1
    ):
        if not np.array_equal(old, prefilled):
            raise RuntimeError(f"Main candidate replay diverged at pick {pick}")


def _summarize_context(rows: list[dict]) -> dict[str, object]:
    forward = np.asarray([row["forward_kl"] for row in rows], dtype=np.float64)
    reverse = np.asarray([row["reverse_kl"] for row in rows], dtype=np.float64)
    forward_tv = np.asarray([row["forward_tv"] for row in rows], dtype=np.float64)
    reverse_tv = np.asarray([row["reverse_tv"] for row in rows], dtype=np.float64)
    control = np.asarray([row["control_kl"] for row in rows], dtype=np.float64)
    hidden_l2 = np.asarray([row["hidden_l2"] for row in rows], dtype=np.float64)
    hidden_cos = np.asarray([row["hidden_cosine"] for row in rows], dtype=np.float64)
    hidden_norm_delta = np.asarray(
        [row["hidden_norm_delta"] for row in rows], dtype=np.float64
    )
    symmetric = 0.5 * (forward + reverse)
    symmetric_tv = 0.5 * (forward_tv + reverse_tv)
    by_quartile: dict[str, dict[str, object]] = {}
    for index, (low, high) in enumerate(MAIN_QUARTILES, start=1):
        section = symmetric[:, low - 1 : high]
        by_quartile[str(index)] = {
            "pick_range": [low, high],
            "mean_symmetric_kl": float(section.mean()),
            "p90_symmetric_kl": float(np.quantile(section, 0.90)),
            "mean_hidden_l2": float(hidden_l2[:, low - 1 : high].mean()),
            "mean_hidden_cosine": float(hidden_cos[:, low - 1 : high].mean()),
        }
    return {
        "episodes": len(rows),
        "main_pick_steps": int(symmetric.size),
        "forward_mean_kl": float(forward.mean()),
        "reverse_mean_kl": float(reverse.mean()),
        "mean_symmetric_kl": float(symmetric.mean()),
        "p90_symmetric_kl": float(np.quantile(symmetric, 0.90)),
        "max_symmetric_kl": float(symmetric.max()),
        "mean_symmetric_tv": float(symmetric_tv.mean()),
        "control_mean_kl": float(control.mean()),
        "mean_hidden_l2": float(hidden_l2.mean()),
        "mean_hidden_cosine": float(hidden_cos.mean()),
        "mean_hidden_norm_delta": float(hidden_norm_delta.mean()),
        "by_quartile": by_quartile,
        "forced_row_decks": [row["forced_row_deck"] for row in rows],
        "prefilled_decks": [row["prefilled_deck"] for row in rows],
    }


def _markdown(payload: dict[str, object]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# Stage 0 Assignment Compatibility",
        "",
        f"Checkpoint: `{payload['checkpoint']}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Contexts | {aggregate['contexts']} |",
        f"| Episodes/context | {payload['episodes_per_context']} |",
        f"| Symmetric main-pick KL | {aggregate['mean_symmetric_kl']:.6f} |",
        f"| P90 symmetric KL | {aggregate['p90_symmetric_kl']:.6f} |",
        f"| Symmetric TV | {aggregate['mean_symmetric_tv']:.6f} |",
        f"| Determinism control KL | {aggregate['control_mean_kl']:.3e} |",
        f"| Hidden cosine | {aggregate['mean_hidden_cosine']:.6f} |",
        f"| Hidden L2 | {aggregate['mean_hidden_l2']:.6f} |",
        f"| Probe decisions/sec | {payload['performance']['policy_decisions_per_second']:.1f} |",
        "",
        "## Contexts",
        "",
        "| Element | Gate | Leader | Sym KL | TV | Hidden cosine |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for context in payload["contexts"]:
        lines.append(
            f"| {context['element']} | {context['gate']} | {context['leader']} | "
            f"{context['mean_symmetric_kl']:.6f} | "
            f"{context['mean_symmetric_tv']:.6f} | "
            f"{context['mean_hidden_cosine']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Learned Leader Frequencies",
            "",
            "The policy-selected arm is diagnostic only; assigned-leader arms above cover all contexts.",
            "",
            "| Gate | Leader | Picks |",
            "| --- | --- | ---: |",
        ]
    )
    for gate, counts in payload["policy_leader_counts"].items():
        for leader, count in sorted(counts.items()):
            lines.append(f"| {gate} | {leader} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("python/config/azuki_deckbuild_3090.ini"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--leader-frequency-episodes", type=int, default=24)
    parser.add_argument(
        "--elements",
        default="",
        help="Optional comma-separated element subset for sharded execution",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.leader_frequency_episodes < 1:
        raise ValueError("Episode counts must be positive")

    from policy.v2 import tcg_sampler

    forced = EpisodeRunner(
        args.config, args.checkpoint, args.device, uniform_assignment=False
    )
    prefilled = EpisodeRunner(
        args.config, args.checkpoint, args.device, uniform_assignment=True
    )
    tcg_sampler.set_sampling_params(subaction_temperature=1.0, smoothing_eps=0.0)
    leaders = _leaders_by_element(forced)
    requested_elements = {
        value.strip().upper() for value in args.elements.split(",") if value.strip()
    }
    unknown_elements = requested_elements - set(GATE_CODE_PAIRS)
    if unknown_elements:
        raise ValueError(f"Unknown --elements values: {sorted(unknown_elements)}")
    selected_gate_pairs = {
        element: gate_pair
        for element, gate_pair in GATE_CODE_PAIRS.items()
        if not requested_elements or element in requested_elements
    }
    opponent_element = forced.catalog.records_by_code[OPPONENT_GATE].element
    opponent_leader = leaders[opponent_element][0]
    records_by_id = forced.catalog.records_by_def_id
    context_payloads: list[dict[str, object]] = []
    decision_count = 0
    started = time.perf_counter()
    try:
        for element_index, (element, gate_pair) in enumerate(selected_gate_pairs.items()):
            for gate_index, gate in enumerate(gate_pair):
                for leader_index, leader in enumerate(leaders[element]):
                    rows: list[dict[str, object]] = []
                    for episode in range(args.episodes):
                        seed = (
                            7_100_003
                            + 200_003 * element_index
                            + 40_009 * gate_index
                            + 10_007 * leader_index
                            + 7_919 * episode
                        )
                        actions_old, probs_old, candidates_old = forced.run_episode(
                            seed,
                            gate,
                            None,
                            p0_leader_code=leader,
                            p1_leader_code=opponent_leader,
                        )
                        old_hidden = list(forced.last_pick_hidden)
                        old_deck = list(forced.base_env._states[0].main_card_def_ids)

                        _, probs_prefilled, candidates_prefilled = prefilled.run_episode(
                            seed,
                            gate,
                            actions_old[2:],
                            p0_leader_code=leader,
                            p1_leader_code=opponent_leader,
                        )
                        prefilled_hidden = list(prefilled.last_pick_hidden)
                        prefilled_deck = list(
                            prefilled.base_env._states[0].main_card_def_ids
                        )
                        _, probs_control, candidates_control = forced.run_episode(
                            seed,
                            gate,
                            actions_old,
                            p0_leader_code=leader,
                            p1_leader_code=opponent_leader,
                        )
                        _assert_main_replay(candidates_old, candidates_prefilled)
                        _assert_main_replay(candidates_old, candidates_control[1:])

                        actions_pre_free, probs_pre_free, candidates_pre_free = (
                            prefilled.run_episode(
                                seed,
                                gate,
                                None,
                                p0_leader_code=leader,
                                p1_leader_code=opponent_leader,
                            )
                        )
                        pre_free_deck = list(
                            prefilled.base_env._states[0].main_card_def_ids
                        )
                        _, probs_old_replay, candidates_old_replay = forced.run_episode(
                            seed,
                            gate,
                            actions_old[:2] + actions_pre_free,
                            p0_leader_code=leader,
                            p1_leader_code=opponent_leader,
                        )
                        old_replay_deck = list(
                            forced.base_env._states[0].main_card_def_ids
                        )
                        _assert_main_replay(candidates_old_replay, candidates_pre_free)

                        hidden = [
                            _hidden_metrics(left, right)
                            for left, right in zip(old_hidden[1:], prefilled_hidden)
                        ]
                        rows.append(
                            {
                                "seed": seed,
                                "forward_kl": [
                                    kl(left, right)
                                    for left, right in zip(
                                        probs_old[1:], probs_prefilled
                                    )
                                ],
                                "reverse_kl": [
                                    kl(left, right)
                                    for left, right in zip(
                                        probs_pre_free, probs_old_replay[1:]
                                    )
                                ],
                                "forward_tv": [
                                    tv(left, right)
                                    for left, right in zip(
                                        probs_old[1:], probs_prefilled
                                    )
                                ],
                                "reverse_tv": [
                                    tv(left, right)
                                    for left, right in zip(
                                        probs_pre_free, probs_old_replay[1:]
                                    )
                                ],
                                "control_kl": [
                                    kl(left, right)
                                    for left, right in zip(
                                        probs_old[1:], probs_control[1:]
                                    )
                                ],
                                "hidden_l2": [item[0] for item in hidden],
                                "hidden_cosine": [item[1] for item in hidden],
                                "hidden_norm_delta": [item[2] for item in hidden],
                                "forced_row_deck": _deck_summary(old_deck),
                                "prefilled_deck": _deck_summary(prefilled_deck),
                                "reverse_prefilled_deck": _deck_summary(pre_free_deck),
                                "reverse_forced_row_deck": _deck_summary(
                                    old_replay_deck
                                ),
                                "forward_replay_decks_identical": (
                                    old_deck == prefilled_deck
                                ),
                                "reverse_replay_decks_identical": (
                                    pre_free_deck == old_replay_deck
                                ),
                            }
                        )
                        decision_count += 51 + 50 + 51 + 50 + 51
                    summary = _summarize_context(rows)
                    context_payloads.append(
                        {
                            "element": element,
                            "gate": gate,
                            "leader": leader,
                            **summary,
                        }
                    )
                    print(
                        f"[{element}/{gate}/{leader}] sym_KL="
                        f"{summary['mean_symmetric_kl']:.6f} hidden_cos="
                        f"{summary['mean_hidden_cosine']:.6f}",
                        flush=True,
                    )

        policy_leader_counts: dict[str, dict[str, int]] = {}
        for element, gate_pair in selected_gate_pairs.items():
            for gate in gate_pair:
                counts: Counter[str] = Counter()
                for episode in range(args.leader_frequency_episodes):
                    seed = 9_100_003 + 50_021 * len(policy_leader_counts) + 7_919 * episode
                    forced.run_episode(
                        seed,
                        gate,
                        None,
                        p1_leader_code=opponent_leader,
                    )
                    selected = int(forced.base_env._states[0].leader_card_def_id)
                    counts[records_by_id[selected].card_code] += 1
                    decision_count += 51
                policy_leader_counts[gate] = dict(sorted(counts.items()))
    finally:
        forced.vecenv.close()
        prefilled.vecenv.close()

    elapsed = time.perf_counter() - started
    aggregate = {
        "contexts": len(context_payloads),
        "mean_symmetric_kl": float(
            np.mean([item["mean_symmetric_kl"] for item in context_payloads])
        ),
        "p90_symmetric_kl": float(
            np.mean([item["p90_symmetric_kl"] for item in context_payloads])
        ),
        "mean_symmetric_tv": float(
            np.mean([item["mean_symmetric_tv"] for item in context_payloads])
        ),
        "control_mean_kl": float(
            np.mean([item["control_mean_kl"] for item in context_payloads])
        ),
        "mean_hidden_l2": float(
            np.mean([item["mean_hidden_l2"] for item in context_payloads])
        ),
        "mean_hidden_cosine": float(
            np.mean([item["mean_hidden_cosine"] for item in context_payloads])
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "episodes_per_context": args.episodes,
        "leader_frequency_episodes_per_gate": args.leader_frequency_episodes,
        "elements": list(selected_gate_pairs),
        "forced_row_policy_rows": 51,
        "prefilled_policy_rows": 50,
        "replayed_main_history_fixed": True,
        "contexts": context_payloads,
        "aggregate": aggregate,
        "policy_leader_counts": policy_leader_counts,
        "performance": {
            "wall_time_seconds": elapsed,
            "policy_decisions": decision_count,
            "policy_decisions_per_second": decision_count / max(elapsed, 1e-9),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(_markdown(payload), encoding="utf-8")
    print(f"wrote {args.json} and {args.md}")


if __name__ == "__main__":
    main()
