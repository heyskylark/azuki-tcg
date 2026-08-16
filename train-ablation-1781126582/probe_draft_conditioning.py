#!/usr/bin/env python3
"""Causal main-pick sensitivity to sibling gates and selected leaders."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from probe_gate_kl import EpisodeRunner, GATE_CODE_PAIRS, kl, tv


MAIN_QUARTILES = ((1, 13), (14, 25), (26, 38), (39, 50))


def _validate_replay(
    left_candidates: list[np.ndarray],
    right_candidates: list[np.ndarray],
) -> None:
    if len(left_candidates) != 51 or len(right_candidates) != 51:
        raise RuntimeError(
            f"Expected leader plus 50 main rows, got "
            f"{len(left_candidates)} and {len(right_candidates)}"
        )
    for pick, (left, right) in enumerate(
        zip(left_candidates[1:], right_candidates[1:]), start=1
    ):
        if not np.array_equal(left, right):
            raise RuntimeError(f"Replay candidate pool diverged at main pick {pick}")


def _metric_summary(
    forward: list[list[float]],
    reverse: list[list[float]],
    forward_tv: list[list[float]],
    reverse_tv: list[list[float]],
    controls: list[list[float]],
) -> dict[str, object]:
    forward_array = np.asarray(forward, dtype=np.float64)
    reverse_array = np.asarray(reverse, dtype=np.float64)
    forward_tv_array = np.asarray(forward_tv, dtype=np.float64)
    reverse_tv_array = np.asarray(reverse_tv, dtype=np.float64)
    control_array = np.asarray(controls, dtype=np.float64)
    if forward_array.ndim != 2 or forward_array.shape[1] != 50:
        raise RuntimeError(f"Expected [episodes, 50] KL data, got {forward_array.shape}")
    symmetric = 0.5 * (forward_array + reverse_array)
    symmetric_tv = 0.5 * (forward_tv_array + reverse_tv_array)
    by_quartile = {}
    for index, (low, high) in enumerate(MAIN_QUARTILES, start=1):
        values = symmetric[:, low - 1:high]
        by_quartile[str(index)] = {
            "pick_range": [low, high],
            "mean_symmetric_kl": float(values.mean()),
            "p90_symmetric_kl": float(np.quantile(values, 0.90)),
        }
    return {
        "episodes": int(forward_array.shape[0]),
        "main_pick_steps": int(forward_array.size),
        "forward_mean_kl": float(forward_array.mean()),
        "reverse_mean_kl": float(reverse_array.mean()),
        "mean_symmetric_kl": float(symmetric.mean()),
        "p90_symmetric_kl": float(np.quantile(symmetric, 0.90)),
        "max_symmetric_kl": float(symmetric.max()),
        "mean_symmetric_tv": float(symmetric_tv.mean()),
        "control_mean_kl": float(control_array.mean()),
        "control_max_kl": float(control_array.max()),
        "by_quartile": by_quartile,
    }


def _compare_contexts(
    runner: EpisodeRunner,
    *,
    episodes: int,
    seed_base: int,
    gate_a: str,
    gate_b: str,
    leader_a: str | None = None,
    leader_b: str | None = None,
) -> dict[str, object]:
    forward: list[list[float]] = []
    reverse: list[list[float]] = []
    forward_tv: list[list[float]] = []
    reverse_tv: list[list[float]] = []
    controls: list[list[float]] = []
    for episode in range(episodes):
        seed = seed_base + 7_919 * episode
        actions_a, probs_a, candidates_a = runner.run_episode(
            seed,
            gate_a,
            None,
            p0_leader_code=leader_a,
        )
        _, probs_b_replay, candidates_b_replay = runner.run_episode(
            seed,
            gate_b,
            actions_a,
            p0_leader_code=leader_b,
        )
        _, probs_a_control, candidates_a_control = runner.run_episode(
            seed,
            gate_a,
            actions_a,
            p0_leader_code=leader_a,
        )
        _validate_replay(candidates_a, candidates_b_replay)
        _validate_replay(candidates_a, candidates_a_control)
        forward.append([kl(left, right) for left, right in zip(probs_a[1:], probs_b_replay[1:])])
        forward_tv.append(
            [tv(left, right) for left, right in zip(probs_a[1:], probs_b_replay[1:])]
        )
        controls.append(
            [kl(left, right) for left, right in zip(probs_a[1:], probs_a_control[1:])]
        )

        actions_b, probs_b, candidates_b = runner.run_episode(
            seed,
            gate_b,
            None,
            p0_leader_code=leader_b,
        )
        _, probs_a_replay, candidates_a_replay = runner.run_episode(
            seed,
            gate_a,
            actions_b,
            p0_leader_code=leader_a,
        )
        _validate_replay(candidates_b, candidates_a_replay)
        reverse.append([kl(left, right) for left, right in zip(probs_b[1:], probs_a_replay[1:])])
        reverse_tv.append(
            [tv(left, right) for left, right in zip(probs_b[1:], probs_a_replay[1:])]
        )
    return _metric_summary(forward, reverse, forward_tv, reverse_tv, controls)


def _leaders_by_element(runner: EpisodeRunner) -> dict[str, tuple[str, ...]]:
    records = runner.catalog.records_by_def_id
    return {
        element: tuple(sorted(records[int(def_id)].card_code for def_id in def_ids))
        for element, def_ids in runner.catalog.leader_def_ids_by_element.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("python/config/azuki_deckbuild_3090.ini"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    from policy.v2 import tcg_sampler

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    tcg_sampler.set_sampling_params(subaction_temperature=1.0, smoothing_eps=0.0)
    leaders = _leaders_by_element(runner)
    sibling_gate = {}
    leader_conditioned = {}
    try:
        for element_index, (element, (gate_a, gate_b)) in enumerate(
            GATE_CODE_PAIRS.items()
        ):
            metrics = _compare_contexts(
                runner,
                episodes=args.episodes,
                seed_base=4_100_001 + 100_003 * element_index,
                gate_a=gate_a,
                gate_b=gate_b,
            )
            sibling_gate[element] = {
                "gate_a": gate_a,
                "gate_b": gate_b,
                **metrics,
            }
            print(
                f"[gate/{element}] main symmetric KL="
                f"{metrics['mean_symmetric_kl']:.6f} "
                f"control={metrics['control_mean_kl']:.2e}",
                flush=True,
            )

        for gate_index, (element, gate_pair) in enumerate(GATE_CODE_PAIRS.items()):
            element_leaders = leaders.get(element, ())
            if len(element_leaders) < 2:
                raise RuntimeError(f"Element {element} has fewer than two leaders")
            for gate in gate_pair:
                pair_results = []
                for pair_index, (leader_a, leader_b) in enumerate(
                    combinations(element_leaders, 2)
                ):
                    metrics = _compare_contexts(
                        runner,
                        episodes=args.episodes,
                        seed_base=(
                            5_100_001
                            + 200_003 * gate_index
                            + 20_011 * pair_index
                            + (0 if gate == gate_pair[0] else 10_007)
                        ),
                        gate_a=gate,
                        gate_b=gate,
                        leader_a=leader_a,
                        leader_b=leader_b,
                    )
                    pair_results.append(
                        {
                            "leader_a": leader_a,
                            "leader_b": leader_b,
                            **metrics,
                        }
                    )
                leader_conditioned[gate] = {
                    "element": element,
                    "leader_pairs": pair_results,
                    "mean_symmetric_kl": float(
                        np.mean([item["mean_symmetric_kl"] for item in pair_results])
                    ),
                }
                print(
                    f"[leader/{gate}] main symmetric KL="
                    f"{leader_conditioned[gate]['mean_symmetric_kl']:.6f}",
                    flush=True,
                )
    finally:
        runner.vecenv.close()

    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "episodes_per_direction": args.episodes,
        "leader_row_excluded": True,
        "replayed_action_history_fixed": True,
        "sibling_gate": sibling_gate,
        "leader_conditioned": leader_conditioned,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
