#!/usr/bin/env python3
"""Replay direct-placement choices and branch Garden versus Alley.

The extractor finds logged plays of Garden-conditional On Play entities where
the same hand card was legal in both destinations. The runner reconstructs the
draft and battle prefix exactly, forces one destination, and samples the frozen
policy to terminal with common rollout seeds. The report clusters uncertainty
by source game so repeated points from one game are not treated as independent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np

from action import ActionType
from analyze_opportunity_rates import GARDEN_CONDITIONAL_ON_PLAY
from analyze_selfplay_games import ELEMENT_OF_GATE, load_games
from probe_block_counterfactual import BranchRunner


NOOP = int(ActionType.NOOP)
PLAY_GARDEN = int(ActionType.PLAY_ENTITY_TO_GARDEN)
PLAY_ALLEY = int(ActionType.PLAY_ENTITY_TO_ALLEY)


def _legal_actions(step: dict) -> list[tuple[int, int, int, int]]:
    raw = step.get("legal")
    if not isinstance(raw, list):
        raise ValueError(
            "Input lacks legal-action traces; generate it with "
            "play_selfplay_games.py --log-legal-actions"
        )
    actions = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 4:
            raise ValueError(f"Malformed legal action: {entry!r}")
        actions.append(tuple(int(value) for value in entry))
    return actions


def extract_points(paths: list[Path], max_per_game: int, rng_seed: int) -> list[dict]:
    games = load_games(paths)
    per_game: list[list[dict]] = []
    for game_index, game in enumerate(games):
        candidates = []
        for step_index, step in enumerate(game["steps"]):
            selected = tuple(int(value) for value in step["a"])
            card = str(step.get("d", {}).get("card", ""))
            if selected[0] != PLAY_ALLEY or card not in GARDEN_CONDITIONAL_ON_PLAY:
                continue
            legal = _legal_actions(step)
            garden = [
                action
                for action in legal
                if action[0] == PLAY_GARDEN and action[1] == selected[1]
            ]
            if selected not in legal or not garden:
                continue
            if int(step["i"]) != step_index:
                raise ValueError(
                    f"Non-contiguous battle step index in game {game_index}: "
                    f"position={step_index}, logged={step['i']}"
                )
            player = int(step["p"])
            candidates.append(
                {
                    "game_index": game_index,
                    "seed": int(game["seed"]),
                    "branch_step": step_index,
                    "player": player,
                    "card": card,
                    "card_name": GARDEN_CONDITIONAL_ON_PLAY[card],
                    "element": ELEMENT_OF_GATE[str(game["decks"][player]["gate"])],
                    "hand_index": selected[1],
                    "alley_action": list(selected),
                    "garden_action": list(garden[0]),
                    "orig_winner": int(game["outcome"]["winner"]),
                    "hp_at_branch": list(step.get("hp", [])),
                    "decks": [
                        {
                            "gate": str(game["decks"][seat]["gate"]),
                            "leader": str(game["decks"][seat]["leader"]),
                            "main": [str(code) for code in game["decks"][seat]["main"]],
                        }
                        for seat in range(2)
                    ],
                    "prefix": [
                        [int(prefix_step["p"])]
                        + [int(value) for value in prefix_step["a"]]
                        for prefix_step in game["steps"][:step_index]
                    ],
                }
            )
        if candidates:
            per_game.append(candidates)

    rng = random.Random(rng_seed)
    points = []
    for candidates in per_game:
        rng.shuffle(candidates)
        if max_per_game > 0:
            candidates = candidates[:max_per_game]
        points.extend(candidates)
    points.sort(key=lambda point: (point["seed"], point["branch_step"], point["card"]))
    return points


class GardenBranchRunner(BranchRunner):
    def _verify_battle_setup(self, point: dict) -> str | None:
        for player, expected in enumerate(point["decks"]):
            raw = self.inner._raw_observation(player)
            actual_gate = self.code(raw.my_observation_data.gate.card_def_id)
            actual_leader = self.code(raw.my_observation_data.leader.card_def_id)
            if actual_gate != expected["gate"]:
                return f"gate mismatch for player {player}: {actual_gate} != {expected['gate']}"
            if actual_leader != expected["leader"]:
                return (
                    f"leader mismatch for player {player}: "
                    f"{actual_leader} != {expected['leader']}"
                )
        return None

    def rollout(self, point: dict, arm: str, rollout_seed: int, step_cap: int = 4000) -> dict:
        import torch
        import azk_puffer.pytorch as azk_pytorch

        if arm not in {"garden", "alley"}:
            raise ValueError(f"Unknown arm: {arm}")
        runner = self.runner
        torch.manual_seed(rollout_seed)
        runner.vecenv.async_reset(seed=point["seed"])
        obs, rew, term, trunc, info, env_id, masks = runner.vecenv.recv()
        del rew, term, trunc, info, env_id
        state = {}
        if runner.use_rnn:
            state = {
                "lstm_h": torch.zeros(
                    runner.vecenv.num_agents,
                    runner.policy.hidden_size,
                    device=runner.device,
                ),
                "lstm_c": torch.zeros(
                    runner.vecenv.num_agents,
                    runner.policy.hidden_size,
                    device=runner.device,
                ),
            }

        self._point_decks = point["decks"]
        cursors = [0, 0]
        prefix = point["prefix"]
        battle_step = -1
        battle_verified = False
        branched = False
        effect_seen = False
        effect_engaged = False
        post_branch_steps = 0

        for _ in range(step_cap):
            building = runner.base_env._building
            active = int(runner.base_env._active_player_index)
            if not building:
                battle_step += 1
                if not battle_verified:
                    setup_error = self._verify_battle_setup(point)
                    if setup_error is not None:
                        return {"error": setup_error}
                    battle_verified = True

            raw_active = self.inner._raw_observation(active)
            effect_active = False
            if branched and arm == "garden" and active == point["player"]:
                ability = raw_active.ability_context
                if bool(ability.has_source_card_def_id):
                    source = self.code(ability.source_card_def_id)
                    if source == point["card"]:
                        effect_seen = True
                        effect_active = True

            obs_tensor = torch.as_tensor(obs, device=runner.device)
            step_state = {"mask": torch.as_tensor(masks, device=runner.device)}
            if runner.use_rnn:
                step_state["lstm_h"] = state["lstm_h"]
                step_state["lstm_c"] = state["lstm_c"]
            with torch.no_grad():
                logits, _ = runner.policy.forward_eval(obs_tensor, step_state)
            if runner.use_rnn:
                state["lstm_h"] = step_state["lstm_h"]
                state["lstm_c"] = step_state["lstm_c"]

            acts = np.zeros((runner.vecenv.num_agents, 4), dtype=np.int32)
            if building:
                forced, error = self._draft_pick_action(cursors)
                if error is not None:
                    return {"error": error}
                acts[active] = forced
            elif battle_step < point["branch_step"]:
                logged = prefix[battle_step]
                if logged[0] != active:
                    return {
                        "error": (
                            f"prefix actor mismatch at battle step {battle_step}: "
                            f"{active} != {logged[0]}"
                        )
                    }
                acts[active] = np.asarray(logged[1:], dtype=np.int32)
            elif battle_step == point["branch_step"]:
                if active != point["player"]:
                    return {"error": f"actor mismatch: {active} != {point['player']}"}
                hand_index = int(point["hand_index"])
                hand = raw_active.my_observation_data.hand
                actual_card = self.code(hand[hand_index].card_def_id)
                if actual_card != point["card"]:
                    return {
                        "error": (
                            f"branch card mismatch: {actual_card} != {point['card']}"
                        )
                    }
                choice = tuple(int(value) for value in point[f"{arm}_action"])
                if choice not in self.legal_actions_for(active):
                    return {"error": f"forced {arm} action is no longer legal"}
                acts[active] = np.asarray(choice, dtype=np.int32)
                branched = True
            else:
                sampled, _, _ = azk_pytorch.sample_logits(logits)
                acts = sampled.cpu().numpy().astype(np.int32, copy=True)
                if effect_active and int(acts[active, 0]) != NOOP:
                    effect_engaged = True
                post_branch_steps += 1

            runner.vecenv.send(acts)
            obs, rewards, terminals, truncations, info, env_id, masks = runner.vecenv.recv()
            del info, env_id
            terminals = np.asarray(terminals).reshape(-1)
            truncations = np.asarray(truncations).reshape(-1)
            if bool(terminals.any()) or bool(truncations.any()):
                if not branched:
                    return {
                        "error": (
                            f"episode ended at battle step {battle_step} before "
                            f"branch {point['branch_step']}"
                        )
                    }
                rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
                winner = -1
                if rewards[0] > rewards[1]:
                    winner = 0
                elif rewards[1] > rewards[0]:
                    winner = 1
                score = 0.5 if winner < 0 else float(winner == point["player"])
                return {
                    "arm": arm,
                    "winner": winner,
                    "player_score": score,
                    "rollout_seed": rollout_seed,
                    "effect_seen": effect_seen,
                    "effect_engaged": effect_engaged,
                    "post_branch_steps": post_branch_steps,
                }
        return {"error": "step cap hit"}


def _cluster_interval(values_by_cluster: dict[int, list[float]], seed: int) -> list[float] | None:
    if len(values_by_cluster) < 2:
        return None
    keys = sorted(values_by_cluster)
    rng = np.random.default_rng(seed)
    draws = np.empty(20_000, dtype=np.float64)
    for index in range(draws.size):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        draws[index] = np.mean(
            [value for key in sampled for value in values_by_cluster[int(key)]]
        )
    return [
        float(np.quantile(draws, 0.10)),
        float(np.quantile(draws, 0.90)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _summarize_points(points: list[dict], bootstrap_seed: int) -> dict:
    if not points:
        return {
            "points": 0,
            "source_games": 0,
            "garden_score": None,
            "alley_score": None,
            "paired_delta": None,
            "positive_points": 0,
            "negative_points": 0,
            "neutral_points": 0,
        }
    values_by_seed: dict[int, list[float]] = defaultdict(list)
    for point in points:
        values_by_seed[int(point["seed"])].append(float(point["delta"]))
    interval = _cluster_interval(values_by_seed, bootstrap_seed)
    result = {
        "points": len(points),
        "source_games": len(values_by_seed),
        "garden_score": float(np.mean([point["garden_score"] for point in points])),
        "alley_score": float(np.mean([point["alley_score"] for point in points])),
        "paired_delta": float(np.mean([point["delta"] for point in points])),
        "positive_points": sum(point["delta"] > 0.05 for point in points),
        "negative_points": sum(point["delta"] < -0.05 for point in points),
        "neutral_points": sum(abs(point["delta"]) <= 0.05 for point in points),
    }
    if interval is not None:
        result["cluster_bootstrap_80"] = interval[:2]
        result["cluster_bootstrap_95"] = interval[2:]
    return result


def report(paths: list[Path], bootstrap_seed: int) -> dict:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    errors = [row for row in rows if "error" in row]
    ok = [row for row in rows if "error" not in row]

    grouped = defaultdict(lambda: {"garden": {}, "alley": {}})
    for row in ok:
        key = (int(row["seed"]), int(row["branch_step"]), str(row["card"]))
        grouped[key][str(row["arm"])][int(row["rollout_seed"])] = row

    points = []
    for (seed, branch_step, card), arms in sorted(grouped.items()):
        common = sorted(set(arms["garden"]) & set(arms["alley"]))
        if not common:
            continue
        garden_score = float(np.mean([arms["garden"][key]["player_score"] for key in common]))
        alley_score = float(np.mean([arms["alley"][key]["player_score"] for key in common]))
        points.append(
            {
                "seed": seed,
                "branch_step": branch_step,
                "card": card,
                "card_name": arms["garden"][common[0]]["card_name"],
                "paired_rollouts": len(common),
                "garden_score": garden_score,
                "alley_score": alley_score,
                "delta": garden_score - alley_score,
                "effect_seen_rate": float(
                    np.mean([arms["garden"][key]["effect_seen"] for key in common])
                ),
                "effect_engaged_rate": float(
                    np.mean([arms["garden"][key]["effect_engaged"] for key in common])
                ),
            }
        )

    by_card = {}
    for index, card in enumerate(sorted({point["card"] for point in points})):
        card_points = [point for point in points if point["card"] == card]
        summary = _summarize_points(card_points, bootstrap_seed + index + 1)
        summary["card_name"] = card_points[0]["card_name"]
        summary["effect_seen_rate"] = float(
            np.mean([point["effect_seen_rate"] for point in card_points])
        )
        summary["effect_engaged_rate"] = float(
            np.mean([point["effect_engaged_rate"] for point in card_points])
        )
        by_card[card] = summary

    payload = {
        "schema_version": 1,
        "rows": len(rows),
        "valid_rows": len(ok),
        "errors": len(errors),
        "error_kinds": dict(
            Counter(str(row["error"]).split(":", 1)[0] for row in errors)
        ),
        "overall": _summarize_points(points, bootstrap_seed),
        "by_card": by_card,
        "points": points,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("inputs", nargs="+", type=Path)
    extract.add_argument("--points", type=Path, required=True)
    extract.add_argument("--max-per-game", type=int, default=0)
    extract.add_argument("--seed", type=int, default=7319)

    run = subparsers.add_parser("run")
    run.add_argument("--points", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument(
        "--config",
        type=Path,
        default=Path("python/config/azuki_deckbuild_3090.ini"),
    )
    run.add_argument("--device", default="cpu")
    run.add_argument("--rollouts", type=int, default=10)
    run.add_argument("--slice", default="0/1")
    run.add_argument("--out", type=Path, required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("inputs", nargs="+", type=Path)
    report_parser.add_argument("--json", type=Path)
    report_parser.add_argument("--bootstrap-seed", type=int, default=9137)

    args = parser.parse_args()
    if args.command == "extract":
        if args.max_per_game < 0:
            raise ValueError("--max-per-game must be nonnegative")
        points = extract_points(args.inputs, args.max_per_game, args.seed)
        args.points.parent.mkdir(parents=True, exist_ok=True)
        args.points.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")
        counts = Counter(point["card"] for point in points)
        print(
            f"wrote {args.points}: points={len(points)} games="
            f"{len({point['seed'] for point in points})} by_card={dict(counts)}"
        )
        return

    if args.command == "report":
        payload = report(args.inputs, args.bootstrap_seed)
        print(json.dumps({key: payload[key] for key in ("rows", "valid_rows", "errors", "overall", "by_card")}, indent=2))
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    shard, shard_count = (int(value) for value in args.slice.split("/", 1))
    if shard_count <= 0 or not 0 <= shard < shard_count:
        raise ValueError("--slice must be K/N with 0 <= K < N")
    points = json.loads(args.points.read_text(encoding="utf-8"))
    selected = [point for index, point in enumerate(points) if index % shard_count == shard]
    runner = GardenBranchRunner(args.config, args.checkpoint, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    total = len(selected) * 2 * args.rollouts
    with args.out.open("w", encoding="utf-8") as handle:
        for point in selected:
            for arm in ("garden", "alley"):
                for rollout_index in range(args.rollouts):
                    rollout_seed = (
                        810_000_000
                        + point["seed"] * 131
                        + point["branch_step"] * 17
                        + rollout_index
                    ) % 2_000_000_000
                    result = runner.rollout(point, arm, rollout_seed)
                    result.update(
                        {
                            "seed": point["seed"],
                            "branch_step": point["branch_step"],
                            "player": point["player"],
                            "card": point["card"],
                            "card_name": point["card_name"],
                            "element": point["element"],
                        }
                    )
                    handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                    handle.flush()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"[{completed}/{total}]", flush=True)
                    if "error" in result:
                        break
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
