#!/usr/bin/env python3
"""Play logged self-play games for a checkpoint with randomized gates.

Both seats are played by the same checkpoint under eval sampling
(temperature 1.0, no smoothing). Gates are randomized per episode by the
env's own draft sampler (sibling-matchup oversampling disabled). For every
battle step we log the acting player, phase, the raw 4-head action, a
semantic decode against the actor's observation struct (card codes for
plays/attacks/portals/abilities/selections), and a compact board snapshot
(leader HP, garden/alley contents, hand of the actor, IKZ, discard counts).
One JSON object per game is appended to --out (JSONL).

Usage:
  OMP_NUM_THREADS=6 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/play_selfplay_games.py \
    --checkpoint CKPT --games 200 --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from action import ActionType
from analyze_decks import GATE_NAMES
from probe_gate_kl import EpisodeRunner

PHASE_NAMES = {
    0: "MULLIGAN",
    1: "START_OF_TURN",
    2: "MAIN",
    3: "RESPONSE",
    4: "COMBAT_RESOLVE",
    5: "END_TURN_ACTION",
    6: "END_TURN",
    7: "END_MATCH",
}

GARDEN_SIZE = 5


class GameLogger:
    def __init__(self, runner: EpisodeRunner):
        self.runner = runner
        self.records = runner.catalog.records_by_def_id
        # inner battle env (exposes _raw_observation / ctypes obs struct)
        self.inner = runner.base_env.env

    def code(self, def_id) -> str:
        rec = self.records.get(int(def_id))
        return rec.card_code if rec else f"?{int(def_id)}"

    # ---- observation snapshot helpers -------------------------------------

    @staticmethod
    def _valid(entries, count=None):
        out = []
        for i, e in enumerate(entries):
            if count is not None and i >= count:
                break
            if int(e.card_def_id) >= 0:
                out.append(e)
        return out

    def _board_list(self, slots):
        out = []
        for e in self._valid(slots):
            atk = int(e.cur_stats.cur_atk) if e.has_cur_stats else -1
            hp = int(e.cur_stats.cur_hp) if e.has_cur_stats else -1
            s = f"{self.code(e.card_def_id)}@{int(e.zone_index)}:{atk}/{hp}"
            if e.tap_state.tapped:
                s += "T"
            if e.has_defender:
                s += "D"
            if int(e.weapon_count) > 0:
                s += "+" + ",".join(
                    self.code(w.card_def_id) for w in e.weapons[: int(e.weapon_count)]
                )
            out.append(s)
        return out

    def _slot_map(self, slots):
        return {int(e.zone_index): self.code(e.card_def_id) for e in self._valid(slots)}

    def snapshot(self, actor: int) -> dict:
        raw = self.inner._raw_observation(actor)
        me = raw.my_observation_data
        opp = raw.opponent_observation_data
        hand = [self.code(h.card_def_id) for h in self._valid(me.hand, int(me.hand_count))]
        ikz_untapped = sum(
            1 for e in self._valid(me.ikz_area) if not e.tap_state.tapped
        )
        ikz_total = len(self._valid(me.ikz_area))
        snap = {
            "phase": PHASE_NAMES.get(int(raw.phase), str(int(raw.phase))),
            "hand": hand,
            "my_garden": self._board_list(me.garden),
            "my_alley": self._board_list(me.alley),
            "opp_garden": self._board_list(opp.garden),
            "opp_alley": self._board_list(opp.alley),
            "my_hp": int(me.leader.cur_stats.cur_hp),
            "opp_hp": int(opp.leader.cur_stats.cur_hp),
            "my_leader_weapons": [
                self.code(w.card_def_id)
                for w in me.leader.weapons[: int(me.leader.weapon_count)]
            ],
            "opp_leader_weapons": [
                self.code(w.card_def_id)
                for w in opp.leader.weapons[: int(opp.leader.weapon_count)]
            ],
            "ikz": [ikz_untapped, ikz_total],
            "ikz_token": bool(me.has_ikz_token),
            "my_discard_n": len(self._valid(me.discard)),
            "opp_discard_n": len(self._valid(opp.discard)),
            "my_deck_n": int(me.deck_count),
            "gate_tapped": bool(me.gate.tap_state.tapped),
        }
        ac = raw.ability_context
        if bool(ac.has_source_card_def_id):
            snap["ability_src"] = self.code(ac.source_card_def_id)
            snap["ability_phase"] = int(ac.phase)
        # selection zone uses stable slot indices with holes after picks
        # (ctx->selection.cards[i] = 0), so keep positions; holes become None
        sel_n = int(me.selection_count)
        if any(int(me.selection[i].card_def_id) >= 0 for i in range(sel_n)):
            snap["selection"] = [
                self.code(me.selection[i].card_def_id)
                if int(me.selection[i].card_def_id) >= 0
                else None
                for i in range(sel_n)
            ]
        cc = raw.combat_context
        if bool(cc.combat_active):
            snap["combat"] = {
                "attacker": "LEADER" if cc.attacker_is_leader else self.code(cc.attacker_card_def_id),
                "target": "LEADER" if cc.target_is_leader else self.code(cc.target_card_def_id),
                "attacker_is_self": bool(cc.attacker_is_self),
                "response_open": bool(cc.response_window_active),
                "intercepted": bool(cc.defender_intercepted),
            }
        return snap

    # ---- action decoding ---------------------------------------------------

    def decode(self, act, snap) -> dict:
        t, s1, s2, s3 = (int(v) for v in act)
        try:
            name = ActionType(t).name
        except ValueError:
            name = f"ACT_{t}"
        d = {"t": name}
        hand = snap["hand"]
        my_g = self._parse_slot_map(snap["my_garden"])
        my_a = self._parse_slot_map(snap["my_alley"])
        opp_g = self._parse_slot_map(snap["opp_garden"])
        opp_a = self._parse_slot_map(snap["opp_alley"])

        def hand_at(i):
            return hand[i] if 0 <= i < len(hand) else f"hand[{i}]?"

        if t in (ActionType.PLAY_ENTITY_TO_GARDEN, ActionType.PLAY_ENTITY_TO_ALLEY):
            d["card"] = hand_at(s1)
            d["slot"] = s2
        elif t == ActionType.PLAY_SPELL_FROM_HAND:
            d["card"] = hand_at(s1)
        elif t == ActionType.ATTACH_WEAPON_FROM_HAND:
            d["card"] = hand_at(s1)
            d["target"] = "MY_LEADER" if s2 == GARDEN_SIZE else my_g.get(s2, f"g{s2}?")
        elif t == ActionType.ATTACK:
            d["attacker"] = "MY_LEADER" if s1 == GARDEN_SIZE else my_g.get(s1, f"g{s1}?")
            if s2 == GARDEN_SIZE:
                d["target"] = "OPP_LEADER"
            elif s2 > GARDEN_SIZE:
                d["target"] = "ALLEY:" + opp_a.get(s2 - GARDEN_SIZE - 1, f"a{s2 - GARDEN_SIZE - 1}?")
            else:
                d["target"] = opp_g.get(s2, f"g{s2}?")
        elif t == ActionType.GATE_PORTAL:
            d["card"] = my_a.get(s1, f"a{s1}?")
            d["slot"] = s2
        elif t == ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY:
            d["card"] = "MY_LEADER" if s1 == GARDEN_SIZE else my_g.get(s1, f"g{s1}?")
            d["ability"] = s2
        elif t == ActionType.ACTIVATE_ALLEY_ABILITY:
            # schema order: (ability_index, alley_index)
            d["card"] = my_a.get(s2, f"a{s2}?")
            d["ability"] = s1
        elif t == ActionType.DECLARE_DEFENDER:
            d["card"] = my_g.get(s1, f"g{s1}?")
        elif t == ActionType.SELECT_FROM_SELECTION:
            sel = snap.get("selection", [])
            d["card"] = sel[s1] if 0 <= s1 < len(sel) and sel[s1] else f"sel[{s1}]?"
        elif t in (
            ActionType.SELECT_TO_ALLEY,
            ActionType.SELECT_TO_GARDEN,
            ActionType.SELECT_TO_EQUIP,
            ActionType.BOTTOM_DECK_CARD,
            ActionType.TOP_DECK_CARD,
        ):
            sel = snap.get("selection", [])
            d["card"] = sel[s1] if 0 <= s1 < len(sel) and sel[s1] else f"sel[{s1}]?"
            d["arg"] = s2
        elif t in (ActionType.SELECT_COST_TARGET, ActionType.SELECT_EFFECT_TARGET):
            d["raw"] = [s1, s2, s3]
        if "ability_src" in snap and t not in (ActionType.NOOP,):
            d["src"] = snap["ability_src"]
        return d

    @staticmethod
    def _parse_slot_map(board_list):
        out = {}
        for s in board_list:
            head = s.split(":", 1)[0]
            code_part, slot = head.split("@")
            out[int(slot)] = code_part
        return out

    # ---- game loop ----------------------------------------------------------

    def deck_record(self):
        decks = []
        for i in range(2):
            st = self.runner.base_env._states[i]
            decks.append(
                {
                    "gate": self.code(st.gate_card_def_id),
                    "gate_name": GATE_NAMES.get(self.code(st.gate_card_def_id), "?"),
                    "leader": self.code(st.leader_card_def_id),
                    "main": [
                        self.code(st.main_card_def_ids[k]) for k in range(st.main_count)
                    ],
                }
            )
        return decks

    def play_game(self, game_index: int, seed: int, step_cap: int = 4000) -> dict:
        runner = self.runner
        torch.manual_seed(seed)
        runner.vecenv.async_reset(seed=seed)
        obs, rew, term, trunc, info, env_id, masks = runner.vecenv.recv()
        state = {}
        if runner.use_rnn:
            state = {
                "lstm_h": torch.zeros(
                    runner.vecenv.num_agents, runner.policy.hidden_size, device=runner.device
                ),
                "lstm_c": torch.zeros(
                    runner.vecenv.num_agents, runner.policy.hidden_size, device=runner.device
                ),
            }
        steps = []
        decks = None
        draft_steps = 0
        last_snap = None
        outcome = None
        for _ in range(step_cap):
            building = runner.base_env._building
            active = int(runner.base_env._active_player_index)
            if not building and decks is None:
                decks = self.deck_record()
            snap = None
            if not building:
                snap = self.snapshot(active)
                last_snap = (active, snap)
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
            import azk_puffer.pytorch as azk_pytorch

            acts, _, _ = azk_pytorch.sample_logits(logits)
            acts = acts.cpu().numpy().astype(np.int32, copy=True)
            if building:
                draft_steps += 1
            else:
                a = acts[active].tolist()
                rec = {
                    "i": len(steps),
                    "p": active,
                    "ph": snap["phase"],
                    "a": a,
                    "d": self.decode(a, snap),
                    "hp": [
                        snap["my_hp"] if active == 0 else snap["opp_hp"],
                        snap["my_hp"] if active == 1 else snap["opp_hp"],
                    ],
                    "ikz": snap["ikz"],
                    "hand": snap["hand"],
                    "my_garden": snap["my_garden"],
                    "my_alley": snap["my_alley"],
                    "opp_garden": snap["opp_garden"],
                    "opp_alley": snap["opp_alley"],
                    "dc": [
                        snap["my_discard_n"] if active == 0 else snap["opp_discard_n"],
                        snap["my_discard_n"] if active == 1 else snap["opp_discard_n"],
                    ],
                    "gate_tapped": snap["gate_tapped"],
                }
                if snap.get("ikz_token"):
                    rec["ikz_token"] = True
                for k in ("ability_src", "selection", "combat", "my_leader_weapons", "opp_leader_weapons"):
                    if snap.get(k):
                        rec[k] = snap[k]
                steps.append(rec)
            runner.vecenv.send(acts)
            obs, rew, term, trunc, info, env_id, masks = runner.vecenv.recv()
            rew = np.asarray(rew, dtype=np.float64).reshape(-1)
            term = np.asarray(term).reshape(-1)
            trunc = np.asarray(trunc).reshape(-1)
            if steps:
                r0, r1 = float(rew[0]), float(rew[1])
                if r0 != 0.0 or r1 != 0.0:
                    steps[-1]["r"] = [round(r0, 4), round(r1, 4)]
            if bool(term.any()) or bool(trunc.any()):
                r0, r1 = float(rew[0]), float(rew[1])
                winner = -1
                if r0 > r1:
                    winner = 0
                elif r1 > r0:
                    winner = 1
                outcome = {
                    "winner": winner,
                    "terminal_rewards": [round(r0, 4), round(r1, 4)],
                    "terminated": bool(term.any()),
                    "truncated": bool(trunc.any()),
                }
                break
        if outcome is None:
            outcome = {"winner": -1, "terminal_rewards": [0.0, 0.0], "terminated": False,
                       "truncated": True, "step_cap_hit": True}
        final_hp = None
        if last_snap is not None:
            active, snap = last_snap
            final_hp = [
                snap["my_hp"] if active == 0 else snap["opp_hp"],
                snap["my_hp"] if active == 1 else snap["opp_hp"],
            ]
        return {
            "game": game_index,
            "seed": seed,
            "decks": decks,
            "draft_steps": draft_steps,
            "battle_steps": len(steps),
            "outcome": outcome,
            "final_hp_before_last_action": final_hp,
            "steps": steps,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=550_000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    # natural per-episode gate randomization; disable sibling-matchup skew
    runner.base_env._same_element_matchup_prob = 0.0

    from policy.v2 import tcg_sampler

    tcg_sampler.set_sampling_params(subaction_temperature=1.0, smoothing_eps=0.0)

    logger = GameLogger(runner)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wins = [0, 0, 0]
    with args.out.open("w") as fh:
        for g in range(args.games):
            game = logger.play_game(g, args.seed0 + 7919 * g)
            fh.write(json.dumps(game, separators=(",", ":")) + "\n")
            fh.flush()
            w = game["outcome"]["winner"]
            wins[w if w >= 0 else 2] += 1
            if (g + 1) % 10 == 0 or g == 0:
                dt = time.time() - t0
                print(
                    f"[{g + 1}/{args.games}] {dt:.1f}s ({dt / (g + 1):.1f}s/game) "
                    f"p0={wins[0]} p1={wins[1]} draws={wins[2]} "
                    f"steps={game['battle_steps']} gates="
                    f"{game['decks'][0]['gate_name']}v{game['decks'][1]['gate_name']}",
                    flush=True,
                )
    print(f"wrote {args.out} ({args.games} games, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
