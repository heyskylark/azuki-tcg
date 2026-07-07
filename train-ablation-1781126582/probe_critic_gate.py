#!/usr/bin/env python3
"""Critic gate-sensitivity probe (go/no-go for long-run sibling-gate conditioning).

Question: does the VALUE FUNCTION distinguish sibling gates (same element,
different gate card) when the entire trajectory is held fixed except the gate
identity? If the critic is blind to the distinction, pick-step advantages
cannot carry a gate-conditional signal and longer training alone will not
create gate-conditional drafting (pivot to privileged critic / portal-EV).

Method: reuses the interventional gate-swap machinery from probe_gate_kl.py.
  ep-rec  : P0 gate=A, sample policy actions; record actions + V(s_t) at every
            P0 pick step and at the first battle observation.
  ep-swap : P0 gate=B, FORCE the recorded actions -> identical deck/board/RNG,
            observations differ only through gate identity (incl. LSTM history).
  ep-ctrl : replay with gate=A again -> numerical noise floor (must be ~0).

Readouts per (element, seed):
  - |dV| over P0 pick steps (rec vs swap), split early/mid/late draft
  - signed dV at battle start + sign consistency across seeds
  - win-prob aux head delta at battle start (sigmoid of logits)
  - control floor (rec vs ctrl) for both readouts
  - scale reference: cross-seed std of battle-start V (natural variation)

Usage:
  probe_critic_gate.py --checkpoint CKPT [--episodes 8] [--device cpu]
                       [--elements LIGHTNING,WATER] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_gate_kl import GATE_CODE_PAIRS, OPPONENT_GATE, EpisodeRunner


def _row0(t) -> float:
    return float(t.detach().flatten()[0].cpu())


class CriticEpisodeRunner(EpisodeRunner):
    def run_episode_values(self, seed: int, p0_gate: str, forced_actions: list | None):
        """Like run_episode but captures value/win-prob readouts for P0 (row 0)."""
        self.force_gates(p0_gate, OPPONENT_GATE)
        torch.manual_seed(seed)
        self.vecenv.async_reset(seed=seed)
        obs, _, _, _, _, _, masks = self.vecenv.recv()
        state = {}
        if self.use_rnn:
            state = {
                "lstm_h": torch.zeros(self.vecenv.num_agents, self.policy.hidden_size, device=self.device),
                "lstm_c": torch.zeros(self.vecenv.num_agents, self.policy.hidden_size, device=self.device),
            }
        actions_log = []
        v_picks: list[float] = []
        wp_picks: list[float | None] = []
        step = 0
        while self.base_env._building:
            active = self.base_env._active_player_index
            obs_tensor = torch.as_tensor(obs, device=self.device)
            step_state = {"mask": torch.as_tensor(masks, device=self.device)}
            if self.use_rnn:
                step_state["lstm_h"] = state["lstm_h"]
                step_state["lstm_c"] = state["lstm_c"]
            with torch.no_grad():
                logits, values = self.policy.forward_eval(obs_tensor, step_state)
            if self.use_rnn:
                state["lstm_h"] = step_state["lstm_h"]
                state["lstm_c"] = step_state["lstm_c"]
            if active == 0:
                v_picks.append(_row0(values))
                wp = step_state.get("_azk_win_prob_logits")
                wp_picks.append(float(torch.sigmoid(wp.detach().flatten()[0]).cpu()) if torch.is_tensor(wp) else None)
            if forced_actions is None:
                import azk_puffer.pytorch as azk_pytorch

                acts, _, _ = azk_pytorch.sample_logits(logits)
                acts = acts.cpu().numpy().astype(np.int32, copy=True)
            else:
                acts = forced_actions[step]
            actions_log.append(np.array(acts, copy=True))
            self.vecenv.send(acts)
            obs, _, _, _, _, _, masks = self.vecenv.recv()
            step += 1
            if step > 400:
                raise RuntimeError("draft did not complete in 400 steps")
        # battle-start readout: identical decks/board across rec/swap by construction
        obs_tensor = torch.as_tensor(obs, device=self.device)
        step_state = {"mask": torch.as_tensor(masks, device=self.device)}
        if self.use_rnn:
            step_state["lstm_h"] = state["lstm_h"]
            step_state["lstm_c"] = state["lstm_c"]
        with torch.no_grad():
            _, values = self.policy.forward_eval(obs_tensor, step_state)
        v_battle = _row0(values)
        wp = step_state.get("_azk_win_prob_logits")
        wp_battle = float(torch.sigmoid(wp.detach().flatten()[0]).cpu()) if torch.is_tensor(wp) else None
        return actions_log, {
            "v_picks": v_picks,
            "wp_picks": wp_picks,
            "v_battle": v_battle,
            "wp_battle": wp_battle,
        }


def _tertile_means(deltas_by_index: dict[int, list[float]]) -> dict[str, float]:
    if not deltas_by_index:
        return {}
    max_idx = max(deltas_by_index)
    bounds = [(0, max_idx // 3), (max_idx // 3 + 1, 2 * max_idx // 3), (2 * max_idx // 3 + 1, max_idx)]
    out = {}
    for name, (lo, hi) in zip(("early", "mid", "late"), bounds):
        vals = [v for i, vs in deltas_by_index.items() if lo <= i <= hi for v in vs]
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--elements", type=str, default="")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    pairs = GATE_CODE_PAIRS
    if args.elements:
        keep = {e.strip().upper() for e in args.elements.split(",") if e.strip()}
        pairs = {k: v for k, v in pairs.items() if k in keep}

    runner = CriticEpisodeRunner(args.config, args.checkpoint, args.device)
    results = {}
    for element, (gate_a, gate_b) in pairs.items():
        pick_dv, pick_dv_ctrl = [], []
        dv_by_index: dict[int, list[float]] = {}
        battle_dv_signed, battle_dv_ctrl = [], []
        battle_v_rec, battle_wp_rec = [], []
        battle_dwp_signed = []
        within_ep_pick_std = []
        for ep in range(args.episodes):
            seed = 31337 + 7919 * ep
            actions, rec = runner.run_episode_values(seed, gate_a, None)
            _, swap = runner.run_episode_values(seed, gate_b, actions)
            _, ctrl = runner.run_episode_values(seed, gate_a, actions)
            if len(rec["v_picks"]) > 1:
                within_ep_pick_std.append(float(np.std(rec["v_picks"])))
            n = min(len(rec["v_picks"]), len(swap["v_picks"]), len(ctrl["v_picks"]))
            for i in range(n):
                d = rec["v_picks"][i] - swap["v_picks"][i]
                pick_dv.append(abs(d))
                dv_by_index.setdefault(i, []).append(abs(d))
                pick_dv_ctrl.append(abs(rec["v_picks"][i] - ctrl["v_picks"][i]))
            battle_dv_signed.append(rec["v_battle"] - swap["v_battle"])
            battle_dv_ctrl.append(abs(rec["v_battle"] - ctrl["v_battle"]))
            battle_v_rec.append(rec["v_battle"])
            if rec["wp_battle"] is not None and swap["wp_battle"] is not None:
                battle_dwp_signed.append(rec["wp_battle"] - swap["wp_battle"])
                battle_wp_rec.append(rec["wp_battle"])
        signed = np.array(battle_dv_signed)
        scale = float(np.std(battle_v_rec)) if len(battle_v_rec) > 1 else float("nan")
        pick_v_within_std = float(np.mean(within_ep_pick_std)) if within_ep_pick_std else float("nan")
        res = {
            "gate_a": gate_a,
            "gate_b": gate_b,
            "episodes": args.episodes,
            "pick_mean_abs_dv": float(np.mean(pick_dv)),
            "pick_p90_abs_dv": float(np.quantile(pick_dv, 0.9)),
            "pick_ctrl_mean_abs_dv": float(np.mean(pick_dv_ctrl)),
            "pick_dv_tertiles": _tertile_means(dv_by_index),
            "battle_mean_abs_dv": float(np.mean(np.abs(signed))),
            "battle_mean_signed_dv": float(np.mean(signed)),
            "battle_sign_pos": int((signed > 0).sum()),
            "battle_ctrl_mean_abs_dv": float(np.mean(battle_dv_ctrl)),
            "battle_v_scale_std": scale,
            "battle_v_mean": float(np.mean(battle_v_rec)),
            "battle_v_values": [round(float(v), 6) for v in battle_v_rec],
            "battle_dv_signed_values": [round(float(v), 6) for v in battle_dv_signed],
            "pick_v_within_ep_std": pick_v_within_std,
            "battle_sens_ratio": float(np.mean(np.abs(signed)) / scale) if scale and scale > 0 else float("nan"),
            "battle_mean_abs_dwp": float(np.mean(np.abs(battle_dwp_signed))) if battle_dwp_signed else None,
            "battle_mean_signed_dwp": float(np.mean(battle_dwp_signed)) if battle_dwp_signed else None,
            "battle_wp_scale_std": float(np.std(battle_wp_rec)) if len(battle_wp_rec) > 1 else None,
        }
        results[element] = res
        print(
            f"[{element}] {gate_a} vs {gate_b}: pick|dV|={res['pick_mean_abs_dv']:.5f} "
            f"(ctrl {res['pick_ctrl_mean_abs_dv']:.2e}) battle|dV|={res['battle_mean_abs_dv']:.5f} "
            f"(ctrl {res['battle_ctrl_mean_abs_dv']:.2e}) signed={res['battle_mean_signed_dv']:+.5f} "
            f"sign+={res['battle_sign_pos']}/{args.episodes} scale_std={res['battle_v_scale_std']:.4f} "
            f"ratio={res['battle_sens_ratio']:.3f} Vmean={res['battle_v_mean']:+.4f} "
            f"withinEpStd={res['pick_v_within_ep_std']:.4f}"
        )
        if res["battle_mean_abs_dwp"] is not None:
            print(
                f"    win-prob head: |dWP|={res['battle_mean_abs_dwp']:.5f} "
                f"signed={res['battle_mean_signed_dwp']:+.5f} scale_std={res['battle_wp_scale_std']:.4f}"
            )
        print(f"    draft tertiles |dV| early/mid/late: {res['pick_dv_tertiles']}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
