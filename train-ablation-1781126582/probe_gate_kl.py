#!/usr/bin/env python3
"""Interventional gate-conditioning probe.

Does the policy's DRAFT distribution actually condition on which gate card it
was assigned, beyond the element (candidate pool)?

Method: within an element, gates A and B share the identical candidate pool.
Episode 1 (record): force P0 gate=A, opponent gate fixed; run the policy and
record every action. Episode 2 (replay): force P0 gate=B, same env seed, step
with the RECORDED actions. The observation streams are then identical except
the gate-identity feature (including through LSTM history), so per-pick-step
KL(P_A || P_B) measures exactly the policy's causal sensitivity to the gate
card. A same-gate replay (A vs A) is run as a determinism control (KL must
be ~0). Reports mean/max KL and TV per element pair plus the cards whose pick
probability shifts most.

Usage:
  probe_gate_kl.py --checkpoint CKPT [--episodes 8] [--device cpu]
                   [--config python/config/azuki_deckbuild_3090.ini]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import azk_puffer.vector as azk_vector
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config, _unwrap_base_env
from train import _load_model_weights
from training_utils import build_policy, build_vecenv, install_tcg_sampler, load_training_config

GATE_CODE_PAIRS = {
    "LIGHTNING": ("STT01-002", "AZK01-120"),
    "WATER": ("STT02-002", "AZK01-126"),
    "FIRE": ("AZK01-122", "STT04-002"),
    "EARTH": ("AZK01-124", "STT03-002"),
}
OPPONENT_GATE = "STT02-002"  # fixed for every arm so P1 draft/battle is constant


def masked_probs(dist, row: int) -> np.ndarray:
    logits = dist.legal_action_logits[row]
    count = int(dist.legal_action_count[row])
    count = max(count, 1)
    probs = torch.softmax(logits[:count], dim=-1)
    return probs.detach().float().cpu().numpy()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    n = min(len(p), len(q))
    p, q = p[:n] + 1e-9, q[:n] + 1e-9
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def tv(p: np.ndarray, q: np.ndarray) -> float:
    n = min(len(p), len(q))
    return float(0.5 * np.abs(p[:n] - q[:n]).sum())


class EpisodeRunner:
    def __init__(
        self,
        config_path: Path,
        checkpoint: Path | None,
        device: str,
        *,
        uniform_assignment: bool = False,
    ):
        trainer_args = load_training_config(config_path, [])
        trainer_args["train"]["device"] = device
        trainer_args.setdefault("env", {})["deck_building_enabled"] = True
        _apply_checkpoint_resume_policy_config(trainer_args, checkpoint)
        trainer_args["env"]["native"] = False
        trainer_args["env"].pop("native_envs_per_instance", None)
        trainer_args["env"]["draft_uniform_assignment"] = bool(uniform_assignment)
        # A privileged-critic checkpoint expects the drafted-deck lists filled;
        # probing it on sanitized obs would mismeasure the critic.
        if trainer_args.get("policy", {}).get("privileged_critic_enabled"):
            trainer_args["env"]["deck_building_privileged_decks"] = True
        install_tcg_sampler()
        self.device = device
        self.uniform_assignment = bool(uniform_assignment)
        self.vecenv = build_vecenv(trainer_args, backend=azk_vector.Serial, num_envs=1, seed=7)
        base = _unwrap_base_env(self.vecenv.envs[0])
        while not getattr(type(base), "is_deck_building_wrapper", False) and hasattr(base, "env"):
            base = base.env
        self.base_env = base
        self.catalog = base._catalog
        self.policy = build_policy(self.vecenv, trainer_args)
        self.use_rnn = bool(trainer_args["train"].get("use_rnn", True))
        # warm forward so lazy modules exist before weight load
        self.vecenv.async_reset(seed=7)
        obs, _, _, _, _, _, masks = self.vecenv.recv()
        state = self._fresh_state(masks)
        with torch.no_grad():
            self.policy.forward_eval(torch.as_tensor(obs, device=device), state)
        if checkpoint is not None:
            _load_model_weights(self.policy, checkpoint, device=device, strict=False)
        self.policy.eval()
        self.code_to_def = {r.card_code: d for d, r in self.catalog.records_by_def_id.items()}
        self.last_pick_hidden: list[np.ndarray] = []

    def _fresh_state(self, masks):
        state = {"mask": torch.as_tensor(masks, device=self.device)}
        if self.use_rnn:
            state["lstm_h"] = torch.zeros(self.vecenv.num_agents, self.policy.hidden_size, device=self.device)
            state["lstm_c"] = torch.zeros(self.vecenv.num_agents, self.policy.hidden_size, device=self.device)
        return state

    def force_gates(self, p0_code: str, p1_code: str):
        forced = [self.code_to_def[p0_code], self.code_to_def[p1_code]]
        calls = {"n": 0}

        def fake_sample():
            value = forced[calls["n"] % 2]
            calls["n"] += 1
            return int(value)

        self.base_env._sample_gate_def_id = fake_sample

    def force_assigned_leaders(self, p0_code: str, p1_code: str) -> None:
        forced = [self.code_to_def[p0_code], self.code_to_def[p1_code]]
        calls = {"n": 0}

        def fake_sample(_gate_def_id: int) -> int:
            value = forced[calls["n"] % 2]
            calls["n"] += 1
            return int(value)

        self.base_env._sample_assigned_leader_def_id = fake_sample

    def run_episode(
        self,
        seed: int,
        p0_gate: str,
        forced_actions: list | None,
        *,
        p0_leader_code: str | None = None,
        p1_leader_code: str | None = None,
    ):
        """Returns (actions, probs_per_pick_step, candidate_ids_per_step)."""
        self.force_gates(p0_gate, OPPONENT_GATE)
        if self.uniform_assignment:
            if p0_leader_code is None or p1_leader_code is None:
                raise ValueError(
                    "Uniform-assignment probes require explicit leaders for both seats"
                )
            self.force_assigned_leaders(p0_leader_code, p1_leader_code)
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
        probs_log = []
        cands_log = []
        self.last_pick_hidden = []
        step = 0
        while self.base_env._building:
            active = self.base_env._active_player_index
            record_pick = active == 0
            obs_tensor = torch.as_tensor(obs, device=self.device)
            step_state = {"mask": torch.as_tensor(masks, device=self.device)}
            if self.use_rnn:
                step_state["lstm_h"] = state["lstm_h"]
                step_state["lstm_c"] = state["lstm_c"]
            with torch.no_grad():
                logits, _ = self.policy.forward_eval(obs_tensor, step_state)
            if self.use_rnn:
                state["lstm_h"] = step_state["lstm_h"]
                state["lstm_c"] = step_state["lstm_c"]
            if record_pick:
                probs_log.append(masked_probs(logits, 0))
                ctx = self.base_env._deck_context_for_player(0, include_candidates=True)
                cands_log.append(np.array(ctx["candidate_card_def_ids"][: ctx["candidate_count"]]))
                if self.use_rnn:
                    self.last_pick_hidden.append(
                        step_state["lstm_h"][0].detach().float().cpu().numpy().copy()
                    )
            if forced_actions is None:
                import azk_puffer.pytorch as azk_pytorch
                acts, _, _ = azk_pytorch.sample_logits(logits)
                acts = acts.cpu().numpy().astype(np.int32, copy=True)
            else:
                acts = np.array(forced_actions[step], dtype=np.int32, copy=True)
            desired_leader_code = (
                p0_leader_code if active == 0 else p1_leader_code
            )
            if (
                desired_leader_code is not None
                and self.base_env._states[active].leader_card_def_id < 0
            ):
                desired_def_id = int(self.code_to_def[desired_leader_code])
                ctx = self.base_env._deck_context_for_player(
                    active, include_candidates=True
                )
                candidates = np.array(
                    ctx["candidate_card_def_ids"][: ctx["candidate_count"]]
                )
                matches = np.flatnonzero(candidates == desired_def_id)
                if matches.size != 1:
                    raise RuntimeError(
                        f"Leader {desired_leader_code} is not uniquely available for "
                        f"player {active}"
                    )
                acts[active, 0] = 3
                acts[active, 1] = int(matches[0])
                acts[active, 2:] = 0
            actions_log.append(np.array(acts, copy=True))
            self.vecenv.send(acts)
            obs, _, _, _, _, _, masks = self.vecenv.recv()
            step += 1
            if step > 400:
                raise RuntimeError("draft did not complete in 400 steps")
        return actions_log, probs_log, cands_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("python/config/azuki_deckbuild_3090.ini"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    runner = EpisodeRunner(args.config, args.checkpoint, args.device)
    meta = {d: r for d, r in runner.catalog.records_by_def_id.items()}
    results = {}
    for element, (gate_a, gate_b) in GATE_CODE_PAIRS.items():
        kls, tvs, ctrl_kls = [], [], []
        shift_by_card = defaultdict(list)
        for ep in range(args.episodes):
            seed = 31337 + 7919 * ep
            actions, probs_a, cands_a = runner.run_episode(seed, gate_a, None)
            _, probs_b, _ = runner.run_episode(seed, gate_b, actions)
            _, probs_c, _ = runner.run_episode(seed, gate_a, actions)  # control
            for pa, pb, pc, cand in zip(probs_a, probs_b, probs_c, cands_a):
                kls.append(kl(pa, pb))
                tvs.append(tv(pa, pb))
                ctrl_kls.append(kl(pa, pc))
                n = min(len(pa), len(pb), len(cand))
                for i in range(n):
                    shift_by_card[int(cand[i])].append(float(pa[i] - pb[i]))
        top_shifts = sorted(
            ((np.mean(v), d) for d, v in shift_by_card.items() if len(v) >= args.episodes),
            key=lambda t: -abs(t[0]),
        )[:8]
        results[element] = {
            "gate_a": gate_a,
            "gate_b": gate_b,
            "mean_kl": float(np.mean(kls)),
            "p90_kl": float(np.quantile(kls, 0.9)),
            "mean_tv": float(np.mean(tvs)),
            "control_mean_kl": float(np.mean(ctrl_kls)),
            "pick_steps": len(kls),
            "top_prob_shifts": [
                {
                    "card": meta[d].card_code,
                    "name": getattr(meta[d], "name", "?") if hasattr(meta[d], "name") else "?",
                    "mean_dprob_a_minus_b": round(float(s), 5),
                }
                for s, d in top_shifts
            ],
        }
        r = results[element]
        print(
            f"[{element}] {gate_a} vs {gate_b}: mean_KL={r['mean_kl']:.5f} p90={r['p90_kl']:.5f} "
            f"mean_TV={r['mean_tv']:.4f} control_KL={r['control_mean_kl']:.2e} steps={r['pick_steps']}"
        )
        for t in r["top_prob_shifts"][:5]:
            print(f"    shift {t['mean_dprob_a_minus_b']:+.4f} {t['card']}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
