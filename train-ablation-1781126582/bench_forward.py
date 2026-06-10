#!/usr/bin/env python3
"""Microbenchmark policy forward/backward cost on real observations.

Measures, on cuda:
  - rollout-style forward_eval (LSTMCell path) per 1440-row batch
  - train-style forward (nn.LSTM BPTT path) per minibatch
  - encode/LSTM/decode split (manual timing)
under float32 and bf16 autocast. Run with deck building on or off via flag.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from training_utils import build_policy, make_azuki_env


def collect_obs(deck_building: bool, rows: int):
    env = make_azuki_env(
        seed=9,
        deck_building_enabled=deck_building,
        direct_parallel=not deck_building,
        deck_pool_path=".codex/docs/azuki_tcg_decks_final.json",
    )
    obs_list = []
    obs, _ = env.reset(seed=21)
    rng = np.random.default_rng(0)
    inner = env.env.env if deck_building else env.env.env
    while len(obs_list) < rows:
        for agent_obs in obs.values() if isinstance(obs, dict) else [obs]:
            pass
        # env is the PettingZooPufferEnv; step with random legal actions
        base = inner
        action = base.random_legal_action(rng)
        actions = {a: action for a in base.possible_agents}
        step_out = base.step(actions)
        observations = step_out[0]
        terms, truncs = step_out[2], step_out[3]
        for agent, agent_obs in observations.items():
            obs_list.append(agent_obs)
        if any(terms.values()) or any(truncs.values()):
            base.reset()
    # Flatten through the puffer emulation to packed arrays
    flat = env.observation_space.sample()
    # Use the emulation env to pack: simpler to use env.reset/step outputs
    return env, obs_list[:rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-building", action="store_true")
    parser.add_argument("--rows", type=int, default=1440)
    parser.add_argument("--bptt", type=int, default=16)
    parser.add_argument("--minibatch", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import azk_puffer.vector as azk_vector

    from training_utils import build_vecenv, install_tcg_sampler, load_training_config
    from pathlib import Path

    install_tcg_sampler()
    trainer_args = load_training_config(
        Path("python/config/azuki_deckbuild_3090.ini"), []
    )
    trainer_args["train"]["device"] = args.device
    trainer_args["env"]["deck_building_enabled"] = bool(args.deck_building)
    if not args.deck_building:
        trainer_args["env"]["direct_parallel"] = True

    vecenv = build_vecenv(trainer_args, backend=azk_vector.Serial, num_envs=2, seed=11)
    vecenv.async_reset(seed=11)
    rng = np.random.default_rng(0)
    rows = []
    obs, *_ = vecenv.recv()
    base_envs = [e.env.env for e in vecenv.envs]
    while len(rows) < max(args.rows, args.minibatch):
        for row in np.asarray(obs):
            rows.append(np.array(row, copy=True))
        actions = []
        for base in base_envs:
            action = base.random_legal_action(rng)
            actions.extend([action, action])
        vecenv.send(np.stack(actions[: vecenv.num_agents]).astype(np.int32))
        obs, *_ = vecenv.recv()
    batch_np = np.stack(rows[: args.minibatch])

    policy = build_policy(vecenv, trainer_args)
    policy.eval()
    device = args.device

    eval_batch = torch.from_numpy(batch_np[: args.rows]).to(device)
    train_batch = torch.from_numpy(batch_np).to(device)

    def bench(label, fn, iters=args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        print(f"{label:44s} {dt*1e3:9.1f} ms")
        return dt

    state = {
        "lstm_h": torch.zeros(args.rows, policy.hidden_size, device=device),
        "lstm_c": torch.zeros(args.rows, policy.hidden_size, device=device),
    }

    with torch.no_grad():
        bench("forward_eval fp32 (rollout path)", lambda: policy.forward_eval(eval_batch, dict(state)))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bench("forward_eval bf16", lambda: policy.forward_eval(eval_batch, dict(state)))

    segments = args.minibatch // args.bptt
    train_state = {
        "lstm_h": torch.zeros(1, segments, policy.hidden_size, device=device),
        "lstm_c": torch.zeros(1, segments, policy.hidden_size, device=device),
    }
    train_view = train_batch.reshape(segments, args.bptt, *train_batch.shape[1:])

    def train_fwd():
        st = {k: v.clone() for k, v in train_state.items()}
        policy.forward(train_view, st)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bench("train forward bf16", train_fwd, iters=max(2, args.iters // 2))

    # Backward cost
    policy.train()

    def train_fwd_bwd():
        st = {k: v.clone() for k, v in train_state.items()}
        logits, value = policy.forward(train_view, st)
        loss = value.float().mean()
        if hasattr(logits, "log_prob_sum"):
            pass
        loss.backward()
        policy.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bench("train fwd+bwd(value-only) bf16", train_fwd_bwd, iters=max(2, args.iters // 2))

    # Raw LSTM cost for reference
    lstm = policy.lstm if hasattr(policy, "lstm") else None
    if lstm is not None:
        x = torch.randn(args.bptt, segments, policy.input_size, device=device)
        h0 = torch.zeros(1, segments, policy.hidden_size, device=device)
        c0 = torch.zeros(1, segments, policy.hidden_size, device=device)
        with torch.no_grad():
            bench("raw nn.LSTM fp32 [bptt,seg,in]", lambda: lstm(x, (h0, c0)))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bench("raw nn.LSTM bf16", lambda: lstm(x.bfloat16(), (h0.bfloat16(), c0.bfloat16())))
        cell = policy.cell if hasattr(policy, "cell") else None
        if cell is not None:
            xe = torch.randn(args.rows, policy.input_size, device=device)
            he = torch.zeros(args.rows, policy.hidden_size, device=device)
            ce = torch.zeros(args.rows, policy.hidden_size, device=device)
            with torch.no_grad():
                bench("raw LSTMCell fp32 [rows,in]", lambda: cell(xe, (he, ce)))


if __name__ == "__main__":
    main()
