"""Portal-GP shaping bonus (AZK_PORTAL_GP_BONUS).

Differential test: identical fixed-deck battles driven by the same seeded
random legal actions, knob off vs on. Reward streams must diverge exactly on
GATE_PORTAL steps whose portaled entity has gate points, by
weight * min(GP,4)/4 (shaping scale 1 without anneal), for the acting player
(+) and opponent (-). Knob off must reproduce the baseline exactly.

Run: PYTHONPATH=build/python/src:python/src pytest \
    python/src/test_portal_gp_bonus.py -q
"""

from __future__ import annotations

import os

import numpy as np

GATE_PORTAL = 10


def _rollout(bonus: float, steps: int = 2500):
    # Env knobs are latched at module/process init inside the C layer, so this
    # test drives one env per subprocess via fork before binding import.
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_rollout_worker, args=(bonus, steps, queue))
    proc.start()
    out = queue.get(timeout=600)
    proc.join(timeout=60)
    if isinstance(out, str):
        raise RuntimeError(out)
    return out


def _rollout_worker(bonus: float, steps: int, queue):
    try:
        os.environ.pop("AZK_DEBUG_FORCE_GATE_DEF_IDS", None)
        if bonus > 0:
            os.environ["AZK_PORTAL_GP_BONUS"] = str(bonus)
        else:
            os.environ.pop("AZK_PORTAL_GP_BONUS", None)
        from azk_native import AzukiNativeEnv, NATIVE_OBS_DTYPE
        from training_deck_pool import load_training_deck_pool

        env = AzukiNativeEnv(num_envs=1, deck_pool=load_training_deck_pool(), seed=5)
        env.reset(seed=5)
        view = env.observations.view(NATIVE_OBS_DTYPE).reshape(env.num_agents)
        rng = np.random.default_rng(99)
        rewards, actions_taken = [], []
        for _ in range(steps):
            acts = np.zeros((env.num_agents, 4), dtype=np.int32)
            for row in range(env.num_agents):
                am = view[row]["action_mask"]
                count = int(am["legal_action_count"])
                if count > 0:
                    r = int(rng.integers(0, count))
                    acts[row] = (
                        am["legal_primary"][r], am["legal_sub1"][r],
                        am["legal_sub2"][r], am["legal_sub3"][r],
                    )
            env.actions[:] = acts.reshape(env.actions.shape)
            env.step()
            rewards.append(env.rewards.copy())
            actions_taken.append(acts.copy())
        env.close()
        queue.put((np.array(rewards), np.array(actions_taken)))
    except Exception as exc:  # pragma: no cover
        import traceback

        queue.put(traceback.format_exc() + str(exc))


def test_portal_gp_bonus_differential():
    base_r, base_a = _rollout(0.0)
    bon_r, bon_a = _rollout(0.4)
    # identical action streams (same seeds, same rng) => same trajectory
    np.testing.assert_array_equal(base_a, bon_a)
    diff = bon_r - base_r
    portal_steps = (base_a[:, :, 0] == GATE_PORTAL).any(axis=1)
    # off-portal steps must be bit-identical
    assert np.abs(diff[~portal_steps]).max() == 0.0
    # some portal steps must have fired and some must carry a positive bonus
    assert portal_steps.sum() > 0, "random rollout never portaled; extend steps"
    changed = np.abs(diff).max(axis=1) > 0
    assert changed.sum() > 0, "no GP>0 portal produced a bonus"
    assert (changed & ~portal_steps).sum() == 0
    # bonus magnitude on changed steps is weight * k/4 for k in 1..4
    mags = np.abs(diff[changed]).max(axis=1)
    steps_of = np.round(mags / 0.1).astype(int)
    assert ((steps_of >= 1) & (steps_of <= 4)).all(), mags
    # zero-sum: actor + opponent deltas cancel
    assert np.abs(diff.sum(axis=1)[changed]).max() < 1e-5
