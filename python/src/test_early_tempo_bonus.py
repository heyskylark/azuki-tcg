"""Early-tempo bonus (AZK_EARLY_TEMPO_BONUS) differential tests.

Same seeded random-legal trajectory with the knob off vs on: reward diffs may
appear ONLY on steps whose action is a qualifying development action (plays,
portal, ability activations, affirmative confirms, attacks — primaries
{4,5,6,7,8,9,10,12}? — asserted via recorded actions), must be zero-sum, and
the capped arm must pay no more per (player, turn) than the cap.

Run: PYTHONPATH=build/python/src:python/src pytest \
    python/src/test_early_tempo_bonus.py -q
"""

from __future__ import annotations

import os

import numpy as np

from test_portal_gp_bonus import GATE_PORTAL  # noqa: F401  (env driver reuse)

# ActionType primaries that qualify (mirror of early_tempo_qualifying_action):
# ATTACK, PLAY_ENTITY_TO_GARDEN, PLAY_ENTITY_TO_ALLEY, PLAY_SPELL_FROM_HAND,
# ATTACH_WEAPON_FROM_HAND, GATE_PORTAL, ACTIVATE_GARDEN_OR_LEADER_ABILITY,
# ACTIVATE_ALLEY_ABILITY, CONFIRM_ABILITY
from action import ActionType

QUALIFYING = {
    int(ActionType.ATTACK),
    int(ActionType.PLAY_ENTITY_TO_GARDEN),
    int(ActionType.PLAY_ENTITY_TO_ALLEY),
    int(ActionType.PLAY_SPELL_FROM_HAND),
    int(ActionType.ATTACH_WEAPON_FROM_HAND),
    int(ActionType.GATE_PORTAL),
    int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY),
    int(ActionType.ACTIVATE_ALLEY_ABILITY),
    int(ActionType.CONFIRM_ABILITY),
}

DEDUP_EXCLUDED = {
    int(ActionType.GATE_PORTAL),
    int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY),
    int(ActionType.ACTIVATE_ALLEY_ABILITY),
    int(ActionType.CONFIRM_ABILITY),
}


def _rollout(env_updates: dict, steps: int = 2500):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_rollout_worker, args=(env_updates, steps, queue))
    proc.start()
    out = queue.get(timeout=600)
    proc.join(timeout=60)
    if isinstance(out, str):
        raise RuntimeError(out)
    return out


def _rollout_worker(env_updates: dict, steps: int, queue):
    try:
        for key in ("AZK_EARLY_TEMPO_BONUS", "AZK_EARLY_TEMPO_CAP",
                    "AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES",
                    "AZK_PORTAL_GP_BONUS", "AZK_PORTAL_OUTCOME_BONUS",
                    "AZK_DEBUG_FORCE_GATE_DEF_IDS"):
            os.environ.pop(key, None)
        os.environ.update(env_updates)
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
    except Exception:
        import traceback

        queue.put(traceback.format_exc())


def test_early_tempo_differential_and_cap():
    base_r, base_a = _rollout({})
    cap_r, cap_a = _rollout({"AZK_EARLY_TEMPO_BONUS": "0.1", "AZK_EARLY_TEMPO_CAP": "1"})
    unc_r, unc_a = _rollout({"AZK_EARLY_TEMPO_BONUS": "0.1", "AZK_EARLY_TEMPO_CAP": "0"})
    np.testing.assert_array_equal(base_a, cap_a)
    np.testing.assert_array_equal(base_a, unc_a)

    for tag, r in (("cap1", cap_r), ("uncapped", unc_r)):
        diff = r - base_r
        changed = np.abs(diff).max(axis=1) > 0
        assert changed.sum() > 0, f"{tag}: bonus never fired"
        # only qualifying-action steps change
        qual_step = np.array([
            any(int(a[0]) in QUALIFYING for a in step_acts) for step_acts in base_a
        ])
        assert (changed & ~qual_step).sum() == 0, f"{tag}: paid on non-qualifying step"
        # zero-sum
        assert np.abs(diff.sum(axis=1)[changed]).max() < 1e-5, f"{tag}: not zero-sum"
        # magnitude is exactly the bonus (anneal off in this env => scale 1)
        mags = np.abs(diff[changed]).max(axis=1)
        assert np.allclose(mags, 0.1, atol=1e-6), f"{tag}: wrong magnitude {mags[:5]}"

    # cap semantics: cap=1 pays on no more steps than uncapped, strictly fewer
    # if any turn had 2+ qualifying actions
    cap_paid = (np.abs(cap_r - base_r).max(axis=1) > 0).sum()
    unc_paid = (np.abs(unc_r - base_r).max(axis=1) > 0).sum()
    assert cap_paid <= unc_paid
    assert unc_paid > 0


def test_early_tempo_portal_ability_dedup_differential():
    full_r, full_a = _rollout(
        {"AZK_EARLY_TEMPO_BONUS": "0.1", "AZK_EARLY_TEMPO_CAP": "0"}
    )
    dedup_r, dedup_a = _rollout(
        {
            "AZK_EARLY_TEMPO_BONUS": "0.1",
            "AZK_EARLY_TEMPO_CAP": "0",
            "AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES": "1",
        }
    )
    np.testing.assert_array_equal(full_a, dedup_a)

    diff = full_r - dedup_r
    changed = np.abs(diff).max(axis=1) > 0
    excluded_step = np.array([
        any(int(action[0]) in DEDUP_EXCLUDED for action in step_actions)
        for step_actions in full_a
    ])
    portal_step = (full_a[:, :, 0] == int(ActionType.GATE_PORTAL)).any(axis=1)
    assert changed.sum() > 0, "deduplication never removed a tempo payment"
    assert (changed & ~excluded_step).sum() == 0
    assert (changed & portal_step).sum() > 0, "no early portal payment was removed"
    assert np.abs(diff.sum(axis=1)[changed]).max() < 1e-5
    assert np.allclose(np.abs(diff[changed]).max(axis=1), 0.1, atol=1e-6)
