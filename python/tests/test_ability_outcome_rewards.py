from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np


_PROBE = textwrap.dedent(
    """
    import hashlib
    import json
    import numpy as np
    from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
    from deck_building import build_deck_build_catalog
    from training_deck_pool import load_training_deck_pool
    from training_utils import make_azuki_env

    catalog = build_deck_build_catalog(load_training_deck_pool())
    contexts = [
        ('STT01-002', 'STT01-001'),
        ('STT02-002', 'STT02-001'),
        ('AZK01-124', 'STT03-001'),
        ('AZK01-122', 'STT04-001'),
    ]
    env = make_azuki_env(
        seed=43, native=True, native_envs_per_instance=1,
        deck_building_enabled=True, draft_uniform_assignment=True,
        evaluation_mode=True, reward_telemetry=True,
        pbrs_mode='discounted', pbrs_gamma=1.0,
        pbrs_terminal_closure=True, reward_decomposed_schedule=True,
    )
    env._reward_scales[:] = [0.0, 1.0]
    games = []
    try:
        for context_index, (gate_code, leader_code) in enumerate(contexts):
            for repeat in range(2):
                seed = 6100 + context_index * 10 + repeat
                gate = catalog.records_by_code[gate_code].card_def_id
                leader = catalog.records_by_code[leader_code].card_def_id
                env.reset_evaluation_games([{
                    'env_index': 0, 'seed': seed,
                    'gate0': gate, 'gate1': gate,
                    'leader0': leader, 'leader1': leader,
                }])
                rows = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
                rng = np.random.default_rng(seed)
                digest = hashlib.sha256()
                total = np.zeros(2, dtype=np.float64)
                terminal = np.zeros(2, dtype=np.float64)
                shaped = np.zeros(2, dtype=np.float64)
                for step in range(1400):
                    active = int(env.active_players()[0])
                    if active < 0:
                        break
                    digest.update(env.observations.tobytes())
                    mask = rows[active]['action_mask']
                    choice = int(rng.integers(int(mask['legal_action_count'])))
                    env.actions.fill(0)
                    env.actions[active] = [mask[key][choice] for key in
                        ('legal_primary', 'legal_sub1', 'legal_sub2', 'legal_sub3')]
                    digest.update(env.actions.tobytes())
                    drafting = int(rows[0]['deck_context']['mode']) != 0
                    env.step()
                    np.testing.assert_allclose(
                        env.rewards, env.terminal_rewards + env.shaped_rewards,
                        rtol=0, atol=1e-6,
                    )
                    if drafting:
                        np.testing.assert_array_equal(env.rewards, [0.0, 0.0])
                    if not env.terminals.any():
                        np.testing.assert_array_equal(env.terminal_rewards, [0.0, 0.0])
                    total += env.rewards
                    terminal += env.terminal_rewards
                    shaped += env.shaped_rewards
                else:
                    raise AssertionError('legal rollout did not finish')
                records = env.drain_evaluation_records()
                assert len(records) == 1
                record = records[0]
                assert not env.drain_evaluation_records()
                games.append({
                    'seed': seed, 'trajectory_sha256': digest.hexdigest(),
                    'total': total.tolist(), 'terminal': terminal.tolist(),
                    'shaped': shaped.tolist(), 'record': record,
                })
        print(json.dumps(games))
    finally:
        env.close()
    """
)


def _rollouts(bonus: float) -> list[dict]:
    process_env = os.environ.copy()
    for name in (
        "AZK_REWARD_LEADER_DELTA_WEIGHT", "AZK_REWARD_BOARD_DELTA_WEIGHT",
        "AZK_REWARD_NOOP_PENALTY", "AZK_PORTAL_GP_BONUS",
        "AZK_PORTAL_OUTCOME_BONUS", "AZK_EARLY_TEMPO_BONUS",
        "AZK_DMG_MITIGATION_BONUS", "AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP",
        "AZK_GENERATED_IKZ_CONVERSION_BONUS", "AZK_TEMP_CHARGE_REALIZATION_BONUS",
        "AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE", "AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS",
        "AZK_MAX_TICKS_CURRICULUM",
    ):
        process_env[name] = "0"
    process_env["AZK_MAX_TICKS_PER_EPISODE"] = "1200"
    process_env["AZK_ABILITY_OUTCOME_BONUS"] = str(bonus)
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], env=process_env,
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_resolved_outcomes_pay_once_without_changing_play_or_terminal_labels() -> None:
    control = _rollouts(0.0)
    treatment = _rollouts(2.0)
    observed_outcomes = 0
    for baseline, shaped in zip(control, treatment, strict=True):
        assert baseline["trajectory_sha256"] == shaped["trajectory_sha256"]
        assert baseline["terminal"] == shaped["terminal"]
        np.testing.assert_array_equal(baseline["shaped"], [0.0, 0.0])
        record = shaped["record"]
        counts = np.asarray([
            player["gate_ability_outcomes"] + player["leader_ability_outcomes"]
            for player in record["players"]
        ])
        observed_outcomes += int(counts.sum())
        expected_shaping = 2.0 * (counts - counts[::-1])
        np.testing.assert_allclose(shaped["shaped"], expected_shaping, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            shaped["total"], np.asarray(shaped["terminal"]) + expected_shaping,
            rtol=0, atol=1e-6,
        )
        if record["end_reason"] == 0:
            np.testing.assert_array_equal(sorted(shaped["terminal"]), [-5.0, 5.0])
        else:
            np.testing.assert_array_equal(shaped["terminal"], [0.0, 0.0])
        diagnostics = record["reward_telemetry"]
        assert diagnostics["raw_reconstruction_max_abs_error"] < 1e-5
        assert diagnostics["scaled_reconstruction_max_abs_error"] < 1e-5
        for seat, player in enumerate(record["players"]):
            telemetry = player["reward_telemetry"]
            assert telemetry["terminal_return"] == shaped["terminal"][seat]
            assert abs(telemetry["scaled_shaping_return"] - expected_shaping[seat]) < 1e-5
    # Without exercising actual effects the equality checks could pass with
    # reward delivery entirely disconnected from successful resolutions.
    assert observed_outcomes > 0
