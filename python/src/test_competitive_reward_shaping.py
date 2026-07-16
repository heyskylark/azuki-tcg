"""Differential checks for the competitive-play reward shaping signals."""

from __future__ import annotations

import numpy as np

from action import ActionType
from test_early_tempo_bonus import _rollout


def _assert_same_trajectory_and_zero_sum(base, shaped):
  base_rewards, base_actions = base
  shaped_rewards, shaped_actions = shaped
  np.testing.assert_array_equal(base_actions, shaped_actions)
  diff = shaped_rewards - base_rewards
  changed = np.abs(diff).max(axis=1) > 1e-9
  assert changed.any(), "reward never fired; extend the deterministic rollout"
  assert np.abs(diff.sum(axis=1)[changed]).max() < 1e-5
  return diff, changed, base_actions


def test_entity_damage_exchange_is_capped_and_quantized():
  base = _rollout({}, steps=5000)
  shaped = _rollout(
      {
          "AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP": "0.025",
          "AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP": "6",
      },
      steps=5000,
  )
  diff, changed, _ = _assert_same_trajectory_and_zero_sum(base, shaped)
  units = np.abs(diff[changed]).max(axis=1) / 0.025
  assert np.allclose(units, np.round(units), atol=1e-4)
  assert (np.round(units) >= 1).all()
  assert (np.round(units) <= 6).all()


def test_generated_ikz_only_pays_on_conversion():
  common = {"AZK_REWARD_UNTAPPED_IKZ_WEIGHT": "0"}
  base = _rollout(common, steps=10000)
  shaped = _rollout(
      {
          **common,
          "AZK_GENERATED_IKZ_CONVERSION_BONUS": "0.05",
          "AZK_GENERATED_IKZ_CONVERSION_STEP_CAP": "4",
      },
      steps=10000,
  )
  diff, changed, _ = _assert_same_trajectory_and_zero_sum(base, shaped)
  units = np.abs(diff[changed]).max(axis=1) / 0.05
  assert np.allclose(units, np.round(units), atol=1e-4)
  assert (np.round(units) >= 1).all()
  assert (np.round(units) <= 4).all()


def test_temporary_effect_realization_has_bounded_components():
  base = _rollout({}, steps=10000)
  shaped = _rollout(
      {
          "AZK_TEMP_CHARGE_REALIZATION_BONUS": "0.08",
          "AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE": "0.025",
          "AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP": "4",
      },
      steps=10000,
  )
  diff, changed, _ = _assert_same_trajectory_and_zero_sum(base, shaped)
  magnitudes = np.abs(diff[changed]).max(axis=1)
  valid = []
  for magnitude in magnitudes:
    valid.append(
        any(
            np.isclose(magnitude, charge * 0.08 + damage * 0.025, atol=1e-5)
            for charge in (0, 1)
            for damage in range(5)
            if charge or damage
        )
    )
  assert all(valid), magnitudes[:10]


def test_contextual_reserve_pays_on_attack_opening_only():
  common = {"AZK_REWARD_UNTAPPED_IKZ_WEIGHT": "0"}
  base = _rollout(common, steps=5000)
  shaped = _rollout(
      {**common, "AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS": "0.08"},
      steps=5000,
  )
  diff, changed, actions = _assert_same_trajectory_and_zero_sum(base, shaped)
  attack_steps = (
      actions[:, :, 0] == int(ActionType.ATTACK)
  ).any(axis=1)
  assert not (changed & ~attack_steps).any()
  magnitudes = np.abs(diff[changed]).max(axis=1)
  assert np.allclose(magnitudes, 0.08, atol=1e-6)
