from __future__ import annotations

import unittest

import numpy as np

from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool
from training_utils import make_azuki_env


class RewardTelemetryTests(unittest.TestCase):
  @staticmethod
  def _first_legal_action(env, structured, env_index: int = 0) -> np.ndarray:
    player = int(env.active_players()[env_index])
    if player < 0:
      raise AssertionError("environment has no active player")
    row = 2 * env_index + player
    mask = structured[row]["action_mask"]
    if int(mask["legal_action_count"]) <= 0:
      raise AssertionError("active player has no legal action")
    return np.asarray(
      [
        mask["legal_primary"][0],
        mask["legal_sub1"][0],
        mask["legal_sub2"][0],
        mask["legal_sub3"][0],
      ],
      dtype=np.int32,
    )

  @staticmethod
  def _component_sums(entries) -> tuple[float, float]:
    raw = sum(float(entry[1]) for entry in entries)
    scaled = sum(float(entry[7]) for entry in entries)
    return raw, scaled

  @staticmethod
  def _discounted_potential_sum(entries) -> float:
    return sum(float(entry[3]) for entry in entries if 4 <= int(entry[0]) <= 7)

  @staticmethod
  def _discounted_scaled_potential_sum(entries) -> float:
    return sum(float(entry[9]) for entry in entries if 4 <= int(entry[0]) <= 7)

  def test_enabled_telemetry_preserves_trajectory_and_reconstructs_rewards(self) -> None:
    deck_pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(deck_pool)
    gate = catalog.records_by_code["STT01-002"].card_def_id
    leaders = catalog.leader_def_ids_by_element[catalog.records_by_def_id[gate].element]
    common = {
      "seed": 71,
      "native": True,
      "native_envs_per_instance": 1,
      "deck_building_enabled": True,
      "draft_uniform_assignment": True,
      "evaluation_mode": True,
    }
    control = make_azuki_env(**common)
    treatment = make_azuki_env(**common, reward_telemetry=True)
    try:
      games = [
        {
          "env_index": 0,
          "seed": 1907,
          "gate0": gate,
          "gate1": gate,
          "leader0": leaders[0],
          "leader1": leaders[1],
        }
      ]
      control.reset_evaluation_games(games)
      treatment.reset_evaluation_games(games)
      control_view = control.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
      treatment_view = treatment.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)

      for _ in range(180):
        np.testing.assert_array_equal(control.observations, treatment.observations)
        self.assertEqual(control.active_players().tolist(), treatment.active_players().tolist())
        if int(control_view[0]["deck_context"]["mode"]) == 0:
          break
        action = self._first_legal_action(control, control_view)
        player = int(control.active_players()[0])
        control.actions.fill(0)
        treatment.actions.fill(0)
        control.actions[player] = action
        treatment.actions[player] = action
        control.step()
        treatment.step()
        np.testing.assert_array_equal(control.rewards, treatment.rewards)
        np.testing.assert_array_equal(control.terminals, treatment.terminals)
        np.testing.assert_array_equal(control.truncations, treatment.truncations)
      else:
        self.fail("draft did not finish")

      for _ in range(20):
        if int(control.active_players()[0]) < 0:
          break
        np.testing.assert_array_equal(control.observations, treatment.observations)
        action = self._first_legal_action(control, control_view)
        player = int(control.active_players()[0])
        control.actions.fill(0)
        treatment.actions.fill(0)
        control.actions[player] = action
        treatment.actions[player] = action
        control.step()
        treatment.step()
        np.testing.assert_array_equal(control.rewards, treatment.rewards)
        np.testing.assert_array_equal(control.terminals, treatment.terminals)
        np.testing.assert_array_equal(control.truncations, treatment.truncations)

      if int(control.active_players()[0]) >= 0:
        control.force_evaluation_truncations([0])
        treatment.force_evaluation_truncations([0])
      control_records = control.drain_evaluation_records()
      treatment_records = treatment.drain_evaluation_records()
      self.assertEqual(len(control_records), 1)
      self.assertEqual(len(treatment_records), 1)
      self.assertNotIn("reward_telemetry", control_records[0])
      record = treatment_records[0]
      diagnostics = record["reward_telemetry"]
      self.assertLessEqual(float(diagnostics["raw_reconstruction_max_abs_error"]), 2e-6)
      self.assertLessEqual(float(diagnostics["scaled_reconstruction_max_abs_error"]), 2e-6)
      self.assertAlmostEqual(float(diagnostics["gamma"]), 0.99, places=7)

      for player in record["players"]:
        telemetry = player["reward_telemetry"]
        raw, scaled = self._component_sums(telemetry["overall"])
        expected_raw = float(telemetry["raw_shaping_return"]) + float(telemetry["terminal_return"])
        expected_scaled = float(telemetry["scaled_shaping_return"]) + float(telemetry["terminal_return"])
        self.assertAlmostEqual(raw, expected_raw, places=5)
        self.assertAlmostEqual(scaled, expected_scaled, places=5)
        for entry in telemetry["overall"]:
          if int(entry[0]) <= 3:
            self.assertAlmostEqual(float(entry[1]), float(entry[7]), places=7)

      metrics = treatment._deckbuild_helper.process_records(treatment_records)
      self.assertLessEqual(metrics["reward_telemetry/raw_reconstruction_max_abs_error"], 2e-6)
      self.assertEqual(
        metrics["reward_component/all/portal_outcome/raw_abs_sum"],
        0.0,
      )
      self.assertIn("reward_component/action/noop/noop_penalty/raw_abs_sum", metrics)
      self.assertTrue(
        any(key.startswith("reward_component/turn/turn_") for key in metrics)
      )
    finally:
      control.close()
      treatment.close()

  def test_decomposed_schedule_scales_exploration_without_scaling_potential(self):
    deck_pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(deck_pool)
    gate = catalog.records_by_code["STT01-002"].card_def_id
    leaders = catalog.leader_def_ids_by_element[
      catalog.records_by_def_id[gate].element
    ]
    env = make_azuki_env(
      seed=73,
      native=True,
      native_envs_per_instance=1,
      deck_building_enabled=True,
      draft_uniform_assignment=True,
      evaluation_mode=True,
      reward_telemetry=True,
      reward_decomposed_schedule=True,
    )
    env._reward_scales[:] = np.asarray([1.0, 0.0], dtype=np.float32)
    try:
      env.reset_evaluation_games(
        [
          {
            "env_index": 0,
            "seed": 1909,
            "gate0": gate,
            "gate1": gate,
            "leader0": leaders[0],
            "leader1": leaders[1],
          }
        ]
      )
      structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
      rng = np.random.default_rng(1909)
      for _ in range(800):
        player = int(env.active_players()[0])
        if player < 0:
          break
        mask = structured[player]["action_mask"]
        legal_index = int(rng.integers(int(mask["legal_action_count"])))
        env.actions.fill(0)
        env.actions[player] = np.asarray(
          [
            mask["legal_primary"][legal_index],
            mask["legal_sub1"][legal_index],
            mask["legal_sub2"][legal_index],
            mask["legal_sub3"][legal_index],
          ],
          dtype=np.int32,
        )
        env.step()
      else:
        self.fail("random legal policy did not terminate")

      records = env.drain_evaluation_records()
      self.assertEqual(len(records), 1)
      diagnostics = records[0]["reward_telemetry"]
      self.assertLessEqual(
        float(diagnostics["scaled_reconstruction_max_abs_error"]), 2e-6
      )
      potential_abs_sum = 0.0
      exploration_abs_sum = 0.0
      for player_record in records[0]["players"]:
        for entry in player_record["reward_telemetry"]["overall"]:
          component = int(entry[0])
          raw = float(entry[1])
          scaled = float(entry[7])
          if component <= 3:
            self.assertAlmostEqual(scaled, raw, places=7)
          elif component <= 9:
            potential_abs_sum += abs(raw)
            self.assertAlmostEqual(scaled, raw, places=7)
          else:
            exploration_abs_sum += abs(raw)
            self.assertAlmostEqual(scaled, 0.0, places=7)
      self.assertGreater(potential_abs_sum, 0.0)
      self.assertGreater(exploration_abs_sum, 0.0)
    finally:
      env.close()

  def test_discounted_pbrs_terminal_closure_telescopes(self) -> None:
    deck_pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(deck_pool)
    gate = catalog.records_by_code["STT01-002"].card_def_id
    leaders = catalog.leader_def_ids_by_element[catalog.records_by_def_id[gate].element]
    env = make_azuki_env(
      seed=79,
      native=True,
      native_envs_per_instance=1,
      deck_building_enabled=True,
      draft_uniform_assignment=True,
      evaluation_mode=True,
      reward_telemetry=True,
      pbrs_mode="discounted",
      pbrs_gamma=0.99,
      pbrs_terminal_closure=True,
    )
    try:
      env.reset_evaluation_games(
        [
          {
            "env_index": 0,
            "seed": 1913,
            "gate0": gate,
            "gate1": gate,
            "leader0": leaders[0],
            "leader1": leaders[1],
          }
        ]
      )
      structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
      for _ in range(180):
        if int(structured[0]["deck_context"]["mode"]) == 0:
          break
        player = int(env.active_players()[0])
        env.actions.fill(0)
        env.actions[player] = self._first_legal_action(env, structured)
        env.step()
      else:
        self.fail("draft did not finish")

      for _ in range(24):
        if int(env.active_players()[0]) < 0:
          break
        player = int(env.active_players()[0])
        env.actions.fill(0)
        env.actions[player] = self._first_legal_action(env, structured)
        env.step()
      if int(env.active_players()[0]) >= 0:
        env.force_evaluation_truncations([0])

      records = env.drain_evaluation_records()
      self.assertEqual(len(records), 1)
      record = records[0]
      diagnostics = record["reward_telemetry"]
      self.assertEqual(diagnostics["pbrs_mode"], "discounted")
      self.assertAlmostEqual(float(diagnostics["pbrs_gamma"]), 0.99, places=7)
      self.assertLessEqual(float(diagnostics["raw_reconstruction_max_abs_error"]), 2e-6)
      self.assertLessEqual(float(diagnostics["scaled_reconstruction_max_abs_error"]), 2e-6)
      for player in record["players"]:
        telemetry = player["reward_telemetry"]
        discounted_potential = self._discounted_potential_sum(telemetry["overall"])
        discounted_scaled_potential = self._discounted_scaled_potential_sum(
          telemetry["overall"]
        )
        self.assertAlmostEqual(float(telemetry["terminal_return"]), 0.0, places=7)
        self.assertAlmostEqual(
          discounted_potential,
          -float(telemetry["initial_potential"])
          + float(telemetry["final_discount"])
          * float(telemetry["final_potential"]),
          delta=2e-5,
        )
        self.assertAlmostEqual(
          discounted_scaled_potential,
          -float(telemetry["initial_scaled_potential"])
          + float(telemetry["final_discount"])
          * float(telemetry["final_scaled_potential"]),
          delta=2e-5,
        )
        for entry in telemetry["overall"]:
          if int(entry[0]) <= 3:
            self.assertAlmostEqual(float(entry[1]), float(entry[7]), places=7)

      env.reset_evaluation_games(
        [
          {
            "env_index": 0,
            "seed": 1919,
            "gate0": gate,
            "gate1": gate,
            "leader0": leaders[0],
            "leader1": leaders[1],
          }
        ]
      )
      for _ in range(180):
        if int(structured[0]["deck_context"]["mode"]) == 0:
          break
        player_index = int(env.active_players()[0])
        env.actions.fill(0)
        env.actions[player_index] = self._first_legal_action(env, structured)
        env.step()
      else:
        self.fail("second draft did not finish")

      rng = np.random.default_rng(1919)
      for _ in range(600):
        player_index = int(env.active_players()[0])
        if player_index < 0:
          break
        mask = structured[player_index]["action_mask"]
        legal_index = int(rng.integers(int(mask["legal_action_count"])))
        env.actions.fill(0)
        env.actions[player_index] = np.asarray(
          [
            mask["legal_primary"][legal_index],
            mask["legal_sub1"][legal_index],
            mask["legal_sub2"][legal_index],
            mask["legal_sub3"][legal_index],
          ],
          dtype=np.int32,
        )
        env.step()
      else:
        self.fail("random legal policy did not reach game over")

      terminal_records = env.drain_evaluation_records()
      self.assertEqual(len(terminal_records), 1)
      terminal_record = terminal_records[0]
      self.assertEqual(int(terminal_record["end_reason"]), 0)
      for player in terminal_record["players"]:
        telemetry = player["reward_telemetry"]
        self.assertIn(float(telemetry["terminal_return"]), (-5.0, 0.0, 5.0))
        self.assertAlmostEqual(
          self._discounted_potential_sum(telemetry["overall"]),
          -float(telemetry["initial_potential"]),
          delta=2e-5,
        )
        self.assertAlmostEqual(
          self._discounted_scaled_potential_sum(telemetry["overall"]),
          -float(telemetry["initial_scaled_potential"]),
          delta=2e-5,
        )
        for entry in telemetry["overall"]:
          if int(entry[0]) <= 3:
            self.assertAlmostEqual(float(entry[1]), float(entry[7]), places=7)
    finally:
      env.close()

  def test_discounted_pbrs_gamma_one_tracks_scale_turnoff_at_truncation(self) -> None:
    deck_pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(deck_pool)
    gate = catalog.records_by_code["STT01-002"].card_def_id
    leaders = catalog.leader_def_ids_by_element[
      catalog.records_by_def_id[gate].element
    ]
    env = make_azuki_env(
      seed=97,
      native=True,
      native_envs_per_instance=1,
      deck_building_enabled=True,
      draft_uniform_assignment=True,
      evaluation_mode=True,
      reward_telemetry=True,
      reward_decomposed_schedule=True,
      pbrs_mode="discounted",
      pbrs_gamma=1.0,
      pbrs_terminal_closure=True,
    )
    try:
      env._reward_scales[:] = np.asarray([1.0, 0.0], dtype=np.float32)
      env.reset_evaluation_games(
        [
          {
            "env_index": 0,
            "seed": 1937,
            "gate0": gate,
            "gate1": gate,
            "leader0": leaders[0],
            "leader1": leaders[1],
          }
        ]
      )
      structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
      for _ in range(180):
        if int(structured[0]["deck_context"]["mode"]) == 0:
          break
        player = int(env.active_players()[0])
        env.actions.fill(0)
        env.actions[player] = self._first_legal_action(env, structured)
        env.step()
      else:
        self.fail("draft did not finish")

      for step_index in range(8):
        player = int(env.active_players()[0])
        if player < 0:
          break
        if step_index == 3:
          env._reward_scales[0] = 0.5
        elif step_index == 6:
          env._reward_scales[0] = 0.0
        env.actions.fill(0)
        env.actions[player] = self._first_legal_action(env, structured)
        env.step()
      if int(env.active_players()[0]) >= 0:
        env.force_evaluation_truncations([0])

        np.testing.assert_array_equal(
          env.terminal_rewards, np.zeros_like(env.terminal_rewards)
        )
        np.testing.assert_array_equal(env.rewards, np.zeros_like(env.rewards))
        np.testing.assert_array_equal(
          env.shaped_rewards, np.zeros_like(env.shaped_rewards)
        )
      record = env.drain_evaluation_records()[0]
      self.assertNotEqual(int(record["end_reason"]), 0)
      self.assertAlmostEqual(
        float(record["reward_telemetry"]["gamma"]), 1.0, places=7
      )
      for player in record["players"]:
        telemetry = player["reward_telemetry"]
        component_ids = {int(entry[0]) for entry in telemetry["overall"]}
        self.assertTrue(component_ids.isdisjoint({0, 1, 2, 3}))
        self.assertAlmostEqual(float(telemetry["terminal_return"]), 0.0, places=7)
        self.assertAlmostEqual(
          self._discounted_scaled_potential_sum(telemetry["overall"]),
          -float(telemetry["initial_scaled_potential"])
          + float(telemetry["final_scaled_potential"]),
          delta=2e-5,
        )
    finally:
      env.close()

  def test_pbrs_configuration_rejects_invalid_combinations(self) -> None:
    with self.assertRaisesRegex(ValueError, "legacy.*discounted"):
      make_azuki_env(
        native=True,
        pbrs_mode="invalid",
      )
    with self.assertRaisesRegex(ValueError, "pbrs_gamma"):
      make_azuki_env(
        native=True,
        pbrs_mode="discounted",
        pbrs_gamma=1.01,
      )
    with self.assertRaisesRegex(ValueError, "requires terminal closure"):
      make_azuki_env(
        native=True,
        pbrs_mode="discounted",
        pbrs_terminal_closure=False,
      )
    with self.assertRaisesRegex(ValueError, "requires pbrs_mode"):
      make_azuki_env(
        native=True,
        pbrs_terminal_closure=True,
      )
    with self.assertRaisesRegex(ValueError, "native"):
      make_azuki_env(
        native=False,
        pbrs_mode="discounted",
      )

  def test_telemetry_requires_native_deckbuilding(self) -> None:
    with self.assertRaisesRegex(ValueError, "deck_building"):
      make_azuki_env(native=True, reward_telemetry=True)
    with self.assertRaisesRegex(ValueError, "native"):
      make_azuki_env(native=False, reward_telemetry=True)


if __name__ == "__main__":
  unittest.main()
