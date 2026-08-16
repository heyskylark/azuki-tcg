from __future__ import annotations

import unittest

import numpy as np

from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import build_deck_build_catalog
from training_deck_pool import load_training_deck_pool
from training_utils import make_azuki_env


class NativeEvaluationControlTests(unittest.TestCase):
  def test_per_env_forcing_pause_and_record_telemetry(self) -> None:
    deck_pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(deck_pool)
    gate = catalog.records_by_code["STT01-002"].card_def_id
    env = make_azuki_env(
      seed=17,
      native=True,
      native_envs_per_instance=2,
      deck_building_enabled=True,
      evaluation_mode=True,
    )
    try:
      env.reset_evaluation_games(
        [
          {
            "env_index": index,
            "seed": 12345,
            "gate0": gate,
            "gate1": gate,
            "reference_seat": -1,
            "reference_deck_index": -1,
          }
          for index in range(2)
        ]
      )
      structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
      records = []
      for _ in range(500):
        players = env.active_players()
        if bool((players < 0).all()):
          break
        env.actions.fill(0)
        for env_index, player_raw in enumerate(players.tolist()):
          player = int(player_raw)
          if player < 0:
            continue
          row = 2 * env_index + player
          mask = structured[row]["action_mask"]
          self.assertGreater(int(mask["legal_action_count"]), 0)
          env.actions[row] = np.asarray(
            [
              mask["legal_primary"][0],
              mask["legal_sub1"][0],
              mask["legal_sub2"][0],
              mask["legal_sub3"][0],
            ],
            dtype=np.int32,
          )
        env.step()
        records.extend(env.drain_evaluation_records())
      self.assertEqual(len(records), 2)
      self.assertEqual(env.active_players().tolist(), [-1, -1])
      self.assertTrue(
        {
          "garden_or_leader_ability_rate",
          "alley_ability_rate",
          "play_entity_to_garden_rate",
          "play_entity_to_alley_rate",
          "contextual_response_opportunities",
          "temporary_charge_realized",
          "temporary_attack_damage_realized",
          "generated_ikz_created",
          "generated_ikz_converted",
          "entity_damage_dealt",
          "entity_damage_taken",
        }.issubset(records[0]["players"][0])
      )
      comparable = [
        (
          record["end_reason"],
          record["starting_player"],
          record["seed"],
          [player["gate"] for player in record["players"]],
          [player["win"] for player in record["players"]],
        )
        for record in records
      ]
      self.assertEqual(comparable[0], comparable[1])

      env.reset_evaluation_games(
        [
          {
            "env_index": 0,
            "seed": 999,
            "gate0": gate,
            "gate1": gate,
            "reference_seat": -1,
            "reference_deck_index": -1,
          }
        ]
      )
      self.assertGreaterEqual(int(env.active_players()[0]), 0)
      self.assertEqual(int(env.active_players()[1]), -1)
      for _ in range(150):
        player = int(env.active_players()[0])
        row = player
        if int(structured[row]["deck_context"]["mode"]) == 0:
          break
        mask = structured[row]["action_mask"]
        env.actions.fill(0)
        env.actions[row] = np.asarray(
          [
            mask["legal_primary"][0],
            mask["legal_sub1"][0],
            mask["legal_sub2"][0],
            mask["legal_sub3"][0],
          ],
          dtype=np.int32,
        )
        env.step()
        self.assertEqual(env.drain_evaluation_records(), [])
      else:
        self.fail("Forced evaluation game did not finish drafting")
      env.force_evaluation_truncations([0])
      forced = env.drain_evaluation_records()
      self.assertEqual(len(forced), 1)
      self.assertEqual(int(forced[0]["end_reason"]), 1)
      self.assertEqual(int(forced[0]["env_index"]), 0)
    finally:
      env.close()

  def test_uniform_assignment_can_pin_compatible_leaders(self) -> None:
    deck_pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(deck_pool)
    gate = catalog.records_by_code["STT01-002"].card_def_id
    element = catalog.records_by_def_id[gate].element
    leaders = catalog.leader_def_ids_by_element[element]
    env = make_azuki_env(
      seed=31,
      native=True,
      native_envs_per_instance=1,
      deck_building_enabled=True,
      draft_uniform_assignment=True,
      evaluation_mode=True,
    )
    try:
      env.reset_evaluation_games(
        [
          {
            "env_index": 0,
            "seed": 456,
            "gate0": gate,
            "gate1": gate,
            "leader0": leaders[0],
            "leader1": leaders[1],
          }
        ]
      )
      structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
      self.assertEqual(int(structured[0]["deck_context"]["mode"]), 2)
      self.assertEqual(int(structured[1]["deck_context"]["mode"]), 2)
      self.assertEqual(
        [
          int(structured[index]["deck_context"]["leader_card_def_id"])
          for index in range(2)
        ],
        list(leaders[:2]),
      )
    finally:
      env.close()


if __name__ == "__main__":
  unittest.main()
