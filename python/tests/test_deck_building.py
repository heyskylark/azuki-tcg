from __future__ import annotations

import unittest

import numpy as np
from pettingzoo import ParallelEnv

from action import ActionType, build_action_space
from deck_building import (
  IKZ_CARD_CODE,
  IKZ_CARD_COUNT,
  MAIN_CARD_TYPES,
  MAX_MAIN_COPIES,
  DeckBuildingParallelEnv,
  build_deck_build_catalog,
  empty_training_observation,
)
from observation import (
  DECK_CONTEXT_MODE_BATTLE,
  DECK_CONTEXT_MODE_PICK_LEADER,
  DECK_CONTEXT_MODE_PICK_MAIN,
  MAX_DECK_SIZE,
  build_observation_space,
)
from training_deck_pool import load_training_deck_pool


class _FakeBattleEnv(ParallelEnv):
  metadata = {"render_modes": ["ansi"], "name": "fake_battle"}

  def __init__(self):
    self.possible_agents = [0, 1]
    self.agents = self.possible_agents[:]
    self.render_mode = "ansi"
    self._observation_space = build_observation_space()
    self._action_space = build_action_space()
    self._active_player_index = 0
    self.reset_decks = None
    self.step_actions = []
    self.terminate_next_step = False
    self.win_agent = 0

  def observation_space(self, agent):
    return self._observation_space

  def action_space(self, agent):
    return self._action_space

  def _battle_obs(self):
    observations = {}
    for agent in self.possible_agents:
      obs = empty_training_observation()
      obs["critic_privileged"]["self_deck"] = tuple(
        {"card_def_id": 0, "zone_index": i} for i in range(MAX_DECK_SIZE)
      )
      obs["critic_privileged"]["opponent_deck"] = tuple(
        {"card_def_id": 0, "zone_index": i} for i in range(MAX_DECK_SIZE)
      )
      legal = obs["action_mask"]["legal_actions"]
      obs["action_mask"]["primary_action_mask"][int(ActionType.NOOP)] = True
      obs["action_mask"]["legal_action_count"] = 1
      legal["legal_primary"][0] = int(ActionType.NOOP)
      observations[agent] = obs
    return observations

  def reset_with_decks(self, *, seed: int, player_decks):
    self.reset_decks = (int(seed), tuple(player_decks))
    self.agents = self.possible_agents[:]
    self._active_player_index = 0
    return self._battle_obs(), {agent: {} for agent in self.possible_agents}

  def step(self, actions):
    self.step_actions.append(actions)
    observations = self._battle_obs()
    rewards = {agent: 0.0 for agent in self.possible_agents}
    terminations = {agent: bool(self.terminate_next_step) for agent in self.possible_agents}
    truncations = {agent: False for agent in self.possible_agents}
    infos = {
      agent: {"win": 1.0 if self.terminate_next_step and agent == self.win_agent else 0.0}
      for agent in self.possible_agents
    }
    if self.terminate_next_step:
      self.agents = []
    return observations, rewards, terminations, truncations, infos

  def render(self):
    return "fake battle\n"


class DeckBuildCatalogTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.deck_pool = load_training_deck_pool()
    cls.catalog = build_deck_build_catalog(cls.deck_pool)

  def test_gate_population_matches_deck_pool_size(self):
    self.assertEqual(len(self.catalog.gate_def_id_population), len(self.deck_pool))
    for gate_def_id in self.catalog.gate_def_id_population:
      self.assertEqual(self.catalog.records_by_def_id[gate_def_id].card_type, "GATE")

  def test_valid_candidates_match_gate_element_rules(self):
    gate_def_id = self.catalog.gate_def_id_population[0]
    gate = self.catalog.records_by_def_id[gate_def_id]
    leader_candidates = self.catalog.leader_def_ids_by_element[gate.element]
    main_candidates = self.catalog.main_def_ids_by_element[gate.element]
    self.assertGreater(len(leader_candidates), 0)
    self.assertGreater(len(main_candidates), 0)
    for card_def_id in leader_candidates:
      record = self.catalog.records_by_def_id[card_def_id]
      self.assertEqual(record.card_type, "LEADER")
      self.assertEqual(record.element, gate.element)
    for card_def_id in main_candidates:
      record = self.catalog.records_by_def_id[card_def_id]
      self.assertIn(record.card_type, MAIN_CARD_TYPES)
      self.assertIn(record.element, ("NORMAL", gate.element))


class DeckBuildingWrapperTests(unittest.TestCase):
  def setUp(self):
    deck_pool = load_training_deck_pool()
    self.env = DeckBuildingParallelEnv(
      _FakeBattleEnv(),
      deck_pool=deck_pool,
      seed=123,
    )

  def _pick_first_until_battle(self):
    observations, _ = self.env.reset(seed=7)
    infos = {agent: {} for agent in self.env.possible_agents}
    steps = 0
    while self.env._building:
      active = self.env._active_player_index
      obs = observations[active]
      legal_count = int(obs["action_mask"]["legal_action_count"])
      self.assertGreater(legal_count, 0)
      action = np.asarray([int(ActionType.DECK_PICK_CARD), 0, 0, 0], dtype=np.int32)
      observations, _, _, _, infos = self.env.step({active: action})
      steps += 1
      self.assertLessEqual(steps, 102)
    return observations, infos, steps

  def test_reset_starts_with_leader_pick_and_hidden_battle_state(self):
    observations, _ = self.env.reset(seed=1)
    active_obs = observations[self.env._active_player_index]
    inactive_obs = observations[1 - self.env._active_player_index]
    self.assertEqual(active_obs["deck_context"]["mode"], DECK_CONTEXT_MODE_PICK_LEADER)
    self.assertGreater(active_obs["deck_context"]["candidate_count"], 0)
    self.assertEqual(inactive_obs["action_mask"]["legal_action_count"], 0)
    self.assertTrue(
      all(card["card_def_id"] == -1 for card in active_obs["critic_privileged"]["self_deck"])
    )

  def test_builds_two_complete_decks_then_starts_battle(self):
    observations, infos, steps = self._pick_first_until_battle()
    self.assertEqual(steps, 102)
    self.assertIsNotNone(self.env.env.reset_decks)
    _, player_decks = self.env.env.reset_decks
    for deck in player_decks:
      total_cards = sum(quantity for _, quantity in deck)
      self.assertEqual(total_cards, MAX_DECK_SIZE + 2 + IKZ_CARD_COUNT)
      counts = dict(deck)
      self.assertEqual(counts[IKZ_CARD_CODE], IKZ_CARD_COUNT)
      for card_code, quantity in deck:
        if card_code != IKZ_CARD_CODE and quantity > 1:
          self.assertLessEqual(quantity, MAX_MAIN_COPIES)
    for agent, obs in observations.items():
      self.assertEqual(obs["deck_context"]["mode"], DECK_CONTEXT_MODE_BATTLE)
      self.assertEqual(obs["deck_context"]["main_count"], MAX_DECK_SIZE)
      self.assertEqual(obs["deck_context"]["candidate_count"], 0)
      self.assertTrue(
        all(card["card_def_id"] == -1 for card in obs["critic_privileged"]["self_deck"])
      )
      info = infos[agent]
      self.assertEqual(info["deckbuild/completed"], 1.0)
      self.assertEqual(info["azk_step_deckbuild/completed"], 1.0)
      self.assertEqual(info["deckbuild/main_count"], float(MAX_DECK_SIZE))
      self.assertGreater(info["deckbuild/main_unique"], 0.0)
      self.assertGreater(info["deckbuild/main_unique_share"], 0.0)
      self.assertGreaterEqual(info["deckbuild/main_copy_entropy"], 0.0)
      self.assertGreaterEqual(info["deckbuild/main_copy_entropy_norm"], 0.0)
      self.assertLessEqual(info["deckbuild/main_copy_entropy_norm"], 1.0)
      type_count_total = sum(
        value
        for key, value in info.items()
        if key.startswith("deckbuild/main_type_count/")
      )
      self.assertEqual(type_count_total, float(MAX_DECK_SIZE))
      type_share_total = sum(
        value
        for key, value in info.items()
        if key.startswith("deckbuild/main_type_share/")
      )
      self.assertAlmostEqual(type_share_total, 1.0)
      element_count_total = sum(
        value
        for key, value in info.items()
        if key.startswith("deckbuild/main_element_count/")
      )
      self.assertEqual(element_count_total, float(MAX_DECK_SIZE))
      element_share_total = sum(
        value
        for key, value in info.items()
        if key.startswith("deckbuild/main_element_share/")
      )
      self.assertAlmostEqual(element_share_total, 1.0)
      self.assertFalse(any(key.startswith("deckbuild/card_count/") for key in info))

  def test_deck_pick_actions_are_not_passed_to_battle_env(self):
    observations, _, _ = self._pick_first_until_battle()
    active = self.env._active_player_index
    self.env.step({active: np.asarray([int(ActionType.NOOP), 0, 0, 0], dtype=np.int32)})
    self.assertEqual(len(self.env.env.step_actions), 1)
    passed = self.env.env.step_actions[0][active]
    self.assertNotEqual(int(passed[0]), int(ActionType.DECK_PICK_CARD))

  def test_terminal_battle_infos_include_deckbuild_result_metrics(self):
    self._pick_first_until_battle()
    self.env.env.terminate_next_step = True
    self.env.env.win_agent = 0
    _, _, terminations, _, infos = self.env.step(
      {0: np.asarray([int(ActionType.NOOP), 0, 0, 0], dtype=np.int32)}
    )
    self.assertTrue(all(terminations.values()))
    self.assertEqual(infos[0]["deckbuild_result/game"], 1.0)
    self.assertEqual(infos[0]["deckbuild_result/win"], 1.0)
    self.assertEqual(infos[1]["deckbuild_result/game"], 1.0)
    self.assertEqual(infos[1]["deckbuild_result/win"], 0.0)
    self.assertIn("deckbuild_result/gate_match_win_joint", infos[0])
    self.assertIn("deckbuild_result/gate_mismatch_win_joint", infos[0])
    self.assertTrue(
      "deckbuild_result/gate_match/win" in infos[0]
      or "deckbuild_result/gate_mismatch/win" in infos[0]
    )
    self.assertTrue(any(key.startswith("deckbuild_result/gate/") for key in infos[0]))
    self.assertTrue(any(key.startswith("deckbuild_result/gate_pair/") for key in infos[0]))
    gate_freq_total = sum(
      value
      for key, value in infos[0].items()
      if key.startswith("deckbuild_result/gate_freq/")
    )
    self.assertEqual(gate_freq_total, 1.0)
    gate_pair_freq_total = sum(
      value
      for key, value in infos[0].items()
      if key.startswith("deckbuild_result/gate_pair_freq/")
    )
    self.assertEqual(gate_pair_freq_total, 1.0)


if __name__ == "__main__":
  unittest.main()
