from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest


_PROBE = textwrap.dedent(
  """
  import json
  import numpy as np

  from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
  from deck_building import build_deck_build_catalog
  from training_deck_pool import load_training_deck_pool
  from training_utils import make_azuki_env

  pool = load_training_deck_pool()
  catalog = build_deck_build_catalog(pool)
  gate = catalog.records_by_code["STT01-002"].card_def_id
  leaders = catalog.leader_def_ids_by_element[catalog.records_by_def_id[gate].element]
  env = make_azuki_env(
    seed=83,
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
    env.reset_evaluation_games([
      {
        "env_index": 0,
        "seed": 1931,
        "gate0": gate,
        "gate1": gate,
        "leader0": leaders[0],
        "leader1": leaders[1],
      }
    ])
    structured = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
    for _ in range(180):
      if int(structured[0]["deck_context"]["mode"]) == 0:
        break
      player = int(env.active_players()[0])
      mask = structured[player]["action_mask"]
      env.actions.fill(0)
      env.actions[player] = np.asarray(
        [
          mask["legal_primary"][0],
          mask["legal_sub1"][0],
          mask["legal_sub2"][0],
          mask["legal_sub3"][0],
        ],
        dtype=np.int32,
      )
      env.step()
    else:
      raise AssertionError("draft did not finish")

    rng = np.random.default_rng(1931)
    for _ in range(600):
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
      raise AssertionError("random legal policy did not terminate")
    if int(env.active_players()[0]) >= 0:
      env.force_evaluation_truncations([0])

    record = env.drain_evaluation_records()[0]
    players = []
    for player in record["players"]:
      entries = {
        int(entry[0]): {
          "raw_abs_sum": float(entry[2]),
          "raw_discounted_sum": float(entry[3]),
        }
        for entry in player["reward_telemetry"]["overall"]
      }
      players.append(
        {
          "initial_potential": float(player["reward_telemetry"]["initial_potential"]),
          "potential": {str(component): entries.get(component, {}) for component in range(4, 8)},
        }
      )
    print(json.dumps(players))
  finally:
    env.close()
  """
)


class RewardPotentialWeightTests(unittest.TestCase):
  def _probe(self, weights: tuple[float, float, float, float]) -> list[dict[str, object]]:
    env = os.environ.copy()
    names = (
      "AZK_REWARD_LEADER_HEALTH_WEIGHT",
      "AZK_REWARD_GARDEN_ATTACK_WEIGHT",
      "AZK_REWARD_UNTAPPED_GARDEN_WEIGHT",
      "AZK_REWARD_UNTAPPED_IKZ_WEIGHT",
    )
    for name, weight in zip(names, weights, strict=True):
      env[name] = str(weight)
    completed = subprocess.run(
      [sys.executable, "-c", _PROBE],
      check=True,
      capture_output=True,
      env=env,
      text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])

  def test_resource_only_profile_removes_combat_state_components(self) -> None:
    players = self._probe((0.0, 0.0, 0.15, 0.15))
    resource_signal = 0.0
    for player in players:
      potential = player["potential"]
      self.assertEqual(potential["4"], {})
      self.assertEqual(potential["5"], {})
      resource_signal += sum(
        float(potential[str(component)].get("raw_abs_sum", 0.0))
        for component in (6, 7)
      )
    self.assertGreater(resource_signal, 0.0)

  def test_zero_profile_disables_phi_exactly(self) -> None:
    players = self._probe((0.0, 0.0, 0.0, 0.0))
    for player in players:
      self.assertEqual(float(player["initial_potential"]), 0.0)
      self.assertEqual(player["potential"], {str(component): {} for component in range(4, 8)})


if __name__ == "__main__":
  unittest.main()
