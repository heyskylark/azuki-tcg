"""Shared fixtures: C reference env harness + deck pool."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))


@pytest.fixture(scope="session")
def native_pool():
  from training_deck_pool import load_training_deck_pool

  return load_training_deck_pool(str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))


class CRef:
  """Reference C environment with semantic accessors over the obs structs."""

  def __init__(self, seed: int, deck_pool=None):
    from tcg_parallel import AzukiTCGParallel

    self.env = AzukiTCGParallel(seed=seed, deck_pool=deck_pool)

  def raw(self, player: int):
    return self.env._raw_observation(player)

  @property
  def active_player(self) -> int:
    import binding

    return int(binding.env_active_player(self.env.c_envs))

  @property
  def phase(self) -> int:
    return int(self.raw(0).phase)

  def hand_def_ids(self, player: int) -> list[int]:
    my = self.raw(player).my_observation_data
    return [int(my.hand[i].card_def_id) for i in range(int(my.hand_count))]

  def deck_def_ids_top_first(self, player: int) -> list[int]:
    priv = self.raw(player).critic_privileged
    my = self.raw(player).my_observation_data
    return [int(priv.self_deck[i].card_def_id) for i in range(int(my.deck_count))]

  def counts(self, player: int) -> dict:
    my = self.raw(player).my_observation_data
    return {
        "hand": int(my.hand_count),
        "deck": int(my.deck_count),
        "ikz_pile": int(my.ikz_pile_count),
        "selection": int(my.selection_count),
        "has_ikz_token": bool(my.has_ikz_token),
    }

  def leader(self, player: int) -> dict:
    leader = self.raw(player).my_observation_data.leader
    return {
        "def_id": int(leader.card_def_id),
        "atk": int(leader.cur_stats.cur_atk),
        "hp": int(leader.cur_stats.cur_hp),
        "tapped": bool(leader.tap_state.tapped),
    }

  def gate(self, player: int) -> dict:
    gate = self.raw(player).my_observation_data.gate
    return {"def_id": int(gate.card_def_id), "tapped": bool(gate.tap_state.tapped)}

  def legal_actions(self, player: int) -> list[tuple[int, int, int, int]]:
    mask = self.raw(player).action_mask
    n = int(mask.legal_action_count)
    return [
        (
            int(mask.legal_primary[i]),
            int(mask.legal_sub1[i]),
            int(mask.legal_sub2[i]),
            int(mask.legal_sub3[i]),
        )
        for i in range(n)
    ]

  def step(self, action4) -> None:
    actions = {
        agent: np.asarray(action4, dtype=np.int32)
        for agent in self.env.possible_agents
    }
    self.env.step(actions)

  def reset(self, seed: int) -> None:
    self.env.reset(seed=seed)

  def reset_with_decks(self, seed: int, deck0, deck1) -> None:
    self.env.reset_with_decks(seed=seed, player_decks=(deck0, deck1))

  def rewards(self) -> list[float]:
    return [float(self.env._rewards[i]) for i in range(2)]

  def dones(self) -> tuple[bool, bool]:
    return bool(self.env._terminals.all()), bool(self.env._truncations.all())

  def close(self):
    self.env.close()


@pytest.fixture()
def make_cref(native_pool):
  refs = []

  def factory(seed: int, deck_pool=native_pool):
    ref = CRef(seed, deck_pool=deck_pool)
    refs.append(ref)
    return ref

  yield factory
  for ref in refs:
    ref.close()


def jax_hand_def_ids(state, player: int) -> list[int]:
  import numpy as np

  from azuki_jax.constants import Zone

  zone = np.asarray(state.zone[player])
  zpos = np.asarray(state.zpos[player])
  ids = np.asarray(state.def_id[player])
  in_hand = zone == int(Zone.HAND)
  order = np.argsort(zpos[in_hand])
  return [int(x) for x in ids[in_hand][order]]


def jax_deck_def_ids_top_first(state, player: int) -> list[int]:
  import numpy as np

  from azuki_jax.constants import Zone

  zone = np.asarray(state.zone[player])
  zpos = np.asarray(state.zpos[player])
  ids = np.asarray(state.def_id[player])
  in_deck = zone == int(Zone.DECK)
  order = np.argsort(-zpos[in_deck])
  return [int(x) for x in ids[in_deck][order]]
