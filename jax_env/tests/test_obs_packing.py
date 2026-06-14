"""L1/L3 obs packing: byte equality of packed TrainingObservationData.

Drives the C env and the JAX engine with identical action sequences on
vanilla decks (test_l2_vanilla driver pattern) and asserts that
azuki_jax.observe.packed_observation_pair produces EXACTLY the bytes the
training wrapper stack produces per agent: observation_to_dict(c_struct)
emulated (azk_puffer.emulation.emulate) into the aligned struct dtype.

The JAX engine does not populate state.recent_actions yet, so the driver
mirrors the C ActionContext history on the host (per-player most-recent-first,
subactions clamped to 0..49) and injects it into the state before packing.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_l2_vanilla import DRIVER_TYPES, VANILLA_DECK


def _leaf_table():
  """[(path, offset, size)] for every leaf of the packed struct dtype."""
  from azuki_jax.observe import STRUCT_DTYPE

  leaves = []

  def walk(dtype, prefix, base):
    if dtype.fields is not None:
      for name, (fdtype, off) in dtype.fields.items():
        walk(fdtype, f"{prefix}/{name}", base + off)
    elif dtype.subdtype is not None:
      sub, shape = dtype.subdtype
      leaves.append((prefix, base, int(np.prod(shape)) * sub.itemsize))
    else:
      leaves.append((prefix, base, dtype.itemsize))

  walk(STRUCT_DTYPE, "", 0)
  return leaves


def _describe_mismatch(c_row: np.ndarray, j_row: np.ndarray, limit=12) -> str:
  diff = np.flatnonzero(c_row != j_row)
  leaves = _leaf_table()
  hits = []
  for off in diff:
    for path, base, size in leaves:
      if base <= off < base + size:
        if not hits or hits[-1][0] != path:
          hits.append((path, base, size))
        break
    else:
      hits.append((f"<padding@{off}>", int(off), 1))
    if len(hits) >= limit:
      break
  lines = [f"{len(diff)} differing bytes; first leaves:"]
  for path, base, size in hits:
    lines.append(
        f"  {path} @ {base}: C={c_row[base:base + size].tolist()}"
        f" JAX={j_row[base:base + size].tolist()}"
    )
  return "\n".join(lines)


def _clamp_sub(value: int) -> int:
  return 0 if value < 0 else (49 if value >= 50 else value)


def _recent_actions_array(history: dict[int, list[tuple[int, int, int, int]]]):
  """state.recent_actions encoding of the C per-player recent history."""
  out = np.zeros((2, 4, 6), np.int16)
  for player in (0, 1):
    recent = list(reversed(history[player]))[:4]
    for k, (act, s1, s2, s3) in enumerate(recent):
      out[player, k] = (
          1, act, _clamp_sub(s1), _clamp_sub(s2), _clamp_sub(s3),
          1 if act == 0 else 0,
      )
  return out


def _strip_unported_leader_rows(obs_dict) -> None:
  """Remove leader-activation rows (type 11, sub1 == 5) from the C mask:
  leaders are not ported yet, so the JAX enumerator intentionally omits them
  (same normalization as the batch tests' comparable())."""
  am = obs_dict["action_mask"]
  la = am["legal_actions"]
  n = int(am["legal_action_count"])
  prim, s1 = la["legal_primary"], la["legal_sub1"]
  keep = [k for k in range(n) if not (prim[k] == 11 and s1[k] == 5)]
  if len(keep) == n:
    return
  for key in ("legal_primary", "legal_sub1", "legal_sub2", "legal_sub3"):
    arr = la[key]
    kept = arr[keep].copy()
    arr[:] = 0
    arr[: len(kept)] = kept
  am["legal_action_count"] = len(keep)
  if not any(prim[k] == 11 for k in range(len(keep))):
    am["primary_action_mask"][11] = False


def _c_packed_pair(cref, strip_leader_rows=False) -> np.ndarray:
  """Reference bytes: observation_to_dict -> emulate, as in training."""
  from azk_puffer.emulation import emulate
  from observation import observation_to_dict

  from azuki_jax.observe import ITEMSIZE, STRUCT_DTYPE

  buf = np.zeros((2, ITEMSIZE), np.uint8)
  struct = buf.view(STRUCT_DTYPE)
  for player in (0, 1):
    obs_dict = observation_to_dict(cref.raw(player))
    if strip_leader_rows:
      _strip_unported_leader_rows(obs_dict)
    emulate(struct[player], obs_dict)
  return buf


@pytest.mark.parametrize("seed", [11, 222])
def test_packed_observation_byte_equality(seed, make_cref):
  _run_byte_equality(seed, make_cref, VANILLA_DECK, DRIVER_TYPES)


# Ability/selection deck: live ability-context + selection blocks must match
# byte-for-byte through confirmation, cost/effect selection, and the
# selection-zone phases (driver includes the ability actions).
ABILITY_DRIVER_TYPES = DRIVER_TYPES | {
    8, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24,
}


def _ability_deck():
  from test_l3_abilities_batch3 import DECK_GROUPS

  return DECK_GROUPS["T"]


@pytest.mark.parametrize("seed", [11, 222])
def test_packed_observation_ability_deck(seed, make_cref):
  _run_byte_equality(
      seed, make_cref, _ability_deck(), ABILITY_DRIVER_TYPES,
      strip_leader_rows=True,
  )


def _run_byte_equality(seed, make_cref, deck, driver_types,
                       strip_leader_rows=False):
  import jax
  import jax.numpy as jnp

  from azuki_jax.engine.step import engine_step, stabilize
  from azuki_jax.env import init_state_with_decks
  from azuki_jax.observe import packed_observation_pair
  from azuki_jax.setup import deck_tables_from_card_lists

  cref = make_cref(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)

  tables = deck_tables_from_card_lists(deck, deck)
  state = stabilize(init_state_with_decks(seed, tables))

  jit_step = jax.jit(engine_step)
  jit_pack = jax.jit(packed_observation_pair)

  history: dict[int, list[tuple[int, int, int, int]]] = {0: [], 1: []}
  rng = np.random.default_rng(seed)
  for step_index in range(400):
    c_pair = _c_packed_pair(cref, strip_leader_rows)
    packed_state = state._replace(
        recent_actions=jnp.asarray(_recent_actions_array(history))
    )
    j_pair = np.asarray(jit_pack(packed_state))
    for player in (0, 1):
      assert np.array_equal(c_pair[player], j_pair[player]), (
          f"step {step_index} player {player} (phase"
          f" {int(state.phase)}, active {int(state.active_player)}):\n"
          + _describe_mismatch(c_pair[player], j_pair[player])
      )

    active = cref.active_player
    driver_rows = [
        row for row in cref.legal_actions(active)
        if row[0] in driver_types and not (row[0] == 11 and row[1] == 5)
    ]
    assert driver_rows, f"step {step_index}: no driver-supported actions"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    history[active].append(action)
    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_terminal, c_trunc = cref.dones()
    if c_terminal or c_trunc:
      # final (post-terminal) observation must also match byte-for-byte
      c_pair = _c_packed_pair(cref, strip_leader_rows)
      packed_state = state._replace(
          recent_actions=jnp.asarray(_recent_actions_array(history))
      )
      j_pair = np.asarray(jit_pack(packed_state))
      for player in (0, 1):
        assert np.array_equal(c_pair[player], j_pair[player]), (
            f"terminal obs player {player}:\n"
            + _describe_mismatch(c_pair[player], j_pair[player])
        )
      break
