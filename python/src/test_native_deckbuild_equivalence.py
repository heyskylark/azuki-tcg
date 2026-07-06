"""Draft-semantics parity: native C draft vs DeckBuildingParallelEnv.

Forces the same gates on both paths (env hook + monkeypatch), replays an
identical pick sequence, and asserts per-step equality of draft state:
active player, per-player deck_context (mode, gate/leader ids, pick-order
main log, candidate ids + copy counts) and the legal-action mask rows.
Also checks the policy's packed decode returns the same deck_context values
the struct holds, and that the Python metric helper matches the legacy
wrapper's battle-start metrics for the same drafted decks.

Run: PYTHONPATH=build/python/src:python/src pytest \
    python/src/test_native_deckbuild_equivalence.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("AZK_DEBUG_FORCE_GATE_DEF_IDS", "3,161")  # Surge(L), Stonehaven(E)

import numpy as np

from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import DeckBuildingParallelEnv, build_deck_build_catalog
from tcg_parallel import AzukiTCGParallel
from training_deck_pool import load_training_deck_pool

FORCED_GATES = (3, 161)
MAX_DECK_SIZE = 50


def _make_legacy(pool):
  env = AzukiTCGParallel(seed=123, deck_pool=pool)
  wrapper = DeckBuildingParallelEnv(env, deck_pool=pool, seed=123)
  calls = {"n": 0}

  def forced_gate():
    value = FORCED_GATES[calls["n"] % 2]
    calls["n"] += 1
    return int(value)

  wrapper._sample_gate_def_id = forced_gate
  return wrapper


def test_native_draft_matches_legacy_wrapper():
  pool = load_training_deck_pool()
  legacy = _make_legacy(pool)
  legacy_obs, _ = legacy.reset(seed=99)

  native = AzukiNativeEnv(num_envs=1, deck_pool=pool, seed=7, deck_building=True)
  native.reset(seed=7)
  view = native.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(native.num_agents)

  rng = np.random.default_rng(2024)
  draft_steps = 0
  while legacy._building:
    active = legacy._active_player_index
    n_active_view = view[active]["deck_context"]
    l_ctx = legacy_obs[active]["deck_context"]

    # active player must agree (native exposes it via who has legal actions)
    native_active = [
      a for a in range(2) if view[a]["action_mask"]["legal_action_count"] > 0
    ]
    assert native_active == [active], (draft_steps, native_active, active)

    for agent in range(2):
      n_ctx = view[agent]["deck_context"]
      ctx = legacy_obs[agent]["deck_context"]
      assert int(n_ctx["mode"]) == int(ctx["mode"]), (draft_steps, agent)
      assert int(n_ctx["gate_card_def_id"]) == int(ctx["gate_card_def_id"])
      assert int(n_ctx["leader_card_def_id"]) == int(ctx["leader_card_def_id"])
      assert int(n_ctx["main_count"]) == int(ctx["main_count"])
      np.testing.assert_array_equal(
        np.asarray(n_ctx["main_card_def_ids"], dtype=np.int16),
        np.asarray(ctx["main_card_def_ids"], dtype=np.int16),
      )
      assert int(n_ctx["candidate_count"]) == int(ctx["candidate_count"]), (
        draft_steps, agent,
      )
      count = int(ctx["candidate_count"])
      if count:
        np.testing.assert_array_equal(
          np.asarray(n_ctx["candidate_card_def_ids"][:count], dtype=np.int16),
          np.asarray(ctx["candidate_card_def_ids"][:count], dtype=np.int16),
        )
        np.testing.assert_array_equal(
          np.asarray(n_ctx["candidate_copy_counts"][:count], dtype=np.uint8),
          np.asarray(ctx["candidate_copy_counts"][:count], dtype=np.uint8),
        )
      n_mask = view[agent]["action_mask"]
      l_mask = legacy_obs[agent]["action_mask"]
      assert int(n_mask["legal_action_count"]) == int(l_mask["legal_action_count"])
      rows = int(l_mask["legal_action_count"])
      if rows:
        legal = l_mask["legal_actions"]
        np.testing.assert_array_equal(
          np.asarray(n_mask["legal_primary"][:rows], dtype=np.int16),
          np.asarray(legal["legal_primary"][:rows], dtype=np.int16),
        )
        np.testing.assert_array_equal(
          np.asarray(n_mask["legal_sub1"][:rows], dtype=np.int16),
          np.asarray(legal["legal_sub1"][:rows], dtype=np.int16),
        )

    count = int(l_ctx["candidate_count"])
    pick = int(rng.integers(0, count))
    action = np.array([3, pick, 0, 0], dtype=np.int32)
    legacy_obs, _, _, _, _ = legacy.step({active: action})

    native.actions[:] = 0
    native.actions[active] = action
    native.step()
    draft_steps += 1
    assert draft_steps < 210, "draft did not terminate"

  assert draft_steps == 102  # 51 picks x 2 players, strict alternation

  # First battle observation: deck_context must agree (mode 0, full log, no
  # candidates) and the native mask must not expose DECK_PICK_CARD.
  for agent in range(2):
    n_ctx = view[agent]["deck_context"]
    ctx = legacy_obs[agent]["deck_context"]
    assert int(n_ctx["mode"]) == 0 and int(ctx["mode"]) == 0
    assert int(n_ctx["main_count"]) == MAX_DECK_SIZE
    np.testing.assert_array_equal(
      np.asarray(n_ctx["main_card_def_ids"], dtype=np.int16),
      np.asarray(ctx["main_card_def_ids"], dtype=np.int16),
    )
    assert int(n_ctx["candidate_count"]) == 0
    # privileged decks sanitized on the native battle path
    priv = view[agent]["critic_privileged"]
    assert int(np.max(priv["self_deck"]["card_def_id"])) == -1
    assert int(np.max(priv["opponent_deck"]["card_def_id"])) == -1

  native.close()
  legacy.close()


def test_policy_packed_decode_reads_deck_context():
  import torch
  from policy.v2.tcg_policy import _build_packed_specs, TCG

  pool = load_training_deck_pool()
  native = AzukiNativeEnv(num_envs=1, deck_pool=pool, seed=3, deck_building=True)
  native.reset(seed=3)
  view = native.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(native.num_agents)

  specs = _build_packed_specs(NATIVE_DECKBUILD_OBS_DTYPE)
  obs_u8 = torch.as_tensor(
    native.observations.reshape(native.num_agents, -1).copy()
  )
  dc_spec = specs["deck_context"]
  for agent in range(2):
    row = obs_u8[agent : agent + 1]
    for field in (
      "mode", "gate_card_def_id", "leader_card_def_id", "main_count",
      "candidate_count",
    ):
      decoded = int(dc_spec[field].extract(row).reshape(-1)[0].item())
      assert decoded == int(view[agent]["deck_context"][field]), field
    decoded_main = dc_spec["main_card_def_ids"].extract(row).reshape(-1).numpy()
    np.testing.assert_array_equal(
      decoded_main.astype(np.int16),
      np.asarray(view[agent]["deck_context"]["main_card_def_ids"], dtype=np.int16),
    )
    count = int(view[agent]["deck_context"]["candidate_count"])
    if count:
      decoded_c = dc_spec["candidate_card_def_ids"].extract(row).reshape(-1).numpy()
      np.testing.assert_array_equal(
        decoded_c[:count].astype(np.int16),
        np.asarray(
          view[agent]["deck_context"]["candidate_card_def_ids"][:count],
          dtype=np.int16,
        ),
      )
  native.close()


def test_metric_helper_matches_legacy_battle_start_metrics():
  pool = load_training_deck_pool()
  legacy = _make_legacy(pool)
  legacy_obs, _ = legacy.reset(seed=42)
  rng = np.random.default_rng(7)
  transition_infos = None
  while legacy._building:
    active = legacy._active_player_index
    count = int(legacy_obs[active]["deck_context"]["candidate_count"])
    action = np.array([3, int(rng.integers(0, count)), 0, 0], dtype=np.int32)
    legacy_obs, _, _, _, infos = legacy.step({active: action})
    if not legacy._building:
      transition_infos = infos

  from deckbuild_metrics import NativeDeckbuildHelper

  helper = NativeDeckbuildHelper(deck_pool=pool)
  record = {
    "seed": 0,
    "episode_length": 0.0,
    "players": [
      {
        "gate": legacy._states[i].gate_card_def_id,
        "leader": legacy._states[i].leader_card_def_id,
        "main": list(legacy._states[i].main_card_def_ids),
        "win": 0.0,
      }
      for i in range(2)
    ],
  }
  players = record["players"]
  for idx in (0, 1):
    player = dict(players[idx])
    player["episode_length"] = 0.0
    mine = helper._player_metrics(player, players[1 - idx])
    legacy_metrics = {
      k: v
      for k, v in transition_infos[idx].items()
      if k.startswith("deckbuild/") or k.startswith("azk_step_deckbuild")
      or k.startswith("deckbuild_gatecard")
    }
    for key, expected in legacy_metrics.items():
      assert key in mine, f"missing metric {key}"
      assert abs(float(mine[key]) - float(expected)) < 1e-9, (
        key, mine[key], expected,
      )
  legacy.close()


if __name__ == "__main__":
  test_native_draft_matches_legacy_wrapper()
  test_policy_packed_decode_reads_deck_context()
  test_metric_helper_matches_legacy_battle_start_metrics()
  print("all parity checks passed")
