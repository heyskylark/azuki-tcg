from __future__ import annotations

from collections import Counter, defaultdict
import os
import subprocess
import sys

import numpy as np

import binding
from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
from deckbuild_metrics import NativeDeckbuildHelper
from deck_building import GATE_CARD_TYPE, LEADER_CARD_TYPE, MAIN_CARD_TYPES, build_deck_build_catalog
from training_deck_pool import load_training_deck_pool


def _groups_and_signatures():
  pool = load_training_deck_pool()
  catalog = build_deck_build_catalog(pool)
  by_context = defaultdict(list)
  signatures = {}
  for deck_index, deck in enumerate(pool):
    gate = leader = -1
    main = []
    for card_code, quantity in deck:
      record = catalog.records_by_code[card_code]
      if record.card_type == GATE_CARD_TYPE:
        gate = record.card_def_id
      elif record.card_type == LEADER_CARD_TYPE:
        leader = record.card_def_id
      elif record.card_type in MAIN_CARD_TYPES:
        main.extend([record.card_def_id] * int(quantity))
    by_context[(gate, leader)].append(deck_index)
    signatures[deck_index] = (gate, leader, tuple(main))
  groups = tuple(
    tuple(indices)
    for _, indices in sorted(by_context.items())
    if len(indices) >= 2
  )
  assert groups
  return pool, groups, signatures


def _structured(env):
  return env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)


def test_prebuilt_episode_uses_exact_paired_decks_without_draft_or_setup_reward():
  pool, groups, signatures = _groups_and_signatures()
  env = AzukiNativeEnv(
    num_envs=1,
    seed=771,
    deck_pool=pool,
    deck_building=True,
    prebuilt_deck_groups=groups,
    prebuilt_probability=1.0,
    pbrs_mode="discounted",
    reward_telemetry=True,
  )
  probability = env.prebuilt_probability
  try:
    env.reset()
    rows = _structured(env)
    assert [int(row["deck_context"]["mode"]) for row in rows] == [0, 0]
    assert [int(row["deck_context"]["candidate_count"]) for row in rows] == [0, 0]
    assert np.array_equal(env.rewards, np.zeros(2, dtype=np.float32))
    assert np.array_equal(env.terminal_rewards, np.zeros(2, dtype=np.float32))
    assert np.array_equal(env.shaped_rewards, np.zeros(2, dtype=np.float32))

    snapshots = [env.draft_snapshot(seat) for seat in range(2)]
    binding.vec_force_evaluation_truncations(env._handle, [0])
    records = binding.vec_drain_deck_records(env._handle)
    assert len(records) == 1
    record = records[0]
    assert record["prebuilt"] is True
    selected = record["prebuilt_deck_indices"]
    assert len(selected) == 2
    for seat in range(2):
      gate, leader, main = signatures[selected[seat]]
      assert snapshots[seat] == {
        "gate": gate,
        "leader": leader,
        "main_count": 50,
        "main": list(main),
      }
      assert record["players"][seat]["gate"] == gate
      assert record["players"][seat]["leader"] == leader
      assert record["players"][seat]["main"] == list(main)

    probability[0] = 0.0
    env.step()
    rows = _structured(env)
    assert all(int(row["deck_context"]["mode"]) != 0 for row in rows)
  finally:
    env.close()


def test_explicit_evaluation_schedule_bypasses_prebuilt_probability_one():
  pool, groups, signatures = _groups_and_signatures()
  gate, leader, main = signatures[0]
  env = AzukiNativeEnv(
    num_envs=1,
    seed=19,
    deck_pool=pool,
    deck_building=True,
    evaluation_mode=True,
    draft_uniform_assignment=True,
    prebuilt_deck_groups=groups,
    prebuilt_probability=1.0,
  )
  try:
    env.reset_evaluation_games([{
      "env_index": 0, "seed": 98765,
      "gate0": gate, "leader0": leader, "gate1": gate, "leader1": leader,
      "reference_seat": 0, "reference_deck_index": 0,
    }])
    rows = _structured(env)
    assert list(rows["deck_context"]["mode"]) == [0, 2]
    assert list(rows["deck_context"]["main_count"]) == [50, 0]
    assert list(rows[0]["deck_context"]["main_card_def_ids"]) == list(main)
    assert int(rows[1]["deck_context"]["gate_card_def_id"]) == gate
  finally:
    env.close()


def test_prebuilt_records_do_not_report_draft_pick_metrics():
  pool, groups, signatures = _groups_and_signatures()
  selected = groups[0][0]
  gate, leader, main = signatures[selected]
  helper = NativeDeckbuildHelper(deck_pool=pool)
  metrics = helper.process_records(
    [
      {
        "prebuilt": True,
        "prebuilt_deck_indices": [selected, selected],
        "episode_length": 37.0,
        "ref_seat": -1,
        "players": [
          {"gate": gate, "leader": leader, "main": list(main)}
          for _ in range(2)
        ],
      }
    ]
  )
  assert metrics["curriculum/prebuilt_game_fraction"] == 1.0
  assert metrics["curriculum/drafted_game_fraction"] == 0.0
  assert metrics["curriculum/battle_length"] == 37.0
  assert metrics["curriculum/prebuilt_battle_length"] == 37.0
  assert metrics["curriculum/drafted_battle_length"] == 0.0
  assert "deckbuild/picks" not in metrics


def test_prebuilt_sampling_is_deterministic_and_balances_context_before_deck():
  pool, groups, _ = _groups_and_signatures()

  def sample(seed):
    env = AzukiNativeEnv(
      num_envs=64,
      seed=seed,
      deck_pool=pool,
      deck_building=True,
      prebuilt_deck_groups=groups,
      prebuilt_probability=1.0,
    )
    try:
      env.reset()
      binding.vec_force_evaluation_truncations(env._handle, list(range(64)))
      return [
        tuple(record["prebuilt_deck_indices"])
        for record in binding.vec_drain_deck_records(env._handle)
      ]
    finally:
      env.close()

  first = sample(4242)
  assert first == sample(4242)
  deck_to_group = {
    deck_index: group_index
    for group_index, group in enumerate(groups)
    for deck_index in group
  }
  context_counts = Counter(
    deck_to_group[deck_index]
    for pair in first
    for deck_index in pair
  )
  assert set(context_counts) == set(range(len(groups)))
  assert max(context_counts.values()) <= 5 * min(context_counts.values())


def test_one_shot_prebuilt_group_is_not_consumed_before_sampling():
  # A consumed iterator previously produced an empty group and a C SIGFPE.
  probe = """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import binding
from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
from training_deck_pool import load_training_deck_pool
env = AzukiNativeEnv(
    deck_building=True, deck_pool=load_training_deck_pool(),
    prebuilt_deck_groups=(iter([0]),), prebuilt_probability=1.0,
)
try:
    env.reset()
    rows = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
    assert list(rows['deck_context']['mode']) == [0, 0]
    binding.vec_force_evaluation_truncations(env._handle, [0])
    record, = binding.vec_drain_deck_records(env._handle)
    assert record['prebuilt_deck_indices'] == [0, 0]
finally:
    env.close()
"""
  result = subprocess.run(
    [sys.executable, "-c", probe], env=os.environ.copy(),
    capture_output=True, text=True, timeout=20,
  )
  assert result.returncode == 0, result.stdout + result.stderr


def test_sundering_strike_is_draftable_in_every_context_up_to_four_copies():
  pool = load_training_deck_pool()
  catalog = build_deck_build_catalog(pool)
  strike = catalog.records_by_code["AZK01-127"].card_def_id
  contexts = [
    (gate, leader)
    for gate in sorted(set(catalog.gate_def_id_population))
    for leader in catalog.leader_def_ids_by_element[catalog.records_by_def_id[gate].element]
  ]
  env = AzukiNativeEnv(
    num_envs=len(contexts), seed=127, deck_pool=pool,
    deck_building=True, evaluation_mode=True, draft_uniform_assignment=True,
  )
  try:
    env.reset_evaluation_games([
      {"env_index": index, "seed": 127 + index,
       "gate0": gate, "leader0": leader, "gate1": gate, "leader1": leader}
      for index, (gate, leader) in enumerate(contexts)
    ])
    for pick in range(9):
      env.actions.fill(0)
      for seat, row in enumerate(_structured(env)):
        context = row["deck_context"]
        count = int(context["candidate_count"])
        if count == 0:
          continue
        candidates = context["candidate_card_def_ids"][:count]
        copies = int(np.count_nonzero(context["main_card_def_ids"] == strike))
        if pick == 8:
          assert copies == 4
          assert strike not in candidates
          continue
        assert copies < 4
        indices = np.flatnonzero(candidates == strike)
        assert len(indices) == 1, (contexts[seat // 2], copies)
        candidate_index = int(indices[0])
        mask = row["action_mask"]
        legal = slice(int(mask["legal_action_count"]))
        assert np.any(
          (mask["legal_primary"][legal] == 3)
          & (mask["legal_sub1"][legal] == candidate_index)
        )
        env.actions[seat] = (3, candidate_index, 0, 0)
      if pick < 8:
        env.step()
    assert np.all(_structured(env)["deck_context"]["main_count"] == 4)
  finally:
    env.close()
