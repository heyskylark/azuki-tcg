from __future__ import annotations

import numpy as np
import pytest
import torch

from action import ActionType
from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
from deck_building import build_deck_build_catalog
from policy.tcg_distribution import TCGLegalActionDistribution
from policy.v2.tcg_sampler import tcg_argmax_logits, tcg_sample_logits
from training_deck_pool import load_training_deck_pool


def _rows(env):
  return env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)


def _legal_actions(row):
  mask = row["action_mask"]
  count = int(mask["legal_action_count"])
  return np.stack([
    mask[key][:count]
    for key in ("legal_primary", "legal_sub1", "legal_sub2", "legal_sub3")
  ], axis=-1).astype(np.int64)


def test_sprout_draft_legality_is_earth_only_and_stops_at_four_copies():
  pool = load_training_deck_pool()
  catalog = build_deck_build_catalog(pool)
  sprout = catalog.records_by_code["STT03-017"].card_def_id
  contexts = [
    (gate, leader)
    for gate in sorted(set(catalog.gate_def_id_population))
    for leader in catalog.leader_def_ids_by_element[catalog.records_by_def_id[gate].element]
  ]
  env = AzukiNativeEnv(
    num_envs=len(contexts), seed=317, deck_pool=pool,
    deck_building=True, evaluation_mode=True, draft_uniform_assignment=True,
  )
  try:
    env.reset_evaluation_games([
      {"env_index": index, "seed": 317 + index,
       "gate0": gate, "leader0": leader, "gate1": gate, "leader1": leader}
      for index, (gate, leader) in enumerate(contexts)
    ])
    for pick in range(9):
      env.actions.fill(0)
      for seat, row in enumerate(_rows(env)):
        context = row["deck_context"]
        count = int(context["candidate_count"])
        if count == 0:
          continue
        candidates = context["candidate_card_def_ids"][:count]
        earth = catalog.records_by_def_id[contexts[seat // 2][0]].element == "EARTH"
        copies = int(np.count_nonzero(context["main_card_def_ids"] == sprout))
        actions = _legal_actions(row)
        if not earth or pick == 8:
          assert sprout not in candidates
          assert copies == (4 if earth else 0)
          if pick < 8:
            env.actions[seat] = actions[0]
          continue
        candidate_index, = np.flatnonzero(candidates == sprout)
        matching = actions[
          (actions[:, 0] == ActionType.DECK_PICK_CARD)
          & (actions[:, 1] == candidate_index)
        ]
        assert len(matching) == 1
        assert copies < 4
        env.actions[seat] = matching[0]
      if pick < 8:
        env.step()
  finally:
    env.close()


@pytest.mark.parametrize("mode", [0, 1], ids=["ramp", "draw"])
def test_policy_selected_sprout_mode_reaches_native_resolution(mode):
  # A legal 50-card Earth deck. Passing turns reaches a cast naturally; no
  # debug state mutations or alternate spell execution path are used.
  deck = (
    ("STT03-001", 1), ("STT03-002", 1), ("IKZ-001", 10),
    ("STT03-017", 4),
    *((f"STT03-{number:03d}", 4) for number in range(3, 14)),
    ("STT03-014", 2),
  )
  pool = (deck,)
  catalog = build_deck_build_catalog(pool)
  sprout = catalog.records_by_code["STT03-017"].card_def_id
  env = AzukiNativeEnv(
    num_envs=1, seed=317, deck_pool=pool, deck_building=True,
    prebuilt_deck_groups=((0,),), prebuilt_probability=1.0,
  )
  try:
    env.reset()
    selected = None
    for _ in range(80):
      env.actions.fill(0)
      for seat, row in enumerate(_rows(env)):
        actions = _legal_actions(row)
        own = row["my_observation_data"]
        for hand_index in range(int(own["hand_count"])):
          if int(own["hand"][hand_index]["card_def_id"]) != sprout:
            continue
          spell_rows = actions[
            (actions[:, 0] == ActionType.PLAY_SPELL_FROM_HAND)
            & (actions[:, 1] == hand_index)
            & (actions[:, 3] == 0)
          ]
          if len(spell_rows) == 0:
            continue
          assert set(spell_rows[:, 2]) == {0, 1}
          target_index, = np.flatnonzero(
            (actions[:, 0] == ActionType.PLAY_SPELL_FROM_HAND)
            & (actions[:, 1] == hand_index)
            & (actions[:, 2] == mode)
            & (actions[:, 3] == 0)
          )
          scores = torch.full((1, len(actions)), -4.0)
          scores[0, target_index] = 4.0
          scores.requires_grad_()
          distribution = TCGLegalActionDistribution(
            legal_action_logits=scores,
            legal_actions=torch.from_numpy(actions).unsqueeze(0),
            legal_action_count=torch.tensor([len(actions)]),
          )
          choice = tcg_argmax_logits(distribution)
          _, logprob, entropy = tcg_sample_logits(distribution, action=choice)
          assert torch.isfinite(logprob).all() and torch.isfinite(entropy).all()
          (-logprob.sum()).backward()
          assert scores.grad is not None and scores.grad[0, target_index] < 0
          env.actions[seat] = choice[0].numpy()
          selected = seat, own.copy()
          break
        if selected is not None:
          break
      env.step()
      if selected is not None:
        break
    assert selected is not None, "No payable Sprout reached within 80 turn decisions"
    seat, before = selected
    after = _rows(env)[seat]["my_observation_data"]
    assert int(after["hand_count"]) == int(before["hand_count"]) - (mode == 0)
    assert int(after["deck_count"]) == int(before["deck_count"]) - (mode == 1)
    assert int(after["ikz_pile_count"]) == int(before["ikz_pile_count"]) - (mode == 0)
    assert int(after["leader"]["cur_stats"]["cur_hp"]) == int(before["leader"]["cur_stats"]["cur_hp"])
    assert np.count_nonzero(after["discard"]["card_def_id"] == sprout) == 1
    assert not env.terminals.any() and not env.truncations.any()
  finally:
    env.close()
