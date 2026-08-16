from __future__ import annotations

from types import SimpleNamespace
import unittest

from deck_building import GATE_CARD_TYPE, LEADER_CARD_TYPE, MAIN_CARD_TYPES, build_deck_build_catalog
from league_promotion import PromotionGameRecord, build_reference_schedule
from native_reference_eval import (
  assign_uniform_candidate_leaders,
  multiset_jaccard,
  summarize_native_reference_result,
)
from training_deck_pool import load_training_deck_labels, load_training_deck_pool


class NativeReferenceMetricTests(unittest.TestCase):
  def test_uniform_reference_seeds_pair_every_context_across_both_seats(self) -> None:
    pool = load_training_deck_pool()
    catalog = build_deck_build_catalog(pool)
    schedules = []
    for seed_index, seed in enumerate((11, 13)):
      schedules.extend(
        assign_uniform_candidate_leaders(
          build_reference_schedule(
            [0],
            catalog.records_by_code,
            base_seed=seed,
            schedule_id=f"seed{seed_index:02d}",
          ),
          catalog,
          seed_index=seed_index,
        )
      )
    contexts = set()
    for game in schedules:
      candidate_gate = game.gate0 if game.candidate_seat == 0 else game.gate1
      candidate_leader = game.leader0 if game.candidate_seat == 0 else game.leader1
      reference_leader = game.leader1 if game.candidate_seat == 0 else game.leader0
      contexts.add((candidate_gate, candidate_leader, game.candidate_seat))
      gate_record = catalog.records_by_def_id[candidate_gate]
      valid_leaders = set(catalog.leader_def_ids_by_element[gate_record.element])
      self.assertIn(candidate_leader, valid_leaders)
      self.assertIn(reference_leader, valid_leaders)
      self.assertNotEqual(candidate_leader, reference_leader)
      self.assertIn("uniform-context-v1", game.schedule_version)
    self.assertEqual(len(schedules), 32)
    self.assertEqual(len(contexts), 32)

  def test_multiset_jaccard_counts_copies(self) -> None:
    from collections import Counter

    self.assertAlmostEqual(
      multiset_jaccard(Counter({"a": 4, "b": 1}), Counter({"a": 2, "c": 2})),
      2 / 7,
    )

  def test_summarizes_candidate_deck_and_battle_records(self) -> None:
    pool = load_training_deck_pool()
    labels = load_training_deck_labels()
    catalog = build_deck_build_catalog(pool)
    main_ids = []
    gate_id = -1
    leader_id = -1
    for card_code, quantity in pool[0]:
      record = catalog.records_by_code[card_code]
      if record.card_type in MAIN_CARD_TYPES:
        main_ids.extend([record.card_def_id] * int(quantity))
      elif record.card_type == GATE_CARD_TYPE:
        gate_id = record.card_def_id
      elif record.card_type == LEADER_CARD_TYPE:
        leader_id = record.card_def_id
    self.assertEqual(len(main_ids), 50)
    self.assertGreaterEqual(gate_id, 0)
    self.assertGreaterEqual(leader_id, 0)

    promotion_records = []
    raw_records = []
    for index, candidate_seat in enumerate((0, 1)):
      game_id = f"game-{index}"
      promotion_records.append(
        PromotionGameRecord(
          game_id=game_id,
          block_id=game_id,
          phase="reference",
          opponent_id="reference:0",
          seed=100 + index,
          candidate_seat=candidate_seat,
          candidate_gate=gate_id,
          opponent_gate=gate_id,
          winner_seat=candidate_seat if index == 0 else 1 - candidate_seat,
          steps=80,
          end_reason="gameover",
        )
      )
      candidate = {
        "gate": gate_id,
        "leader": leader_id,
        "main": list(main_ids),
        "attack_rate": 0.2,
        "spell_rate": 0.1,
        "weapon_rate": 0.05,
        "portal_rate": 0.02,
        "play_entity_rate": 0.4,
        "noop_rate": 0.1,
        "ability_rate": 0.03,
        "leader_health": 0.5,
      }
      reference = dict(candidate)
      players = [reference, reference]
      players[candidate_seat] = candidate
      raw_records.append(
        {
          "game_id": game_id,
          "candidate_seat": candidate_seat,
          "seed": 100 + index,
          "starting_player": candidate_seat,
          "ref_deck_index": 0,
          "episode_length": 75.0,
          "players": players,
        }
      )

    result = SimpleNamespace(
      records=tuple(promotion_records),
      raw_records=tuple(raw_records),
    )
    games, summary = summarize_native_reference_result(
      result=result,
      catalog=catalog,
      pool=pool,
      reference_labels=labels,
      training_reference_indices=tuple(range(0, len(pool), 2)),
      holdout_reference_indices=tuple(range(1, len(pool), 2)),
    )
    self.assertEqual(len(games), 2)
    self.assertEqual(summary["episodes"], 2)
    self.assertAlmostEqual(summary["score"], 0.5)
    self.assertEqual(summary["deck_metrics"]["distinct_main_decks"], 1)
    self.assertAlmostEqual(
      summary["deck_metrics"]["nearest_train_reference_jaccard_mean"], 1.0
    )
    self.assertAlmostEqual(summary["battle_metrics"]["attack_rate_mean"], 0.2)
    self.assertEqual(len(summary["by_candidate_leader"]), 1)
    self.assertEqual(len(summary["by_candidate_context"]), 1)


if __name__ == "__main__":
  unittest.main()
