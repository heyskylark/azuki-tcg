from __future__ import annotations

import unittest
from dataclasses import dataclass

from league_promotion import (
  CONFIRMATION_PHASE,
  GATE_CODES,
  PromotionGameRecord,
  PromotionThresholds,
  SCREEN_PHASE,
  build_panel_schedule,
  build_reference_schedule,
  compare_panel_records,
  compare_reference_records,
  decide_promotion,
  screen_passes,
  summarize_panel,
)
from league_promotion_store import promotion_record_from_dict


@dataclass(frozen=True)
class _Record:
  card_def_id: int
  element: str


def _catalog():
  elements = (
    "LIGHTNING", "LIGHTNING", "FIRE", "FIRE",
    "EARTH", "EARTH", "WATER", "WATER",
  )
  return {
    code: _Record(card_def_id=index + 10, element=elements[index])
    for index, code in enumerate(GATE_CODES)
  }


def _leaders():
  return {
    "LIGHTNING": (100, 101),
    "FIRE": (102, 103),
    "EARTH": (104, 105),
    "WATER": (106, 107),
  }


def _result(spec, *, candidate_wins: bool, end_reason: str = "gameover"):
  winner = spec.candidate_seat if candidate_wins else 1 - spec.candidate_seat
  candidate_gate = spec.gate0 if spec.candidate_seat == 0 else spec.gate1
  opponent_gate = spec.gate1 if spec.candidate_seat == 0 else spec.gate0
  candidate_leader = spec.leader0 if spec.candidate_seat == 0 else spec.leader1
  opponent_leader = spec.leader1 if spec.candidate_seat == 0 else spec.leader0
  return PromotionGameRecord(
    game_id=spec.game_id,
    block_id=spec.block_id,
    phase=spec.phase,
    opponent_id=spec.opponent_id,
    seed=spec.seed,
    candidate_seat=spec.candidate_seat,
    candidate_gate=candidate_gate,
    opponent_gate=opponent_gate,
    winner_seat=winner,
    steps=100,
    end_reason=end_reason,
    schedule_version=spec.schedule_version,
    reference_seat=spec.reference_seat,
    reference_deck_index=spec.reference_deck_index,
    candidate_leader=candidate_leader,
    opponent_leader=opponent_leader,
  )


class PromotionScheduleTests(unittest.TestCase):
  def test_uniform_record_cache_round_trip_retains_leaders(self):
    spec = build_panel_schedule(
      ["p1"],
      _catalog(),
      base_seed=7,
      include_confirmation=False,
      leader_ids_by_element=_leaders(),
    )[0]
    original = _result(spec, candidate_wins=True)
    restored = promotion_record_from_dict(original.to_dict())
    self.assertEqual(restored.candidate_leader, original.candidate_leader)
    self.assertEqual(restored.opponent_leader, original.opponent_leader)
    self.assertEqual(restored.to_dict(), original.to_dict())

  def test_screen_is_seat_and_gate_balanced(self):
    games = build_panel_schedule(["p1"], _catalog(), base_seed=7, include_confirmation=False)
    self.assertEqual(len(games), 16)
    self.assertEqual({game.phase for game in games}, {SCREEN_PHASE})
    self.assertEqual(sum(game.candidate_seat == 0 for game in games), 8)
    self.assertEqual(sum(game.candidate_seat == 1 for game in games), 8)
    self.assertTrue(all(game.leader0 == -1 and game.leader1 == -1 for game in games))

    by_block = {}
    for game in games:
      by_block.setdefault(game.block_id, []).append(game)
    self.assertEqual(len(by_block), 8)
    for block in by_block.values():
      self.assertEqual({game.candidate_seat for game in block}, {0, 1})
      self.assertEqual(len({game.seed for game in block}), 1)
      self.assertEqual(len({game.gate0 for game in block}), 1)
      self.assertTrue(all(game.gate0 == game.gate1 for game in block))

  def test_confirmation_balances_every_gate_across_both_seats(self):
    games = build_panel_schedule(["p1"], _catalog(), base_seed=7, include_confirmation=True)
    confirm = [game for game in games if game.phase == CONFIRMATION_PHASE]
    self.assertEqual(len(confirm), 16)
    gate_seats = {(gate_id, seat): 0 for gate_id in range(10, 18) for seat in (0, 1)}
    for game in confirm:
      candidate_gate = game.gate0 if game.candidate_seat == 0 else game.gate1
      gate_seats[(candidate_gate, game.candidate_seat)] += 1
    self.assertTrue(all(count == 1 for count in gate_seats.values()))

  def test_uniform_screen_pairs_all_sixteen_gate_leader_contexts(self):
    games = build_panel_schedule(
      ["p1"],
      _catalog(),
      base_seed=7,
      include_confirmation=False,
      leader_ids_by_element=_leaders(),
    )
    self.assertEqual(len(games), 32)
    self.assertTrue(all(game.schedule_version == "paired-context-v2" for game in games))
    contexts = {}
    for game in games:
      self.assertEqual(game.gate0, game.gate1)
      self.assertEqual(game.leader0, game.leader1)
      contexts.setdefault((game.gate0, game.leader0), []).append(game)
    self.assertEqual(len(contexts), 16)
    for block in contexts.values():
      self.assertEqual({game.candidate_seat for game in block}, {0, 1})
      self.assertEqual(len({game.seed for game in block}), 1)

  def test_uniform_confirmation_covers_every_context_once_for_candidate(self):
    games = build_panel_schedule(
      ["p1"],
      _catalog(),
      base_seed=7,
      include_confirmation=True,
      leader_ids_by_element=_leaders(),
    )
    confirm = [game for game in games if game.phase == CONFIRMATION_PHASE]
    self.assertEqual(len(confirm), 16)
    candidate_contexts = []
    for game in confirm:
      if game.candidate_seat == 0:
        candidate_contexts.append((game.gate0, game.leader0))
      else:
        candidate_contexts.append((game.gate1, game.leader1))
    self.assertEqual(len(set(candidate_contexts)), 16)

  def test_uniform_schedule_rejects_missing_sibling_leader(self):
    leaders = _leaders()
    leaders["FIRE"] = (102,)
    with self.assertRaisesRegex(ValueError, "requires two leaders for FIRE"):
      build_panel_schedule(
        ["p1"],
        _catalog(),
        base_seed=7,
        include_confirmation=False,
        leader_ids_by_element=leaders,
      )

  def test_uniform_summary_and_comparison_report_context_axes(self):
    specs = build_panel_schedule(
      ["p1"],
      _catalog(),
      base_seed=7,
      include_confirmation=False,
      leader_ids_by_element=_leaders(),
    )
    candidate = [
      _result(spec, candidate_wins=(index % 2 == 0))
      for index, spec in enumerate(specs)
    ]
    anchor = [_result(spec, candidate_wins=False) for spec in specs]
    summary = summarize_panel(
      candidate,
      thresholds=PromotionThresholds(bootstrap_samples=1_000),
      bootstrap_seed=9,
    )
    comparison = compare_panel_records(
      candidate,
      anchor,
      confidence=0.8,
      samples=1_000,
      seed=11,
    )
    self.assertEqual(len(summary.leader_scores), 8)
    self.assertEqual(len(summary.context_scores), 16)
    self.assertEqual(len(comparison.leader_deltas), 8)
    self.assertEqual(len(comparison.context_deltas), 16)

  def test_reference_schedule_balances_deck_seat_and_gate(self):
    games = build_reference_schedule([0, 2, 4], _catalog(), base_seed=11)
    self.assertEqual(len(games), 3 * 2 * 8)
    for deck_index in (0, 2, 4):
      selected = [game for game in games if game.reference_deck_index == deck_index]
      self.assertEqual(sum(game.candidate_seat == 0 for game in selected), 8)
      self.assertEqual(sum(game.candidate_seat == 1 for game in selected), 8)
      self.assertTrue(all(game.reference_seat == 1 - game.candidate_seat for game in selected))

  def test_uniform_reference_schedule_covers_candidate_contexts_without_more_games(self):
    games = build_reference_schedule(
      [0],
      _catalog(),
      base_seed=11,
      leader_ids_by_element=_leaders(),
    )
    self.assertEqual(len(games), 16)
    contexts = set()
    for game in games:
      candidate_leader = game.leader0 if game.candidate_seat == 0 else game.leader1
      reference_leader = game.leader1 if game.candidate_seat == 0 else game.leader0
      candidate_gate = game.gate0 if game.candidate_seat == 0 else game.gate1
      contexts.add((candidate_gate, candidate_leader))
      gate_code = GATE_CODES[candidate_gate - 10]
      element = _catalog()[gate_code].element
      self.assertIn(reference_leader, _leaders()[element])
      self.assertNotEqual(reference_leader, candidate_leader)
    self.assertEqual(len(contexts), 16)

  def test_uniform_reference_seed_pair_swaps_leaders_between_candidate_seats(self):
    first = build_reference_schedule(
      [0],
      _catalog(),
      base_seed=11,
      schedule_id="seed00",
      leader_ids_by_element=_leaders(),
      leader_assignment_offset=0,
    )
    second = build_reference_schedule(
      [0],
      _catalog(),
      base_seed=12,
      schedule_id="seed01",
      leader_ids_by_element=_leaders(),
      leader_assignment_offset=1,
    )
    by_gate_seat = {}
    for game in first + second:
      candidate_gate = game.gate0 if game.candidate_seat == 0 else game.gate1
      candidate_leader = game.leader0 if game.candidate_seat == 0 else game.leader1
      by_gate_seat.setdefault((candidate_gate, game.candidate_seat), set()).add(
        candidate_leader
      )
    self.assertEqual(len(by_gate_seat), 16)
    self.assertTrue(all(len(leaders) == 2 for leaders in by_gate_seat.values()))

  def test_uniform_reference_rejects_invalid_leader_assignment_offset(self):
    with self.assertRaisesRegex(ValueError, "leader_assignment_offset must be 0 or 1"):
      build_reference_schedule(
        [0],
        _catalog(),
        base_seed=11,
        leader_ids_by_element=_leaders(),
        leader_assignment_offset=2,
      )

  def test_reference_schedule_ids_are_unique_across_seed_namespaces(self):
    first = build_reference_schedule(
      [0], _catalog(), base_seed=11, schedule_id="seed00"
    )
    second = build_reference_schedule(
      [0], _catalog(), base_seed=13, schedule_id="seed01"
    )
    game_ids = [game.game_id for game in first + second]
    self.assertEqual(len(game_ids), len(set(game_ids)))
    self.assertTrue(all(game.schedule_version.endswith("seed00") for game in first))
    self.assertTrue(all(game.schedule_version.endswith("seed01") for game in second))


class PromotionDecisionTests(unittest.TestCase):
  def setUp(self):
    self.thresholds = PromotionThresholds(bootstrap_samples=1_000)
    self.specs = build_panel_schedule(
      ["p1", "p2", "p3", "p4"], _catalog(), base_seed=13, include_confirmation=True
    )

  def test_strong_broad_candidate_passes(self):
    records = []
    by_opponent_count = {}
    for spec in self.specs:
      index = by_opponent_count.get(spec.opponent_id, 0)
      by_opponent_count[spec.opponent_id] = index + 1
      # 20/32 against each opponent, balanced enough to clear all floors.
      records.append(_result(spec, candidate_wins=(index % 8 < 5)))
    decision = decide_promotion(
      records,
      thresholds=self.thresholds,
      reference=None,
      require_reference=False,
      bootstrap_seed=17,
    )
    self.assertTrue(decision.admitted)
    self.assertEqual(decision.route, "standard")

  def test_specialist_fails_matchup_floor(self):
    records = []
    counts = {}
    for spec in self.specs:
      index = counts.get(spec.opponent_id, 0)
      counts[spec.opponent_id] = index + 1
      wins = index < (4 if spec.opponent_id == "p4" else 26)
      records.append(_result(spec, candidate_wins=wins))
    decision = decide_promotion(
      records,
      thresholds=self.thresholds,
      reference=None,
      require_reference=False,
      bootstrap_seed=19,
    )
    self.assertFalse(decision.admitted)
    self.assertIn("opponent_floor_failed", decision.reasons)

  def test_timeout_is_non_waivable(self):
    records = [_result(spec, candidate_wins=True) for spec in self.specs]
    records[0] = _result(self.specs[0], candidate_wins=True, end_reason="timeout")
    decision = decide_promotion(
      records,
      thresholds=self.thresholds,
      reference=None,
      require_reference=False,
      bootstrap_seed=23,
    )
    self.assertFalse(decision.admitted)
    self.assertIn("timeout_rate_regressed", decision.reasons)

  def test_screen_rejects_only_after_complete_blocks(self):
    screen_specs = [spec for spec in self.specs if spec.phase == SCREEN_PHASE]
    records = [_result(spec, candidate_wins=False) for spec in screen_specs]
    summary = summarize_panel(records, thresholds=self.thresholds, bootstrap_seed=29)
    passed, reasons = screen_passes(summary, self.thresholds)
    self.assertFalse(passed)
    self.assertIn("screen_pooled_score_too_low", reasons)

  def test_reference_comparison_uses_matched_game_ids(self):
    specs = build_reference_schedule([0], _catalog(), base_seed=31)
    candidate = [_result(spec, candidate_wins=index < 12) for index, spec in enumerate(specs)]
    anchor = [_result(spec, candidate_wins=index < 8) for index, spec in enumerate(specs)]
    comparison = compare_reference_records(
      candidate,
      anchor,
      confidence=0.8,
      samples=1_000,
      seed=37,
    )
    self.assertEqual(comparison.games, 16)
    self.assertAlmostEqual(comparison.delta, 0.25)

  def test_reference_absolute_floor_cannot_be_waived_by_relative_gain(self):
    panel_records = []
    counts = {}
    for spec in self.specs:
      index = counts.get(spec.opponent_id, 0)
      counts[spec.opponent_id] = index + 1
      panel_records.append(_result(spec, candidate_wins=(index % 8 < 5)))
    reference_specs = build_reference_schedule([0], _catalog(), base_seed=31)
    candidate = [
      _result(spec, candidate_wins=index < 7)
      for index, spec in enumerate(reference_specs)
    ]
    anchor = [
      _result(spec, candidate_wins=index < 4)
      for index, spec in enumerate(reference_specs)
    ]
    comparison = compare_reference_records(
      candidate,
      anchor,
      confidence=0.8,
      samples=1_000,
      seed=37,
    )
    decision = decide_promotion(
      panel_records,
      thresholds=self.thresholds,
      reference=comparison,
      require_reference=True,
      bootstrap_seed=39,
    )
    self.assertFalse(decision.admitted)
    self.assertIn("reference_candidate_floor_failed", decision.reasons)

  def test_anchor_relative_rule_accepts_anchor_equivalent_cyclic_policy(self):
    candidate = []
    counts = {}
    for spec in self.specs:
      index = counts.get(spec.opponent_id, 0)
      counts[spec.opponent_id] = index + 1
      wins = index < (8 if spec.opponent_id == "p4" else 20)
      candidate.append(_result(spec, candidate_wins=wins))
    anchor = list(candidate)
    comparison = compare_panel_records(
      candidate,
      anchor,
      confidence=0.8,
      samples=1_000,
      seed=41,
    )
    decision = decide_promotion(
      candidate,
      thresholds=self.thresholds,
      reference=None,
      require_reference=False,
      bootstrap_seed=43,
      panel_comparison=comparison,
    )
    self.assertTrue(decision.admitted)
    self.assertEqual(decision.route, "standard")
    self.assertAlmostEqual(comparison.delta, 0.0)
    self.assertEqual(min(decision.summary.opponent_scores.values()), 0.25)

  def test_anchor_relative_rule_rejects_matchup_collapse(self):
    anchor = []
    candidate = []
    counts = {}
    for spec in self.specs:
      index = counts.get(spec.opponent_id, 0)
      counts[spec.opponent_id] = index + 1
      anchor.append(_result(spec, candidate_wins=(index < 16)))
      candidate.append(
        _result(
          spec,
          candidate_wins=(index < (4 if spec.opponent_id == "p4" else 20)),
        )
      )
    comparison = compare_panel_records(
      candidate,
      anchor,
      confidence=0.8,
      samples=1_000,
      seed=47,
    )
    decision = decide_promotion(
      candidate,
      thresholds=self.thresholds,
      reference=None,
      require_reference=False,
      bootstrap_seed=53,
      panel_comparison=comparison,
    )
    self.assertFalse(decision.admitted)
    self.assertIn("opponent_catastrophic_floor_failed", decision.reasons)
    self.assertIn("panel_relative_opponent_floor_failed", decision.reasons)


if __name__ == "__main__":
  unittest.main()
