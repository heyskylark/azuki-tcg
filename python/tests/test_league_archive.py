from __future__ import annotations

from dataclasses import asdict
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from league_archive import (
  PayoffStat,
  PromotionArchiveState,
  admit_quality_policy,
  ensure_panel_manifest,
  ensure_production_anchor,
  load_promotion_archive_state,
  save_promotion_archive_state,
)
from league_eval import MatchResult
from league_manager import (
  LeagueManager,
  LeagueManagerConfig,
  preserve_training_rng_state,
  summarize_external_strength_window,
)
from league_promotion import PromotionGameRecord, PromotionGameSpec
from league_promotion_store import checkpoint_sha256, load_records, write_immutable_json
from league_state import LeaguePolicyEntry, LeagueState, register_policy


class ArchiveStateTests(unittest.TestCase):
  def _league_entries(self, root: Path, count: int = 6):
    state = LeagueState()
    entries = []
    for index in range(count):
      path = root / f"model_{index:06d}.pt"
      path.write_bytes(f"checkpoint-{index}".encode())
      entries.append(
        register_policy(
          state,
          checkpoint_path=path,
          created_epoch=index * 100,
          source="checkpoint",
        )
      )
    return state, entries

  def test_panel_is_unique_and_refresh_replaces_at_most_one_member(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      league, entries = self._league_entries(Path(tmpdir))
      archive = PromotionArchiveState()
      ensure_production_anchor(archive, policy_id=entries[0].policy_id, epoch=0)
      admit_quality_policy(
        archive,
        policy_id=entries[1].policy_id,
        epoch=100,
        run_id="r1",
        route="standard",
        max_active=6,
      )
      admit_quality_policy(
        archive,
        policy_id=entries[2].policy_id,
        epoch=200,
        run_id="r2",
        route="standard",
        max_active=6,
      )
      archive.payoff_matrix[entries[0].policy_id] = {
        entries[3].policy_id: PayoffStat(games=16, points=4.0, wins=4, losses=12),
      }

      panel1 = ensure_panel_manifest(
        archive,
        league.policies,
        epoch=0,
        panel_size=4,
        refresh_epochs=600,
        base_seed=42,
        checkpoint_hash=lambda policy_id: f"hash-{policy_id}",
      )
      self.assertIsNotNone(panel1)
      ids1 = [member.policy_id for member in panel1.members]
      self.assertEqual(len(ids1), 4)
      self.assertEqual(len(set(ids1)), 4)
      self.assertEqual(ids1[0], entries[0].policy_id)

      frozen = ensure_panel_manifest(
        archive,
        league.policies,
        epoch=599,
        panel_size=4,
        refresh_epochs=600,
        base_seed=42,
        checkpoint_hash=lambda policy_id: f"hash-{policy_id}",
      )
      self.assertEqual(frozen.version, panel1.version)

      admit_quality_policy(
        archive,
        policy_id=entries[4].policy_id,
        epoch=500,
        run_id="r4",
        route="standard",
        max_active=6,
      )
      panel2 = ensure_panel_manifest(
        archive,
        league.policies,
        epoch=600,
        panel_size=4,
        refresh_epochs=600,
        base_seed=42,
        checkpoint_hash=lambda policy_id: f"hash-{policy_id}",
      )
      ids2 = [member.policy_id for member in panel2.members]
      self.assertEqual(panel2.version, panel1.version + 1)
      self.assertEqual(ids2[0], entries[0].policy_id)
      self.assertLessEqual(len(set(ids1).symmetric_difference(ids2)), 2)

  def test_archive_state_roundtrip_preserves_payoff_stats(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "promotion.json"
      state = PromotionArchiveState(
        production_anchor_policy_id="p1",
        payoff_matrix={"p1": {"p2": PayoffStat(games=2, points=1.5, wins=1, draws=1)}},
      )
      save_promotion_archive_state(path, state)
      loaded = load_promotion_archive_state(path)
      self.assertEqual(loaded.production_anchor_policy_id, "p1")
      self.assertEqual(loaded.payoff_matrix["p1"]["p2"].games, 2)
      self.assertEqual(loaded.payoff_matrix["p1"]["p2"].score, 0.75)

  def test_panel_refresh_retains_bootstrap_hardness_until_live_score_is_harder(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      league, entries = self._league_entries(Path(tmpdir))
      archive = PromotionArchiveState()
      ensure_production_anchor(archive, policy_id=entries[0].policy_id, epoch=0)
      for index in (1, 2):
        admit_quality_policy(
          archive,
          policy_id=entries[index].policy_id,
          epoch=index * 100,
          run_id=f"r{index}",
          route="standard",
          max_active=6,
        )
      archive.payoff_matrix[entries[0].policy_id] = {
        entries[3].policy_id: PayoffStat(games=64, points=19.0, wins=19, losses=45),
      }
      panel1 = ensure_panel_manifest(
        archive,
        league.policies,
        epoch=0,
        panel_size=4,
        refresh_epochs=600,
        base_seed=42,
        checkpoint_hash=lambda policy_id: f"hash-{policy_id}",
      )
      self.assertEqual(panel1.members[2].policy_id, entries[3].policy_id)

      archive.payoff_matrix[entries[0].policy_id] = {
        entries[4].policy_id: PayoffStat(games=16, points=8.0, wins=8, losses=8),
      }
      unchanged = ensure_panel_manifest(
        archive,
        league.policies,
        epoch=600,
        panel_size=4,
        refresh_epochs=600,
        base_seed=42,
        checkpoint_hash=lambda policy_id: f"hash-{policy_id}",
      )
      self.assertEqual(unchanged.version, panel1.version)
      self.assertEqual(unchanged.members[2].policy_id, entries[3].policy_id)

      archive.payoff_matrix[entries[0].policy_id][entries[4].policy_id] = PayoffStat(
        games=16,
        points=3.0,
        wins=3,
        losses=13,
      )
      replaced = ensure_panel_manifest(
        archive,
        league.policies,
        epoch=600,
        panel_size=4,
        refresh_epochs=600,
        base_seed=42,
        checkpoint_hash=lambda policy_id: f"hash-{policy_id}",
      )
      self.assertEqual(replaced.version, panel1.version + 1)
      self.assertIn(entries[4].policy_id, [member.policy_id for member in replaced.members])
      self.assertNotIn(entries[3].policy_id, [member.policy_id for member in replaced.members])


class PromotionStoreTests(unittest.TestCase):
  def test_immutable_json_refuses_changed_payload_and_records_reload(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "artifact.json"
      record = PromotionGameRecord(
        game_id="g0",
        block_id="b0",
        phase="screen",
        opponent_id="p1",
        seed=3,
        candidate_seat=0,
        candidate_gate=4,
        opponent_gate=4,
        winner_seat=0,
        steps=10,
        end_reason="gameover",
      )
      payload = {"games": [record.to_dict()]}
      write_immutable_json(path, payload)
      write_immutable_json(path, payload)
      with self.assertRaises(FileExistsError):
        write_immutable_json(path, {"games": []})
      loaded = load_records(path)
      self.assertEqual(loaded, [record])


class _AllWinEvaluator:
  evaluator_version = "fake-v1"

  def __init__(self):
    self.calls = []

  def evaluate_schedule(self, trainer_args, *, policy_a, policy_b, request, games):
    self.calls.append((policy_a is policy_b, tuple(game.phase for game in games)))
    records = tuple(
      PromotionGameRecord(
        game_id=game.game_id,
        block_id=game.block_id,
        phase=game.phase,
        opponent_id=game.opponent_id,
        seed=game.seed,
        candidate_seat=game.candidate_seat,
        candidate_gate=game.gate0 if game.candidate_seat == 0 else game.gate1,
        opponent_gate=game.gate1 if game.candidate_seat == 0 else game.gate0,
        winner_seat=game.candidate_seat,
        steps=20,
        end_reason="gameover",
        schedule_version=game.schedule_version,
        evaluator_version=self.evaluator_version,
        candidate_leader=(
          game.leader0 if game.candidate_seat == 0 else game.leader1
        ),
        opponent_leader=(
          game.leader1 if game.candidate_seat == 0 else game.leader0
        ),
      )
      for game in games
    )
    return MatchResult(
      wins_a=len(records),
      wins_b=0,
      draws=0,
      episodes=len(records),
      records=records,
    )


class ArchiveManagerShadowTests(unittest.TestCase):
  def test_external_strength_window_is_schedule_matched_and_capped(self) -> None:
    history = [
      {
        "event": "external_strength_observation",
        "schedule_hash": "same",
        "epoch": epoch,
        "candidate_id": f"p{epoch}",
        "candidate_score": score,
        "delta": score - 0.5,
        "paired_lcb": score - 0.1,
      }
      for epoch, score in ((1, 0.4), (2, 0.5), (3, 0.6))
    ]
    history.append({
      "event": "external_strength_observation",
      "schedule_hash": "other",
      "epoch": 4,
      "candidate_id": "other",
      "candidate_score": 1.0,
      "delta": 1.0,
      "paired_lcb": 1.0,
    })
    window = summarize_external_strength_window(
      history,
      {
        "event": "external_strength_observation",
        "schedule_hash": "same",
        "epoch": 5,
        "candidate_id": "p5",
        "candidate_score": 0.7,
        "delta": 0.2,
        "paired_lcb": 0.6,
      },
    )
    self.assertEqual(window["count"], 3)
    self.assertEqual(window["epochs"], [2, 3, 5])
    self.assertEqual(window["candidate_ids"], ["p2", "p3", "p5"])
    self.assertAlmostEqual(window["score_mean"], 0.6)
    self.assertGreater(window["score_slope_per_100_updates"], 0.0)

  def test_evaluation_rng_scope_restores_python_numpy_and_torch(self) -> None:
    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    with preserve_training_rng_state():
      random.random()
      np.random.random()
      torch.rand(32)

    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    self.assertEqual(actual, expected)

  def test_panel_policy_uses_its_checkpoint_policy_config(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      checkpoint = root / "historical.pt"
      checkpoint.write_bytes(b"checkpoint")
      checkpoint.with_suffix(".pt.meta.json").write_text(
        json.dumps(
          {
            "resume_config_fingerprint": {
              "policy_gate_id_embedding_enabled": True,
            }
          }
        ),
        encoding="utf-8",
      )
      manager = LeagueManager(LeagueManagerConfig(enabled=True))
      entry = LeaguePolicyEntry(
        policy_id="historical",
        checkpoint_path=str(checkpoint),
        created_epoch=0,
        created_ts=0.0,
        source="test",
      )
      trainer_args = {
        "train": {"device": "cpu"},
        "policy": {"gate_id_embedding_enabled": False},
      }
      built_args = []
      loaded = []

      policy = manager._load_policy(
        entry=entry,
        vecenv=None,
        trainer_args=trainer_args,
        build_policy_fn=lambda vecenv, args: (
          built_args.append(args) or torch.nn.Linear(1, 1)
        ),
        load_weights_fn=lambda *args, **kwargs: loaded.append((args, kwargs)),
      )

      self.assertIsInstance(policy, torch.nn.Linear)
      self.assertTrue(built_args[0]["policy"]["gate_id_embedding_enabled"])
      self.assertFalse(trainer_args["policy"]["gate_id_embedding_enabled"])
      self.assertEqual(len(loaded), 1)

  def test_reference_anchor_result_is_cached_across_candidates(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      manager = LeagueManager(
        LeagueManagerConfig(
          enabled=True,
          state_path=str(root / "league.json"),
          eval_mode="native_panel",
          promotion_mode="archive_panel",
          promotion_shadow_mode=True,
          promotion_state_path=str(root / "promotion.json"),
          promotion_records_dir=str(root / "records"),
          promotion_require_reference=True,
          promotion_reference_seeds=(11, 13),
          quick_eval_interval=1,
          full_eval_interval=1,
          promotion_bootstrap_samples=100,
        )
      )
      seeds = []
      for index in range(4):
        path = root / f"seed_{index}.pt"
        path.write_bytes(f"seed-{index}".encode())
        seeds.append(path)
      manager.ensure_seed_policies(seeds)
      evaluator = _AllWinEvaluator()
      manager.evaluator = evaluator
      trainer_args = {
        "env": {"deck_pool_path": ".codex/docs/azuki_tcg_decks_final.json"},
        "train": {"device": "cpu", "use_rnn": False},
      }
      policies = []

      for epoch in (1, 2):
        candidate_path = root / f"candidate_{epoch}.pt"
        candidate_path.write_bytes(f"candidate-{epoch}".encode())
        manager.maybe_add_checkpoint(candidate_path, epoch=epoch)
        learner = torch.nn.Linear(1, 1)
        policies.append(learner)
        metrics = manager.maybe_evaluate_and_promote(
          epoch=epoch,
          trainer_args=trainer_args,
          vecenv=None,
          build_policy_fn=lambda vecenv, args: torch.nn.Linear(1, 1),
          load_weights_fn=lambda *args, **kwargs: None,
          learner_policy=learner,
        )
        payload = json.loads(Path(str(metrics["league/eval_raw_record_path"])).read_text())
        self.assertEqual(len(payload["games"]), 416)

      self_controlled_reference_calls = sum(
        is_self and phases[0] == "reference" for is_self, phases in evaluator.calls
      )
      self.assertEqual(self_controlled_reference_calls, 3)
      self.assertEqual(len(evaluator.calls), 23)
      self.assertEqual(len(manager.archive_state.panel_cache), 1)
      self.assertEqual(len(manager.archive_state.reference_cache), 1)
      observations = [
        item
        for item in manager.archive_state.history
        if item.get("event") == "external_strength_observation"
      ]
      self.assertEqual(len(observations), 2)
      self.assertEqual(observations[-1]["window"]["count"], 2)
      self.assertEqual(observations[-1]["window"]["epochs"], [1, 2])
      self.assertEqual(
        observations[-1]["window"]["score_mean"],
        observations[-1]["candidate_score"],
      )

  def test_uniform_assignment_manager_uses_context_panel_and_swapped_reference_seeds(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      manager = LeagueManager(
        LeagueManagerConfig(
          enabled=True,
          state_path=str(root / "league.json"),
          eval_mode="native_panel",
          promotion_mode="archive_panel",
          promotion_shadow_mode=True,
          promotion_state_path=str(root / "promotion.json"),
          promotion_records_dir=str(root / "records"),
          promotion_require_reference=True,
          promotion_reference_seeds=(11, 13),
          quick_eval_interval=1,
          full_eval_interval=1,
          promotion_bootstrap_samples=100,
        )
      )
      seeds = []
      for index in range(4):
        path = root / f"seed_{index}.pt"
        path.write_bytes(f"seed-{index}".encode())
        seeds.append(path)
      manager.ensure_seed_policies(seeds)
      candidate_path = root / "candidate.pt"
      candidate_path.write_bytes(b"candidate")
      manager.maybe_add_checkpoint(candidate_path, epoch=1)
      manager.evaluator = _AllWinEvaluator()

      trainer_args = {
        "env": {
          "deck_pool_path": ".codex/docs/azuki_tcg_decks_final.json",
          "draft_uniform_assignment": True,
        },
        "train": {"device": "cpu", "use_rnn": False},
      }
      metrics = manager.maybe_evaluate_and_promote(
        epoch=1,
        trainer_args=trainer_args,
        vecenv=None,
        build_policy_fn=lambda vecenv, args: torch.nn.Linear(1, 1),
        load_weights_fn=lambda *args, **kwargs: None,
        learner_policy=torch.nn.Linear(1, 1),
      )

      payload = json.loads(Path(str(metrics["league/eval_raw_record_path"])).read_text())
      panel = [game for game in payload["games"] if game["phase"] != "reference"]
      references = [game for game in payload["games"] if game["phase"] == "reference"]
      self.assertEqual(len(panel), 192)
      self.assertEqual(len(references), 288)
      self.assertEqual(
        payload["metadata"]["reference_manifest"]["assignment_contract"],
        "uniform_gate_same_element_leader",
      )
      screen_contexts = {
        (game["candidate_gate"], game["candidate_leader"], game["candidate_seat"])
        for game in panel
        if game["phase"] == "screen"
      }
      self.assertEqual(len(screen_contexts), 32)
      reference_contexts = {
        (game["candidate_gate"], game["candidate_leader"], game["candidate_seat"])
        for game in references
      }
      self.assertEqual(len(reference_contexts), 32)
      self.assertEqual(
        len([key for key in metrics if key.startswith("league/promotion_leader_score/")]),
        8,
      )
      self.assertEqual(
        len([key for key in metrics if key.startswith("league/promotion_context_score/")]),
        16,
      )

      cached_metrics = manager.maybe_evaluate_and_promote(
        epoch=2,
        trainer_args=trainer_args,
        vecenv=None,
        build_policy_fn=lambda vecenv, args: torch.nn.Linear(1, 1),
        load_weights_fn=lambda *args, **kwargs: None,
        learner_policy=torch.nn.Linear(1, 1),
      )
      self.assertEqual(cached_metrics["league/promotion_panel_games"], 192.0)
      self.assertEqual(len(manager.evaluator.calls), 23)

  def test_retrospective_panel_is_evaluation_only(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      checkpoints = []
      for index in range(4):
        path = root / f"historical_{index}.pt"
        path.write_bytes(f"history-{index}".encode())
        checkpoints.append(path)
      panel_path = root / "panel.json"
      panel_path.write_text(
        json.dumps(
          {
            "panel_version": 1,
            "anchor_id": "anchor",
            "schedule_seed": 123,
            "members": [
              {
                "policy_id": "anchor" if index == 0 else f"history_{index}",
                "role": "production_anchor" if index == 0 else "historical",
                "checkpoint_path": str(path),
                "checkpoint_hash": checkpoint_sha256(path),
                "quality_qualified": index in (0, 1),
              }
              for index, path in enumerate(checkpoints)
            ],
          }
        ),
        encoding="utf-8",
      )
      manager = LeagueManager(
        LeagueManagerConfig(
          enabled=True,
          state_path=str(root / "league.json"),
          eval_mode="native_panel",
          promotion_mode="archive_panel",
          promotion_state_path=str(root / "promotion.json"),
          production_anchor_checkpoint=str(checkpoints[0]),
          promotion_bootstrap_panel_path=str(panel_path),
        )
      )

      manager.ensure_seed_policies([])

      self.assertEqual(len(manager.state.policies), 0)
      self.assertEqual(len(manager.opponent_entries_for_training()), 0)
      panel = manager.archive_state.panels[0]
      self.assertEqual(len(panel.members), 4)
      self.assertEqual(
        panel.members[0].policy_id,
        manager.archive_state.production_anchor_policy_id,
      )
      self.assertIsNone(manager.state.champion_policy_id)
      self.assertTrue(all(manager._promotion_entry(item.policy_id) is not None for item in panel.members))

      candidate_path = root / "candidate.pt"
      candidate_path.write_bytes(b"candidate")
      candidate = manager.maybe_add_checkpoint(candidate_path, epoch=1)
      self.assertIsNotNone(candidate)
      self.assertIsNone(manager.state.champion_policy_id)
      self.assertEqual(
        [item.policy_id for item in manager.opponent_entries_for_training()],
        [candidate.policy_id],
      )

  def test_shadow_admission_does_not_mutate_authoritative_league_state(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      config = LeagueManagerConfig(
        enabled=True,
        state_path=str(root / "league.json"),
        eval_mode="native_panel",
        promotion_mode="archive_panel",
        promotion_shadow_mode=True,
        promotion_state_path=str(root / "promotion.json"),
        promotion_records_dir=str(root / "records"),
        promotion_require_reference=False,
        quick_eval_interval=1,
        full_eval_interval=1,
        promotion_bootstrap_samples=100,
      )
      manager = LeagueManager(config)
      seeds = []
      for index in range(4):
        path = root / f"seed_{index}.pt"
        path.write_bytes(f"seed-{index}".encode())
        seeds.append(path)
      manager.ensure_seed_policies(seeds)
      candidate_path = root / "candidate.pt"
      candidate_path.write_bytes(b"candidate")
      candidate = manager.maybe_add_checkpoint(candidate_path, epoch=1)
      self.assertIsNotNone(candidate)
      before = asdict(manager.state)
      manager.evaluator = _AllWinEvaluator()

      metrics = manager.maybe_evaluate_and_promote(
        epoch=1,
        trainer_args={
          "env": {"deck_pool_path": ".codex/docs/azuki_tcg_decks_final.json"},
          "train": {"device": "cpu", "use_rnn": False},
        },
        vecenv=None,
        build_policy_fn=lambda vecenv, args: torch.nn.Linear(1, 1),
        load_weights_fn=lambda *args, **kwargs: None,
        learner_policy=torch.nn.Linear(1, 1),
      )

      self.assertEqual(asdict(manager.state), before)
      self.assertEqual(metrics["league/promotion_decision_admitted"], 1.0)
      self.assertEqual(metrics["league/promotion_accepted"], 0.0)
      self.assertEqual(
        [item.policy_id for item in manager.archive_state.quality_archive if item.active],
        [manager.archive_state.production_anchor_policy_id],
      )
      artifact = Path(str(metrics["league/eval_raw_record_path"]))
      self.assertTrue(artifact.exists())
      payload = json.loads(artifact.read_text())
      self.assertEqual(len(payload["games"]), 128)
      self.assertNotIn(candidate.policy_id, [item["policy_id"] for item in payload["metadata"]["panel"]["members"]])

  def test_archive_admission_does_not_change_training_retention_by_default(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      manager = LeagueManager(
        LeagueManagerConfig(
          enabled=True,
          state_path=str(root / "league.json"),
          eval_mode="native_panel",
          promotion_mode="archive_panel",
          promotion_shadow_mode=False,
          promotion_state_path=str(root / "promotion.json"),
          promotion_records_dir=str(root / "records"),
          promotion_require_reference=False,
          promotion_archive_affects_training_pool=False,
          quick_eval_interval=1,
          full_eval_interval=1,
          promotion_bootstrap_samples=100,
          keep_recent=4,
          keep_mid=0,
          keep_old=1,
        )
      )
      seeds = []
      for index in range(4):
        path = root / f"seed_{index}.pt"
        path.write_bytes(f"seed-{index}".encode())
        seeds.append(path)
      manager.ensure_seed_policies(seeds)

      candidate_path = root / "candidate.pt"
      candidate_path.write_bytes(b"candidate")
      candidate = manager.maybe_add_checkpoint(candidate_path, epoch=1)
      self.assertIsNotNone(candidate)
      manager.evaluator = _AllWinEvaluator()
      metrics = manager.maybe_evaluate_and_promote(
        epoch=1,
        trainer_args={
          "env": {"deck_pool_path": ".codex/docs/azuki_tcg_decks_final.json"},
          "train": {"device": "cpu", "use_rnn": False},
        },
        vecenv=None,
        build_policy_fn=lambda vecenv, args: torch.nn.Linear(1, 1),
        load_weights_fn=lambda *args, **kwargs: None,
        learner_policy=torch.nn.Linear(1, 1),
      )
      self.assertEqual(metrics["league/promotion_accepted"], 1.0)
      self.assertIn(
        candidate.policy_id,
        [item.policy_id for item in manager.archive_state.quality_archive if item.active],
      )

      manager.config.keep_recent = 1
      successor_path = root / "successor.pt"
      successor_path.write_bytes(b"successor")
      manager.maybe_add_checkpoint(successor_path, epoch=2)

      self.assertFalse(manager.state.policies[candidate.policy_id].active)
      self.assertNotIn(
        candidate.policy_id,
        [item.policy_id for item in manager.opponent_entries_for_training()],
      )


if __name__ == "__main__":
  unittest.main()
