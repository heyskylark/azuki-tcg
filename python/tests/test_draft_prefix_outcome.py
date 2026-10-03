from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from draft_prefix_outcome import (
  ARTIFACT_SCHEMA_VERSION,
  FrozenDraftPrefixPredictor,
  PrefixOutcomeRedistributor,
  file_sha256,
  prefix_quartile,
)
from league_training import LeaguePuffeRL
from observation import DECKBUILD_OBSERVATION_CTYPE


def _write_artifact(path: Path) -> None:
  fc1_weight = np.zeros((1, 8), dtype=np.float32)
  fc1_weight[0, 0] = 10.0
  np.savez(
    path,
    schema_version=np.asarray([ARTIFACT_SCHEMA_VERSION], dtype=np.int32),
    gate_ids=np.asarray([3], dtype=np.int16),
    leader_ids=np.asarray([2], dtype=np.int16),
    card_embedding=np.asarray([[0.0], [1.0], [-1.0], [0.5], [0.25]], dtype=np.float32),
    gate_embedding=np.zeros((1, 1), dtype=np.float32),
    leader_embedding=np.zeros((1, 1), dtype=np.float32),
    seat_embedding=np.zeros((2, 1), dtype=np.float32),
    fc1_weight=fc1_weight,
    fc1_bias=np.zeros(1, dtype=np.float32),
    fc2_weight=np.ones((1, 1), dtype=np.float32),
    fc2_bias=np.zeros(1, dtype=np.float32),
  )


class FrozenDraftPrefixPredictorTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()
    self.path = Path(self.temp_dir.name) / "predictor.npz"
    _write_artifact(self.path)

  def tearDown(self) -> None:
    self.temp_dir.cleanup()

  def test_predicts_batched_prefixes_and_verifies_hash(self) -> None:
    predictor = FrozenDraftPrefixPredictor(
      self.path, expected_sha256=file_sha256(self.path)
    )
    cards = np.full((2, 50), -1, dtype=np.int16)
    cards[1, 0] = 1
    predictions = predictor.predict_probability(
      gate_ids=np.asarray([3, 3]),
      leader_ids=np.asarray([2, 2]),
      seats=np.asarray([0, 1]),
      main_card_ids=cards,
      main_counts=np.asarray([0, 1]),
    )
    self.assertAlmostEqual(float(predictions[0]), 0.5, places=6)
    self.assertGreater(float(predictions[1]), float(predictions[0]))
    with self.assertRaisesRegex(ValueError, "hash mismatch"):
      FrozenDraftPrefixPredictor(self.path, expected_sha256="0" * 64)

  def test_rejects_unknown_selected_card(self) -> None:
    predictor = FrozenDraftPrefixPredictor(self.path)
    cards = np.full((1, 50), -1, dtype=np.int16)
    cards[0, 0] = 9
    with self.assertRaisesRegex(ValueError, "unknown card"):
      predictor.predict_probability(
        gate_ids=np.asarray([3]),
        leader_ids=np.asarray([2]),
        seats=np.asarray([0]),
        main_card_ids=cards,
        main_counts=np.asarray([1]),
      )

  def test_redistribution_telescopes_at_terminal_and_truncation(self) -> None:
    predictor = FrozenDraftPrefixPredictor(self.path)
    redistributor = PrefixOutcomeRedistributor(
      predictor, total_agents=2, coefficient=1.0
    )
    cards = np.full((1, 50), -1, dtype=np.int16)

    def step(*, count: int, done: bool = False, terminal: bool = False):
      return redistributor.step(
        agent_ids=np.asarray([0]),
        episode_ids=np.asarray([7]),
        trainable=np.asarray([True]),
        done=np.asarray([done]),
        terminal=np.asarray([terminal]),
        modes=np.asarray([2 if count < 50 else 0]),
        gate_ids=np.asarray([3]),
        leader_ids=np.asarray([2]),
        main_card_ids=cards,
        main_counts=np.asarray([count]),
      )[0]

    self.assertEqual(float(step(count=0)), 0.0)
    cards[0, 0] = 1
    first = float(step(count=1))
    cards[0, 1:50] = 4
    final = float(step(count=50))
    residual = float(step(count=0, done=True, terminal=True))
    self.assertGreater(first, 0.0)
    self.assertAlmostEqual(first + final + residual, 0.0, places=6)
    metrics = redistributor.window_metrics()
    self.assertEqual(metrics["completed"], 1.0)
    self.assertLess(metrics["telescope_abs_max"], 1e-6)

    cards.fill(-1)
    self.assertEqual(float(step(count=0)), 0.0)
    cards[0, 0] = 1
    delta = float(step(count=1))
    correction = float(step(count=0, done=True, terminal=False))
    self.assertAlmostEqual(delta + correction, 0.0, places=6)
    self.assertEqual(redistributor.window_metrics()["truncated"], 1.0)

  def test_quartile_boundaries(self) -> None:
    self.assertEqual(prefix_quartile(0), "q1_00_12")
    self.assertEqual(prefix_quartile(12), "q1_00_12")
    self.assertEqual(prefix_quartile(13), "q2_13_25")
    self.assertEqual(prefix_quartile(25), "q2_13_25")
    self.assertEqual(prefix_quartile(26), "q3_26_37")
    self.assertEqual(prefix_quartile(38), "q4_38_50")
    self.assertEqual(prefix_quartile(50), "q4_38_50")

  def test_league_decoder_applies_exact_terminal_residual(self) -> None:
    predictor = FrozenDraftPrefixPredictor(self.path)
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._prefix_outcome_enabled = True
    trainer._prefix_outcome_layout = None
    trainer._prefix_outcome_inference_seconds = 0.0
    trainer._prefix_outcome_redistributor = PrefixOutcomeRedistributor(
      predictor, total_agents=2
    )
    trainer._agents_per_env = 2
    trainer._env_episode_ids = np.asarray([11], dtype=np.int64)

    structured = np.zeros(2, dtype=np.dtype(DECKBUILD_OBSERVATION_CTYPE))
    structured["deck_context"]["mode"] = 2
    structured["deck_context"]["gate_card_def_id"] = 3
    structured["deck_context"]["leader_card_def_id"] = 2
    structured["deck_context"]["main_card_def_ids"] = -1
    observations = torch.from_numpy(structured.view(np.uint8).reshape(2, -1))

    def decode(*, done: bool = False, terminal: bool = False):
      return trainer._compute_prefix_outcome_redistribution(
        observations_cpu=observations,
        env_id_np=np.asarray([0, 1]),
        trainable_rows_np=np.asarray([True, False]),
        done_mask=np.asarray([done, done]),
        terminal_mask=np.asarray([terminal, terminal]),
      )[0]

    self.assertEqual(float(decode()), 0.0)
    structured["deck_context"]["main_card_def_ids"][0, 0] = 1
    structured["deck_context"]["main_count"][0] = 1
    delta = float(decode())
    residual = float(decode(done=True, terminal=True))
    self.assertGreater(delta, 0.0)
    self.assertAlmostEqual(delta + residual, 0.0, places=6)


if __name__ == "__main__":
  unittest.main()
