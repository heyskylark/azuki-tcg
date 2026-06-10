from __future__ import annotations

import unittest

from sps_autoresearch import (
  _best_result_for_target,
  _extract_metric_value,
  _propose_adaptive_variants,
  _result_meets_target,
  evaluate_action_sanity,
  parse_env_profile_line,
  parse_epoch_log_line,
)


class SPSAutoresearchTests(unittest.TestCase):
  def test_parse_epoch_log_line(self):
    line = (
      "[epoch 8] {'SPS': 468.29, 'epoch': 8, "
      "'performance/env': 1.25, 'environment/0/azk_zero_legal_action_truncation': 0.0}"
    )
    parsed = parse_epoch_log_line(line)
    self.assertIsNotNone(parsed)
    assert parsed is not None
    self.assertEqual(parsed["epoch_index"], 8)
    self.assertEqual(parsed["epoch"], 8)
    self.assertAlmostEqual(parsed["SPS"], 468.29)
    self.assertAlmostEqual(parsed["performance/env"], 1.25)

  def test_parse_env_profile_line(self):
    line = (
      "[EnvProfile] steps=4000 avg_step_us=591.42 avg_tick_us=554.14 "
      "avg_refresh_us=33.18 avg_auto_ticks=4.56 tick_share=0.937 refresh_share=0.056"
    )
    parsed = parse_env_profile_line(line)
    self.assertIsNotNone(parsed)
    assert parsed is not None
    self.assertEqual(parsed["steps"], 4000.0)
    self.assertAlmostEqual(parsed["avg_step_us"], 591.42)
    self.assertAlmostEqual(parsed["tick_share"], 0.937)

  def test_action_sanity_passes_for_balanced_mix(self):
    summary = evaluate_action_sanity(
      {
        "environment/0/azk_zero_legal_action_truncation": 0.0,
        "environment/1/azk_zero_legal_action_truncation": 0.0,
        "environment/0/azk_noop_selected_rate": 0.31,
        "environment/1/azk_noop_selected_rate": 0.28,
        "environment/0/azk_attack_selected_rate": 0.12,
        "environment/1/azk_attack_selected_rate": 0.15,
        "environment/0/azk_play_selected_rate": 0.34,
        "environment/1/azk_play_selected_rate": 0.33,
        "environment/0/azk_ability_selected_rate": 0.08,
        "environment/1/azk_ability_selected_rate": 0.07,
        "environment/0/azk_target_selected_rate": 0.15,
        "environment/1/azk_target_selected_rate": 0.17,
      }
    )
    self.assertTrue(summary["pass"])
    self.assertEqual(summary["reasons"], [])

  def test_action_sanity_fails_for_truncation_and_collapse(self):
    summary = evaluate_action_sanity(
      {
        "environment/0/azk_zero_legal_action_truncation": 0.01,
        "environment/1/azk_zero_legal_action_truncation": 0.0,
        "environment/0/azk_noop_selected_rate": 0.92,
        "environment/1/azk_noop_selected_rate": 0.91,
        "environment/0/azk_attack_selected_rate": 0.02,
        "environment/1/azk_attack_selected_rate": 0.02,
        "environment/0/azk_play_selected_rate": 0.03,
        "environment/1/azk_play_selected_rate": 0.03,
        "environment/0/azk_ability_selected_rate": 0.01,
        "environment/1/azk_ability_selected_rate": 0.01,
        "environment/0/azk_target_selected_rate": 0.01,
        "environment/1/azk_target_selected_rate": 0.01,
      }
    )
    self.assertFalse(summary["pass"])
    self.assertGreaterEqual(len(summary["reasons"]), 3)

  def test_extract_metric_value_nested_path(self):
    payload = {"tail_mean": {"SPS": 468.29}, "resource_summary": {"gpu_mem_used_mb_max": 16744.0}}
    self.assertAlmostEqual(_extract_metric_value(payload, "tail_mean.SPS"), 468.29)
    self.assertAlmostEqual(_extract_metric_value(payload, "resource_summary.gpu_mem_used_mb_max"), 16744.0)
    self.assertIsNone(_extract_metric_value(payload, "tail_mean.missing"))

  def test_result_meets_target_requires_guardrails_when_requested(self):
    result = {
      "status": "ok",
      "tail_mean": {"SPS": 470.0},
      "action_sanity": {"pass": True},
    }
    self.assertTrue(
      _result_meets_target(
        result,
        target_metric="tail_mean.SPS",
        target_value=460.0,
        target_mode="at_least",
        require_guardrails=True,
      )
    )
    result["action_sanity"]["pass"] = False
    self.assertFalse(
      _result_meets_target(
        result,
        target_metric="tail_mean.SPS",
        target_value=460.0,
        target_mode="at_least",
        require_guardrails=True,
      )
    )
    self.assertTrue(
      _result_meets_target(
        result,
        target_metric="tail_mean.SPS",
        target_value=460.0,
        target_mode="at_least",
        require_guardrails=False,
      )
    )

  def test_best_result_for_target_prefers_guardrail_passing_result(self):
    results = [
      {
        "label": "faster_but_bad",
        "status": "ok",
        "tail_mean": {"SPS": 500.0},
        "action_sanity": {"pass": False},
      },
      {
        "label": "slower_but_good",
        "status": "ok",
        "tail_mean": {"SPS": 470.0},
        "action_sanity": {"pass": True},
      },
    ]
    best = _best_result_for_target(
      results,
      target_metric="tail_mean.SPS",
      target_mode="at_least",
      require_guardrails=True,
    )
    self.assertIsNotNone(best)
    assert best is not None
    self.assertEqual(best["label"], "slower_but_good")

  def test_best_result_for_target_uses_metric_when_guardrails_are_optional(self):
    results = [
      {
        "label": "faster_but_bad",
        "status": "ok",
        "tail_mean": {"SPS": 500.0},
        "action_sanity": {"pass": False},
      },
      {
        "label": "slower_but_good",
        "status": "ok",
        "tail_mean": {"SPS": 470.0},
        "action_sanity": {"pass": True},
      },
    ]
    best = _best_result_for_target(
      results,
      target_metric="tail_mean.SPS",
      target_mode="at_least",
      require_guardrails=False,
    )
    self.assertIsNotNone(best)
    assert best is not None
    self.assertEqual(best["label"], "faster_but_bad")

  def test_propose_adaptive_variants_learner_bound_prioritizes_compile_and_precision(self):
    proposals = _propose_adaptive_variants(
      {
        "vec.num_envs": 120,
        "vec.batch_size": 120,
        "vec.num_workers": 4,
        "vec.zero_copy": True,
        "env.direct_parallel": True,
        "policy.legal_action_scorer_use_references": True,
        "train.precision": "bfloat16",
        "train.compile": False,
        "league.enable": False,
      },
      tried_signatures=set(),
      search_space={
        "vec.num_envs": [96, 120],
        "vec.num_workers": [2, 4, 6],
        "vec.zero_copy": [True, False],
        "env.direct_parallel": [True, False],
        "policy.legal_action_scorer_use_references": [True, False],
        "train.precision": ["bfloat16", "float32"],
        "train.compile": [False, True],
        "league.enable": [False, True],
      },
      bound_classification="learner_bound",
      batch_size=3,
    )
    self.assertEqual(len(proposals), 3)
    self.assertEqual(proposals[0].overrides, {"train.compile": True})
    self.assertEqual(proposals[1].overrides, {"train.precision": "float32"})
    self.assertEqual(
      proposals[2].overrides,
      {"policy.legal_action_scorer_use_references": False},
    )


if __name__ == "__main__":
  unittest.main()
