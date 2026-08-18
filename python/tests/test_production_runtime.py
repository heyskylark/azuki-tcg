from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from production_runtime import (
  CheckpointPolicy,
  apply_process_environment,
  checkpoint_files,
  finalize_checkpoint_state,
  prune_checkpoints,
)
from train import _JsonlLogger, _trusted_parent_source_drift


def test_process_environment_is_declarative_and_rejects_resume_bypasses(monkeypatch) -> None:
  monkeypatch.setenv("AZK_DRAFT_PREFIX_PROBS", "stale")
  applied = apply_process_environment(
    {
      "process_env": {
        "azk_draft_episode_credit_coef": 1.0,
        "azk_draft_prefix_lengths": (0, 1, 2, 4),
        "azk_draft_prefix_probs": "",
      }
    }
  )
  assert applied == {
    "AZK_DRAFT_EPISODE_CREDIT_COEF": "1.0",
    "AZK_DRAFT_PREFIX_LENGTHS": "0,1,2,4",
  }
  assert "AZK_DRAFT_PREFIX_PROBS" not in __import__("os").environ

  with pytest.raises(ValueError, match="escape hatch"):
    apply_process_environment(
      {"process_env": {"azk_resume_allow_source_drift": 1}}
    )


def test_atomic_parent_authorizes_only_hash_verified_initial_source_drift(
  tmp_path: Path,
) -> None:
  model = tmp_path / "model.pt"
  metadata = tmp_path / "model.pt.meta.json"
  trainer = tmp_path / "trainer.pt"
  for path in (model, metadata, trainer):
    path.write_text(path.name, encoding="utf-8")
  manifest = tmp_path / "manifest.json"
  payload = {"schema_version": 2}
  for key, path in (("model", model), ("metadata", metadata), ("trainer", trainer)):
    payload[key] = {
      "path": str(path),
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
  manifest.write_text(json.dumps(payload), encoding="utf-8")

  assert _trusted_parent_source_drift(
    {"parent_manifest": str(manifest)}, model, trainer
  )
  model.write_text("tampered", encoding="utf-8")
  with pytest.raises(ValueError, match="hash mismatch"):
    _trusted_parent_source_drift(
      {"parent_manifest": str(manifest)}, model, trainer
    )


def test_checkpoint_policy_unifies_recovery_evaluation_and_milestones() -> None:
  policy = CheckpointPolicy(
    recovery_interval=100,
    evaluation_interval=250,
    milestone_interval=1000,
    recovery_keep=3,
    evaluation_keep=2,
  )
  assert policy.scheduler_interval == 50
  assert policy.due(100)
  assert policy.due(250)
  assert not policy.due(150)
  assert policy.due(175, done=True)
  assert policy.roles(1000) == ("recovery", "evaluation", "milestone")
  assert policy.retained_updates(
    [100, 200, 250, 300, 400, 500, 1000, 1100],
    protected_updates=[200],
  ) == {200, 500, 1000, 1100}


def test_checkpoint_pruning_keeps_roles_and_active_dependencies(tmp_path: Path) -> None:
  policy = CheckpointPolicy(100, 250, 1000, recovery_keep=2, evaluation_keep=1)
  for update in (100, 200, 250, 300, 400, 500):
    for path in checkpoint_files(tmp_path, update):
      path.write_text(str(update), encoding="utf-8")

  removed = prune_checkpoints(tmp_path, policy, protected_updates={200})
  removed_names = {path.name for path in removed}
  assert "model_azuki_local_000100.pt" in removed_names
  assert "model_azuki_local_000300.pt" in removed_names
  for update in (200, 400, 500):
    assert (tmp_path / f"model_azuki_local_{update:06d}.pt").is_file()


def test_checkpoint_manifest_captures_model_trainer_and_league_state(tmp_path: Path) -> None:
  checkpoint = tmp_path / "model_azuki_local_000250.pt"
  metadata = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
  trainer = tmp_path / "trainer_state_000250.pt"
  league = tmp_path / "live_league.json"
  promotion = tmp_path / "live_promotion.json"
  config = tmp_path / "production.ini"
  for path in (checkpoint, metadata, trainer, league, promotion, config):
    path.write_text(path.name, encoding="utf-8")

  manifest_path = finalize_checkpoint_state(
    checkpoint,
    league_state_path=league,
    promotion_state_path=promotion,
    config_path=config,
    roles=("evaluation",),
  )
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  assert manifest["update"] == 250
  assert manifest["roles"] == ["evaluation"]
  assert len(manifest["artifacts"]) == 5
  assert (tmp_path / "league_state_000250.json").is_file()
  assert (tmp_path / "promotion_state_000250.json").is_file()


def test_compact_jsonl_logger_aggregates_windows(tmp_path: Path) -> None:
  logger = _JsonlLogger(
    {
      "tag": "compact",
      "logging": {
        "interval_updates": 2,
        "metric_patterns": "SPS,epoch,environment/count,environment/timeout",
        "sum_patterns": "environment/count",
        "max_patterns": "environment/timeout",
      },
    },
    tmp_path,
  )
  logger.log(
    {"SPS": 100.0, "epoch": 1, "environment/count": 2, "environment/timeout": 0},
    10,
  )
  logger.log(
    {"SPS": 120.0, "epoch": 2, "environment/count": 3, "environment/timeout": 1},
    20,
  )
  logger.close()

  path = next(tmp_path.glob("*.jsonl"))
  rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
  assert len(rows) == 3
  assert rows[1]["_window_updates"] == 2
  assert rows[1]["SPS"] == pytest.approx(110.0)
  assert rows[1]["epoch"] == 2.0
  assert rows[1]["environment/count"] == 5.0
  assert rows[1]["environment/timeout"] == 1.0
