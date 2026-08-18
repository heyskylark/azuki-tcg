import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train import (
    RESUME_REWARD_ENV_VARS,
    _apply_saved_reward_env,
    _checkpoint_metadata_path,
    _approved_resume_cfg_mismatch_prefixes,
    _enabled_episode_schedule_completion,
    _extract_completed_episodes_from_mapping,
    _load_model_weights,
    _seed_training_process,
    _peek_resume_env_completed_episodes,
    _restart_lr_schedule_for_remaining_epochs,
    _resume_cfg_mismatches,
    _select_resume_completed_episodes,
    _trainer_shaped_reward_schedule_state,
)


def _draw_rng_sample() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(()))


def test_process_seeding_is_disabled_by_default() -> None:
    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    expected = _draw_rng_sample()

    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    assert _seed_training_process({"seed": 42}) is None
    assert _draw_rng_sample() == expected


def test_process_seeding_reproducibly_seeds_all_rngs() -> None:
    config = {"seed": 42, "seed_process_rngs": True}
    assert _seed_training_process(config) == 42
    expected = _draw_rng_sample()

    _draw_rng_sample()
    assert _seed_training_process(config) == 42
    assert _draw_rng_sample() == expected


class _CachedPolicy(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.cache_invalidations = 0

    def _invalidate_text_feature_table(self) -> None:
        self.cache_invalidations += 1


class _PolicyWrapper(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.policy = _CachedPolicy(value)


def test_random_draft_prefix_configuration_is_resume_fingerprinted() -> None:
    assert {
        "AZK_DRAFT_PREFIX_LENGTHS",
        "AZK_DRAFT_PREFIX_PROBS",
        "AZK_DRAFT_PREFIX_SEED",
    }.issubset(RESUME_REWARD_ENV_VARS)


def test_load_model_weights_invalidates_derived_policy_cache(tmp_path: Path) -> None:
    source = _PolicyWrapper(7.0)
    checkpoint = tmp_path / "model.pt"
    torch.save(source.state_dict(), checkpoint)
    target = _PolicyWrapper(-2.0)

    _load_model_weights(target, checkpoint, device="cpu", strict=True)

    assert target.policy.weight.item() == pytest.approx(7.0)
    assert target.policy.cache_invalidations == 1


def test_extract_completed_episodes_accepts_native_aggregate() -> None:
    assert (
        _extract_completed_episodes_from_mapping(
            {"environment/completed_episodes": 25.83823529411765}
        )
        == 26
    )


def test_extract_completed_episodes_uses_largest_known_counter() -> None:
    assert (
        _extract_completed_episodes_from_mapping(
            {
                "0/azk_completed_episodes": [22.0, 27.0],
                "environment/completed_episodes": 25.4,
            }
        )
        == 27
    )


def test_enabled_episode_schedule_completion_uses_latest_boundary(monkeypatch) -> None:
    monkeypatch.setenv("AZK_MAX_TICKS_CURRICULUM", "1")
    monkeypatch.setenv("AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES", "3")
    monkeypatch.setenv("AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES", "7")
    monkeypatch.setenv("AZK_REWARD_SHAPING_ANNEAL", "1")
    monkeypatch.setenv("AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES", "12")
    monkeypatch.setenv("AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES", "40")

    assert _enabled_episode_schedule_completion() == 52


def test_peek_resume_uses_monotonic_maximum_across_sidecars(tmp_path: Path) -> None:
    model_path = tmp_path / "model_azuki_local_002930.pt"
    model_path.touch()
    trainer_state_path = tmp_path / "trainer_state_002930.pt"
    torch.save(
        {"env_completed_episodes": 200, "global_step": 27_000_000},
        trainer_state_path,
    )
    _checkpoint_metadata_path(model_path).write_text(
        json.dumps({"env_completed_episodes": 242, "global_step": 27_600_000})
    )

    assert (
        _peek_resume_env_completed_episodes(
            model_path,
            trainer_state_path,
            num_envs_hint=1440,
        )
        == 242
    )


def test_legacy_resume_pins_enabled_reward_schedule_at_final_phase(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model_azuki_local_002930.pt"
    model_path.touch()
    _checkpoint_metadata_path(model_path).write_text(
        json.dumps({"global_step": 27_600_000})
    )
    monkeypatch.setenv("AZK_REWARD_SHAPING_ANNEAL", "1")
    monkeypatch.setenv("AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES", "12")
    monkeypatch.setenv("AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES", "40")

    assert (
        _peek_resume_env_completed_episodes(
            model_path,
            None,
            num_envs_hint=1440,
        )
        == 52
    )


def test_resume_progression_cannot_rewind_without_explicit_escape_hatch() -> None:
    assert (
        _select_resume_completed_episodes(
            saved_completed_episodes=242,
            manual_completed_episodes=0,
            allow_schedule_rewind=False,
        )
        == 242
    )
    assert (
        _select_resume_completed_episodes(
            saved_completed_episodes=242,
            manual_completed_episodes=0,
            allow_schedule_rewind=True,
        )
        == 0
    )
    assert (
        _select_resume_completed_episodes(
            saved_completed_episodes=242,
            manual_completed_episodes=300,
            allow_schedule_rewind=False,
        )
        == 300
    )


def test_restart_lr_schedule_uses_only_remaining_epochs() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=0.0)
    optimizer.param_groups[0]["initial_lr"] = 0.003
    trainer = SimpleNamespace(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=977),
        epoch=977,
        total_epochs=2930,
        config={"learning_rate": 0.0003},
    )

    remaining, lrs = _restart_lr_schedule_for_remaining_epochs(trainer)

    assert remaining == 1953
    assert lrs == pytest.approx([0.0003])
    assert trainer.scheduler.T_max == 1953
    for _ in range(1953):
        trainer.optimizer.step()
        trainer.scheduler.step()
    assert trainer.scheduler.get_last_lr() == pytest.approx([0.0], abs=1e-12)


def test_saved_reward_env_is_restored_by_default(monkeypatch) -> None:
    monkeypatch.setenv("AZK_EARLY_TEMPO_BONUS", "0.9")
    monkeypatch.setenv("AZK_DMG_MITIGATION_BONUS", "0.8")

    _apply_saved_reward_env(
        {
            "reward_env": {
                "AZK_EARLY_TEMPO_BONUS": "0.1",
                "AZK_DMG_MITIGATION_BONUS": "",
            }
        }
    )

    assert os.environ["AZK_EARLY_TEMPO_BONUS"] == "0.1"
    assert "AZK_DMG_MITIGATION_BONUS" not in os.environ


def test_reward_env_override_keeps_caller_values(monkeypatch) -> None:
    monkeypatch.setenv("AZK_RESUME_KEEP_CURRENT_REWARD_ENV", "1")
    monkeypatch.setenv("AZK_EARLY_TEMPO_BONUS", "0.2")

    _apply_saved_reward_env(
        {"reward_env": {"AZK_EARLY_TEMPO_BONUS": "0.1"}}
    )

    assert os.environ["AZK_EARLY_TEMPO_BONUS"] == "0.2"


def test_legacy_fingerprint_without_reward_group_remains_compatible() -> None:
    saved = {"schedule_env": {"AZK_REWARD_SHAPING_ANNEAL": "1"}}
    current = {
        "schedule_env": {"AZK_REWARD_SHAPING_ANNEAL": "1"},
        "reward_env": {"AZK_EARLY_TEMPO_BONUS": "0.1"},
    }
    assert _resume_cfg_mismatches(saved, current) == []


def test_deck_pool_migration_excuses_only_deck_path(monkeypatch) -> None:
    monkeypatch.delenv("AZK_RESUME_ALLOW_SOURCE_DRIFT", raising=False)
    monkeypatch.delenv("AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV", raising=False)
    monkeypatch.delenv("AZK_RESUME_KEEP_CURRENT_REWARD_ENV", raising=False)

    assert _approved_resume_cfg_mismatch_prefixes(
        allow_trusted_source_drift=False,
        allow_deck_pool_migration=True,
    ) == ("deck_pool_path",)


def test_checkpoint_schedule_state_is_auditable() -> None:
    trainer = SimpleNamespace(
        trainer_shaped_reward_schedule_state=lambda: {
            "enabled": True,
            "start_epoch": 6000,
            "end_epoch": 6900,
            "absolute_update": 6450,
            "multiplier": 0.5,
        }
    )
    assert _trainer_shaped_reward_schedule_state(trainer) == {
        "enabled": True,
        "start_epoch": 6000,
        "end_epoch": 6900,
        "absolute_update": 6450,
        "multiplier": 0.5,
    }
