from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest
import torch

from league_training import LeaguePuffeRL, prebuilt_exposure_probability
from observation import DECKBUILD_OBSERVATION_CTYPE
from train import (
    _checkpoint_metadata_path,
    _peek_prebuilt_battle_decisions,
    _save_per_checkpoint_trainer_state,
)


def _curriculum_clock(progress: int = 0):
    trainer = LeaguePuffeRL.__new__(LeaguePuffeRL)
    trainer._prebuilt_enabled = True
    trainer._prebuilt_initial_probability = 0.8
    trainer._prebuilt_obs_dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
    trainer.config = {
        "prebuilt_final_probability": 0.2,
        "prebuilt_anneal_start_battle_decisions": 2,
        "prebuilt_anneal_end_battle_decisions": 8,
    }
    trainer.vecenv = SimpleNamespace(prebuilt_probability=np.zeros((2, 1), dtype=np.float32))
    trainer.restore_prebuilt_battle_decisions(progress)
    return trainer


def test_curriculum_clock_ignores_drafting_frozen_passive_and_terminal_rows():
    trainer = _curriculum_clock(2)
    observations = np.zeros((6, trainer._prebuilt_obs_dtype.itemsize), dtype=np.uint8)
    rows = observations.view(trainer._prebuilt_obs_dtype).reshape(-1)
    rows["action_mask"]["primary_action_mask"][:, 6] = True
    rows["deck_context"]["mode"][0] = 2  # Drafting.
    rows["action_mask"]["primary_action_mask"][4] = False
    rows["action_mask"]["primary_action_mask"][4, 0] = True  # Passive seat.
    trainable = np.array([True, True, False, True, True, True])
    terminal = np.array([False, False, False, True, False, False])

    trainer._advance_prebuilt_curriculum(torch.from_numpy(observations), trainable, terminal)

    assert trainer.prebuilt_battle_decisions == 4
    assert np.allclose(trainer.vecenv.prebuilt_probability, 0.6)
    trainer.restore_prebuilt_battle_decisions(20)
    assert np.allclose(trainer.vecenv.prebuilt_probability, 0.2)


def test_curriculum_checkpoint_preserves_mid_ramp_clock(tmp_path: Path):
    trainer = _curriculum_clock(5)
    model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda epoch: 1.0)
    trainer.global_step = 1000
    trainer.epoch = 10
    trainer.logger = SimpleNamespace(run_id="curriculum-regression")
    checkpoint = tmp_path / "model_azuki_local_000010.pt"
    checkpoint.touch()
    state_path = _save_per_checkpoint_trainer_state(
        trainer, checkpoint, {}, {"prebuilt_curriculum": True}
    )

    progress = _peek_prebuilt_battle_decisions(checkpoint, state_path)
    resumed = _curriculum_clock(progress)

    assert progress == 5
    assert np.allclose(resumed.vecenv.prebuilt_probability, 0.5)
    assert np.array_equal(resumed.vecenv.prebuilt_probability, trainer.vecenv.prebuilt_probability)


def test_curriculum_resume_rejects_lost_or_conflicting_clock(tmp_path: Path):
    checkpoint = tmp_path / "model_azuki_local_000010.pt"
    checkpoint.touch()
    metadata = {"resume_config_fingerprint": {"prebuilt_curriculum": True}}
    _checkpoint_metadata_path(checkpoint).write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        _peek_prebuilt_battle_decisions(checkpoint, None)

    metadata["prebuilt_battle_decisions"] = 5
    _checkpoint_metadata_path(checkpoint).write_text(json.dumps(metadata))
    state_path = tmp_path / "trainer_state_000010.pt"
    torch.save({"model_name": checkpoint.name, "prebuilt_battle_decisions": 4}, state_path)
    with pytest.raises(ValueError):
        _peek_prebuilt_battle_decisions(checkpoint, state_path)


def test_curriculum_rejects_nonfinite_probability():
    with pytest.raises(ValueError):
        prebuilt_exposure_probability(0, float("nan"), {})
