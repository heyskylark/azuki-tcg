"""Damage-mitigation bonus (AZK_DMG_MITIGATION_BONUS) differential tests.

Same seeded random-legal trajectory, knob off vs on: reward diffs may appear
only on steps at/after a DECLARE_DEFENDER by the credited player (the
resolution step), must be zero-sum, and magnitudes must be w*k/cap for
integer soak k in 1..cap.

Run: PYTHONPATH=build/python/src:python/src pytest \
    python/src/test_dmg_mitigation_bonus.py -q
"""

from __future__ import annotations

import numpy as np

from action import ActionType
from test_early_tempo_bonus import _rollout

DECLARE_DEFENDER = int(ActionType.DECLARE_DEFENDER)


def test_dmg_mitigation_differential():
    base_r, base_a = _rollout({}, steps=4000)
    mit_r, mit_a = _rollout(
        {"AZK_DMG_MITIGATION_BONUS": "0.15", "AZK_DMG_MITIGATION_CAP": "10"},
        steps=4000,
    )
    np.testing.assert_array_equal(base_a, mit_a)
    diff = mit_r - base_r
    changed = np.abs(diff).max(axis=1) > 1e-9
    assert changed.sum() > 0, "bonus never fired — extend steps or check declare rate"
    # zero-sum on every changed step
    assert np.abs(diff.sum(axis=1)[changed]).max() < 1e-5
    # magnitudes quantized to 0.15*k/10
    mags = np.abs(diff[changed]).max(axis=1)
    ks = mags / 0.015
    assert np.allclose(ks, np.round(ks), atol=1e-4), f"non-quantized magnitudes {mags[:5]}"
    assert (np.round(ks) >= 1).all() and (np.round(ks) <= 10).all()
    # every changed step follows a DECLARE_DEFENDER somewhere earlier in the
    # episode stream (weak ordering check: at least one declare must exist)
    declares = (base_a[:, :, 0] == DECLARE_DEFENDER).any(axis=1)
    assert declares.sum() > 0
    first_declare = int(np.argmax(declares))
    first_changed = int(np.argmax(changed))
    assert first_changed >= first_declare
