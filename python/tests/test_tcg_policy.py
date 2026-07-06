import unittest

import torch

from policy.v2.tcg_policy import (
    ACT_ACTIVATE_ALLEY_ABILITY,
    ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
    ACT_ATTACK,
    ACT_ATTACH_WEAPON_FROM_HAND,
    ACT_BOTTOM_DECK_ALL,
    ACT_BOTTOM_DECK_CARD,
    ACT_DECLARE_DEFENDER,
    ACT_GATE_PORTAL,
    ACT_NOOP,
    ACT_PLAY_ENTITY_TO_ALLEY,
    ACT_PLAY_ENTITY_TO_GARDEN,
    ACT_PLAY_SPELL_FROM_HAND,
    ACT_SELECT_COST_TARGET,
    ACT_SELECT_EFFECT_TARGET,
    ACT_SELECT_FROM_SELECTION,
    ACT_SELECT_TO_ALLEY,
    ACT_SELECT_TO_EQUIP,
    ACT_SELECT_TO_GARDEN,
    ACT_TOP_DECK_CARD,
    LEGAL_ACTION_ARG_KIND_ABILITY_INDEX,
    LEGAL_ACTION_ARG_KIND_BOOL,
    LEGAL_ACTION_ARG_KIND_GENERIC_TARGET,
    LEGAL_ACTION_ARG_KIND_HAND,
    LEGAL_ACTION_ARG_KIND_OPP_DEFENDER,
    LEGAL_ACTION_ARG_KIND_SELECTION,
    LEGAL_ACTION_ARG_KIND_SELF_ALLEY,
    LEGAL_ACTION_ARG_KIND_SELF_GARDEN,
    LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER,
    LEGAL_ACTION_ARG_KIND_UNUSED,
    ALLEY_SIZE,
    GARDEN_SIZE,
    TCG,
    UNIT_EMBED_SIZE,
    _build_legal_action_arg_kind_table,
)


class LegalActionArgKindTableTest(unittest.TestCase):
    def test_expected_primary_action_mappings(self) -> None:
        table = _build_legal_action_arg_kind_table()
        expected = {
            ACT_NOOP: (
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_PLAY_ENTITY_TO_GARDEN: (
                LEGAL_ACTION_ARG_KIND_HAND,
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN,
                LEGAL_ACTION_ARG_KIND_BOOL,
            ),
            ACT_PLAY_ENTITY_TO_ALLEY: (
                LEGAL_ACTION_ARG_KIND_HAND,
                LEGAL_ACTION_ARG_KIND_SELF_ALLEY,
                LEGAL_ACTION_ARG_KIND_BOOL,
            ),
            ACT_ATTACK: (
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER,
                LEGAL_ACTION_ARG_KIND_OPP_DEFENDER,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_ATTACH_WEAPON_FROM_HAND: (
                LEGAL_ACTION_ARG_KIND_HAND,
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER,
                LEGAL_ACTION_ARG_KIND_BOOL,
            ),
            ACT_PLAY_SPELL_FROM_HAND: (
                LEGAL_ACTION_ARG_KIND_HAND,
                LEGAL_ACTION_ARG_KIND_ABILITY_INDEX,
                LEGAL_ACTION_ARG_KIND_BOOL,
            ),
            ACT_DECLARE_DEFENDER: (
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_GATE_PORTAL: (
                LEGAL_ACTION_ARG_KIND_SELF_ALLEY,
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY: (
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER,
                LEGAL_ACTION_ARG_KIND_ABILITY_INDEX,
                LEGAL_ACTION_ARG_KIND_BOOL,
            ),
            ACT_ACTIVATE_ALLEY_ABILITY: (
                LEGAL_ACTION_ARG_KIND_ABILITY_INDEX,
                LEGAL_ACTION_ARG_KIND_SELF_ALLEY,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_SELECT_COST_TARGET: (
                LEGAL_ACTION_ARG_KIND_GENERIC_TARGET,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_SELECT_EFFECT_TARGET: (
                LEGAL_ACTION_ARG_KIND_GENERIC_TARGET,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_SELECT_FROM_SELECTION: (
                LEGAL_ACTION_ARG_KIND_SELECTION,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_SELECT_TO_ALLEY: (
                LEGAL_ACTION_ARG_KIND_SELECTION,
                LEGAL_ACTION_ARG_KIND_SELF_ALLEY,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_SELECT_TO_EQUIP: (
                LEGAL_ACTION_ARG_KIND_SELECTION,
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_SELECT_TO_GARDEN: (
                LEGAL_ACTION_ARG_KIND_SELECTION,
                LEGAL_ACTION_ARG_KIND_SELF_GARDEN,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_TOP_DECK_CARD: (
                LEGAL_ACTION_ARG_KIND_SELECTION,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_BOTTOM_DECK_CARD: (
                LEGAL_ACTION_ARG_KIND_SELECTION,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
            ACT_BOTTOM_DECK_ALL: (
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
                LEGAL_ACTION_ARG_KIND_UNUSED,
            ),
        }

        for action_id, arg_kinds in expected.items():
            with self.subTest(action_id=action_id):
                self.assertEqual(tuple(table[action_id].tolist()), arg_kinds)


class LegalActionRefGatherTest(unittest.TestCase):
    def test_garden_or_leader_gather_uses_combined_table_and_zeroes_invalid(self) -> None:
        tcg = object.__new__(TCG)
        garden_rows = torch.arange(2 * GARDEN_SIZE * UNIT_EMBED_SIZE, dtype=torch.float32).reshape(
            2, GARDEN_SIZE, UNIT_EMBED_SIZE
        )
        leader_rows = torch.full((2, 1, UNIT_EMBED_SIZE), -7.0, dtype=torch.float32)
        combined = torch.cat([garden_rows, leader_rows], dim=1)
        indices = torch.tensor(
            [
                [0, GARDEN_SIZE - 1, GARDEN_SIZE, GARDEN_SIZE + 2],
                [1, 2, GARDEN_SIZE, GARDEN_SIZE + 9],
            ],
            dtype=torch.long,
        )

        refs = tcg._gather_garden_or_leader_refs(combined, indices)

        self.assertTrue(torch.equal(refs[0, 0], garden_rows[0, 0]))
        self.assertTrue(torch.equal(refs[0, 1], garden_rows[0, GARDEN_SIZE - 1]))
        self.assertTrue(torch.equal(refs[0, 2], leader_rows[0, 0]))
        self.assertTrue(torch.equal(refs[1, 2], leader_rows[1, 0]))
        self.assertTrue(torch.equal(refs[0, 3], torch.zeros(UNIT_EMBED_SIZE)))
        self.assertTrue(torch.equal(refs[1, 3], torch.zeros(UNIT_EMBED_SIZE)))

    def test_opponent_defender_gather_uses_combined_table_and_zeroes_invalid(self) -> None:
        tcg = object.__new__(TCG)
        garden_rows = torch.arange(GARDEN_SIZE * UNIT_EMBED_SIZE, dtype=torch.float32).reshape(
            1, GARDEN_SIZE, UNIT_EMBED_SIZE
        )
        leader_rows = torch.full((1, 1, UNIT_EMBED_SIZE), 33.0, dtype=torch.float32)
        alley_rows = torch.arange(
            ALLEY_SIZE * UNIT_EMBED_SIZE,
            2 * ALLEY_SIZE * UNIT_EMBED_SIZE,
            dtype=torch.float32,
        ).reshape(1, ALLEY_SIZE, UNIT_EMBED_SIZE)
        combined = torch.cat([garden_rows, leader_rows, alley_rows], dim=1)
        indices = torch.tensor(
            [[0, GARDEN_SIZE, GARDEN_SIZE + 1, GARDEN_SIZE + ALLEY_SIZE, GARDEN_SIZE + ALLEY_SIZE + 1]],
            dtype=torch.long,
        )

        refs = tcg._gather_opponent_defender_refs(combined, indices)

        self.assertTrue(torch.equal(refs[0, 0], garden_rows[0, 0]))
        self.assertTrue(torch.equal(refs[0, 1], leader_rows[0, 0]))
        self.assertTrue(torch.equal(refs[0, 2], alley_rows[0, 0]))
        self.assertTrue(torch.equal(refs[0, 3], alley_rows[0, ALLEY_SIZE - 1]))
        self.assertTrue(torch.equal(refs[0, 4], torch.zeros(UNIT_EMBED_SIZE)))


class LegalActionCandidateTrimTest(unittest.TestCase):
    # Trim buckets to powers of two with a floor of 32 so torch.compile /
    # CUDA graphs see a bounded shape set; all active rows must be kept.
    def test_trim_keeps_all_active_rows_within_bucket(self) -> None:
        tcg = object.__new__(TCG)
        legal_actions = torch.arange(2 * 8 * 4, dtype=torch.long).reshape(2, 8, 4)
        legal_action_count = torch.tensor([3, 5], dtype=torch.long)

        trimmed = tcg._trim_active_legal_action_candidates(legal_actions, legal_action_count)

        # Bucket floor is 32, clamped to the table width (8): nothing trimmed.
        self.assertEqual(trimmed.shape, (2, 8, 4))
        self.assertTrue(torch.equal(trimmed, legal_actions))

    def test_trim_rounds_up_to_power_of_two_bucket(self) -> None:
        tcg = object.__new__(TCG)
        legal_actions = torch.arange(2 * 1024 * 4, dtype=torch.long).reshape(2, 1024, 4)
        legal_action_count = torch.tensor([33, 70], dtype=torch.long)

        trimmed = tcg._trim_active_legal_action_candidates(legal_actions, legal_action_count)

        self.assertEqual(trimmed.shape, (2, 128, 4))
        self.assertTrue(torch.equal(trimmed, legal_actions[:, :128]))

    def test_preserves_fallback_rows_when_batch_has_no_legal_actions(self) -> None:
        tcg = object.__new__(TCG)
        legal_actions = torch.arange(2 * 6 * 4, dtype=torch.long).reshape(2, 6, 4)
        legal_action_count = torch.zeros(2, dtype=torch.long)

        trimmed = tcg._trim_active_legal_action_candidates(legal_actions, legal_action_count)

        # Must keep at least one row for the sampler's fallback; bucket floor
        # (32) clamps to the table width, so the whole table survives.
        self.assertGreaterEqual(trimmed.shape[1], 1)
        self.assertEqual(trimmed.shape, (2, 6, 4))
        self.assertTrue(torch.equal(trimmed, legal_actions))

if __name__ == "__main__":
    unittest.main()
