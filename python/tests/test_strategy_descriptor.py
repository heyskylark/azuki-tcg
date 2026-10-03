from __future__ import annotations

import copy
import unittest

from action import ActionType
from strategy_descriptor import (
    build_strategy_descriptor,
    evaluate_strategy_events,
    payoff_vector_distance,
)


FILLER = "AZK01-003"
ENTITY = "STT03-013"
WEAPON = "STT01-012"
SPELL = "AZK01-002"


def _post(step: dict, **updates) -> dict:
    actor = int(step["p"])
    hp = step.get("hp", [20, 20])
    result = {
        "phase": step.get("ph", "MAIN"),
        "hand": list(step.get("hand", [])),
        "my_garden": list(step.get("my_garden", [])),
        "my_alley": list(step.get("my_alley", [])),
        "opp_garden": list(step.get("opp_garden", [])),
        "opp_alley": list(step.get("opp_alley", [])),
        "my_hp": int(hp[actor]),
        "opp_hp": int(hp[1 - actor]),
        "my_leader_atk": int(step.get("my_leader_atk", 0)),
        "opp_leader_atk": int(step.get("opp_leader_atk", 0)),
        "my_leader_weapons": list(step.get("my_leader_weapons", [])),
        "opp_leader_weapons": list(step.get("opp_leader_weapons", [])),
        "ikz": list(step.get("ikz", [3, 3])),
        "ikz_token": bool(step.get("ikz_token", False)),
        "my_discard": list(step.get("my_discard", [])),
        "opp_discard": list(step.get("opp_discard", [])),
        "my_deck_n": int(step.get("my_deck_n", 30)),
        "gate_tapped": bool(step.get("gate_tapped", False)),
    }
    result.update(updates)
    return result


def _step(
    action: ActionType,
    decoded: dict,
    *,
    raw: tuple[int, int, int, int] | None = None,
    legal: list[tuple[int, int, int, int]] | None = None,
    phase: str = "MAIN",
    **state,
) -> dict:
    selected = raw or (int(action), 0, 0, 0)
    row = {
        "i": 0,
        "p": 0,
        "ph": phase,
        "a": list(selected),
        "d": {"t": action.name, **decoded},
        "hp": state.pop("hp", [20, 20]),
        "ikz": state.pop("ikz", [3, 3]),
        "hand": state.pop("hand", []),
        "my_deck_n": state.pop("my_deck_n", 30),
        "my_garden": state.pop("my_garden", []),
        "my_alley": state.pop("my_alley", []),
        "opp_garden": state.pop("opp_garden", []),
        "opp_alley": state.pop("opp_alley", []),
        "my_discard": state.pop("my_discard", []),
        "opp_discard": state.pop("opp_discard", []),
        "my_leader_atk": state.pop("my_leader_atk", 0),
        "opp_leader_atk": state.pop("opp_leader_atk", 0),
        "my_leader_weapons": state.pop("my_leader_weapons", []),
        "opp_leader_weapons": state.pop("opp_leader_weapons", []),
        "gate_tapped": state.pop("gate_tapped", False),
        "legal": [list(value) for value in (legal or [selected])],
    }
    if state:
        raise ValueError(f"unused state: {state}")
    row["post"] = _post(row)
    return row


def _game(
    steps: list[dict],
    *,
    gate: str,
    leader: str,
    main: list[str] | None = None,
    winner: int = 0,
    mode: str = "argmax",
) -> dict:
    for index, step in enumerate(steps):
        step["i"] = index
    main_cards = list(main or [FILLER] * 50)
    return {
        "trace_schema_version": 2,
        "policy_action_mode": mode,
        "game": 0,
        "seed": 7,
        "decks": [
            {"gate": gate, "leader": leader, "main": main_cards},
            {"gate": "STT02-002", "leader": "STT02-001", "main": [FILLER] * 50},
        ],
        "draft": [],
        "outcome": {"winner": winner, "terminated": True, "truncated": False},
        "steps": steps,
    }


def _attack(attacker: str, *, hp: list[int] | None = None) -> dict:
    step = _step(
        ActionType.ATTACK,
        {"attacker": attacker, "target": "OPP_LEADER"},
        raw=(int(ActionType.ATTACK), 0, 5, 0),
        hp=hp or [20, 20],
        my_garden=[f"{attacker}@0:3/3"],
    )
    step["post"]["opp_hp"] = int(step["hp"][1]) - 3
    return step


def _sequence_games() -> dict[str, tuple[dict, str]]:
    attach = _step(
        ActionType.ATTACH_WEAPON_FROM_HAND,
        {"card": WEAPON, "target": ENTITY},
        raw=(int(ActionType.ATTACH_WEAPON_FROM_HAND), 0, 0, 0),
        hand=[WEAPON],
        my_garden=[f"{ENTITY}@0:3/3"],
    )
    attach["post"]["hand"] = []
    attach["post"]["my_garden"] = [f"{ENTITY}@0:4/3+{WEAPON}"]
    discard_seen = _step(
        ActionType.PLAY_ENTITY_TO_GARDEN,
        {"card": FILLER, "slot": 1},
        hand=[FILLER],
        my_discard=[WEAPON],
    )
    discard_seen["post"]["hand"] = []
    discard_seen["post"]["my_garden"] = [f"{FILLER}@1:1/1"]
    portal = _step(
        ActionType.GATE_PORTAL,
        {"card": ENTITY, "slot": 0},
        raw=(int(ActionType.GATE_PORTAL), 0, 0, 0),
        legal=[(int(ActionType.GATE_PORTAL), 0, 0, 0)],
        my_alley=[f"{ENTITY}@0:3/3"],
        my_discard=[WEAPON],
    )
    portal["post"]["my_alley"] = []
    portal["post"]["my_garden"] = [f"{ENTITY}@0:3/3"]
    portal["post"]["my_discard"] = []
    surge_equip = _step(
        ActionType.SELECT_TO_EQUIP,
        {"card": WEAPON, "arg": 0, "src": "STT01-002"},
        raw=(int(ActionType.SELECT_TO_EQUIP), 0, 0, 0),
        my_garden=[f"{ENTITY}@0:3/3"],
    )
    surge_equip["post"]["my_garden"] = [f"{ENTITY}@0:4/3+{WEAPON}"]
    surge_attack = _attack(ENTITY)
    surge_attack["my_garden"] = [f"{ENTITY}@0:4/3+{WEAPON}"]
    surge_attack["post"] = _post(surge_attack, opp_hp=16)
    surge = _game(
        [attach, discard_seen, portal, surge_equip, surge_attack],
        gate="STT01-002",
        leader="STT01-001",
        main=[WEAPON, ENTITY] + [FILLER] * 48,
    )

    storm_attach = copy.deepcopy(attach)
    storm_portal = copy.deepcopy(portal)
    storm_portal["my_garden"] = [f"{ENTITY}@0:4/3+{WEAPON}"]
    storm_portal["my_discard"] = []
    storm_portal["post"] = _post(
        storm_portal,
        my_alley=[],
        my_garden=[f"{ENTITY}@0:3/3", f"{FILLER}@1:1/1"],
        my_discard=[],
    )
    storm_equip = _step(
        ActionType.SELECT_TO_EQUIP,
        {"card": WEAPON, "arg": 5, "src": "AZK01-120"},
        raw=(int(ActionType.SELECT_TO_EQUIP), 0, 5, 0),
        my_garden=[f"{ENTITY}@0:3/3", f"{FILLER}@1:1/1"],
    )
    storm_equip["post"]["my_leader_weapons"] = [WEAPON]
    storm_attack = _attack("MY_LEADER")
    storm_attack["a"] = [int(ActionType.ATTACK), 5, 5, 0]
    storm_attack["legal"] = [storm_attack["a"]]
    storm_attack["my_leader_weapons"] = [WEAPON]
    storm_attack["my_leader_atk"] = 3
    storm_attack["post"] = _post(storm_attack, opp_hp=17)
    stormchain = _game(
        [storm_attach, storm_portal, storm_equip, storm_attack],
        gate="AZK01-120",
        leader="STT01-001",
        main=[WEAPON, ENTITY] + [FILLER] * 48,
    )

    leader_use = _step(
        ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY,
        {"card": "MY_LEADER", "ability": 0},
        raw=(int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY), 5, 0, 0),
    )
    leader_use["post"]["ikz"] = [2, 3]

    first_spell = _step(
        ActionType.PLAY_SPELL_FROM_HAND,
        {"card": SPELL},
        raw=(int(ActionType.PLAY_SPELL_FROM_HAND), 0, 0, 0),
        hand=[SPELL],
    )
    first_spell["post"]["hand"] = []
    first_spell["post"]["my_discard"] = [SPELL]
    water_portal = copy.deepcopy(portal)
    water_portal["my_discard"] = [SPELL]
    water_portal["post"]["my_discard"] = []
    water_portal["post"]["hand"] = [SPELL]
    recovered = _step(
        ActionType.PLAY_ENTITY_TO_GARDEN,
        {"card": FILLER, "slot": 0},
        hand=[SPELL, FILLER],
    )
    recovered["post"]["hand"] = [SPELL]
    replay = copy.deepcopy(first_spell)
    replay["my_discard"] = []
    replay["hp"] = [14, 20]
    replay["post"] = _post(replay, hand=[], my_discard=[SPELL], my_hp=16)
    water_replay = _game(
        [first_spell, water_portal, recovered, replay],
        gate="AZK01-126",
        leader="STT02-001",
        main=[SPELL, ENTITY] + [FILLER] * 48,
    )

    heal = copy.deepcopy(first_spell)
    heal["hp"] = [14, 20]
    heal["post"] = _post(heal, hand=[], my_hp=16, my_discard=[SPELL])
    healing = _game([heal], gate="AZK01-126", leader="STT02-001", main=[SPELL] + [FILLER] * 49)

    shao_use = _step(
        ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY,
        {"card": "MY_LEADER", "ability": 0},
        raw=(int(ActionType.ACTIVATE_GARDEN_OR_LEADER_ABILITY), 5, 0, 0),
        phase="RESPONSE",
    )
    shao_use["combat"] = {"attacker": "ENEMY", "target": "LEADER"}
    shao_target = _step(
        ActionType.SELECT_EFFECT_TARGET,
        {"src": "STT02-001", "target": "ENEMY"},
        raw=(int(ActionType.SELECT_EFFECT_TARGET), 0, 0, 0),
        phase="RESPONSE",
        opp_garden=["ENEMY@0:4/4"],
    )
    shao_target["post"]["opp_garden"] = ["ENEMY@0:3/4"]
    shao = _game([shao_use, shao_target], gate="AZK01-126", leader="STT02-001")

    devotion = copy.deepcopy(portal)
    devotion["my_garden"] = ["SACRIFICE@1:1/4"]
    devotion["opp_garden"] = ["TARGET@0:3/5"]
    devotion["post"] = _post(
        devotion,
        my_alley=[],
        my_garden=[f"{ENTITY}@0:3/3"],
        opp_garden=["TARGET@0:3/1"],
    )
    devotion_game = _game([devotion], gate="AZK01-124", leader="STT03-001")

    bobu_use = copy.deepcopy(leader_use)
    bobu_use["hp"] = [14, 20]
    bobu_use["my_garden"] = [f"{ENTITY}@0:1/1"]
    bobu_use["post"] = _post(bobu_use, ikz=[2, 3])
    bobu_destroy = _step(
        ActionType.PLAY_SPELL_FROM_HAND,
        {"card": SPELL},
        raw=(int(ActionType.PLAY_SPELL_FROM_HAND), 0, 0, 0),
        hand=[SPELL],
        my_garden=[f"{ENTITY}@0:1/1"],
        hp=[14, 20],
    )
    bobu_destroy["post"] = _post(bobu_destroy, hand=[], my_garden=[], my_discard=[SPELL], my_hp=15)
    bobu = _game([bobu_use, bobu_destroy], gate="STT03-002", leader="STT03-001", main=[SPELL] + [FILLER] * 49)

    defender = _step(
        ActionType.DECLARE_DEFENDER,
        {"card": "DEFENDER"},
        raw=(int(ActionType.DECLARE_DEFENDER), 0, 0, 0),
        phase="RESPONSE",
        my_garden=["DEFENDER@0:1/4"],
    )
    defender["post"]["combat"] = {"intercepted": True}
    stone = _game([defender, copy.deepcopy(portal), _attack(ENTITY)], gate="STT03-002", leader="STT03-001", main=[ENTITY] + [FILLER] * 49)

    extra = _step(
        ActionType.SELECT_TO_GARDEN,
        {"card": "CHARGER", "slot": 0, "src": "AZK01-122"},
        hand=["CHARGER"],
    )
    extra["post"]["hand"] = []
    extra["post"]["my_garden"] = ["CHARGER@0:3/2"]
    rushfire = _game([copy.deepcopy(portal), extra, _attack("CHARGER")], gate="AZK01-122", leader="AZK01-121", main=["CHARGER"] + [FILLER] * 49)

    zero_use = copy.deepcopy(leader_use)
    zero_use["post"]["my_hp"] = 19
    zero_target = _step(
        ActionType.SELECT_EFFECT_TARGET,
        {"src": "STT04-001", "target": "BUFFED"},
        raw=(int(ActionType.SELECT_EFFECT_TARGET), 0, 0, 0),
        my_garden=["BUFFED@0:2/3"],
    )
    zero_target["post"]["my_garden"] = ["BUFFED@0:3/2"]
    zero = _game([zero_use, zero_target, _attack("BUFFED")], gate="STT04-002", leader="STT04-001", main=["BUFFED"] + [FILLER] * 49)

    play_one = _step(ActionType.PLAY_ENTITY_TO_GARDEN, {"card": "ONE", "slot": 0}, hand=["ONE"])
    play_one["post"]["hand"] = []
    play_one["post"]["my_garden"] = ["ONE@0:1/1"]
    play_two = _step(ActionType.PLAY_ENTITY_TO_GARDEN, {"card": "TWO", "slot": 1}, hand=["TWO"])
    play_two["post"]["hand"] = []
    play_two["post"]["my_garden"] = ["TWO@1:1/1"]
    kagoro_attack = _step(
        ActionType.ATTACK,
        {"attacker": "MY_LEADER", "target": "OPP_LEADER"},
        raw=(int(ActionType.ATTACK), 5, 5, 0),
        my_leader_atk=2,
    )
    kagoro_attack["post"]["opp_hp"] = 18
    kagoro_use = copy.deepcopy(leader_use)
    kagoro_use["post"]["my_leader_atk"] = 2
    kagoro = _game([play_one, play_two, kagoro_use, kagoro_attack], gate="AZK01-122", leader="AZK01-121", main=["ONE", "TWO"] + [FILLER] * 48)

    return {
        "lightning.surge_weapon_recovery_attack": (surge, "LIGHTNING"),
        "lightning.stormchain_weapon_requip_attack": (stormchain, "LIGHTNING"),
        "water.echoed_waves_spell_replay": (water_replay, "WATER"),
        "water.healing_flutter_timing": (healing, "WATER"),
        "water.shao_attacker_target": (shao, "WATER"),
        "earth.devotion_sacrifice_conversion": (devotion_game, "EARTH"),
        "earth.bobu_before_destruction": (bobu, "EARTH"),
        "earth.stone_defender_portal_attack": (stone, "EARTH"),
        "fire.rushfire_charge_conversion": (rushfire, "FIRE"),
        "fire.zero_before_attack": (zero, "FIRE"),
        "fire.kagoro_after_multi_play": (kagoro, "FIRE"),
    }


class StrategyDescriptorTest(unittest.TestCase):
    def test_scripted_sequences_require_order_and_conversion(self) -> None:
        for sequence_id, (positive, _) in _sequence_games().items():
            with self.subTest(sequence_id=sequence_id):
                result = evaluate_strategy_events([copy.deepcopy(positive)])
                sequence = result["sequences"][sequence_id]
                self.assertEqual(sequence["eligible"], 1)
                self.assertEqual(sequence["completed"], 1)
                self.assertEqual(sequence["converted"], 1)

                negative = copy.deepcopy(positive)
                if len(negative["steps"]) == 1:
                    only = negative["steps"][0]
                    only["post"] = _post(only)
                else:
                    negative["steps"] = list(reversed(negative["steps"]))
                result = evaluate_strategy_events([negative])
                invalidated = result["sequences"][sequence_id]
                self.assertTrue(
                    invalidated["completed"] == 0 or invalidated["converted"] == 0
                )

    def test_lightning_sequences_are_gate_specific_and_report_stages(self) -> None:
        games = _sequence_games()
        surge = games["lightning.surge_weapon_recovery_attack"][0]
        result = evaluate_strategy_events([copy.deepcopy(surge)])
        surge_sequence = result["sequences"]["lightning.surge_weapon_recovery_attack"]
        storm_sequence = result["sequences"]["lightning.stormchain_weapon_requip_attack"]
        self.assertEqual(surge_sequence["eligible"], 1)
        self.assertEqual(surge_sequence["completed"], 1)
        self.assertEqual(surge_sequence["converted"], 1)
        self.assertEqual(storm_sequence["eligible"], 0)
        self.assertEqual(
            surge_sequence["stages"]["weapon_recovered_equipped"]["reached"], 1
        )
        self.assertEqual(
            surge_sequence["stages"]["attack_resolved"]["reached_per_eligible"], 1.0
        )

        milled_weapon = copy.deepcopy(surge)
        milled_weapon["steps"] = milled_weapon["steps"][2:]
        for index, step in enumerate(milled_weapon["steps"]):
            step["i"] = index
        milled_sequence = evaluate_strategy_events([milled_weapon])["sequences"][
            "lightning.surge_weapon_recovery_attack"
        ]
        self.assertEqual(milled_sequence["eligible"], 1)
        self.assertEqual(milled_sequence["completed"], 1)
        self.assertEqual(
            milled_sequence["stages"]["prior_attach_then_discard_observed"][
                "reached"
            ],
            0,
        )

        delayed_equip = copy.deepcopy(milled_weapon)
        delayed_equip["steps"].insert(
            1,
            _step(ActionType.NOOP, {}, raw=(int(ActionType.NOOP), 0, 0, 0)),
        )
        for index, step in enumerate(delayed_equip["steps"]):
            step["i"] = index
        delayed_sequence = evaluate_strategy_events([delayed_equip])["sequences"][
            "lightning.surge_weapon_recovery_attack"
        ]
        self.assertEqual(delayed_sequence["eligible"], 1)
        self.assertEqual(delayed_sequence["completed"], 0)

        wrong_gate = copy.deepcopy(surge)
        wrong_gate["decks"][0]["gate"] = "STT03-002"
        invalid = evaluate_strategy_events([wrong_gate])["sequences"][
            "lightning.surge_weapon_recovery_attack"
        ]
        self.assertEqual(invalid["eligible"], 0)
        self.assertEqual(invalid["completed"], 0)

    def test_opportunity_rates_ignore_irrelevant_noops(self) -> None:
        game = _sequence_games()["earth.stone_defender_portal_attack"][0]
        baseline = evaluate_strategy_events([copy.deepcopy(game)])
        with_noop = copy.deepcopy(game)
        noop = _step(ActionType.NOOP, {}, raw=(int(ActionType.NOOP), 0, 0, 0))
        with_noop["steps"].insert(1, noop)
        observed = evaluate_strategy_events([with_noop])
        self.assertEqual(baseline["funnels"], observed["funnels"])
        for sequences in (baseline["sequences"], observed["sequences"]):
            for sequence in sequences.values():
                sequence.pop("examples")
        self.assertEqual(baseline["sequences"], observed["sequences"])

    def test_attack_alternatives_and_lethal_are_opportunity_normalized(self) -> None:
        face = _attack("PRESSURE", hp=[20, 3])
        face["my_garden"] = ["PRESSURE@0:5/5"]
        face["legal"] = [
            [int(ActionType.ATTACK), 0, 5, 0],
            [int(ActionType.ATTACK), 0, 0, 0],
        ]
        face["post"] = _post(face, opp_hp=0)
        game = _game([face], gate="STT04-002", leader="STT04-001")
        result = evaluate_strategy_events([game])
        self.assertEqual(result["funnels"]["attack.nominal_lethal_face"]["opportunities"], 1)
        self.assertEqual(result["funnels"]["attack.nominal_lethal_face"]["selected"], 1)
        self.assertEqual(result["game_profile"]["face_share_when_both_legal"], 1.0)

        entity = copy.deepcopy(face)
        entity["a"] = [int(ActionType.ATTACK), 0, 0, 0]
        entity["d"]["target"] = "TARGET"
        entity["opp_garden"] = ["TARGET@0:2/2"]
        entity["post"] = _post(entity, opp_garden=[])
        result = evaluate_strategy_events([_game([entity], gate="STT04-002", leader="STT04-001")])
        self.assertEqual(result["funnels"]["attack.nominal_lethal_face"]["opportunities"], 1)
        self.assertEqual(result["funnels"]["attack.nominal_lethal_face"]["selected"], 0)
        self.assertEqual(result["game_profile"]["face_share_when_both_legal"], 0.0)

    def test_selected_card_must_resolve_before_conversion(self) -> None:
        unresolved = _step(
            ActionType.PLAY_SPELL_FROM_HAND,
            {"card": SPELL},
            raw=(int(ActionType.PLAY_SPELL_FROM_HAND), 0, 0, 0),
            hand=[SPELL],
        )
        resolved = copy.deepcopy(unresolved)
        resolved["post"]["hand"] = []
        unresolved_result = evaluate_strategy_events([
            _game([unresolved], gate="AZK01-126", leader="STT02-001", main=[SPELL] + [FILLER] * 49)
        ])
        resolved_result = evaluate_strategy_events([
            _game([resolved], gate="AZK01-126", leader="STT02-001", main=[SPELL] + [FILLER] * 49)
        ])
        self.assertEqual(unresolved_result["card_funnels"][SPELL]["play"]["resolved"], 0)
        self.assertEqual(resolved_result["card_funnels"][SPELL]["play"]["resolved"], 1)

    def test_card_lifecycle_resource_reserve_and_temporary_conversion(self) -> None:
        lightning = _sequence_games()["lightning.surge_weapon_recovery_attack"][0]
        lightning_result = evaluate_strategy_events([copy.deepcopy(lightning)])
        weapon = lightning_result["card_funnels"][WEAPON]["lifecycle"]
        self.assertEqual(weapon["recovered"], 1)
        self.assertEqual(weapon["re_equipped"], 1)

        before_draw = _step(
            ActionType.PLAY_ENTITY_TO_GARDEN,
            {"card": FILLER, "slot": 0},
            hand=[FILLER],
            my_deck_n=30,
        )
        before_draw["post"] = _post(before_draw, hand=[], my_deck_n=30)
        after_draw = _step(
            ActionType.PLAY_SPELL_FROM_HAND,
            {"card": SPELL},
            raw=(int(ActionType.PLAY_SPELL_FROM_HAND), 0, 0, 0),
            hand=[SPELL],
            my_deck_n=29,
        )
        after_draw["post"] = _post(
            after_draw, hand=[], my_discard=[SPELL], my_deck_n=29
        )
        drawn = evaluate_strategy_events(
            [
                _game(
                    [before_draw, after_draw],
                    gate="AZK01-126",
                    leader="STT02-001",
                    main=[SPELL] + [FILLER] * 49,
                )
            ]
        )
        self.assertEqual(drawn["card_funnels"][SPELL]["lifecycle"]["drawn"], 1)

        response_card = "STT01-017"
        reserve = _step(
            ActionType.PLAY_ENTITY_TO_GARDEN,
            {"card": FILLER, "slot": 0},
            hand=[response_card, FILLER],
            ikz=[1, 1],
        )
        reserve["post"] = _post(
            reserve, hand=[response_card], ikz=[1, 1], my_garden=[f"{FILLER}@0:1/1"]
        )
        reserved = evaluate_strategy_events(
            [
                _game(
                    [reserve],
                    gate="STT01-002",
                    leader="STT01-001",
                    main=[response_card] + [FILLER] * 49,
                )
            ]
        )["funnels"]["response.reserved"]
        self.assertEqual(reserved["opportunities"], 1)
        self.assertEqual(reserved["selected"], 1)

        devotion = _sequence_games()["earth.devotion_sacrifice_conversion"][0]
        portal = evaluate_strategy_events([copy.deepcopy(devotion)])["funnels"]["portal"]
        self.assertEqual(portal["resolved"], 1)
        self.assertEqual(portal["converted"], 1)

        rushfire = _sequence_games()["fire.rushfire_charge_conversion"][0]
        charge = evaluate_strategy_events([copy.deepcopy(rushfire)])["funnels"][
            "temporary.charge"
        ]
        self.assertEqual(charge["opportunities"], 1)
        self.assertEqual(charge["converted"], 1)

        zero = _sequence_games()["fire.zero_before_attack"][0]
        attack = evaluate_strategy_events([copy.deepcopy(zero)])["funnels"][
            "temporary.attack"
        ]
        self.assertEqual(attack["opportunities"], 1)
        self.assertEqual(attack["converted"], 1)

    def test_eventual_winner_does_not_resolve_a_failed_attack(self) -> None:
        attack = _attack("PRESSURE")
        attack["post"] = _post(attack)
        result = evaluate_strategy_events([_game([attack], gate="STT04-002", leader="STT04-001")])
        self.assertEqual(result["funnels"]["attack.face"]["resolved"], 0)
        attack["post"]["combat"] = {"attacker": "PRESSURE"}
        result = evaluate_strategy_events([_game([attack], gate="STT04-002", leader="STT04-001")])
        self.assertEqual(result["funnels"]["attack.face"]["resolution_observed"], 0)

    def test_entity_attack_tapping_is_not_damage(self) -> None:
        attack = _attack("PRESSURE")
        attack["a"] = [int(ActionType.ATTACK), 0, 0, 0]
        attack["legal"] = [attack["a"]]
        attack["d"]["target"] = "TARGET"
        attack["opp_garden"] = ["TARGET@0:3/3"]
        attack["post"] = _post(attack, my_garden=["PRESSURE@0:3/3T"])
        result = evaluate_strategy_events([_game([attack], gate="STT04-002", leader="STT04-001")])
        self.assertEqual(result["funnels"]["attack.entity"]["resolved"], 0)

    def test_new_ikz_is_not_counted_twice_or_as_conversion(self) -> None:
        step = _step(ActionType.GATE_PORTAL, {"card": ENTITY}, ikz=[2, 2])
        step["post"] = _post(step, ikz=[3, 3])
        result = evaluate_strategy_events([_game([step], gate="STT02-002", leader="STT02-001")])
        self.assertEqual(result["resource_profile"]["ikz_net_added"], 1)
        self.assertEqual(result["resource_profile"]["ikz_net_readied"], 0)
        self.assertIsNone(result["resource_profile"]["generated_ikz_conversion"])

    def test_second_spell_copy_is_not_a_replay_or_useful_effect(self) -> None:
        first = _step(ActionType.PLAY_SPELL_FROM_HAND, {"card": SPELL}, hand=[SPELL, SPELL])
        first["post"] = _post(first, hand=[SPELL], my_discard=[SPELL])
        second = _step(ActionType.PLAY_SPELL_FROM_HAND, {"card": SPELL}, hand=[SPELL], my_discard=[SPELL])
        second["post"] = _post(second, hand=[], my_discard=[SPELL, SPELL])
        result = evaluate_strategy_events([_game([first, second], gate="AZK01-126", leader="STT02-001", main=[SPELL, SPELL] + [FILLER] * 48)])
        self.assertEqual(result["card_funnels"][SPELL]["lifecycle"]["replayed"], 0)
        self.assertEqual(result["card_funnels"][SPELL]["play"]["converted"], 0)

    def test_spell_self_destruction_alone_is_not_positive_effect(self) -> None:
        spell = _step(ActionType.PLAY_SPELL_FROM_HAND, {"card": SPELL}, hand=[SPELL], my_garden=[f"{ENTITY}@0:3/3"])
        spell["post"] = _post(spell, hand=[], my_garden=[], my_discard=[SPELL, ENTITY])
        result = evaluate_strategy_events([_game([spell], gate="AZK01-126", leader="STT02-001", main=[SPELL, ENTITY] + [FILLER] * 48)])
        self.assertEqual(result["card_funnels"][SPELL]["play"]["converted"], 0)

    def test_expired_or_unpaid_fire_buff_does_not_convert(self) -> None:
        game = _sequence_games()["fire.zero_before_attack"][0]
        game["steps"][0]["post"]["my_hp"] = 20
        result = evaluate_strategy_events([game])
        self.assertEqual(result["sequences"]["fire.zero_before_attack"]["converted"], 0)
        game = _sequence_games()["fire.zero_before_attack"][0]
        enemy_turn = _step(ActionType.NOOP, {})
        enemy_turn["p"] = 1
        game["steps"].insert(2, enemy_turn)
        result = evaluate_strategy_events([game])
        self.assertEqual(result["sequences"]["fire.zero_before_attack"]["converted"], 0)

    def test_bobu_destruction_without_healing_is_not_conversion(self) -> None:
        game = _sequence_games()["earth.bobu_before_destruction"][0]
        game["steps"][-1]["post"]["my_hp"] = 14
        sequence = evaluate_strategy_events([game])["sequences"]["earth.bobu_before_destruction"]
        self.assertEqual(sequence["completed"], 1)
        self.assertEqual(sequence["converted"], 0)

    def test_recovered_weapon_must_remain_on_the_attacker(self) -> None:
        game = _sequence_games()["lightning.surge_weapon_recovery_attack"][0]
        game["steps"][-1]["my_garden"] = [f"{ENTITY}@0:3/3"]
        sequence = evaluate_strategy_events([game])["sequences"]["lightning.surge_weapon_recovery_attack"]
        self.assertEqual(sequence["completed"], 0)

    def test_deferred_echoed_recovery_tracks_the_portal_discard(self) -> None:
        game = _sequence_games()["water.echoed_waves_spell_replay"][0]
        portal = game["steps"][1]
        portal["post"]["hand"] = []
        selection = _step(
            ActionType.SELECT_FROM_SELECTION,
            {"card": SPELL, "src": "AZK01-126"},
            hand=[],
        )
        selection["post"] = _post(selection, hand=[SPELL])
        game["steps"].insert(2, selection)
        result = evaluate_strategy_events([game])
        self.assertEqual(result["sequences"]["water.echoed_waves_spell_replay"]["converted"], 1)
        self.assertEqual(result["card_funnels"][SPELL]["lifecycle"]["recovered"], 1)
        self.assertEqual(result["card_funnels"][SPELL]["lifecycle"]["replayed"], 1)
        game["steps"][1]["hand"] = [SPELL]
        result = evaluate_strategy_events([game])
        self.assertEqual(result["sequences"]["water.echoed_waves_spell_replay"]["completed"], 0)
        self.assertEqual(result["card_funnels"][SPELL]["lifecycle"]["replayed"], 0)

    def test_alley_damage_uses_slot_after_leader(self) -> None:
        attack = _attack("PRESSURE")
        attack["a"] = [int(ActionType.ATTACK), 0, 6, 0]
        attack["legal"] = [attack["a"]]
        attack["d"]["target"] = "ALLEY:TARGET"
        attack["opp_alley"] = ["TARGET@0:3/3", "OTHER@1:2/5"]
        attack["post"] = _post(attack, opp_alley=["OTHER@1:2/5"])
        result = evaluate_strategy_events([_game([attack], gate="STT04-002", leader="STT04-001")])
        self.assertEqual(result["funnels"]["attack.entity"]["resolved"], 1)

    def test_equipping_one_copy_does_not_credit_another_copys_attack(self) -> None:
        for sequence_id in (
            "lightning.surge_weapon_recovery_attack",
            "lightning.stormchain_weapon_requip_attack",
        ):
            with self.subTest(sequence=sequence_id):
                game = _sequence_games()[sequence_id][0]
                equip, attack = game["steps"][-2:]
                equip["a"][2] = equip["d"]["arg"] = 0
                equip["legal"] = [equip["a"]]
                equip["my_garden"] = [f"{ENTITY}@0:3/3", f"{ENTITY}@1:4/3+{WEAPON}"]
                equip["post"] = _post(equip, my_garden=[f"{ENTITY}@0:4/3+{WEAPON}", f"{ENTITY}@1:4/3+{WEAPON}"])
                attack["a"][1] = 1
                attack["legal"] = [attack["a"]]
                attack["d"]["attacker"] = ENTITY
                attack["my_garden"] = list(equip["post"]["my_garden"])
                attack["post"] = _post(attack, opp_hp=16)
                result = evaluate_strategy_events([game])
                self.assertEqual(result["sequences"][sequence_id]["converted"], 0)

    def test_rushfire_charge_requires_the_new_copys_attack(self) -> None:
        game = _sequence_games()["fire.rushfire_charge_conversion"][0]
        extra, attack = game["steps"][-2:]
        extra["a"][2] = extra["d"]["arg"] = 1
        extra["legal"] = [extra["a"]]
        extra["my_garden"] = ["CHARGER@0:3/2"]
        extra["post"] = _post(extra, hand=[], my_garden=["CHARGER@0:3/2", "CHARGER@1:3/2"])
        attack["my_garden"] = list(extra["post"]["my_garden"])
        attack["post"] = _post(attack, opp_hp=17)
        result = evaluate_strategy_events([game])
        self.assertEqual(result["sequences"]["fire.rushfire_charge_conversion"]["converted"], 0)
        self.assertEqual(result["funnels"]["temporary.charge"]["converted"], 0)

    def test_bobu_healing_observes_opponent_destruction_until_next_turn(self) -> None:
        game = _sequence_games()["earth.bobu_before_destruction"][0]
        destruction = game["steps"][-1]
        destruction["p"] = 1
        destruction["my_garden"] = []
        destruction["opp_garden"] = [f"{ENTITY}@0:1/1"]
        destruction["post"] = _post(destruction, hand=[], opp_garden=[], opp_hp=15)
        descriptor = build_strategy_descriptor([game], label="opponent-heal")
        context = next(row for row in descriptor["elemental_strategy"]["contexts"].values() if row["seat"] == 0)
        self.assertEqual(context["counts"]["leader_hp_restored"], 1)
        self.assertEqual(context["counts"]["earth.bobu_before_destruction/converted"], 1)
        game["steps"].insert(1, _step(ActionType.NOOP, {}))
        game["steps"][1]["p"] = 1
        game["steps"].insert(2, _step(ActionType.NOOP, {}))
        descriptor = build_strategy_descriptor([game], label="expired-heal")
        context = next(row for row in descriptor["elemental_strategy"]["contexts"].values() if row["seat"] == 0)
        self.assertEqual(context["counts"]["leader_hp_restored"], 1)
        self.assertEqual(context["counts"]["earth.bobu_before_destruction/converted"], 0)

    def test_water_resource_spell_conversion_excludes_existing_ikz(self) -> None:
        portal = _step(
            ActionType.GATE_PORTAL, {"card": ENTITY, "slot": 0},
            my_alley=[f"{ENTITY}@0:3/3"], ikz=[0, 2],
        )
        portal["post"] = _post(portal, my_alley=[], my_garden=[f"{ENTITY}@0:3/3"], ikz=[2, 2])
        spell = _step(ActionType.PLAY_SPELL_FROM_HAND, {"card": SPELL}, hand=[SPELL], ikz=[2, 2], hp=[14, 20])
        spell["post"] = _post(spell, hand=[], ikz=[1, 2], my_hp=16)
        game = _game([portal, spell], gate="STT02-002", leader="STT02-001", main=[SPELL, ENTITY] + [FILLER] * 48)
        report = build_strategy_descriptor([game], label="resource")
        context = next(row for row in report["elemental_strategy"]["contexts"].values() if row["seat"] == 0)
        self.assertEqual(context["counts"]["hydromancy_ikz_to_spell_lower_bound"], 1)
        portal["ikz"] = [1, 2]
        report = build_strategy_descriptor([game], label="resource")
        context = next(row for row in report["elemental_strategy"]["contexts"].values() if row["seat"] == 0)
        self.assertEqual(context["counts"]["hydromancy_ikz_to_spell_lower_bound"], 0)

    def test_illegal_selected_action_invalidates_trace(self) -> None:
        step = _attack("PRESSURE")
        step["legal"] = [[int(ActionType.NOOP), 0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "absent from legal trace"):
            evaluate_strategy_events([_game([step], gate="STT04-002", leader="STT04-001")])

    def test_context_relationships_hold_leader_fixed(self) -> None:
        games = [
            _game([], gate="STT04-002", leader="STT04-001"),
            _game([], gate="AZK01-122", leader="STT04-001"),
            _game([], gate="STT04-002", leader="AZK01-121"),
            _game([], gate="AZK01-122", leader="AZK01-121", main=[ENTITY] + [FILLER] * 49),
        ]
        relationships = build_strategy_descriptor(games, label="siblings")["deck"]["relationships"]
        self.assertEqual(relationships["sibling_exact_collision_pairs"], 1)
        self.assertEqual(len(relationships["sibling_pairs"]), 2)
        for pair in relationships["sibling_pairs"]:
            self.assertEqual(pair["left_context"].split(":")[1], pair["right_context"].split(":")[1])
        self.assertTrue(any(pair["multiset_distance"] > 0 for pair in relationships["sibling_pairs"]))

    def test_completed_run_cannot_qualify_without_converted_element_breadth(self) -> None:
        from build_terminal_safe_reward_final_decision import qualification_checks

        mode = {
            "by_element": {element: {"converted_sequences": 1} for element in ("EARTH", "FIRE", "LIGHTNING", "WATER")},
            "elemental_strategy": {"contexts": {}},
        }
        arm = {
            "training": {
                "integrity_maxima": {"timeout": 0},
                "max_invalid_metric": 0,
                "reward_raw_reconstruction_max_abs_error": 0,
                "reward_scaled_reconstruction_max_abs_error": 0,
                "ppo_component_reconstruction_max_abs_error": 0,
                "last_epoch": 2930,
            },
            "endpoint_curated": {"completed": 32, "games": 32},
            "windows": [
                {"update": update, "sample": copy.deepcopy(mode), "argmax": copy.deepcopy(mode)}
                for update in (1950, 2925)
            ],
        }
        checks, _ = qualification_checks(arm)
        self.assertTrue(all(checks.values()))
        arm["windows"][-1]["argmax"]["by_element"]["EARTH"]["converted_sequences"] = 0
        checks, _ = qualification_checks(arm)
        self.assertFalse(checks["argmax_late_converted_breadth"])

    def test_descriptor_versions_mode_and_payoff_context(self) -> None:
        game = _sequence_games()["water.healing_flutter_timing"][0]
        cells = [
            {"context_key": "gate=G|leader=L|opp=O|seat=0|starter=1", "score": 0.75, "games": 8}
        ]
        descriptor = build_strategy_descriptor(
            [game], label="fixture", checkpoint_sha256="abc", payoff_cells=cells
        )
        self.assertEqual(descriptor["schema_id"], "azuki.strategy_descriptor")
        self.assertEqual(descriptor["trace_capabilities"]["policy_action_modes"], {"argmax": 1})
        self.assertEqual(descriptor["payoff_vector"], cells)
        self.assertTrue(str(descriptor["descriptor_id"]).startswith("sha256:"))

        common, distance = payoff_vector_distance(
            cells,
            [{"context_key": cells[0]["context_key"], "score": 0.25, "games": 8}],
        )
        self.assertEqual(common, 1)
        self.assertEqual(distance, 0.5)
        self.assertEqual(
            payoff_vector_distance(cells, [{"context_key": "other", "score": 0.5}]),
            (0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
