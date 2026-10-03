"""Constructed legal-history diagnostics; not a natural-play win-rate estimate.

Port of results/leader_history_benchmark_v1/history_benchmark.py to the specialist
snapshot runtime.  Fixture logic is unchanged; only the harness imports, output
directory and runner construction differ.  The draft catalog is the original
v3 deck pool so the fixed MAIN deck is draftable exactly as before.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from tb_runtime import (EpisodeRunner, GameLogger, DISABLED_PROCESS_ENV, POLICIES, V3_POOL, RT as RUNTIME,  # noqa: E402
                        _load_model_weights, tcg_sampler, terminal_winner, sha256)
import torch  # noqa: E402
from action import ActionType  # noqa: E402
from deck_building import PlayerDeckBuildState  # noqa: E402

os.environ.update(DISABLED_PROCESS_ENV)
torch.set_num_threads(1)
tcg_sampler.set_sampling_params(primary_temperature=1., subaction_temperature=1., smoothing_eps=0., legal_row_temperature=1., deck_pick_smoothing_eps=0.)
VANILLA = ['STT02-004', 'AZK01-067', 'AZK01-012', 'STT03-008', 'STT04-006', 'STT04-011']
MAIN = [c for c in VANILLA + ['STT01-007', 'STT01-008', 'AZK01-033', 'AZK01-069', 'AZK01-070', 'AZK01-094'] for _ in range(4)] + ['AZK01-073'] * 2
DAGGER, REI, ED, FRIDA, STATION = 'AZK01-094', 'STT02-004', 'AZK01-012', 'AZK01-067', 'AZK01-103'
LEADERS = {'raizan': 'STT01-001', 'piko': 'AZK01-119'}


def specs():
    return {key: dict(checkpoint=str(path), checkpoint_sha256=sha256(path), deck_pool=str(V3_POOL)) for key, path in POLICIES.items()}


class Fixture:
    def __init__(self, runner, seed, seat, leader, observer=None):
        self.runner, self.seed, self.seat, self.leader = runner, seed, seat, leader
        self.observer = observer
        self.log = GameLogger(runner, log_legal_actions=True)
        self.actions = []
        self.terminal = False
        self.winner = -1
        normal = [REI, FRIDA, ED, STATION]
        for code, def_id in sorted(runner.code_to_def.items()):
            rec = runner.catalog.records_by_def_id[def_id]
            if rec.card_type == 'ENTITY' and rec.element in ('NORMAL', 'EARTH') and code not in normal:
                normal.append(code)
        self.decks = [MAIN if i == seat else [c for c in normal for _ in range(4)][:50] for i in range(2)]
        assert all(len(deck) == 50 for deck in self.decks)
        initial = []
        for i in range(2):
            state = PlayerDeckBuildState.create(runner.code_to_def['AZK01-120' if i == seat else 'STT03-002'])
            state.leader_card_def_id = runner.code_to_def[LEADERS[leader] if i == seat else 'STT03-001']
            initial.append(state)
        runner.base_env._initial_states = lambda: copy.deepcopy(initial)
        runner.vecenv.async_reset(seed=seed)
        self.recv()

    def recv(self):
        self.obs, _, term, trunc, _, _, self.masks = self.runner.vecenv.recv()
        self.terminal = bool(np.asarray(term).any() or np.asarray(trunc).any())
        if self.terminal:
            self.winner = terminal_winner(self.runner.base_env.env, terminated=bool(np.asarray(term).any()), truncated=bool(np.asarray(trunc).any()))

    @property
    def actor(self):
        return int(self.runner.base_env._active_player_index)

    def raw(self, seat=None):
        return self.runner.base_env.env._raw_observation(self.seat if seat is None else seat)

    def snap(self, seat=None):
        seat = self.seat if seat is None else seat
        snap = self.log.snapshot(seat)
        snap['garden_flags'] = [dict(slot=int(c.zone_index), card=self.log.code(c.card_def_id), tapped=bool(c.tap_state.tapped), cooldown=bool(c.tap_state.cooldown), charge=bool(c.has_charge), attack=int(c.cur_stats.cur_atk), hp=int(c.cur_stats.cur_hp), weapons=int(c.weapon_count)) for c in self.raw(seat).my_observation_data.garden if c.card_def_id >= 0]
        return snap

    def rows(self):
        return self.log.snapshot(self.actor)['legal']

    def step(self, action):
        assert not self.terminal
        action = [int(v) for v in action]
        building = bool(self.runner.base_env._building)
        if building:
            context = self.runner.base_env._deck_context_for_player(self.actor, include_candidates=True)
            assert action[0] == int(ActionType.DECK_PICK_CARD) and 0 <= action[1] < int(context['candidate_count'])
        else:
            assert action in self.rows(), (action, self.actor, self.snap(self.actor))
        if self.observer is not None and self.actor == self.seat:
            self.observer(self)
        self.actions.append({'seat': self.actor, 'draft': building, 'action': action})
        actions = np.zeros((2, 4), dtype=np.int32)
        actions[self.actor] = action
        self.runner.vecenv.send(actions)
        self.recv()
        assert len(self.actions) < 1000, 'fixture exceeded action cap'

    def draft(self):
        while self.runner.base_env._building:
            actor = self.actor
            count = self.runner.base_env._states[actor].main_count
            desired = self.runner.code_to_def[self.decks[actor][count]]
            candidates, _ = self.runner.base_env._candidate_def_ids(actor)
            assert desired in candidates, (self.decks[actor][count], actor, count)
            self.step([int(ActionType.DECK_PICK_CARD), list(candidates).index(desired), 0, 0])
        self.to_main(self.seat)

    def is_main(self, seat=None):
        seat = self.seat if seat is None else seat
        raw = self.raw(seat)
        return self.actor == seat and int(raw.phase) == 2 and int(raw.ability_context.phase) == 0

    def default(self):
        rows = self.rows()
        return next((a for a in rows if a[0] == 0), rows[0])

    def settle(self):
        while not self.terminal and not self.is_main(self.actor):
            self.step(self.default())

    def to_main(self, seat):
        while not self.terminal and not self.is_main(seat):
            self.step(self.default())
        assert not self.terminal, 'game ended during legal setup'

    def next_turn(self, seat=None):
        seat = self.seat if seat is None else seat
        self.to_main(seat)
        self.step([0, 0, 0, 0])
        self.to_main(seat)

    def hand(self, seat=None):
        return self.snap(seat)['hand']

    def slot(self, code, zone='garden', seat=None):
        cards = getattr(self.raw(seat).my_observation_data, zone)
        return next((int(c.zone_index) for c in cards if c.card_def_id >= 0 and self.log.code(c.card_def_id) == code), None)

    def resources(self):
        snap = self.snap()
        return snap['ikz'][0] + int(snap['ikz_token'])

    def wait(self, predicate):
        for _ in range(60):
            if predicate():
                return
            assert self.snap()['my_deck_n'] >= 30, 'fixture prerequisites not drawn within first twenty cards'
            self.next_turn()
        raise AssertionError('setup predicate unavailable within 60 turns')

    def play(self, code, zone='garden', seat=None):
        seat = self.seat if seat is None else seat
        self.to_main(seat)
        index = self.hand(seat).index(code)
        action_type = 1 if zone == 'garden' else 2
        row = next(a for a in self.rows() if a[0] == action_type and a[1] == index)
        slot = row[2]
        self.step(row)
        self.settle()
        return slot

    def equip(self, slot):
        index = self.hand().index(DAGGER)
        self.step(next(a for a in self.rows() if a[0] == 7 and a[1] == index and a[2] == slot))
        self.settle()

    def attack(self, slot, target=5):
        self.step(next(a for a in self.rows() if a[0] == 6 and a[1] == slot and a[2] == target))
        self.settle()

    def activate(self, slot):
        self.step(next(a for a in self.rows() if a[0] == 11 and a[1] == 5))
        rows = self.rows()
        selected = next(a for a in rows if a[0] == 14 and a[1] == slot)
        self.step(selected)
        self.settle()

    def gate_sink(self):
        assert FRIDA in self.hand()
        slot = self.play(FRIDA, 'alley')
        self.step(next(a for a in self.rows() if a[0] == 10 and a[1] == slot))
        self.settle()
        assert self.snap()['gate_tapped']

    def spend_to(self, remaining):
        amount = self.resources() - remaining
        assert amount >= 0
        hand = self.hand()
        options = [(i, code, self.runner.catalog.records_by_def_id[self.runner.code_to_def[code]].ikz_cost) for i, code in enumerate(hand) if code in VANILLA]
        solutions = {0: []}
        for i, code, cost in options:
            for total, chosen in list(solutions.items()):
                if total + cost <= amount and (total + cost not in solutions or len(solutions[total + cost]) > len(chosen) + 1):
                    solutions[total + cost] = chosen + [code]
        assert amount in solutions, ('cannot legally spend to budget', amount, hand)
        for code in solutions[amount]:
            self.play(code, 'alley')
        assert self.resources() == remaining

    def save(self, name, goal, target, notes):
        assert self.is_main()
        snap = self.snap()
        assert snap['hand_n'] <= 30 and self.snap(1 - self.seat)['hand_n'] <= 30, 'fixture exceeds observation hand capacity'
        case = dict(id=f'{self.leader}_{name}_seat{self.seat}_seed{self.seed}', name=name, leader=self.leader, seat=self.seat, seed=self.seed, goal=goal, target=target, notes=notes, decks=self.log.deck_record(), prefix=copy.deepcopy(self.actions), checkpoint=snap)
        return case


def setup(runner, seed, seat, leader, trade=False, zero=False, unnecessary=False):
    game = Fixture(runner, seed, seat, leader)
    game.draft()
    needed = [REI, FRIDA] + ([] if trade else [ED])
    game.wait(lambda: all(c in game.hand() for c in needed) and game.hand().count(DAGGER) >= (1 if zero else 4) and game.resources() >= 10)
    payload = None
    if leader == 'piko':
        payload = game.play(REI)
    workhorse = None if trade else game.play(ED)
    game.next_turn()
    if not zero:
        for _ in range(3):
            game.equip(5 if trade else workhorse)
            game.attack(5 if trade else workhorse)
            game.next_turn()
    if not trade:
        attacks = (5 if zero else 1) if leader == 'piko' else 2
        for i in range(attacks):
            game.attack(workhorse)
            if i + 1 < attacks or unnecessary:
                game.next_turn()
        assert game.snap()['opp_hp'] == (5 if leader == 'piko' else 2)
        if unnecessary and leader == 'piko':
            game.attack(workhorse)
            game.attack(payload)
            game.next_turn()
            assert game.snap()['opp_hp'] == 1
    else:
        opponent = 1 - seat
        target_code = REI if leader == 'raizan' else STATION
        game.to_main(opponent)
        for _ in range(50):
            if target_code in game.hand(opponent) and game.snap(opponent)['ikz'][0] >= 5:
                break
            game.next_turn(opponent)
        target_slot = game.play(target_code, seat=opponent)
        game.next_turn(opponent)
        game.attack(target_slot)
        game.to_main(seat)
    if leader == 'raizan':
        payload = game.play(REI)
    game.gate_sink()
    game.spend_to(2 if leader == 'raizan' else 4)
    return game, payload


def generate(runner, seeds):
    cases, failures, accepted = [], [], []
    for seat in (0, 1):
        found = 0
        for seed in seeds:
            batch = []
            try:
                for leader in LEADERS:
                    game, payload = setup(runner, seed, seat, leader)
                    batch.append(game.save('equipment_lethal', 'win', payload, 'Only remaining weapon is in hand; compare body versus leader equipment.'))
                    game.equip(payload)
                    batch.append(game.save('ability_lethal', 'win', payload, 'Final damage requires enabling this body; other genuine wins also count.'))
                    if leader == 'piko':
                        game.attack(payload)
                        batch.append(game.save('ability_tapped', 'restraint', payload, 'Equipped body already attacked; end-of-turn attack buff cannot yield another attack.'))
                        game, payload = setup(runner, seed, seat, leader, zero=True)
                        game.equip(payload)
                        batch.append(game.save('ability_zero', 'restraint', payload, 'Zero discarded weapons; activation consumes three IKZ for zero attack.'))
                    game, payload = setup(runner, seed, seat, leader, unnecessary=True)
                    batch.append(game.save('equipment_equivalent_win', 'win', payload, 'A ready attacker already wins without equipment; either equipment host can still win.'))
                    game.equip(payload)
                    batch.append(game.save('ability_equivalent_win', 'win', payload, 'A ready attacker already wins; activation is optional, not the objective.'))
                    game, payload = setup(runner, seed, seat, leader, trade=True)
                    target = game.slot(REI if leader == 'raizan' else STATION, seat=1-seat)
                    batch.append(game.save('equipment_trade', 'trade', target, 'Compare body versus leader equipment; remove the tapped enemy and preserve the attacker.'))
                    game.equip(payload)
                    batch.append(game.save('ability_trade', 'trade', target, 'Ability enables favorable removal this turn.'))
                cases.extend(batch)
                accepted.append({'seat': seat, 'seed': seed, 'case_count': len(batch)})
                found += 1
                print(json.dumps({'generated_seat': seat, 'seed': seed, 'cases': len(batch)}), flush=True)
                (OUT / 'cases.json').write_text(json.dumps(cases, indent=2))
                if found == 2:
                    break
            except (AssertionError, StopIteration, ValueError) as exc:
                failures.append({'seat': seat, 'seed': seed, 'error': str(exc), 'traceback': traceback.format_exc()})
                (OUT / 'fixture_generation_failures.json').write_text(json.dumps(failures, indent=2))
                print(json.dumps({'rejected_seat': seat, 'seed': seed, 'reason': traceback.format_exc()[-500:]}), flush=True)
        assert found == 2, (seat, failures[-5:])
    (OUT / 'fixture_generation.json').write_text(json.dumps({'accepted': accepted, 'rejected': failures, 'selection_rule': 'first two seeds per seat where all legal fixtures construct; no policy outputs consulted'}, indent=2))
    return cases


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--generate', action='store_true')
    args.add_argument('--seed-start', type=int, default=101)
    args.add_argument('--seed-stop', type=int, default=1101)
    opts = args.parse_args()
    runner = EpisodeRunner(POLICIES['u8223'], V3_POOL, 'cpu')
    try:
        if opts.generate:
            generate(runner, range(opts.seed_start, opts.seed_stop))
    finally:
        runner.vecenv.close()

if __name__ == '__main__':
    main()
