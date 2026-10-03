"""Constructed legal-history fixtures from curated constructed decks.

Same contract as leader_history_benchmark_v1.Fixture (every action is checked
against the engine's legal-action list for the recorded actor; the checkpoint is
a full observation snapshot incl. legal rows), but both seats start from
complete curated decks (the specialists' 80% prebuilt regime) instead of a
forced draft.  The world seed fixes the shuffle; setup code only waits for
cards to be drawn naturally and plays them legally.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from tb_runtime import CURATED_POOL, GameLogger, terminal_winner

DECKS = json.loads(CURATED_POOL.read_text())['decks']
GARDEN, LEADER_SLOT = 5, 5
NOOP = [0, 0, 0, 0]


def native_deck(index):
    return tuple((c['card_id'], int(c['quantity'])) for c in DECKS[index]['cards'])


class FixedFixture:
    def __init__(self, runner, seed, seat, my_deck, opp_deck, observer=None):
        self.runner, self.seed, self.seat = runner, seed, seat
        self.my_deck, self.opp_deck = my_deck, opp_deck
        self.observer = observer
        self.log = GameLogger(runner, log_legal_actions=True)
        self.actions = []
        self.terminal = False
        self.winner = -1
        base = runner.base_env
        decks = [my_deck if i == seat else opp_deck for i in range(2)]
        initial = [base._fixed_state_from_deck(native_deck(d)) for d in decks]
        base._initial_states = lambda: copy.deepcopy(initial)
        runner.vecenv.async_reset(seed=seed)
        self.recv()
        assert not base._building

    # ---- engine plumbing --------------------------------------------------
    def recv(self):
        self.obs, _, term, trunc, _, _, self.masks = self.runner.vecenv.recv()
        self.terminal = bool(np.asarray(term).any() or np.asarray(trunc).any())
        if self.terminal:
            self.winner = terminal_winner(self.runner.base_env.env, terminated=bool(np.asarray(term).any()),
                                          truncated=bool(np.asarray(trunc).any()))

    @property
    def actor(self):
        return int(self.runner.base_env._active_player_index)

    @property
    def opp(self):
        return 1 - self.seat

    def raw(self, seat=None):
        return self.runner.base_env.env._raw_observation(self.seat if seat is None else seat)

    def snap(self, seat=None):
        seat = self.seat if seat is None else seat
        snap = self.log.snapshot(seat)
        snap['garden_flags'] = [dict(slot=int(c.zone_index), card=self.log.code(c.card_def_id), tapped=bool(c.tap_state.tapped),
                                     cooldown=bool(c.tap_state.cooldown), charge=bool(c.has_charge), attack=int(c.cur_stats.cur_atk),
                                     hp=int(c.cur_stats.cur_hp), weapons=int(c.weapon_count))
                                for c in self.raw(seat).my_observation_data.garden if c.card_def_id >= 0]
        return snap

    def rows(self):
        return self.log.snapshot(self.actor)['legal']

    def step(self, action):
        assert not self.terminal
        action = [int(v) for v in action]
        assert action in self.rows(), (action, self.actor, self.snap(self.actor))
        if self.observer is not None and self.actor == self.seat:
            self.observer(self)
        self.actions.append({'seat': self.actor, 'draft': False, 'action': action})
        actions = np.zeros((2, 4), dtype=np.int32)
        actions[self.actor] = action
        self.runner.vecenv.send(actions)
        self.recv()
        assert len(self.actions) < 3000, 'fixture exceeded action cap'

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

    def to_main(self, seat=None):
        seat = self.seat if seat is None else seat
        while not self.terminal and not self.is_main(seat):
            self.step(self.default())
        assert not self.terminal, 'game ended during legal setup'

    def next_turn(self, seat=None):
        seat = self.seat if seat is None else seat
        self.to_main(seat)
        self.step(NOOP)
        self.to_main(seat)

    # ---- board queries ------------------------------------------------------
    def code(self, def_id):
        return self.log.code(def_id)

    def record(self, code):
        return self.runner.catalog.records_by_def_id[self.runner.code_to_def[code]]

    def hand(self, seat=None):
        return self.snap(seat)['hand']

    def garden(self, seat=None):
        return {int(c.zone_index): c for c in self.raw(seat).my_observation_data.garden if c.card_def_id >= 0}

    def alley(self, seat=None):
        return {int(c.zone_index): c for c in self.raw(seat).my_observation_data.alley if c.card_def_id >= 0}

    def slot(self, code, zone='garden', seat=None):
        cards = getattr(self.raw(seat).my_observation_data, zone)
        return next((int(c.zone_index) for c in cards if c.card_def_id >= 0 and self.code(c.card_def_id) == code), None)

    def ikz(self, seat=None):
        snap = self.snap(seat)
        return snap['ikz'][0] + int(snap['ikz_token'])

    def hp(self, seat=None):
        return self.snap(seat)['my_hp']

    # ---- legal actions --------------------------------------------------------
    def play(self, code, zone='garden', seat=None, slot=None):
        seat = self.seat if seat is None else seat
        self.to_main(seat)
        index = self.hand(seat).index(code)
        action_type = 1 if zone == 'garden' else 2
        occupied = set(self.garden(seat) if zone == 'garden' else self.alley(seat))
        rows = [a for a in self.rows() if a[0] == action_type and a[1] == index and (slot is None or a[2] == slot)]
        free = [a for a in rows if a[2] not in occupied]
        assert free, ('no free slot; refusing to replace a card', zone, code)
        row = free[0]
        self.step(row)
        self.settle()
        return row[2]

    def spell(self, code, seat=None):
        seat = self.seat if seat is None else seat
        index = self.hand(seat).index(code)
        self.step(next(a for a in self.rows() if a[0] == 8 and a[1] == index))

    def attack(self, slot, target=GARDEN):
        self.step(next(a for a in self.rows() if a[0] == 6 and a[1] == slot and a[2] == target))
        self.settle()

    def ready_attackers(self, seat=None):
        """Garden slots (and 5 = leader) holding a legal face-attack row."""
        return sorted({a[1] for a in self.rows() if a[0] == 6 and a[2] == GARDEN}) if self.is_main(seat) else []

    def save(self, family, name, goal, notes, **params):
        assert self.is_main() or params.get('checkpoint') == 'response'
        snap = self.snap()
        assert snap['hand_n'] <= 30 and self.snap(self.opp)['hand_n'] <= 30, 'fixture exceeds observation hand capacity'
        return dict(id=f'{family}_{name}_seat{self.seat}_seed{self.seed}', family=family, name=name, seat=self.seat, seed=self.seed,
                    my_deck=self.my_deck, opp_deck=self.opp_deck, goal=goal, notes=notes, params=params,
                    decks=self.log.deck_record(), prefix=copy.deepcopy(self.actions), checkpoint=snap)


def replay(runner, case, observer=None, cls=FixedFixture):
    game = cls(runner, case['seed'], case['seat'], case['my_deck'], case['opp_deck'], observer=observer)
    for row in case['prefix']:
        assert game.actor == row['seat'], ('actor drift', case['id'], len(game.actions))
        game.step(row['action'])
    assert game.snap() == case['checkpoint'], ('checkpoint replay differs', case['id'])
    return game
