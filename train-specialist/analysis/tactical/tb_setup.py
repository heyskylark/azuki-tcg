"""Legal setup primitives shared by the Fire/Water/Earth fixture families.

Everything here only issues actions that appear in the engine's legal list for
the acting seat.  The tested seat and the opponent are both scripted during
setup; no policy output is consulted.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from tb_fixture import FixedFixture, GARDEN, NOOP

ST = json.loads((Path(__file__).resolve().parent / 'card_stats.json').read_text())
# entities with native Defender (official card JSON + engine prefab tags): never placed by setup spending
DEFENDERS = {'AZK01-001', 'AZK01-025', 'AZK01-035', 'AZK01-040', 'AZK01-049', 'AZK01-051', 'AZK01-052', 'AZK01-106', 'AZK01-129', 'STT02-006', 'STT03-004', 'STT03-009'}


class Reject(AssertionError):
    """Seed cannot construct the requested legal fixture."""


def subset_exact(items, target):
    """items: list[(key, value)] -> subset of keys summing exactly to target (fewest items), else None."""
    best = None
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            if sum(v for _, v in combo) == target:
                return [k for k, _ in combo]
    return best


def subset_max(items, cap):
    best, best_sum = [], 0
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            total = sum(v for _, v in combo)
            if best_sum < total <= cap:
                best, best_sum = [k for k, _ in combo], total
    return best


class TF(FixedFixture):
    # ---- queries -------------------------------------------------------------
    def flags(self, seat=None):
        seat = self.seat if seat is None else seat
        return {f['slot']: f for f in self.snap(seat)['garden_flags']}

    def opp_hp(self):
        return self.snap()['opp_hp']

    def face_rows(self):
        assert self.is_main(self.actor)
        return [a for a in self.rows() if a[0] == 6 and a[2] == GARDEN]

    def ready(self, exclude=()):
        """[(slot, attack)] for the actor's garden entities that may attack the leader now."""
        flags = self.flags(self.actor)
        return [(a[1], flags[a[1]]['attack']) for a in self.face_rows() if a[1] != GARDEN and a[1] not in exclude and flags[a[1]]['attack'] > 0]

    def attack_slots(self, slots, target=GARDEN):
        for s in slots:
            self.attack(s, target)
            if self.terminal:
                return

    # ---- playing ---------------------------------------------------------------
    def affordable_entities(self, allowed, seat=None):
        seat = self.seat if seat is None else seat
        hand = self.hand(seat)
        ikz = self.ikz(seat)
        return [c for c in hand if c in allowed and ST[c]['type'] == 'ENTITY' and ST[c]['cost'] <= ikz]

    def garden_count(self, seat=None):
        return len(self.flags(seat))

    def develop(self, allowed, budget_atk, seat=None, max_entities=5, zone='garden'):
        """Play cheap allowed entities to the garden while their total attack stays <= budget_atk."""
        seat = self.seat if seat is None else seat
        added = 0
        while True:
            self.to_main(seat)
            if self.garden_count(seat) >= max_entities:
                return added
            options = sorted(self.affordable_entities(allowed, seat), key=lambda c: (ST[c]['cost'], -ST[c]['attack']))
            options = [c for c in options if added + ST[c]['attack'] <= budget_atk and ST[c]['attack'] > 0]
            if not options:
                return added
            self.play(options[0], 'garden', seat=seat)
            added += ST[options[0]]['attack']

    def cycle(self, our=None, opp=None):
        """Finish our turn (after optional our()), run the opponent turn (opp() at its main), return to our main."""
        self.to_main(self.seat)
        if our:
            our()
        self.to_main(self.seat)
        self.step(NOOP)
        self.to_main(self.opp)
        if opp:
            opp()
        self.to_main(self.opp)
        self.step(NOOP)
        self.to_main(self.seat)

    def wait_for(self, predicate, opp=None, our=None, limit=25):
        for _ in range(limit):
            self.to_main(self.seat)
            if predicate():
                return
            if self.snap()['my_deck_n'] < 12:
                break
            self.cycle(our=our, opp=opp)
        raise Reject('setup predicate unavailable')

    # ---- HP shaping --------------------------------------------------------------
    def chip_opp(self, target, reserve=(), fodder=(), opp=None):
        """Attack the (passive) opponent leader over turns so that on the returned (our main) turn the opponent
        is exactly at `target` and every non-reserved ready attacker has already attacked this turn."""
        for _ in range(30):
            self.to_main()
            ready = self.ready(exclude=reserve)
            S = sum(v for _, v in ready)
            gap = self.opp_hp() - target
            if gap < 0:
                raise Reject('opponent below target')
            if gap == S and (S > 0 or not ready):
                self.attack_slots([s for s, _ in ready])
                assert self.opp_hp() == target and not self.terminal
                return
            if gap < S:
                raise Reject('chip overshoot: too many ready attackers')
            x = gap - S if gap - S <= S else S
            chosen = subset_exact(ready, x) or subset_max(ready, x)
            self.attack_slots(chosen)
            spent = sum(v for s, v in ready if s in chosen)
            gap_next = gap - spent
            self.develop(fodder, budget_atk=max(0, gap_next - S), max_entities=3)
            self.cycle(opp=opp)
        raise Reject('chip_opp did not converge')

    def opp_total_attack(self):
        return sum(f['attack'] for f in self.flags(self.opp).values() if f['attack'] > 0)

    def chip_me(self, opp_allowed, target=None, max_opp=4, our=None, offset=0):
        """Opponent develops whitelisted entities and attacks our leader.  target=None: finish when
        my_hp == the opponent's total garden attack (survival design); else finish at my_hp == target.
        The finishing opponent turn makes no attack, so its entities are untapped on our checkpoint turn.
        Returns at our next main."""
        for _ in range(30):
            self.to_main(self.opp)
            total = self.opp_total_attack()
            my_hp = self.snap()['my_hp']
            want = total + offset if target is None else target
            room = my_hp - want
            if room < 0:
                raise Reject('my_hp below desired')
            if room == 0 and (target is not None or total > 0):
                self.step(NOOP)
                self.to_main(self.seat)
                return
            ready = self.ready()
            chosen = subset_exact(ready, room) or subset_max(ready, room)
            for s in chosen:
                self.attack(s)
                self.to_main(self.opp)
            new_hp = self.snap()['my_hp']
            if new_hp <= 0 or self.terminal:
                raise Reject('setup killed tested leader')
            budget = (new_hp - total - offset) if target is None else min(12, new_hp - target)
            if self.garden_count(self.opp) < max_opp and budget > 0:
                self.develop(opp_allowed, budget_atk=budget, seat=self.opp, max_entities=max_opp)
            self.to_main(self.opp)
            self.step(NOOP)
            self.to_main(self.seat)
            if our:
                our()
                self.to_main(self.seat)
            self.step(NOOP)
        raise Reject('chip_me did not converge')

    # ---- resources -----------------------------------------------------------------
    def gate_sink(self, allowed):
        """Tap the gate by portaling a cheap allowed entity played to the alley this turn; decline the gate effect."""
        self.to_main()
        options = sorted(self.affordable_entities(allowed), key=lambda c: ST[c]['cost'])
        alley_before = set(self.alley())
        if not options:
            raise Reject('no gate-sink fodder')
        slot = self.play(options[0], 'alley')
        occupied = set(self.garden())
        portal = [a for a in self.rows() if a[0] == 10 and a[1] == slot and a[2] not in occupied]
        if not portal:
            raise Reject('gate portal unavailable')
        self.step(portal[0])
        self.settle()
        if not self.snap()['gate_tapped']:
            raise Reject('gate did not tap')
        return alley_before

    def spend_to(self, remaining, allowed, zone='alley', free_garden=2, weapon_sinks=()):
        """Spend IKZ down to `remaining` by playing allowed entities (and, if weapon_sinks are given, by equipping
        allowed weapons onto those already-tapped friendly entities, where they cannot add damage this turn)."""
        self.to_main()
        amount = self.ikz() - remaining
        if amount < 0:
            raise Reject('not enough IKZ')
        if amount == 0:
            return
        hand = self.hand()
        kinds = ('ENTITY', 'WEAPON') if weapon_sinks else ('ENTITY',)
        items = [(i, ST[c]['cost']) for i, c in enumerate(hand) if c in allowed and c not in DEFENDERS and ST[c]['type'] in kinds and ST[c]['cost'] > 0]
        chosen = subset_exact(items, amount)
        if chosen is None:
            raise Reject(('cannot legally spend to budget', amount))
        codes = [hand[i] for i in chosen]
        for code in codes:
            if ST[code]['type'] == 'WEAPON':
                idx = self.hand().index(code)
                row = next((a for a in self.rows() if a[0] == 7 and a[1] == idx and a[2] in weapon_sinks), None)
                if row is None:
                    raise Reject('weapon sink unavailable')
                self.step(row)
                self.settle()
                continue
            order = [zone, 'garden' if zone == 'alley' else 'alley']
            room = {'alley': len(self.alley()) < 5, 'garden': self.garden_count() < 5 - free_garden}
            target_zone = next((z for z in order if room[z]), None)
            if target_zone is None:
                raise Reject('no room to spend')
            self.play(code, target_zone)
        if self.ikz() != remaining:
            raise Reject(('spend drift', self.ikz(), remaining))

    def spend_at_most(self, alternatives, allowed, zone='alley', free_garden=2):
        """Spend down until every alternative card in hand is unaffordable (leftover < its cost)."""
        self.to_main()
        costs = [ST[c]['cost'] for c in self.hand() if c in alternatives]
        cap = min(costs) - 1 if costs else self.ikz()
        cap = min(cap, self.ikz())
        for remaining in range(cap, -1, -1):
            try:
                hand = self.hand()
                amount = self.ikz() - remaining
                items = [(i, ST[c]['cost']) for i, c in enumerate(hand) if c in allowed and c not in DEFENDERS and ST[c]['type'] == 'ENTITY' and ST[c]['cost'] > 0]
                if amount == 0 or subset_exact(items, amount) is not None:
                    self.spend_to(remaining, allowed, zone=zone, free_garden=free_garden)
                    return remaining
            except Reject:
                raise
        raise Reject(('cannot spend below alternatives', cap))

    def has_friendly_defender(self):
        return any(s.split(':')[1].endswith('D') or 'D+' in s.split(':')[1] for s in self.snap()['my_garden'])
