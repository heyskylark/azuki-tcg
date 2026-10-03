"""LIGHTNING (constructed) family: Raizan (STT01-001, pay 1 IKZ: Charge to an equipped entity) and Surge
(STT01-002, portal -> play a weapon from the discard).  Complements the ported draft-history benchmark with the
specialists' 80% regime: curated Raizan decks 4 ('case1_banchan', Stormchain) and 10 ('inv_ducky', Surge) vs
curated Bobu deck 14 (passive)."""
from tb_setup import TF, Reject, ST
from tactical import op_leader, op_target_code, op_portal, op_attack_face, op_equip, first

FAMILY = 'lightning_c'
STORM, SURGE, OPP = 4, 10, 14
HOSTS = ['AZK01-011', 'STT01-004', 'STT02-004', 'STT01-005', 'STT01-003', 'AZK01-033', 'STT03-008']
FODDER = ['STT01-004', 'STT01-003', 'STT02-004', 'AZK01-011', 'STT01-005', 'AZK01-033', 'STT03-008']
WEAPONS = ['AZK01-094', 'STT01-012']  # 1-cost, +1 attack, no on-play prompt
SPEND = FODDER + ['STT01-009', 'AZK01-040']


def op_maybe(type_):
    def op(game):
        return next((a for a in game.rows() if a[0] == type_), 'skip')
    return op


def op_equip_from_selection(host):
    def op(game):
        garden = game.garden()
        return next((a for a in game.rows() if a[0] == 22 and a[2] in garden and game.code(garden[a[2]].card_def_id) == host), None)
    return op


def build(runner, seed, seat):
    cases = []
    # L1 raizan_charge_lethal
    g = TF(runner, seed, seat, STORM, OPP)

    def ready1():
        hand = g.hand()
        return any(h in hand for h in HOSTS) and any(w in hand for w in WEAPONS)
    g.wait_for(ready1)
    host = max((h for h in HOSTS if h in g.hand()), key=lambda h: ST[h]['attack'])  # 2-attack hosts keep BJ Dagger-on-leader (2) short
    weapon = next(w for w in WEAPONS if w in g.hand())
    target = ST[host]['attack'] + ST[weapon]['attack']
    keep = {host, weapon}
    g.chip_opp(target, fodder=[c for c in FODDER if c not in keep])
    if host not in g.hand() or weapon not in g.hand() or g.ikz() < ST[host]['cost'] + ST[weapon]['cost'] + 2:
        raise Reject('raizan pieces unavailable on checkpoint turn')
    hslot = g.play(host)
    hand = g.hand()
    g.step(next(a for a in g.rows() if a[0] == 7 and a[1] == hand.index(weapon) and a[2] == hslot))
    g.settle()
    g.gate_sink([c for c in FODDER if c != host])  # Stormchain re-equip would otherwise offer weapon moves
    sinks = tuple(s for s, f in g.flags().items() if f['tapped'])
    g.spend_to(1, SPEND + ['AZK01-001', 'STT01-006'] + WEAPONS, zone='alley', weapon_sinks=sinks)
    if g.ready():
        raise Reject('raizan checkpoint invalid: another ready attacker')
    if 'STT01-013' in g.hand() and target <= 2:
        raise Reject('Black Jade Dagger on the leader (2 attack for 1 IKZ) would be a Raizan-free lethal')
    cases.append(g.save(FAMILY, 'raizan_charge_lethal', 'win', f'{host} entered this turn with {weapon} equipped (cooldown); opponent at exactly its attack ({target}). Raizan (1 IKZ, all that is left) gives it Charge -> attack for lethal.',
                        horizon='turn', host=host, leader_needed=True, gate_needed=False))
    # L2 surge_lethal
    g = TF(runner, seed, seat, SURGE, OPP)
    g.wait_for(lambda: any(h in g.hand() for h in HOSTS) and any(w in g.hand() for w in WEAPONS) and g.ikz() >= 2)
    host = next(h for h in HOSTS if h in g.hand())
    hslot = g.play(host)
    weapon = next(w for w in WEAPONS if w in g.hand())
    g.step(next(a for a in g.rows() if a[0] == 7 and a[1] == g.hand().index(weapon) and a[2] == hslot))
    g.settle()
    g.cycle()  # weapon goes to the discard at end of turn; host becomes ready
    if not any(c in WEAPONS for c in g.snap()['my_discard']):
        raise Reject('no cheap weapon in discard')

    def body_ok():
        return any(c in SPEND and ST[c]['gp'] >= 1 for c in g.hand())
    g.wait_for(body_ok)
    target = ST[host]['attack'] + 1
    g.chip_opp(target, reserve=(hslot,), fodder=[c for c in FODDER if c != host])
    bodies = [c for c in g.hand() if c in SPEND and ST[c]['gp'] >= 1 and ST[c]['cost'] <= g.ikz()]
    if not bodies:
        raise Reject('no surge body on checkpoint turn')
    body = min(bodies, key=lambda c: ST[c]['cost'])
    g.play(body, 'alley')
    g.spend_to(0, [c for c in SPEND], zone='garden')
    if g.ready() != [(hslot, ST[host]['attack'])] or g.snap()['gate_tapped']:
        raise Reject('surge checkpoint invalid')
    cases.append(g.save(FAMILY, 'surge_lethal', 'win', f'Ready {host} is one damage short; IKZ 0. Portal {body} with Surge, play a 1-cost weapon from the discard onto {host}, attack for exact lethal.',
                        horizon='turn', host=host, portal=body, leader_needed=False, gate_needed=True))
    return cases


def plans(case):
    p = case['params']
    if case['name'] == 'raizan_charge_lethal':
        leader_weapon = [first(lambda g, a: a[0] == 7 and a[2] == 5), op_maybe(16), first(lambda g, a: a[0] == 6 and a[1] == 5 and a[2] == 5)]
        return {'with_ability': [op_leader(), op_target_code(p['host']), op_attack_face(p['host'])], 'without_ability': [op_attack_face()],
                'leader_weapon': leader_weapon}
    if case['name'] == 'surge_lethal':
        return {'with_gate': [op_portal(p['portal']), op_maybe(16), op_maybe(18), op_equip_from_selection(p['host']), op_attack_face(p['host'])],
                'without_gate': [op_attack_face(p['host'])]}
    raise KeyError(case['name'])


def check_oracle(case, rows):
    ok = {k: v['success'] for k, v in rows.items()}
    positive = {'raizan_charge_lethal': 'with_ability', 'surge_lethal': 'with_gate'}[case['name']]
    assert ok[positive] and not any(v for k, v in ok.items() if k != positive), (case['id'], ok)
