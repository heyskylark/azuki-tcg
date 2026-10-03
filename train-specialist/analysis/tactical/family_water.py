"""WATER family: Shao leader (STT02-001, [Response] pay 1 IKZ: -1 attack) + Hydromancy (STT02-002, untap IKZ = gate power).

Tested seat plays curated Shao/Hydromancy deck 11 ('inv_aez_shao_v1'; no free Aquatic Veil); the opponent
plays curated Goro deck 13 and only attacks where a case scripts it (aggressive script: every ready entity
attacks the leader, highest attack first).
"""
from tb_fixture import NOOP
from tb_setup import TF, Reject, ST
from tactical import op_leader, op_portal, op_equip, op_maybe_confirm, first, opp_turn_action

FAMILY = 'water'
DECK, OPP = 11, 13
FODDER = ['AZK01-021', 'STT02-003', 'STT02-008', 'STT02-012', 'STT02-013']
SPEND = FODDER + ['AZK01-024', 'AZK01-022']
BODIES = ['STT02-012', 'STT02-013', 'AZK01-014']  # gate power 2
TENSHIN = 'STT01-014'
OPP_ATTACKERS = ['AZK01-069', 'AZK01-012', 'AZK01-019', 'AZK01-050', 'STT02-004', 'AZK01-105']


def setup_hydro(g, remaining):
    def ready():
        hand = g.hand()
        return TENSHIN in hand and any(b in hand for b in BODIES)
    g.wait_for(ready)
    body = next(b for b in BODIES if b in g.hand())
    keep = {body, TENSHIN}
    g.chip_opp(1, fodder=[c for c in FODDER if c not in keep])
    if body not in g.hand() or TENSHIN not in g.hand() or g.ikz() < ST[body]['cost'] + remaining:
        raise Reject('hydromancy pieces unavailable on checkpoint turn')
    g.play(body, 'alley')
    g.spend_to(remaining, [c for c in SPEND if c not in keep], zone='garden')
    if g.ready() or g.snap()['gate_tapped']:
        raise Reject('hydromancy checkpoint invalid')
    return body


def build(runner, seed, seat):
    cases = []
    g = TF(runner, seed, seat, DECK, OPP)
    body = setup_hydro(g, 0)
    cases.append(g.save(FAMILY, 'hydro_tenshin_lethal', 'win', f'Opponent at 1, IKZ 0, no attacker. Portal {body} (gate power 2): Hydromancy untaps 2 IKZ, cast Tenshin (2) - its on-play ping or the equipped leader attack is lethal. Without the portal nothing is castable.',
                        horizon='turn', portal=body, leader_needed=False, gate_needed=True))
    g = TF(runner, seed, seat, DECK, OPP)
    body = setup_hydro(g, 2)
    cases.append(g.save(FAMILY, 'hydro_equivalent', 'win', 'Same board with 2 IKZ already open: Tenshin is castable without the portal (control; portal optional).',
                        horizon='turn', portal=body, leader_needed=False, gate_needed=False))
    # W4 hold_ikz (main checkpoint) and W3 shao_response (response checkpoint on the hold line)
    g = TF(runner, seed, seat, DECK, OPP)
    g.chip_me(OPP_ATTACKERS, target=None)
    if 'AZK01-029' in g.hand():
        raise Reject('free Aquatic Veil alternative')
    g.gate_sink(SPEND)  # tap Hydromancy now so it cannot refund IKZ later this turn
    g.spend_to(1, SPEND, zone='alley')
    if g.has_friendly_defender():
        raise Reject('friendly defender on board')
    if g.snap()['ikz_token']:
        raise Reject('token adds a second resource')
    if not any(ST.get(c, {}).get('cost', 9) <= 1 and ST[c]['type'] in ('ENTITY', 'SPELL', 'WEAPON') for c in g.hand()):
        raise Reject('no 1-cost temptation in hand')
    total = g.opp_total_attack()
    cases.append(g.save(FAMILY, 'hold_ikz', 'survive', f'Opponent board attacks for exactly our {total} HP next turn. Holding the last IKZ for Shao (-1 attack in response) survives; spending it loses.',
                        horizon='opp_turn', leader_needed=True, gate_needed=False, opp_total_attack=total))
    g.step(NOOP)
    for _ in range(40):
        if g.actor == g.seat and not g.is_main():
            break
        if g.terminal:
            raise Reject('terminal before response')
        g.step(opp_turn_action(g) if g.is_main(g.opp) else g.default())
    if not any(a[0] == 11 and a[1] == 5 for a in g.rows()):
        raise Reject('Shao not legal at response window')
    cases.append(g.save(FAMILY, 'shao_response', 'survive', 'Opponent has declared the first of its attacks; total incoming damage equals our HP. Activate Shao (1 IKZ) to reduce an attacker by 1 and survive.',
                        horizon='opp_turn', checkpoint='response', leader_needed=True, gate_needed=False))
    return cases


def op_tenshin_ping():
    def op(game):
        rows = game.rows()
        for want in (11, 5):
            row = next((a for a in rows if a[0] == 14 and a[1] == want), None)
            if row is not None:
                return row
        return 'skip'
    return op


def op_shao_target():
    return first(lambda g, a: a[0] == 14 and a[1] < 5)


def op_spend_one():
    def op(game):
        hand = game.hand()
        return next((a for a in game.rows() if a[0] in (1, 2) and a[1] < len(hand) and ST[hand[a[1]]]['cost'] == 1), None)
    return op


def plans(case):
    p = case['params']
    name = case['name']
    tenshin_line = [op_equip(TENSHIN), op_maybe_confirm(), op_tenshin_ping(), first(lambda g, a: a[0] == 6 and a[1] == 5 and a[2] == 5)]
    if name in ('hydro_tenshin_lethal', 'hydro_equivalent'):
        return {'with_gate': [op_portal(p['portal'])] + tenshin_line, 'without_gate': list(tenshin_line)}
    if name == 'hold_ikz':
        return {'hold_then_shao': [op_leader(), op_shao_target()], 'spend_then_shao': [op_spend_one(), op_leader(), op_shao_target()],
                'spend_then_block': [op_spend_one(), first(lambda g, a: a[0] == 9)], 'hold_no_shao': []}
    if name == 'shao_response':
        return {'with_ability': [op_leader(), op_shao_target()], 'without_ability': []}
    raise KeyError(name)


def check_oracle(case, rows):
    ok = {k: v['success'] for k, v in rows.items()}
    name = case['name']
    if name == 'hydro_equivalent':
        assert all(ok.values()), (case['id'], ok)
    else:
        positive = {'hydro_tenshin_lethal': 'with_gate', 'hold_ikz': 'hold_then_shao', 'shao_response': 'with_ability'}[name]
        assert ok[positive] and not any(v for k, v in ok.items() if k != positive), (case['id'], ok)
    if name == 'hold_ikz':
        assert rows['hold_then_shao']['leader_uses'] == 1 and rows['spend_then_shao']['leader_uses'] == 0, (case['id'], 'spend branch must lose access to Shao')
