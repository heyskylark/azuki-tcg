"""EARTH family: Bobu (STT03-001) / Goro (AZK01-123) + Stonehaven (STT03-002) + Quicksand (STT03-016).

Tested seat: curated Bobu/Stonehaven decks 14 ('pasadena_cat') and 15 ('case1_bobu_3rd'), curated Goro deck 13.
Opponents: curated Shao deck 11 (has Foamback Crab defenders) or curated Zero deck 6.
"""
from tb_fixture import NOOP, GARDEN
from tb_setup import TF, Reject, ST
from tactical import (op_leader, op_portal, op_target_code, op_attack_face, op_attack_slot_code, op_spell, op_defend,
                      op_maybe_confirm, first, opp_turn_action)

FAMILY = 'earth'
CAT, BOBU3, GORO = 14, 15, 13
OPP_W, OPP_F = 11, 6
OPP_W_ATTACKERS = ['AZK01-021', 'STT02-003', 'STT02-008', 'STT02-013']
FOAMBACK = 'STT02-006'
QUICKSAND = 'STT03-016'
# no heals (Shroom Tender, Mo), no free responses (Mocking Dummy), no Root (Sandcoil), no Taunt/self-tap (Stone Masked)
SAFE = ['STT03-003', 'AZK01-045', 'STT03-005', 'AZK01-048', 'STT03-012', 'AZK01-054', 'AZK01-047']  # Warding Totem has native Defender
STONE_BODIES = [c for c in SAFE if ST[c]['health'] <= ST[c]['gp']]
FODDER = ['STT03-003', 'AZK01-045', 'AZK01-048']
# cards that would offer a non-Stonehaven survival line if affordable
ALTS = ['AZK01-128', 'STT03-015', 'STT03-016', 'STT03-017', 'AZK01-050', 'AZK01-015', 'AZK01-002', 'STT03-014', 'AZK01-070']


def setup_stonehaven(g, offset):
    g.chip_me(OPP_W_ATTACKERS, target=None, offset=offset)
    if 'AZK01-070' in [g.code(c.card_def_id) for c in g.garden().values()]:
        raise Reject('Mocking Dummy response alternative')
    bodies = [c for c in g.hand() if c in STONE_BODIES and ST[c]['cost'] <= g.ikz()]
    if not bodies:
        raise Reject('no Stonehaven body')
    body = min(bodies, key=lambda c: ST[c]['cost'])
    g.play(body, 'alley')
    g.spend_at_most(ALTS, [c for c in SAFE if c != body], zone='garden')
    if g.snap()['gate_tapped'] or g.has_friendly_defender():
        raise Reject('stonehaven checkpoint invalid')
    return body


def build(runner, seed, seat):
    cases = []
    # E1 stonehaven_block + E2 declare_defender (response checkpoint on the oracle line)
    g = TF(runner, seed, seat, CAT, OPP_W)
    body = setup_stonehaven(g, 0)
    total = g.opp_total_attack()
    cases.append(g.save(FAMILY, 'stonehaven_block', 'survive', f'Opponent attacks next turn for exactly our {total} HP; no defender, IKZ 0. Portal {body} with Stonehaven, grant it Defender (base HP <= gate power), then block an attack to survive.',
                        horizon='opp_turn', portal=body, leader_needed=False, gate_needed=True, opp_total_attack=total))
    # follow the oracle line to the first defender decision
    from tactical import PlanChooser
    chooser = PlanChooser([op_portal(body), op_maybe_confirm(), op_target_code(body)])
    while g.is_main() or g.actor == g.seat and not g.is_main() and g.snap().get('ability_src'):
        g.step(chooser(g))
        if g.is_main() and chooser.i >= 3:
            break
    if not any(f['card'] == body for f in g.flags().values()):
        raise Reject('portal failed in setup')
    g.step(NOOP)
    for _ in range(60):
        if g.terminal:
            raise Reject('terminal before defender decision')
        if g.actor == g.seat and any(a[0] == 9 for a in g.rows()):
            break
        if g.actor == g.seat:
            g.step(g.default())
        else:
            g.step(opp_turn_action(g) if g.is_main(g.opp) else g.default())
    else:
        raise Reject('no defender decision')
    cases.append(g.save(FAMILY, 'declare_defender', 'survive', 'Stonehaven Defender already granted; opponent declared its first attack and total incoming damage equals our HP. Declare the defender to survive.',
                        horizon='opp_turn', checkpoint='response', leader_needed=False, gate_needed=False))
    # E5 stonehaven_unneeded (control)
    g = TF(runner, seed, seat, CAT, OPP_W)
    body = setup_stonehaven(g, 1)
    cases.append(g.save(FAMILY, 'stonehaven_unneeded', 'survive', 'Same structure but incoming damage is one less than our HP: survival needs nothing (control; portal/defender optional).',
                        horizon='opp_turn', portal=body, leader_needed=False, gate_needed=False))
    # E3 quicksand_lethal (Bobu deck 15 vs Foamback)
    g = TF(runner, seed, seat, BOBU3, OPP_W)

    def opp_foam():
        if FOAMBACK in g.hand(g.opp) and g.ikz(g.opp) >= 2 and not any(f['card'] == FOAMBACK for f in g.flags(g.opp).values()):
            g.play(FOAMBACK, seat=g.opp)
    g.wait_for(lambda: QUICKSAND in g.hand() and any(c in g.hand() for c in FODDER) and g.ikz() >= 1, opp=opp_foam)
    host = next(c for c in FODDER if c in g.hand())
    hslot = g.play(host)
    g.cycle(opp=opp_foam)
    g.chip_opp(ST[host]['attack'], reserve=(hslot,), fodder=[c for c in FODDER], opp=opp_foam)
    if not any(f['card'] == FOAMBACK and not f['tapped'] for f in g.flags(g.opp).values()):
        raise Reject('opponent has no untapped Foamback')
    if QUICKSAND not in g.hand():
        raise Reject('quicksand gone')
    g.spend_to(5, [c for c in SAFE if c in ST] + ['AZK01-050', 'AZK01-015'], zone='alley', free_garden=1)
    if g.snap()['gate_tapped'] is False and g.alley():
        pass
    if g.ready() != [(hslot, ST[host]['attack'])]:
        raise Reject('host not the only ready attacker')
    cases.append(g.save(FAMILY, 'quicksand_lethal', 'win', f'One ready attacker ({host}) with exact lethal, but the opponent has an untapped Foamback Crab defender and blocks. Quicksand (5 IKZ, all 5 open) clears every <=2-HP entity first. Bobu\'s 1-IKZ ability first makes Quicksand uncastable (harmful).',
                        horizon='turn', opp_block=True, host=host, leader_needed=False, gate_needed=False, harmful_leader=True))
    # E4 goro_trade (Goro deck 13 vs Zero deck 6)
    g = TF(runner, seed, seat, GORO, OPP_F)
    A_CAND = ['STT02-004', 'AZK01-050', 'AZK01-069']
    T_CAND = ['AZK01-058', 'AZK01-062', 'AZK01-056', 'STT04-005', 'STT02-004', 'STT04-004']

    def pair():
        mine, theirs = g.flags(), g.flags(g.opp)
        for ts, t in theirs.items():
            if t['cooldown'] or t['card'] not in T_CAND:
                continue
            for s, a in mine.items():
                if a['card'] in A_CAND and t['attack'] == a['hp'] and t['hp'] <= a['attack']:
                    others = [b for bs, b in mine.items() if bs != s and b['attack'] >= t['hp'] and b['hp'] > t['attack']]
                    if not others:
                        return ts, s
        return None
    found = None
    for _ in range(14):
        g.to_main()
        g.develop(A_CAND, budget_atk=4, max_entities=3)
        g.step(NOOP)
        g.to_main(g.opp)
        g.develop(T_CAND, budget_atk=6, seat=g.opp, max_entities=3)
        g.to_main(g.opp)
        found = pair()
        if found:
            ts = found[0]
            if [6, ts, GARDEN, 0] not in g.rows():
                found = None
            else:
                g.attack(ts)
        g.to_main(g.opp)
        g.step(NOOP)
        g.to_main()
        if found:
            break
    if not found:
        raise Reject('no Goro trade pair')
    ts, s = found
    t_code = g.flags(g.opp)[ts]['card']
    a_code = g.flags()[s]['card']
    if not g.flags(g.opp)[ts]['tapped']:
        raise Reject('target not tapped')
    # Tumbleweed excluded: portaled from the alley it offers a sacrifice-removal line outside the combat-trade grade
    g.spend_to(1, ['STT02-004', 'AZK01-069', 'AZK01-019', 'AZK01-012', 'AZK01-004', 'AZK01-050', 'AZK01-070'], zone='alley')
    if any(f['card'] == 'AZK01-105' for f in g.flags().values()):
        raise Reject('Tumbleweed removal alternative on board')
    p = pair_check = [b for bs, b in g.flags().items() if bs != s and not b['cooldown'] and not b['tapped'] and b['attack'] >= g.flags(g.opp)[ts]['hp'] and b['hp'] > g.flags(g.opp)[ts]['attack']]
    if pair_check:
        raise Reject('alternative favorable attacker')
    if any(g.code(c.card_def_id) == 'AZK01-105' for c in list(g.alley().values()) + list(g.garden().values())):
        raise Reject('Tumbleweed sacrifice-removal alternative')
    cases.append(g.save(FAMILY, 'goro_trade', 'trade', f'Tapped enemy {t_code} would trade evenly with our {a_code} (its attack equals our health). Goro (1 IKZ) gives +1 health first, so the attack removes it and our attacker survives.',
                        horizon='turn', host=a_code, target_code=t_code, leader_needed=True, gate_needed=False))
    return cases


def plans(case):
    p = case['params']
    name = case['name']
    if name in ('stonehaven_block', 'stonehaven_unneeded'):
        return {'portal_grant_block': [op_portal(p['portal']), op_maybe_confirm(), op_target_code(p['portal']), op_defend()],
                'portal_no_block': [op_portal(p['portal']), op_maybe_confirm(), op_target_code(p['portal'])],
                'block_without_portal': [op_defend()],
                'no_portal': []}
    if name == 'declare_defender':
        return {'block': [op_defend()], 'no_block': []}
    if name == 'quicksand_lethal':
        return {'quicksand_then_attack': [op_spell(QUICKSAND), op_attack_face(p['host'])],
                'attack_only': [op_attack_face(p['host'])],
                'leader_then_quicksand': [op_leader(), op_spell(QUICKSAND), op_attack_face(p['host'])]}
    if name == 'goro_trade':
        return {'goro_then_attack': [op_leader(), op_target_code(p['host']), op_attack_slot_code(p['host'], p['target_code'])],
                'attack_only': [op_attack_slot_code(p['host'], p['target_code'])]}
    raise KeyError(name)


def check_oracle(case, rows):
    ok = {k: v['success'] for k, v in rows.items()}
    name = case['name']
    if name == 'stonehaven_unneeded':
        assert all(ok.values()), (case['id'], ok)
        return
    positive = {'stonehaven_block': 'portal_grant_block', 'declare_defender': 'block', 'quicksand_lethal': 'quicksand_then_attack', 'goro_trade': 'goro_then_attack'}[name]
    assert ok[positive] and not any(v for k, v in ok.items() if k != positive), (case['id'], ok)
    if name == 'goro_trade':
        assert any(c['target_removed'] and not c['attacker_survived'] for c in rows['attack_only']['combats']), (case['id'], 'attack-only must be an even trade')
    if name == 'quicksand_lethal':
        assert rows['leader_then_quicksand']['leader_uses'] == 1
