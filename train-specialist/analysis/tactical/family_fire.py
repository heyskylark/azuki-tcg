"""FIRE family: Zero leader (STT04-001) + Rushfire (AZK01-122) / Ragefire (STT04-002).

Tested seat plays curated Zero decks (5 = Rushfire 'pasadena_ducky_drewwskiee', 18 = Ragefire
'inv_drewwskiee'); the opponent plays curated Bobu/Stonehaven deck 14 and is passive except where a
case scripts its attacks during setup.
"""
from tb_setup import TF, Reject, ST
from tactical import op_leader, op_target_code, op_portal, op_select_garden, op_attack_face, op_maybe_confirm, first

FAMILY = 'fire'
RUSH, RAGE, OPP = 5, 18, 14
FODDER = ['STT02-004', 'STT04-004', 'STT04-005', 'AZK01-056', 'STT04-003', 'AZK01-059', 'AZK01-062', 'STT04-007']
SPEND = FODDER + ['AZK01-058', 'STT04-009']  # Charge bodies (Scarlett, Tenmoku) excluded: they would add attackers
HOSTS = ['STT02-004', 'STT04-004']  # vanilla-ish 1/2 bodies that survive Zero's 1 damage
OPP_ATTACKERS = ['STT03-003', 'AZK01-050', 'STT03-012', 'AZK01-054', 'STT03-014', 'AZK01-045']
RITUALIST = 'STT04-009'


def card_slot(g, code, zone='garden'):
    s = g.slot(code, zone)
    if s is None:
        raise Reject(f'{code} missing from {zone}')
    return s


def no_ready_except(g, keep=()):
    extra = [s for s, _ in g.ready() if s not in keep]
    if extra:
        raise Reject('unexpected ready attackers at checkpoint')


def setup_host(g, target_offset):
    """Ready 1/2 host + opponent at host_attack + target_offset; IKZ 0, gate tapped, nothing else ready."""
    g.wait_for(lambda: any(c in g.hand() for c in HOSTS) and g.ikz() >= 1)
    host = next(c for c in HOSTS if c in g.hand())
    hslot = g.play(host)
    g.cycle()
    target = ST[host]['attack'] + target_offset
    g.chip_opp(target, reserve=(hslot,), fodder=[c for c in FODDER if c not in HOSTS])
    g.gate_sink(FODDER)
    g.spend_to(0, SPEND)
    if g.ready() != [(hslot, ST[host]['attack'])]:
        raise Reject('host not the only ready attacker')
    return host


def build(runner, seed, seat):
    cases = []
    # F1 zero_lethal / F2 zero_equivalent
    for name, offset, goal, notes in (('zero_lethal', 1, 'win', 'Zero pings the ready 1/2 host (+1 attack); the host then attacks for exact lethal. Without Zero the host is 1 short; IKZ 0, gate tapped.'),
                                      ('zero_equivalent', 0, 'win', 'Host already has lethal attack; Zero is optional (control for simple conversion / unnecessary activation).')):
        g = TF(runner, seed, seat, RUSH, OPP)
        host = setup_host(g, offset)
        cases.append(g.save(FAMILY, name, goal, notes, horizon='turn', host=host, leader_needed=offset > 0, gate_needed=False))
    # F3 zero_ritualist_burn
    g = TF(runner, seed, seat, RUSH, OPP)
    g.wait_for(lambda: RITUALIST in g.hand())
    g.chip_opp(1, fodder=FODDER)
    if g.ikz() < 3:
        raise Reject('cannot afford ritualist on checkpoint turn')
    g.play(RITUALIST)
    g.gate_sink(FODDER)
    g.spend_to(0, SPEND)
    no_ready_except(g)
    cases.append(g.save(FAMILY, 'zero_ritualist_burn', 'win', 'Opponent at 1, no attacker can attack. Zero targets Cinderwake Ritualist; its damage trigger (capped 2) then hits the opponent leader. Without Zero nothing deals damage.',
                        horizon='turn', leader_needed=True, gate_needed=False))
    # F4 rushfire_lethal
    g = TF(runner, seed, seat, RUSH, OPP)

    def rush_ready():
        hand = g.hand()
        portals = [c for c in hand if c in SPEND and ST[c]['gp'] >= 1]
        if not portals:
            return False
        best_gp = max(ST[c]['gp'] for c in portals)
        payloads = [c for c in hand if c in FODDER + ['AZK01-058'] and ST[c]['cost'] <= best_gp and ST[c]['attack'] >= 1]
        return bool(payloads) and len(hand) >= 3
    g.wait_for(rush_ready)
    hand = g.hand()
    portal = max((c for c in hand if c in SPEND and ST[c]['gp'] >= 1), key=lambda c: (ST[c]['gp'], -ST[c]['cost']))
    gp = ST[portal]['gp']
    rest = list(hand)
    rest.remove(portal)
    payload = max((c for c in rest if c in FODDER + ['AZK01-058'] and ST[c]['cost'] <= gp and ST[c]['attack'] >= 1), key=lambda c: ST[c]['attack'], default=None)
    if payload is None:
        raise Reject('no payload distinct from portal body')
    payload_atk = ST[payload]['attack']
    keep = {portal, payload}
    g.chip_opp(payload_atk, fodder=[c for c in FODDER if c not in keep])
    if g.ikz() < ST[portal]['cost'] or portal not in g.hand():
        raise Reject('cannot afford portal body')
    g.play(portal, 'alley')
    hand = g.hand()
    if payload not in hand:
        raise Reject('payload drawn away')
    spendable = [c for c in SPEND if c not in keep]
    g.spend_to(0, spendable, zone='garden')
    if payload not in g.hand():
        raise Reject('payload spent')
    no_ready_except(g)
    if g.snap()['gate_tapped']:
        raise Reject('gate tapped')
    cases.append(g.save(FAMILY, 'rushfire_lethal', 'win', f'No ready attacker, IKZ 0. Portal {portal} (gate power {gp}) with Rushfire, play {payload} from hand with Charge and attack for exact lethal.',
                        horizon='turn', portal=portal, payload=payload, leader_needed=False, gate_needed=True))
    # F5 zero_suicide (restraint)
    g = TF(runner, seed, seat, RUSH, OPP)
    g.chip_me(OPP_ATTACKERS, target=1, our=lambda: g.develop(FODDER, budget_atk=3, max_entities=2))
    g.to_main()
    if not g.garden() and not g.alley():
        raise Reject('no friendly entity: Zero would be illegal, restraint untestable')
    cases.append(g.save(FAMILY, 'zero_suicide', 'restraint', 'Own leader at 1 HP and no lethal available. Zero costs 1 damage to your own leader, so any activation loses on the spot (as do Detonation Pact / Fire Orb).',
                        horizon='turn', leader_needed=False, gate_needed=False))
    # F6 ragefire_combo (Ragefire deck)
    g = TF(runner, seed, seat, RAGE, OPP)
    rage_fodder = [c for c in FODDER if c != 'STT04-003']
    g.wait_for(lambda: any(c in g.hand() for c in HOSTS) and g.ikz() >= 1)
    host = next(c for c in HOSTS if c in g.hand())
    hslot = g.play(host)
    g.cycle()

    def portal_body():
        options = [c for c in g.hand() if c in SPEND and c != 'STT04-003' and ST[c]['gp'] >= 1 and ST[c]['cost'] <= 4]
        return max(options, key=lambda c: (ST[c]['gp'], -ST[c]['cost'])) if options else None
    g.wait_for(lambda: portal_body() is not None)
    body = portal_body()
    target = ST[host]['attack'] + 1 + ST[body]['gp']
    g.chip_opp(target, reserve=(hslot,), fodder=[c for c in rage_fodder if c not in HOSTS and c != body])
    body = portal_body()
    if body is None or ST[host]['attack'] + 1 + ST[body]['gp'] != target or g.ikz() < ST[body]['cost']:
        raise Reject('ragefire body unavailable on checkpoint turn')
    g.play(body, 'alley')
    g.spend_to(0, [c for c in SPEND if c != 'STT04-003'], zone='garden')
    if any(g.code(c.card_def_id) == 'STT04-003' for c in g.garden().values()):
        raise Reject('Seer in garden gives a Zero-free Ragefire target')
    if g.ready() != [(hslot, ST[host]['attack'])] or g.snap()['gate_tapped']:
        raise Reject('ragefire checkpoint invalid')
    cases.append(g.save(FAMILY, 'ragefire_combo', 'win', f'Zero pings {host} (+1, now damaged this turn), portal {body} (gate power {ST[body]["gp"]}) so Ragefire adds +{ST[body]["gp"]} to the damaged host, attack for exact lethal. Needs both leader and gate.',
                        horizon='turn', host=host, portal=body, leader_needed=True, gate_needed=True))
    return cases


def plans(case):
    p = case['params']
    name = case['name']
    if name in ('zero_lethal', 'zero_equivalent'):
        return {'with_ability': [op_leader(), op_target_code(p['host']), op_attack_face(p['host'])],
                'without_ability': [op_attack_face(p['host'])]}
    if name == 'zero_ritualist_burn':
        return {'with_ability': [op_leader(), op_target_code(RITUALIST), first(lambda g, a: a[0] == 16), first(lambda g, a: a[0] == 14 and a[1] == 11)],
                'ability_hits_own_leader': [op_leader(), op_target_code(RITUALIST), first(lambda g, a: a[0] == 16), first(lambda g, a: a[0] == 14 and a[1] == 10)],
                'without_ability': []}
    if name == 'rushfire_lethal':
        return {'with_gate': [op_portal(p['portal']), op_select_garden(p['payload']), op_attack_face(p['payload'])],
                'without_gate': [op_attack_face()]}
    if name == 'zero_suicide':
        return {'with_ability': [op_leader(), first(lambda g, a: a[0] == 14)], 'without_ability': []}
    if name == 'ragefire_combo':
        return {'leader_and_gate': [op_leader(), op_target_code(p['host']), op_portal(p['portal']), op_maybe_confirm(), op_target_code(p['host']), op_attack_face(p['host'])],
                'gate_only': [op_portal(p['portal']), op_attack_face(p['host'])],
                'leader_only': [op_leader(), op_target_code(p['host']), op_attack_face(p['host'])]}
    raise KeyError(name)


def check_oracle(case, rows):
    name = case['name']
    ok = {k: v['success'] for k, v in rows.items()}
    if name == 'zero_equivalent':
        assert all(ok.values()), (case['id'], ok)
    elif name == 'zero_suicide':
        assert ok['without_ability'] and not ok['with_ability'] and rows['with_ability']['lost'], (case['id'], ok)
    else:
        positive = {'zero_lethal': 'with_ability', 'zero_ritualist_burn': 'with_ability', 'rushfire_lethal': 'with_gate', 'ragefire_combo': 'leader_and_gate'}[name]
        assert ok[positive] and not any(v for k, v in ok.items() if k != positive), (case['id'], ok)
