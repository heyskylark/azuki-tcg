"""Replay complete legal prefixes; grade actual outcomes rather than activation quotas."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from history_benchmark import OUT, RUNTIME, Fixture, EpisodeRunner, REI, FRIDA, DAGGER, specs, _load_model_weights, tcg_sampler, V3_POOL


def case_digest(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


class ObservedFixture(Fixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combats = []
        self.pending_attack = None

    def step(self, action):
        if not self.runner.base_env._building and self.actor == self.seat and int(action[0]) == 6 and int(action[2]) < 5:
            raw = self.raw()
            source_slot, target_slot = int(action[1]), int(action[2])
            source = raw.my_observation_data.leader if source_slot == 5 else raw.my_observation_data.garden[source_slot]
            self.pending_attack = dict(source_slot=source_slot, source_id=int(source.card_def_id), target_slot=target_slot)
        super().step(action)
        if self.pending_attack is not None and not self.raw().combat_context.combat_active:
            event = self.pending_attack
            raw = self.raw()
            source = raw.my_observation_data.leader if event['source_slot'] == 5 else raw.my_observation_data.garden[event['source_slot']]
            event['attacker_survived'] = int(source.card_def_id) == event['source_id'] and int(source.cur_stats.cur_hp) > 0
            event['target_removed'] = int(raw.opponent_observation_data.garden[event['target_slot']].card_def_id) < 0
            self.combats.append(event)
            self.pending_attack = None


def replay(runner, case, brain=None):
    if brain is not None:
        brain.reset()
    game = ObservedFixture(runner, case['seed'], case['seat'], case['leader'], observer=None if brain is None else brain.observe)
    for row in case['prefix']:
        assert game.actor == row['seat'] and bool(runner.base_env._building) == row['draft']
        game.step(row['action'])
    assert game.snap() == case['checkpoint'], ('checkpoint replay differs', case['id'])
    return game


class Brain:
    def __init__(self, policy):
        self.policy = policy
        self.cache = {}
        self.computed = 0
        self.hits = 0
        self.reset()

    def reset(self):
        self.head = bytes(32)
        self.own = {name: torch.zeros(1, self.policy.hidden_size) for name in ('lstm_h', 'lstm_c')}
        self.steps = 0

    def forward(self, game, commit):
        row = game.obs[game.seat:game.seat + 1]
        mask = np.asarray(game.masks)[game.seat:game.seat + 1]
        key = hashlib.sha256(self.head + row.tobytes() + mask.tobytes()).digest()
        if key in self.cache:
            after, rows, probs = self.cache[key]
            self.hits += 1
        else:
            state = {name: value.expand(2, -1).contiguous() for name, value in self.own.items()}
            state['mask'] = torch.as_tensor(mask).repeat(2)
            with torch.no_grad():
                logits, _ = self.policy.forward_eval(torch.as_tensor(row).expand(2, *row.shape[1:]), state)
                distribution, _, _ = tcg_sampler.legal_action_row_distribution(logits)
            n = int(logits.legal_action_count[0])
            rows = logits.legal_actions[0, :n].cpu().numpy().copy()
            probs = distribution[0, :n].cpu().clone()
            after = {name: state[name][:1].clone() for name in ('lstm_h', 'lstm_c')}
            self.cache[key] = after, rows, probs
            self.computed += 1
        if commit:
            self.head = key
            self.own = after
            self.steps += 1
        return rows, probs

    def observe(self, game):
        self.forward(game, commit=True)

    def inspect(self, game):
        rows, probs = self.forward(game, commit=False)
        leader = (rows[:, 0] == 11) & (rows[:, 1] == 5)
        entity_equip = (rows[:, 0] == 7) & (rows[:, 2] < 5)
        leader_equip = (rows[:, 0] == 7) & (rows[:, 2] == 5)
        order = torch.argsort(probs, descending=True).tolist()
        return dict(own_history_steps=self.steps, history_sha256=self.head.hex(), hidden_norms={k: float(v.norm()) for k, v in self.own.items()}, leader_probability=float(probs[leader].sum()), entity_equip_probability=float(probs[entity_equip].sum()), leader_equip_probability=float(probs[leader_equip].sum()), actions=[dict(action=rows[i].tolist(), probability=float(probs[i])) for i in order])


def finish_turn(game):
    for _ in range(120):
        if game.terminal or game.is_main(1 - game.seat):
            return
        game.step(game.default())
    raise AssertionError('turn completion exceeded action budget')


def outcome(game, case, prefix_length):
    continuation = game.actions[prefix_length:]
    uses = sum(a['seat'] == game.seat and a['action'][0] == 11 and a['action'][1] == 5 for a in continuation)
    final = game.snap()
    payload_alive = game.slot(REI) is not None
    removed = False
    if case['goal'] == 'trade':
        card = game.raw().opponent_observation_data.garden[case['target']]
        removed = int(card.card_def_id) < 0
    won = game.terminal and game.winner == game.seat
    favorable = any(e['target_slot'] == case['target'] and e['target_removed'] and e['attacker_survived'] for e in game.combats)
    success = won if case['goal'] == 'win' else won or favorable if case['goal'] == 'trade' else uses == 0
    return dict(success=success, won=won, terminal=game.terminal, winner=game.winner, target_removed=removed, favorable_trade=favorable, combats=game.combats, payload_survives=payload_alive, leader_uses=uses, ikz_remaining=game.resources(), final=final, actions=continuation)


def oracle(runner, case, branch):
    game = replay(runner, case)
    n = len(game.actions)
    payload = game.slot(REI)
    if branch == 'losing_attacker_exchange':
        payload = game.slot(FRIDA)
    assert payload is not None
    if case['name'].startswith('equipment_') and branch != 'no_equipment':
        game.equip(5 if branch == 'leader_equipment' else payload)
    use_ability = branch in ('entity_equipment', 'with_ability', 'losing_attacker_exchange')
    if use_ability:
        row = next((a for a in game.rows() if a[0] == 11 and a[1] == 5), None)
        if row is not None:
            game.activate(payload)
    if case['goal'] == 'trade':
        for _ in range(6):
            if game.terminal or int(game.raw().opponent_observation_data.garden[case['target']].card_def_id) < 0:
                break
            raw = game.raw()
            def positive_attack(a):
                if a[0] != 6 or a[2] != case['target']:
                    return False
                source = raw.my_observation_data.leader if a[1] == 5 else raw.my_observation_data.garden[a[1]]
                return source.cur_stats.cur_atk > 0
            row = next((a for a in game.rows() if positive_attack(a)), None)
            if row is None:
                break
            game.attack(row[1], case['target'])
    else:
        for _ in range(6):
            if game.terminal:
                break
            row = next((a for a in game.rows() if a[0] == 6 and a[2] == 5), None)
            if row is None:
                break
            game.attack(row[1])
    finish_turn(game)
    return dict(case=case['id'], case_sha256=case_digest(case), branch=branch, **outcome(game, case, n))


def verify_oracles(runner, cases):
    results = []
    for case in cases:
        branches = ('entity_equipment', 'leader_equipment', 'no_equipment') if case['name'].startswith('equipment_') else ('with_ability', 'without_ability')
        if case['leader'] == 'raizan' and case['name'] == 'equipment_trade':
            branches += ('losing_attacker_exchange',)
        rows = {branch: oracle(runner, case, branch) for branch in branches}
        if 'losing_attacker_exchange' in rows:
            exchange = rows['losing_attacker_exchange']
            assert exchange['target_removed'] and exchange['payload_survives'] and not exchange['favorable_trade'], (case['id'], exchange)
        if 'equivalent_win' in case['name']:
            assert all(row['won'] for row in rows.values()), (case['id'], rows)
        elif case['goal'] != 'restraint':
            positive = 'entity_equipment' if case['name'].startswith('equipment_') else 'with_ability'
            assert rows[positive]['success'], (case['id'], rows)
            assert all(not row['success'] for branch, row in rows.items() if branch != positive), (case['id'], rows)
        else:
            assert rows['with_ability']['leader_uses'] == 1 and rows['without_ability']['leader_uses'] == 0
            assert rows['with_ability']['final']['opp_hp'] == rows['without_ability']['final']['opp_hp'], case['id']
            assert rows['with_ability']['final']['my_garden'] == rows['without_ability']['final']['my_garden'], case['id']
        results.extend(rows.values())
        print(json.dumps({'oracle_verified': case['id'], 'branches': {k: v['success'] for k, v in rows.items()}}), flush=True)
    (OUT / 'oracle_results.json').write_text(json.dumps(results, indent=2))
    return results


def rollout(runner, case, brain, mode, sample_seed, forced_host=None):
    game = replay(runner, case, brain)
    n = len(game.actions)
    root = brain.inspect(game)
    if forced_host is not None:
        game.equip(5 if forced_host == 'leader' else game.slot(REI))
    rng = torch.Generator().manual_seed(sample_seed)
    for _ in range(120):
        if game.terminal or game.is_main(1 - game.seat):
            break
        if game.actor == game.seat:
            rows, probs = brain.forward(game, commit=False)
            index = int(torch.argmax(probs)) if mode == 'argmax' else int(torch.multinomial(probs, 1, generator=rng))
            game.step(rows[index])
        else:
            game.step(game.default())
    else:
        raise AssertionError(('model turn exceeded action budget', case['id']))
    return dict(case=case['id'], mode=mode, sample_seed=sample_seed, forced_host=forced_host, root=root, **outcome(game, case, n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--policies', action='store_true')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--model')
    opts = parser.parse_args()
    cases = json.loads((OUT / 'cases.json').read_text())
    all_specs = specs()
    (OUT / 'evaluation_contract.json').write_text(json.dumps(dict(runtime=str(RUNTIME), fixtures_sha256=hashlib.sha256((OUT / 'cases.json').read_bytes()).hexdigest(), policies=all_specs, samples=opts.samples, sample_seeds=list(range(91001, 91001 + opts.samples)), history='Every active candidate draft and battle observation is replayed through its own checkpoint. Shared-prefix memoization is scoped to one model and keyed by the entire observation/mask history.', grade='Actual terminal win, target removal with the actual attacker surviving combat, or avoidance of provably useless activation. Equivalent wins all succeed.', limitations=['Constructed conditional tactical fixtures, not naturally occurring frequency or broad matchup strength.', 'Opponent passes responses; no claim of robustness to adversarial responses.', 'One deck family, two feasible world seeds per seat; no population confidence claims.', 'Favorable trade is an available tactic, not proof that face damage or development is globally inferior.', 'Frozen Raizan outcome counter ignored; engine winners and board transitions used.']), indent=2))
    for key, spec in all_specs.items():
        if opts.model and opts.model != key:
            continue
        if not opts.policies and key != 'u8223':
            continue
        checkpoint = Path(spec['checkpoint'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == spec['checkpoint_sha256']
        runner = EpisodeRunner(checkpoint, V3_POOL, 'cpu')
        try:
            if opts.oracle:
                verify_oracles(runner, cases)
                opts.oracle = False
            if opts.policies:
                assert (OUT / 'oracle_results.json').exists(), 'oracle verification required first'
                inputs = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in ('cases.json', 'oracle_results.json', 'history_benchmark.py', 'evaluate_history.py')}
                verified_cases = {r['case']: r['case_sha256'] for r in json.loads((OUT / 'oracle_results.json').read_text())}
                assert verified_cases == {c['id']: case_digest(c) for c in cases}, 'oracle fixtures changed'
                brain = Brain(runner.policy)
                results = []
                for case in cases:
                    modes = [('argmax', 0)] + [('sample', seed) for seed in range(91001, 91001 + opts.samples)]
                    for mode, seed in modes:
                        results.append(rollout(runner, case, brain, mode, seed))
                    if case['name'].startswith('equipment_'):
                        for host in ('entity', 'leader'):
                            results.append(rollout(runner, case, brain, 'argmax', 0, forced_host=host))
                    (OUT / (key + '_results.json')).write_text(json.dumps(dict(model=key, checkpoint=spec, inputs=inputs, computed_forwards=brain.computed, cache_hits=brain.hits, results=results), indent=2))
                    print(json.dumps({'model': key, 'case': case['id'], 'completed_rollouts': len(results), 'computed_forwards': brain.computed}), flush=True)
        finally:
            runner.vecenv.close()

if __name__ == '__main__':
    main()
