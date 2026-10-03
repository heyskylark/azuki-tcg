"""Rollouts, oracle verification and policy evaluation for the constructed families.

Same grading contract as leader_history_benchmark_v1: replay the complete legal
prefix through each checkpoint's own recurrent state, then let the policy act
(argmax and multinomial samples) until the case horizon; grade the actual
engine outcome (win / survival / favorable trade / restraint), not activation
quotas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tb_runtime import OUT, POLICIES, CURATED_POOL, EpisodeRunner, sha256, tcg_sampler, set_eval_sampling
from tb_fixture import GARDEN, NOOP, replay
from tb_setup import TF

SAMPLE_SEEDS = list(range(91001, 91017))  # 16 samples; the first 4 equal the original benchmark's seeds


def case_digest(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


class Brain:
    """Per-checkpoint recurrent history; memoized by full own-observation history (as in the original)."""

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
        groups = {'leader': (rows[:, 0] == 11) & (rows[:, 1] == 5), 'portal': rows[:, 0] == 10, 'pass': rows[:, 0] == 0,
                  'attack': rows[:, 0] == 6, 'spell': rows[:, 0] == 8, 'defend': rows[:, 0] == 9}
        order = torch.argsort(probs, descending=True).tolist()[:12]
        return dict(own_history_steps=self.steps, history_sha256=self.head.hex(), hidden_norms={k: float(v.norm()) for k, v in self.own.items()},
                    group_probability={k: float(probs[torch.as_tensor(m)].sum()) for k, m in groups.items()},
                    actions=[dict(action=rows[i].tolist(), probability=float(probs[i])) for i in order])


class ModelChooser:
    def __init__(self, brain, mode, seed):
        self.brain, self.mode = brain, mode
        self.rng = torch.Generator().manual_seed(seed)

    def __call__(self, game):
        rows, probs = self.brain.forward(game, commit=False)
        index = int(torch.argmax(probs)) if self.mode == 'argmax' else int(torch.multinomial(probs, 1, generator=self.rng))
        return [int(v) for v in rows[index]]


class PlanChooser:
    """Oracle: ordered ops (callables game -> action|None).  An op that is not yet applicable yields the
    engine default (NOOP: end turn at main / decline at prompts) without advancing."""

    def __init__(self, ops):
        self.ops = list(ops)
        self.i = 0

    def __call__(self, game):
        while self.i < len(self.ops):
            act = self.ops[self.i](game)
            if act is None:
                return game.default()
            self.i += 1
            if act == 'skip':
                continue
            return [int(v) for v in act]
        return game.default()


# ---- op constructors for oracle plans -------------------------------------------
def first(pred):
    def op(game):
        return next((a for a in game.rows() if pred(game, a)), None)
    return op


def op_leader():
    return first(lambda g, a: a[0] == 11 and a[1] == GARDEN)


def op_target(index):
    return first(lambda g, a: a[0] == 14 and a[1] == index)


def op_target_code(code, own=True, leader=False):
    def op(game):
        for a in game.rows():
            if a[0] != 14:
                continue
            if leader and a[1] == GARDEN:
                return a
            if not leader:
                garden = game.garden(game.seat if own else game.opp)
                c = garden.get(a[1])
                if c is not None and game.code(c.card_def_id) == code:
                    return a
        return None
    return op


def op_portal(code=None):
    def op(game):
        alley, garden = game.alley(), game.garden()
        rows = [a for a in game.rows() if a[0] == 10 and (code is None or (a[1] in alley and game.code(alley[a[1]].card_def_id) == code))]
        return next((a for a in rows if a[2] not in garden), None)
    return op


def op_select_garden(code):
    def op(game):
        sel = game.snap().get('selection') or []
        return next((a for a in game.rows() if a[0] == 23 and a[1] < len(sel) and sel[a[1]] == code), None)
    return op


def op_attack_face(code=None):
    def op(game):
        if not game.is_main():
            return None
        garden = game.garden()
        return next((a for a in game.rows() if a[0] == 6 and a[2] == GARDEN and a[1] != GARDEN and (code is None or game.code(garden[a[1]].card_def_id) == code)), None)
    return op


def op_attack_slot_code(attacker, target_code):
    def op(game):
        if not game.is_main():
            return None
        mine, theirs = game.garden(), game.garden(game.opp)
        for a in game.rows():
            if a[0] == 6 and a[1] in mine and a[2] in theirs and game.code(mine[a[1]].card_def_id) == attacker and game.code(theirs[a[2]].card_def_id) == target_code:
                return a
        return None
    return op


def op_spell(code):
    def op(game):
        hand = game.hand()
        return next((a for a in game.rows() if a[0] == 8 and a[1] < len(hand) and hand[a[1]] == code), None)
    return op


def op_equip(code, target=GARDEN):
    def op(game):
        hand = game.hand()
        return next((a for a in game.rows() if a[0] == 7 and a[1] < len(hand) and hand[a[1]] == code and a[2] == target), None)
    return op


def op_defend():
    return first(lambda g, a: a[0] == 9)


def op_maybe_confirm():
    def op(game):
        return next((a for a in game.rows() if a[0] == 16), 'skip')
    return op


def op_any(type_):
    return first(lambda g, a: a[0] == type_)


# ---- rollout ---------------------------------------------------------------------
def opp_turn_action(game):
    """Scripted aggressive opponent main: attack our leader with its highest-attack ready entity, else end turn."""
    flags = game.flags(game.opp)
    rows = [a for a in game.rows() if a[0] == 6 and a[2] == GARDEN and a[1] != GARDEN and flags.get(a[1], {}).get('attack', 0) > 0]
    if not rows:
        return NOOP
    return max(rows, key=lambda a: (flags[a[1]]['attack'], -a[1]))


def opp_response(game, block):
    rows = game.rows()
    if block:
        defend = [a for a in rows if a[0] == 9]
        if defend:
            return defend[0]
    return game.default()


def play_out(runner, case, chooser, observer=None):
    game = replay(runner, case, observer=observer, cls=TF)
    n = len(game.actions)
    root = observer.__self__.inspect(game) if observer is not None else None
    params = case['params']
    horizon = params['horizon']
    block = params.get('opp_block', False)
    seen_opp = params.get('checkpoint') == 'response'
    combats, pending = [], None
    for _ in range(400):
        if game.terminal:
            break
        if game.actor == game.seat:
            if seen_opp and game.is_main():
                break
            act = chooser(game)
            if act[0] == 6 and act[2] < GARDEN:
                src = game.raw().my_observation_data.leader if act[1] == GARDEN else game.garden().get(act[1])
                pending = dict(source_slot=act[1], source_id=int(src.card_def_id), target_slot=act[2],
                               target_id=int(game.garden(game.opp)[act[2]].card_def_id))
        else:
            if game.is_main(game.opp):
                seen_opp = True
                if horizon == 'turn':
                    break
                act = opp_turn_action(game)
            else:
                act = opp_response(game, block)
        game.step(act)
        if pending is not None and not game.terminal and not game.raw().combat_context.combat_active:
            src = game.raw().my_observation_data.leader if pending['source_slot'] == GARDEN else game.garden().get(pending['source_slot'])
            tgt = game.garden(game.opp).get(pending['target_slot'])
            pending['attacker_survived'] = src is not None and int(src.card_def_id) == pending['source_id'] and int(src.cur_stats.cur_hp) > 0
            pending['target_removed'] = tgt is None or int(tgt.card_def_id) != pending['target_id']
            combats.append(pending)
            pending = None
    else:
        raise AssertionError(('rollout exceeded action budget', case['id']))
    return game, n, root, combats


def outcome(game, case, n, combats):
    cont = game.actions[n:]
    mine = [a['action'] for a in cont if a['seat'] == game.seat]
    won = game.terminal and game.winner == game.seat
    lost = game.terminal and game.winner == 1 - game.seat
    leader_uses = sum(a[0] == 11 and a[1] == GARDEN for a in mine)
    goal = case['goal']
    target = case['params'].get('target_code')
    favorable = any(c['target_removed'] and c['attacker_survived'] and (target is None or game.code(c['target_id']) == target) for c in combats)
    if goal == 'win':
        success = won
    elif goal == 'survive':
        success = not lost
    elif goal == 'trade':
        success = won or favorable
    elif goal == 'restraint':
        success = not lost and leader_uses == 0
    else:
        raise ValueError(goal)
    return dict(success=bool(success), won=bool(won), lost=bool(lost), terminal=game.terminal, winner=game.winner, favorable_trade=favorable,
                combats=combats, leader_uses=leader_uses, portals=sum(a[0] == 10 for a in mine), defends=sum(a[0] == 9 for a in mine),
                spells=sum(a[0] == 8 for a in mine), first_action=mine[0] if mine else None, final=game.snap(), actions=cont)


# ---- family registry --------------------------------------------------------------
def load_family(element):
    import importlib
    return importlib.import_module(f'family_{element}')


def verify_oracles(runner, element, cases):
    fam = load_family(element)
    results = []
    for case in cases:
        rows = {}
        for branch, ops in fam.plans(case).items():
            game, n, _, combats = play_out(runner, case, PlanChooser(ops))
            rows[branch] = dict(case=case['id'], case_sha256=case_digest(case), branch=branch, **outcome(game, case, n, combats))
        fam.check_oracle(case, rows)
        results.extend(rows.values())
        print(json.dumps({'oracle_verified': case['id'], 'branches': {k: v['success'] for k, v in rows.items()}}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('element')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--model', action='append', default=[])
    parser.add_argument('--samples', type=int, default=16)
    opts = parser.parse_args()
    fdir = OUT / opts.element
    cases = json.loads((fdir / 'cases.json').read_text())
    if opts.oracle:
        runner = EpisodeRunner(POLICIES['u8223'], CURATED_POOL, 'cpu')
        try:
            results = verify_oracles(runner, opts.element, cases)
        finally:
            runner.close()
        (fdir / 'oracle_results.json').write_text(json.dumps(results, indent=2))
        return
    verified = {r['case']: r['case_sha256'] for r in json.loads((fdir / 'oracle_results.json').read_text())}
    assert verified == {c['id']: case_digest(c) for c in cases}, 'oracle fixtures changed'
    inputs = {name: hashlib.sha256((fdir / name).read_bytes()).hexdigest() for name in ('cases.json', 'oracle_results.json')}
    seeds = SAMPLE_SEEDS[:opts.samples]
    for key in opts.model:
        checkpoint = POLICIES[key]
        runner = EpisodeRunner(checkpoint, CURATED_POOL, 'cpu')
        set_eval_sampling()
        try:
            brain = Brain(runner.policy)
            results = []
            for case in cases:
                for mode, seed in [('argmax', 0)] + [('sample', s) for s in seeds]:
                    brain.reset()
                    game, n, root, combats = play_out(runner, case, ModelChooser(brain, mode, seed), observer=brain.observe)
                    results.append(dict(case=case['id'], mode=mode, sample_seed=seed, root=root, **outcome(game, case, n, combats)))
                print(json.dumps({'model': key, 'case': case['id'], 'rollouts': len(results), 'argmax_success': results[-len(seeds) - 1]['success'],
                                  'sample_success': sum(r['success'] for r in results[-len(seeds):])}), flush=True)
            (fdir / f'{key}_results.json').write_text(json.dumps(dict(model=key, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                                                                          inputs=inputs, samples=seeds, computed_forwards=brain.computed,
                                                                          cache_hits=brain.hits, results=results), indent=1))
        finally:
            runner.close()


if __name__ == '__main__':
    main()
