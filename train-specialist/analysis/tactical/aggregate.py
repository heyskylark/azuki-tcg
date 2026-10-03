"""Aggregate tactical-fixture results: per family x case type x policy x mode success, leader/gate use,
harmful activations; paired specialist-minus-baseline deltas with cluster (fixture-instance) bootstrap CIs and
exact Clopper-Pearson intervals.  Writes summary.json and tables.md."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random

from scipy.stats import beta

OUT = Path(__file__).resolve().parent
FAMILIES = {  # family dir -> (own specialist, off-element controls)
    'lightning': ('lightning', ['fire', 'water', 'earth']),
    'lightning_c': ('lightning', ['fire']),
    'fire': ('fire', ['water']),
    'water': ('water', ['earth']),
    'earth': ('earth', ['lightning']),
}
BASE = ['u8223', 'u9305']
B = 4000


def cp(k, n, a=0.05):
    if n == 0:
        return (math.nan, math.nan)
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return (round(float(lo), 3), round(float(hi), 3))


LIGHTNING_CATEGORY = {'ability_lethal': 'ability_dependent_lethal', 'equipment_lethal': 'setup_dependent_lethal', 'equipment_trade': 'favorable_trade',
                      'ability_trade': 'favorable_trade', 'equipment_equivalent_win': 'simpler_wins', 'ability_equivalent_win': 'simpler_wins',
                      'ability_tapped': 'pointless_piko_avoided', 'ability_zero': 'pointless_piko_avoided'}


def case_type(case):
    if 'family' in case:
        return case['name']
    return LIGHTNING_CATEGORY[case['name']]


def instance(case):
    return (case['seat'], case['seed'])


def harmful(case, row):
    """Activations that provably cost the outcome (from the case's oracle design)."""
    name = case_type(case)
    acts = [a['action'] for a in row['actions'] if a['seat'] == case['seat']]
    if name in ('zero_suicide',):
        return row['lost']  # opponent is passive during our turn: any loss is self-inflicted (Zero / Detonation Pact / Fire Orb)
    if name == 'quicksand_lethal':
        return row['leader_uses'] > 0 and not row['success']
    if name == 'zero_ritualist_burn':
        return any(a[0] == 14 and a[1] == 10 for a in acts)  # Ritualist trigger aimed at own leader
    return False


def pointless(case, row):
    name = case_type(case)
    return name == 'pointless_piko_avoided' and row['leader_uses'] > 0


def load(fam, model):
    path = OUT / fam / f'{model}_results.json'
    if not path.exists():
        return None
    rows = json.loads(path.read_text())['results']
    return [r for r in rows if not r.get('forced_host')]


def rate_by_instance(rows, cases, mode):
    acc = defaultdict(list)
    for r in rows:
        if r['mode'] != mode:
            continue
        c = cases[r['case']]
        acc[(case_type(c), instance(c))].append(r['success'])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def boot_delta(a, b, keys, seed=0):
    rng = random.Random(seed)
    keys = list(keys)
    if not keys:
        return (math.nan, math.nan, math.nan)
    point = sum(a[k] - b[k] for k in keys) / len(keys)
    draws = []
    for _ in range(B):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(sum(a[k] - b[k] for k in sample) / len(sample))
    draws.sort()
    return (round(point, 3), round(draws[int(0.025 * B)], 3), round(draws[int(0.975 * B) - 1], 3))


def main():
    summary = {'families': {}}
    lines = []
    for fam, (own, controls) in FAMILIES.items():
        cpath = OUT / fam / 'cases.json'
        if not cpath.exists():
            continue
        cases = {c['id']: c for c in json.loads(cpath.read_text())}
        models = [m for m in BASE + [own] + controls if load(fam, m) is not None]
        data = {m: load(fam, m) for m in models}
        types = sorted({case_type(c) for c in cases.values()})
        fam_out = {'models': models, 'case_types': {}, 'paired': {}}
        lines.append(f'\n### {fam} (own specialist: {own}; controls: {", ".join(c for c in controls if c in models)})\n')
        lines.append('| case type | n inst | policy | greedy succ | sampled succ (95% CP) | leader uses g/s | gate portals g/s | harmful g/s |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for t in types:
            insts = sorted({instance(c) for c in cases.values() if case_type(c) == t})
            fam_out['case_types'][t] = {}
            for m in models:
                rows = [r for r in data[m] if case_type(cases[r['case']]) == t]
                g = [r for r in rows if r['mode'] == 'argmax']
                s = [r for r in rows if r['mode'] == 'sample']
                kg, ks = sum(r['success'] for r in g), sum(r['success'] for r in s)
                entry = dict(greedy=[kg, len(g)], sampled=[ks, len(s)], sampled_ci=cp(ks, len(s)), greedy_ci=cp(kg, len(g)),
                             leader_uses_greedy=sum(r['leader_uses'] for r in g), leader_uses_sampled=sum(r['leader_uses'] for r in s),
                             leader_rollouts_sampled=sum(r['leader_uses'] > 0 for r in s),
                             portals_greedy=sum(r.get('portals', 0) for r in g), portals_sampled=sum(r.get('portals', 0) for r in s),
                             harmful_greedy=sum(harmful(cases[r['case']], r) for r in g), harmful_sampled=sum(harmful(cases[r['case']], r) for r in s),
                             pointless_greedy=sum(pointless(cases[r['case']], r) for r in g), pointless_sampled=sum(pointless(cases[r['case']], r) for r in s),
                             success_with_leader_sampled=sum(r['success'] and r['leader_uses'] > 0 for r in s),
                             failed_with_leader_greedy=sum((not r['success']) and r['leader_uses'] > 0 for r in g),
                             failed_with_leader_sampled=sum((not r['success']) and r['leader_uses'] > 0 for r in s),
                             defends_sampled=sum(r.get('defends', 0) for r in s),
                             root_leader_prob=round(sum(r['root'].get('group_probability', {}).get('leader', r['root'].get('leader_probability', 0)) for r in g) / max(1, len(g)), 4))
                fam_out['case_types'][t][m] = entry
                hp = entry['harmful_greedy'] + entry['pointless_greedy'], entry['harmful_sampled'] + entry['pointless_sampled']
                lines.append(f"| {t} | {len(insts)} | {m} | {kg}/{len(g)} | {ks}/{len(s)} ({entry['sampled_ci'][0]:.2f}-{entry['sampled_ci'][1]:.2f}) | {entry['leader_uses_greedy']}/{entry['leader_uses_sampled']} | {entry['portals_greedy']}/{entry['portals_sampled']} | {hp[0]}/{hp[1]} |")
        # paired deltas (specialist and controls vs each baseline), per case type and pooled
        lines.append('\n| comparison | case type | greedy delta [95% boot CI] | sampled delta [95% boot CI] |')
        lines.append('|---|---|---|---|')
        for m in [own] + controls:
            if m not in models:
                continue
            for base in BASE:
                if base not in models:
                    continue
                for mode in ('argmax', 'sample'):
                    pass
                ra = {mode: rate_by_instance(data[m], cases, mode) for mode in ('argmax', 'sample')}
                rb = {mode: rate_by_instance(data[base], cases, mode) for mode in ('argmax', 'sample')}
                for t in types + ['ALL']:
                    res = {}
                    for mode in ('argmax', 'sample'):
                        keys = [k for k in ra[mode] if (t == 'ALL' or k[0] == t) and k in rb[mode]]
                        res[mode] = boot_delta(ra[mode], rb[mode], keys, seed=hash((fam, m, base, t, mode)) & 0xffff)
                    fam_out['paired'][f'{m}-{base}:{t}'] = res
                    if base == 'u8223' or t == 'ALL':
                        lines.append(f"| {m} - {base} | {t} | {res['argmax'][0]:+.3f} [{res['argmax'][1]:+.2f}, {res['argmax'][2]:+.2f}] | {res['sample'][0]:+.3f} [{res['sample'][1]:+.2f}, {res['sample'][2]:+.2f}] |")
        summary['families'][fam] = fam_out
    summary['notes'] = ['Bootstrap resamples fixture instances (seat x world seed) within case type; sampled rates are per-instance means over sample seeds.',
                        'CP intervals treat rollouts as independent; sample repetitions share fixtures, so they understate uncertainty.',
                        'harmful = activation that the oracle shows costs the outcome (zero_suicide any Zero use; quicksand_lethal Bobu use that leaves Quicksand uncastable; ritualist trigger aimed at own leader). pointless = original Piko zero/tapped definition.']
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    (OUT / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
