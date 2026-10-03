"""Generate a family's fixtures: first two world seeds per seat (scanning upward from 101) where every
case type constructs legally AND its oracle branches verify.  No policy output is consulted."""
import argparse
import json
import traceback

from tb_runtime import OUT, POLICIES, CURATED_POOL, EpisodeRunner
from tb_setup import Reject
from tactical import load_family, verify_oracles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('element')
    parser.add_argument('--seed-start', type=int, default=101)
    parser.add_argument('--seed-stop', type=int, default=400)
    parser.add_argument('--per-seat', type=int, default=2)
    parser.add_argument('--seats', default='0,1')
    opts = parser.parse_args()
    fam = load_family(opts.element)
    fdir = OUT / opts.element
    fdir.mkdir(exist_ok=True)
    runner = EpisodeRunner(POLICIES['u8223'], CURATED_POOL, 'cpu')
    cases, oracle, accepted, rejected = [], [], [], []
    try:
        for seat in (int(s) for s in opts.seats.split(',')):
            found = 0
            for seed in range(opts.seed_start, opts.seed_stop):
                try:
                    batch = fam.build(runner, seed, seat)
                    rows = verify_oracles(runner, opts.element, batch)
                except (Reject, AssertionError, StopIteration, ValueError, KeyError) as exc:
                    rejected.append({'seat': seat, 'seed': seed, 'error': repr(exc)[:400], 'where': traceback.format_exc()[-1200:]})
                    print(json.dumps({'rejected_seat': seat, 'seed': seed, 'reason': repr(exc)[:300]}), flush=True)
                    continue
                cases.extend(batch)
                oracle.extend(rows)
                accepted.append({'seat': seat, 'seed': seed, 'case_count': len(batch)})
                found += 1
                print(json.dumps({'generated_seat': seat, 'seed': seed, 'cases': len(batch)}), flush=True)
                (fdir / 'cases.json').write_text(json.dumps(cases, indent=1))
                (fdir / 'oracle_results.json').write_text(json.dumps(oracle, indent=1))
                if found == opts.per_seat:
                    break
    finally:
        runner.close()
        (fdir / 'fixture_generation.json').write_text(json.dumps({'accepted': accepted, 'rejected': rejected,
            'selection_rule': f'first {opts.per_seat} seeds per seat (from {opts.seed_start}) where every case type constructs legally and verifies against its oracle; no policy outputs consulted'}, indent=1))


if __name__ == '__main__':
    main()
