"""Replay the frozen v3 fixture prefixes in the snapshot runtime and report drift.

For every original case: does every prefix action remain legal for the recorded
actor, and does the checkpoint observation snapshot match byte-for-byte?
"""
import json

from history_benchmark import OUT, EpisodeRunner, POLICIES, V3_POOL
from evaluate_history import ObservedFixture


def main():
    cases = json.loads((OUT / 'original/cases_v3.json').read_text())
    runner = EpisodeRunner(POLICIES['u8223'], V3_POOL, 'cpu')
    rows = []
    try:
        for case in cases:
            game = ObservedFixture(runner, case['seed'], case['seat'], case['leader'])
            info = dict(case=case['id'], prefix_len=len(case['prefix']))
            try:
                for i, row in enumerate(case['prefix']):
                    if game.actor != row['seat'] or bool(runner.base_env._building) != row['draft']:
                        raise AssertionError(f'actor/phase mismatch at {i}')
                    game.step(row['action'])
                snap = game.snap()
                diff = {k: (case['checkpoint'].get(k), snap.get(k)) for k in set(snap) | set(case['checkpoint']) if snap.get(k) != case['checkpoint'].get(k)}
                info.update(replayed=True, checkpoint_match=not diff, diff_keys=sorted(diff), diff={k: v for k, v in diff.items() if k != 'legal'})
                if 'legal' in diff:
                    old, new = diff['legal']
                    info['legal_added'] = [a for a in new or [] if a not in (old or [])]
                    info['legal_removed'] = [a for a in old or [] if a not in (new or [])]
            except Exception as exc:  # noqa: BLE001
                info.update(replayed=False, failed_at=len(game.actions), error=repr(exc)[:600])
            rows.append(info)
            print(json.dumps({k: info[k] for k in ('case', 'replayed') if k in info} | {'match': info.get('checkpoint_match'), 'diff': info.get('diff_keys'), 'err': info.get('error', '')[:200]}), flush=True)
    finally:
        runner.close()
    summary = dict(cases=len(rows), replayed=sum(r['replayed'] for r in rows), checkpoint_match=sum(bool(r.get('checkpoint_match')) for r in rows), rows=rows)
    (OUT / 'drift_v3_replay.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('cases', 'replayed', 'checkpoint_match')}))


if __name__ == '__main__':
    main()
