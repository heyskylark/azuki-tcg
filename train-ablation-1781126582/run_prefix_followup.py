#!/usr/bin/env python3
"""Run registered retained-checkpoint evaluation only; never train or promote."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from monitor_local_production import _atomic_json, _discord_post
from strategy_descriptor import build_strategy_descriptor

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def summarize(paths: list[Path], registration: dict, key: str, mode: str, output: Path) -> dict:
    expected = {task['task_id'] for task in registration['tasks']}
    seen = set()
    counts = defaultdict(Counter)
    scores = defaultdict(list)
    trace_hashes = {}
    for path in paths:
        trace_hashes[str(path.relative_to(ROOT))] = digest(path)
        grouped = defaultdict(list)
        with path.open() as handle:
            for line in handle:
                game = json.loads(line)
                task = game['paired_eval']
                identity = task['task_id']
                if identity in seen or identity not in expected:
                    raise ValueError(f'Duplicate or unexpected task {identity}')
                seen.add(identity)
                if not game['outcome']['terminated'] or game['outcome']['truncated']:
                    raise ValueError(f'Incomplete game {identity}; retain trace and hold review')
                if task['checkpoint_key'] != key or game['policy_action_mode'] != mode:
                    raise ValueError('Trace belongs to a different checkpoint or mode')
                grouped[task['opponent_id'], task['candidate_seat']].append(game)
                starter = next((s['p'] for s in game['steps'] if s.get('ph') == 'MAIN'), 'unknown')
                cell = f"{task['candidate_gate']}|{task['candidate_leader']}|opponent={task['opponent_id']}|{task['opponent_gate']}|{task['opponent_leader']}|seat={task['candidate_seat']}|starter={starter}"
                scores[cell].append(task['candidate_score'])
        for (opponent, seat), games in grouped.items():
            descriptor = build_strategy_descriptor(games, label=f'{key}_{mode}_{opponent}_seat{seat}', checkpoint_sha256=registration['checkpoints'][key]['checkpoint_sha256'])
            for context_key, context in descriptor['elemental_strategy']['contexts'].items():
                if context['seat'] == seat:
                    counts[f'opponent={opponent}|{context_key}'].update(context['counts'])
    if seen != expected:
        raise ValueError(f'Missing tasks: {len(seen)}/{len(expected)}')
    result = {
        'schema_id': 'azuki.prefix_paired_eval_result', 'schema_version': 1,
        'checkpoint_key': key, 'mode': mode, 'games': len(seen), 'incomplete': 0,
        'production_qualified': False, 'candidate_only_context_counts': {k: dict(v) for k, v in sorted(counts.items())},
        'supporting_scores': {k: {'games': len(v), 'score': sum(v) / len(v)} for k, v in sorted(scores.items())},
        'trace_sha256': trace_hashes,
        'interpretation': 'Free draft with continuous recurrent state versus frozen checkpoint opponents; candidate-seat counts only. Observed effects are not causal action quality. Deferred effects remain incompletely observed.'
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--discord-webhook-file', type=Path, default=Path.home() / '.config/azuki-tcg/discord-webhook-url')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    registration_path = args.registration.resolve()
    registration = json.loads(registration_path.read_text())
    if registration['schema_id'] != 'azuki.prefix_paired_eval_registration' or registration.get('production_qualified') is not False:
        raise ValueError('Only non-production paired evaluation registration is accepted')
    folder = registration_path.parent
    status_path = folder / 'campaign_status.json'
    if status_path.exists():
        raise FileExistsError('Refusing to replace campaign status; review retained output before recovery')
    for name, expected in registration['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Registered source changed: {name}')
    status = {'schema_id': 'azuki.prefix_followup_status', 'schema_version': 1, 'registration_sha256': digest(registration_path), 'state': 'running', 'started_at': time.time(), 'production_qualified': False, 'completed': [], 'notifications': [], 'pending_notifications': []}
    live = []
    last_notice = 0.0

    def save() -> None:
        status['updated_at'] = time.time()
        _atomic_json(status_path, status)

    def notify(message: str) -> None:
        status['pending_notifications'].append(message)
        deliver()

    def deliver() -> None:
        while status['pending_notifications']:
            message = status['pending_notifications'][0]
            with contextlib.redirect_stdout(io.StringIO()):
                sent = _discord_post(args, message)
            if not sent:
                break
            status['pending_notifications'].pop(0)
            status['notifications'].append({'at': time.time(), 'message': message, 'delivered': True})
        save()

    try:
        notify('**Azuki retained-checkpoint evaluation started**\nControl/random-prefix/strategic-prefix; early/middle/final; both modes and leaders; frozen opponents. Evaluation only, no training or 1B.')
        for key in registration['checkpoints']:
            for mode in ('sample', 'argmax'):
                batch = folder / 'eval' / key / mode
                batch.mkdir(parents=True, exist_ok=True)
                status['active'] = {'checkpoint': key, 'mode': mode, 'expected_games': len(registration['tasks'])}
                save()
                traces = []
                logs = []
                live = []
                try:
                    for shard in range(args.workers):
                        trace = batch / f'trace_{shard:02d}.jsonl'
                        log = (batch / f'worker_{shard:02d}.log').open('x')
                        logs.append(log)
                        traces.append(trace)
                        command = [sys.executable, str(ROOT / 'train-ablation-1781126582/run_prefix_paired_eval.py'), '--registration', str(registration_path), '--checkpoint-key', key, '--mode', mode, '--shards', str(args.workers), '--shard-index', str(shard), '--out', str(trace)]
                        environment = os.environ.copy()
                        environment.update(registration['process_env'])
                        live.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT))
                    while any(p.poll() is None for p in live):
                        if any(p.poll() not in (None, 0) for p in live):
                            raise RuntimeError(f'Worker failed: {key}/{mode}; inspect retained worker logs')
                        total = 0
                        for trace in traces:
                            if trace.exists():
                                with trace.open('rb') as handle:
                                    total += sum(1 for _ in handle)
                        status['active']['written_games'] = total
                        save()
                        now = time.time()
                        if now - last_notice >= 1800:
                            notify(f'**Azuki paired evaluation progress**\n{key} / {mode}: {total}/{len(registration["tasks"])} games; {len(status["completed"])}/{2 * len(registration["checkpoints"])} batches complete. No training or 1B.')
                            last_notice = now
                        elif status['pending_notifications']:
                            deliver()
                        time.sleep(30)
                    if any(p.returncode != 0 for p in live):
                        raise RuntimeError(f'Worker failed: {key}/{mode}')
                finally:
                    for p in live:
                        if p.poll() is None:
                            p.terminate()
                    for p in live:
                        try:
                            p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            p.kill()
                            p.wait()
                    for log in logs:
                        log.close()
                result_path = batch / 'result.json'
                result = summarize(traces, registration, key, mode, result_path)
                status['completed'].append({'checkpoint': key, 'mode': mode, 'games': result['games'], 'result': str(result_path.relative_to(ROOT)), 'result_sha256': digest(result_path)})
                save()
                print(f'[prefix-followup] completed {key}/{mode} games={result["games"]}', flush=True)
        status.update(state='completed_observations_require_review', finished_at=time.time())
        status.pop('active', None)
        notify('**PAIRED EVALUATIONS FINISHED — READY FOR REVIEW**\nAll registered retained-checkpoint comparisons and candidate-seat summaries are complete. Resume the assistant session for review. No training, promotion, or 1B launched.')
    except BaseException as exc:
        status.update(state='stopped', error=str(exc), finished_at=time.time())
        notify('**Azuki paired evaluation stopped**\nRetained evidence requires inspection; no fallback or training launched. Read prefix followup campaign_status.json and worker logs.')
        raise
    finally:
        save()
    while status['pending_notifications']:
        time.sleep(60)
        deliver()


if __name__ == '__main__':
    main()
