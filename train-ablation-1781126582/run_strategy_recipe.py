#!/usr/bin/env python3
"""Run registered strategy acquisition diagnostics; never rank by wins or launch 1B."""
from __future__ import annotations

import argparse
import configparser
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from evaluate_reward_screens import _checkpoint_from_manifest, _training_summary
from monitor_local_production import _atomic_json, _discord_post
from run_prefix_followup import summarize
from run_strategy_discovery import metric_failure, read_metrics, sha256, verify_registration_files
from report_strategy_recipe import report

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
RECIPES = {'CONTROL', 'RANDOM20', 'STRATEGIC20', 'STRATEGIC50', 'STRATEGIC20_HOLD_EXPLORATION'}


def validate(registration: dict) -> None:
    if (registration.get('schema_id') != 'azuki.ablation_registration'
            or registration.get('family') not in ('strategy_recipe_v1', 'strategy_retention_v1', 'random50_continuation_v1')
            or registration.get('scope') != 'diagnostic_only'
            or registration.get('status') != 'registered'
            or registration.get('production_qualified') is not False):
        raise ValueError('Only registered non-production strategy-recipe diagnostics are accepted')
    verify_registration_files(registration)
    arms = registration['arms']
    pairs = {(arm['recipe'], arm['seed']) for arm in arms}
    smoke = registration.get('smoke') is True
    retention = registration['family'] == 'strategy_retention_v1'
    continuation = registration['family'] == 'random50_continuation_v1'
    recipes = {'STRATEGIC50', 'RANDOM50'} if retention else {'STRATEGIC50'} if smoke else RECIPES
    expected = ({('RANDOM50', seed) for seed in (43, 44)} if continuation else
                {(recipe, seed) for recipe in recipes for seed in ((43,) if smoke else (43, 44))})
    if pairs != expected or len(arms) != len(expected) or len({a['id'] for a in arms}) != len(arms):
        raise ValueError('Recipe/seed panel does not match the authorized comparison')
    for arm in arms:
        config = configparser.ConfigParser(interpolation=None)
        config.read(ROOT / arm['config'])
        rows = 153_600 if smoke else 50_012_160 if retention else 15_006_720
        updates = [10] if smoke else [325, 650, 975, 1625, 2275, 3250] if retention else [325, 650, 975]
        if continuation:
            rows = (3266 if smoke else 13021) * 15_360
            updates = [3256, 3266] if smoke else [3256, 6511, 13021]
        if (arm['total_timesteps'] != rows or config.getint('train', 'total_timesteps') != rows
                or arm['total_updates'] != rows // 15_360 or arm['evaluation_updates'] != updates
                or config.getint('train', 'seed') != arm['seed']
                or arm.get('production_qualified') is not False):
            raise ValueError(f'Unregistered horizon/seed: {arm["id"]}')
        for field in ('config', 'evaluation_config'):
            if sha256(ROOT / arm[field]) != arm[field + '_sha256']:
                raise ValueError(f'Changed {field}: {arm["id"]}')
        if dict(config['process_env']) != {k.lower(): str(v) for k, v in arm['process_env'].items()}:
            raise ValueError(f'Process environment differs from registration: {arm["id"]}')
        if continuation:
            validate_continuation_arm(arm, config)
        if retention or continuation:
            expected_pool = registration['prefix_pool']['path'] if arm['recipe'] == 'STRATEGIC50' else ''
            if (config.get('process_env', 'azk_draft_prefix_probs') != '0.5,0.5'
                    or config.get('process_env', 'azk_draft_prefix_lengths') != '0,4'
                    or config.get('process_env', 'azk_draft_prefix_pool_path') != expected_pool
                    or config.getfloat('process_env', 'azk_exploration_scale_final') != 0.15
                    or config.getint('process_env', 'azk_exploration_anneal_end_rows') != 15_006_720
                    or config.getint('process_env', 'azk_potential_anneal_end_rows') != 15_006_720):
                raise ValueError(f'Retention treatment differs from authorized recipe: {arm["id"]}')
    tasks = registration['paired_template']['tasks']
    if len(tasks) != (4 if smoke else 512) or len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Wrong paired task coverage')
    if continuation:
        heldout = registration['heldout_template']
        paired = registration['paired_template']
        expected_tasks = [{**task, 'seed': task['seed'] + 1_000_000_000,
                           'task_id': 'heldout_' + task['task_id']} for task in tasks]
        if heldout['tasks'] != expected_tasks or any(
                heldout[key] != paired[key] for key in paired if key != 'tasks'):
            raise ValueError('Held-out panel must change only task IDs and seeds')
        if registration['expected_evaluation_games'] != len(arms) * len(updates) * 2 * len(tasks) * 2:
            raise ValueError('Evaluation allocation must count both panels')


def guard_parent(arm: dict) -> None:
    parent = arm['resume_checkpoint']
    for field in ('checkpoint', 'trainer_state'):
        if sha256(ROOT / parent[field]) != parent[field + '_sha256']:
            raise ValueError(f'Changed parent {field}: {arm["id"]}')
    league = arm['league_restore']
    for item in [league['state'], league['promotion_state'], *league['checkpoints']]:
        if sha256(ROOT / item['source']) != item['sha256']:
            raise ValueError(f'Changed parent league artifact: {item["source"]}')
        if item.get('metadata_source') and sha256(ROOT / item['metadata_source']) != item['metadata_sha256']:
            raise ValueError('Changed parent league checkpoint metadata')


def validate_continuation_arm(arm: dict, config: configparser.ConfigParser) -> None:
    if (arm.get('initialization') != 'full_state_continuation' or arm.get('parent_update') != 3256
            or arm['resume_checkpoint']['update'] != 3256 or arm['id'] != f'RANDOM50_S{arm["seed"]}'
            or config.getfloat('train', 'learning_rate') != 3e-5):
        raise ValueError('Unauthorized continuation identity or learning rate')
    required = {'load_optimizer': True, 'strict': True, 'restart_lr_schedule': True, 'auto_reset_critic': False}
    if set(config['resume']) != set(required) or any(config.getboolean('resume', k) != v for k, v in required.items()):
        raise ValueError('Unsafe continuation resume flags')
    if any(key.upper().startswith('AZK_RESUME_') for key in arm['process_env']):
        raise ValueError('Resume environment overrides are prohibited')
    original = configparser.ConfigParser(interpolation=None)
    if sha256(ROOT / arm['parent_config']) != arm['parent_config_sha256']:
        raise ValueError('Changed parent training config')
    original.read(ROOT / arm['parent_config'])
    allowed = {
        'base': {'tag', 'jsonl_log'}, 'train': {'total_timesteps', 'data_dir', 'learning_rate'},
        'league': {'state_path', 'promotion_state_path', 'promotion_records_dir', 'opponent_dir'},
        'resume': set(required), 'artifacts': {'evaluation_interval_updates', 'milestone_interval_updates'}}
    if arm['total_updates'] == 3266:
        allowed['train'].add('checkpoint_interval')
        allowed['artifacts'].add('recovery_interval_updates')
    for section in set(original.sections()) | set(config.sections()):
        before = dict(original[section]) if original.has_section(section) else {}
        after = dict(config[section]) if config.has_section(section) else {}
        for key in set(before) | set(after):
            if key not in allowed.get(section, set()) and before.get(key) != after.get(key):
                raise ValueError(f'Unauthorized parent config change: {section}.{key}')
    parent = arm['resume_checkpoint']
    model = ROOT / parent['checkpoint']
    if (ROOT / parent['trainer_state']).resolve() != model.with_name(f'trainer_state_{model.stem.rsplit("_", 1)[-1]}.pt').resolve():
        raise ValueError('Parent must have its update-specific trainer state')
    metadata = json.loads(model.with_suffix(model.suffix + '.meta.json').read_text())
    saved_env = metadata['resume_config_fingerprint']
    for group in ('schedule_env', 'reward_env'):
        if any(str(arm['process_env'].get(k, '')) != str(v) for k, v in saved_env[group].items()):
            raise ValueError(f'Continuation changes parent {group}')
    child = (ROOT / arm['result_root']).resolve()
    if child == (ROOT / parent['experiment_dir']).resolve() or child in model.resolve().parents:
        raise ValueError('Child output overlaps parent')
    league = arm['league_restore']
    for key, field in [('state', 'state_path'), ('promotion_state', 'promotion_state_path')]:
        if (ROOT / config.get('league', field)).resolve() != (ROOT / league[key]['destination']).resolve():
            raise ValueError('Trainer league state path differs from frozen restoration destination')
    for section, field in [('base', 'jsonl_log'), ('train', 'data_dir'),
                           ('league', 'promotion_records_dir'), ('league', 'opponent_dir')]:
        if not (ROOT / config.get(section, field)).resolve().is_relative_to(child):
            raise ValueError(f'Trainer writable path is outside child: {section}.{field}')
    for item in [league['state'], league['promotion_state'], *league['checkpoints']]:
        for field in ('destination', 'metadata_destination'):
            if field in item and not (ROOT / item[field]).resolve().is_relative_to(child):
                raise ValueError('League restoration must write only within child output')
    guard_parent(arm)
    import torch
    saved = torch.load(ROOT / parent['trainer_state'], map_location='cpu', weights_only=False)
    if (saved.get('update') != 3256 or saved.get('model_name') != model.name
            or not saved.get('optimizer_state_dict') or not saved.get('scheduler_state_dict')
            or not saved.get('coordinator_rng_state')
            or saved.get('global_step') != {43: 30_412_921, 44: 30_441_468}[arm['seed']]
            or saved.get('env_completed_episodes') != {43: 272, 44: 274}[arm['seed']]):
        raise ValueError('Parent trainer state does not support the registered full-state continuation')


def restore_league(arm: dict) -> None:
    guard_parent(arm)
    league = arm['league_restore']
    def rewrite(value):
        if isinstance(value, str):
            return league['path_rewrites'].get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value
    for item in league['checkpoints']:
        for source, destination in [('source', 'destination'), ('metadata_source', 'metadata_destination')]:
            if source in item:
                target = ROOT / item[destination]
                target.parent.mkdir(parents=True, exist_ok=True)
                with (ROOT / item[source]).open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    for key in ('state', 'promotion_state'):
        item = league[key]
        target = ROOT / item['destination']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as handle:
            json.dump(rewrite(json.loads((ROOT / item['source']).read_text())), handle, indent=2)


def stop(process: subprocess.Popen) -> None:
    """Terminate the owned subprocess group, including native worker children."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class Campaign:
    def __init__(self, args: argparse.Namespace, registration: dict):
        self.args = args
        self.registration = registration
        self.continuation = registration['family'] == 'random50_continuation_v1'
        self.panels = ('paired', 'heldout') if self.continuation else ('paired',)
        self.folder = args.registration.resolve().parent
        self.path = self.folder / 'campaign_status.json'
        if self.path.exists():
            raise FileExistsError(f'Refusing to overwrite execution evidence: {self.path}')
        self.last_notice = time.monotonic()
        self.state = {
            'schema_id': 'azuki.strategy_recipe_status', 'schema_version': 1,
            'registration_sha256': sha256(args.registration), 'production_qualified': False,
            'state': 'running', 'started_at': time.time(), 'arms': [],
            'notifications': [], 'pending_notifications': [],
            'objective': 'strategy emergence, persistence and diversity; no winrate gate',
        }

    def save(self) -> None:
        self.state['updated_at'] = time.time()
        _atomic_json(self.path, self.state)

    def notify(self, message: str) -> None:
        self.state['pending_notifications'].append(message)
        self.deliver()
        self.last_notice = time.monotonic()

    def deliver(self) -> None:
        while self.state['pending_notifications']:
            message = self.state['pending_notifications'][0]
            with contextlib.redirect_stdout(io.StringIO()):
                delivered = _discord_post(self.args, message)
            if not delivered:
                break
            self.state['pending_notifications'].pop(0)
            self.state['notifications'].append({'at': time.time(), 'message': message, 'delivered': True})
        self.save()

    def progress(self) -> None:
        self.save()
        if time.monotonic() - self.last_notice >= 1800:
            self.notify('**Strategy recipe progress**\n' + json.dumps(self.state.get('active', {}))
                        + '\nStrategy-first diagnostics; no 1B or automatic promotion.')
        elif self.state['pending_notifications']:
            self.deliver()

    def await_monitor(self) -> None:
        if not (self.continuation or self.args.require_discord_monitor):
            return
        self.state['active'] = {'phase': 'awaiting_independent_discord_attachment'}
        self.save()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            path = self.folder / 'discord_monitor_state.json'
            if path.exists():
                watcher = json.loads(path.read_text())
                checked = watcher.get('last_checked_at', 0)
                if (watcher.get('registration_sha256') == self.state['registration_sha256']
                        and 'attached' in watcher.get('sent_events', [])
                        and self.state['started_at'] <= checked <= time.time()
                        and time.time() - checked < 120):
                    self.state['monitor_attached_at'] = checked
                    self.save()
                    return
            time.sleep(2)
        raise RuntimeError('Independent Discord watcher did not deliver a fresh attachment within 300s')

    def train(self, arm: dict) -> dict:
        verify_registration_files(self.registration)
        if self.continuation:
            validate(self.registration)
        folder = ROOT / arm['result_root']
        if folder.exists():
            raise FileExistsError(f'Fresh training output already exists: {folder}')
        folder.mkdir(parents=True)
        if self.continuation:
            restore_league(arm)
        config = configparser.ConfigParser(interpolation=None)
        config.read(ROOT / arm['config'])
        data_dir = ROOT / config.get('train', 'data_dir')
        env = {k: v for k, v in os.environ.items() if not k.startswith('AZK_')}
        env.update(PYTHONPATH=f'{ROOT / "build/python/src"}:{ROOT / "python/src"}',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        state = {'state': 'running', 'arm': arm['id'], 'seed': arm['seed'],
                 'registration_sha256': self.state['registration_sha256'],
                 'config_sha256': arm['config_sha256'], 'started_at': time.time()}
        status_path = folder / 'run_status.json'
        _atomic_json(status_path, state)
        self.state['active'] = {'phase': 'training', 'arm': arm['id'], 'rows': arm['total_timesteps']}
        start = arm.get('parent_update', 0)
        self.notify(f'**Strategy recipe training started**\n{arm["id"]}: updates {start}->{arm["total_updates"]}; {arm["total_timesteps"]:,} cumulative configured rows; initialization={arm.get("initialization", "fresh")}. No winrate admission gate.')
        command = [str(PYTHON), 'python/src/train.py', '--config', arm['config']]
        if self.continuation:
            guard_parent(arm)
            command += ['--resume-checkpoint', arm['resume_checkpoint']['checkpoint'],
                        '--resume-load-optimizer', '--resume-strict',
                        '--resume-restart-lr-schedule', '--no-resume-auto-reset-critic']
        try:
            with (folder / 'training_console.log').open('x') as console:
                process = subprocess.Popen(command,
                                           cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    while process.poll() is None:
                        time.sleep(10)
                        if self.continuation:
                            console_text = (folder / 'training_console.log').read_text()
                            if any(marker in console_text for marker in (
                                    'skipping optimizer+step restore', 'optimizer restore skipped',
                                    'scheduler restore skipped', 'optimizer_restored=False',
                                    'coordinator_rng_restored=False', 'critic head reset')):
                                raise RuntimeError('Trainer rejected full-state restoration; refusing fallback')
                        metrics = read_metrics(folder / 'logs/production.jsonl')
                        failure = metric_failure(metrics)
                        if failure:
                            raise RuntimeError(failure)
                        if metrics:
                            state['progress'] = {key: metrics[-1][key] for key in
                                                 ('epoch', 'agent_steps', 'SPS') if key in metrics[-1]}
                            if self.continuation:
                                epoch = int(metrics[-1].get('epoch', start))
                                if epoch < start or epoch > arm['total_updates']:
                                    raise RuntimeError('Continuation logged an out-of-range absolute update')
                                state['progress'].update(continuation_updates=epoch - start,
                                                         cumulative_configured_rows=epoch * 15_360)
                            self.state['active'].update(state['progress'])
                            _atomic_json(status_path, state)
                        self.progress()
                    if process.returncode != 0:
                        raise RuntimeError(f'Training exited {process.returncode}; inspect {folder}')
                finally:
                    stop(process)
            metrics = read_metrics(folder / 'logs/production.jsonl')
            failure = metric_failure(metrics)
            if failure or not metrics:
                raise RuntimeError(failure or 'No training metrics')
            directories = [p for p in data_dir.glob(f'azuki_local_{arm["tag"]}_*') if p.is_dir()]
            if len(directories) != 1:
                raise RuntimeError('Expected exactly one fresh checkpoint directory')
            experiment = directories[0]
            _checkpoint_from_manifest(experiment, arm['total_updates'])
            for update in arm['evaluation_updates']:
                if self.continuation and update == start:
                    continue
                _checkpoint_from_manifest(experiment, update)
            state.update(state='completed', finished_at=time.time(),
                         latest_experiment_dir=str(experiment.relative_to(ROOT)),
                         training=_training_summary(folder / 'logs/production.jsonl'),
                         verified_final_update=arm['total_updates'])
            if state['training']['max_invalid_metric'] > 0:
                raise RuntimeError('Training recorded invalid-action metrics')
            telemetry_keys = sorted({key for row in metrics for key in row
                                     if key.startswith(('environment/draft_prefix/', 'environment/anneal/'))})
            state['intervention_telemetry'] = {
                key: {'first': values[0], 'last': values[-1], 'minimum': min(values),
                      'maximum': max(values), 'metric_windows': len(values)}
                for key in telemetry_keys
                if (values := [row[key] for row in metrics if isinstance(row.get(key), (int, float))])
            }
            if self.continuation:
                guard_parent(arm)
                console_text = (folder / 'training_console.log').read_text()
                if ('optimizer_restored=True' not in console_text
                        or f'epoch={start},' not in console_text
                        or 'scheduler_restored=True' not in console_text
                        or f'global_step={ {43: 30_412_921, 44: 30_441_468}[arm["seed"]]},' not in console_text
                        or f'remaining_epochs={arm["total_updates"] - start},' not in console_text
                        or 'coordinator_rng_restored=True' not in console_text):
                    raise RuntimeError('Full-state restoration was not confirmed by trainer')
                state.update(parent_update=start, continuation_updates=arm['total_updates'] - start,
                             cumulative_configured_rows=arm['total_timesteps'])
            _atomic_json(status_path, state)
            return state
        except BaseException as exc:
            state.update(state='failed', error=str(exc), finished_at=time.time())
            _atomic_json(status_path, state)
            raise

    def evaluate(self, arm: dict, training: dict | None, panel: str = 'paired') -> None:
        verify_registration_files(self.registration)
        parent_only = training is None
        if self.continuation:
            guard_parent(arm)
        folder = self.folder / panel / arm['name']
        folder.mkdir(parents=True, exist_ok=self.continuation)
        evaluation = json.loads(json.dumps(self.registration[f'{panel}_template']))
        evaluation['source_sha256'] = dict(self.registration['source_sha256'])
        evaluation['checkpoints'] = {}
        evaluation['parent_registration_sha256'] = self.state['registration_sha256']
        updates = [arm['parent_update']] if parent_only else arm['evaluation_updates']
        for update in updates:
            if self.continuation and update == arm['parent_update']:
                parent = arm['resume_checkpoint']
                checkpoint, digest = ROOT / parent['checkpoint'], parent['checkpoint_sha256']
            else:
                checkpoint, digest = _checkpoint_from_manifest(ROOT / training['latest_experiment_dir'], update)
            key = f'{arm["id"]}_p{update:06d}'
            evaluation['checkpoints'][key] = {'config': arm['evaluation_config'],
                                             'checkpoint': str(checkpoint.relative_to(ROOT)),
                                             'checkpoint_sha256': digest}
            metadata = checkpoint.with_suffix(checkpoint.suffix + '.meta.json')
            if metadata.exists():
                evaluation['source_sha256'][str(metadata.relative_to(ROOT))] = sha256(metadata)
        registration_path = folder / ('parent_registration.json' if parent_only else 'registration.json')
        with registration_path.open('x') as handle:
            json.dump(evaluation, handle, indent=2)
            handle.write('\n')
        self.notify(f'**Strategy recipe {panel} evaluation started**\n{arm["id"]}; free draft, prefixes disabled, both modes, candidate-only mechanics.')
        for key in evaluation['checkpoints']:
            if self.continuation and not parent_only and key == f'{arm["id"]}_p{arm["parent_update"]:06d}':
                continue
            for mode in ('sample', 'argmax'):
                batch = folder / 'eval' / key / mode
                batch.mkdir(parents=True)
                self.state['active'] = {'phase': f'{panel}_evaluation', 'arm': arm['id'], 'panel': panel,
                                        'checkpoint': key, 'mode': mode, 'expected_games': len(evaluation['tasks'])}
                live, logs, traces = [], [], []
                try:
                    workers = min(self.args.workers, len(evaluation['tasks']))
                    for shard in range(workers):
                        trace = batch / f'trace_{shard:02d}.jsonl'
                        log = (batch / f'worker_{shard:02d}.log').open('x')
                        traces.append(trace)
                        logs.append(log)
                        env = {k: v for k, v in os.environ.items() if not k.startswith('AZK_')}
                        env.update(evaluation['process_env'])
                        command = [str(PYTHON), 'train-ablation-1781126582/run_prefix_paired_eval.py',
                                   '--registration', str(registration_path), '--checkpoint-key', key,
                                   '--mode', mode, '--shards', str(workers), '--shard-index', str(shard),
                                   '--out', str(trace)]
                        live.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                     stderr=subprocess.STDOUT, start_new_session=True))
                    while any(p.poll() is None for p in live):
                        if any(p.poll() not in (None, 0) for p in live):
                            raise RuntimeError(f'Paired evaluation worker failed: {key}/{mode}')
                        self.state['active']['written_games'] = sum(
                            sum(1 for _ in p.open('rb')) for p in traces if p.exists())
                        self.progress()
                        time.sleep(10)
                    if any(p.returncode != 0 for p in live):
                        raise RuntimeError(f'Paired evaluation worker failed: {key}/{mode}')
                finally:
                    for process in live:
                        stop(process)
                    for log in logs:
                        log.close()
                result_path = batch / 'result.json'
                result = summarize(traces, evaluation, key, mode, result_path)
                if sum(c['player_games'] for c in result['candidate_only_context_counts'].values()) != result['games']:
                    raise RuntimeError('Candidate-only denominator mismatch')
                print(f'[strategy-recipe] evaluated {panel}/{key}/{mode}: {result["games"]} games', flush=True)
        report(self.args.registration, panel=panel)

    def run(self) -> None:
        try:
            print(f'[strategy-recipe] ready: {len(self.registration["arms"])} registered arms; no 1B', flush=True)
            self.save()
            self.await_monitor()
            if self.continuation:
                for arm in self.registration['arms']:
                    for panel in self.panels:
                        self.evaluate(arm, None, panel)
            for arm in self.registration['arms']:
                training = self.train(arm)
                for panel in self.panels:
                    self.evaluate(arm, training, panel)
                self.state['arms'].append({'id': arm['id'], 'state': 'evaluated'})
                self.save()
            for panel in self.panels:
                evidence = report(self.args.registration, panel=panel)
                if not evidence['complete']:
                    raise RuntimeError(f'{panel} evaluation remains provisional')
            self.state.update(state='completed_observations_require_review', finished_at=time.time())
            self.state.pop('active', None)
            self.notify('**STRATEGY EVALUATIONS FINISHED — READY FOR REVIEW**\nReplicated emergence, retention and diversity evidence is in strategy_report.json. Prompt the assistant to review this campaign and decide the next step. No winrate gate, automatic recipe selection or 1B launch.')
        except BaseException as exc:
            self.state.update(state='stopped', error=str(exc), finished_at=time.time())
            self.notify('**Strategy recipe campaign stopped**\n' + str(exc)
                        + '\nEvidence retained; no fallback training or 1B launched.')
            raise
        finally:
            self.save()
            while self.state['pending_notifications']:
                time.sleep(60)
                self.deliver()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--require-discord-monitor', action='store_true')
    parser.add_argument('--discord-webhook-file', type=Path,
                        default=Path.home() / '.config/azuki-tcg/discord-webhook-url')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    registration = json.loads(args.registration.read_text())
    validate(registration)
    print(f'[strategy-recipe] verified {len(registration["arms"])} arms', flush=True)
    if not args.validate_only:
        Campaign(args, registration).run()


if __name__ == '__main__':
    main()
