"""Snapshot-runtime harness for constructed tactical fixtures.

Self-contained port of the pieces of probe_gate_kl.EpisodeRunner and
play_selfplay_games.GameLogger that leader_history_benchmark_v1 used, rebound to
/home/skylark/git/azuki-tcg-specialist-runtime (legacy single-env Python driver,
CPU inference).  Every policy is evaluated with the SAME env/deck-pool so all
models observe byte-identical histories.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

RT = Path('/home/skylark/git/azuki-tcg-specialist-runtime')
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(RT / 'build/python/src'), str(RT / 'python/src')]

DISABLED_PROCESS_ENV = {
    'AZK_DRAFT_REF_SEAT_PROB': '0', 'AZK_DRAFT_REF_LEARNER_FIXED': '0', 'AZK_DRAFT_REF_OPPONENT_ONLY': '0',
    'AZK_DRAFT_REF_DECK_INDICES': '', 'AZK_DRAFT_PREFIX_PROBS': '', 'AZK_DRAFT_PREFIX_POOL_PATH': '',
    'AZK_DRAFT_NORMAL_PENALTY_CONFIG': '', 'AZK_DRAFT_NORMAL_PENALTY_COEF_INITIAL': '0',
    'AZK_DRAFT_NORMAL_PENALTY_COEF_FINAL': '0', 'AZK_FIXED_SEAT_DECK_INDICES': '', 'AZK_DECKBUILD_SNAPSHOT_DIR': '',
}
os.environ.update(DISABLED_PROCESS_ENV)

import torch  # noqa: E402
import azk_puffer.vector as azk_vector  # noqa: E402
from action import ActionType  # noqa: E402,F401
from deck_building import PlayerDeckBuildState  # noqa: E402,F401
from evaluate_checkpoint import _apply_checkpoint_resume_policy_config, _unwrap_base_env  # noqa: E402
from policy.v2 import tcg_sampler  # noqa: E402
from train import _load_model_weights  # noqa: E402
from training_utils import build_policy, build_vecenv, install_tcg_sampler, load_training_config  # noqa: E402

torch.set_num_threads(1)

CONFIG = RT / 'train-specialist/runs/water/specialist_water_s30m_r1.ini'
V3_POOL = Path('/home/skylark/git/azuki-tcg/train-ablation-1781126582/results/bptt_league_15m_v3/deck_pool.json')
CURATED_POOL = RT / 'train-specialist/decks/curated_deck_pool.json'
_A = Path('/home/skylark/git/azuki-tcg/train-ablation-1781126582/results/control_strategy_scale_30m_v1/runs/control/artifacts')
_S = RT / 'train-specialist/runs'
POLICIES = {
    'u8223': _A / 'azuki_local_control_strategy_scale_30m_s43_179038406361/model_azuki_local_008223.pt',
    'u9305': _A / 'azuki_local_control_strategy_scale_30m_s43_179039793722/model_azuki_local_009305.pt',
    'water': _S / 'water/artifacts/azuki_local_specialist_water_s43_179083097333/model_azuki_local_011361.pt',
    'earth': _S / 'earth/artifacts/azuki_local_specialist_earth_s43_179084927273/model_azuki_local_011422.pt',
    'lightning': _S / 'lightning/artifacts/azuki_local_specialist_lightning_s43_179087061766/model_azuki_local_011502.pt',
    'fire': _S / 'fire/artifacts/azuki_local_specialist_fire_s43_179089277894/model_azuki_local_011477.pt',
    # Earth continuation (runs/earth_c1): +75M (106M total) and final +100M (130M total).
    'earth_c75': _S / 'earth_c1/artifacts/azuki_local_specialist_earth_s43_179098246978/model_azuki_local_019500.pt',
    'earth_c100': _S / 'earth_c1/artifacts/azuki_local_specialist_earth_s43_179098246978/model_azuki_local_022062.pt',
}
PHASE_NAMES = {0: 'MULLIGAN', 1: 'START_OF_TURN', 2: 'MAIN', 3: 'RESPONSE', 4: 'COMBAT_RESOLVE', 5: 'END_TURN_ACTION', 6: 'END_TURN', 7: 'END_MATCH'}
GATE_NAMES = {'STT01-002': 'Surge(L)', 'AZK01-120': 'Stormchain(L)', 'STT02-002': 'Hydromancy(W)', 'AZK01-126': 'EchoedWaves(W)',
              'AZK01-122': 'Rushfire(F)', 'STT04-002': 'Ragefire(F)', 'AZK01-124': 'Devotion(E)', 'STT03-002': 'Stonehaven(E)'}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_winner(env, *, terminated: bool, truncated: bool) -> int:
    if not terminated or truncated:
        return -1
    r0, r1 = env._terminal_rewards
    return 0 if r0 > r1 else 1 if r1 > r0 else -1


class EpisodeRunner:
    """Legacy (non-native) single env + one policy.  deck_pool is forced so that
    every checkpoint sees the same draft catalog / fixed-deck catalog."""

    def __init__(self, checkpoint: Path | None, deck_pool: Path, device: str = 'cpu', config: Path = CONFIG):
        args = load_training_config(config, [])
        args['train']['device'] = device
        _apply_checkpoint_resume_policy_config(args, checkpoint)
        env = args['env']
        env.update(deck_building_enabled=True, native=False, reward_telemetry=False, pbrs_mode='legacy',
                   pbrs_terminal_closure=False, reward_decomposed_schedule=False, draft_uniform_assignment=True,
                   draft_same_element_matchup_prob=0.0, draft_cross_gate_replay_prob=0.0,
                   prebuilt_curriculum=False, prebuilt_probability=0.0, learner_element='none',
                   deck_pool_path=str(deck_pool))
        env.pop('native_envs_per_instance', None)
        install_tcg_sampler()
        self.device = device
        self.deck_pool = str(deck_pool)
        self.vecenv = build_vecenv(args, backend=azk_vector.Serial, num_envs=1, seed=7)
        base = _unwrap_base_env(self.vecenv.envs[0])
        while not getattr(type(base), 'is_deck_building_wrapper', False) and hasattr(base, 'env'):
            base = base.env
        self.base_env = base
        self.catalog = base._catalog
        self.policy = build_policy(self.vecenv, args)
        self.vecenv.async_reset(seed=7)
        obs, _, _, _, _, _, masks = self.vecenv.recv()
        state = {'mask': torch.as_tensor(masks), 'lstm_h': torch.zeros(self.vecenv.num_agents, self.policy.hidden_size),
                 'lstm_c': torch.zeros(self.vecenv.num_agents, self.policy.hidden_size)}
        with torch.no_grad():
            self.policy.forward_eval(torch.as_tensor(obs), state)
        if checkpoint is not None:
            _load_model_weights(self.policy, Path(checkpoint), device=device, strict=True)
        self.policy.eval().requires_grad_(False)
        self.code_to_def = {r.card_code: d for d, r in self.catalog.records_by_def_id.items()}

    def close(self):
        self.vecenv.close()


class GameLogger:
    def __init__(self, runner: EpisodeRunner, log_legal_actions: bool = True):
        self.runner = runner
        self.log_legal_actions = log_legal_actions
        self.records = runner.catalog.records_by_def_id
        self.inner = runner.base_env.env

    def code(self, def_id) -> str:
        rec = self.records.get(int(def_id))
        return rec.card_code if rec else f'?{int(def_id)}'

    @staticmethod
    def _valid(entries, count=None):
        out = []
        for i, e in enumerate(entries):
            if count is not None and i >= count:
                break
            if int(e.card_def_id) >= 0:
                out.append(e)
        return out

    def _board_list(self, slots):
        out = []
        for e in self._valid(slots):
            atk = int(e.cur_stats.cur_atk) if e.has_cur_stats else -1
            hp = int(e.cur_stats.cur_hp) if e.has_cur_stats else -1
            s = f'{self.code(e.card_def_id)}@{int(e.zone_index)}:{atk}/{hp}'
            if e.tap_state.tapped:
                s += 'T'
            if e.has_defender:
                s += 'D'
            if int(e.weapon_count) > 0:
                s += '+' + ','.join(self.code(w.card_def_id) for w in e.weapons[: int(e.weapon_count)])
            out.append(s)
        return out

    def snapshot(self, actor: int, include_legal: bool | None = None) -> dict:
        raw = self.inner._raw_observation(actor)
        me, opp = raw.my_observation_data, raw.opponent_observation_data
        include_legal = self.log_legal_actions if include_legal is None else include_legal
        snap = {
            'phase': PHASE_NAMES.get(int(raw.phase), str(int(raw.phase))),
            'hand': [self.code(h.card_def_id) for h in self._valid(me.hand, int(me.hand_count))],
            'hand_n': int(me.hand_count),
            'my_garden': self._board_list(me.garden), 'my_alley': self._board_list(me.alley),
            'opp_garden': self._board_list(opp.garden), 'opp_alley': self._board_list(opp.alley),
            'my_hp': int(me.leader.cur_stats.cur_hp), 'opp_hp': int(opp.leader.cur_stats.cur_hp),
            'my_leader_atk': int(me.leader.cur_stats.cur_atk), 'opp_leader_atk': int(opp.leader.cur_stats.cur_atk),
            'my_leader_weapons': [self.code(w.card_def_id) for w in me.leader.weapons[: int(me.leader.weapon_count)]],
            'opp_leader_weapons': [self.code(w.card_def_id) for w in opp.leader.weapons[: int(opp.leader.weapon_count)]],
            'ikz': [sum(1 for e in self._valid(me.ikz_area) if not e.tap_state.tapped), len(self._valid(me.ikz_area))],
            'ikz_token': bool(me.has_ikz_token),
            'my_discard': [self.code(c.card_def_id) for c in self._valid(me.discard)],
            'opp_discard': [self.code(c.card_def_id) for c in self._valid(opp.discard)],
            'my_discard_n': len(self._valid(me.discard)), 'opp_discard_n': len(self._valid(opp.discard)),
            'my_deck_n': int(me.deck_count), 'gate_tapped': bool(me.gate.tap_state.tapped),
        }
        ac = raw.ability_context
        if bool(ac.has_source_card_def_id):
            snap['ability_src'] = self.code(ac.source_card_def_id)
            snap['ability_phase'] = int(ac.phase)
        sel_n = int(me.selection_count)
        if any(int(me.selection[i].card_def_id) >= 0 for i in range(sel_n)):
            snap['selection'] = [self.code(me.selection[i].card_def_id) if int(me.selection[i].card_def_id) >= 0 else None for i in range(sel_n)]
        cc = raw.combat_context
        if bool(cc.combat_active):
            snap['combat'] = {'attacker': 'LEADER' if cc.attacker_is_leader else self.code(cc.attacker_card_def_id),
                              'target': 'LEADER' if cc.target_is_leader else self.code(cc.target_card_def_id),
                              'attacker_is_self': bool(cc.attacker_is_self), 'response_open': bool(cc.response_window_active),
                              'intercepted': bool(cc.defender_intercepted)}
        if include_legal:
            mask = raw.action_mask
            snap['legal'] = [[int(mask.legal_primary[i]), int(mask.legal_sub1[i]), int(mask.legal_sub2[i]), int(mask.legal_sub3[i])]
                             for i in range(int(mask.legal_action_count))]
        return snap

    def deck_record(self):
        decks = []
        for i in range(2):
            st = self.runner.base_env._states[i]
            decks.append({'gate': self.code(st.gate_card_def_id), 'gate_name': GATE_NAMES.get(self.code(st.gate_card_def_id), '?'),
                          'leader': self.code(st.leader_card_def_id),
                          'main': [self.code(st.main_card_def_ids[k]) for k in range(st.main_count)]})
        return decks


def set_eval_sampling():
    tcg_sampler.set_sampling_params(primary_temperature=1., subaction_temperature=1., smoothing_eps=0.,
                                    legal_row_temperature=1., deck_pick_smoothing_eps=0.)


set_eval_sampling()
