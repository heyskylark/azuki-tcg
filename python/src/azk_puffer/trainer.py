## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import sys
import glob
import ast
import time
import random
import shutil
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque

import numpy as np
import psutil

import pufferlib as upstream_pufferlib
import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import azk_puffer as pufferlib
import azk_puffer.pytorch
import azk_puffer.vector

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME
)
# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)


def terminal_win_labels_from_rewards(
        env_id: np.ndarray,
        label_mask: np.ndarray,
        terminal_rewards: np.ndarray,
        agents_per_env: int) -> dict[int, dict[int, float]]:
    """Build per-seat binary win labels from true terminal rows."""
    env_ids = np.asarray(env_id, dtype=np.int64).reshape(-1)
    labels = np.asarray(label_mask, dtype=np.bool_).reshape(-1)
    rewards = np.asarray(terminal_rewards, dtype=np.float32).reshape(-1)
    if env_ids.shape != labels.shape or env_ids.shape != rewards.shape:
        raise ValueError('env_id, label_mask, and terminal_rewards must have matching shapes')
    if agents_per_env < 1:
        raise ValueError('agents_per_env must be positive')

    out: dict[int, dict[int, float]] = {}
    for agent_id, is_terminal, reward in zip(env_ids, labels, rewards):
        if not bool(is_terminal) or float(reward) == 0.0:
            continue
        env_index = int(agent_id // agents_per_env)
        seat = int(agent_id % agents_per_env)
        out.setdefault(env_index, {})[seat] = 1.0 if float(reward) > 0.0 else 0.0
    return out


def trainer_shaped_reward_multiplier(
        update: int,
        *,
        enabled: bool,
        start_epoch: int,
        end_epoch: int) -> float:
    """Return the absolute-update multiplier for trainer-side shaped reward."""
    if not enabled:
        return 1.0
    if start_epoch < 0 or end_epoch <= start_epoch:
        raise ValueError('trainer shaped-reward anneal requires 0 <= start_epoch < end_epoch')
    absolute_update = int(update)
    if absolute_update <= start_epoch:
        return 1.0
    if absolute_update >= end_epoch:
        return 0.0
    return float(end_epoch - absolute_update) / float(end_epoch - start_epoch)


def recombine_reward_components(
        raw_total: torch.Tensor,
        terminal: torch.Tensor,
        shaped: torch.Tensor,
        multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale only shaped reward while preserving exact pre-anneal totals."""
    scale = float(multiplier)
    if not 0.0 <= scale <= 1.0:
        raise ValueError('shaped reward multiplier must be in [0, 1]')
    if scale == 1.0:
        return raw_total, shaped
    scaled_shaped = shaped * scale
    return torch.clamp(terminal + scaled_shaped, -1, 1), scaled_shaped


class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision('high')
        if torch.cuda.is_available() and config['device'] == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        # The project entrypoint optionally seeds process RNGs before policy
        # construction. This seed controls the vector-environment reset.
        seed = config['seed']

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience
        if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
            raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
        elif config['batch_size'] == 'auto':
            config['batch_size'] = total_agents * config['bptt_horizon']
        elif config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = config['batch_size'] // total_agents

        batch_size = config['batch_size']
        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        device = config['device']
        self.observations = torch.zeros(segments, horizon, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self._rollout_obs_device = None
        if device == 'cuda' and not config['cpu_offload']:
            self._rollout_obs_device = torch.empty(
                (vecenv.agents_per_batch, *obs_space.shape),
                dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[
                    obs_space.dtype
                ],
                device=device,
            )
        self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(segments, horizon, device=device)
        self.terminal_values = torch.zeros(segments, horizon, device=device)
        self.shaped_values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminal_reward_components = torch.zeros(segments, horizon, device=device)
        self.shaped_reward_components = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents
        # Group rows by GAME, not by worker instance: a native driver_env packs
        # many independent games (num_agents = 2 * envs_per_instance), but the
        # global row layout is [g0_p0, g0_p1, g1_p0, ...] on every path, so
        # game index = row // agents_per_match. Envs expose agents_per_match;
        # falling back to num_agents preserves legacy 2-agent wrappers.
        driver = vecenv.driver_env
        self._agents_per_env = max(
            1, int(getattr(driver, 'agents_per_match', getattr(driver, 'num_agents', 1)))
        )
        self._num_envs_total = max(1, int(total_agents // self._agents_per_env))
        self._env_episode_ids = np.arange(self._num_envs_total, dtype=np.int64)
        self._next_env_episode_id = int(self._num_envs_total)
        self.win_prob_targets = torch.zeros(segments, horizon, device=device)
        self.win_prob_target_mask = torch.zeros(segments, horizon, device=device, dtype=torch.bool)
        self.win_prob_episode_ids = torch.full((segments, horizon), -1, device=device, dtype=torch.int64)
        self.win_prob_agent_ids = torch.full((segments, horizon), -1, device=device, dtype=torch.int32)
        self._win_prob_zero_label_epochs = 0

        anneal_flag = os.environ.get('AZK_TRAINER_SHAPED_REWARD_ANNEAL', '')
        self._trainer_shaped_reward_anneal_enabled = (
            anneal_flag.strip().lower() not in {'', '0', 'false', 'no', 'off'}
        )
        self._trainer_shaped_reward_start_epoch = int(
            os.environ.get('AZK_TRAINER_SHAPED_REWARD_ANNEAL_START_EPOCH', '0') or 0
        )
        self._trainer_shaped_reward_end_epoch = int(
            os.environ.get('AZK_TRAINER_SHAPED_REWARD_ANNEAL_END_EPOCH', '1') or 1
        )
        if self._trainer_shaped_reward_anneal_enabled:
            trainer_shaped_reward_multiplier(
                0,
                enabled=True,
                start_epoch=self._trainer_shaped_reward_start_epoch,
                end_epoch=self._trainer_shaped_reward_end_epoch,
            )
        self._trainer_shaped_reward_multiplier = 1.0
        self._trainer_shaped_reward_update = 0

        # LSTM
        if config['use_rnn']:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

        # A-DRAFTAUX: draft->battle boundary auxiliary pick credit (design in
        # train-ablation-1781126582/draft-aux-design.md). Off unless env knobs set.
        self._draftaux_vboot = float(os.environ.get('AZK_DRAFT_VBOOT_COEF', '0') or 0.0)
        self._draftaux_sibdiff = float(os.environ.get('AZK_DRAFT_SIBDIFF_COEF', '0') or 0.0)
        self._draftaux_cap = float(os.environ.get('AZK_DRAFT_SIBDIFF_CAP', '0.05') or 0.05)
        self._draftaux_enabled = (
            self._draftaux_vboot > 0.0 or self._draftaux_sibdiff > 0.0
        ) and bool(config['use_rnn'])
        self._draftaux_layout = None
        self._draftaux_prev = {}
        self._draftaux_injected = 0.0
        self._draftaux_events = 0
        # S1: scale the aux by the env's live reward_shaping_scale so the
        # seeding force fades with the anneal instead of Goodharting at length.
        self._draftaux_anneal = os.environ.get('AZK_DRAFT_AUX_ANNEAL') == '1'
        self._draftaux_last_scale = 1.0

        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f'minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly')

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f'batch_size {batch_size} must be >= minibatch_size {minibatch_size}'
            )

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        if config['compile']:
            base = getattr(policy, 'policy', None)
            if base is not None and hasattr(base, 'encode_observations') and hasattr(base, 'decode_actions'):
                # Azuki TCG policy: compile the encode/decode hot paths and keep
                # the LSTM cell + glue eager. The default recompile_limit (8) is
                # load-bearing: the _PackedField.extract frames specialize per
                # field spec and must cap to eager quickly, while the hot
                # encoder/decoder graphs compile once per shape family. Raising
                # the limit (or dynamo-disabling the canonicalizer) was measured
                # 10-70% slower end to end.
                # Keep the eager originals reachable: CUDA-graph capture must
                # record the eager kernels (dynamo guards read the CUDA RNG
                # seed, which is illegal inside stream capture).
                base._eager_encode_observations = base.encode_observations
                base._eager_decode_actions = base.decode_actions
                base.encode_observations = torch.compile(base.encode_observations, mode=config['compile_mode'])
                base.decode_actions = torch.compile(base.decode_actions, mode=config['compile_mode'])
            else:
                self.policy = torch.compile(policy, mode=config['compile_mode'])
                self.policy.forward_eval = torch.compile(policy, mode=config['compile_mode'])
                pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, mode=config['compile_mode'])

        # Manual CUDA graphs for the launch-bound rollout forward (Azuki
        # policy only; captures one graph per legal-action trim bucket).
        if config.get('cuda_graphs', False) and 'cuda' in str(config['device']):
            enable_graphs = getattr(policy, 'enable_rollout_cuda_graphs', None)
            if enable_graphs is not None:
                enable_graphs()

        # Optimizer
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
                fused='cuda' in str(config['device']),
            )
        elif config['optimizer'] == 'muon':
            import heavyball
            from heavyball import ForeachMuon
            warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
            heavyball.utils.compile_mode = "default"

            # # optionally a little bit better/faster alternative to newtonschulz iteration
            # import heavyball.utils
            # heavyball.utils.zeroth_power_mode = 'thinky_polar_express'

            # heavyball_momentum=True introduced in heavyball 2.1.1
            # recovers heavyball-1.7.2 behaviour - previously swept hyperparameters work well
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
                heavyball_momentum=True,
            )
        else:
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config['total_timesteps'] // config['batch_size']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = contextlib.nullcontext()
        if config.get('amp', True) and config['device'] == 'cuda':
            self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        if self._trainer_shaped_reward_anneal_enabled:
            print(
                '[trainer-shaped-reward] enabled: '
                f'start_epoch={self._trainer_shaped_reward_start_epoch}, '
                f'end_epoch={self._trainer_shaped_reward_end_epoch}'
            )

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def _base_policy_module(self):
        return getattr(self.uncompiled_policy, 'policy', self.uncompiled_policy)

    def _win_prob_aux_enabled(self) -> bool:
        base_policy = self._base_policy_module()
        return bool(getattr(base_policy, 'win_prob_aux_enabled', False)) and getattr(base_policy, 'win_prob_fn', None) is not None

    def _win_prob_aux_coef(self) -> float:
        base_policy = self._base_policy_module()
        return float(getattr(base_policy, 'win_prob_aux_coef', 0.0))

    def _split_value_heads_enabled(self) -> bool:
        base_policy = self._base_policy_module()
        return bool(getattr(base_policy, 'split_value_heads_enabled', False)) and getattr(base_policy, 'value_terminal_fn', None) is not None and getattr(base_policy, 'value_shaped_fn', None) is not None

    def _split_value_component_coef(self) -> float:
        base_policy = self._base_policy_module()
        return float(getattr(base_policy, 'split_value_component_coef', 0.0))

    def _reset_win_prob_rollout_buffers(self):
        self.win_prob_targets.zero_()
        self.win_prob_target_mask.zero_()
        self.win_prob_episode_ids.fill_(-1)
        self.win_prob_agent_ids.fill_(-1)

    def _reset_split_value_rollout_buffers(self):
        self.terminal_values.zero_()
        self.shaped_values.zero_()
        self.terminal_reward_components.zero_()
        self.shaped_reward_components.zero_()

    def _prepare_trainer_shaped_reward_anneal(self) -> float:
        absolute_update = int(self.epoch) + 1
        multiplier = trainer_shaped_reward_multiplier(
            absolute_update,
            enabled=self._trainer_shaped_reward_anneal_enabled,
            start_epoch=self._trainer_shaped_reward_start_epoch,
            end_epoch=self._trainer_shaped_reward_end_epoch,
        )
        self._trainer_shaped_reward_update = absolute_update
        self._trainer_shaped_reward_multiplier = multiplier
        self.stats['trainer_shaped_reward_multiplier'].append(multiplier)
        self.stats['trainer_shaped_reward_update'].append(float(absolute_update))
        return multiplier

    def _record_effective_reward_shaping_scale(self):
        native_scales = self.stats.get('reward_shaping_scale')
        if not native_scales:
            return
        numeric = [float(value) for value in native_scales if np.isfinite(float(value))]
        if not numeric:
            return
        self.stats['effective_reward_shaping_scale'].append(
            float(np.mean(numeric)) * self._trainer_shaped_reward_multiplier
        )

    def trainer_shaped_reward_schedule_state(self) -> dict[str, object]:
        return {
            'enabled': bool(self._trainer_shaped_reward_anneal_enabled),
            'start_epoch': int(self._trainer_shaped_reward_start_epoch),
            'end_epoch': int(self._trainer_shaped_reward_end_epoch),
            'absolute_update': int(self._trainer_shaped_reward_update),
            'multiplier': float(self._trainer_shaped_reward_multiplier),
        }

    def _build_info_by_env(self, info, env_indices: np.ndarray) -> dict[int, object]:
        ordered_envs = np.unique(env_indices)
        info_entries = list(info) if isinstance(info, (list, tuple)) else []
        if len(info_entries) == len(ordered_envs):
            return {
                int(env_idx): info_entries[pos]
                for pos, env_idx in enumerate(ordered_envs)
            }
        if len(info_entries) == len(env_indices):
            info_by_env: dict[int, object] = {}
            for pos, env_idx in enumerate(env_indices):
                info_by_env.setdefault(int(env_idx), info_entries[pos])
            return info_by_env
        return {}

    def _extract_agent_info(self, env_info, seat: int):
        if not isinstance(env_info, dict):
            return None
        seat_info = env_info.get(seat)
        if seat_info is None:
            seat_info = env_info.get(str(seat))
        return seat_info if isinstance(seat_info, dict) else None

    def _extract_step_reward_components(
        self,
        info,
        env_id: np.ndarray,
        total_rewards: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if env_id.size == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        env_indices = (env_id // self._agents_per_env).astype(np.int32)
        info_by_env = self._build_info_by_env(info, env_indices)
        terminal = np.zeros(env_id.shape[0], dtype=np.float32)
        shaped = np.asarray(total_rewards, dtype=np.float32).copy()

        for row_index, agent_id in enumerate(env_id):
            env_idx = int(env_indices[row_index])
            seat = int(agent_id % self._agents_per_env)
            seat_info = self._extract_agent_info(info_by_env.get(env_idx), seat)
            if seat_info is None:
                continue

            terminal_value = seat_info.get('azk_step_terminal_reward', 0.0)
            shaped_value = seat_info.get('azk_step_shaped_reward')
            terminal[row_index] = float(np.clip(float(terminal_value), -1.0, 1.0))
            if shaped_value is not None:
                shaped[row_index] = float(np.clip(float(shaped_value), -1.0, 1.0))
            else:
                shaped[row_index] = float(total_rewards[row_index]) - terminal[row_index]

        return terminal, shaped

    def _component_values_from_state(self, state: dict, shape: torch.Size | tuple[int, ...]):
        terminal_value = state.get('_azk_value_terminal')
        shaped_value = state.get('_azk_value_shaped')
        if not torch.is_tensor(terminal_value) or not torch.is_tensor(shaped_value):
            raise RuntimeError('split_value_heads_enabled is true but component value tensors were not published by the policy')
        return terminal_value.view(shape), shaped_value.view(shape)

    def _clipped_value_loss(self, new_value, old_value, returns, vf_clip: float):
        v_clipped = old_value + torch.clamp(new_value - old_value, -vf_clip, vf_clip)
        v_loss_unclipped = (new_value - returns) ** 2
        v_loss_clipped = (v_clipped - returns) ** 2
        return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

    def _episode_envs_from_done_mask(self, env_id: np.ndarray, done_mask: np.ndarray) -> np.ndarray:
        if env_id.size == 0:
            return np.zeros((0,), dtype=np.int32)
        env_indices = (env_id // self._agents_per_env).astype(np.int32)
        finished_envs: list[int] = []
        for env_idx in np.unique(env_indices):
            selector = env_indices == env_idx
            if selector.any() and bool(done_mask[selector].all()):
                finished_envs.append(int(env_idx))
        return np.asarray(finished_envs, dtype=np.int32)

    def _stamp_win_prob_rollout_metadata(self, batch_rows, step_index: int, env_id: np.ndarray):
        if env_id.size == 0:
            return
        env_indices = (env_id // self._agents_per_env).astype(np.int32)
        episode_ids = self._env_episode_ids[env_indices]
        device = self.config['device']
        self.win_prob_episode_ids[batch_rows, step_index] = torch.as_tensor(
            episode_ids,
            device=device,
            dtype=torch.int64,
        )
        self.win_prob_agent_ids[batch_rows, step_index] = torch.as_tensor(
            env_id,
            device=device,
            dtype=torch.int32,
        )

    def _extract_terminal_win_labels(self, env_info) -> dict[int, float]:
        if not isinstance(env_info, dict):
            return {}

        labels: dict[int, float] = {}
        for seat in range(self._agents_per_env):
            seat_info = env_info.get(seat)
            if seat_info is None:
                seat_info = env_info.get(str(seat))
            if not isinstance(seat_info, dict):
                continue
            win_value = seat_info.get('win')
            if win_value is None:
                continue
            labels[int(seat)] = float(win_value)
        return labels

    def _assign_terminal_win_prob_targets(
        self,
        info,
        env_id: np.ndarray,
        done_mask: np.ndarray,
        *,
        terminal_rewards: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
    ):
        finished_envs = self._episode_envs_from_done_mask(env_id, done_mask)
        if finished_envs.size == 0:
            return finished_envs

        env_indices = (env_id // self._agents_per_env).astype(np.int32)
        info_by_env = self._build_info_by_env(info, env_indices)
        direct_labels: dict[int, dict[int, float]] = {}
        if terminal_rewards is not None:
            direct_labels = terminal_win_labels_from_rewards(
                env_id,
                done_mask if label_mask is None else label_mask,
                terminal_rewards,
                self._agents_per_env,
            )

        seat_ids = torch.remainder(self.win_prob_agent_ids.long(), self._agents_per_env)
        for env_idx in finished_envs:
            episode_id = int(self._env_episode_ids[int(env_idx)])
            episode_mask = self.win_prob_episode_ids == episode_id
            if bool(episode_mask.any().item()):
                if terminal_rewards is not None:
                    seat_labels = direct_labels.get(int(env_idx), {})
                else:
                    seat_labels = self._extract_terminal_win_labels(
                        info_by_env.get(int(env_idx))
                    )
                for seat, win_value in seat_labels.items():
                    seat_mask = episode_mask & (seat_ids == int(seat))
                    if not bool(seat_mask.any().item()):
                        continue
                    self.win_prob_targets[seat_mask] = float(win_value)
                    self.win_prob_target_mask[seat_mask] = True

            self._env_episode_ids[int(env_idx)] = self._next_env_episode_id
            self._next_env_episode_id += 1

        return finished_envs

    def _compute_win_prob_aux(self, state: dict, idx: torch.Tensor):
        """Masked win-prob BCE with metrics kept as device tensors (no syncs)."""
        device = self.config['device']
        zero = torch.zeros((), device=device)
        metrics = {
            'enabled': False,
            'raw_loss': zero,
            'labeled_frac': zero,
            'example_count': zero,
            'correct_sum': zero,
            'brier_sum': zero,
            'pred_sum': zero,
            'target_sum': zero,
        }
        if not self._win_prob_aux_enabled():
            return zero, metrics

        metrics['enabled'] = True
        win_prob_logits = state.get('_azk_win_prob_logits')
        if not torch.is_tensor(win_prob_logits):
            return zero, metrics

        targets = self.win_prob_targets[idx].to(device=win_prob_logits.device, dtype=win_prob_logits.dtype)
        labeled = self.win_prob_target_mask[idx].to(device=win_prob_logits.device, dtype=win_prob_logits.dtype)
        if win_prob_logits.shape != targets.shape:
            win_prob_logits = win_prob_logits.view_as(targets)

        count = labeled.sum()
        denom = count.clamp(min=1.0)
        per_element = F.binary_cross_entropy_with_logits(win_prob_logits, targets, reduction='none')
        raw_loss = (per_element * labeled).sum() / denom

        with torch.no_grad():
            probs = torch.sigmoid(win_prob_logits)
            predictions = (probs >= 0.5).to(dtype=targets.dtype)
            metrics['labeled_frac'] = labeled.mean()
            metrics['example_count'] = count
            metrics['raw_loss'] = raw_loss.detach()
            metrics['correct_sum'] = ((predictions == targets).to(dtype=labeled.dtype) * labeled).sum()
            metrics['brier_sum'] = (torch.square(probs - targets) * labeled).sum()
            metrics['pred_sum'] = (probs * labeled).sum()
            metrics['target_sum'] = (targets * labeled).sum()

        weighted_loss = raw_loss * self._win_prob_aux_coef()
        return weighted_loss, metrics

    def _draftaux_aux_scale(self):
        """Scale draft auxiliary credit with both active shaping schedules."""
        native_scale = 1.0
        if self._draftaux_anneal:
            vals = self.stats.get('reward_shaping_scale')
            if vals:
                self._draftaux_last_scale = float(vals[-1])
            native_scale = self._draftaux_last_scale
        return native_scale * self._trainer_shaped_reward_multiplier

    def _draftaux_init_layout(self, obs_row_bytes, device):
        """Byte offsets into the packed deckbuild obs + sibling-gate lookup.

        Disables the aux (with a message) if the obs layout is not the native
        deck-building struct."""
        import numpy as np

        try:
            from observation import DECKBUILD_OBSERVATION_CTYPE
            dt = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
            if int(obs_row_bytes) != dt.itemsize:
                raise ValueError(
                    f"obs row {obs_row_bytes}B != deckbuild struct {dt.itemsize}B"
                )

            def field_offset(dtype, path):
                off = 0
                for name in path:
                    sub, sub_off = dtype.fields[name][0], dtype.fields[name][1]
                    off += sub_off
                    dtype = sub
                return off

            layout = {
                "mode_off": field_offset(dt, ("deck_context", "mode")),
                "gate_ctx_off": field_offset(dt, ("deck_context", "gate_card_def_id")),
                "gate_zone_off": field_offset(dt, ("my_observation_data", "gate", "card_def_id")),
            }
            from deck_building import build_deck_build_catalog
            from training_deck_pool import load_training_deck_pool

            catalog = build_deck_build_catalog(load_training_deck_pool())
            records = catalog.records_by_def_id
            gate_ids = sorted({int(g) for g in catalog.gate_def_id_population})
            by_element = {}
            for g in gate_ids:
                by_element.setdefault(records[g].element, []).append(g)
            vocab = max(records) + 1
            sibling = torch.full((vocab,), -1, dtype=torch.int64, device=device)
            for element_gates in by_element.values():
                if len(element_gates) < 2:
                    continue
                for i, g in enumerate(element_gates):
                    sibling[g] = element_gates[(i + 1) % len(element_gates)]
            layout["sibling"] = sibling
            self._draftaux_layout = layout
            print(
                f"[draftaux] enabled: vboot={self._draftaux_vboot} "
                f"sibdiff={self._draftaux_sibdiff} cap={self._draftaux_cap}"
            )
        except Exception as exc:
            print(f"[draftaux] disabled (layout init failed: {exc})")
            self._draftaux_enabled = False
        return self._draftaux_layout

    def _draftaux_step(self, o_device, value, batch_rows, l, env_id, mask, h_prev, c_prev):
        """Detect draft->battle boundaries and inject aux credit at the last pick."""
        lay = self._draftaux_layout
        if lay is None:
            lay = self._draftaux_init_layout(o_device.shape[-1], o_device.device)
            if lay is None:
                return
        group = env_id.start
        mo = lay["mode_off"]
        mode = o_device[:, mo:mo + 4].contiguous().view(torch.int32).flatten()
        prev = self._draftaux_prev.get(group)
        self._draftaux_prev[group] = {"mode": mode.clone(), "rows": batch_rows, "l": l}
        if prev is None:
            return
        boundary = (mode == 0) & (prev["mode"] != 0)
        if not bool(boundary.any()):
            return
        b = boundary.nonzero(as_tuple=False).flatten()
        v_own = value.flatten()[b].detach().float()
        aux = self._draftaux_vboot * v_own
        if self._draftaux_sibdiff > 0.0:
            g_off = lay["gate_ctx_off"]
            z_off = lay["gate_zone_off"]
            rows = o_device[b].clone()
            gid = rows[:, g_off:g_off + 2].contiguous().view(torch.int16).flatten().long()
            sib = lay["sibling"][gid.clamp(min=0, max=lay["sibling"].numel() - 1)]
            ok = (gid >= 0) & (sib >= 0)
            if bool(ok.any()):
                sb_bytes = sib.to(torch.int16).view(torch.uint8).reshape(-1, 2)
                rows[:, g_off:g_off + 2] = sb_bytes
                rows[:, z_off:z_off + 2] = sb_bytes
                mask_t = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
                cf_state = {
                    "mask": mask_t.to(o_device.device)[b],
                    "lstm_h": h_prev[b].clone(),
                    "lstm_c": c_prev[b].clone(),
                }
                with torch.no_grad(), self.amp_context:
                    _, v_sib = self.policy.forward_eval(rows, cf_state)
                diff = (v_own - v_sib.flatten().detach().float()).clamp(
                    min=0.0, max=self._draftaux_cap
                )
                aux = aux + self._draftaux_sibdiff * diff * ok.float()
        # Inject at the previous step's stored slot (the last pick) — cached
        # coords survive segment rollover because they are the written coords.
        prev_rows, prev_l = prev["rows"], prev["l"]
        idx = torch.arange(prev_rows.start, prev_rows.stop, device=value.device)[b]
        aux = aux * self._draftaux_aux_scale()
        aux_cast = aux.to(self.rewards.dtype)
        self.rewards[idx, prev_l] += aux_cast
        self.shaped_reward_components[idx, prev_l] += aux_cast
        self._draftaux_injected += float(aux.sum().item())
        self._draftaux_events += int(b.numel())

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']
        reward_multiplier = self._prepare_trainer_shaped_reward_anneal()

        if config['use_rnn']:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        if self._draftaux_enabled:
            # Prev-step buffer coords from the last epoch are recycled slots;
            # never inject across the epoch boundary.
            self._draftaux_prev.clear()
            if self._draftaux_events:
                self.stats['draftaux_events'].append(float(self._draftaux_events))
                self.stats['draftaux_injected_mean'].append(
                    self._draftaux_injected / max(self._draftaux_events, 1)
                )
            self._draftaux_events = 0
            self._draftaux_injected = 0.0

        self.full_rows = 0
        # Python mirrors of ep_lengths/ep_indices for the slice reads below:
        # reading the device tensors would sync the CPU against the whole
        # enqueued forward on every step.
        self._ep_len_py = {}
        self._ep_row_py = {}
        self._reset_win_prob_rollout_buffers()
        self._reset_split_value_rollout_buffers()
        win_prob_enabled = self._win_prob_aux_enabled()
        if win_prob_enabled:
            # Per-agent episode bookkeeping, all device-resident and rebased
            # each epoch: ids advance by total_agents on episode end so they
            # stay unique per agent-episode; the win table is keyed by them.
            self.agent_episode_ids = torch.arange(
                self.total_agents, device=device, dtype=torch.int64)
            # Each agent's id advances by total_agents per episode end; an
            # agent sees at most ceil(batch/agents_per_batch) recv rounds.
            agents_per_batch = max(1, int(getattr(self.vecenv, 'agents_per_batch', self.total_agents)))
            max_rounds = -(-int(config['batch_size']) // agents_per_batch)
            win_table_size = self.total_agents * (max_rounds + 2)
            self.episode_win_table = torch.full((win_table_size,), -1.0, device=device)

        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id_np = np.asarray(env_id, dtype=np.int64)
            env_id = slice(int(env_id_np[0]), int(env_id_np[-1]) + 1)
            self.global_step += int(mask.sum())

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            if self._rollout_obs_device is None:
                o_device = o.to(device, non_blocking=True)
            else:
                self._rollout_obs_device.copy_(o, non_blocking=False)
                o_device = self._rollout_obs_device
            r = torch.as_tensor(r).to(device, non_blocking=True)
            d = torch.as_tensor(d).to(device, non_blocking=True)
            t_dev = torch.as_tensor(t).to(device, non_blocking=True)

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state['lstm_h'] = self.lstm_h[env_id.start]
                    state['lstm_c'] = self.lstm_c[env_id.start]
                draftaux_h_prev = self.lstm_h[env_id.start] if self._draftaux_enabled else None
                draftaux_c_prev = self.lstm_c[env_id.start] if self._draftaux_enabled else None

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                # Reward components derive from (reward, done): the env
                # guarantees rewards carry only the terminal component on
                # episode-end steps and only the shaped component otherwise.
                done_dev = d | t_dev
                r_clipped = torch.clamp(r, -1, 1)
                zeros = torch.zeros((), device=device, dtype=r_clipped.dtype)
                reward_components_terminal = torch.where(done_dev, r_clipped, zeros)
                reward_components_shaped = torch.where(done_dev, zeros, r_clipped)

                # Fast path for fully vectorized envs
                group = env_id.start
                l = self._ep_len_py.get(group, 0)
                row_start = self._ep_row_py.get(group, group)
                batch_rows = slice(row_start, row_start + (env_id.stop - env_id.start))

                if config['cpu_offload']:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                scaled_total, scaled_shaped = recombine_reward_components(
                    r_clipped,
                    reward_components_terminal,
                    reward_components_shaped,
                    reward_multiplier,
                )
                self.rewards[batch_rows, l] = scaled_total
                self.terminal_reward_components[batch_rows, l] = reward_components_terminal
                self.shaped_reward_components[batch_rows, l] = scaled_shaped
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()
                if self._draftaux_enabled:
                    self._draftaux_step(
                        o_device, value, batch_rows, l, env_id, mask,
                        draftaux_h_prev, draftaux_c_prev,
                    )
                if self._split_value_heads_enabled():
                    terminal_value, shaped_value = self._component_values_from_state(
                        state,
                        self.values[batch_rows, l].shape,
                    )
                    self.terminal_values[batch_rows, l] = terminal_value.detach().float()
                    self.shaped_values[batch_rows, l] = shaped_value.detach().float()

                if win_prob_enabled:
                    # Stamp episode ids for these rows and record win labels on
                    # episode end (win := terminal end with positive terminal
                    # reward; truncations label 0 — matching the env's
                    # winner-based per-episode stats).
                    ids = self.agent_episode_ids[env_id]
                    self.win_prob_episode_ids[batch_rows, l] = ids
                    win_now = (d & (r > 0)).float()
                    prev = self.episode_win_table[ids]
                    self.episode_win_table[ids] = torch.where(done_dev, win_now, prev)
                    self.agent_episode_ids[env_id] = ids + done_dev.long() * self.total_agents

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                self._ep_len_py[group] = l + 1
                if l+1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
                    self.ep_lengths[env_id] = 0
                    self._ep_row_py[group] = self.free_idx
                    self._ep_len_py[group] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile('env', epoch)
            self.vecenv.send(action)

        profile('eval_misc', epoch)
        if win_prob_enabled:
            # Resolve per-row win targets from the episode table in one gather.
            row_ids = self.win_prob_episode_ids.clamp(min=0)
            wins = self.episode_win_table[row_ids]
            labeled = (self.win_prob_episode_ids >= 0) & (wins >= 0)
            self.win_prob_targets = torch.where(labeled, wins, torch.zeros_like(wins))
            self.win_prob_target_mask = labeled

        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        self._record_effective_reward_shaping_scale()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        profile('train_misc', epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        device = config['device']
        win_prob_enabled = self._win_prob_aux_enabled()
        split_value_enabled = self._split_value_heads_enabled()
        win_prob_correct_sum = 0.0
        win_prob_brier_sum = 0.0
        win_prob_pred_sum = 0.0
        win_prob_target_sum = 0.0
        win_prob_example_count = 0

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile('train_misc', epoch)

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(self.values, self.rewards,
                self.terminals, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
            terminal_advantages = None
            shaped_advantages = None
            if split_value_enabled:
                terminal_advantages = torch.zeros(shape, device=device)
                terminal_advantages = compute_puff_advantage(
                    self.terminal_values,
                    self.terminal_reward_components,
                    self.terminals,
                    self.ratio,
                    terminal_advantages,
                    config['gamma'],
                    config['gae_lambda'],
                    config['vtrace_rho_clip'],
                    config['vtrace_c_clip'],
                )
                shaped_advantages = torch.zeros(shape, device=device)
                shaped_advantages = compute_puff_advantage(
                    self.shaped_values,
                    self.shaped_reward_components,
                    self.terminals,
                    self.ratio,
                    shaped_advantages,
                    config['gamma'],
                    config['gae_lambda'],
                    config['vtrace_rho_clip'],
                    config['vtrace_c_clip'],
                )

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta

            profile('train_copy', epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]
            if split_value_enabled:
                mb_terminal_values = self.terminal_values[idx]
                mb_shaped_values = self.shaped_values[idx]
                mb_terminal_returns = terminal_advantages[idx] + mb_terminal_values
                mb_shaped_returns = shaped_advantages[idx] + mb_shaped_values

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            with self.amp_context:
                logits, newvalue = self.policy(mb_obs, state)
                actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            # NOTE: Commenting this out since adv is replaced below
            # adv = advantages[idx]
            # adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
            #     ratio, adv, config['gamma'], config['gae_lambda'],
            #     config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Weight advantages by priority and normalize
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            total_v_loss = self._clipped_value_loss(newvalue, mb_values, mb_returns, vf_clip)
            component_v_loss = torch.zeros((), device=device)
            terminal_v_loss = torch.zeros((), device=device)
            shaped_v_loss = torch.zeros((), device=device)
            if split_value_enabled:
                new_terminal_value, new_shaped_value = self._component_values_from_state(
                    state,
                    mb_returns.shape,
                )
                terminal_v_loss = self._clipped_value_loss(
                    new_terminal_value,
                    mb_terminal_values,
                    mb_terminal_returns,
                    vf_clip,
                )
                shaped_v_loss = self._clipped_value_loss(
                    new_shaped_value,
                    mb_shaped_values,
                    mb_shaped_returns,
                    vf_clip,
                )
                component_v_loss = 0.5 * (terminal_v_loss + shaped_v_loss)

            entropy_loss = entropy.mean()
            win_prob_aux_loss, win_prob_aux_metrics = self._compute_win_prob_aux(state, idx)

            value_loss_for_optim = total_v_loss + self._split_value_component_coef() * component_v_loss
            loss = pg_loss + config['vf_coef']*value_loss_for_optim - config['ent_coef']*entropy_loss + win_prob_aux_loss

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()
            if split_value_enabled:
                self.terminal_values[idx] = new_terminal_value.detach().float()
                self.shaped_values[idx] = new_shaped_value.detach().float()

            # Logging: accumulate as device tensors; a single sync happens at
            # the end of the epoch when the dashboard converts to floats.
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.detach() / self.total_minibatches
            losses['value_loss'] += value_loss_for_optim.detach() / self.total_minibatches
            losses['value_loss_total'] += total_v_loss.detach() / self.total_minibatches
            if split_value_enabled:
                losses['value_loss_terminal'] += terminal_v_loss.detach() / self.total_minibatches
                losses['value_loss_shaped'] += shaped_v_loss.detach() / self.total_minibatches
            losses['entropy'] += entropy_loss.detach() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl / self.total_minibatches
            losses['approx_kl'] += approx_kl / self.total_minibatches
            losses['clipfrac'] += clipfrac / self.total_minibatches
            losses['importance'] += ratio.detach().mean() / self.total_minibatches
            if win_prob_enabled:
                losses['win_prob_aux_loss'] += win_prob_aux_metrics['raw_loss'] / self.total_minibatches
                losses['win_prob_aux_labeled_frac'] += win_prob_aux_metrics['labeled_frac'] / self.total_minibatches
                win_prob_correct_sum += win_prob_aux_metrics['correct_sum']
                win_prob_brier_sum += win_prob_aux_metrics['brier_sum']
                win_prob_pred_sum += win_prob_aux_metrics['pred_sum']
                win_prob_target_sum += win_prob_aux_metrics['target_sum']
                win_prob_example_count += win_prob_aux_metrics['example_count']

            # Learn on accumulated minibatches
            profile('learn_backward', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                profile('learn_opt', epoch)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var
        if split_value_enabled:
            terminal_y_pred = self.terminal_values.flatten()
            terminal_y_true = terminal_advantages.flatten() + self.terminal_values.flatten()
            terminal_var_y = terminal_y_true.var()
            losses['explained_variance_terminal'] = (
                torch.nan
                if terminal_var_y == 0
                else (1 - (terminal_y_true - terminal_y_pred).var() / terminal_var_y).item()
            )
            shaped_y_pred = self.shaped_values.flatten()
            shaped_y_true = shaped_advantages.flatten() + self.shaped_values.flatten()
            shaped_var_y = shaped_y_true.var()
            losses['explained_variance_shaped'] = (
                torch.nan
                if shaped_var_y == 0
                else (1 - (shaped_y_true - shaped_y_pred).var() / shaped_var_y).item()
            )
        if win_prob_enabled:
            denom = win_prob_example_count
            if torch.is_tensor(denom):
                denom = denom.clamp(min=1.0)
            else:
                denom = max(denom, 1.0)
            losses['win_prob_aux_accuracy'] = win_prob_correct_sum / denom
            losses['win_prob_aux_brier'] = win_prob_brier_sum / denom
            losses['win_prob_aux_pred_mean'] = win_prob_pred_sum / denom
            losses['win_prob_aux_target_mean'] = win_prob_target_sum / denom

        # Single host sync for all logged scalars.
        losses = {k: (float(v.item()) if torch.is_tensor(v) else v) for k, v in losses.items()}

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            'SPS': dist_sum(self.sps, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               self.logger.log(logs, agent_steps)
               return logs
           else:
               return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[dim default]', b1='[bright_cyan]', b2='[default]'):
        config = self.config
        sps = dist_sum(self.sps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        version = str(getattr(upstream_pufferlib, '__version__', '4.0.0')).split('+', 1)[0]
        table.add_row(
            f'{b1}PufferLib {b2}{version} {idx[0]*" "}:blowfish:',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{b2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{b2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{b2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{b2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{b2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{b2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{b2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf('  Env', b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf('  Backward', b2, delta, profile.learn_backward, b2, c2))
        p.add_row(*fmt_perf('  Optimizer', b2, delta, profile.learn_opt, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

try:
    from pufferlib.torch_pufferl import compute_puff_advantage as upstream_compute_puff_advantage
except Exception:
    upstream_compute_puff_advantage = None


def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''Use the upstream 4.0 kernel when available and fall back to a Python implementation.'''

    if upstream_compute_puff_advantage is not None:
        return upstream_compute_puff_advantage(
            values,
            rewards,
            terminals,
            ratio,
            advantages,
            gamma,
            gae_lambda,
            vtrace_rho_clip,
            vtrace_c_clip,
        )

    next_advantage = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
    next_value = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
    clipped_rho = torch.clamp(ratio, max=vtrace_rho_clip)
    clipped_c = torch.clamp(ratio, max=vtrace_c_clip)

    for step in range(values.shape[1] - 1, -1, -1):
        nonterminal = 1.0 - terminals[:, step]
        delta = clipped_rho[:, step] * (
            rewards[:, step] + gamma * nonterminal * next_value - values[:, step]
        )
        next_advantage = delta + gamma * gae_lambda * nonterminal * clipped_c[:, step] * next_advantage
        advantages[:, step] = next_advantage
        next_value = values[:, step]

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []
        self.pending = []
        self.cuda_timing = torch.cuda.is_available()

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def _boundary(self):
        tick = time.perf_counter()
        event = None
        if self.cuda_timing:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return tick, event

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        tick, event = self._boundary()
        if len(self.stack) != 0 and not nest:
            self.pop(tick, event)

        self.stack.append((name, tick, event))

    def pop(self, end_tick, end_event):
        name, start_tick, start_event = self.stack.pop()
        self.pending.append(
            (name, end_tick - start_tick, start_event, end_event)
        )

    def end(self):
        if not self.stack:
            return

        end_tick, end_event = self._boundary()
        while self.stack:
            self.pop(end_tick, end_event)
        if end_event is not None:
            end_event.synchronize()

        for name, cpu_delta, start_event, segment_end_event in self.pending:
            delta = cpu_delta
            if start_event is not None and segment_end_event is not None:
                gpu_delta = start_event.elapsed_time(segment_end_event) / 1000.0
                delta = max(cpu_delta, gpu_delta)
            profile = self.profiles[name]
            profile['delta'] += delta
            # Multiply delta by freq to account for skipped epochs
            profile['elapsed'] += delta * self.frequency
        self.pending.clear()

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self._reported_gpu_metrics_error = False
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                   time.sleep(self.delay)
                   continue

                try:
                    self.gpu_util.append(torch.cuda.utilization())
                    free, total = torch.cuda.mem_get_info()
                    self.gpu_mem.append(100*(total-free)/total)
                except Exception:
                    self._reported_gpu_metrics_error = True
                    self.gpu_util.append(0)
                    self.gpu_mem.append(0)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100*time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
        pass

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune['model'].track_files(model_path)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'
 
class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
        import wandb
        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args['wandb_project'],
            group=args['wandb_group'],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'

def train(env_name, args=None, vecenv=None, policy=None, logger=None, should_stop_early=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = { **args['train'], 'env': env_name }
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    all_logs = []
    while pufferl.global_step < train_config['total_timesteps']:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20*train_config['total_timesteps']:
                all_logs.append(logs)

            if should_stop_early is not None and should_stop_early(logs):
                model_path = pufferl.close()
                pufferl.logger.close(model_path)
                return all_logs

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        stats = pufferl.evaluate()
        if i >= 32 and stats:
            break

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs

def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')

def stop_if_loss_nan(logs):
    return any("losses/" in k and np.isnan(v) for k, v in logs.items())

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')

    import azk_puffer.sweep as azk_sweep

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(azk_sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sweep.suggest(args)
        all_logs = train(env_name, args=args, should_stop_early=stop_if_loss_nan)
        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        if len(timesteps) > 0 and timesteps[-1] < 0.7 * total_timesteps:  # 0.7 is arbitrary
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace("trace.json")

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f'Saved {len(weights)} weights to {path}')

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['rnn'])

    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    return process_config(p, parser=parser)

def load_config_file(file_path, fill_in_default=True, parser=None):
    if not os.path.exists(file_path):
        raise pufferlib.APIUsageError('No config file found')

    config_paths = [file_path]

    if fill_in_default:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        # Process the puffer defaults first
        config_paths.insert(0, os.path.join(puffer_dir, 'config/default.ini'))

    p = configparser.ConfigParser()
    p.read(config_paths)

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb or neptune')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{upstream_pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=auto_type if value == 'auto' else type(value)
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == 'train':
        train(env_name=env_name)
    elif mode == 'eval':
        eval(env_name=env_name)
    elif mode == 'sweep':
        sweep(env_name=env_name)
    elif mode == 'autotune':
        autotune(env_name=env_name)
    elif mode == 'profile':
        profile(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
