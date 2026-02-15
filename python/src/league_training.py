from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

import pufferlib
import pufferlib.pytorch
from pufferlib import pufferl


@dataclass
class LeagueConfig:
  enabled: bool = False
  latest_ratio: float = 0.85
  randomize_learner_seat: bool = True
  seed: int = 0


def compute_learner_row_mask(
  env_ids: np.ndarray,
  *,
  agents_per_env: int,
  env_learner_seat: np.ndarray,
) -> np.ndarray:
  env_indices = (env_ids // agents_per_env).astype(np.int32)
  seat_indices = (env_ids % agents_per_env).astype(np.int32)
  return seat_indices == env_learner_seat[env_indices]


class LeaguePuffeRL(pufferl.PuffeRL):
  """PuffeRL variant with dual-policy league rollouts and learner-only updates."""

  def __init__(self, config, vecenv, policy, opponent_policies, league_cfg: LeagueConfig, logger=None):
    super().__init__(config, vecenv, policy, logger=logger)
    self.league_cfg = league_cfg
    self.opponent_policies = list(opponent_policies)
    self._rng = np.random.default_rng(int(league_cfg.seed))

    self._agents_per_env = int(vecenv.driver_env.num_agents)
    if self._agents_per_env <= 1:
      raise ValueError("League mode requires at least 2 agents per env")

    if hasattr(vecenv, "num_environments"):
      self._num_envs_total = int(vecenv.num_environments)
    elif hasattr(vecenv, "envs"):
      self._num_envs_total = int(len(vecenv.envs))
    else:
      self._num_envs_total = int(self.total_agents // self._agents_per_env)

    self._env_learner_seat = self._rng.integers(
      0, self._agents_per_env, size=self._num_envs_total, dtype=np.int32
    )
    self._env_opp_policy = np.zeros(self._num_envs_total, dtype=np.int32)
    self._env_use_latest = np.ones(self._num_envs_total, dtype=np.bool_)
    self._resample_matchups(np.arange(self._num_envs_total, dtype=np.int32))

    self._segment_is_trainable = torch.zeros(self.segments, device=self.config["device"], dtype=torch.bool)

    self._use_rnn = bool(self.config.get("use_rnn", False))
    if self._use_rnn:
      hidden_size = int(policy.hidden_size)
      device = self.config["device"]
      self._learner_lstm_h = torch.zeros(self.total_agents, hidden_size, device=device)
      self._learner_lstm_c = torch.zeros(self.total_agents, hidden_size, device=device)
      self._opp_lstm_h = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
      self._opp_lstm_c = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
    else:
      self._learner_lstm_h = None
      self._learner_lstm_c = None
      self._opp_lstm_h = []
      self._opp_lstm_c = []

    for opp in self.opponent_policies:
      opp.eval()
      for param in opp.parameters():
        param.requires_grad_(False)

  def _resample_matchups(self, env_indices: np.ndarray) -> None:
    if env_indices.size == 0:
      return

    if self.league_cfg.randomize_learner_seat:
      self._env_learner_seat[env_indices] = self._rng.integers(
        0, self._agents_per_env, size=env_indices.size, dtype=np.int32
      )

    if len(self.opponent_policies) == 0:
      self._env_use_latest[env_indices] = True
      self._env_opp_policy[env_indices] = -1
      return

    latest_draw = self._rng.random(env_indices.size)
    self._env_use_latest[env_indices] = latest_draw < float(self.league_cfg.latest_ratio)
    self._env_opp_policy[env_indices] = self._rng.integers(
      0, len(self.opponent_policies), size=env_indices.size, dtype=np.int32
    )

  def set_opponent_policies(self, opponent_policies: list[torch.nn.Module]) -> None:
    self.opponent_policies = list(opponent_policies)
    for opp in self.opponent_policies:
      opp.eval()
      for param in opp.parameters():
        param.requires_grad_(False)
    self._opp_lstm_h = []
    self._opp_lstm_c = []
    if self._use_rnn:
      hidden_size = int(self.policy.hidden_size)
      device = self.config["device"]
      self._opp_lstm_h = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
      self._opp_lstm_c = [
        torch.zeros(self.total_agents, hidden_size, device=device) for _ in self.opponent_policies
      ]
    self._resample_matchups(np.arange(self._num_envs_total, dtype=np.int32))

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

  def _zero_done_states(self, done_rows: np.ndarray) -> None:
    if done_rows.size == 0 or not self._use_rnn:
      return
    done_t = torch.as_tensor(done_rows, device=self.config["device"], dtype=torch.bool)
    self._learner_lstm_h[done_t] = 0
    self._learner_lstm_c[done_t] = 0
    for idx in range(len(self.opponent_policies)):
      self._opp_lstm_h[idx][done_t] = 0
      self._opp_lstm_c[idx][done_t] = 0

  def _infer_actions(self, o_device: torch.Tensor, mask_t: torch.Tensor, env_id_np: np.ndarray):
    device = self.config["device"]
    batch_n = o_device.shape[0]

    env_indices = (env_id_np // self._agents_per_env).astype(np.int32)
    learner_rows_np = compute_learner_row_mask(
      env_id_np,
      agents_per_env=self._agents_per_env,
      env_learner_seat=self._env_learner_seat,
    )

    actions_out = torch.zeros((batch_n, *self.vecenv.single_action_space.shape), device=device, dtype=torch.int32)
    logprobs_out = torch.zeros(batch_n, device=device)
    values_out = torch.zeros(batch_n, device=device)

    learner_idx = np.nonzero(learner_rows_np)[0]
    if learner_idx.size > 0:
      learner_idx_t = torch.as_tensor(learner_idx, device=device, dtype=torch.long)
      learner_state = {"mask": mask_t[learner_idx_t]}
      if self._use_rnn:
        learner_state["lstm_h"] = self._learner_lstm_h[learner_idx_t]
        learner_state["lstm_c"] = self._learner_lstm_c[learner_idx_t]
      logits, values = self._safe_forward_eval(self.policy, o_device[learner_idx_t], learner_state)
      with torch.no_grad(), self.amp_context:
        actions, logprobs, _ = pufferlib.pytorch.sample_logits(logits)
      actions_out[learner_idx_t] = actions.to(dtype=torch.int32)
      logprobs_out[learner_idx_t] = logprobs.to(dtype=logprobs_out.dtype)
      values_out[learner_idx_t] = values.flatten().to(dtype=values_out.dtype)
      if self._use_rnn:
        self._learner_lstm_h[learner_idx_t] = learner_state["lstm_h"]
        self._learner_lstm_c[learner_idx_t] = learner_state["lstm_c"]

    opp_rows_np = np.nonzero(~learner_rows_np)[0]
    if opp_rows_np.size > 0:
      opp_env_indices = env_indices[opp_rows_np]
      opp_use_latest = self._env_use_latest[opp_env_indices]
      latest_rows_np = opp_rows_np[opp_use_latest]
      frozen_rows_np = opp_rows_np[~opp_use_latest]

      if latest_rows_np.size > 0:
        latest_idx_t = torch.as_tensor(latest_rows_np, device=device, dtype=torch.long)
        latest_state = {"mask": mask_t[latest_idx_t]}
        if self._use_rnn:
          latest_state["lstm_h"] = self._learner_lstm_h[latest_idx_t]
          latest_state["lstm_c"] = self._learner_lstm_c[latest_idx_t]
        latest_logits, _ = self._safe_forward_eval(self.policy, o_device[latest_idx_t], latest_state)
        with torch.no_grad(), self.amp_context:
          latest_actions, _, _ = pufferlib.pytorch.sample_logits(latest_logits)
        actions_out[latest_idx_t] = latest_actions.to(dtype=torch.int32)
        if self._use_rnn:
          self._learner_lstm_h[latest_idx_t] = latest_state["lstm_h"]
          self._learner_lstm_c[latest_idx_t] = latest_state["lstm_c"]

      if frozen_rows_np.size > 0:
        frozen_env_indices = env_indices[frozen_rows_np]
        frozen_policy_ids = self._env_opp_policy[frozen_env_indices]
        for policy_id in np.unique(frozen_policy_ids):
          rows_np = frozen_rows_np[frozen_policy_ids == policy_id]
          if rows_np.size == 0:
            continue
          rows_t = torch.as_tensor(rows_np, device=device, dtype=torch.long)
          opp_policy = self.opponent_policies[int(policy_id)]
          opp_state = {"mask": mask_t[rows_t]}
          if self._use_rnn:
            opp_state["lstm_h"] = self._opp_lstm_h[int(policy_id)][rows_t]
            opp_state["lstm_c"] = self._opp_lstm_c[int(policy_id)][rows_t]
          opp_logits, _ = self._safe_forward_eval(opp_policy, o_device[rows_t], opp_state)
          with torch.no_grad(), self.amp_context:
            opp_actions, _, _ = pufferlib.pytorch.sample_logits(opp_logits)
          actions_out[rows_t] = opp_actions.to(dtype=torch.int32)
          if self._use_rnn:
            self._opp_lstm_h[int(policy_id)][rows_t] = opp_state["lstm_h"]
            self._opp_lstm_c[int(policy_id)][rows_t] = opp_state["lstm_c"]

    return actions_out, logprobs_out, values_out, learner_rows_np, env_indices

  def _safe_forward_eval(self, model, obs: torch.Tensor, state: dict):
    with torch.no_grad(), self.amp_context:
      if obs.shape[0] > 1:
        return model.forward_eval(obs, state)

      obs_pad = torch.cat([obs, obs], dim=0)
      state_pad = {"mask": torch.cat([state["mask"], state["mask"]], dim=0)}
      if self._use_rnn:
        state_pad["lstm_h"] = torch.cat([state["lstm_h"], state["lstm_h"]], dim=0)
        state_pad["lstm_c"] = torch.cat([state["lstm_c"], state["lstm_c"]], dim=0)
      logits, values = model.forward_eval(obs_pad, state_pad)
      if self._use_rnn:
        state["lstm_h"] = state_pad["lstm_h"][:1]
        state["lstm_c"] = state_pad["lstm_c"][:1]

      if hasattr(logits, "primary_logits"):
        logits = type(logits)(
          primary_logits=logits.primary_logits[:1],
          primary_action_mask=logits.primary_action_mask[:1],
          legal_actions=logits.legal_actions[:1],
          legal_action_count=logits.legal_action_count[:1],
          target_matrix=logits.target_matrix[:1],
          unit1_projection=logits.unit1_projection[:1],
          unit2_projection=logits.unit2_projection[:1],
          bins2_logits=logits.bins2_logits[:1],
          bins3_logits=logits.bins3_logits[:1],
          gate1_table=logits.gate1_table,
          gate2_table=logits.gate2_table,
        )
      return logits, values[:1]

  def evaluate(self):
    profile = self.profile
    epoch = self.epoch
    profile("eval", epoch)
    profile("eval_misc", epoch, nest=True)

    config = self.config
    device = config["device"]

    self.full_rows = 0
    self._segment_is_trainable.zero_()
    while self.full_rows < self.segments:
      profile("env", epoch)
      o, r, d, t, info, env_id, mask = self.vecenv.recv()

      profile("eval_misc", epoch)
      env_id_slice = slice(env_id[0], env_id[-1] + 1)
      done_mask = (d + t).astype(np.bool_)

      profile("eval_copy", epoch)
      o = torch.as_tensor(o)
      o_device = o.to(device)
      r_t = torch.as_tensor(r).to(device)
      d_t = torch.as_tensor(d).to(device)
      mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
      env_id_np = np.asarray(env_id, dtype=np.int64)

      profile("eval_forward", epoch)
      actions_t, logprobs_t, values_t, learner_rows_np, env_indices = self._infer_actions(
        o_device, mask_t, env_id_np
      )

      learner_step_mask = np.logical_and(mask.astype(np.bool_), learner_rows_np)
      self.global_step += int(learner_step_mask.sum())

      profile("eval_copy", epoch)
      with torch.no_grad():
        l = self.ep_lengths[env_id_slice.start].item()
        batch_rows = slice(
          self.ep_indices[env_id_slice.start].item(),
          1 + self.ep_indices[env_id_slice.stop - 1].item(),
        )

        if config["cpu_offload"]:
          self.observations[batch_rows, l] = o
        else:
          self.observations[batch_rows, l] = o_device

        self.actions[batch_rows, l] = actions_t
        self.logprobs[batch_rows, l] = logprobs_t
        self.rewards[batch_rows, l] = torch.clamp(r_t, -1, 1)
        self.terminals[batch_rows, l] = d_t.float()
        self.values[batch_rows, l] = values_t.float()
        self._segment_is_trainable[batch_rows] = torch.as_tensor(learner_rows_np, device=device, dtype=torch.bool)

        self.ep_lengths[env_id_slice] += 1
        if l + 1 >= config["bptt_horizon"]:
          num_full = env_id_slice.stop - env_id_slice.start
          self.ep_indices[env_id_slice] = self.free_idx + torch.arange(num_full, device=device).int()
          self.ep_lengths[env_id_slice] = 0
          self.free_idx += num_full
          self.full_rows += num_full

      done_rows = env_id_np[done_mask]
      self._zero_done_states(done_rows)
      finished_envs = self._episode_envs_from_done_mask(env_id_np, done_mask)
      self._resample_matchups(finished_envs)

      actions_np = actions_t.cpu().numpy()

      profile("eval_misc", epoch)
      for i in info:
        for k, v in pufferlib.unroll_nested_dict(i):
          if isinstance(v, np.ndarray):
            v = v.tolist()
          elif isinstance(v, (list, tuple)):
            self.stats[k].extend(v)
          else:
            self.stats[k].append(v)

      self.stats["league/learner_row_fraction"].append(float(np.mean(learner_rows_np)))
      self.stats["league/latest_opponent_fraction"].append(float(np.mean(self._env_use_latest[env_indices])))

      profile("env", epoch)
      self.vecenv.send(actions_np)

    # Match base PuffeRL buffer lifecycle: reset row indexing state after each
    # evaluate pass so the next epoch starts with fresh contiguous row slots.
    self.free_idx = self.total_agents
    self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
    self.ep_lengths.zero_()

    profile.end()
    return self.stats

  def train(self):
    profile = self.profile
    epoch = self.epoch
    profile("train", epoch)
    profile("train_misc", epoch, nest=True)
    losses = defaultdict(float)
    config = self.config
    device = config["device"]

    b0 = config["prio_beta0"]
    a = config["prio_alpha"]
    clip_coef = config["clip_coef"]
    vf_clip = config["vf_clip_coef"]
    anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
    self.ratio[:] = 1

    trainable_idx = torch.nonzero(self._segment_is_trainable, as_tuple=False).flatten()
    if trainable_idx.numel() == 0:
      trainable_idx = torch.arange(self.segments, device=device)

    for mb in range(self.total_minibatches):
      profile("train_misc", epoch)
      amp_cm = self.amp_context if self.amp_context is not None else contextlib.nullcontext()
      amp_cm.__enter__()

      shape = self.values.shape
      advantages = torch.zeros(shape, device=device)
      advantages = pufferl.compute_puff_advantage(
        self.values,
        self.rewards,
        self.terminals,
        self.ratio,
        advantages,
        config["gamma"],
        config["gae_lambda"],
        config["vtrace_rho_clip"],
        config["vtrace_c_clip"],
      )

      adv = advantages.abs().sum(axis=1)
      prio_weights_all = torch.nan_to_num(adv**a, 0, 0, 0)
      prio_weights = prio_weights_all[trainable_idx]
      if float(prio_weights.sum().item()) <= 0.0:
        prio_weights = torch.ones_like(prio_weights)
      prio_probs = prio_weights / (prio_weights.sum() + 1e-8)
      rel_idx = torch.multinomial(prio_probs, self.minibatch_segments, replacement=True)
      idx = trainable_idx[rel_idx]
      mb_prio = (max(trainable_idx.numel(), 1) * prio_probs[rel_idx, None]) ** -anneal_beta

      profile("train_copy", epoch)
      mb_obs = self.observations[idx]
      mb_actions = self.actions[idx]
      mb_logprobs = self.logprobs[idx]
      mb_values = self.values[idx]
      mb_returns = advantages[idx] + mb_values
      mb_advantages = advantages[idx]

      profile("train_forward", epoch)
      if not config["use_rnn"]:
        mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

      state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
      logits, newvalue = self.policy(mb_obs, state)
      _, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

      profile("train_misc", epoch)
      newlogprob = newlogprob.reshape(mb_logprobs.shape)
      logratio = newlogprob - mb_logprobs
      ratio = logratio.exp()
      self.ratio[idx] = ratio.detach()

      with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

      adv_norm = mb_prio * (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
      pg_loss1 = -adv_norm * ratio
      pg_loss2 = -adv_norm * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
      pg_loss = torch.max(pg_loss1, pg_loss2).mean()

      newvalue = newvalue.view(mb_returns.shape)
      v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
      v_loss_unclipped = (newvalue - mb_returns) ** 2
      v_loss_clipped = (v_clipped - mb_returns) ** 2
      v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
      entropy_loss = entropy.mean()
      loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss

      self.values[idx] = newvalue.detach().float()

      profile("train_misc", epoch)
      losses["policy_loss"] += pg_loss.item() / self.total_minibatches
      losses["value_loss"] += v_loss.item() / self.total_minibatches
      losses["entropy"] += entropy_loss.item() / self.total_minibatches
      losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
      losses["approx_kl"] += approx_kl.item() / self.total_minibatches
      losses["clipfrac"] += clipfrac.item() / self.total_minibatches
      losses["importance"] += ratio.mean().item() / self.total_minibatches

      profile("learn", epoch)
      loss.backward()
      if (mb + 1) % self.accumulate_minibatches == 0:
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()

      amp_cm.__exit__(None, None, None)

    profile("train_misc", epoch)
    if config["anneal_lr"]:
      self.scheduler.step()

    eval_mask = self._segment_is_trainable
    if not bool(eval_mask.any().item()):
      eval_mask = torch.ones_like(eval_mask, dtype=torch.bool)
    y_pred = self.values[eval_mask].flatten()
    y_true = (advantages[eval_mask] + self.values[eval_mask]).flatten()
    var_y = y_true.var()
    explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
    losses["explained_variance"] = explained_var

    profile.end()
    logs = None
    self.epoch += 1
    done_training = self.global_step >= config["total_timesteps"]
    if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
      logs = self.mean_and_log()
      self.losses = losses
      self.print_dashboard()
      self.stats = defaultdict(list)
      self.last_log_time = time.time()
      self.last_log_step = self.global_step
      profile.clear()

    if self.epoch % config["checkpoint_interval"] == 0 or done_training:
      self.save_checkpoint()
      self.msg = f"Checkpoint saved at update {self.epoch}"

    return logs
