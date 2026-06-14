"""JAX GPU vectorized environment backend (battle-only).

`JaxVecEnv` exposes the azk_puffer vector interface (async_reset/recv/send/
close + the attributes the trainer reads) over `jax_env/azuki_jax`: one jitted
call steps all B envs and emits observations already packed in the byte
layout the policy consumes (azuki_jax.observe == emulation byte parity).

Semantics mirror the Multiprocessing + PettingZooPufferEnv(AzukiTCGParallel)
path:
- 2 agents per env; rows are env-major (row = env * 2 + player).
- Both players' obs/rewards/terminals/truncations are emitted every step.
- auto-reset happens inside azuki_jax.step.step (the action submitted on a
  reset transition is discarded, rewards are zero, dones False -- same as the
  Serial worker calling env.reset() when done).
- recv() `mask` is all True (PettingZooPufferEnv sets masks True for both
  agents every step of a parallel env).

KNOWN LIMITATION: only ability-free (vanilla) gameplay is implemented in the
JAX engine; ability cards' abilities NO-OP (state.ab_scratch[3] counts hits).
Training in this backend is a self-consistent MDP but not C-equivalent on
decks with abilities.

Env vars:
- AZK_JAX_CHECK_LEGAL: when set to a non-zero integer N, validates incoming
  actions against the previous step's legal mask on host for the first N
  send() calls (N=1 means the default of 256) and warn-logs misses. Illegal
  actions would otherwise silently diverge (azuki_jax.step does not abort,
  unlike the C engine).

TODO(perf): recv() currently returns host numpy (device->host copy of ~25MB
at B=512); switch to dlpack zero-copy into torch once the trainer accepts
torch tensors from recv().
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Share the GPU with torch: do not let XLA preallocate 75% of VRAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jaxcache")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JAX_ENV_DIR = _REPO_ROOT / "jax_env"
if str(_JAX_ENV_DIR) not in sys.path:
  sys.path.insert(0, str(_JAX_ENV_DIR))

import azk_puffer as pufferlib
from azk_puffer.emulation import emulate_action_space, emulate_observation_space

RESET = 0
SEND = 2
RECV = 3

_DEFAULT_CHECK_STEPS = 256


def _check_legal_steps() -> int:
  raw = os.getenv("AZK_JAX_CHECK_LEGAL", "0").strip()
  try:
    value = int(raw)
  except ValueError:
    value = 1 if raw.lower() in {"true", "yes", "on"} else 0
  if value <= 0:
    return 0
  return _DEFAULT_CHECK_STEPS if value == 1 else value


class JaxDriverEnv:
  """Driver shim mirroring the PettingZooPufferEnv attributes the trainer,
  policy ctor (env.emulated) and LSTMWrapper (single_observation_space) read."""

  def __init__(self):
    from action import build_action_space
    from observation import build_observation_space

    self.env_single_observation_space = build_observation_space()
    self.env_single_action_space = build_action_space()
    self.single_observation_space, self.obs_dtype = emulate_observation_space(
        self.env_single_observation_space)
    self.single_action_space, self.atn_dtype = emulate_action_space(
        self.env_single_action_space)
    self.emulated = dict(
        observation_dtype=self.single_observation_space.dtype,
        emulated_observation_dtype=self.obs_dtype,
    )
    self.num_agents = 2

  def close(self):
    pass


class JaxVecEnv:
  """Vecenv over azuki_jax: B envs stepped by one jitted device call."""

  @property
  def num_envs(self):
    return self.agents_per_batch

  def __init__(self, num_envs: int, deck_pool, seed: int = 0, **kwargs):
    import jax
    import jax.numpy as jnp

    from azuki_jax.engine.step import stabilize
    from azuki_jax.env import init_state
    from azuki_jax.observe import ITEMSIZE, packed_observation_with_mask
    from azuki_jax.setup import build_deck_pool_tables
    from azuki_jax.step import step as env_step

    self._jax = jax
    self._jnp = jnp
    self.obs_bytes = ITEMSIZE

    pool = build_deck_pool_tables(deck_pool)
    self._pool = pool

    self.driver_env = JaxDriverEnv()
    self.num_environments = int(num_envs)
    self.num_agents = self.driver_env.num_agents * self.num_environments
    self.agents_per_batch = self.num_agents
    self.single_observation_space = self.driver_env.single_observation_space
    self.single_action_space = self.driver_env.single_action_space
    self.observation_space = pufferlib.spaces.joint_space(
        self.single_observation_space, self.agents_per_batch)
    self.action_space = pufferlib.spaces.joint_space(
        self.single_action_space, self.agents_per_batch)
    self.emulated = self.driver_env.emulated
    self.agent_ids = np.arange(self.num_agents)
    self.seed = int(seed)

    def _init_one(env_seed):
      return stabilize(init_state(env_seed, pool))

    def _observe_one(state):
      return packed_observation_with_mask(state)

    def _step_one(state, actions, prev_terms, prev_truncs):
      state, rewards, terms, truncs = env_step(
          state, actions, prev_terms, prev_truncs, pool)
      obs, legal, count = packed_observation_with_mask(state)
      return state, obs, rewards, terms, truncs, legal, count, state.active_player

    self._init_fn = jax.jit(jax.vmap(_init_one))
    self._observe_fn = jax.jit(jax.vmap(_observe_one))
    self._step_fn = jax.jit(jax.vmap(_step_one))

    self._states = None
    self._terms = None
    self._truncs = None
    self._pending = None

    # action legality checking (host-side, first N sends)
    self._check_steps_left = _check_legal_steps()
    self._check_total = 0
    self._check_misses = 0
    self._last_mask_host = None

    self.flag = RESET
    self.initialized = False

  # -- vecenv interface ----------------------------------------------------

  def async_reset(self, seed=None):
    jnp = self._jnp
    if seed is None:
      seed = self.seed
    seeds = (np.uint32(seed) + np.arange(self.num_environments, dtype=np.uint64)
             ).astype(np.uint32)
    states = self._init_fn(jnp.asarray(seeds))
    obs, legal, count = self._observe_fn(states)
    b = self.num_environments
    self._states = states
    self._terms = jnp.zeros((b, 2), jnp.bool_)
    self._truncs = jnp.zeros((b, 2), jnp.bool_)
    rewards = jnp.zeros((b, 2), jnp.float32)
    active = states.active_player
    self._pending = (obs, rewards, self._terms, self._truncs, legal, count, active)
    self.flag = RECV

  def recv(self):
    if self.flag != RECV:
      raise pufferlib.APIUsageError('Call reset before stepping')
    self.flag = SEND

    obs, rewards, terms, truncs, legal, count, active = self._pending
    o = np.asarray(obs).reshape(self.num_agents, self.obs_bytes)
    r = np.asarray(rewards).astype(np.float32, copy=False).ravel()
    d = np.asarray(terms).ravel()
    t = np.asarray(truncs).ravel()

    if self._check_steps_left > 0:
      self._last_mask_host = (
          np.asarray(legal),
          np.asarray(count).astype(np.int32),
          np.asarray(active).astype(np.int32),
      )

    mask = np.ones(self.num_agents, dtype=bool)
    infos = []  # TODO: surface episode stats (returns/lengths/win rates)
    return o, r, d, t, infos, self.agent_ids, mask

  def send(self, actions):
    if self.flag != SEND:
      raise pufferlib.APIUsageError('Call (async) reset + recv before sending')
    self.flag = RECV

    jnp = self._jnp
    acts = np.asarray(actions).astype(np.int32, copy=False)
    acts = np.ascontiguousarray(acts).reshape(self.num_environments, 2, 4)

    if self._check_steps_left > 0 and self._last_mask_host is not None:
      self._validate_actions(acts)
      self._check_steps_left -= 1
      if self._check_steps_left == 0:
        self._report_check(final=True)

    states, obs, rewards, terms, truncs, legal, count, active = self._step_fn(
        self._states, jnp.asarray(acts), self._terms, self._truncs)
    self._states = states
    self._terms = terms
    self._truncs = truncs
    self._pending = (obs, rewards, terms, truncs, legal, count, active)

  def notify(self):
    pass

  def close(self):
    if self._check_total:
      self._report_check(final=True)
    self._states = None
    self._pending = None
    self.driver_env.close()

  # -- legality checking ----------------------------------------------------

  def _validate_actions(self, acts: np.ndarray):
    legal, count, active = self._last_mask_host
    b = self.num_environments
    rows = np.arange(b)
    chosen = acts[rows, np.clip(active, 0, 1)]  # (B, 4)
    hit = (legal.astype(np.int32) == chosen[:, None, :]).all(axis=-1)  # (B, 1024)
    in_range = np.arange(legal.shape[1])[None, :] < count[:, None]
    ok = (hit & in_range).any(axis=1) | (count == 0)  # count==0: action discarded
    checked = int((count > 0).sum())
    misses = np.flatnonzero(~ok)
    self._check_total += checked
    if misses.size:
      self._check_misses += int(misses.size)
      head = misses[:4]
      detail = "; ".join(
          f"env {int(e)} active {int(active[e])} action {chosen[e].tolist()}"
          f" (legal_count {int(count[e])})"
          for e in head
      )
      print(
          f"[JaxVecEnv][WARN] {misses.size} illegal action(s) this step: {detail}",
          file=sys.stderr,
          flush=True,
      )

  def _report_check(self, final=False):
    if self._check_total == 0 and self._check_misses == 0:
      return
    print(
        f"[JaxVecEnv] legality check: {self._check_misses} misses /"
        f" {self._check_total} checked actions",
        file=sys.stderr,
        flush=True,
    )
    if final:
      self._check_total = 0
      self._check_misses = 0
      self._last_mask_host = None

  # -- sync helpers (vector.reset/step parity) ------------------------------

  def reset(self, seed=42):
    self.async_reset(seed)
    obs, _, _, _, infos, _, _ = self.recv()
    return obs, infos

  def step(self, actions):
    self.send(np.asarray(actions))
    obs, rewards, terminals, truncations, infos, _, _ = self.recv()
    return obs, rewards, terminals, truncations, infos


def bench(num_envs: int = 512, steps: int = 100, seed: int = 1):
  """Quick SPS probe: random legal actions chosen on host from the obs mask."""
  from training_deck_pool import load_training_deck_pool

  pool = load_training_deck_pool()
  env = JaxVecEnv(num_envs, pool, seed=seed)
  env.async_reset(seed)
  rng = np.random.default_rng(seed)

  def pick_actions():
    obs_legal = np.asarray(env._pending[4])
    obs_count = np.asarray(env._pending[5]).astype(np.int32)
    b = env.num_environments
    idx = rng.integers(0, np.maximum(obs_count, 1))
    rows = obs_legal[np.arange(b), idx].astype(np.int32)  # (B, 4)
    acts = np.zeros((b, 2, 4), np.int32)
    act_p = np.asarray(env._pending[6]).astype(np.int32)
    acts[np.arange(b), np.clip(act_p, 0, 1)] = rows
    return acts.reshape(b * 2, 4)

  env.recv()
  env.send(pick_actions())
  env.recv()  # compile + warmup
  start = time.perf_counter()
  for _ in range(steps):
    env.send(pick_actions())
    env.recv()
  elapsed = time.perf_counter() - start
  sps = steps * env.num_agents / elapsed
  print(f"JaxVecEnv bench: {num_envs} envs, {sps:,.0f} agent-steps/s"
        f" ({elapsed / steps * 1e3:.1f} ms/step)")
  env.close()


if __name__ == "__main__":
  bench(*(int(x) for x in sys.argv[1:]))
