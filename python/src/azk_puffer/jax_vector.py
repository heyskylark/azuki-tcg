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

Parity note: full-pool C/JAX parity is verified by `jax_env/tests`, not by this
benchmark shim. Keep this backend's API simple and let the parity suite remain
the authority for gameplay equivalence.

Env vars:
- AZK_JAX_CHECK_LEGAL: when set to a non-zero integer N, validates incoming
  actions against the previous step's legal mask on host for the first N
  send() calls (N=1 means the default of 256) and warn-logs misses. Illegal
  actions would otherwise silently diverge (azuki_jax.step does not abort,
  unlike the C engine).
- AZK_JAX_SPLIT_ACTIONS: defaults to 1. When enabled, compile separate
  static-action kernels and merge rows by primary action type instead of
  compiling the monolithic dynamic engine_step.

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

    from azuki_jax import cards as jax_cards
    from azuki_jax.constants import (
        Act,
        AbilityPhase,
        CardType,
        GARDEN_SIZE,
        Phase,
        TOKEN_INSTANCE,
        Zone,
    )
    from azuki_jax.engine.apply import apply_mulligan
    from azuki_jax.env import init_state, reset_state
    from azuki_jax.observe import ITEMSIZE, packed_observation_with_mask
    from azuki_jax.setup import build_deck_pool_tables
    from azuki_jax.step import (
        _simple_start_turn_no_triggers,
        step_activate_stt02_001_fast,
        step_activate_azk01_070_fast,
        step_activate_azk01_105_fast,
        step_activate_azk01_121_fast,
        step_activate_stt03_001_fast,
        step_activate_stt02_011_fast,
        step_activate_stt04_001_fast,
        step_activate_stt01_005_fast,
        step_attack_azk01_004_leader_fast,
        step_attack_azk01_060_confirm_fast,
        step_attack_stt01_006_effect_fast,
        step_attach_stt01_013_confirm_fast,
        step_attach_weapon_simple_fast,
        step_attack_entity_mutual_destroy_fast,
        step_attack_leader_simple_fast,
        step_attack_leader_response_fast,
        step_bottom_deck_all_fast,
        step_bottom_deck_card_fast,
        step_confirm_azk01_060_fast,
        step_confirm_azk01_060_response_fast,
        step_confirm_azk01_058_fast,
        step_confirm_clear_fast,
        step_confirm_stt01_002_fast,
        step_confirm_stt01_007_fast,
        step_confirm_stt01_013_fast,
        step_confirm_stt01_004_fast,
        step_confirm_stt02_009_fast,
        step_confirm_stt04_004_fast,
        step_declare_defender_fast,
        step_effect_azk01_032_fast,
        step_effect_azk01_007_fast,
        step_effect_azk01_009_fast,
        step_effect_azk01_040_fast,
        step_effect_azk01_058_fast,
        step_effect_azk01_059_fast,
        step_effect_azk01_065_fast,
        step_effect_azk01_070_fast,
        step_effect_azk01_105_fast,
        step_effect_azk01_127_fast,
        step_effect_stt01_005_fast,
        step_effect_stt01_006_fast,
        step_effect_stt01_014_fast,
        step_effect_stt01_017_fast,
        step_effect_stt02_001_fast,
        step_effect_stt02_011_fast,
        step_effect_stt02_016_fast,
        step_effect_stt02_009_fast,
        step_effect_stt03_002_fast,
        step_effect_stt03_006_fast,
        step_effect_stt04_001_fast,
        step_effect_stt04_004_fast,
        step_effect_stt04_016_fast,
        step_gate_portal_simple_fast,
        step_main_noop_azk01_011_fast,
        step_main_noop_fast,
        step_main_noop_simple_fast,
        step_main_noop_stt04_003_fast,
        step_play_azk01_007_effect_fast,
        step_play_azk01_003_reveal_fast,
        step_play_azk01_033_reveal_fast,
        step_play_azk01_045_reveal_fast,
        step_play_azk01_056_reveal_fast,
        step_play_azk01_097_reveal_fast,
        step_play_entity_simple_fast,
        step_play_stt01_007_confirm_fast,
        step_play_stt02_003_reveal_fast,
        step_play_stt02_013_reveal_fast,
        step_play_stt02_009_confirm_fast,
        step_play_spell_azk01_002_fast,
        step_play_spell_azk01_032_fast,
        step_play_spell_azk01_009_fast,
        step_play_spell_azk01_065_fast,
        step_play_spell_azk01_127_fast,
        step_play_spell_stt01_017_fast,
        step_play_spell_stt02_016_fast,
        step_play_spell_stt03_016_fast,
        step_play_spell_stt04_016_fast,
        step_response_noop_azk01_040_fast,
        step_response_noop_entity_combat_fast,
        step_response_noop_leader_combat_fast,
        step_selection_pick_noop_fast,
        step_select_azk01_097_fast,
        step_select_azk01_122_place_fast,
        step_select_azk01_126_pick_fast,
        step_select_cost_azk01_032_fast,
        step_select_cost_stt01_007_fast,
        step_select_cost_stt02_016_fast,
        step_select_stt01_002_equip_fast,
        step_select_stt02_003_pick_fast,
        step_select_stt02_013_pick_fast,
        step_select_stt01_004_pick_fast,
        step_select_azk01_045_pick_fast,
        step_select_azk01_056_pick_fast,
        step_select_cost_stt04_016_fast,
        step_select_cost_stt01_004_fast,
        step_select_cost_stt02_009_fast,
        step_select_azk01_003_pick_fast,
        step_select_azk01_033_pick_fast,
        step_zero_legal_fast,
        step_with_legal_count as env_step,
        step_with_legal_count_static_action as env_step_static,
    )
    from azuki_jax.abilities import tables as ab_tables

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
      # A fresh game already rests at the mulligan decision point; compiling
      # stabilize() here pulls in the full auto-resolve loop without changing
      # the reset state.
      return init_state(env_seed, pool)

    def _observe_one(state):
      return packed_observation_with_mask(state)

    def _step_one(state, actions, prev_terms, prev_truncs, legal_count):
      state, rewards, terms, truncs = env_step(
          state, actions, prev_terms, prev_truncs, pool, legal_count)
      return state, rewards, terms, truncs

    def _zero_legal_one(state, actions, prev_terms, prev_truncs, legal_count):
      return step_zero_legal_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _pregame_one(state, actions, prev_terms, prev_truncs, legal_count):
      was_done = (prev_terms[0] & prev_terms[1]) | (
          prev_truncs[0] & prev_truncs[1]
      )
      fresh = reset_state(state, pool)
      fresh = fresh._replace(
          completed_episodes=state.completed_episodes + 1,
          tick=jnp.int32(0),
      )
      state = jax.tree.map(lambda a, b: jnp.where(was_done, a, b), fresh, state)
      state = state._replace(tick=state.tick + 1)
      acting = state.active_player.astype(jnp.int32)
      action = actions[acting]
      zero_legal = (legal_count.astype(jnp.int32) == 0) & ~was_done
      stepped = apply_mulligan(state, action[0])
      stepped = jax.lax.cond(
          stepped.phase == Phase.START_OF_TURN,
          _simple_start_turn_no_triggers,
          lambda x: x,
          stepped,
      )
      state = jax.tree.map(
          lambda a, b: jnp.where(was_done | zero_legal, b, a), stepped, state
      )
      rewards = jnp.zeros((2,), jnp.float32)
      terminals = jnp.zeros((2,), jnp.bool_)
      truncations = jnp.broadcast_to(zero_legal, (2,))
      return state, rewards, terminals, truncations

    def _main_noop_simple_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_main_noop_simple_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _main_noop_azk01_011_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_main_noop_azk01_011_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _main_noop_one(args):
      state, actions, prev_terms, prev_truncs, legal_count = args
      return step_main_noop_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _main_noop_batch(states, actions, prev_terms, prev_truncs, legal_count):
      return jax.lax.map(
          _main_noop_one,
          (states, actions, prev_terms, prev_truncs, legal_count),
      )

    def _make_play_entity_simple_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_entity_simple_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_entity_simple_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_entity_simple_one))

    def _make_play_azk01_007_effect_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_azk01_007_effect_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_azk01_007_effect_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_azk01_007_effect_one))

    def _make_play_stt01_007_confirm_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_stt01_007_confirm_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_stt01_007_confirm_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_stt01_007_confirm_one))

    def _make_play_azk01_003_reveal_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_azk01_003_reveal_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_azk01_003_reveal_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_azk01_003_reveal_one))

    def _make_play_azk01_033_reveal_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_azk01_033_reveal_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_azk01_033_reveal_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_azk01_033_reveal_one))

    def _make_play_azk01_045_reveal_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_azk01_045_reveal_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_azk01_045_reveal_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_azk01_045_reveal_one))

    def _make_play_azk01_056_reveal_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_azk01_056_reveal_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_azk01_056_reveal_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_azk01_056_reveal_one))

    def _make_play_azk01_097_reveal_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_azk01_097_reveal_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_azk01_097_reveal_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_azk01_097_reveal_one))

    def _make_play_stt02_003_reveal_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_stt02_003_reveal_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_stt02_003_reveal_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_stt02_003_reveal_one))

    def _make_play_stt02_013_reveal_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_stt02_013_reveal_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_stt02_013_reveal_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_stt02_013_reveal_one))

    def _make_play_stt02_009_confirm_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _play_stt02_009_confirm_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_play_stt02_009_confirm_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_play_stt02_009_confirm_one))

    def _gate_portal_simple_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_gate_portal_simple_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_stt04_016_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_stt04_016_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_stt02_016_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_stt02_016_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_stt01_017_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_stt01_017_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_azk01_032_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_azk01_032_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_azk01_002_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_azk01_002_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_azk01_009_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_azk01_009_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_azk01_065_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_azk01_065_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_azk01_127_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_azk01_127_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _play_spell_stt03_016_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_play_spell_stt03_016_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_cost_stt04_016_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_cost_stt04_016_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_cost_stt01_004_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_cost_stt01_004_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_cost_stt02_009_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_cost_stt02_009_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_cost_azk01_032_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_cost_azk01_032_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_cost_stt02_016_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_cost_stt02_016_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt04_016_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt04_016_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt02_009_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt02_009_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_032_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_032_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt02_016_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt02_016_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt01_017_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt01_017_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt04_004_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt04_004_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt01_006_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt01_006_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt03_002_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt03_002_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt03_006_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt03_006_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt01_005_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt01_005_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt01_014_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt01_014_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_stt01_005_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_stt01_005_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_clear_one(state, actions, prev_terms, prev_truncs, legal_count):
      return step_confirm_clear_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_stt01_002_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_stt01_002_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_stt01_007_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_stt01_007_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_stt01_013_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_stt01_013_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_stt01_004_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_stt01_004_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_stt04_004_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_stt04_004_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_stt02_009_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_stt02_009_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_azk01_058_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_azk01_058_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_azk01_060_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_azk01_060_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _confirm_azk01_060_response_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_confirm_azk01_060_response_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_007_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_007_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_cost_stt01_007_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_cost_stt01_007_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_009_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_009_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_040_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_040_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_058_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_058_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_059_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_059_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_070_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_070_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_065_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_065_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_127_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_127_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt04_001_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt04_001_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt02_001_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt02_001_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_stt02_011_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_stt02_011_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _effect_azk01_105_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_effect_azk01_105_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_azk01_121_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_azk01_121_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_stt02_001_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_stt02_001_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_stt03_001_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_stt03_001_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_stt04_001_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_stt04_001_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_stt02_011_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_stt02_011_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_azk01_070_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_azk01_070_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _activate_azk01_105_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_activate_azk01_105_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_azk01_003_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_azk01_003_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_azk01_033_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_azk01_033_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_azk01_045_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_azk01_045_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_azk01_056_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_azk01_056_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_stt02_003_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_stt02_003_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_stt02_013_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_stt02_013_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_stt01_004_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_stt01_004_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_stt01_002_equip_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_stt01_002_equip_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_azk01_126_pick_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_azk01_126_pick_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _select_azk01_097_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_select_azk01_097_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _selection_pick_noop_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_selection_pick_noop_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _make_select_azk01_122_place_fn(placement_zone: int):
      placement_zone = int(placement_zone)

      def _select_azk01_122_place_one(
          state, actions, prev_terms, prev_truncs, legal_count
      ):
        return step_select_azk01_122_place_fast(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            placement_zone,
        )

      return jax.jit(jax.vmap(_select_azk01_122_place_one))

    def _bottom_deck_card_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_bottom_deck_card_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _bottom_deck_all_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_bottom_deck_all_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attach_weapon_simple_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attach_weapon_simple_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attach_stt01_013_confirm_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attach_stt01_013_confirm_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attack_leader_simple_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attack_leader_simple_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attack_azk01_004_leader_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attack_azk01_004_leader_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attack_leader_response_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attack_leader_response_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _response_noop_leader_combat_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_response_noop_leader_combat_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _response_noop_entity_combat_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_response_noop_entity_combat_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _response_noop_azk01_040_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_response_noop_azk01_040_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attack_entity_mutual_destroy_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attack_entity_mutual_destroy_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attack_azk01_060_confirm_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attack_azk01_060_confirm_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _attack_stt01_006_effect_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_attack_stt01_006_effect_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _declare_defender_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_declare_defender_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _main_noop_stt04_003_one(
        state, actions, prev_terms, prev_truncs, legal_count
    ):
      return step_main_noop_stt04_003_fast(
          state, actions, prev_terms, prev_truncs, pool, legal_count
      )

    def _make_step_type_fn(action_type: int):
      action_type = int(action_type)

      def _step_one_type(args):
        state, actions, prev_terms, prev_truncs, legal_count = args
        state, rewards, terms, truncs = env_step_static(
            state,
            actions,
            prev_terms,
            prev_truncs,
            pool,
            legal_count,
            action_type,
        )
        return state, rewards, terms, truncs

      def _step_batch_type(states, actions, prev_terms, prev_truncs, legal_count):
        # `vmap` rewrites scalar `lax.cond`/`lax.switch` into select-like
        # batched control flow, reintroducing the "evaluate every branch" cost
        # that the split kernels are meant to avoid. `lax.map` keeps the scalar
        # branchy body and maps it across the fixed batch.
        return jax.lax.map(
            _step_one_type,
            (states, actions, prev_terms, prev_truncs, legal_count),
        )

      return jax.jit(_step_batch_type)

    self._init_fn = jax.jit(jax.vmap(_init_one))
    self._observe_fn = jax.jit(jax.vmap(_observe_one))
    self._step_fn = jax.jit(jax.vmap(_step_one))
    self._zero_legal_step_fn = jax.jit(jax.vmap(_zero_legal_one))
    self._pregame_step_fn = jax.jit(jax.vmap(_pregame_one))
    self._main_noop_simple_step_fn = jax.jit(jax.vmap(_main_noop_simple_one))
    self._main_noop_azk01_011_step_fn = jax.jit(
        jax.vmap(_main_noop_azk01_011_one)
    )
    self._main_noop_stt04_003_step_fn = jax.jit(
        jax.vmap(_main_noop_stt04_003_one)
    )
    self._main_noop_step_fn = jax.jit(_main_noop_batch)
    self._play_entity_simple_fns = {
        int(Zone.GARDEN): _make_play_entity_simple_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_entity_simple_fn(Zone.ALLEY),
    }
    self._play_azk01_007_effect_fns = {
        int(Zone.GARDEN): _make_play_azk01_007_effect_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_azk01_007_effect_fn(Zone.ALLEY),
    }
    self._play_stt01_007_confirm_fns = {
        int(Zone.GARDEN): _make_play_stt01_007_confirm_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_stt01_007_confirm_fn(Zone.ALLEY),
    }
    self._play_azk01_003_reveal_fns = {
        int(Zone.GARDEN): _make_play_azk01_003_reveal_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_azk01_003_reveal_fn(Zone.ALLEY),
    }
    self._play_azk01_033_reveal_fns = {
        int(Zone.GARDEN): _make_play_azk01_033_reveal_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_azk01_033_reveal_fn(Zone.ALLEY),
    }
    self._play_azk01_045_reveal_fns = {
        int(Zone.GARDEN): _make_play_azk01_045_reveal_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_azk01_045_reveal_fn(Zone.ALLEY),
    }
    self._play_azk01_056_reveal_fns = {
        int(Zone.GARDEN): _make_play_azk01_056_reveal_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_azk01_056_reveal_fn(Zone.ALLEY),
    }
    self._play_azk01_097_reveal_fns = {
        int(Zone.GARDEN): _make_play_azk01_097_reveal_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_azk01_097_reveal_fn(Zone.ALLEY),
    }
    self._play_stt02_003_reveal_fns = {
        int(Zone.GARDEN): _make_play_stt02_003_reveal_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_stt02_003_reveal_fn(Zone.ALLEY),
    }
    self._play_stt02_013_reveal_fns = {
        int(Zone.GARDEN): _make_play_stt02_013_reveal_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_stt02_013_reveal_fn(Zone.ALLEY),
    }
    self._play_stt02_009_confirm_fns = {
        int(Zone.GARDEN): _make_play_stt02_009_confirm_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_play_stt02_009_confirm_fn(Zone.ALLEY),
    }
    self._play_spell_stt04_016_step_fn = jax.jit(
        jax.vmap(_play_spell_stt04_016_one)
    )
    self._play_spell_stt02_016_step_fn = jax.jit(
        jax.vmap(_play_spell_stt02_016_one)
    )
    self._play_spell_stt01_017_step_fn = jax.jit(
        jax.vmap(_play_spell_stt01_017_one)
    )
    self._play_spell_azk01_032_step_fn = jax.jit(
        jax.vmap(_play_spell_azk01_032_one)
    )
    self._play_spell_azk01_002_step_fn = jax.jit(
        jax.vmap(_play_spell_azk01_002_one)
    )
    self._play_spell_azk01_009_step_fn = jax.jit(
        jax.vmap(_play_spell_azk01_009_one)
    )
    self._play_spell_azk01_065_step_fn = jax.jit(
        jax.vmap(_play_spell_azk01_065_one)
    )
    self._play_spell_azk01_127_step_fn = jax.jit(
        jax.vmap(_play_spell_azk01_127_one)
    )
    self._play_spell_stt03_016_step_fn = jax.jit(
        jax.vmap(_play_spell_stt03_016_one)
    )
    self._select_cost_stt04_016_step_fn = jax.jit(
        jax.vmap(_select_cost_stt04_016_one)
    )
    self._select_cost_stt01_004_step_fn = jax.jit(
        jax.vmap(_select_cost_stt01_004_one)
    )
    self._select_cost_stt02_009_step_fn = jax.jit(
        jax.vmap(_select_cost_stt02_009_one)
    )
    self._select_cost_azk01_032_step_fn = jax.jit(
        jax.vmap(_select_cost_azk01_032_one)
    )
    self._select_cost_stt02_016_step_fn = jax.jit(
        jax.vmap(_select_cost_stt02_016_one)
    )
    self._effect_stt04_016_step_fn = jax.jit(jax.vmap(_effect_stt04_016_one))
    self._effect_stt02_009_step_fn = jax.jit(jax.vmap(_effect_stt02_009_one))
    self._effect_azk01_032_step_fn = jax.jit(jax.vmap(_effect_azk01_032_one))
    self._effect_stt02_016_step_fn = jax.jit(
        jax.vmap(_effect_stt02_016_one)
    )
    self._effect_stt01_017_step_fn = jax.jit(
        jax.vmap(_effect_stt01_017_one)
    )
    self._effect_stt02_011_step_fn = jax.jit(
        jax.vmap(_effect_stt02_011_one)
    )
    self._effect_azk01_105_step_fn = jax.jit(
        jax.vmap(_effect_azk01_105_one)
    )
    self._effect_stt04_001_step_fn = jax.jit(jax.vmap(_effect_stt04_001_one))
    self._effect_stt02_001_step_fn = jax.jit(jax.vmap(_effect_stt02_001_one))
    self._effect_stt04_004_step_fn = jax.jit(jax.vmap(_effect_stt04_004_one))
    self._effect_stt01_005_step_fn = jax.jit(jax.vmap(_effect_stt01_005_one))
    self._effect_stt01_006_step_fn = jax.jit(jax.vmap(_effect_stt01_006_one))
    self._effect_stt01_014_step_fn = jax.jit(
        jax.vmap(_effect_stt01_014_one)
    )
    self._effect_stt03_002_step_fn = jax.jit(jax.vmap(_effect_stt03_002_one))
    self._effect_stt03_006_step_fn = jax.jit(jax.vmap(_effect_stt03_006_one))
    self._activate_stt01_005_step_fn = jax.jit(
        jax.vmap(_activate_stt01_005_one)
    )
    self._gate_portal_simple_step_fn = jax.jit(jax.vmap(_gate_portal_simple_one))
    self._confirm_clear_step_fn = jax.jit(jax.vmap(_confirm_clear_one))
    self._confirm_stt01_002_step_fn = jax.jit(
        jax.vmap(_confirm_stt01_002_one)
    )
    self._confirm_stt01_007_step_fn = jax.jit(
        jax.vmap(_confirm_stt01_007_one)
    )
    self._confirm_stt01_013_step_fn = jax.jit(
        jax.vmap(_confirm_stt01_013_one)
    )
    self._confirm_stt01_004_step_fn = jax.jit(
        jax.vmap(_confirm_stt01_004_one)
    )
    self._confirm_stt04_004_step_fn = jax.jit(
        jax.vmap(_confirm_stt04_004_one)
    )
    self._confirm_stt02_009_step_fn = jax.jit(
        jax.vmap(_confirm_stt02_009_one)
    )
    self._confirm_azk01_058_step_fn = jax.jit(
        jax.vmap(_confirm_azk01_058_one)
    )
    self._confirm_azk01_060_step_fn = jax.jit(
        jax.vmap(_confirm_azk01_060_one)
    )
    self._confirm_azk01_060_response_step_fn = jax.jit(
        jax.vmap(_confirm_azk01_060_response_one)
    )
    self._effect_azk01_007_step_fn = jax.jit(
        jax.vmap(_effect_azk01_007_one)
    )
    self._select_cost_stt01_007_step_fn = jax.jit(
        jax.vmap(_select_cost_stt01_007_one)
    )
    self._effect_azk01_009_step_fn = jax.jit(
        jax.vmap(_effect_azk01_009_one)
    )
    self._effect_azk01_040_step_fn = jax.jit(
        jax.vmap(_effect_azk01_040_one)
    )
    self._effect_azk01_058_step_fn = jax.jit(
        jax.vmap(_effect_azk01_058_one)
    )
    self._effect_azk01_059_step_fn = jax.jit(
        jax.vmap(_effect_azk01_059_one)
    )
    self._effect_azk01_070_step_fn = jax.jit(
        jax.vmap(_effect_azk01_070_one)
    )
    self._effect_azk01_065_step_fn = jax.jit(
        jax.vmap(_effect_azk01_065_one)
    )
    self._effect_azk01_127_step_fn = jax.jit(
        jax.vmap(_effect_azk01_127_one)
    )
    self._activate_azk01_121_step_fn = jax.jit(
        jax.vmap(_activate_azk01_121_one)
    )
    self._activate_stt02_001_step_fn = jax.jit(
        jax.vmap(_activate_stt02_001_one)
    )
    self._activate_stt03_001_step_fn = jax.jit(
        jax.vmap(_activate_stt03_001_one)
    )
    self._activate_stt04_001_step_fn = jax.jit(
        jax.vmap(_activate_stt04_001_one)
    )
    self._activate_stt02_011_step_fn = jax.jit(
        jax.vmap(_activate_stt02_011_one)
    )
    self._activate_azk01_070_step_fn = jax.jit(
        jax.vmap(_activate_azk01_070_one)
    )
    self._activate_azk01_105_step_fn = jax.jit(
        jax.vmap(_activate_azk01_105_one)
    )
    self._select_azk01_003_pick_step_fn = jax.jit(
        jax.vmap(_select_azk01_003_pick_one)
    )
    self._select_azk01_033_pick_step_fn = jax.jit(
        jax.vmap(_select_azk01_033_pick_one)
    )
    self._select_azk01_045_pick_step_fn = jax.jit(
        jax.vmap(_select_azk01_045_pick_one)
    )
    self._select_azk01_056_pick_step_fn = jax.jit(
        jax.vmap(_select_azk01_056_pick_one)
    )
    self._select_stt02_003_pick_step_fn = jax.jit(
        jax.vmap(_select_stt02_003_pick_one)
    )
    self._select_stt02_013_pick_step_fn = jax.jit(
        jax.vmap(_select_stt02_013_pick_one)
    )
    self._select_stt01_004_pick_step_fn = jax.jit(
        jax.vmap(_select_stt01_004_pick_one)
    )
    self._select_stt01_002_equip_step_fn = jax.jit(
        jax.vmap(_select_stt01_002_equip_one)
    )
    self._select_azk01_126_pick_step_fn = jax.jit(
        jax.vmap(_select_azk01_126_pick_one)
    )
    self._select_azk01_097_step_fn = jax.jit(
        jax.vmap(_select_azk01_097_one)
    )
    self._selection_pick_noop_step_fn = jax.jit(
        jax.vmap(_selection_pick_noop_one)
    )
    self._select_azk01_122_place_fns = {
        int(Zone.GARDEN): _make_select_azk01_122_place_fn(Zone.GARDEN),
        int(Zone.ALLEY): _make_select_azk01_122_place_fn(Zone.ALLEY),
    }
    self._bottom_deck_card_step_fn = jax.jit(jax.vmap(_bottom_deck_card_one))
    self._bottom_deck_all_step_fn = jax.jit(jax.vmap(_bottom_deck_all_one))
    self._attach_weapon_simple_step_fn = jax.jit(
        jax.vmap(_attach_weapon_simple_one)
    )
    self._attach_stt01_013_confirm_step_fn = jax.jit(
        jax.vmap(_attach_stt01_013_confirm_one)
    )
    self._attack_leader_simple_step_fn = jax.jit(
        jax.vmap(_attack_leader_simple_one)
    )
    self._attack_azk01_004_leader_step_fn = jax.jit(
        jax.vmap(_attack_azk01_004_leader_one)
    )
    self._attack_leader_response_step_fn = jax.jit(
        jax.vmap(_attack_leader_response_one)
    )
    self._response_noop_leader_combat_step_fn = jax.jit(
        jax.vmap(_response_noop_leader_combat_one)
    )
    self._response_noop_entity_combat_step_fn = jax.jit(
        jax.vmap(_response_noop_entity_combat_one)
    )
    self._response_noop_azk01_040_step_fn = jax.jit(
        jax.vmap(_response_noop_azk01_040_one)
    )
    self._attack_entity_mutual_destroy_step_fn = jax.jit(
        jax.vmap(_attack_entity_mutual_destroy_one)
    )
    self._attack_azk01_060_confirm_step_fn = jax.jit(
        jax.vmap(_attack_azk01_060_confirm_one)
    )
    self._attack_stt01_006_effect_step_fn = jax.jit(
        jax.vmap(_attack_stt01_006_effect_one)
    )
    self._declare_defender_step_fn = jax.jit(
        jax.vmap(_declare_defender_one)
    )
    self._make_step_type_fn = _make_step_type_fn
    self._step_type_fns = {}
    self._timing_eot = np.asarray(ab_tables.TIMING_END_OF_TURN, dtype=bool)
    self._timing_on_play = np.asarray(ab_tables.TIMING_ON_PLAY, dtype=bool)
    self._timing_when_equipped = np.asarray(
        ab_tables.TIMING_WHEN_EQUIPPED, dtype=bool
    )
    self._timing_when_attacking = np.asarray(
        ab_tables.TIMING_WHEN_ATTACKING, dtype=bool
    )
    self._timing_when_attacked = np.asarray(
        ab_tables.TIMING_WHEN_ATTACKED, dtype=bool
    )
    self._timing_after_attacking = np.asarray(
        ab_tables.TIMING_AFTER_ATTACKING, dtype=bool
    )
    self._timing_when_destroyed = np.asarray(
        ab_tables.TIMING_WHEN_DESTROYED, dtype=bool
    )
    self._timing_when_returned = np.asarray(
        ab_tables.TIMING_WHEN_RETURNED_TO_HAND, dtype=bool
    )
    self._timing_takes_damage = np.asarray(
        ab_tables.TIMING[:, ab_tables.TIMING_INDEX["AWhenTakesDamage"]],
        dtype=bool,
    )
    self._timing_deals_damage = np.asarray(
        ab_tables.TIMING[:, ab_tables.TIMING_INDEX["AWhenDealsDamage"]],
        dtype=bool,
    )
    self._timing_is_response = np.asarray(ab_tables.TIMING_IS_RESPONSE, dtype=bool)
    self._has_ability = np.asarray(ab_tables.HAS_ABILITY, dtype=bool)
    self._response_play_from_hand = np.asarray(
        ab_tables.RESPONSE_PLAY_FROM_HAND, dtype=bool
    )
    self._timing_enter_garden = np.asarray(
        ab_tables.TIMING_WHEN_ENTERS_GARDEN, dtype=bool
    )
    self._timing_start = np.asarray(ab_tables.TIMING_START_OF_TURN, dtype=bool)
    self._timing_start_each = np.asarray(
        ab_tables.TIMING_START_OF_EACH_TURN, dtype=bool
    )
    self._implemented = np.asarray(ab_tables.IMPLEMENTED, dtype=bool)
    self._ability_ikz_cost = np.asarray(
        ab_tables.ABILITY_IKZ_COST, dtype=np.int8
    )
    self._force_tapped = np.asarray(jax_cards.ATTR_GARDEN_FORCE_TAPPED, dtype=bool)
    self._card_type = np.asarray(jax_cards.TYPE, dtype=np.int8)
    self._card_element = np.asarray(jax_cards.ELEMENT, dtype=np.int8)
    self._water_element = 2
    self._card_type_leader = int(CardType.LEADER)
    self._card_type_entity = int(CardType.ENTITY)
    self._card_type_weapon = int(CardType.WEAPON)
    self._card_type_spell = int(CardType.SPELL)
    self._black_jade = np.asarray(
        jax_cards.SUBTYPE_MATRIX[:, jax_cards.subtype_index("BlackJade")],
        dtype=bool,
    )
    self._watercrafting = np.asarray(
        jax_cards.SUBTYPE_MATRIX[:, jax_cards.subtype_index("Watercrafting")],
        dtype=bool,
    )
    self._steelborn = np.asarray(
        jax_cards.SUBTYPE_MATRIX[:, jax_cards.subtype_index("Steelborn")],
        dtype=bool,
    )
    self._obsidian = np.asarray(
        jax_cards.SUBTYPE_MATRIX[:, jax_cards.subtype_index("Obsidian")],
        dtype=bool,
    )
    self._scorchweaver = np.asarray(
        jax_cards.SUBTYPE_MATRIX[:, jax_cards.subtype_index("Scorchweaver")],
        dtype=bool,
    )
    self._counts_as_ikz = np.asarray(
        jax_cards.ATTR_COUNTS_AS_IKZ_SOURCE, dtype=bool
    )
    self._has_ikz_cost = np.asarray(jax_cards.HAS_IKZ_COST, dtype=bool)
    self._ikz_cost = np.asarray(jax_cards.IKZ_COST, dtype=np.int8)
    self._gate_points = np.asarray(jax_cards.GATE_POINTS, dtype=np.int8)
    self._base_hp = np.asarray(jax_cards.BASE_HP, dtype=np.int8)
    self._has_base_stats = np.asarray(jax_cards.HAS_BASE_STATS, dtype=bool)
    self._garden_size = int(GARDEN_SIZE)
    self._act_play_garden = int(Act.PLAY_ENTITY_TO_GARDEN)
    self._act_play_alley = int(Act.PLAY_ENTITY_TO_ALLEY)
    self._act_gate_portal = int(Act.GATE_PORTAL)
    self._act_attach_weapon = int(Act.ATTACH_WEAPON_FROM_HAND)
    self._act_play_spell = int(Act.PLAY_SPELL_FROM_HAND)
    self._act_activate_garden = int(Act.ACTIVATE_GARDEN_OR_LEADER_ABILITY)
    self._act_activate_alley = int(Act.ACTIVATE_ALLEY_ABILITY)
    self._act_attack = int(Act.ATTACK)
    self._act_declare_defender = int(Act.DECLARE_DEFENDER)
    self._act_confirm = int(Act.CONFIRM_ABILITY)
    self._act_select_from_selection = int(Act.SELECT_FROM_SELECTION)
    self._act_select_to_alley = int(Act.SELECT_TO_ALLEY)
    self._act_select_to_garden = int(Act.SELECT_TO_GARDEN)
    self._act_select_to_equip = int(Act.SELECT_TO_EQUIP)
    self._act_select_cost = int(Act.SELECT_COST_TARGET)
    self._act_select_effect = int(Act.SELECT_EFFECT_TARGET)
    self._act_bottom_deck_card = int(Act.BOTTOM_DECK_CARD)
    self._act_bottom_deck_all = int(Act.BOTTOM_DECK_ALL)
    self._act_noop = int(Act.NOOP)
    self._ability_confirmation = int(AbilityPhase.CONFIRMATION)
    self._ability_cost_selection = int(AbilityPhase.COST_SELECTION)
    self._ability_effect_selection = int(AbilityPhase.EFFECT_SELECTION)
    self._ability_selection_pick = int(AbilityPhase.SELECTION_PICK)
    self._ability_bottom_deck = int(AbilityPhase.BOTTOM_DECK)
    self._token_instance = int(TOKEN_INSTANCE)
    self._zone_attached = int(Zone.ATTACHED)
    self._zone_token = int(Zone.TOKEN)
    self._zone_deck = int(Zone.DECK)
    self._zone_hand = int(Zone.HAND)
    self._zone_discard = int(Zone.DISCARD)
    self._zone_selection = int(Zone.SELECTION)
    self._zone_garden = int(Zone.GARDEN)
    self._zone_alley = int(Zone.ALLEY)
    self._zone_ikz_area = int(Zone.IKZ_AREA)
    self._zone_leader = int(Zone.LEADER)
    self._zone_gate = int(Zone.GATE)
    self._simple_play_watch_ids = np.asarray(
        [
            jax_cards.CODE_TO_ID["STT01-008"],
            jax_cards.CODE_TO_ID["STT01-009"],
            jax_cards.CODE_TO_ID["STT01-011"],
            jax_cards.CODE_TO_ID["AZK01-010"],
            jax_cards.CODE_TO_ID["AZK01-019"],
            jax_cards.CODE_TO_ID["AZK01-073"],
            jax_cards.CODE_TO_ID["STT02-012"],
            jax_cards.CODE_TO_ID["AZK01-043"],
            jax_cards.CODE_TO_ID["AZK01-095"],
            jax_cards.CODE_TO_ID["STT03-013"],
        ],
        dtype=np.int16,
    )
    self._simple_play_implemented_ids = np.asarray(
        [
            jax_cards.CODE_TO_ID["STT01-003"],
            jax_cards.CODE_TO_ID["STT01-004"],
            jax_cards.CODE_TO_ID["STT02-005"],
            jax_cards.CODE_TO_ID["STT02-007"],
            jax_cards.CODE_TO_ID["STT03-009"],
            jax_cards.CODE_TO_ID["AZK01-098"],
            jax_cards.CODE_TO_ID["STT04-004"],
        ],
        dtype=np.int16,
    )
    self._simple_attach_watch_ids = np.asarray(
        [
            jax_cards.CODE_TO_ID["STT01-008"],
            jax_cards.CODE_TO_ID["STT01-009"],
            jax_cards.CODE_TO_ID["STT01-011"],
            jax_cards.CODE_TO_ID["AZK01-010"],
            jax_cards.CODE_TO_ID["AZK01-019"],
            jax_cards.CODE_TO_ID["AZK01-073"],
            jax_cards.CODE_TO_ID["STT02-012"],
            jax_cards.CODE_TO_ID["AZK01-043"],
            jax_cards.CODE_TO_ID["AZK01-095"],
        ],
        dtype=np.int16,
    )
    self._stt01_002_id = int(jax_cards.CODE_TO_ID["STT01-002"])
    self._stt01_003_id = int(jax_cards.CODE_TO_ID["STT01-003"])
    self._stt01_004_id = int(jax_cards.CODE_TO_ID["STT01-004"])
    self._stt01_005_id = int(jax_cards.CODE_TO_ID["STT01-005"])
    self._stt01_006_id = int(jax_cards.CODE_TO_ID["STT01-006"])
    self._stt01_007_id = int(jax_cards.CODE_TO_ID["STT01-007"])
    self._stt01_008_id = int(jax_cards.CODE_TO_ID["STT01-008"])
    self._stt01_013_id = int(jax_cards.CODE_TO_ID["STT01-013"])
    self._stt01_014_id = int(jax_cards.CODE_TO_ID["STT01-014"])
    self._stt01_017_id = int(jax_cards.CODE_TO_ID["STT01-017"])
    self._stt02_001_id = int(jax_cards.CODE_TO_ID["STT02-001"])
    self._stt02_002_id = int(jax_cards.CODE_TO_ID["STT02-002"])
    self._stt02_003_id = int(jax_cards.CODE_TO_ID["STT02-003"])
    self._stt02_009_id = int(jax_cards.CODE_TO_ID["STT02-009"])
    self._stt02_010_id = int(jax_cards.CODE_TO_ID["STT02-010"])
    self._stt02_011_id = int(jax_cards.CODE_TO_ID["STT02-011"])
    self._stt02_012_id = int(jax_cards.CODE_TO_ID["STT02-012"])
    self._stt02_013_id = int(jax_cards.CODE_TO_ID["STT02-013"])
    self._stt02_016_id = int(jax_cards.CODE_TO_ID["STT02-016"])
    self._stt03_001_id = int(jax_cards.CODE_TO_ID["STT03-001"])
    self._stt03_002_id = int(jax_cards.CODE_TO_ID["STT03-002"])
    self._stt03_006_id = int(jax_cards.CODE_TO_ID["STT03-006"])
    self._stt03_009_id = int(jax_cards.CODE_TO_ID["STT03-009"])
    self._stt03_016_id = int(jax_cards.CODE_TO_ID["STT03-016"])
    self._azk01_002_id = int(jax_cards.CODE_TO_ID["AZK01-002"])
    self._azk01_003_id = int(jax_cards.CODE_TO_ID["AZK01-003"])
    self._azk01_004_id = int(jax_cards.CODE_TO_ID["AZK01-004"])
    self._azk01_007_id = int(jax_cards.CODE_TO_ID["AZK01-007"])
    self._azk01_009_id = int(jax_cards.CODE_TO_ID["AZK01-009"])
    self._azk01_011_id = int(jax_cards.CODE_TO_ID["AZK01-011"])
    self._azk01_032_id = int(jax_cards.CODE_TO_ID["AZK01-032"])
    self._azk01_033_id = int(jax_cards.CODE_TO_ID["AZK01-033"])
    self._azk01_040_id = int(jax_cards.CODE_TO_ID["AZK01-040"])
    self._azk01_045_id = int(jax_cards.CODE_TO_ID["AZK01-045"])
    self._azk01_056_id = int(jax_cards.CODE_TO_ID["AZK01-056"])
    self._azk01_034_id = int(jax_cards.CODE_TO_ID["AZK01-034"])
    self._azk01_036_id = int(jax_cards.CODE_TO_ID["AZK01-036"])
    self._azk01_058_id = int(jax_cards.CODE_TO_ID["AZK01-058"])
    self._azk01_059_id = int(jax_cards.CODE_TO_ID["AZK01-059"])
    self._azk01_060_id = int(jax_cards.CODE_TO_ID["AZK01-060"])
    self._azk01_065_id = int(jax_cards.CODE_TO_ID["AZK01-065"])
    self._azk01_070_id = int(jax_cards.CODE_TO_ID["AZK01-070"])
    self._azk01_062_id = int(jax_cards.CODE_TO_ID["AZK01-062"])
    self._azk01_097_id = int(jax_cards.CODE_TO_ID["AZK01-097"])
    self._azk01_098_id = int(jax_cards.CODE_TO_ID["AZK01-098"])
    self._azk01_105_id = int(jax_cards.CODE_TO_ID["AZK01-105"])
    self._azk01_120_id = int(jax_cards.CODE_TO_ID["AZK01-120"])
    self._azk01_121_id = int(jax_cards.CODE_TO_ID["AZK01-121"])
    self._azk01_122_id = int(jax_cards.CODE_TO_ID["AZK01-122"])
    self._azk01_126_id = int(jax_cards.CODE_TO_ID["AZK01-126"])
    self._azk01_127_id = int(jax_cards.CODE_TO_ID["AZK01-127"])
    self._stt04_001_id = int(jax_cards.CODE_TO_ID["STT04-001"])
    self._stt04_002_id = int(jax_cards.CODE_TO_ID["STT04-002"])
    self._stt04_003_id = int(jax_cards.CODE_TO_ID["STT04-003"])
    self._stt04_004_id = int(jax_cards.CODE_TO_ID["STT04-004"])
    self._stt04_014_id = int(jax_cards.CODE_TO_ID["STT04-014"])
    self._stt04_016_id = int(jax_cards.CODE_TO_ID["STT04-016"])
    self._azk01_044_id = int(jax_cards.CODE_TO_ID["AZK01-044"])
    self._inherent_defender = np.asarray(jax_cards.INHERENT_DEFENDER, dtype=bool)
    self._inherent_infiltrate = np.asarray(
        jax_cards.INHERENT_INFILTRATE, dtype=bool
    )
    self._inherent_godmode = np.asarray(jax_cards.INHERENT_GODMODE, dtype=bool)
    split_raw = os.getenv("AZK_JAX_SPLIT_ACTIONS", "1").strip().lower()
    self._split_actions = split_raw not in {"0", "false", "no", "off"}

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

    if self._split_actions:
      states, rewards, terms, truncs = self._split_step_by_action(acts)
    else:
      states, rewards, terms, truncs = self._step_fn(
          self._states,
          jnp.asarray(acts),
          self._terms,
          self._truncs,
          self._pending[5],
      )
    obs, legal, count = self._observe_fn(states)
    active = states.active_player
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

  def _get_step_type_fn(self, action_type: int):
    action_type = int(action_type)
    fn = self._step_type_fns.get(action_type)
    if fn is None:
      fn = self._make_step_type_fn(action_type)
      self._step_type_fns[action_type] = fn
    return fn

  def _split_step_by_action(self, acts: np.ndarray):
    jnp = self._jnp
    b = self.num_environments
    active = np.asarray(self._pending[6]).astype(np.int32, copy=False)
    chosen = acts[np.arange(b), np.clip(active, 0, 1), 0].astype(np.int32)
    phase = np.asarray(self._states.phase).astype(np.int32, copy=False)
    zero_legal_mask_host = (
        np.asarray(self._pending[5]).astype(np.int32, copy=False) == 0
    )
    pregame_mask_host = phase == 0  # Phase.PREGAME_MULLIGAN
    play_garden_simple_mask_host = self._play_entity_simple_fast_mask(
        acts, chosen, phase, self._zone_garden, self._act_play_garden
    )
    play_alley_simple_mask_host = self._play_entity_simple_fast_mask(
        acts, chosen, phase, self._zone_alley, self._act_play_alley
    )
    play_azk01_007_garden_mask_host = (
        self._play_azk01_007_effect_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_azk01_007_alley_mask_host = (
        self._play_azk01_007_effect_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_stt01_007_garden_mask_host = (
        self._play_stt01_007_confirm_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_stt01_007_alley_mask_host = (
        self._play_stt01_007_confirm_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_azk01_003_garden_mask_host = (
        self._play_azk01_003_reveal_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_azk01_003_alley_mask_host = (
        self._play_azk01_003_reveal_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_azk01_033_garden_mask_host = (
        self._play_azk01_033_reveal_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_azk01_033_alley_mask_host = (
        self._play_azk01_033_reveal_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_azk01_045_garden_mask_host = (
        self._play_azk01_045_reveal_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_azk01_045_alley_mask_host = (
        self._play_azk01_045_reveal_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_azk01_056_garden_mask_host = (
        self._play_azk01_056_reveal_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_azk01_056_alley_mask_host = (
        self._play_azk01_056_reveal_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_azk01_097_garden_mask_host = (
        self._play_azk01_097_reveal_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_azk01_097_alley_mask_host = (
        self._play_azk01_097_reveal_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_stt02_003_garden_mask_host = (
        self._play_stt02_003_reveal_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_stt02_003_alley_mask_host = (
        self._play_stt02_003_reveal_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_stt02_013_garden_mask_host = (
        self._play_stt02_013_reveal_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_stt02_013_alley_mask_host = (
        self._play_stt02_013_reveal_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    play_stt02_009_garden_mask_host = (
        self._play_stt02_009_confirm_fast_mask(
            acts, chosen, phase, self._zone_garden, self._act_play_garden
        )
    )
    play_stt02_009_alley_mask_host = (
        self._play_stt02_009_confirm_fast_mask(
            acts, chosen, phase, self._zone_alley, self._act_play_alley
        )
    )
    gate_portal_simple_mask_host = self._gate_portal_simple_fast_mask(
        acts, chosen, phase
    )
    play_spell_stt04_016_mask_host = self._play_spell_stt04_016_fast_mask(
        acts, chosen, phase
    )
    play_spell_stt02_016_mask_host = self._play_spell_stt02_016_fast_mask(
        acts, chosen, phase
    )
    play_spell_stt01_017_mask_host = self._play_spell_stt01_017_fast_mask(
        acts, chosen, phase
    )
    play_spell_azk01_032_mask_host = self._play_spell_azk01_032_fast_mask(
        acts, chosen, phase
    )
    play_spell_azk01_002_mask_host = self._play_spell_azk01_002_fast_mask(
        acts, chosen, phase
    )
    play_spell_azk01_009_mask_host = self._play_spell_azk01_009_fast_mask(
        acts, chosen, phase
    )
    play_spell_azk01_065_mask_host = self._play_spell_azk01_065_fast_mask(
        acts, chosen, phase
    )
    play_spell_azk01_127_mask_host = self._play_spell_azk01_127_fast_mask(
        acts, chosen, phase
    )
    play_spell_stt03_016_mask_host = (
        self._play_spell_stt03_016_fast_mask(acts, chosen, phase)
    )
    select_cost_stt04_016_mask_host = self._select_cost_stt04_016_fast_mask(
        acts, chosen
    )
    select_cost_stt01_007_mask_host = self._select_cost_stt01_007_fast_mask(
        acts, chosen
    )
    select_cost_stt01_004_mask_host = self._select_cost_stt01_004_fast_mask(
        acts, chosen
    )
    select_cost_stt02_009_mask_host = self._select_cost_stt02_009_fast_mask(
        acts, chosen
    )
    select_cost_azk01_032_mask_host = self._select_cost_azk01_032_fast_mask(
        acts, chosen
    )
    select_cost_stt02_016_mask_host = self._select_cost_stt02_016_fast_mask(
        acts, chosen
    )
    effect_stt04_016_mask_host = self._effect_stt04_016_fast_mask(
        acts, chosen
    )
    effect_stt04_001_mask_host = self._effect_stt04_001_fast_mask(
        acts, chosen
    )
    effect_stt04_004_mask_host = self._effect_stt04_004_fast_mask(
        acts, chosen
    )
    effect_stt01_005_mask_host = self._effect_stt01_005_fast_mask(
        acts, chosen
    )
    effect_stt01_006_mask_host = self._effect_stt01_006_fast_mask(
        acts, chosen
    )
    effect_stt01_014_mask_host = self._effect_stt01_014_fast_mask(
        acts, chosen
    )
    effect_stt01_017_mask_host = self._effect_stt01_017_fast_mask(
        acts, chosen
    )
    effect_stt02_001_mask_host = self._effect_stt02_001_fast_mask(
        acts, chosen
    )
    effect_stt02_011_mask_host = self._effect_stt02_011_fast_mask(
        acts, chosen
    )
    effect_azk01_105_mask_host = self._effect_azk01_105_fast_mask(
        acts, chosen
    )
    effect_stt02_009_mask_host = self._effect_stt02_009_fast_mask(
        acts, chosen
    )
    effect_azk01_032_mask_host = self._effect_azk01_032_fast_mask(
        acts, chosen
    )
    effect_stt02_016_mask_host = self._effect_stt02_016_fast_mask(
        acts, chosen
    )
    effect_stt03_002_mask_host = self._effect_stt03_002_fast_mask(
        acts, chosen
    )
    effect_stt03_006_mask_host = self._effect_stt03_006_fast_mask(
        acts, chosen
    )
    confirm_clear_mask_host = self._confirm_clear_fast_mask(acts, chosen)
    confirm_stt01_002_mask_host = self._confirm_stt01_002_fast_mask(
        acts, chosen
    )
    confirm_stt01_007_mask_host = self._confirm_stt01_007_fast_mask(
        acts, chosen
    )
    confirm_stt01_013_mask_host = self._confirm_stt01_013_fast_mask(
        acts, chosen
    )
    confirm_stt01_004_mask_host = self._confirm_stt01_004_fast_mask(
        acts, chosen
    )
    confirm_stt04_004_mask_host = self._confirm_stt04_004_fast_mask(
        acts, chosen
    )
    confirm_stt02_009_mask_host = self._confirm_stt02_009_fast_mask(
        acts, chosen
    )
    confirm_azk01_058_mask_host = self._confirm_azk01_058_fast_mask(
        acts, chosen
    )
    confirm_azk01_060_mask_host = self._confirm_azk01_060_fast_mask(
        acts, chosen
    )
    confirm_azk01_060_response_mask_host = (
        self._confirm_azk01_060_response_fast_mask(acts, chosen)
    )
    confirm_clear_mask_host = (
        confirm_clear_mask_host
        & ~confirm_azk01_060_mask_host
        & ~confirm_azk01_060_response_mask_host
    )
    effect_azk01_007_mask_host = self._effect_azk01_007_fast_mask(
        acts, chosen
    )
    effect_azk01_009_mask_host = self._effect_azk01_009_fast_mask(
        acts, chosen
    )
    effect_azk01_040_mask_host = self._effect_azk01_040_fast_mask(
        acts, chosen
    )
    effect_azk01_058_mask_host = self._effect_azk01_058_fast_mask(
        acts, chosen
    )
    effect_azk01_059_mask_host = self._effect_azk01_059_fast_mask(
        acts, chosen
    )
    effect_azk01_070_mask_host = self._effect_azk01_070_fast_mask(
        acts, chosen
    )
    effect_azk01_065_mask_host = self._effect_azk01_065_fast_mask(
        acts, chosen
    )
    effect_azk01_127_mask_host = self._effect_azk01_127_fast_mask(
        acts, chosen
    )
    activate_stt04_001_mask_host = self._activate_stt04_001_fast_mask(
        acts, chosen, phase
    )
    activate_stt02_011_mask_host = self._activate_stt02_011_fast_mask(
        acts, chosen, phase
    )
    activate_stt02_001_mask_host = self._activate_stt02_001_fast_mask(
        acts, chosen, phase
    )
    activate_azk01_121_mask_host = self._activate_azk01_121_fast_mask(
        acts, chosen, phase
    )
    activate_stt03_001_mask_host = self._activate_stt03_001_fast_mask(
        acts, chosen, phase
    )
    activate_stt01_005_mask_host = self._activate_stt01_005_fast_mask(
        acts, chosen, phase
    )
    activate_azk01_070_mask_host = self._activate_azk01_070_fast_mask(
        acts, chosen, phase
    )
    activate_azk01_105_mask_host = self._activate_azk01_105_fast_mask(
        acts, chosen, phase
    )
    select_azk01_003_pick_mask_host = self._select_azk01_003_pick_fast_mask(
        acts, chosen
    )
    select_azk01_033_pick_mask_host = self._select_azk01_033_pick_fast_mask(
        acts, chosen
    )
    select_azk01_045_pick_mask_host = self._select_azk01_045_pick_fast_mask(
        acts, chosen
    )
    select_azk01_056_pick_mask_host = self._select_azk01_056_pick_fast_mask(
        acts, chosen
    )
    select_stt02_003_pick_mask_host = self._select_stt02_003_pick_fast_mask(
        acts, chosen
    )
    select_stt02_013_pick_mask_host = self._select_stt02_013_pick_fast_mask(
        acts, chosen
    )
    select_stt01_004_pick_mask_host = self._select_stt01_004_pick_fast_mask(
        acts, chosen
    )
    select_stt01_002_equip_mask_host = (
        self._select_stt01_002_equip_fast_mask(acts, chosen)
    )
    select_azk01_126_pick_mask_host = self._select_azk01_126_pick_fast_mask(
        acts, chosen
    )
    select_azk01_097_mask_host = self._select_azk01_097_fast_mask(
        acts, chosen
    )
    selection_pick_noop_mask_host = self._selection_pick_noop_fast_mask(
        acts, chosen
    )
    select_azk01_122_garden_mask_host = (
        self._select_azk01_122_place_fast_mask(
            acts, chosen, self._zone_garden, self._act_select_to_garden
        )
    )
    select_azk01_122_alley_mask_host = (
        self._select_azk01_122_place_fast_mask(
            acts, chosen, self._zone_alley, self._act_select_to_alley
        )
    )
    bottom_deck_card_mask_host = self._bottom_deck_azk01_003_fast_mask(
        acts, chosen, all_cards=False
    )
    bottom_deck_all_mask_host = self._bottom_deck_azk01_003_fast_mask(
        acts, chosen, all_cards=True
    )
    attach_weapon_simple_mask_host = self._attach_weapon_simple_fast_mask(
        acts, chosen, phase
    )
    attach_stt01_013_confirm_mask_host = (
        self._attach_stt01_013_confirm_fast_mask(acts, chosen, phase)
    )
    declare_defender_mask_host = self._declare_defender_fast_mask(
        acts, chosen, phase
    )
    attack_entity_simple_mask_host = (
        self._attack_entity_mutual_destroy_fast_mask(acts, chosen, phase)
    )
    attack_azk01_060_confirm_mask_host = (
        self._attack_azk01_060_confirm_fast_mask(acts, chosen, phase)
    )
    attack_stt01_006_effect_mask_host = (
        self._attack_stt01_006_effect_fast_mask(acts, chosen, phase)
    )
    attack_azk01_004_leader_mask_host = (
        self._attack_azk01_004_leader_fast_mask(acts, chosen, phase)
    )
    attack_leader_response_mask_host = (
        self._attack_leader_response_fast_mask(acts, chosen, phase)
    )
    attack_entity_response_mask_host = (
        self._attack_entity_response_fast_mask(acts, chosen, phase)
    )
    attack_leader_simple_mask_host = self._attack_leader_simple_fast_mask(
        acts, chosen, phase
    )
    response_noop_leader_combat_mask_host = (
        self._response_noop_leader_combat_fast_mask(acts, chosen, phase)
    )
    response_noop_entity_combat_mask_host = (
        self._response_noop_entity_combat_fast_mask(acts, chosen, phase)
    )
    response_noop_azk01_040_mask_host = (
        self._response_noop_azk01_040_fast_mask(acts, chosen, phase)
    )
    main_noop_simple_mask_host = self._main_noop_simple_fast_mask(chosen, phase)
    main_noop_azk01_011_mask_host = self._main_noop_azk01_011_fast_mask(
        chosen, phase
    )
    main_noop_stt04_003_mask_host = self._main_noop_stt04_003_fast_mask(
        chosen, phase
    )
    main_noop_mask_host = (
        self._main_noop_fast_mask(chosen, phase)
        & ~main_noop_simple_mask_host
        & ~main_noop_azk01_011_mask_host
        & ~main_noop_stt04_003_mask_host
    )
    if os.getenv("AZK_JAX_SPLIT_TRACE", "0") != "0":
      remaining_mask = ~(
          zero_legal_mask_host
          | pregame_mask_host
          | play_garden_simple_mask_host
          | play_alley_simple_mask_host
          | play_azk01_007_garden_mask_host
          | play_azk01_007_alley_mask_host
          | play_stt01_007_garden_mask_host
          | play_stt01_007_alley_mask_host
          | play_azk01_003_garden_mask_host
          | play_azk01_003_alley_mask_host
          | play_azk01_033_garden_mask_host
          | play_azk01_033_alley_mask_host
          | play_azk01_045_garden_mask_host
          | play_azk01_045_alley_mask_host
          | play_azk01_056_garden_mask_host
          | play_azk01_056_alley_mask_host
          | play_azk01_097_garden_mask_host
          | play_azk01_097_alley_mask_host
          | play_stt02_003_garden_mask_host
          | play_stt02_003_alley_mask_host
          | play_stt02_013_garden_mask_host
          | play_stt02_013_alley_mask_host
          | play_stt02_009_garden_mask_host
          | play_stt02_009_alley_mask_host
          | gate_portal_simple_mask_host
          | play_spell_stt04_016_mask_host
          | play_spell_stt02_016_mask_host
          | play_spell_stt01_017_mask_host
          | play_spell_azk01_032_mask_host
          | play_spell_azk01_002_mask_host
          | play_spell_azk01_009_mask_host
          | play_spell_azk01_065_mask_host
          | play_spell_azk01_127_mask_host
          | play_spell_stt03_016_mask_host
          | select_cost_stt04_016_mask_host
          | select_cost_stt01_007_mask_host
          | select_cost_stt01_004_mask_host
          | select_cost_stt02_009_mask_host
          | select_cost_azk01_032_mask_host
          | select_cost_stt02_016_mask_host
          | effect_stt04_016_mask_host
          | effect_stt04_001_mask_host
          | effect_stt04_004_mask_host
          | effect_stt01_005_mask_host
          | effect_stt01_006_mask_host
          | effect_stt01_014_mask_host
          | effect_stt01_017_mask_host
          | effect_stt02_001_mask_host
          | effect_stt02_011_mask_host
          | effect_azk01_105_mask_host
          | effect_stt02_009_mask_host
          | effect_azk01_032_mask_host
          | effect_stt02_016_mask_host
          | effect_stt03_002_mask_host
          | effect_stt03_006_mask_host
          | confirm_clear_mask_host
          | confirm_stt01_002_mask_host
          | confirm_stt01_007_mask_host
          | confirm_stt01_013_mask_host
          | confirm_stt01_004_mask_host
          | confirm_stt02_009_mask_host
          | confirm_stt04_004_mask_host
          | confirm_azk01_058_mask_host
          | confirm_azk01_060_mask_host
          | confirm_azk01_060_response_mask_host
          | effect_azk01_007_mask_host
          | effect_azk01_009_mask_host
          | effect_azk01_040_mask_host
          | effect_azk01_058_mask_host
          | effect_azk01_059_mask_host
          | effect_azk01_070_mask_host
          | effect_azk01_065_mask_host
          | effect_azk01_127_mask_host
          | activate_stt04_001_mask_host
          | activate_stt02_001_mask_host
          | activate_stt02_011_mask_host
          | activate_azk01_121_mask_host
          | activate_stt03_001_mask_host
          | activate_stt01_005_mask_host
          | activate_azk01_070_mask_host
          | activate_azk01_105_mask_host
          | select_azk01_003_pick_mask_host
          | select_azk01_033_pick_mask_host
          | select_azk01_045_pick_mask_host
          | select_azk01_056_pick_mask_host
          | select_stt02_003_pick_mask_host
          | select_stt02_013_pick_mask_host
          | select_stt01_004_pick_mask_host
          | select_stt01_002_equip_mask_host
          | select_azk01_126_pick_mask_host
          | select_azk01_097_mask_host
          | selection_pick_noop_mask_host
          | select_azk01_122_garden_mask_host
          | select_azk01_122_alley_mask_host
          | bottom_deck_card_mask_host
          | bottom_deck_all_mask_host
          | attach_weapon_simple_mask_host
          | attach_stt01_013_confirm_mask_host
          | declare_defender_mask_host
          | attack_azk01_060_confirm_mask_host
          | attack_stt01_006_effect_mask_host
          | attack_azk01_004_leader_mask_host
          | attack_leader_response_mask_host
          | attack_entity_response_mask_host
          | attack_entity_simple_mask_host
          | attack_leader_simple_mask_host
          | response_noop_leader_combat_mask_host
          | response_noop_entity_combat_mask_host
          | response_noop_azk01_040_mask_host
          | main_noop_simple_mask_host
          | main_noop_azk01_011_mask_host
          | main_noop_stt04_003_mask_host
          | main_noop_mask_host
      )
      remaining_types = ",".join(
          str(int(x)) for x in np.unique(chosen[remaining_mask])
      )
      print(
          "[JaxVecEnv] split masks:"
          f" zero_legal={int(zero_legal_mask_host.sum())}"
          f" pregame={int(pregame_mask_host.sum())}"
          f" play_simple={int(play_garden_simple_mask_host.sum() + play_alley_simple_mask_host.sum())}"
          f" play_azk01_007={int(play_azk01_007_garden_mask_host.sum() + play_azk01_007_alley_mask_host.sum())}"
          f" play_stt01_007={int(play_stt01_007_garden_mask_host.sum() + play_stt01_007_alley_mask_host.sum())}"
          f" play_azk01_003={int(play_azk01_003_garden_mask_host.sum() + play_azk01_003_alley_mask_host.sum())}"
          f" play_azk01_033={int(play_azk01_033_garden_mask_host.sum() + play_azk01_033_alley_mask_host.sum())}"
          f" play_azk01_045={int(play_azk01_045_garden_mask_host.sum() + play_azk01_045_alley_mask_host.sum())}"
          f" play_azk01_056={int(play_azk01_056_garden_mask_host.sum() + play_azk01_056_alley_mask_host.sum())}"
          f" play_azk01_097={int(play_azk01_097_garden_mask_host.sum() + play_azk01_097_alley_mask_host.sum())}"
          f" play_stt02_003={int(play_stt02_003_garden_mask_host.sum() + play_stt02_003_alley_mask_host.sum())}"
          f" play_stt02_013={int(play_stt02_013_garden_mask_host.sum() + play_stt02_013_alley_mask_host.sum())}"
          f" play_stt02_009={int(play_stt02_009_garden_mask_host.sum() + play_stt02_009_alley_mask_host.sum())}"
          f" gate_simple={int(gate_portal_simple_mask_host.sum())}"
          f" spell_stt04_016={int(play_spell_stt04_016_mask_host.sum())}"
          f" spell_stt02_016={int(play_spell_stt02_016_mask_host.sum())}"
          f" spell_stt01_017={int(play_spell_stt01_017_mask_host.sum())}"
          f" spell_azk01_032={int(play_spell_azk01_032_mask_host.sum())}"
          f" spell_azk01_002={int(play_spell_azk01_002_mask_host.sum())}"
          f" spell_azk01_009={int(play_spell_azk01_009_mask_host.sum())}"
          f" spell_azk01_065={int(play_spell_azk01_065_mask_host.sum())}"
          f" spell_azk01_127={int(play_spell_azk01_127_mask_host.sum())}"
          f" spell_stt03_016={int(play_spell_stt03_016_mask_host.sum())}"
          f" cost_stt04_016={int(select_cost_stt04_016_mask_host.sum())}"
          f" cost_stt01_007={int(select_cost_stt01_007_mask_host.sum())}"
          f" cost_stt01_004={int(select_cost_stt01_004_mask_host.sum())}"
          f" cost_stt02_009={int(select_cost_stt02_009_mask_host.sum())}"
          f" cost_azk01_032={int(select_cost_azk01_032_mask_host.sum())}"
          f" cost_stt02_016={int(select_cost_stt02_016_mask_host.sum())}"
          f" effect_stt04_016={int(effect_stt04_016_mask_host.sum())}"
          f" effect_stt04_001={int(effect_stt04_001_mask_host.sum())}"
          f" effect_stt04_004={int(effect_stt04_004_mask_host.sum())}"
          f" effect_stt01_005={int(effect_stt01_005_mask_host.sum())}"
          f" effect_stt01_006={int(effect_stt01_006_mask_host.sum())}"
          f" effect_stt01_014={int(effect_stt01_014_mask_host.sum())}"
          f" effect_stt01_017={int(effect_stt01_017_mask_host.sum())}"
          f" effect_stt02_001={int(effect_stt02_001_mask_host.sum())}"
          f" effect_stt02_011={int(effect_stt02_011_mask_host.sum())}"
          f" effect_azk01_105={int(effect_azk01_105_mask_host.sum())}"
          f" effect_stt02_009={int(effect_stt02_009_mask_host.sum())}"
          f" effect_azk01_032={int(effect_azk01_032_mask_host.sum())}"
          f" effect_stt02_016={int(effect_stt02_016_mask_host.sum())}"
          f" effect_stt03_002={int(effect_stt03_002_mask_host.sum())}"
          f" effect_stt03_006={int(effect_stt03_006_mask_host.sum())}"
          f" confirm_clear={int(confirm_clear_mask_host.sum())}"
          f" confirm_stt01_002={int(confirm_stt01_002_mask_host.sum())}"
          f" confirm_stt01_007={int(confirm_stt01_007_mask_host.sum())}"
          f" confirm_stt01_013={int(confirm_stt01_013_mask_host.sum())}"
          f" confirm_stt01_004={int(confirm_stt01_004_mask_host.sum())}"
          f" confirm_stt02_009={int(confirm_stt02_009_mask_host.sum())}"
          f" confirm_stt04_004={int(confirm_stt04_004_mask_host.sum())}"
          f" confirm_azk01_058={int(confirm_azk01_058_mask_host.sum())}"
          f" confirm_azk01_060={int(confirm_azk01_060_mask_host.sum())}"
          f" confirm_azk01_060_response={int(confirm_azk01_060_response_mask_host.sum())}"
          f" effect_azk01_007={int(effect_azk01_007_mask_host.sum())}"
          f" effect_azk01_009={int(effect_azk01_009_mask_host.sum())}"
          f" effect_azk01_040={int(effect_azk01_040_mask_host.sum())}"
          f" effect_azk01_058={int(effect_azk01_058_mask_host.sum())}"
          f" effect_azk01_059={int(effect_azk01_059_mask_host.sum())}"
          f" effect_azk01_070={int(effect_azk01_070_mask_host.sum())}"
          f" effect_azk01_065={int(effect_azk01_065_mask_host.sum())}"
          f" effect_azk01_127={int(effect_azk01_127_mask_host.sum())}"
          f" activate_stt04_001={int(activate_stt04_001_mask_host.sum())}"
          f" activate_stt02_001={int(activate_stt02_001_mask_host.sum())}"
          f" activate_stt02_011={int(activate_stt02_011_mask_host.sum())}"
          f" activate_azk01_121={int(activate_azk01_121_mask_host.sum())}"
          f" activate_stt03_001={int(activate_stt03_001_mask_host.sum())}"
          f" activate_stt01_005={int(activate_stt01_005_mask_host.sum())}"
          f" activate_azk01_070={int(activate_azk01_070_mask_host.sum())}"
          f" activate_azk01_105={int(activate_azk01_105_mask_host.sum())}"
          f" select_azk01_003={int(select_azk01_003_pick_mask_host.sum())}"
          f" select_azk01_033={int(select_azk01_033_pick_mask_host.sum())}"
          f" select_azk01_045={int(select_azk01_045_pick_mask_host.sum())}"
          f" select_azk01_056={int(select_azk01_056_pick_mask_host.sum())}"
          f" select_stt02_003={int(select_stt02_003_pick_mask_host.sum())}"
          f" select_stt02_013={int(select_stt02_013_pick_mask_host.sum())}"
          f" select_stt01_004={int(select_stt01_004_pick_mask_host.sum())}"
          f" select_stt01_002_equip={int(select_stt01_002_equip_mask_host.sum())}"
          f" select_azk01_126={int(select_azk01_126_pick_mask_host.sum())}"
          f" select_azk01_097={int(select_azk01_097_mask_host.sum())}"
          f" selection_noop={int(selection_pick_noop_mask_host.sum())}"
          f" select_azk01_122={int(select_azk01_122_garden_mask_host.sum() + select_azk01_122_alley_mask_host.sum())}"
          f" bottom_deck={int(bottom_deck_card_mask_host.sum() + bottom_deck_all_mask_host.sum())}"
          f" attach_simple={int(attach_weapon_simple_mask_host.sum())}"
          f" attach_stt01_013={int(attach_stt01_013_confirm_mask_host.sum())}"
          f" declare_defender={int(declare_defender_mask_host.sum())}"
          f" attack_azk01_060={int(attack_azk01_060_confirm_mask_host.sum())}"
          f" attack_stt01_006={int(attack_stt01_006_effect_mask_host.sum())}"
          f" attack_azk01_004={int(attack_azk01_004_leader_mask_host.sum())}"
          f" attack_leader_response={int(attack_leader_response_mask_host.sum())}"
          f" attack_entity_response={int(attack_entity_response_mask_host.sum())}"
          f" attack_entity_simple={int(attack_entity_simple_mask_host.sum())}"
          f" attack_simple={int(attack_leader_simple_mask_host.sum())}"
          f" response_noop={int(response_noop_leader_combat_mask_host.sum() + response_noop_entity_combat_mask_host.sum() + response_noop_azk01_040_mask_host.sum())}"
          f" response_noop_azk01_040={int(response_noop_azk01_040_mask_host.sum())}"
          f" main_noop_simple={int(main_noop_simple_mask_host.sum())}"
          f" main_noop_azk01_011={int(main_noop_azk01_011_mask_host.sum())}"
          f" main_noop_stt04_003={int(main_noop_stt04_003_mask_host.sum())}"
          f" main_noop={int(main_noop_mask_host.sum())}"
          f" generic={int(remaining_mask.sum())}"
          f" generic_types={remaining_types or '-'}",
          flush=True,
      )

    acts_dev = jnp.asarray(acts)
    states_acc = self._states
    rewards_acc = jnp.zeros((b, 2), jnp.float32)
    terms_acc = self._terms
    truncs_acc = self._truncs

    def merge_tree(old, new, mask):
      def merge_leaf(a, c):
        m = mask
        while m.ndim < c.ndim:
          m = m[..., None]
        return jnp.where(m, c, a)

      return self._jax.tree.map(merge_leaf, old, new)

    if np.any(zero_legal_mask_host):
      st_t, rw_t, tm_t, tr_t = self._zero_legal_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(zero_legal_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(pregame_mask_host):
      st_t, rw_t, tm_t, tr_t = self._pregame_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(pregame_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_azk01_007_garden_mask_host),
        (self._zone_alley, play_azk01_007_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_azk01_007_effect_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_stt01_007_garden_mask_host),
        (self._zone_alley, play_stt01_007_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_stt01_007_confirm_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_azk01_003_garden_mask_host),
        (self._zone_alley, play_azk01_003_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_azk01_003_reveal_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_azk01_033_garden_mask_host),
        (self._zone_alley, play_azk01_033_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_azk01_033_reveal_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_azk01_045_garden_mask_host),
        (self._zone_alley, play_azk01_045_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_azk01_045_reveal_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_azk01_056_garden_mask_host),
        (self._zone_alley, play_azk01_056_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_azk01_056_reveal_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_azk01_097_garden_mask_host),
        (self._zone_alley, play_azk01_097_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_azk01_097_reveal_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_stt02_003_garden_mask_host),
        (self._zone_alley, play_stt02_003_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_stt02_003_reveal_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_stt02_013_garden_mask_host),
        (self._zone_alley, play_stt02_013_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_stt02_013_reveal_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_stt02_009_garden_mask_host),
        (self._zone_alley, play_stt02_009_alley_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_stt02_009_confirm_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, play_mask_host in (
        (self._zone_garden, play_garden_simple_mask_host),
        (self._zone_alley, play_alley_simple_mask_host),
    ):
      if np.any(play_mask_host):
        st_t, rw_t, tm_t, tr_t = self._play_entity_simple_fns[placement_zone](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(play_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(gate_portal_simple_mask_host):
      st_t, rw_t, tm_t, tr_t = self._gate_portal_simple_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(gate_portal_simple_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_stt04_016_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_stt04_016_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_stt04_016_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_stt02_016_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_stt02_016_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_stt02_016_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_stt01_017_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_stt01_017_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_stt01_017_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_azk01_032_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_azk01_032_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_azk01_032_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_azk01_002_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_azk01_002_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_azk01_002_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_azk01_009_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_azk01_009_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_azk01_009_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_azk01_065_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_azk01_065_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_azk01_065_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_azk01_127_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_azk01_127_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_azk01_127_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(play_spell_stt03_016_mask_host):
      st_t, rw_t, tm_t, tr_t = self._play_spell_stt03_016_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(play_spell_stt03_016_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_cost_stt04_016_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_cost_stt04_016_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_cost_stt04_016_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_cost_stt01_007_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_cost_stt01_007_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_cost_stt01_007_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_cost_stt01_004_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_cost_stt01_004_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_cost_stt01_004_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_cost_stt02_009_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_cost_stt02_009_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_cost_stt02_009_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_cost_azk01_032_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_cost_azk01_032_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_cost_azk01_032_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_cost_stt02_016_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_cost_stt02_016_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_cost_stt02_016_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt04_016_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt04_016_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt04_016_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt04_001_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt04_001_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt04_001_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt04_004_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt04_004_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt04_004_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt01_005_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt01_005_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt01_005_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt01_006_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt01_006_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt01_006_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt01_014_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt01_014_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt01_014_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt01_017_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt01_017_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt01_017_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt02_001_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt02_001_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt02_001_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt02_011_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt02_011_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt02_011_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_105_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_105_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_105_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt02_009_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt02_009_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt02_009_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_032_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_032_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_032_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt02_016_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt02_016_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt02_016_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt03_002_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt03_002_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt03_002_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_stt03_006_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_stt03_006_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_stt03_006_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_clear_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_clear_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_clear_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_stt01_002_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_stt01_002_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_stt01_002_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_stt01_007_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_stt01_007_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_stt01_007_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_stt01_013_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_stt01_013_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_stt01_013_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_stt01_004_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_stt01_004_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_stt01_004_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_stt04_004_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_stt04_004_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_stt04_004_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_stt02_009_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_stt02_009_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_stt02_009_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_azk01_058_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_azk01_058_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_azk01_058_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_azk01_060_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_azk01_060_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_azk01_060_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(confirm_azk01_060_response_mask_host):
      st_t, rw_t, tm_t, tr_t = self._confirm_azk01_060_response_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(confirm_azk01_060_response_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_007_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_007_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_007_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_009_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_009_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_009_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_040_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_040_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_040_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_058_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_058_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_058_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_059_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_059_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_059_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_070_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_070_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_070_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_065_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_065_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_065_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(effect_azk01_127_mask_host):
      st_t, rw_t, tm_t, tr_t = self._effect_azk01_127_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(effect_azk01_127_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_stt04_001_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_stt04_001_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_stt04_001_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_stt02_001_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_stt02_001_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_stt02_001_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_stt02_011_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_stt02_011_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_stt02_011_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_azk01_121_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_azk01_121_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_azk01_121_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_stt03_001_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_stt03_001_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_stt03_001_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_stt01_005_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_stt01_005_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_stt01_005_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_azk01_070_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_azk01_070_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_azk01_070_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(activate_azk01_105_mask_host):
      st_t, rw_t, tm_t, tr_t = self._activate_azk01_105_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(activate_azk01_105_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_azk01_003_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_azk01_003_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_azk01_003_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_azk01_033_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_azk01_033_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_azk01_033_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_azk01_045_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_azk01_045_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_azk01_045_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_azk01_056_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_azk01_056_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_azk01_056_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_stt02_003_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_stt02_003_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_stt02_003_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_stt02_013_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_stt02_013_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_stt02_013_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_stt01_004_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_stt01_004_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_stt01_004_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_stt01_002_equip_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_stt01_002_equip_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_stt01_002_equip_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_azk01_126_pick_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_azk01_126_pick_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_azk01_126_pick_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(select_azk01_097_mask_host):
      st_t, rw_t, tm_t, tr_t = self._select_azk01_097_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(select_azk01_097_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(selection_pick_noop_mask_host):
      st_t, rw_t, tm_t, tr_t = self._selection_pick_noop_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(selection_pick_noop_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    for placement_zone, select_mask_host in (
        (self._zone_garden, select_azk01_122_garden_mask_host),
        (self._zone_alley, select_azk01_122_alley_mask_host),
    ):
      if np.any(select_mask_host):
        st_t, rw_t, tm_t, tr_t = self._select_azk01_122_place_fns[
            placement_zone
        ](
            self._states,
            acts_dev,
            self._terms,
            self._truncs,
            self._pending[5],
        )
        mask = jnp.asarray(select_mask_host)
        states_acc = merge_tree(states_acc, st_t, mask)
        row_mask = mask[:, None]
        rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
        terms_acc = jnp.where(row_mask, tm_t, terms_acc)
        truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(bottom_deck_card_mask_host):
      st_t, rw_t, tm_t, tr_t = self._bottom_deck_card_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(bottom_deck_card_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(bottom_deck_all_mask_host):
      st_t, rw_t, tm_t, tr_t = self._bottom_deck_all_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(bottom_deck_all_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(attach_weapon_simple_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attach_weapon_simple_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attach_weapon_simple_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(attach_stt01_013_confirm_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attach_stt01_013_confirm_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attach_stt01_013_confirm_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(declare_defender_mask_host):
      st_t, rw_t, tm_t, tr_t = self._declare_defender_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(declare_defender_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(attack_azk01_004_leader_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attack_azk01_004_leader_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attack_azk01_004_leader_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(attack_leader_simple_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attack_leader_simple_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attack_leader_simple_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(attack_azk01_060_confirm_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attack_azk01_060_confirm_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attack_azk01_060_confirm_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(attack_stt01_006_effect_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attack_stt01_006_effect_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attack_stt01_006_effect_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    attack_response_mask_host = (
        attack_leader_response_mask_host | attack_entity_response_mask_host
    )
    if np.any(attack_response_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attack_leader_response_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attack_response_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(response_noop_leader_combat_mask_host):
      st_t, rw_t, tm_t, tr_t = self._response_noop_leader_combat_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(response_noop_leader_combat_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(response_noop_entity_combat_mask_host):
      st_t, rw_t, tm_t, tr_t = self._response_noop_entity_combat_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(response_noop_entity_combat_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(response_noop_azk01_040_mask_host):
      st_t, rw_t, tm_t, tr_t = self._response_noop_azk01_040_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(response_noop_azk01_040_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(attack_entity_simple_mask_host):
      st_t, rw_t, tm_t, tr_t = self._attack_entity_mutual_destroy_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(attack_entity_simple_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(main_noop_simple_mask_host):
      st_t, rw_t, tm_t, tr_t = self._main_noop_simple_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(main_noop_simple_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(main_noop_azk01_011_mask_host):
      st_t, rw_t, tm_t, tr_t = self._main_noop_azk01_011_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(main_noop_azk01_011_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(main_noop_stt04_003_mask_host):
      st_t, rw_t, tm_t, tr_t = self._main_noop_stt04_003_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(main_noop_stt04_003_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    if np.any(main_noop_mask_host):
      st_t, rw_t, tm_t, tr_t = self._main_noop_step_fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(main_noop_mask_host)
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    remaining = ~(
        zero_legal_mask_host
        | pregame_mask_host
        | play_garden_simple_mask_host
        | play_alley_simple_mask_host
        | play_azk01_007_garden_mask_host
        | play_azk01_007_alley_mask_host
        | play_stt01_007_garden_mask_host
        | play_stt01_007_alley_mask_host
        | play_azk01_003_garden_mask_host
        | play_azk01_003_alley_mask_host
        | play_azk01_033_garden_mask_host
        | play_azk01_033_alley_mask_host
        | play_azk01_045_garden_mask_host
        | play_azk01_045_alley_mask_host
        | play_azk01_056_garden_mask_host
        | play_azk01_056_alley_mask_host
        | play_azk01_097_garden_mask_host
        | play_azk01_097_alley_mask_host
        | play_stt02_003_garden_mask_host
        | play_stt02_003_alley_mask_host
        | play_stt02_013_garden_mask_host
        | play_stt02_013_alley_mask_host
        | play_stt02_009_garden_mask_host
        | play_stt02_009_alley_mask_host
        | gate_portal_simple_mask_host
        | play_spell_stt04_016_mask_host
        | play_spell_stt02_016_mask_host
        | play_spell_stt01_017_mask_host
        | play_spell_azk01_032_mask_host
        | play_spell_azk01_002_mask_host
        | play_spell_azk01_009_mask_host
        | play_spell_azk01_065_mask_host
        | play_spell_azk01_127_mask_host
        | play_spell_stt03_016_mask_host
        | select_cost_stt04_016_mask_host
        | select_cost_stt01_007_mask_host
        | select_cost_stt01_004_mask_host
        | select_cost_stt02_009_mask_host
        | select_cost_azk01_032_mask_host
        | select_cost_stt02_016_mask_host
        | effect_stt04_016_mask_host
        | effect_stt04_001_mask_host
        | effect_stt04_004_mask_host
        | effect_stt01_005_mask_host
        | effect_stt01_006_mask_host
        | effect_stt01_014_mask_host
        | effect_stt01_017_mask_host
        | effect_stt02_001_mask_host
        | effect_stt02_011_mask_host
        | effect_azk01_105_mask_host
        | effect_stt02_009_mask_host
        | effect_azk01_032_mask_host
        | effect_stt02_016_mask_host
        | effect_stt03_002_mask_host
        | effect_stt03_006_mask_host
        | confirm_clear_mask_host
        | confirm_stt01_002_mask_host
        | confirm_stt01_007_mask_host
        | confirm_stt01_013_mask_host
        | confirm_stt01_004_mask_host
        | confirm_stt02_009_mask_host
        | confirm_stt04_004_mask_host
        | confirm_azk01_058_mask_host
        | confirm_azk01_060_mask_host
        | confirm_azk01_060_response_mask_host
        | effect_azk01_007_mask_host
        | effect_azk01_009_mask_host
        | effect_azk01_040_mask_host
        | effect_azk01_058_mask_host
        | effect_azk01_059_mask_host
        | effect_azk01_070_mask_host
        | effect_azk01_065_mask_host
        | effect_azk01_127_mask_host
        | activate_stt04_001_mask_host
        | activate_stt02_001_mask_host
        | activate_stt02_011_mask_host
        | activate_azk01_121_mask_host
        | activate_stt03_001_mask_host
        | activate_stt01_005_mask_host
        | activate_azk01_070_mask_host
        | activate_azk01_105_mask_host
        | select_azk01_003_pick_mask_host
        | select_azk01_033_pick_mask_host
        | select_azk01_045_pick_mask_host
        | select_azk01_056_pick_mask_host
        | select_stt02_003_pick_mask_host
        | select_stt02_013_pick_mask_host
        | select_stt01_004_pick_mask_host
        | select_stt01_002_equip_mask_host
        | select_azk01_126_pick_mask_host
        | select_azk01_097_mask_host
        | selection_pick_noop_mask_host
        | select_azk01_122_garden_mask_host
        | select_azk01_122_alley_mask_host
        | bottom_deck_card_mask_host
        | bottom_deck_all_mask_host
        | attach_weapon_simple_mask_host
        | attach_stt01_013_confirm_mask_host
        | declare_defender_mask_host
        | attack_azk01_060_confirm_mask_host
        | attack_stt01_006_effect_mask_host
        | attack_azk01_004_leader_mask_host
        | attack_leader_response_mask_host
        | attack_entity_response_mask_host
        | attack_entity_simple_mask_host
        | attack_leader_simple_mask_host
        | response_noop_leader_combat_mask_host
        | response_noop_entity_combat_mask_host
        | response_noop_azk01_040_mask_host
        | main_noop_simple_mask_host
        | main_noop_azk01_011_mask_host
        | main_noop_stt04_003_mask_host
        | main_noop_mask_host
    )
    action_types = [int(x) for x in np.unique(chosen[remaining])]

    for action_type in action_types:
      if action_type < 0 or action_type >= 26:
        raise ValueError(f"JAX split-step received invalid action type {action_type}")
      fn = self._get_step_type_fn(action_type)
      st_t, rw_t, tm_t, tr_t = fn(
          self._states,
          acts_dev,
          self._terms,
          self._truncs,
          self._pending[5],
      )
      mask = jnp.asarray(remaining & (chosen == action_type))
      states_acc = merge_tree(states_acc, st_t, mask)
      row_mask = mask[:, None]
      rewards_acc = jnp.where(row_mask, rw_t, rewards_acc)
      terms_acc = jnp.where(row_mask, tm_t, terms_acc)
      truncs_acc = jnp.where(row_mask, tr_t, truncs_acc)

    return states_acc, rewards_acc, terms_acc, truncs_acc

  def _timing_present(self, zone_host, def_host, players, timing_table):
    rows = np.arange(self.num_environments)
    z = zone_host[rows, players]
    ids = def_host[rows, players]
    in_play = (z == 4) | (z == 5) | (z == 2)  # garden, alley, leader
    safe_ids = np.maximum(ids, 0)
    valid = ids >= 0
    timed = timing_table[safe_ids]
    return np.any(in_play & valid & timed, axis=1)

  def _timing_present_any_player(self, zone_host, def_host, timing_table):
    z = zone_host
    ids = def_host
    in_play = (z == 4) | (z == 5) | (z == 2)  # garden, alley, leader
    safe_ids = np.maximum(ids, 0)
    valid = ids >= 0
    timed = timing_table[safe_ids]
    return np.any(in_play & valid & timed, axis=(1, 2))

  def _main_noop_fast_mask(self, chosen, phase):
    states = self._states
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    next_player = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)

    no_eot_timing = ~self._timing_present(
        zone_host, def_host, active, self._timing_eot
    )
    no_start_timing = ~self._timing_present(
        zone_host, def_host, next_player, self._timing_start
    )
    no_start_each_timing = ~self._timing_present_any_player(
        zone_host, def_host, self._timing_start_each
    )

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )

    return (
        (phase == 2)  # Phase.MAIN
        & (chosen == 0)  # Act.NOOP
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
        & no_eot_timing
        & no_start_timing
        & no_start_each_timing
    )

  def _main_noop_azk01_011_fast_mask(self, chosen, phase):
    states = self._states
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    next_player = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)

    no_start_timing = ~self._timing_present(
        zone_host, def_host, next_player, self._timing_start
    )
    no_start_each_timing = ~self._timing_present_any_player(
        zone_host, def_host, self._timing_start_each
    )
    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == 0)  # Act.NOOP
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
        & no_start_timing
        & no_start_each_timing
    )
    if not np.any(base):
      return base

    rows = np.arange(self.num_environments)
    active_zone = zone_host[rows, active]
    active_defs = def_host[rows, active]
    active_valid = active_defs >= 0
    active_safe_defs = np.maximum(active_defs, 0)
    active_in_play = (
        (active_zone == self._zone_garden)
        | (active_zone == self._zone_alley)
        | (active_zone == self._zone_leader)
    )
    active_eot = (
        active_in_play
        & active_valid
        & self._timing_eot[active_safe_defs]
    )
    azk01_011_eot = (
        (active_zone == self._zone_garden)
        & (active_defs == self._azk01_011_id)
    )
    only_single_azk01_011 = (
        (np.sum(active_eot, axis=1) == 1)
        & (np.sum(azk01_011_eot, axis=1) == 1)
    )

    no_attached = ~np.any(zone_host == self._zone_attached, axis=(1, 2))
    no_sacrifice = ~np.any(np.asarray(states.sacrifice_eot), axis=(1, 2))

    token_zone = zone_host[:, :, self._token_instance]
    token_tapped = np.asarray(states.tapped)[:, :, self._token_instance]
    token_expires = np.asarray(states.ikz_token_expires_eot)
    no_token_cleanup = ~np.any(
        (token_zone == self._zone_token) & (token_tapped | token_expires),
        axis=1,
    )

    eot_fields = (
        states.atk_buff_eot,
        states.hp_buff_eot,
        states.cmb_in_eot,
        states.cmb_out_eot,
        states.carapace_eot,
    )
    no_eot_modifiers = ~np.any(
        np.stack([np.asarray(field) != 0 for field in eot_fields], axis=0),
        axis=(0, 2, 3),
    )

    no_positive_start_status = ~np.any(
        (np.asarray(states.frozen_dur) > 0)
        | (np.asarray(states.effect_immune_dur) > 0)
        | (np.asarray(states.shocked_dur) != 0),
        axis=(1, 2),
    )
    no_timed_grants = ~np.any(
        (np.asarray(states.timed_tag) != 0)
        & (np.asarray(states.timed_phase) == 1),  # GRANT_PHASE_START
        axis=(1, 2, 3),
    )

    active_untap_zone = (
        (active_zone == self._zone_garden)
        | (active_zone == self._zone_alley)
        | (active_zone == self._zone_ikz_area)
        | (active_zone == self._zone_leader)
        | (active_zone == self._zone_gate)
    )
    no_force_tapped = ~np.any(
        active_untap_zone
        & active_valid
        & self._force_tapped[active_safe_defs],
        axis=1,
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    inert_watch = def_host == self._stt01_008_id
    no_passive_watch = ~np.any(
        board & np.isin(def_host, self._simple_play_watch_ids) & ~inert_watch,
        axis=(1, 2),
    )
    azk01_011_destroy_clean = (
        ~self._timing_when_destroyed[self._azk01_011_id]
        & ~self._inherent_godmode[self._azk01_011_id]
    )

    return (
        base
        & only_single_azk01_011
        & no_attached
        & no_sacrifice
        & no_token_cleanup
        & no_eot_modifiers
        & no_positive_start_status
        & no_timed_grants
        & no_force_tapped
        & no_passive_watch
        & azk01_011_destroy_clean
    )

  def _main_noop_stt04_003_fast_mask(self, chosen, phase):
    states = self._states
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    next_player = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    rows = np.arange(self.num_environments)

    no_eot_timing = ~self._timing_present(
        zone_host, def_host, active, self._timing_eot
    )
    no_start_timing = ~self._timing_present(
        zone_host, def_host, next_player, self._timing_start
    )
    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == 0)  # Act.NOOP
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
        & no_eot_timing
        & no_start_timing
    )
    if not np.any(base):
      return base

    in_play = (
        (zone_host == self._zone_garden)
        | (zone_host == self._zone_alley)
        | (zone_host == self._zone_leader)
    )
    valid = def_host >= 0
    safe_defs = np.maximum(def_host, 0)
    start_each = in_play & valid & self._timing_start_each[safe_defs]
    seer = start_each & (def_host == self._stt04_003_id)
    start_each_count = np.sum(start_each, axis=(1, 2))
    seer_count = np.sum(seer, axis=(1, 2))
    one_or_two_clean_seers = (
        (start_each_count == seer_count)
        & (seer_count >= 1)
        & (seer_count <= 2)
    )
    seer_damage_ok = ~np.any(seer & (np.asarray(states.cur_hp) <= 0), axis=(1, 2))
    seer_bad_state = (
        seer
        & (
            (np.asarray(states.cmb_in_perm) != 0)
            | (np.asarray(states.cmb_in_eot) != 0)
            | (np.asarray(states.cmb_out_perm) != 0)
            | (np.asarray(states.cmb_out_eot) != 0)
            | (np.asarray(states.carapace_perm) != 0)
            | (np.asarray(states.carapace_eot) != 0)
            | (np.asarray(states.effect_immune_dur) != 0)
            | np.asarray(states.grant_godmode)
        )
    )
    seer_clean = (
        ~np.any(seer_bad_state, axis=(1, 2))
        & ~self._timing_takes_damage[self._stt04_003_id]
        & ~self._timing_deals_damage[self._stt04_003_id]
        & ~self._timing_when_destroyed[self._stt04_003_id]
    )

    no_attached = ~np.any(zone_host == self._zone_attached, axis=(1, 2))
    sacrifice = np.asarray(states.sacrifice_eot)
    sacrifice_ok = ~np.any(
        sacrifice
        & (
            (def_host != self._azk01_060_id)
            | (
                (zone_host != self._zone_garden)
                & (zone_host != self._zone_alley)
            )
        ),
        axis=(1, 2),
    )

    token_zone = zone_host[:, :, self._token_instance]
    token_tapped = np.asarray(states.tapped)[:, :, self._token_instance]
    token_expires = np.asarray(states.ikz_token_expires_eot)
    no_token_cleanup = ~np.any(
        (token_zone == self._zone_token) & (token_tapped | token_expires),
        axis=1,
    )

    atk_eot = np.asarray(states.atk_buff_eot) != 0
    atk_eot_zones = (zone_host == self._zone_garden) | (
        zone_host == self._zone_leader
    )
    atk_eot_ok = ~np.any(atk_eot & ~atk_eot_zones, axis=(1, 2))
    other_eot_fields = (
        states.hp_buff_eot,
        states.cmb_in_eot,
        states.cmb_out_eot,
        states.carapace_eot,
    )
    no_other_eot_modifiers = ~np.any(
        np.stack([np.asarray(field) != 0 for field in other_eot_fields], axis=0),
        axis=(0, 2, 3),
    )

    no_positive_start_status = ~np.any(
        (np.asarray(states.frozen_dur) > 0)
        | (np.asarray(states.effect_immune_dur) > 0)
        | (np.asarray(states.shocked_dur) != 0),
        axis=(1, 2),
    )
    no_start_timed_grants = ~np.any(
        (np.asarray(states.timed_tag) != 0)
        & (np.asarray(states.timed_phase) == 1),  # GRANT_PHASE_START
        axis=(1, 2, 3),
    )

    active_zone = zone_host[rows, active]
    active_defs = def_host[rows, active]
    active_valid = active_defs >= 0
    active_safe_defs = np.maximum(active_defs, 0)
    active_untap_zone = (
        (active_zone == self._zone_garden)
        | (active_zone == self._zone_alley)
        | (active_zone == self._zone_ikz_area)
        | (active_zone == self._zone_leader)
        | (active_zone == self._zone_gate)
    )
    no_force_tapped = ~np.any(
        active_untap_zone
        & active_valid
        & self._force_tapped[active_safe_defs],
        axis=1,
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    inert_watch = def_host == self._stt01_008_id
    no_passive_watch = ~np.any(
        board & np.isin(def_host, self._simple_play_watch_ids) & ~inert_watch,
        axis=(1, 2),
    )

    return (
        base
        & one_or_two_clean_seers
        & seer_damage_ok
        & seer_clean
        & no_attached
        & sacrifice_ok
        & no_token_cleanup
        & atk_eot_ok
        & no_other_eot_modifiers
        & no_positive_start_status
        & no_start_timed_grants
        & no_force_tapped
        & no_passive_watch
    )

  def _play_azk01_003_reveal_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    attached_row = attached_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_match = (zone_row == placement_zone) & (zpos_row == slot[:, None])
    slot_occupied = np.any(slot_match, axis=1)
    slot_inst = np.argmax(slot_match, axis=1)
    zone_full = np.sum(zone_row == placement_zone, axis=1) >= self._garden_size
    slot_ok = ~slot_occupied | zone_full
    displaced_def = def_row[rows, slot_inst]
    displaced_safe_def = np.maximum(displaced_def, 0)
    displaced_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == slot_inst[:, None].astype(attached_row.dtype)),
        axis=1,
    )
    attached_to_displaced = (
        (zone_row == self._zone_attached)
        & (attached_row == slot_inst[:, None].astype(attached_row.dtype))
    )
    attached_defs = def_row
    attached_safe_defs = np.maximum(attached_defs, 0)
    attached_problem = attached_to_displaced & (
        self._timing_when_destroyed[attached_safe_defs]
        | np.isin(attached_defs, self._simple_attach_watch_ids)
    )
    attached_simple = ~np.any(attached_problem, axis=1)
    displaced_simple = (
        ~slot_occupied
        | (
            (displaced_def >= 0)
            & ~self._timing_when_destroyed[displaced_safe_def]
            & ~self._inherent_godmode[displaced_safe_def]
            & (~displaced_has_attached | attached_simple)
        )
    )

    board = (zone_row == self._zone_garden) | (zone_row == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_row, self._simple_play_watch_ids)
        & (def_row != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=1)

    return (
        base
        & hand_exists
        & (def_id == self._azk01_003_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_ok
        & displaced_simple
        & no_passive_watch
    )

  def _play_azk01_007_effect_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    attached_row = attached_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_match = (zone_row == placement_zone) & (zpos_row == slot[:, None])
    slot_occupied = np.any(slot_match, axis=1)
    slot_inst = np.argmax(slot_match, axis=1)
    zone_count = np.sum(zone_row == placement_zone, axis=1)
    zone_full = zone_count >= self._garden_size
    slot_ok = ~slot_occupied | zone_full
    displaced_def = def_row[rows, slot_inst]
    displaced_safe_def = np.maximum(displaced_def, 0)
    displaced_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == slot_inst[:, None].astype(attached_row.dtype)),
        axis=1,
    )
    displaced_simple = (
        ~slot_occupied
        | (
            (displaced_def >= 0)
            & ~self._timing_when_destroyed[displaced_safe_def]
            & ~self._inherent_godmode[displaced_safe_def]
            & ~displaced_has_attached
        )
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._azk01_007_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_ok
        & displaced_simple
        & no_passive_watch
    )

  def _play_stt01_007_confirm_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, active_safe]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    attached_row = attached_host[rows, active_safe]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_match = (zone_row == placement_zone) & (zpos_row == slot[:, None])
    slot_occupied = np.any(slot_match, axis=1)
    slot_inst = np.argmax(slot_match, axis=1)
    zone_count = np.sum(zone_row == placement_zone, axis=1)
    zone_full = zone_count >= self._garden_size
    slot_ok = ~slot_occupied | zone_full
    displaced_def = def_row[rows, slot_inst]
    displaced_safe_def = np.maximum(displaced_def, 0)
    displaced_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == slot_inst[:, None].astype(attached_row.dtype)),
        axis=1,
    )
    displaced_simple = (
        ~slot_occupied
        | (
            (displaced_def >= 0)
            & ~self._timing_when_destroyed[displaced_safe_def]
            & ~self._inherent_godmode[displaced_safe_def]
            & ~displaced_has_attached
        )
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._stt01_007_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_ok
        & displaced_simple
        & no_passive_watch
    )

  def _play_azk01_097_reveal_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_empty = ~np.any(
        (zone_row == placement_zone) & (zpos_row == slot[:, None]),
        axis=1,
    )
    not_full = np.sum(zone_row == placement_zone, axis=1) < self._garden_size

    deck_count = np.sum(zone_row == self._zone_deck, axis=1)
    deck_sort_key = np.where(
        zone_row == self._zone_deck,
        -zpos_row.astype(np.int16, copy=False),
        1_000_000,
    )
    top5_inst = np.argsort(deck_sort_key, axis=1)[:, :5]
    top5_defs = np.take_along_axis(def_row, top5_inst, axis=1)
    top5_has_weapon = np.any(
        self._card_type[np.maximum(top5_defs, 0)] == self._card_type_weapon,
        axis=1,
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._azk01_097_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_empty
        & not_full
        & (deck_count >= 5)
        & top5_has_weapon
        & no_passive_watch
    )

  def _play_stt02_003_reveal_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_empty = ~np.any(
        (zone_row == placement_zone) & (zpos_row == slot[:, None]),
        axis=1,
    )
    not_full = np.sum(zone_row == placement_zone, axis=1) < self._garden_size
    deck_count = np.sum(zone_row == self._zone_deck, axis=1)

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._stt02_003_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_empty
        & not_full
        & (deck_count > 0)
        & no_passive_watch
    )

  def _play_stt02_013_reveal_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_empty = ~np.any(
        (zone_row == placement_zone) & (zpos_row == slot[:, None]),
        axis=1,
    )
    not_full = np.sum(zone_row == placement_zone, axis=1) < self._garden_size
    deck_count = np.sum(zone_row == self._zone_deck, axis=1)

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._stt02_013_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_empty
        & not_full
        & (deck_count >= 3)
        & no_passive_watch
    )

  def _play_stt02_009_confirm_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_empty = ~np.any(
        (zone_row == placement_zone) & (zpos_row == slot[:, None]),
        axis=1,
    )
    not_full = np.sum(zone_row == placement_zone, axis=1) < self._garden_size

    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active
    ].astype(np.int32, copy=False)
    cost = np.maximum(
        self._ikz_cost[safe_def].astype(np.int32) - next_reduction,
        0,
    )
    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    garden_cost_target = (
        (zone_row == self._zone_garden)
        & (def_row >= 0)
        & (self._card_type[safe_row_defs] == self._card_type_entity)
        & self._has_ikz_cost[safe_row_defs]
        & (self._ikz_cost[safe_row_defs].astype(np.int32) >= 2)
    )
    played_can_pay_cost = (
        (placement_zone == self._zone_garden)
        & (def_id == self._stt02_009_id)
        & self._has_ikz_cost[safe_def]
        & (self._ikz_cost[safe_def].astype(np.int32) >= 2)
    )
    cost_available = np.any(garden_cost_target, axis=1) | played_can_pay_cost

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._stt02_009_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_empty
        & not_full
        & can_pay
        & cost_available
        & no_passive_watch
    )

  def _play_azk01_033_reveal_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_empty = ~np.any(
        (zone_row == placement_zone) & (zpos_row == slot[:, None]),
        axis=1,
    )
    not_full = np.sum(zone_row == placement_zone, axis=1) < self._garden_size
    deck_count = np.sum(zone_row == self._zone_deck, axis=1)

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._azk01_033_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_empty
        & not_full
        & (deck_count > 0)
        & no_passive_watch
    )

  def _play_azk01_045_reveal_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_empty = ~np.any(
        (zone_row == placement_zone) & (zpos_row == slot[:, None]),
        axis=1,
    )
    not_full = np.sum(zone_row == placement_zone, axis=1) < self._garden_size
    deck_count = np.sum(zone_row == self._zone_deck, axis=1)

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._azk01_045_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_empty
        & not_full
        & (deck_count > 0)
        & no_passive_watch
    )

  def _play_azk01_056_reveal_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)

    slot_empty = ~np.any(
        (zone_row == placement_zone) & (zpos_row == slot[:, None]),
        axis=1,
    )
    not_full = np.sum(zone_row == placement_zone, axis=1) < self._garden_size
    deck_count = np.sum(zone_row == self._zone_deck, axis=1)

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id)
    )
    no_passive_watch = ~np.any(watched_on_board, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == action_type)
        & hand_exists
        & (def_id == self._azk01_056_id)
        & (self._card_type[safe_def] == self._card_type_entity)
        & slot_empty
        & not_full
        & (deck_count > 0)
        & no_passive_watch
    )

  def _play_spell_stt04_016_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    friendly_garden_entity = (
        (zone_row == self._zone_garden)
        & (def_row >= 0)
        & (self._card_type[safe_row_defs] == self._card_type_entity)
    )
    cost_target_available = np.any(friendly_garden_entity, axis=1)

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._stt04_016_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & can_pay
        & cost_target_available
    )

  def _play_spell_stt02_016_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    opp = (active_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, active_safe]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    tapped_row = tapped_host[rows, active_safe]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active_safe
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    inst_cols = np.arange(def_row.shape[1], dtype=np.int32)
    other_hand_card = np.any(
        (zone_row == self._zone_hand) & (inst_cols[None, :] != hand_inst[:, None]),
        axis=1,
    )
    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    target_available = np.any(opp_zone == self._zone_leader, axis=1) | np.any(
        (opp_zone == self._zone_garden)
        & (opp_defs >= 0)
        & (self._card_type[opp_safe_defs] == self._card_type_entity),
        axis=1,
    )

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._stt02_016_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & can_pay
        & other_hand_card
        & target_available
    )

  def _play_spell_stt01_017_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    opp = (active_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, active_safe]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    tapped_row = tapped_host[rows, active_safe]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active_safe
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    enemy_garden_entity = (
        (opp_zone == self._zone_garden)
        & (opp_defs >= 0)
        & (self._card_type[opp_safe_defs] == self._card_type_entity)
    )

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._stt01_017_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & self._timing_is_response[safe_def]
        & self._has_ability[safe_def]
        & can_pay
        & np.any(enemy_garden_entity, axis=1)
    )

  def _play_spell_azk01_032_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    friendly_cost_target = (
        (zone_row == self._zone_garden)
        & (def_row >= 0)
        & (self._card_type[safe_row_defs] == self._card_type_entity)
        & self._has_ikz_cost[safe_row_defs]
        & (self._ikz_cost[safe_row_defs].astype(np.int32) >= 2)
    )
    cost_target_available = np.any(friendly_cost_target, axis=1)

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._azk01_032_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & can_pay
        & cost_target_available
    )

  def _play_spell_azk01_002_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_alive = leader_exists & (
        np.asarray(states.cur_hp)[rows, active, leader_inst] > 0
    )

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._azk01_002_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & can_pay
        & leader_alive
    )

  def _play_spell_azk01_009_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, active_safe]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    tapped_row = tapped_host[rows, active_safe]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active_safe
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    safe_all_defs = np.maximum(def_host, 0)
    valid_target = (
        (zone_host == self._zone_garden)
        & (def_host >= 0)
        & (self._card_type[safe_all_defs] == self._card_type_entity)
        & (self._ikz_cost[safe_all_defs].astype(np.int32) <= 4)
    )
    target_available = np.any(valid_target, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._azk01_009_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & can_pay
        & target_available
    )

  def _play_spell_azk01_065_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_safe_def = np.maximum(leader_def, 0)
    leader_clean = (
        leader_exists
        & (np.asarray(states.cur_hp)[rows, active, leader_inst] > 3)
        & ~self._timing_takes_damage[leader_safe_def]
        & ~self._inherent_godmode[leader_safe_def]
        & (np.asarray(states.carapace_perm)[rows, active, leader_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, active, leader_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active, leader_inst]
        & (np.asarray(states.effect_immune_dur)[rows, active, leader_inst] == 0)
    )

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._azk01_065_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & can_pay
        & leader_clean
    )

  def _play_spell_azk01_127_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    enemy_garden_entity = (
        (opp_zone == self._zone_garden)
        & (opp_defs >= 0)
        & (self._card_type[opp_safe_defs] == self._card_type_entity)
    )

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._azk01_127_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & self._timing_is_response[safe_def]
        & self._has_ability[safe_def]
        & can_pay
        & np.any(enemy_garden_entity, axis=1)
    )

  def _play_spell_stt03_016_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    opp = (active_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_play_spell)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, active_safe]
    hand_index = active_acts[:, 1]
    ability_index = active_acts[:, 2]
    use_token = active_acts[:, 3] != 0
    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    tapped_row = tapped_host[rows, active_safe]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    safe_def = np.maximum(def_id, 0)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, active_safe
    ].astype(np.int32, copy=False)
    cost = np.maximum(self._ikz_cost[safe_def].astype(np.int32) - next_reduction, 0)

    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    can_pay = token_needed_ok & (payment_sources >= cost)

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    marked = (
        (opp_zone == self._zone_garden)
        & (opp_defs >= 0)
        & (np.asarray(states.cur_hp)[rows, opp] <= 2)
    )
    marked_entity = marked & (
        self._card_type[opp_safe_defs] == self._card_type_entity
    )
    no_destroy_triggers = ~np.any(
        marked_entity & self._timing_when_destroyed[opp_safe_defs],
        axis=1,
    )
    no_godmode = ~np.any(
        marked_entity
        & (
            self._inherent_godmode[opp_safe_defs]
            | np.asarray(states.grant_godmode)[rows, opp]
        ),
        axis=1,
    )
    no_attached_opp = ~np.any(opp_zone == self._zone_attached, axis=1)
    no_passive_death_watch = (
        ~np.any(np.asarray(states.passive_observer_registered), axis=(1, 2))
        & ~np.any(
            (zone_host == self._zone_garden)
            & (def_host == self._stt02_012_id),
            axis=(1, 2),
        )
    )

    return (
        base
        & (active_acts[:, 0] == self._act_play_spell)
        & (ability_index == 0)
        & hand_exists
        & (def_id == self._stt03_016_id)
        & (self._card_type[safe_def] == self._card_type_spell)
        & can_pay
        & np.any(marked_entity, axis=1)
        & no_destroy_triggers
        & no_godmode
        & no_attached_opp
        & no_passive_death_watch
    )

  def _select_cost_stt04_016_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt04_016_id)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & ~np.asarray(states.ab_costs_applied)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    target_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_protected = (
        self._inherent_godmode[target_safe_def]
        | (np.asarray(states.carapace_perm)[rows, owner_safe, target_inst] != 0)
        | (np.asarray(states.carapace_eot)[rows, owner_safe, target_inst] != 0)
        | np.asarray(states.grant_godmode)[rows, owner_safe, target_inst]
        | (np.asarray(states.effect_immune_dur)[rows, owner_safe, target_inst] != 0)
    )
    simple_stt04_003 = (
        target_exists
        & (target_def == self._stt04_003_id)
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~target_protected
    )
    friendly_garden_entity = (
        (zone_row == self._zone_garden)
        & (def_row >= 0)
        & (self._card_type[np.maximum(def_row, 0)] == self._card_type_entity)
    )
    inst_cols = np.arange(def_row.shape[1], dtype=np.int32)
    has_other_garden_entity = np.any(
        friendly_garden_entity & (inst_cols[None, :] != target_inst[:, None]),
        axis=1,
    )
    azk01_059_trigger = (
        target_exists
        & (target_def == self._azk01_059_id)
        & (np.asarray(states.cur_hp)[rows, owner_safe, target_inst] > 1)
        & ((np.asarray(states.once_per_turn_used)[rows, owner_safe, target_inst] & 1) == 0)
        & has_other_garden_entity
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~target_protected
    )
    target_clean = simple_stt04_003 | azk01_059_trigger

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
    )
    return (
        (chosen == self._act_select_cost)
        & (active_acts[:, 0] == self._act_select_cost)
        & (np.asarray(states.ab_phase) == self._ability_cost_selection)
        & source_ok
        & target_clean
        & clean
    )

  def _select_cost_stt01_007_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_007_id)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & ~np.asarray(states.ab_costs_applied)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    target_match = (
        (zone_row == self._zone_hand)
        & (zpos_row == target_index[:, None])
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_ok = target_exists & (target_inst != src)

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_cost)
        & (active_acts[:, 0] == self._act_select_cost)
        & (np.asarray(states.ab_phase) == self._ability_cost_selection)
        & source_ok
        & target_ok
        & clean
    )

  def _select_cost_stt02_009_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_009_id)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & ~np.asarray(states.ab_costs_applied)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    target_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_host[rows, owner_safe] == target_inst[:, None]),
        axis=1,
    )
    target_clean = (
        target_exists
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) >= 2)
        & ~target_has_attached
        & ~self._timing_when_returned[target_safe_def]
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    no_stt02_010_observer = ~np.any(
        (zone_host == self._zone_garden) & (def_host == self._stt02_010_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
        & no_stt02_010_observer
    )
    return (
        (chosen == self._act_select_cost)
        & (active_acts[:, 0] == self._act_select_cost)
        & (np.asarray(states.ab_phase) == self._ability_cost_selection)
        & source_ok
        & target_clean
        & clean
    )

  def _select_cost_azk01_032_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_032_id)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & ~np.asarray(states.ab_costs_applied)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    target_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_host[rows, owner_safe] == target_inst[:, None]),
        axis=1,
    )
    target_clean = (
        target_exists
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) >= 2)
        & ~target_has_attached
        & ~self._timing_when_returned[target_safe_def]
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    no_stt02_010_observer = ~np.any(
        (zone_host == self._zone_garden) & (def_host == self._stt02_010_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
        & no_stt02_010_observer
    )
    return (
        (chosen == self._act_select_cost)
        & (active_acts[:, 0] == self._act_select_cost)
        & (np.asarray(states.ab_phase) == self._ability_cost_selection)
        & source_ok
        & target_clean
        & clean
    )

  def _select_cost_stt02_016_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_016_id)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & ~np.asarray(states.ab_costs_applied)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    target_match = (
        (zone_row == self._zone_hand)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] >= 0)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_ok = target_exists & (target_inst != src_safe)

    clean = (
        (np.asarray(states.phase) == 3)  # Phase.RESPONSE_WINDOW
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_cost)
        & (active_acts[:, 0] == self._act_select_cost)
        & (np.asarray(states.ab_phase) == self._ability_cost_selection)
        & source_ok
        & target_ok
        & clean
    )

  def _select_cost_stt01_004_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_004_id)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & ~np.asarray(states.ab_costs_applied)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    target_match = (zone_row == self._zone_hand) & (
        zpos_row == target_index[:, None]
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_ok = (
        target_exists
        & (self._card_type[target_safe_def] == self._card_type_weapon)
    )
    deck_count = np.sum(zone_row == self._zone_deck, axis=1)
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_cost)
        & (active_acts[:, 0] == self._act_select_cost)
        & (np.asarray(states.ab_phase) == self._ability_cost_selection)
        & source_ok
        & target_ok
        & (deck_count > 0)
        & clean
    )

  def _effect_stt04_016_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt04_016_id)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
        & np.asarray(states.ab_costs_applied)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, opp]
    zpos_row = zpos_host[rows, opp]
    def_row = def_host[rows, opp]
    attached_row = attached_host[rows, opp]
    target_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_is_entity = self._card_type[target_safe_def] == self._card_type_entity
    target_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == target_inst[:, None].astype(attached_row.dtype)),
        axis=1,
    )
    target_clean = (
        target_exists
        & target_is_entity
        & ~target_has_attached
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, opp, target_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, target_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, opp, target_inst]
        & (np.asarray(states.effect_immune_dur)[rows, opp, target_inst] == 0)
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    trig_count = np.asarray(states.trig_count).astype(np.int32, copy=False)
    trig_owner = np.asarray(states.trig_owner)[:, 0].astype(np.int32, copy=False)
    trig_source = np.asarray(states.trig_source)[:, 0].astype(np.int32, copy=False)
    trig_timing = np.asarray(states.trig_timing)[:, 0].astype(np.int32, copy=False)
    trig_source_safe = np.maximum(trig_source, 0)
    queued_azk01_059 = (
        (trig_count == 1)
        & (trig_owner == owner)
        & (trig_source >= 0)
        & (trig_timing == 11)  # TIMING_WHEN_TAKES_DAMAGE
        & (def_host[rows, owner_safe, trig_source_safe] == self._azk01_059_id)
        & (
            (np.asarray(states.once_per_turn_used)[
                rows, owner_safe, trig_source_safe
            ] & 1) == 0
        )
    )
    trigger_ok = (trig_count == 0) | queued_azk01_059
    clean = (
        trigger_ok
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_clean
        & clean
    )

  def _effect_stt02_009_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_009_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, opp]
    zpos_row = zpos_host[rows, opp]
    def_row = def_host[rows, opp]
    target_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_host[rows, opp] == target_inst[:, None]),
        axis=1,
    )
    target_clean = (
        target_exists
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) <= 2)
        & ~target_has_attached
        & ~self._timing_when_returned[target_safe_def]
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    no_stt02_010_observer = ~np.any(
        (zone_host == self._zone_garden) & (def_host == self._stt02_010_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
        & no_stt02_010_observer
    )
    select = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & target_clean
    )
    skip = (
        (chosen == self._act_noop)
        & (active_acts[:, 0] == self._act_noop)
    )
    return (
        (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & clean
        & (select | skip)
    )

  def _effect_azk01_032_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_032_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, opp]
    zpos_row = zpos_host[rows, opp]
    def_row = def_host[rows, opp]
    target_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_host[rows, opp] == target_inst[:, None]),
        axis=1,
    )
    target_clean = (
        target_exists
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) <= 4)
        & ~target_has_attached
        & ~self._timing_when_returned[target_safe_def]
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    no_stt02_010_observer = ~np.any(
        (zone_host == self._zone_garden) & (def_host == self._stt02_010_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
        & no_stt02_010_observer
    )
    select = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & target_clean
    )
    skip = (
        (chosen == self._act_noop)
        & (active_acts[:, 0] == self._act_noop)
    )
    return (
        (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & clean
        & (select | skip)
    )

  def _effect_stt02_016_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_016_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_is_leader = target_index == self._garden_size
    zone_row = zone_host[rows, opp]
    zpos_row = zpos_host[rows, opp]
    def_row = def_host[rows, opp]
    garden_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] >= 0)
        & (target_index[:, None] < self._garden_size)
    )
    leader_match = zone_row == self._zone_leader
    garden_exists = np.any(garden_match, axis=1)
    leader_exists = np.any(leader_match, axis=1)
    garden_inst = np.argmax(garden_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    target_inst = np.where(target_is_leader, leader_inst, garden_inst)
    target_exists = np.where(target_is_leader, leader_exists, garden_exists)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_ok = (
        (target_index >= 0)
        & (target_index <= self._garden_size)
        & target_exists
        & (target_def >= 0)
        & (
            (
                target_is_leader
                & (self._card_type[target_safe_def] == self._card_type_leader)
            )
            | (
                ~target_is_leader
                & (self._card_type[target_safe_def] == self._card_type_entity)
            )
        )
    )

    clean = (
        (np.asarray(states.phase) == 3)  # Phase.RESPONSE_WINDOW
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_ok
        & clean
    )

  def _effect_stt02_001_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_001_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_is_leader = target_index == self._garden_size
    zone_row = zone_host[rows, opp]
    zpos_row = zpos_host[rows, opp]
    def_row = def_host[rows, opp]
    garden_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] >= 0)
        & (target_index[:, None] < self._garden_size)
    )
    leader_match = zone_row == self._zone_leader
    garden_exists = np.any(garden_match, axis=1)
    leader_exists = np.any(leader_match, axis=1)
    garden_inst = np.argmax(garden_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    target_inst = np.where(target_is_leader, leader_inst, garden_inst)
    target_exists = np.where(target_is_leader, leader_exists, garden_exists)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_ok = (
        (target_index >= 0)
        & (target_index <= self._garden_size)
        & target_exists
        & (target_def >= 0)
        & (
            (
                target_is_leader
                & (self._card_type[target_safe_def] == self._card_type_leader)
            )
            | (
                ~target_is_leader
                & (self._card_type[target_safe_def] == self._card_type_entity)
            )
        )
    )

    clean = (
        (np.asarray(states.phase) == 3)  # Phase.RESPONSE_WINDOW
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_ok
        & clean
    )

  def _effect_stt04_001_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    cur_hp = np.asarray(states.cur_hp)

    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt04_001_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    owner_zone = zone_host[rows, owner_safe]
    owner_zpos = zpos_host[rows, owner_safe]
    garden_match = (owner_zone == self._zone_garden) & (
        owner_zpos == target_index[:, None]
    )
    alley_slot = target_index - self._garden_size
    alley_match = (owner_zone == self._zone_alley) & (
        owner_zpos == alley_slot[:, None]
    )
    use_garden = target_index < self._garden_size
    target_match = np.where(use_garden[:, None], garden_match, alley_match)
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_host[rows, owner_safe, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    source_safe_def = np.maximum(source_def, 0)
    valid_target = (
        (target_index >= 0)
        & (target_index < 2 * self._garden_size)
        & target_exists
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & (cur_hp[rows, owner_safe, target_inst] > 0)
    )
    no_damage_triggers = ~(
        self._timing_takes_damage[source_safe_def]
        | self._timing_deals_damage[source_safe_def]
        | self._timing_takes_damage[target_safe_def]
        | self._timing_deals_damage[target_safe_def]
        | self._timing_when_destroyed[target_safe_def]
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & valid_target
        & no_damage_triggers
        & clean
    )

  def _effect_stt02_011_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_011_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    owner_zone = zone_host[rows, owner_safe]
    owner_zpos = zpos_host[rows, owner_safe]
    target_match = (owner_zone == self._zone_garden) & (
        owner_zpos == target_index[:, None]
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_host[rows, owner_safe, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    valid_target = (
        (target_index >= 0)
        & (target_index < self._garden_size)
        & target_exists
        & (target_inst != src_safe)
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & (np.asarray(states.cur_hp)[rows, owner_safe, target_inst] > 0)
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & valid_target
        & clean
    )

  def _effect_azk01_105_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    source_def = def_host[rows, owner_safe, src_safe]
    source_safe_def = np.maximum(source_def, 0)
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_105_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
        & (np.asarray(states.ab_scratch)[:, 0] > 0)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_is_leader = target_index == self._garden_size
    zone_row = zone_host[rows, opp]
    zpos_row = zpos_host[rows, opp]
    def_row = def_host[rows, opp]
    attached_row = attached_host[rows, opp]
    garden_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == target_index[:, None])
        & (target_index[:, None] >= 0)
        & (target_index[:, None] < self._garden_size)
    )
    leader_match = zone_row == self._zone_leader
    garden_exists = np.any(garden_match, axis=1)
    leader_exists = np.any(leader_match, axis=1)
    garden_inst = np.argmax(garden_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    target_inst = np.where(target_is_leader, leader_inst, garden_inst)
    target_exists = np.where(target_is_leader, leader_exists, garden_exists)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_ok = (
        (target_index >= 0)
        & (target_index <= self._garden_size)
        & target_exists
        & (target_def >= 0)
        & (
            (
                target_is_leader
                & (self._card_type[target_safe_def] == self._card_type_leader)
            )
            | (
                ~target_is_leader
                & (self._card_type[target_safe_def] == self._card_type_entity)
            )
        )
    )
    target_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == target_inst[:, None].astype(attached_row.dtype)),
        axis=1,
    )
    damage_clean = (
        ~target_has_attached
        & ~self._timing_takes_damage[source_safe_def]
        & ~self._timing_deals_damage[source_safe_def]
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, opp, target_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, target_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, opp, target_inst]
        & (np.asarray(states.effect_immune_dur)[rows, opp, target_inst] == 0)
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    no_attached = ~np.any(zone_host == self._zone_attached, axis=(1, 2))
    clean = (
        (np.asarray(states.phase) == 2)  # Phase.MAIN
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.asarray(states.eot_abilities_queued)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
        & no_attached
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_ok
        & damage_clean
        & clean
    )

  def _effect_stt04_004_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt04_004_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_alley)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_player = np.where(target_index < self._garden_size, owner_safe, opp)
    target_slot = np.where(
        target_index < self._garden_size,
        target_index,
        target_index - self._garden_size,
    )
    target_zone = zone_host[rows, target_player]
    target_zpos = zpos_host[rows, target_player]
    target_match = (target_zone == self._zone_garden) & (
        target_zpos == target_slot[:, None]
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_host[rows, target_player, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_has_attached = np.any(
        (zone_host[rows, target_player] == self._zone_attached)
        & (
            np.asarray(states.attached_to)[rows, target_player]
            == target_inst[:, None]
        ),
        axis=1,
    )
    target_clean = (
        target_exists
        & (target_index >= 0)
        & (target_index < 2 * self._garden_size)
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & (np.asarray(states.cur_hp)[rows, target_player, target_inst] > 0)
        & ~target_has_attached
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, target_player, target_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, target_player, target_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, target_player, target_inst]
        & (np.asarray(states.effect_immune_dur)[
            rows, target_player, target_inst
        ] == 0)
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board & np.isin(def_host, self._simple_play_watch_ids),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_clean
        & clean
    )

  def _effect_stt01_005_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_005_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) < 2)
        & (np.asarray(states.ab_eff_min) == 2)
        & (np.asarray(states.ab_eff_max) == 2)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    owner_zone = zone_host[rows, owner_safe]
    owner_zpos = zpos_host[rows, owner_safe]
    target_match = (owner_zone == self._zone_hand) & (
        owner_zpos == target_index[:, None]
    )
    target_exists = np.any(target_match, axis=1)

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_exists
        & clean
    )

  def _effect_stt01_006_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_006_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_garden)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
        & (np.asarray(states.combat_attacker) == src)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_is_leader = target_index == self._garden_size
    target_zone = zone_host[rows, opp]
    target_zpos = zpos_host[rows, opp]
    garden_match = (target_zone == self._zone_garden) & (
        target_zpos == target_index[:, None]
    )
    leader_match = target_zone == self._zone_leader
    garden_exists = np.any(garden_match, axis=1)
    leader_exists = np.any(leader_match, axis=1)
    garden_inst = np.argmax(garden_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    target_inst = np.where(target_is_leader, leader_inst, garden_inst)
    target_exists = np.where(target_is_leader, leader_exists, garden_exists)
    target_def = def_host[rows, opp, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_has_attached = np.any(
        (target_zone == self._zone_attached)
        & (
            np.asarray(states.attached_to)[rows, opp]
            == target_inst[:, None]
        ),
        axis=1,
    )
    target_clean = (
        target_exists
        & (target_index >= 0)
        & (target_index <= self._garden_size)
        & (target_def >= 0)
        & (
            target_is_leader
            | (self._card_type[target_safe_def] == self._card_type_entity)
        )
        & (np.asarray(states.cur_hp)[rows, opp, target_inst] > 0)
        & ~target_has_attached
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, opp, target_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, target_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, opp, target_inst]
        & (np.asarray(states.effect_immune_dur)[rows, opp, target_inst] == 0)
    )

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_clean
        & clean
    )

  def _effect_stt01_014_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)

    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_014_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_attached)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_player = np.where(target_index == 0, owner_safe, (owner_safe + 1) % 2)
    target_zone = zone_host[rows, target_player]
    leader_match = target_zone == self._zone_leader
    target_exists = np.any(leader_match, axis=1)
    target_inst = np.argmax(leader_match, axis=1)
    target_def = def_host[rows, target_player, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_clean = (
        target_exists
        & (target_index >= 0)
        & (target_index <= 1)
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_leader)
        & (np.asarray(states.cur_hp)[rows, target_player, target_inst] > 0)
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[
            rows, target_player, target_inst
        ] == 0)
        & (np.asarray(states.carapace_eot)[
            rows, target_player, target_inst
        ] == 0)
        & ~np.asarray(states.grant_godmode)[rows, target_player, target_inst]
        & (np.asarray(states.effect_immune_dur)[
            rows, target_player, target_inst
        ] == 0)
    )

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_clean
        & clean
    )

  def _effect_stt01_017_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    cur_hp = np.asarray(states.cur_hp)

    source_def = def_host[rows, owner_safe, src_safe]
    source_safe_def = np.maximum(source_def, 0)
    selected_count = np.asarray(states.ab_eff_selected).astype(
        np.int32, copy=False
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_017_id)
        & ~np.asarray(states.ab_costs_applied)
        & (selected_count < 2)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 2)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_zone = zone_host[rows, opp]
    target_zpos = zpos_host[rows, opp]
    target_match = (
        (target_zone == self._zone_garden)
        & (target_zpos == target_index[:, None])
        & (target_index[:, None] >= 0)
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_host[rows, opp, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_has_attached = np.any(
        (target_zone == self._zone_attached)
        & (attached_host[rows, opp] == target_inst[:, None]),
        axis=1,
    )
    target_hp = cur_hp[rows, opp, target_inst]
    target_hp_ok = (target_hp > 1) | (
        (target_hp > 0) & ~np.isin(target_def, self._simple_play_watch_ids)
    )
    selected_targets = np.asarray(states.ab_eff_targets)
    selected_players = np.asarray(states.ab_eff_target_players)
    already_selected = (
        (selected_count > 0)
        & (selected_players[:, 0] == opp.astype(np.int8))
        & (selected_targets[:, 0] == target_inst.astype(np.int8))
    )
    target_clean = (
        target_exists
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & target_hp_ok
        & ~target_has_attached
        & ~already_selected
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, opp, target_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, target_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, opp, target_inst]
        & (np.asarray(states.effect_immune_dur)[rows, opp, target_inst] == 0)
    )

    first_inst = np.maximum(selected_targets[:, 0].astype(np.int32), 0)
    first_player = selected_players[:, 0].astype(np.int32, copy=False)
    first_player_safe = np.clip(first_player, 0, 1)
    first_def = def_host[rows, first_player_safe, first_inst]
    first_safe_def = np.maximum(first_def, 0)
    first_hp = cur_hp[rows, first_player_safe, first_inst]
    first_hp_ok = (first_hp > 1) | (
        (first_hp > 0) & ~np.isin(first_def, self._simple_play_watch_ids)
    )
    first_has_attached = np.any(
        (zone_host[rows, first_player_safe] == self._zone_attached)
        & (attached_host[rows, first_player_safe] == first_inst[:, None]),
        axis=1,
    )
    first_clean = (
        (selected_count > 0)
        & (first_player == opp)
        & (zone_host[rows, first_player_safe, first_inst] == self._zone_garden)
        & (first_def >= 0)
        & (self._card_type[first_safe_def] == self._card_type_entity)
        & first_hp_ok
        & ~first_has_attached
        & ~self._timing_takes_damage[first_safe_def]
        & ~self._timing_deals_damage[first_safe_def]
        & ~self._timing_when_destroyed[first_safe_def]
        & ~self._inherent_godmode[first_safe_def]
        & (np.asarray(states.carapace_perm)[
            rows, first_player_safe, first_inst
        ] == 0)
        & (np.asarray(states.carapace_eot)[
            rows, first_player_safe, first_inst
        ] == 0)
        & ~np.asarray(states.grant_godmode)[rows, first_player_safe, first_inst]
        & (np.asarray(states.effect_immune_dur)[
            rows, first_player_safe, first_inst
        ] == 0)
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_passive_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.phase) == 3)  # Phase.RESPONSE_WINDOW
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & ~np.asarray(states.eot_abilities_queued)
        & no_passive_watch
        & ~self._timing_deals_damage[source_safe_def]
    )
    select_allowed = target_clean & (
        (selected_count == 0) | ((selected_count == 1) & first_clean)
    )
    select = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & select_allowed
    )
    skip = (
        (chosen == self._act_noop)
        & (active_acts[:, 0] == self._act_noop)
        & (selected_count >= 1)
        & first_clean
    )
    return (
        (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & clean
        & (select | skip)
    )

  def _effect_stt03_002_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt03_002_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_gate)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
        & (np.asarray(states.ab_scratch)[:, 2] == 1)
    )

    scratch = np.asarray(states.ab_scratch)
    portaled = np.clip(scratch[:, 0].astype(np.int32), 0, def_host.shape[2] - 1)
    portaled_def = def_host[rows, owner_safe, portaled]
    gate_power = np.where(
        scratch[:, 2] == 1,
        self._gate_points[np.maximum(portaled_def, 0)],
        0,
    ).astype(np.int32)

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    target_match = (zone_row == self._zone_garden) & (
        zpos_row == target_index[:, None]
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_row[rows, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_ok = (
        target_exists
        & (target_index >= 0)
        & (target_index < self._garden_size)
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & self._has_base_stats[target_safe_def]
        & (self._base_hp[target_safe_def].astype(np.int32) <= gate_power)
        & ~self._inherent_defender[target_safe_def]
        & ~np.asarray(states.grant_defender)[rows, owner_safe, target_inst]
    )

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_ok
        & clean
    )

  def _effect_stt03_006_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt03_006_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_discard)
        & np.asarray(states.ab_costs_applied)
        & ~np.asarray(states.ab_restores_active)
        & (np.asarray(states.ab_saved_active) < 0)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    hand_match = (
        (zone_host[rows, owner_safe] == self._zone_hand)
        & (zpos_host[rows, owner_safe] == target_index[:, None])
        & (target_index[:, None] >= 0)
    )
    target_exists = np.any(hand_match, axis=1)

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & source_ok
        & target_exists
        & clean
    )

  def _select_azk01_003_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_003_id)
        & (sel_count > 0)
        & (sel_count <= 5)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & self._black_jade[np.maximum(target_def, 0)]
        & (target_def != self._azk01_003_id)
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_stt02_003_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_003_id)
        & (sel_count > 0)
        & (sel_count <= 5)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & self._watercrafting[np.maximum(target_def, 0)]
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_stt02_013_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]
    target_safe_def = np.maximum(target_def, 0)

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_013_id)
        & (sel_count > 0)
        & (sel_count <= 3)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & (target_def >= 0)
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) <= 2)
        & (self._card_element[target_safe_def] == self._water_element)
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_azk01_033_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_033_id)
        & (sel_count > 0)
        & (sel_count <= 5)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & self._steelborn[np.maximum(target_def, 0)]
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_azk01_045_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_045_id)
        & (sel_count > 0)
        & (sel_count <= 5)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & self._obsidian[np.maximum(target_def, 0)]
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_azk01_056_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_056_id)
        & (sel_count > 0)
        & (sel_count <= 5)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & self._scorchweaver[np.maximum(target_def, 0)]
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_stt01_004_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    zone_host = np.asarray(states.zone)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_004_id)
        & (sel_count > 0)
        & (sel_count <= 5)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
        & np.asarray(states.ab_costs_applied)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & (zone_host[rows, owner_safe, inst_safe] == self._zone_selection)
        & (
            self._card_type[np.maximum(target_def, 0)]
            == self._card_type_weapon
        )
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_stt01_002_equip_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]
    target_safe_def = np.maximum(target_def, 0)

    target_index = active_acts[:, 2].astype(np.int32, copy=False)
    owner_zone = zone_host[rows, owner_safe]
    owner_zpos = zpos_host[rows, owner_safe]
    garden_host_match = (owner_zone == self._zone_garden) & (
        owner_zpos == target_index[:, None]
    )
    leader_host_match = owner_zone == self._zone_leader
    garden_host_exists = np.any(garden_host_match, axis=1)
    leader_host_exists = np.any(leader_host_match, axis=1)
    garden_host_inst = np.argmax(garden_host_match, axis=1)
    leader_host_inst = np.argmax(leader_host_match, axis=1)
    host_inst = np.where(
        target_index == self._garden_size, leader_host_inst, garden_host_inst
    )
    host_def = def_host[rows, owner_safe, host_inst]
    host_safe_def = np.maximum(host_def, 0)
    host_exists = np.where(
        target_index == self._garden_size, leader_host_exists, garden_host_exists
    )

    max_cost = np.asarray(states.ab_scratch)[:, 0].astype(np.int32, copy=False)
    allowed_on_play = ~self._timing_on_play[target_safe_def] | (
        target_def == self._stt01_013_id
    )
    trigger_ok = (
        allowed_on_play
        & ~self._timing_when_equipped[target_safe_def]
        & ~self._timing_when_equipped[host_safe_def]
    )
    action_matches = (
        (chosen == self._act_select_to_equip)
        & (active_acts[:, 0] == self._act_select_to_equip)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_002_id)
        & (sel_count > 0)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_scratch)[:, 2] == 2)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & (zone_host[rows, owner_safe, inst_safe] == self._zone_selection)
        & (
            self._card_type[target_safe_def]
            == self._card_type_weapon
        )
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) <= max_cost)
        & (target_index >= 0)
        & (target_index <= self._garden_size)
        & host_exists
        & trigger_ok
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_azk01_126_pick_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]
    target_safe_def = np.maximum(target_def, 0)

    scratch = np.asarray(states.ab_scratch)
    portaled = np.clip(scratch[:, 0].astype(np.int32), 0, def_host.shape[2] - 1)
    portaled_def = def_host[rows, owner_safe, portaled]
    gate_power = np.where(
        (scratch[:, 2] == 1) & (portaled_def >= 0),
        self._gate_points[np.maximum(portaled_def, 0)],
        0,
    ).astype(np.int32)

    action_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_126_id)
        & (sel_count > 0)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
        & np.asarray(states.ab_costs_applied)
        & (scratch[:, 2] == 1)
    )
    action_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & (zone_host[rows, owner_safe, inst_safe] == self._zone_selection)
        & (self._card_type[target_safe_def] == self._card_type_spell)
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) <= gate_power)
        & (gate_power > 0)
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & action_matches & source_ok & action_ok & clean

  def _select_azk01_097_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]

    pick_matches = (
        (chosen == self._act_select_from_selection)
        & (active_acts[:, 0] == self._act_select_from_selection)
        & (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & (
            self._card_type[np.maximum(target_def, 0)]
            == self._card_type_weapon
        )
    )
    decline_matches = (
        (chosen == self._act_noop) & (active_acts[:, 0] == self._act_noop)
    )
    in_selection = np.asarray(states.ab_phase) == self._ability_selection_pick
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_097_id)
        & (sel_count > 0)
        & (sel_count <= 5)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_selection & source_ok & (pick_matches | decline_matches) & clean

  def _selection_pick_noop_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_stt01_002 = (
        (source_def == self._stt01_002_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_scratch)[:, 2] == 2)
    )
    supported_source = (
        (source_def == self._azk01_003_id)
        | (source_def == self._azk01_033_id)
        | (source_def == self._azk01_045_id)
        | (source_def == self._azk01_056_id)
        | (source_def == self._stt02_003_id)
        | (source_def == self._stt02_013_id)
        | (source_def == self._stt01_004_id)
        | source_stt01_002
    )

    action_matches = (
        (chosen == self._act_noop) & (active_acts[:, 0] == self._act_noop)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & supported_source
        & (np.asarray(states.ab_sel_count).astype(np.int32, copy=False) > 0)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
    )
    clean = (
        (np.asarray(states.ab_phase) == self._ability_selection_pick)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return clean & action_matches & source_ok

  def _select_azk01_122_place_fast_mask(
      self, acts: np.ndarray, chosen, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    source_def = def_host[rows, owner_safe, src_safe]

    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    idx = active_acts[:, 1].astype(np.int32, copy=False)
    idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
    inst = sel_cards[rows, idx_safe]
    inst_safe = np.maximum(inst, 0)
    target_def = def_host[rows, owner_safe, inst_safe]
    target_safe_def = np.maximum(target_def, 0)

    scratch = np.asarray(states.ab_scratch)
    portaled = np.clip(
        scratch[:, 0].astype(np.int32, copy=False),
        0,
        def_host.shape[2] - 1,
    )
    portaled_def = def_host[rows, owner_safe, portaled]
    gate_power = np.where(
        (scratch[:, 2] == 1) & (portaled_def >= 0),
        self._gate_points[np.maximum(portaled_def, 0)],
        0,
    ).astype(np.int32, copy=False)

    action_matches = (
        (chosen == action_type)
        & (active_acts[:, 0] == action_type)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_122_id)
        & (sel_count > 0)
        & (np.asarray(states.ab_sel_pick_max) == 1)
        & (np.asarray(states.ab_sel_picked_count) == 0)
        & np.asarray(states.ab_costs_applied)
        & (scratch[:, 2] == 1)
    )
    target_ok = (
        (idx >= 0)
        & (idx < sel_count)
        & (inst >= 0)
        & (zone_host[rows, owner_safe, inst_safe] == self._zone_selection)
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & self._has_ikz_cost[target_safe_def]
        & (self._ikz_cost[target_safe_def].astype(np.int32) <= gate_power)
        & (gate_power > 0)
    )

    slot = active_acts[:, 2].astype(np.int32, copy=False)
    zone_row = zone_host[rows, owner_safe]
    zpos_row = zpos_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    attached_row = attached_host[rows, owner_safe]
    slot_match = (zone_row == placement_zone) & (zpos_row == slot[:, None])
    slot_occupied = np.any(slot_match, axis=1)
    slot_inst = np.argmax(slot_match, axis=1)
    zone_full = np.sum(zone_row == placement_zone, axis=1) >= self._garden_size
    slot_ok = (
        (slot >= 0)
        & (slot < self._garden_size)
        & (~slot_occupied | zone_full)
    )
    displaced_def = def_row[rows, slot_inst]
    displaced_safe_def = np.maximum(displaced_def, 0)
    displaced_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == slot_inst[:, None].astype(attached_row.dtype)),
        axis=1,
    )
    displaced_simple = (
        ~slot_occupied
        | (
            (displaced_def >= 0)
            & ~self._timing_when_destroyed[displaced_safe_def]
            & ~self._inherent_godmode[displaced_safe_def]
            & ~displaced_has_attached
        )
    )

    target_timing = self._timing_on_play[target_safe_def]
    if placement_zone == self._zone_garden:
      target_timing = target_timing | self._timing_enter_garden[target_safe_def]
    target_trigger_ok = ~target_timing

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    played_passive_ok = target_def == self._stt01_008_id
    no_passive_watch = (
        ~np.any(
            board
            & np.isin(def_host, self._simple_play_watch_ids)
            & (def_host != self._stt01_008_id),
            axis=(1, 2),
        )
        & (~np.isin(target_def, self._simple_play_watch_ids) | played_passive_ok)
    )
    clean = (
        (np.asarray(states.ab_phase) == self._ability_selection_pick)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & no_passive_watch
    )
    return (
        clean
        & action_matches
        & source_ok
        & target_ok
        & slot_ok
        & displaced_simple
        & target_trigger_ok
    )

  def _bottom_deck_azk01_003_fast_mask(
      self, acts: np.ndarray, chosen, all_cards: bool
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    sel_cards = np.asarray(states.ab_sel_cards)
    sel_count = np.asarray(states.ab_sel_count).astype(np.int32, copy=False)
    live_selection = sel_cards >= 0

    action_type = (
        self._act_bottom_deck_all if all_cards else self._act_bottom_deck_card
    )
    action_matches = (chosen == action_type) & (active_acts[:, 0] == action_type)
    in_bottom = np.asarray(states.ab_phase) == self._ability_bottom_deck
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (
            (source_def == self._azk01_003_id)
            | (source_def == self._azk01_033_id)
            | (source_def == self._azk01_045_id)
            | (source_def == self._azk01_056_id)
            | (source_def == self._stt02_003_id)
            | (source_def == self._stt02_013_id)
            | (source_def == self._stt01_004_id)
        )
        & (sel_count > 0)
        & (sel_count <= 5)
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if all_cards:
      action_ok = np.any(live_selection, axis=1)
    else:
      idx = active_acts[:, 1].astype(np.int32, copy=False)
      idx_safe = np.clip(idx, 0, sel_cards.shape[1] - 1)
      action_ok = (
          (idx >= 0)
          & (idx < sel_count)
          & (sel_cards[rows, idx_safe] >= 0)
      )
    return in_bottom & action_matches & source_ok & action_ok & clean

  def _play_entity_simple_fast_mask(
      self, acts: np.ndarray, chosen, phase, placement_zone: int, action_type: int
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    main_phase = (phase == 2) & (np.asarray(states.combat_attacker) < 0)
    response_phase = (phase == 3) & (np.asarray(states.combat_attacker) >= 0)
    base = (
        (main_phase | response_phase)
        & (chosen == action_type)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    slot = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    attached_row = attached_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    hand_exists = np.any(hand_match, axis=1)
    hand_inst = np.argmax(hand_match, axis=1)
    def_id = def_row[rows, hand_inst]
    valid_def = hand_exists & (def_id >= 0)
    safe_def = np.maximum(def_id, 0)

    is_entity = self._card_type[safe_def] == self._card_type_entity
    timing_ok = main_phase | self._response_play_from_hand[safe_def]
    slot_match = (zone_row == placement_zone) & (zpos_row == slot[:, None])
    slot_occupied = np.any(slot_match, axis=1)
    slot_inst = np.argmax(slot_match, axis=1)
    zone_count = np.sum(zone_row == placement_zone, axis=1)
    zone_full = zone_count >= self._garden_size
    slot_ok = ~slot_occupied | zone_full
    displaced_def = def_row[rows, slot_inst]
    displaced_safe_def = np.maximum(displaced_def, 0)
    displaced_has_attached = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == slot_inst[:, None].astype(attached_row.dtype)),
        axis=1,
    )
    displaced_simple = (
        ~slot_occupied
        | (
            (displaced_def >= 0)
            & ~self._timing_when_destroyed[displaced_safe_def]
            & ~self._inherent_godmode[displaced_safe_def]
            & ~displaced_has_attached
        )
    )

    timing = self._timing_on_play[safe_def]
    if placement_zone == self._zone_garden:
      timing = timing | self._timing_enter_garden[safe_def]
    supported_implemented = np.isin(def_id, self._simple_play_implemented_ids)
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_scorchweaver = leader_exists & self._scorchweaver[
        np.maximum(leader_def, 0)
    ]
    stt04_014_invalid = (
        (def_id == self._stt04_014_id) & ~leader_scorchweaver
    )
    supported_implemented = supported_implemented | stt04_014_invalid
    implemented_trigger_ok = ~(
        timing
        & self._implemented[safe_def]
        & ~supported_implemented
    )

    board = (zone_row == self._zone_garden) | (zone_row == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_row, self._simple_play_watch_ids)
        & (def_row != self._stt01_008_id)
    )
    played_passive_ok = def_id == self._stt01_008_id
    no_passive_watch = (
        ~np.any(watched_on_board, axis=1)
        & (~np.isin(def_id, self._simple_play_watch_ids) | played_passive_ok)
    )

    return (
        base
        & hand_exists
        & valid_def
        & is_entity
        & timing_ok
        & slot_ok
        & displaced_simple
        & implemented_trigger_ok
        & no_passive_watch
    )

  def _gate_portal_simple_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    passive_events_clean = ~np.any(
        np.asarray(states.stt02_012_event_pending), axis=(1, 2)
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_gate_portal)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_events_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    alley_index = active_acts[:, 1]
    garden_index = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]
    attached_row = np.asarray(states.attached_to)[rows, active]

    gate_match = zone_row == self._zone_gate
    gate_exists = np.any(gate_match, axis=1)
    gate_inst = np.argmax(gate_match, axis=1)
    gate_def = def_row[rows, gate_inst]
    gate_untapped = gate_exists & ~tapped_row[rows, gate_inst]

    alley_match = (zone_row == self._zone_alley) & (
        zpos_row == alley_index[:, None]
    )
    alley_exists = np.any(alley_match, axis=1)
    alley_inst = np.argmax(alley_match, axis=1)
    portaled_def = def_row[rows, alley_inst]
    valid_portaled = alley_exists & (portaled_def >= 0)
    safe_portaled_def = np.maximum(portaled_def, 0)
    alley_ok = alley_exists & ~tapped_row[rows, alley_inst]
    gate_power = np.where(valid_portaled, self._gate_points[safe_portaled_def], 0)

    safe_defs = np.maximum(def_row, 0)
    rushfire_eligible = (
        (zone_row == self._zone_hand)
        & (self._card_type[safe_defs] == self._card_type_entity)
        & self._has_ikz_cost[safe_defs]
        & (self._ikz_cost[safe_defs].astype(np.int32) <= gate_power[:, None])
        & (gate_power[:, None] > 0)
    )
    azk01_122_fizzles = (
        (gate_def == self._azk01_122_id)
        & ~np.any(rushfire_eligible, axis=1)
    )
    echoed_eligible = (
        (zone_row == self._zone_discard)
        & (self._card_type[safe_defs] == self._card_type_spell)
        & self._has_ikz_cost[safe_defs]
        & (self._ikz_cost[safe_defs].astype(np.int32) <= gate_power[:, None])
        & (gate_power[:, None] > 0)
    )
    azk01_126_ok = (
        (gate_def == self._azk01_126_id)
        & np.any(echoed_eligible, axis=1)
    )
    azk01_126_fizzles = (
        (gate_def == self._azk01_126_id)
        & ~np.any(echoed_eligible, axis=1)
    )
    attached = (zone_row == self._zone_attached) & (attached_row >= 0)
    host = np.clip(attached_row.astype(np.int32, copy=False), 0, zone_row.shape[1] - 1)
    host_zone = np.take_along_axis(zone_row, host, axis=1)
    host_is_leader = host_zone == self._zone_leader
    host_ok = host_is_leader | (host_zone == self._zone_garden)
    garden_count = np.sum(zone_row == self._zone_garden, axis=1)
    other_host = np.where(host_is_leader, garden_count[:, None] >= 1, True)
    reequip_eligible = (
        attached
        & host_ok
        & (self._card_type[safe_defs] == self._card_type_weapon)
        & self._has_ikz_cost[safe_defs]
        & (self._ikz_cost[safe_defs].astype(np.int32) <= gate_power[:, None])
        & (gate_power[:, None] > 0)
        & other_host
    )
    azk01_120_fizzles = (
        (gate_def == self._azk01_120_id)
        & ~np.any(reequip_eligible, axis=1)
    )
    garden_match = (zone_row == self._zone_garden) & (
        zpos_row == garden_index[:, None]
    )
    garden_occupied = np.any(garden_match, axis=1)
    garden_inst = np.argmax(garden_match, axis=1)
    garden_full = garden_count >= self._garden_size
    slot_ok = ~garden_occupied | garden_full
    displaced_def = def_row[rows, garden_inst]
    displaced_safe_def = np.maximum(displaced_def, 0)
    displaced_simple = (
        ~garden_occupied
        | (
            (displaced_def >= 0)
            & ~self._timing_when_destroyed[displaced_safe_def]
            & ~self._inherent_godmode[displaced_safe_def]
        )
    )
    no_enter_trigger = ~self._timing_enter_garden[safe_portaled_def]

    stt03_target_ok = (
        (gate_def == self._stt03_002_id)
        & ~garden_occupied
        & (self._card_type[safe_portaled_def] == self._card_type_entity)
        & self._has_base_stats[safe_portaled_def]
        & (self._base_hp[safe_portaled_def].astype(np.int32) <= gate_power)
        & ~self._inherent_defender[safe_portaled_def]
        & ~np.asarray(states.grant_defender)[rows, active, alley_inst]
        & ~np.any(
            np.asarray(states.timed_tag)[rows, active, alley_inst] != 0,
            axis=1,
        )
    )
    inst_index = np.arange(zone_row.shape[1])[None, :]
    replaced_inst = garden_full & garden_occupied
    garden_after_portal = (zone_row == self._zone_garden) & ~(
        replaced_inst[:, None] & (inst_index == garden_inst[:, None])
    )
    entity_after_portal = self._card_type[safe_defs] == self._card_type_entity
    damaged_after_portal = (
        garden_after_portal
        & entity_after_portal
        & np.asarray(states.took_damage_turn)[rows, active]
    )
    portaled_damaged = (
        (self._card_type[safe_portaled_def] == self._card_type_entity)
        & np.asarray(states.took_damage_turn)[rows, active, alley_inst]
    )
    defender_after_portal = (
        self._inherent_defender[safe_defs]
        | np.asarray(states.grant_defender)[rows, active]
        | np.any(np.asarray(states.timed_tag)[rows, active] != 0, axis=2)
    )
    stonehaven_after_portal = (
        garden_after_portal
        & entity_after_portal
        & self._has_base_stats[safe_defs]
        & (self._base_hp[safe_defs].astype(np.int32) <= gate_power[:, None])
        & ~defender_after_portal
    )
    portaled_stonehaven_target = (
        (self._card_type[safe_portaled_def] == self._card_type_entity)
        & self._has_base_stats[safe_portaled_def]
        & (self._base_hp[safe_portaled_def].astype(np.int32) <= gate_power)
        & ~self._inherent_defender[safe_portaled_def]
        & ~np.asarray(states.grant_defender)[rows, active, alley_inst]
        & ~np.any(
            np.asarray(states.timed_tag)[rows, active, alley_inst] != 0,
            axis=1,
        )
    )
    stt03_002_no_targets = (
        (gate_def == self._stt03_002_id)
        & ~np.any(stonehaven_after_portal, axis=1)
        & ~portaled_stonehaven_target
    )
    stt04_002_no_targets = (
        (gate_def == self._stt04_002_id)
        & ~np.any(damaged_after_portal, axis=1)
        & ~portaled_damaged
    )
    supported_gate = (
        (gate_def == self._stt01_002_id)
        | (gate_def == self._stt02_002_id)
        | (gate_def == self._azk01_122_id)
        | azk01_120_fizzles
        | azk01_126_ok
        | azk01_126_fizzles
        | azk01_122_fizzles
        | stt03_target_ok
        | stt03_002_no_targets
        | stt04_002_no_targets
    )
    gate_ok = gate_untapped & supported_gate

    board = (zone_row == self._zone_garden) | (zone_row == self._zone_alley)
    watched_on_board = (
        board
        & np.isin(def_row, self._simple_play_watch_ids)
        & (def_row != self._stt01_008_id)
    )
    portaled_has_weapon = np.any(
        (zone_row == self._zone_attached)
        & (attached_row == alley_inst[:, None])
        & (self._card_type[safe_defs] == self._card_type_weapon),
        axis=1,
    )
    portaled_passive_ok = (
        (portaled_def == self._stt01_008_id)
        & ~portaled_has_weapon
    )
    no_passive_watch = (
        ~np.any(watched_on_board, axis=1)
        & (
            ~np.isin(portaled_def, self._simple_play_watch_ids)
            | portaled_passive_ok
        )
    )

    return (
        base
        & gate_ok
        & alley_ok
        & valid_portaled
        & slot_ok
        & displaced_simple
        & no_enter_trigger
        & no_passive_watch
    )

  def _confirm_clear_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_stt01_002 = (
        (owner >= 0) & (src >= 0) & (source_def == self._stt01_002_id)
    )
    source_stt03_002 = (
        (owner == active) & (src >= 0) & (source_def == self._stt03_002_id)
    )
    source_stt01_014 = (
        (owner == active) & (src >= 0) & (source_def == self._stt01_014_id)
    )

    scratch = np.asarray(states.ab_scratch)
    portaled = np.clip(scratch[:, 0].astype(np.int32), 0, def_host.shape[2] - 1)
    portaled_def = def_host[rows, owner_safe, portaled]
    portaled_safe_def = np.maximum(portaled_def, 0)
    gate_power = np.where(
        scratch[:, 2] == 1,
        self._gate_points[portaled_safe_def],
        0,
    ).astype(np.int32)

    zone_host = np.asarray(states.zone)
    zone_row = zone_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    safe_defs = np.maximum(def_row, 0)
    eligible = (
        (zone_row == self._zone_discard)
        & (self._card_type[safe_defs] == self._card_type_weapon)
        & self._has_ikz_cost[safe_defs]
        & (self._ikz_cost[safe_defs].astype(np.int32) <= gate_power[:, None])
        & (gate_power[:, None] > 0)
    )
    no_eligible = ~np.any(eligible, axis=1)

    in_confirm = np.asarray(states.ab_phase) == self._ability_confirmation
    optional = np.asarray(states.ab_is_optional)
    action_matches = active_acts[:, 0] == chosen
    safe_clear = np.asarray(states.combat_attacker) < 0
    decline = (
        (chosen == self._act_noop)
        & in_confirm
        & optional
        & action_matches
    )
    stt01002_confirm = (
        (chosen == self._act_confirm)
        & in_confirm
        & optional
        & action_matches
        & source_stt01_002
        & no_eligible
    )
    stt03002_skip = (
        (chosen == self._act_noop)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & action_matches
        & source_stt03_002
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
        & (scratch[:, 2] == 1)
    )
    stt01014_skip = (
        (chosen == self._act_noop)
        & (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & action_matches
        & source_stt01_014
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
    )
    return (decline | stt01002_confirm | stt03002_skip | stt01014_skip) & safe_clear

  def _confirm_stt01_002_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    scratch = np.asarray(states.ab_scratch)
    portaled = np.clip(scratch[:, 0].astype(np.int32), 0, def_host.shape[2] - 1)
    portaled_def = def_host[rows, owner_safe, portaled]
    gate_power = np.where(
        (scratch[:, 2] == 1) & (portaled_def >= 0),
        self._gate_points[np.maximum(portaled_def, 0)],
        0,
    ).astype(np.int32)

    zone_row = zone_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    safe_defs = np.maximum(def_row, 0)
    eligible = (
        (zone_row == self._zone_discard)
        & (self._card_type[safe_defs] == self._card_type_weapon)
        & self._has_ikz_cost[safe_defs]
        & (self._ikz_cost[safe_defs].astype(np.int32) <= gate_power[:, None])
        & (gate_power[:, None] > 0)
    )
    has_eligible = np.any(eligible, axis=1)

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & (np.asarray(states.ab_sel_count) == 0)
    )
    return (
        (chosen == self._act_confirm)
        & (active_acts[:, 0] == self._act_confirm)
        & (np.asarray(states.ab_phase) == self._ability_confirmation)
        & (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_002_id)
        & np.asarray(states.ab_is_optional)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 0)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 0)
        & (scratch[:, 2] == 1)
        & has_eligible
        & clean
    )

  def _confirm_stt01_007_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]

    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & (np.asarray(states.ab_sel_count) == 0)
    )
    return (
        (chosen == self._act_confirm)
        & (active_acts[:, 0] == self._act_confirm)
        & (np.asarray(states.ab_phase) == self._ability_confirmation)
        & (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_007_id)
        & np.asarray(states.ab_is_optional)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 0)
        & clean
    )

  def _confirm_stt01_013_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    source_def = def_host[rows, owner_safe, src_safe]
    source_safe_def = np.maximum(source_def, 0)

    zone_row = zone_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_safe_def = np.maximum(leader_def, 0)

    host = np.asarray(states.attached_to)[rows, owner_safe, src_safe].astype(
        np.int32, copy=False
    )
    host_valid = host >= 0
    host_safe = np.maximum(host, 0)
    host_zone = zone_host[rows, owner_safe, host_safe]

    base = (
        (chosen == self._act_confirm)
        & (active_acts[:, 0] == self._act_confirm)
        & (np.asarray(states.ab_phase) == self._ability_confirmation)
        & np.asarray(states.ab_is_optional)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_013_id)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 0)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 0)
    )
    damage_clean = ~(
        self._timing_takes_damage[leader_safe_def]
        | self._timing_when_destroyed[leader_safe_def]
        | self._timing_deals_damage[source_safe_def]
    )
    return (
        base
        & source_ok
        & leader_exists
        & (np.asarray(states.cur_hp)[rows, owner_safe, leader_inst] > 1)
        & damage_clean
        & host_valid
        & (
            (host_zone == self._zone_garden)
            | (host_zone == self._zone_alley)
            | (host_zone == self._zone_leader)
        )
    )

  def _confirm_stt01_004_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    zone_host = np.asarray(states.zone)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt01_004_id)
    )

    zone_row = zone_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    safe_defs = np.maximum(def_row, 0)
    hand_weapons = np.any(
        (zone_row == self._zone_hand)
        & (self._card_type[safe_defs] == self._card_type_weapon),
        axis=1,
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_confirm)
        & (active_acts[:, 0] == self._act_confirm)
        & (np.asarray(states.ab_phase) == self._ability_confirmation)
        & np.asarray(states.ab_is_optional)
        & source_ok
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & hand_weapons
        & clean
    )

  def _confirm_stt04_004_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    zone_host = np.asarray(states.zone)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt04_004_id)
    )
    any_garden_entity = np.any(
        (zone_host == self._zone_garden)
        & (
            self._card_type[np.maximum(def_host, 0)]
            == self._card_type_entity
        ),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_confirm)
        & (active_acts[:, 0] == self._act_confirm)
        & (np.asarray(states.ab_phase) == self._ability_confirmation)
        & np.asarray(states.ab_is_optional)
        & source_ok
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
        & any_garden_entity
        & clean
    )

  def _confirm_stt02_009_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    def_host = np.asarray(states.def_id)
    zone_host = np.asarray(states.zone)
    source_def = def_host[rows, owner_safe, src_safe]
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._stt02_009_id)
    )

    zone_row = zone_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    safe_defs = np.maximum(def_row, 0)
    cost_available = np.any(
        (zone_row == self._zone_garden)
        & (self._card_type[safe_defs] == self._card_type_entity)
        & self._has_ikz_cost[safe_defs]
        & (self._ikz_cost[safe_defs].astype(np.int32) >= 2),
        axis=1,
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        (chosen == self._act_confirm)
        & (active_acts[:, 0] == self._act_confirm)
        & (np.asarray(states.ab_phase) == self._ability_confirmation)
        & np.asarray(states.ab_is_optional)
        & source_ok
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_cost_selected) == 0)
        & (np.asarray(states.ab_cost_max) == 1)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
        & cost_available
        & clean
    )

  def _confirm_azk01_058_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    src_safe = np.maximum(src, 0)
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    attached_to = np.asarray(states.attached_to)
    zone_row = zone_host[rows, owner_safe]
    def_row = def_host[rows, owner_safe]
    source_def = def_host[rows, owner_safe, src_safe]

    in_confirm = np.asarray(states.ab_phase) == self._ability_confirmation
    action_matches = (
        (chosen == self._act_confirm)
        & (active_acts[:, 0] == self._act_confirm)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_058_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_garden)
        & np.asarray(states.ab_is_optional)
        & (np.asarray(states.ab_cost_max) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )
    no_attached_source = ~np.any(
        (zone_row == self._zone_attached) & (attached_to[rows, owner_safe] == src_safe[:, None]),
        axis=1,
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    leader_exists = np.any(zone_host == self._zone_leader, axis=(1, 2))
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        in_confirm
        & action_matches
        & source_ok
        & no_attached_source
        & no_non_inert_watch
        & leader_exists
        & clean
    )

  def _confirm_azk01_060_response_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]
    source_safe_def = np.maximum(source_def, 0)
    safe_source_def = np.maximum(source_def, 0)

    in_confirm = np.asarray(states.ab_phase) == self._ability_confirmation
    decline = (
        (chosen == self._act_noop)
        & (active_acts[:, 0] == self._act_noop)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_060_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_garden)
        & np.asarray(states.ab_is_optional)
        & (np.asarray(states.ab_cost_max) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 0)
        & (np.asarray(states.combat_attacker) == src)
        & (np.asarray(states.combat_defender) >= 0)
        & (np.asarray(states.combat_defender_player).astype(
            np.int32, copy=False
        ) == opp)
    )

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    attacker_infiltrate = (
        self._inherent_infiltrate[safe_source_def]
        | np.asarray(states.grant_infiltrate)[rows, owner_safe, src_safe]
    )
    defender_cards = (
        (opp_zone == self._zone_garden)
        & (
            self._inherent_defender[opp_safe_defs]
            | np.asarray(states.grant_defender)[rows, opp]
        )
        & ~np.asarray(states.tapped)[rows, opp]
    )
    can_declare_defender = (
        np.any(defender_cards, axis=1)
        & ~attacker_infiltrate
        & ~np.asarray(states.combat_intercepted)
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_confirm & decline & source_ok & can_declare_defender & clean

  def _confirm_azk01_060_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    defender = np.asarray(states.combat_defender).astype(np.int32, copy=False)
    defender_safe = np.clip(defender, 0, n_inst - 1)
    defender_player = np.asarray(states.combat_defender_player).astype(
        np.int32, copy=False
    )
    defender_def = def_host[rows, opp, defender_safe]
    defender_zone = zone_host[rows, opp, defender_safe]
    safe_attacker_def = np.maximum(source_def, 0)
    safe_defender_def = np.maximum(defender_def, 0)

    in_confirm = np.asarray(states.ab_phase) == self._ability_confirmation
    action_matches = active_acts[:, 0] == chosen
    action_ok = (
        ((chosen == self._act_confirm) | (chosen == self._act_noop))
        & action_matches
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_060_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_garden)
        & np.asarray(states.ab_is_optional)
        & (np.asarray(states.ab_cost_max) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 0)
        & (np.asarray(states.combat_attacker) == src)
    )
    defender_ok = (
        (defender >= 0)
        & (defender_player == opp)
        & (defender_def >= 0)
        & (
            (
                (defender_zone == self._zone_garden)
                & (self._card_type[safe_defender_def] == self._card_type_entity)
            )
            | (defender_zone == self._zone_leader)
        )
    )

    attached_to = np.asarray(states.attached_to)
    owner_zone = zone_host[rows, owner_safe]
    opp_zone = zone_host[rows, opp]
    no_attached = (
        ~np.any(
            (owner_zone == self._zone_attached)
            & (attached_to[rows, owner_safe] == src_safe[:, None]),
            axis=1,
        )
        & ~np.any(
            (opp_zone == self._zone_attached)
            & (attached_to[rows, opp] == defender_safe[:, None]),
            axis=1,
        )
    )

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, owner_safe, src_safe] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, owner_safe, src_safe] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, owner_safe, src_safe] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, owner_safe, src_safe] == 0)
        & (np.asarray(states.carapace_perm)[rows, owner_safe, src_safe] == 0)
        & (np.asarray(states.carapace_eot)[rows, owner_safe, src_safe] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, opp, defender_safe] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, opp, defender_safe] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, opp, defender_safe] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, opp, defender_safe] == 0)
        & (np.asarray(states.carapace_perm)[rows, opp, defender_safe] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, defender_safe] == 0)
        & (np.asarray(states.frozen_dur)[rows, owner_safe, src_safe] == 0)
        & (np.asarray(states.frozen_dur)[rows, opp, defender_safe] == 0)
        & ~np.asarray(states.grant_godmode)[rows, owner_safe, src_safe]
        & ~np.asarray(states.grant_godmode)[rows, opp, defender_safe]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    cur_atk = np.asarray(states.cur_atk).astype(np.int16, copy=False)
    cur_hp = np.asarray(states.cur_hp).astype(np.int16, copy=False)
    damage_to_defender = cur_atk[rows, owner_safe, src_safe]
    damage_to_attacker = cur_atk[rows, opp, defender_safe]
    attacker_hp = cur_hp[rows, owner_safe, src_safe]
    defender_hp = cur_hp[rows, opp, defender_safe]
    clean_combat_damage = (
        (damage_to_defender >= 0)
        & (damage_to_attacker >= 0)
        & (attacker_hp > 0)
        & (defender_hp > 0)
    )

    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    response_hand = (opp_zone == self._zone_hand) & (
        (
            (self._card_type[opp_safe_defs] == self._card_type_spell)
            & self._timing_is_response[opp_safe_defs]
            & self._has_ability[opp_safe_defs]
        )
        | self._response_play_from_hand[opp_safe_defs]
    )
    in_board = (
        (opp_zone == self._zone_garden)
        | (opp_zone == self._zone_alley)
        | (opp_zone == self._zone_leader)
    )
    response_board = (
        in_board
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
    )
    defender_cards = (
        (opp_zone == self._zone_garden)
        & (
            self._inherent_defender[opp_safe_defs]
            | np.asarray(states.grant_defender)[rows, opp]
        )
        & ~np.asarray(states.tapped)[rows, opp]
    )
    no_response = ~(
        np.any(response_hand, axis=1)
        | np.any(response_board, axis=1)
        | np.any(defender_cards, axis=1)
    )
    no_nondefender_response = ~(
        np.any(response_hand, axis=1)
        | np.any(response_board, axis=1)
    )
    no_kira_redirect = ~np.any(
        (opp_zone == self._zone_alley) & (opp_defs == self._azk01_034_id),
        axis=1,
    )

    trigger_ok = (
        self._timing_when_attacking[safe_attacker_def]
        & ~self._timing_after_attacking[safe_attacker_def]
        & ~self._timing_when_attacked[safe_defender_def]
        & ~self._timing_takes_damage[safe_attacker_def]
        & ~self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~self._timing_when_destroyed[safe_attacker_def]
        & ~self._timing_when_destroyed[safe_defender_def]
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    stt01_008_active = (
        passive_watch
        & (def_host == self._stt01_008_id)
        & (
            (np.asarray(states.passive_atk) != 0)
            | (np.asarray(states.passive_hp) != 0)
        )
    )
    passive_watch_ok = (
        ~np.any(passive_watch & (def_host != self._stt01_008_id), axis=(1, 2))
        & ~np.any(stt01_008_active, axis=(1, 2))
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    confirm = chosen == self._act_confirm
    confirmed_damage_to_defender = damage_to_defender + confirm.astype(np.int16)
    target_is_entity = defender_zone == self._zone_garden
    target_is_leader = defender_zone == self._zone_leader
    target_response_ok = (
        (target_is_entity & no_response)
        | (
            target_is_leader
            & confirm
            & no_nondefender_response
            & (confirmed_damage_to_defender < defender_hp)
        )
    )
    return (
        in_confirm
        & action_ok
        & source_ok
        & defender_ok
        & no_attached
        & no_modifiers
        & clean_combat_damage
        & target_response_ok
        & no_kira_redirect
        & trigger_ok
        & passive_watch_ok
        & clean
    )

  def _effect_azk01_058_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    friendly_garden = (target_index >= 0) & (target_index < self._garden_size)
    enemy_garden = (
        (target_index >= self._garden_size)
        & (target_index < 2 * self._garden_size)
    )
    own_leader = target_index == 2 * self._garden_size
    enemy_leader = target_index == (2 * self._garden_size + 1)

    garden_player = np.where(friendly_garden, owner_safe, opp)
    garden_slot = np.where(
        friendly_garden,
        target_index,
        target_index - self._garden_size,
    )
    garden_match = (
        (zone_host[rows, garden_player] == self._zone_garden)
        & (zpos_host[rows, garden_player] == garden_slot[:, None])
    )
    has_garden = np.any(garden_match, axis=1)
    garden_inst = np.where(has_garden, np.argmax(garden_match, axis=1), -1)

    own_leader_match = zone_host[rows, owner_safe] == self._zone_leader
    has_own_leader = np.any(own_leader_match, axis=1)
    own_leader_inst = np.where(
        has_own_leader, np.argmax(own_leader_match, axis=1), -1
    )
    enemy_leader_match = zone_host[rows, opp] == self._zone_leader
    has_enemy_leader = np.any(enemy_leader_match, axis=1)
    enemy_leader_inst = np.where(
        has_enemy_leader, np.argmax(enemy_leader_match, axis=1), -1
    )

    target_player = np.where(friendly_garden | own_leader, owner_safe, opp)
    target_inst = np.where(
        friendly_garden | enemy_garden,
        garden_inst,
        np.where(own_leader, own_leader_inst, enemy_leader_inst),
    )
    target_safe = np.clip(target_inst, 0, n_inst - 1)
    target_def = def_host[rows, target_player, target_safe]
    target_type = self._card_type[np.maximum(target_def, 0)]
    target_ok = (
        ((own_leader | enemy_leader) & (target_inst >= 0))
        | (
            friendly_garden
            & (target_inst >= 0)
            & (target_type == self._card_type_entity)
        )
    )

    in_effect = np.asarray(states.ab_phase) == self._ability_effect_selection
    action_matches = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_058_id)
        & np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        in_effect
        & action_matches
        & source_ok
        & target_ok
        & no_non_inert_watch
        & clean
    )

  def _effect_azk01_040_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    attacker_p = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]
    source_safe_def = np.maximum(source_def, 0)

    combat_attacker = np.asarray(states.combat_attacker).astype(
        np.int32, copy=False
    )
    combat_defender = np.asarray(states.combat_defender).astype(
        np.int32, copy=False
    )
    safe_attacker = np.clip(combat_attacker, 0, n_inst - 1)
    safe_defender = np.clip(combat_defender, 0, n_inst - 1)
    attacker_zone = zone_host[rows, attacker_p, safe_attacker]
    defender_zone = zone_host[rows, owner_safe, safe_defender]
    attacker_def = def_host[rows, attacker_p, safe_attacker]
    defender_def = def_host[rows, owner_safe, safe_defender]
    safe_attacker_def = np.maximum(attacker_def, 0)
    safe_defender_def = np.maximum(defender_def, 0)
    cur_atk = np.asarray(states.cur_atk)
    cur_hp = np.asarray(states.cur_hp)

    in_effect = np.asarray(states.ab_phase) == self._ability_effect_selection
    skip_action = (chosen == self._act_noop) & (
        active_acts[:, 0] == self._act_noop
    )
    select_action = (chosen == self._act_select_effect) & (
        active_acts[:, 0] == self._act_select_effect
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_040_id)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 0)
        & (np.asarray(states.ab_eff_max) == 1)
        & np.asarray(states.ab_restores_active)
        & (np.asarray(states.ab_saved_active).astype(np.int32, copy=False)
           == attacker_p)
    )
    clean = (
        (np.asarray(states.phase) == 4)  # Phase.COMBAT_RESOLVE
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (combat_attacker >= 0)
        & (combat_defender >= 0)
        & (
            np.asarray(states.combat_defender_player).astype(
                np.int32, copy=False
            )
            == owner
        )
        & (src == combat_defender)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_is_owner = target_index == 0
    target_player = np.where(target_is_owner, owner_safe, attacker_p)
    target_zone_row = zone_host[rows, target_player]
    target_def_row = def_host[rows, target_player]
    leader_match = target_zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    target_def = target_def_row[rows, leader_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_hp = cur_hp[rows, target_player, leader_inst].astype(np.int16)
    select_target_clean = (
        select_action
        & (target_index >= 0)
        & (target_index <= 1)
        & leader_exists
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_leader)
        & (target_hp > 1)
        & ~self._timing_takes_damage[source_safe_def]
        & ~self._timing_deals_damage[source_safe_def]
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & ~np.asarray(states.grant_godmode)[rows, target_player, leader_inst]
        & (np.asarray(states.effect_immune_dur)[
            rows, target_player, leader_inst
        ] == 0)
        & (np.asarray(states.carapace_perm)[
            rows, target_player, leader_inst
        ] == 0)
        & (np.asarray(states.carapace_eot)[
            rows, target_player, leader_inst
        ] == 0)
    )

    attached_to = np.asarray(states.attached_to)
    no_attached = (
        ~np.any(
            (zone_host[rows, attacker_p] == self._zone_attached)
            & (attached_to[rows, attacker_p] == safe_attacker[:, None]),
            axis=1,
        )
        & ~np.any(
            (zone_host[rows, owner_safe] == self._zone_attached)
            & (attached_to[rows, owner_safe] == safe_defender[:, None]),
            axis=1,
        )
    )

    damage_to_attacker = cur_atk[rows, owner_safe, safe_defender].astype(np.int16)
    damage_to_defender = cur_atk[rows, attacker_p, safe_attacker].astype(np.int16)
    attacker_hp = cur_hp[rows, attacker_p, safe_attacker].astype(np.int16)
    defender_hp = cur_hp[rows, owner_safe, safe_defender].astype(np.int16)
    attacker_after = attacker_hp - damage_to_attacker
    defender_after = defender_hp - damage_to_defender
    clean_damage = (
        (damage_to_defender > 0)
        & (damage_to_attacker >= 0)
        & (attacker_hp > 0)
        & (defender_hp > 0)
    )
    attacker_dies = clean_damage & (attacker_after <= 0)
    defender_dies = clean_damage & (defender_after <= 0)
    no_passive_death_watch = (
        ~np.any(np.asarray(states.passive_observer_registered), axis=(1, 2))
        & ~np.any(
            (zone_host == self._zone_garden)
            & (def_host == self._stt02_012_id),
            axis=(1, 2),
        )
    )
    trigger_ok = ~(
        self._timing_after_attacking[safe_attacker_def]
        | self._timing_takes_damage[safe_attacker_def]
        | self._timing_deals_damage[safe_attacker_def]
        | self._timing_takes_damage[safe_defender_def]
        | self._timing_deals_damage[safe_defender_def]
        | (self._timing_when_destroyed[safe_attacker_def] & attacker_dies)
        | (self._timing_when_destroyed[safe_defender_def] & defender_dies)
    )

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.carapace_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.carapace_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, owner_safe, safe_defender] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, owner_safe, safe_defender] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, owner_safe, safe_defender] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, owner_safe, safe_defender] == 0)
        & (np.asarray(states.carapace_perm)[rows, owner_safe, safe_defender] == 0)
        & (np.asarray(states.carapace_eot)[rows, owner_safe, safe_defender] == 0)
        & (np.asarray(states.frozen_dur)[rows, owner_safe, safe_defender] == 0)
        & ~np.asarray(states.grant_godmode)[rows, attacker_p, safe_attacker]
        & ~np.asarray(states.grant_godmode)[rows, owner_safe, safe_defender]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    return (
        in_effect
        & (skip_action | select_target_clean)
        & source_ok
        & clean
        & (attacker_zone == self._zone_garden)
        & (defender_zone == self._zone_garden)
        & (attacker_def >= 0)
        & (defender_def == self._azk01_040_id)
        & (self._card_type[safe_attacker_def] == self._card_type_entity)
        & (self._card_type[safe_defender_def] == self._card_type_entity)
        & np.asarray(states.tapped)[rows, attacker_p, safe_attacker]
        & no_attached
        & clean_damage
        & ((attacker_after > 0) | (attacker_dies & no_passive_death_watch))
        & ((defender_after > 0) | (defender_dies & no_passive_death_watch))
        & trigger_ok
        & no_modifiers
    )

  def _effect_azk01_007_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_match = (
        (zone_host[rows, owner_safe] == self._zone_garden)
        & (zpos_host[rows, owner_safe] == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.where(target_exists, np.argmax(target_match, axis=1), -1)
    target_safe = np.clip(target_inst, 0, n_inst - 1)
    target_def = def_host[rows, owner_safe, target_safe]
    target_ok = (
        target_exists
        & (target_def >= 0)
        & (self._card_type[np.maximum(target_def, 0)] == self._card_type_entity)
    )

    in_effect = np.asarray(states.ab_phase) == self._ability_effect_selection
    action_matches = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
    )
    source_board = (
        (zone_host[rows, owner_safe, src_safe] == self._zone_garden)
        | (zone_host[rows, owner_safe, src_safe] == self._zone_alley)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_007_id)
        & source_board
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        in_effect
        & action_matches
        & source_ok
        & target_ok
        & no_non_inert_watch
        & clean
    )

  def _effect_azk01_070_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    cur_hp = np.asarray(states.cur_hp)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    selected_count = np.asarray(states.ab_eff_selected).astype(
        np.int32, copy=False
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_070_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_garden)
        & np.asarray(states.ab_costs_applied)
        & (selected_count == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_match = (
        (zone_host[rows, opp] == self._zone_garden)
        & (zpos_host[rows, opp] == target_index[:, None])
        & (target_index[:, None] >= 0)
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_def = def_host[rows, opp, target_inst]
    target_safe_def = np.maximum(target_def, 0)
    target_ok = (
        target_exists
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & (cur_hp[rows, opp, target_inst] > 0)
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_non_inert_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.phase) == 3)  # Phase.RESPONSE_WINDOW
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
        & ~np.asarray(states.eot_abilities_queued)
        & no_non_inert_watch
    )
    return (
        (np.asarray(states.ab_phase) == self._ability_effect_selection)
        & (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
        & source_ok
        & target_ok
        & clean
    )

  def _effect_azk01_009_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    friendly_garden = target_index < self._garden_size
    enemy_garden = (
        (target_index >= self._garden_size)
        & (target_index < 2 * self._garden_size)
    )
    target_player = np.where(friendly_garden, owner_safe, opp)
    target_slot = np.where(
        friendly_garden,
        target_index,
        target_index - self._garden_size,
    )
    target_match = (
        (zone_host[rows, target_player] == self._zone_garden)
        & (zpos_host[rows, target_player] == target_slot[:, None])
        & (target_slot[:, None] >= 0)
        & (target_slot[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.where(target_exists, np.argmax(target_match, axis=1), -1)
    target_safe = np.clip(target_inst, 0, n_inst - 1)
    target_player_safe = np.clip(target_player, 0, 1)
    target_def = def_host[rows, target_player_safe, target_safe]
    target_safe_def = np.maximum(target_def, 0)
    target_ok = (
        (friendly_garden | enemy_garden)
        & target_exists
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & (self._ikz_cost[target_safe_def].astype(np.int32) <= 4)
    )

    in_effect = np.asarray(states.ab_phase) == self._ability_effect_selection
    action_matches = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_009_id)
        & (zone_host[rows, owner_safe, src_safe] == self._zone_discard)
        & ~np.asarray(states.ab_costs_applied)
        & ~np.asarray(states.ab_restores_active)
        & (np.asarray(states.ab_saved_active) < 0)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        in_effect
        & action_matches
        & source_ok
        & target_ok
        & no_non_inert_watch
        & clean
    )

  def _effect_azk01_059_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_match = (
        (zone_host[rows, owner_safe] == self._zone_garden)
        & (zpos_host[rows, owner_safe] == target_index[:, None])
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.where(target_exists, np.argmax(target_match, axis=1), -1)
    target_safe = np.clip(target_inst, 0, n_inst - 1)
    target_def = def_host[rows, owner_safe, target_safe]
    target_ok = (
        target_exists
        & (target_inst != src)
        & (target_def >= 0)
        & (self._card_type[np.maximum(target_def, 0)] == self._card_type_entity)
    )

    in_effect = np.asarray(states.ab_phase) == self._ability_effect_selection
    action_matches = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_059_id)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
        & ((np.asarray(states.once_per_turn_used)[rows, owner_safe, src_safe] & 1) == 0)
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        in_effect
        & action_matches
        & source_ok
        & target_ok
        & no_non_inert_watch
        & clean
    )

  def _effect_azk01_065_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    friendly_garden = target_index < self._garden_size
    enemy_garden = (
        (target_index >= self._garden_size)
        & (target_index < 2 * self._garden_size)
    )
    own_leader = target_index == 2 * self._garden_size
    enemy_leader = target_index == (2 * self._garden_size + 1)

    own_garden_match = (
        (zone_host[rows, owner_safe] == self._zone_garden)
        & (zpos_host[rows, owner_safe] == target_index[:, None])
    )
    enemy_slot = target_index - self._garden_size
    enemy_garden_match = (
        (zone_host[rows, opp] == self._zone_garden)
        & (zpos_host[rows, opp] == enemy_slot[:, None])
    )
    own_garden_exists = np.any(own_garden_match, axis=1)
    enemy_garden_exists = np.any(enemy_garden_match, axis=1)
    own_garden_inst = np.argmax(own_garden_match, axis=1)
    enemy_garden_inst = np.argmax(enemy_garden_match, axis=1)

    own_leader_match = zone_host[rows, owner_safe] == self._zone_leader
    enemy_leader_match = zone_host[rows, opp] == self._zone_leader
    own_leader_exists = np.any(own_leader_match, axis=1)
    enemy_leader_exists = np.any(enemy_leader_match, axis=1)
    own_leader_inst = np.argmax(own_leader_match, axis=1)
    enemy_leader_inst = np.argmax(enemy_leader_match, axis=1)

    target_player = np.where(
        friendly_garden | own_leader,
        owner_safe,
        opp,
    )
    target_inst = np.where(
        friendly_garden,
        np.where(own_garden_exists, own_garden_inst, -1),
        np.where(
            enemy_garden,
            np.where(enemy_garden_exists, enemy_garden_inst, -1),
            np.where(
                own_leader,
                np.where(own_leader_exists, own_leader_inst, -1),
                np.where(enemy_leader_exists, enemy_leader_inst, -1),
            ),
        ),
    )
    target_safe = np.clip(target_inst, 0, n_inst - 1)
    target_player_safe = np.clip(target_player, 0, 1)
    target_def = def_host[rows, target_player_safe, target_safe]
    target_safe_def = np.maximum(target_def, 0)
    target_is_entity = (
        target_def >= 0
    ) & (self._card_type[target_safe_def] == self._card_type_entity)
    target_is_leader = own_leader | enemy_leader
    target_ok = (
        ((friendly_garden | enemy_garden) & (target_inst >= 0) & target_is_entity)
        | (target_is_leader & (target_inst >= 0))
    )

    own_leader_safe = np.clip(own_leader_inst, 0, n_inst - 1)
    own_leader_def = def_host[rows, owner_safe, own_leader_safe]
    own_leader_safe_def = np.maximum(own_leader_def, 0)
    self_damage_clean = (
        own_leader_exists
        & (np.asarray(states.cur_hp)[rows, owner_safe, own_leader_safe] > 3)
        & ~self._timing_takes_damage[own_leader_safe_def]
        & ~self._inherent_godmode[own_leader_safe_def]
        & (np.asarray(states.carapace_perm)[rows, owner_safe, own_leader_safe] == 0)
        & (np.asarray(states.carapace_eot)[rows, owner_safe, own_leader_safe] == 0)
        & ~np.asarray(states.grant_godmode)[rows, owner_safe, own_leader_safe]
        & (np.asarray(states.effect_immune_dur)[rows, owner_safe, own_leader_safe] == 0)
    )

    target_has_attached = np.any(
        (zone_host[rows, target_player_safe] == self._zone_attached)
        & (
            attached_host[rows, target_player_safe]
            == target_safe[:, None].astype(attached_host.dtype)
        ),
        axis=1,
    )
    target_base_clean = (
        target_ok
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, target_player_safe, target_safe] == 0)
        & (np.asarray(states.carapace_eot)[rows, target_player_safe, target_safe] == 0)
        & ~np.asarray(states.grant_godmode)[rows, target_player_safe, target_safe]
        & (
            np.asarray(states.effect_immune_dur)[
                rows, target_player_safe, target_safe
            ] == 0
        )
        & (
            target_is_leader
            | (
                ~self._timing_when_destroyed[target_safe_def]
                & ~target_has_attached
            )
        )
    )
    target_owner_zone = zone_host[rows, target_player_safe]
    target_owner_def = def_host[rows, target_player_safe]
    target_owner_safe_def = np.maximum(target_owner_def, 0)
    inst_cols = np.arange(n_inst, dtype=np.int32)
    target_owner_garden_entity = (
        (target_owner_zone == self._zone_garden)
        & (target_owner_def >= 0)
        & (self._card_type[target_owner_safe_def] == self._card_type_entity)
    )
    azk01_059_has_target = np.any(
        target_owner_garden_entity & (inst_cols[None, :] != target_safe[:, None]),
        axis=1,
    )
    target_azk01_059_trigger = (
        target_ok
        & (target_def == self._azk01_059_id)
        & ((np.asarray(states.once_per_turn_used)[
            rows, target_player_safe, target_safe
        ] & 1) == 0)
        & azk01_059_has_target
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~target_has_attached
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, target_player_safe, target_safe] == 0)
        & (np.asarray(states.carapace_eot)[rows, target_player_safe, target_safe] == 0)
        & ~np.asarray(states.grant_godmode)[rows, target_player_safe, target_safe]
        & (
            np.asarray(states.effect_immune_dur)[
                rows, target_player_safe, target_safe
            ] == 0
        )
    )
    target_clean = target_base_clean | target_azk01_059_trigger

    in_effect = np.asarray(states.ab_phase) == self._ability_effect_selection
    action_matches = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_065_id)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
        & ~self._timing_deals_damage[np.maximum(source_def, 0)]
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    clean = (
        (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return (
        in_effect
        & action_matches
        & source_ok
        & self_damage_clean
        & target_clean
        & no_non_inert_watch
        & clean
    )

  def _effect_azk01_127_fast_mask(self, acts: np.ndarray, chosen):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]

    owner = np.asarray(states.ab_owner).astype(np.int32, copy=False)
    src = np.asarray(states.ab_source).astype(np.int32, copy=False)
    owner_safe = np.clip(owner, 0, 1)
    opp = (owner_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    attached_host = np.asarray(states.attached_to)
    n_inst = def_host.shape[2]
    src_safe = np.clip(src, 0, n_inst - 1)
    source_def = def_host[rows, owner_safe, src_safe]

    target_index = active_acts[:, 1].astype(np.int32, copy=False)
    target_match = (
        (zone_host[rows, opp] == self._zone_garden)
        & (zpos_host[rows, opp] == target_index[:, None])
        & (target_index[:, None] >= 0)
        & (target_index[:, None] < self._garden_size)
    )
    target_exists = np.any(target_match, axis=1)
    target_inst = np.argmax(target_match, axis=1)
    target_safe = np.clip(target_inst, 0, n_inst - 1)
    target_def = def_host[rows, opp, target_safe]
    target_safe_def = np.maximum(target_def, 0)
    target_hp = np.asarray(states.cur_hp)[rows, opp, target_safe]
    lethal_target = target_hp <= 1
    target_has_attached = np.any(
        (zone_host[rows, opp] == self._zone_attached)
        & (
            attached_host[rows, opp]
            == target_safe[:, None].astype(attached_host.dtype)
        ),
        axis=1,
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    inert_watch = def_host == self._stt01_008_id
    no_non_inert_watch = ~np.any(
        board & np.isin(def_host, self._simple_play_watch_ids) & ~inert_watch,
        axis=(1, 2),
    )
    target_clean = (
        target_exists
        & (target_index >= 0)
        & (target_index < self._garden_size)
        & (target_def >= 0)
        & (self._card_type[target_safe_def] == self._card_type_entity)
        & (target_hp > 0)
        & (~lethal_target | no_non_inert_watch)
        & (target_def != self._azk01_062_id)
        & ~target_has_attached
        & ~self._timing_takes_damage[target_safe_def]
        & ~self._timing_deals_damage[target_safe_def]
        & ~self._timing_when_destroyed[target_safe_def]
        & ~self._inherent_godmode[target_safe_def]
        & (np.asarray(states.carapace_perm)[rows, opp, target_safe] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, target_safe] == 0)
        & ~np.asarray(states.grant_godmode)[rows, opp, target_safe]
    )

    in_effect = np.asarray(states.ab_phase) == self._ability_effect_selection
    action_matches = (
        (chosen == self._act_select_effect)
        & (active_acts[:, 0] == self._act_select_effect)
    )
    source_ok = (
        (owner == active)
        & (src >= 0)
        & (source_def == self._azk01_127_id)
        & ~np.asarray(states.ab_costs_applied)
        & (np.asarray(states.ab_eff_selected) == 0)
        & (np.asarray(states.ab_eff_min) == 1)
        & (np.asarray(states.ab_eff_max) == 1)
        & ~self._timing_deals_damage[np.maximum(source_def, 0)]
    )
    clean = (
        (np.asarray(states.phase) == 3)  # Phase.RESPONSE_WINDOW
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    return in_effect & action_matches & source_ok & target_clean & clean

  def _activate_stt04_001_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_activate_garden)
        & (active_acts[:, 0] == self._act_activate_garden)
        & (active_acts[:, 1] == self._garden_size)
        & (active_acts[:, 2] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_safe_def = np.maximum(leader_def, 0)

    friendly_targets = (
        ((zone_row == self._zone_garden) | (zone_row == self._zone_alley))
        & (def_row >= 0)
        & (self._card_type[np.maximum(def_row, 0)] == self._card_type_entity)
    )
    no_damage_triggers = ~(
        self._timing_takes_damage[leader_safe_def]
        | self._timing_deals_damage[leader_safe_def]
        | self._timing_when_destroyed[leader_safe_def]
    )
    source_ok = (
        leader_exists
        & (leader_def == self._stt04_001_id)
        & ~tapped_row[rows, leader_inst]
        & (np.asarray(states.frozen_dur)[rows, active, leader_inst] == 0)
        & ((np.asarray(states.once_per_turn_used)[
            rows, active, leader_inst
        ] & 1) == 0)
        & (np.asarray(states.cur_hp)[rows, active, leader_inst] > 1)
        & np.any(friendly_targets, axis=1)
        & no_damage_triggers
    )
    return base & source_ok

  def _activate_stt02_011_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    cur_hp = np.asarray(states.cur_hp)
    n_inst = def_host.shape[2]

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_activate_garden)
        & (active_acts[:, 0] == self._act_activate_garden)
        & (active_acts[:, 2] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    source_slot = active_acts[:, 1].astype(np.int32, copy=False)
    source_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == source_slot[:, None])
        & (source_slot[:, None] >= 0)
        & (source_slot[:, None] < self._garden_size)
    )
    source_exists = np.any(source_match, axis=1)
    source_inst = np.argmax(source_match, axis=1)
    source_safe = np.clip(source_inst, 0, n_inst - 1)
    source_def = def_row[rows, source_safe]

    inst = np.arange(n_inst)[None, :]
    safe_defs = np.maximum(def_row, 0)
    friendly_targets = (
        (zone_row == self._zone_garden)
        & (def_row >= 0)
        & (self._card_type[safe_defs] == self._card_type_entity)
        & (cur_hp[rows, active_safe] > 0)
        & (inst != source_safe[:, None])
    )
    source_ok = (
        source_exists
        & (source_def == self._stt02_011_id)
        & (np.asarray(states.frozen_dur)[rows, active_safe, source_safe] == 0)
        & np.any(friendly_targets, axis=1)
    )
    return base & source_ok

  def _activate_azk01_070_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)
    cur_hp = np.asarray(states.cur_hp)
    n_inst = def_host.shape[2]

    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_activate_garden)
        & (active_acts[:, 0] == self._act_activate_garden)
        & (active_acts[:, 2] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.asarray(states.eot_abilities_queued)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    source_slot = active_acts[:, 1].astype(np.int32, copy=False)
    source_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == source_slot[:, None])
        & (source_slot[:, None] >= 0)
        & (source_slot[:, None] < self._garden_size)
    )
    source_exists = np.any(source_match, axis=1)
    source_inst = np.argmax(source_match, axis=1)
    source_safe = np.clip(source_inst, 0, n_inst - 1)
    source_def = def_row[rows, source_safe]
    source_safe_def = np.maximum(source_def, 0)

    opp = (active_safe + 1) % 2
    opp_def = def_host[rows, opp]
    opp_safe_def = np.maximum(opp_def, 0)
    enemy_targets = (
        (zone_host[rows, opp] == self._zone_garden)
        & (opp_def >= 0)
        & (self._card_type[opp_safe_def] == self._card_type_entity)
        & (cur_hp[rows, opp] > 0)
    )
    enemy_target_available = np.any(enemy_targets, axis=1)

    source_ok = (
        source_exists
        & (source_def == self._azk01_070_id)
        & ~tapped_host[rows, active_safe, source_safe]
        & (np.asarray(states.cooldown)[rows, active_safe, source_safe] == 0)
        & (cur_hp[rows, active_safe, source_safe] > 1)
        & ~self._timing_takes_damage[source_safe_def]
        & ~self._timing_deals_damage[source_safe_def]
        & ~self._inherent_godmode[source_safe_def]
        & (np.asarray(states.carapace_perm)[
            rows, active_safe, source_safe
        ] == 0)
        & (np.asarray(states.carapace_eot)[
            rows, active_safe, source_safe
        ] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active_safe, source_safe]
        & (np.asarray(states.effect_immune_dur)[
            rows, active_safe, source_safe
        ] == 0)
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_non_inert_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    return base & source_ok & enemy_target_available & no_non_inert_watch

  def _activate_azk01_105_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    cur_hp = np.asarray(states.cur_hp)
    n_inst = def_host.shape[2]

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_activate_garden)
        & (active_acts[:, 0] == self._act_activate_garden)
        & (active_acts[:, 2] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    source_slot = active_acts[:, 1].astype(np.int32, copy=False)
    source_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == source_slot[:, None])
        & (source_slot[:, None] >= 0)
        & (source_slot[:, None] < self._garden_size)
    )
    source_exists = np.any(source_match, axis=1)
    source_inst = np.argmax(source_match, axis=1)
    source_safe = np.clip(source_inst, 0, n_inst - 1)
    source_def = def_row[rows, source_safe]

    opp = (active_safe + 1) % 2
    opp_zone = zone_host[rows, opp]
    opp_def = def_host[rows, opp]
    opp_safe_def = np.maximum(opp_def, 0)
    leader_exists = np.any(opp_zone == self._zone_leader, axis=1)
    enemy_garden_entity = (
        (opp_zone == self._zone_garden)
        & (opp_def >= 0)
        & (self._card_type[opp_safe_def] == self._card_type_entity)
    )
    target_available = leader_exists | np.any(enemy_garden_entity, axis=1)

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    no_non_inert_watch = ~np.any(
        board
        & np.isin(def_host, self._simple_play_watch_ids)
        & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    no_attached = ~np.any(zone_host == self._zone_attached, axis=(1, 2))
    source_ok = (
        source_exists
        & (source_def == self._azk01_105_id)
        & (np.asarray(states.frozen_dur)[rows, active_safe, source_safe] == 0)
        & (cur_hp[rows, active_safe, source_safe] > 0)
    )
    return (
        base
        & source_ok
        & target_available
        & no_non_inert_watch
        & no_attached
    )

  def _activate_azk01_121_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_activate_garden)
        & (active_acts[:, 0] == self._act_activate_garden)
        & (active_acts[:, 1] == self._garden_size)
        & (active_acts[:, 2] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_safe_def = np.maximum(leader_def, 0)

    use_token = active_acts[:, 3] != 0
    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    cost = self._ability_ikz_cost[leader_safe_def].astype(np.int32, copy=False)
    can_pay = token_needed_ok & (payment_sources >= cost)

    total_played = (
        np.asarray(states.entities_played_garden_turn)[rows, active].astype(
            np.int32, copy=False
        )
        + np.asarray(states.entities_played_alley_turn)[rows, active].astype(
            np.int32, copy=False
        )
    )
    leader_ok = (
        leader_exists
        & (leader_def == self._azk01_121_id)
        & (np.asarray(states.frozen_dur)[rows, active, leader_inst] == 0)
        & ((np.asarray(states.once_per_turn_used)[
            rows, active, leader_inst
        ] & 1) == 0)
        & (total_played > 0)
        & can_pay
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    return base & leader_ok & no_non_inert_watch

  def _activate_stt03_001_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_activate_garden)
        & (active_acts[:, 0] == self._act_activate_garden)
        & (active_acts[:, 1] == self._garden_size)
        & (active_acts[:, 2] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active]
    def_row = def_host[rows, active]
    tapped_row = tapped_host[rows, active]
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_safe_def = np.maximum(leader_def, 0)

    use_token = active_acts[:, 3] != 0
    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    cost = self._ability_ikz_cost[leader_safe_def].astype(np.int32, copy=False)
    can_pay = token_needed_ok & (payment_sources >= cost)

    leader_ok = (
        leader_exists
        & (leader_def == self._stt03_001_id)
        & (np.asarray(states.frozen_dur)[rows, active, leader_inst] == 0)
        & ((np.asarray(states.once_per_turn_used)[
            rows, active, leader_inst
        ] & 1) == 0)
        & can_pay
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    return base & leader_ok & no_non_inert_watch

  def _activate_stt02_001_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_activate_garden)
        & (active_acts[:, 0] == self._act_activate_garden)
        & (active_acts[:, 1] == self._garden_size)
        & (active_acts[:, 2] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.winner) == -1)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.asarray(states.eot_abilities_queued)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    tapped_row = tapped_host[rows, active_safe]
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_safe_def = np.maximum(leader_def, 0)

    use_token = active_acts[:, 3] != 0
    token_ready = (
        (zone_row[:, self._token_instance] == self._zone_token)
        & ~tapped_row[:, self._token_instance]
    )
    token_needed_ok = ~use_token | token_ready
    payment_sources = np.sum(
        (zone_row == self._zone_ikz_area) & ~tapped_row,
        axis=1,
    )
    safe_row_defs = np.maximum(def_row, 0)
    payment_sources += np.sum(
        (zone_row == self._zone_garden)
        & ~tapped_row
        & self._counts_as_ikz[safe_row_defs],
        axis=1,
    )
    payment_sources += (use_token & token_ready).astype(np.int32)
    cost = self._ability_ikz_cost[leader_safe_def].astype(np.int32, copy=False)
    can_pay = token_needed_ok & (payment_sources >= cost)

    opp = (active_safe + 1) % 2
    opp_zone = zone_host[rows, opp]
    opp_def = def_host[rows, opp]
    opp_safe_def = np.maximum(opp_def, 0)
    enemy_targets = (
        (opp_zone == self._zone_leader)
        | (
            (opp_zone == self._zone_garden)
            & (opp_def >= 0)
            & (self._card_type[opp_safe_def] == self._card_type_entity)
        )
    )
    target_available = np.any(enemy_targets, axis=1)

    leader_ok = (
        leader_exists
        & (leader_def == self._stt02_001_id)
        & (np.asarray(states.frozen_dur)[rows, active_safe, leader_inst] == 0)
        & ((np.asarray(states.once_per_turn_used)[
            rows, active_safe, leader_inst
        ] & 1) == 0)
        & can_pay
        & target_available
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    return base & leader_ok & no_non_inert_watch

  def _activate_stt01_005_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_acts = acts[rows, np.clip(active, 0, 1)]
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_activate_alley)
        & (active_acts[:, 0] == self._act_activate_alley)
        & (active_acts[:, 1] == 0)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]
    alley_slot = active_acts[:, 2].astype(np.int32, copy=False)
    source_match = (zone_row == self._zone_alley) & (
        zpos_row == alley_slot[:, None]
    )
    source_exists = np.any(source_match, axis=1)
    source_inst = np.argmax(source_match, axis=1)
    source_def = def_row[rows, source_inst]

    hand_count = np.sum(zone_row == self._zone_hand, axis=1)
    deck_count = np.sum(zone_row == self._zone_deck, axis=1)
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_non_inert_watch = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    source_ok = (
        source_exists
        & (source_def == self._stt01_005_id)
        & (deck_count > 3)
        & (hand_count >= 2)
    )
    return base & source_ok & no_non_inert_watch

  def _attach_stt01_013_confirm_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attach_weapon)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    entity_index = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    weapon_exists = np.any(hand_match, axis=1)
    weapon_inst = np.argmax(hand_match, axis=1)
    weapon_def = def_row[rows, weapon_inst]
    safe_weapon_def = np.maximum(weapon_def, 0)

    target_is_leader = entity_index == self._garden_size
    garden_target = entity_index < self._garden_size
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    leader_def = def_row[rows, leader_inst]
    leader_safe_def = np.maximum(leader_def, 0)
    target_match = (zone_row == self._zone_garden) & (
        zpos_row == entity_index[:, None]
    )
    garden_exists = garden_target & np.any(target_match, axis=1)
    target_exists = garden_exists | (target_is_leader & leader_exists)
    garden_inst = np.argmax(target_match, axis=1)
    target_inst = np.where(target_is_leader, leader_inst, garden_inst)
    target_def = def_row[rows, target_inst]
    safe_target_def = np.maximum(target_def, 0)

    no_attach_triggers = ~(
        self._timing_when_equipped[safe_weapon_def]
        | self._timing_when_equipped[safe_target_def]
    )
    leader_damage_clean = ~(
        self._timing_takes_damage[leader_safe_def]
        | self._timing_when_destroyed[leader_safe_def]
        | self._timing_deals_damage[safe_weapon_def]
    )

    board = (
        (zone_row == self._zone_garden)
        | (zone_row == self._zone_alley)
        | (zone_row == self._zone_attached)
        | (zone_row == self._zone_leader)
    )
    watched_on_board = (
        board
        & np.isin(def_row, self._simple_attach_watch_ids)
        & (def_row != self._stt01_008_id)
    )
    target_passive_ok = ~np.isin(target_def, self._simple_attach_watch_ids) | (
        target_def == self._stt01_008_id
    )
    no_passive_watch = (
        ~np.any(watched_on_board, axis=1)
        & ~np.isin(weapon_def, self._simple_attach_watch_ids)
        & target_passive_ok
    )

    return (
        base
        & (active_acts[:, 0] == self._act_attach_weapon)
        & weapon_exists
        & (weapon_def == self._stt01_013_id)
        & target_exists
        & leader_exists
        & (np.asarray(states.cur_hp)[rows, active, leader_inst] > 1)
        & no_attach_triggers
        & leader_damage_clean
        & no_passive_watch
    )

  def _attach_weapon_simple_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attach_weapon)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    hand_index = active_acts[:, 1]
    entity_index = active_acts[:, 2]
    zone_row = zone_host[rows, active]
    zpos_row = zpos_host[rows, active]
    def_row = def_host[rows, active]

    hand_match = (zone_row == self._zone_hand) & (zpos_row == hand_index[:, None])
    weapon_exists = np.any(hand_match, axis=1)
    weapon_inst = np.argmax(hand_match, axis=1)
    weapon_def = def_row[rows, weapon_inst]
    valid_weapon = weapon_exists & (weapon_def >= 0)
    safe_weapon_def = np.maximum(weapon_def, 0)
    is_weapon = self._card_type[safe_weapon_def] == self._card_type_weapon

    target_is_leader = entity_index == self._garden_size
    garden_target = entity_index < self._garden_size
    leader_match = zone_row == self._zone_leader
    leader_exists = np.any(leader_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    target_match = (zone_row == self._zone_garden) & (
        zpos_row == entity_index[:, None]
    )
    garden_exists = garden_target & np.any(target_match, axis=1)
    target_exists = garden_exists | (target_is_leader & leader_exists)
    garden_inst = np.argmax(target_match, axis=1)
    target_inst = np.where(target_is_leader, leader_inst, garden_inst)
    target_def = def_row[rows, target_inst]
    safe_target_def = np.maximum(target_def, 0)

    allowed_on_play = ~self._timing_on_play[safe_weapon_def] | (
        weapon_def == self._stt01_014_id
    )
    no_attach_triggers = (
        allowed_on_play
        & ~self._timing_when_equipped[safe_weapon_def]
        & ~self._timing_when_equipped[safe_target_def]
    )

    board = (
        (zone_row == self._zone_garden)
        | (zone_row == self._zone_alley)
        | (zone_row == self._zone_attached)
        | (zone_row == self._zone_leader)
    )
    watched_on_board = (
        board
        & np.isin(def_row, self._simple_attach_watch_ids)
        & (def_row != self._stt01_008_id)
    )
    target_passive_ok = ~np.isin(target_def, self._simple_attach_watch_ids) | (
        target_def == self._stt01_008_id
    )
    no_passive_watch = (
        ~np.any(watched_on_board, axis=1)
        & ~np.isin(weapon_def, self._simple_attach_watch_ids)
        & target_passive_ok
    )

    return (
        base
        & valid_weapon
        & is_weapon
        & target_exists
        & no_attach_triggers
        & no_passive_watch
    )

  def _attack_azk01_060_confirm_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attack)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    attacker_index = active_acts[:, 1]
    defender_index = active_acts[:, 2]
    atk_zone = zone_host[rows, active]
    atk_zpos = zpos_host[rows, active]
    atk_defs = def_host[rows, active]
    def_zone = zone_host[rows, opp]
    def_zpos = zpos_host[rows, opp]
    def_defs = def_host[rows, opp]

    attacker_match = (atk_zone == self._zone_garden) & (
        atk_zpos == attacker_index[:, None]
    )
    attacker_exists = np.any(attacker_match, axis=1)
    attacker_inst = np.argmax(attacker_match, axis=1)
    attacker_def = atk_defs[rows, attacker_inst]
    safe_attacker_def = np.maximum(attacker_def, 0)

    target_is_leader = defender_index == self._garden_size
    target_is_garden = (
        (defender_index >= 0) & (defender_index < self._garden_size)
    )
    garden_match = (def_zone == self._zone_garden) & (
        def_zpos == defender_index[:, None]
    )
    leader_match = def_zone == self._zone_leader
    garden_exists = np.any(garden_match, axis=1)
    leader_exists = np.any(leader_match, axis=1)
    garden_inst = np.argmax(garden_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    defender_exists = np.where(
        target_is_leader, leader_exists, target_is_garden & garden_exists
    )
    defender_inst = np.where(target_is_leader, leader_inst, garden_inst)
    defender_def = def_defs[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)

    exact_cards = (
        (attacker_def == self._azk01_060_id)
        & (defender_def == self._stt01_008_id)
    )
    attached_to = np.asarray(states.attached_to)
    no_attached = (
        ~np.any(
            (atk_zone == self._zone_attached)
            & (attached_to[rows, active] == attacker_inst[:, None]),
            axis=1,
        )
        & ~np.any(
            (def_zone == self._zone_attached)
            & (attached_to[rows, opp] == defender_inst[:, None]),
            axis=1,
        )
    )

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, opp, defender_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active, attacker_inst]
        & ~np.asarray(states.grant_godmode)[rows, opp, defender_inst]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    cur_atk = np.asarray(states.cur_atk).astype(np.int16, copy=False)
    cur_hp = np.asarray(states.cur_hp).astype(np.int16, copy=False)
    damage_to_defender = cur_atk[rows, active, attacker_inst]
    damage_to_attacker = cur_atk[rows, opp, defender_inst]
    attacker_hp = cur_hp[rows, active, attacker_inst]
    defender_hp = cur_hp[rows, opp, defender_inst]
    mutual_destroy = (
        (damage_to_defender > 0)
        & (damage_to_attacker > 0)
        & (damage_to_defender >= defender_hp)
        & (damage_to_attacker >= attacker_hp)
    )

    trigger_ok = (
        self._timing_when_attacking[safe_attacker_def]
        & self._implemented[safe_attacker_def]
        & ~self._timing_after_attacking[safe_attacker_def]
        & ~self._timing_when_attacked[safe_defender_def]
        & ~self._timing_takes_damage[safe_attacker_def]
        & ~self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~self._timing_when_destroyed[safe_attacker_def]
        & ~self._timing_when_destroyed[safe_defender_def]
    )

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    response_hand = (opp_zone == self._zone_hand) & (
        (
            (self._card_type[opp_safe_defs] == self._card_type_spell)
            & self._timing_is_response[opp_safe_defs]
            & self._has_ability[opp_safe_defs]
        )
        | self._response_play_from_hand[opp_safe_defs]
    )
    in_board = (
        (opp_zone == self._zone_garden)
        | (opp_zone == self._zone_alley)
        | (opp_zone == self._zone_leader)
    )
    response_board = (
        in_board
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
    )
    defender_cards = (
        (opp_zone == self._zone_garden)
        & (
            self._inherent_defender[opp_safe_defs]
            | np.asarray(states.grant_defender)[rows, opp]
        )
        & ~np.asarray(states.tapped)[rows, opp]
    )
    no_response = ~(
        np.any(response_hand, axis=1)
        | np.any(response_board, axis=1)
        | np.any(defender_cards, axis=1)
    )
    no_kira_redirect = ~np.any(
        (opp_zone == self._zone_alley) & (opp_defs == self._azk01_034_id),
        axis=1,
    )

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    stt01_008_active = (
        passive_watch
        & (def_host == self._stt01_008_id)
        & (
            (np.asarray(states.passive_atk) != 0)
            | (np.asarray(states.passive_hp) != 0)
        )
    )
    passive_watch_ok = (
        ~np.any(passive_watch & (def_host != self._stt01_008_id), axis=(1, 2))
        & ~np.any(stt01_008_active, axis=(1, 2))
    )

    return (
        base
        & attacker_exists
        & defender_exists
        & (defender_index >= 0)
        & (defender_index <= self._garden_size)
        & (attacker_def == self._azk01_060_id)
        & (defender_def >= 0)
        & (
            target_is_leader
            | (self._card_type[safe_defender_def] == self._card_type_entity)
        )
    )

  def _attack_stt01_006_effect_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attack)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    attacker_index = active_acts[:, 1]
    defender_index = active_acts[:, 2]
    atk_zone = zone_host[rows, active]
    atk_zpos = zpos_host[rows, active]
    atk_defs = def_host[rows, active]
    def_zone = zone_host[rows, opp]
    def_defs = def_host[rows, opp]

    attacker_match = (atk_zone == self._zone_garden) & (
        atk_zpos == attacker_index[:, None]
    )
    attacker_exists = np.any(attacker_match, axis=1)
    attacker_inst = np.argmax(attacker_match, axis=1)
    attacker_def = atk_defs[rows, attacker_inst]
    safe_attacker_def = np.maximum(attacker_def, 0)

    target_is_leader = defender_index == self._garden_size
    garden_match = (def_zone == self._zone_garden) & (
        zpos_host[rows, opp] == defender_index[:, None]
    )
    leader_match = def_zone == self._zone_leader
    garden_exists = np.any(garden_match, axis=1)
    leader_exists = np.any(leader_match, axis=1)
    garden_inst = np.argmax(garden_match, axis=1)
    leader_inst = np.argmax(leader_match, axis=1)
    defender_exists = np.where(target_is_leader, leader_exists, garden_exists)
    defender_inst = np.where(target_is_leader, leader_inst, garden_inst)
    defender_def = def_defs[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)

    attached_to = np.asarray(states.attached_to)
    attacker_attached = (atk_zone == self._zone_attached) & (
        attached_to[rows, active] == attacker_inst[:, None]
    )
    attached_defs = np.where(attacker_attached, atk_defs, -1)
    safe_attached_defs = np.maximum(attached_defs, 0)
    no_extra_attack_triggers = ~(
        np.any(
            self._timing_when_attacking[safe_attached_defs]
            & attacker_attached,
            axis=1,
        )
        | np.any(
            attacker_attached & (attached_defs == self._azk01_044_id),
            axis=1,
        )
    )
    no_attached = ~np.any(attacker_attached, axis=1)
    source_ok = (
        (attacker_def == self._stt01_006_id)
        & self._timing_when_attacking[safe_attacker_def]
        & self._implemented[safe_attacker_def]
        & ~self._timing_after_attacking[safe_attacker_def]
        & ((np.asarray(states.once_per_turn_used)[
            rows, active, attacker_inst
        ] & 1) == 0)
    )

    enemy_safe_defs = np.maximum(def_defs, 0)
    enemy_target = (
        ((def_zone == self._zone_garden) | (def_zone == self._zone_leader))
        & (def_defs >= 0)
        & (np.asarray(states.effect_immune_dur)[rows, opp] == 0)
    )
    effect_available = np.any(enemy_target, axis=1)

    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    passive_watch_ok = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    return (
        base
        & (active_acts[:, 0] == self._act_attack)
        & attacker_exists
        & defender_exists
        & (defender_index >= 0)
        & (defender_index <= self._garden_size)
        & source_ok
        & no_extra_attack_triggers
        & no_attached
        & effect_available
        & passive_watch_ok
    )

  def _declare_defender_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    active_safe = np.clip(active, 0, 1)
    active_acts = acts[rows, active_safe]
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_declare_defender)
        & (active_acts[:, 0] == self._act_declare_defender)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) >= 0)
        & (np.asarray(states.combat_defender_player) == active)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    defender_index = active_acts[:, 1].astype(np.int32, copy=False)
    zone_row = zone_host[rows, active_safe]
    zpos_row = zpos_host[rows, active_safe]
    def_row = def_host[rows, active_safe]
    defender_match = (
        (zone_row == self._zone_garden)
        & (zpos_row == defender_index[:, None])
        & (defender_index[:, None] >= 0)
        & (defender_index[:, None] < self._garden_size)
    )
    defender_exists = np.any(defender_match, axis=1)
    defender_inst = np.argmax(defender_match, axis=1)
    defender_def = def_row[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)
    defender_kw = (
        self._inherent_defender[safe_defender_def]
        | np.asarray(states.grant_defender)[rows, active_safe, defender_inst]
    )

    attacker_p = (active_safe + 1) % 2
    attacker = np.asarray(states.combat_attacker).astype(np.int32, copy=False)
    safe_attacker = np.maximum(attacker, 0)
    attacker_def = def_host[rows, attacker_p, safe_attacker]
    safe_attacker_def = np.maximum(attacker_def, 0)
    attacker_infiltrate = (
        self._inherent_infiltrate[safe_attacker_def]
        | np.asarray(states.grant_infiltrate)[rows, attacker_p, safe_attacker]
    )

    return (
        base
        & defender_exists
        & (defender_def >= 0)
        & defender_kw
        & ~tapped_host[rows, active_safe, defender_inst]
        & ~attacker_infiltrate
    )

  def _attack_entity_mutual_destroy_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    passive_clean = (
        (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attack)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & passive_clean
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    attacker_index = active_acts[:, 1]
    defender_index = active_acts[:, 2]
    atk_zone = zone_host[rows, active]
    atk_zpos = zpos_host[rows, active]
    atk_defs = def_host[rows, active]
    def_zone = zone_host[rows, opp]
    def_zpos = zpos_host[rows, opp]
    def_defs = def_host[rows, opp]

    attacker_match = (atk_zone == self._zone_garden) & (
        atk_zpos == attacker_index[:, None]
    )
    attacker_exists = np.any(attacker_match, axis=1)
    attacker_inst = np.argmax(attacker_match, axis=1)
    attacker_def = atk_defs[rows, attacker_inst]
    safe_attacker_def = np.maximum(attacker_def, 0)

    target_is_garden = defender_index < self._garden_size
    defender_match = (def_zone == self._zone_garden) & (
        def_zpos == defender_index[:, None]
    )
    defender_exists = target_is_garden & np.any(defender_match, axis=1)
    defender_inst = np.argmax(defender_match, axis=1)
    defender_def = def_defs[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)

    old_exact_cards = (
        (attacker_def == self._azk01_058_id)
        & (defender_def == self._stt01_003_id)
    )
    attacker_azk01_062_fizzle = (
        (attacker_def == self._azk01_062_id)
        & (defender_def == self._stt01_008_id)
        & (np.asarray(states.redirect_count) == 0)
    )
    defender_azk01_062_fizzle = (
        (attacker_def == self._stt01_008_id)
        & (defender_def == self._azk01_062_id)
        & (np.asarray(states.redirect_count) == 0)
    )
    azk01_062_fizzle = (
        attacker_azk01_062_fizzle | defender_azk01_062_fizzle
    )
    exact_cards = old_exact_cards | azk01_062_fizzle
    attached_to = np.asarray(states.attached_to)
    no_attached = (
        ~np.any(
            (atk_zone == self._zone_attached)
            & (attached_to[rows, active] == attacker_inst[:, None]),
            axis=1,
        )
        & ~np.any(
            (def_zone == self._zone_attached)
            & (attached_to[rows, opp] == defender_inst[:, None]),
            axis=1,
        )
    )

    cur_atk = np.asarray(states.cur_atk).astype(np.int16, copy=False)
    cur_hp = np.asarray(states.cur_hp).astype(np.int16, copy=False)
    damage_to_defender = cur_atk[rows, active, attacker_inst]
    damage_to_attacker = cur_atk[rows, opp, defender_inst]
    attacker_hp = cur_hp[rows, active, attacker_inst]
    defender_hp = cur_hp[rows, opp, defender_inst]
    mutual_destroy = (
        (damage_to_defender > 0)
        & (damage_to_attacker > 0)
        & (damage_to_defender >= defender_hp)
        & (damage_to_attacker >= attacker_hp)
    )

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, opp, defender_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active, attacker_inst]
        & ~np.asarray(states.grant_godmode)[rows, opp, defender_inst]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    old_trigger_ok = (
        old_exact_cards
        & ~self._timing_when_attacking[safe_attacker_def]
        & self._timing_after_attacking[safe_attacker_def]
        & ~self._timing_when_attacked[safe_defender_def]
        & ~self._timing_takes_damage[safe_attacker_def]
        & ~self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~self._timing_when_destroyed[safe_attacker_def]
        & ~self._timing_when_destroyed[safe_defender_def]
    )
    attacker_azk01_062_trigger_ok = (
        attacker_azk01_062_fizzle
        & ~self._timing_when_attacking[safe_attacker_def]
        & ~self._timing_after_attacking[safe_attacker_def]
        & ~self._timing_when_attacked[safe_defender_def]
        & self._timing_takes_damage[safe_attacker_def]
        & ~self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~self._timing_when_destroyed[safe_attacker_def]
        & ~self._timing_when_destroyed[safe_defender_def]
    )
    defender_azk01_062_trigger_ok = (
        defender_azk01_062_fizzle
        & ~self._timing_when_attacking[safe_attacker_def]
        & ~self._timing_after_attacking[safe_attacker_def]
        & ~self._timing_when_attacked[safe_defender_def]
        & ~self._timing_takes_damage[safe_attacker_def]
        & self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~self._timing_when_destroyed[safe_attacker_def]
        & ~self._timing_when_destroyed[safe_defender_def]
    )
    defender_unimplemented_when_attacked = (
        self._timing_when_attacked[safe_defender_def]
        & self._has_ability[safe_defender_def]
        & ~self._implemented[safe_defender_def]
    )
    defender_azk01_036_when_attacked = (
        (defender_def == self._azk01_036_id)
        & self._timing_when_attacked[safe_defender_def]
        & self._has_ability[safe_defender_def]
        & self._implemented[safe_defender_def]
    )
    simple_trigger_ok = (
        ~self._timing_when_attacking[safe_attacker_def]
        & ~self._timing_after_attacking[safe_attacker_def]
        & (
            ~self._timing_when_attacked[safe_defender_def]
            | defender_unimplemented_when_attacked
            | defender_azk01_036_when_attacked
        )
        & ~self._timing_takes_damage[safe_attacker_def]
        & ~self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~self._timing_when_destroyed[safe_attacker_def]
        & ~self._timing_when_destroyed[safe_defender_def]
    )
    trigger_ok = (
        old_trigger_ok
        | attacker_azk01_062_trigger_ok
        | defender_azk01_062_trigger_ok
        | simple_trigger_ok
    )

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    in_hand = opp_zone == self._zone_hand
    opp_tapped = np.asarray(states.tapped)[rows, opp]
    token_ready = (
        (opp_zone[:, self._token_instance] == self._zone_token)
        & ~opp_tapped[:, self._token_instance]
    )
    payment_sources = np.sum(
        (opp_zone == self._zone_ikz_area) & ~opp_tapped,
        axis=1,
    )
    payment_sources += np.sum(
        (opp_zone == self._zone_garden)
        & ~opp_tapped
        & self._counts_as_ikz[opp_safe_defs],
        axis=1,
    )
    payment_sources += token_ready.astype(np.int32)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, opp
    ].astype(np.int32, copy=False)
    response_cost = np.maximum(
        self._ikz_cost[opp_safe_defs].astype(np.int32) - next_reduction[:, None],
        0,
    )
    response_spell = (
        in_hand
        & (self._card_type[opp_safe_defs] == self._card_type_spell)
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_from_hand = (
        in_hand
        & self._response_play_from_hand[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_hand = response_spell | response_from_hand
    in_board = (
        (opp_zone == self._zone_garden)
        | (opp_zone == self._zone_alley)
        | (opp_zone == self._zone_leader)
    )
    response_board = (
        in_board
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
    )
    defender_cards = (
        (opp_zone == self._zone_garden)
        & (
            self._inherent_defender[opp_safe_defs]
            | np.asarray(states.grant_defender)[rows, opp]
        )
        & ~np.asarray(states.tapped)[rows, opp]
    )
    no_response = ~(
        np.any(response_hand, axis=1)
        | np.any(response_board, axis=1)
        | np.any(defender_cards, axis=1)
    )
    no_kira_redirect = ~np.any(
        (opp_zone == self._zone_alley) & (opp_defs == self._azk01_034_id),
        axis=1,
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    no_passive_watch_strict = ~np.any(passive_watch, axis=(1, 2))
    no_passive_watch_stt01_008_inert = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    passive_watch_ok = (
        (old_exact_cards & no_passive_watch_strict)
        | (azk01_062_fizzle & no_passive_watch_stt01_008_inert)
        | (simple_trigger_ok & no_passive_watch_stt01_008_inert)
    )

    return (
        base
        & attacker_exists
        & defender_exists
        & no_attached
        & no_modifiers
        & trigger_ok
        & no_response
        & no_kira_redirect
        & passive_watch_ok
    )

  def _attack_leader_response_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attack)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    attacker_index = active_acts[:, 1]
    defender_index = active_acts[:, 2]
    atk_zone = zone_host[rows, active]
    atk_zpos = zpos_host[rows, active]
    atk_defs = def_host[rows, active]
    def_zone = zone_host[rows, opp]
    def_defs = def_host[rows, opp]
    def_tapped = tapped_host[rows, opp]

    leader_attacker_match = atk_zone == self._zone_leader
    garden_attacker_match = (atk_zone == self._zone_garden) & (
        atk_zpos == attacker_index[:, None]
    )
    attacker_match = np.where(
        (attacker_index == self._garden_size)[:, None],
        leader_attacker_match,
        garden_attacker_match,
    )
    attacker_exists = np.any(attacker_match, axis=1)
    attacker_inst = np.argmax(attacker_match, axis=1)
    attacker_def = atk_defs[rows, attacker_inst]
    safe_attacker_def = np.maximum(attacker_def, 0)

    leader_match = def_zone == self._zone_leader
    defender_exists = np.any(leader_match, axis=1)
    defender_inst = np.argmax(leader_match, axis=1)
    defender_def = def_defs[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)
    defender_is_leader_target = defender_index == self._garden_size

    attached = (atk_zone == self._zone_attached) & (
        np.asarray(states.attached_to)[rows, active] == attacker_inst[:, None]
    )
    attached_defs = np.where(attached, atk_defs, -1)
    safe_attached_defs = np.maximum(attached_defs, 0)
    no_attacking_triggers = ~(
        self._timing_when_attacking[safe_attacker_def]
        | self._timing_after_attacking[safe_attacker_def]
        | np.any(self._timing_when_attacking[safe_attached_defs] & attached, axis=1)
        | np.any(attached & (attached_defs == self._azk01_044_id), axis=1)
    )
    cur_atk = np.asarray(states.cur_atk)
    cur_hp = np.asarray(states.cur_hp)
    damage = cur_atk[rows, active, attacker_inst].astype(np.int16)
    defender_atk = cur_atk[rows, opp, defender_inst].astype(np.int16)
    defender_hp = cur_hp[rows, opp, defender_inst].astype(np.int16)
    no_defender_triggers = ~(
        self._timing_when_attacked[safe_defender_def]
        | self._timing_takes_damage[safe_defender_def]
        | self._timing_deals_damage[safe_defender_def]
        | (
            self._timing_takes_damage[safe_attacker_def]
            & (defender_atk != 0)
        )
        | self._timing_deals_damage[safe_attacker_def]
    )

    nonlethal = (damage >= 0) & (damage < defender_hp) & (defender_atk == 0)

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, opp, defender_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active, attacker_inst]
        & ~np.asarray(states.grant_godmode)[rows, opp, defender_inst]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    response_board = (
        (
            (opp_zone == self._zone_garden)
            | (opp_zone == self._zone_alley)
            | (opp_zone == self._zone_leader)
        )
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
    )
    defender_cards = (
        (opp_zone == self._zone_garden)
        & (
            self._inherent_defender[opp_safe_defs]
            | np.asarray(states.grant_defender)[rows, opp]
        )
        & ~def_tapped
    )
    token_ready = (
        (opp_zone[:, self._token_instance] == self._zone_token)
        & ~def_tapped[:, self._token_instance]
    )
    payment_sources = np.sum(
        (opp_zone == self._zone_ikz_area) & ~def_tapped,
        axis=1,
    )
    payment_sources += np.sum(
        (opp_zone == self._zone_garden)
        & ~def_tapped
        & self._counts_as_ikz[opp_safe_defs],
        axis=1,
    )
    payment_sources += token_ready.astype(np.int32)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, opp
    ].astype(np.int32, copy=False)
    response_cost = np.maximum(
        self._ikz_cost[opp_safe_defs].astype(np.int32) - next_reduction[:, None],
        0,
    )
    in_hand = opp_zone == self._zone_hand
    response_spell = (
        in_hand
        & (self._card_type[opp_safe_defs] == self._card_type_spell)
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_from_hand = (
        in_hand
        & self._response_play_from_hand[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_hand = response_spell | response_from_hand
    azk01_127_in_hand = (opp_zone == self._zone_hand) & (
        opp_defs == self._azk01_127_id
    )
    active_garden_entity = (
        (atk_zone == self._zone_garden)
        & (atk_defs >= 0)
        & (self._card_type[np.maximum(atk_defs, 0)] == self._card_type_entity)
    )
    has_azk01_127_response = (
        np.any(azk01_127_in_hand, axis=1)
        & (payment_sources >= int(self._ikz_cost[self._azk01_127_id]))
        & np.any(active_garden_entity, axis=1)
    )
    no_other_response_shape = (
        ~np.any(response_hand, axis=1)
        & ~np.any(response_board, axis=1)
        & ~np.any(defender_cards, axis=1)
    )

    azk01_127_response_shape = (
        base
        & (active_acts[:, 0] == self._act_attack)
        & attacker_exists
        & defender_exists
        & defender_is_leader_target
        & (attacker_def == self._azk01_036_id)
        & (defender_def == self._azk01_121_id)
        & no_attacking_triggers
        & no_defender_triggers
        & nonlethal
        & no_modifiers
        & has_azk01_127_response
        & no_other_response_shape
    )
    board_response_shape = (
        base
        & (active_acts[:, 0] == self._act_attack)
        & attacker_exists
        & defender_exists
        & defender_is_leader_target
        & no_attacking_triggers
        & no_defender_triggers
        & nonlethal
        & no_modifiers
        & (
            np.any(response_hand, axis=1)
            | np.any(response_board, axis=1)
            | np.any(defender_cards, axis=1)
        )
    )
    return azk01_127_response_shape | board_response_shape

  def _attack_entity_response_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)
    tapped_host = np.asarray(states.tapped)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attack)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    attacker_index = active_acts[:, 1]
    defender_index = active_acts[:, 2]
    atk_zone = zone_host[rows, active]
    atk_zpos = zpos_host[rows, active]
    atk_defs = def_host[rows, active]
    def_zone = zone_host[rows, opp]
    def_zpos = zpos_host[rows, opp]
    def_defs = def_host[rows, opp]
    def_tapped = tapped_host[rows, opp]

    attacker_match = (atk_zone == self._zone_garden) & (
        atk_zpos == attacker_index[:, None]
    )
    attacker_exists = np.any(attacker_match, axis=1)
    attacker_inst = np.argmax(attacker_match, axis=1)
    attacker_def = atk_defs[rows, attacker_inst]
    safe_attacker_def = np.maximum(attacker_def, 0)

    target_is_garden = defender_index < self._garden_size
    defender_match = (def_zone == self._zone_garden) & (
        def_zpos == defender_index[:, None]
    )
    defender_exists = target_is_garden & np.any(defender_match, axis=1)
    defender_inst = np.argmax(defender_match, axis=1)
    defender_def = def_defs[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)

    attached_to = np.asarray(states.attached_to)
    attacker_attached = (atk_zone == self._zone_attached) & (
        attached_to[rows, active] == attacker_inst[:, None]
    )
    attached_defs = np.where(attacker_attached, atk_defs, -1)
    safe_attached_defs = np.maximum(attached_defs, 0)
    defender_azk01_040_when_attacked = (
        (defender_def == self._azk01_040_id)
        & self._timing_when_attacked[safe_defender_def]
        & self._has_ability[safe_defender_def]
        & self._implemented[safe_defender_def]
    )
    no_attack_declaration_triggers = ~(
        self._timing_when_attacking[safe_attacker_def]
        | self._timing_after_attacking[safe_attacker_def]
        | (
            self._timing_when_attacked[safe_defender_def]
            & ~defender_azk01_040_when_attacked
        )
        | np.any(
            self._timing_when_attacking[safe_attached_defs] & attacker_attached,
            axis=1,
        )
        | np.any(
            attacker_attached & (attached_defs == self._azk01_044_id),
            axis=1,
        )
    )
    no_attached = ~np.any(attacker_attached, axis=1) & ~np.any(
        (def_zone == self._zone_attached)
        & (attached_to[rows, opp] == defender_inst[:, None]),
        axis=1,
    )
    cur_atk = np.asarray(states.cur_atk).astype(np.int16, copy=False)
    cur_hp = np.asarray(states.cur_hp).astype(np.int16, copy=False)
    damage_to_defender = cur_atk[rows, active, attacker_inst]
    damage_to_attacker = cur_atk[rows, opp, defender_inst]
    attacker_hp = cur_hp[rows, active, attacker_inst]
    defender_hp = cur_hp[rows, opp, defender_inst]
    attacker_after = attacker_hp - damage_to_attacker
    defender_after = defender_hp - damage_to_defender
    clean_damage = (
        (damage_to_defender > 0)
        & (damage_to_attacker >= 0)
        & (attacker_hp > 0)
        & (defender_hp > 0)
    )
    attacker_dies = clean_damage & (attacker_after <= 0)
    defender_dies = clean_damage & (defender_after <= 0)
    no_passive_death_watch = (
        ~np.any(np.asarray(states.passive_observer_registered), axis=(1, 2))
        & ~np.any(
            (zone_host == self._zone_garden)
            & (def_host == self._stt02_012_id),
            axis=(1, 2),
        )
    )
    azk01_040_combat_clean = (
        defender_azk01_040_when_attacked
        & clean_damage
        & ~self._timing_takes_damage[safe_attacker_def]
        & ~self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~(self._timing_when_destroyed[safe_attacker_def] & attacker_dies)
        & ~(self._timing_when_destroyed[safe_defender_def] & defender_dies)
        & ((attacker_after > 0) | (attacker_dies & no_passive_death_watch))
        & ((defender_after > 0) | (defender_dies & no_passive_death_watch))
    )
    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, opp, defender_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active, attacker_inst]
        & ~np.asarray(states.grant_godmode)[rows, opp, defender_inst]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    token_ready = (
        (def_zone[:, self._token_instance] == self._zone_token)
        & ~def_tapped[:, self._token_instance]
    )
    payment_sources = np.sum(
        (def_zone == self._zone_ikz_area) & ~def_tapped,
        axis=1,
    )
    def_safe_defs = np.maximum(def_defs, 0)
    payment_sources += np.sum(
        (def_zone == self._zone_garden)
        & ~def_tapped
        & self._counts_as_ikz[def_safe_defs],
        axis=1,
    )
    payment_sources += token_ready.astype(np.int32)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, opp
    ].astype(np.int32, copy=False)
    response_cost = np.maximum(
        self._ikz_cost[def_safe_defs].astype(np.int32) - next_reduction[:, None],
        0,
    )
    in_hand = def_zone == self._zone_hand
    response_spell = (
        in_hand
        & (self._card_type[def_safe_defs] == self._card_type_spell)
        & self._timing_is_response[def_safe_defs]
        & self._has_ability[def_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_from_hand = (
        in_hand
        & self._response_play_from_hand[def_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_board = (
        (
            (def_zone == self._zone_garden)
            | (def_zone == self._zone_alley)
            | (def_zone == self._zone_leader)
        )
        & self._timing_is_response[def_safe_defs]
        & self._has_ability[def_safe_defs]
    )
    defender_cards = (
        (def_zone == self._zone_garden)
        & (
            self._inherent_defender[def_safe_defs]
            | np.asarray(states.grant_defender)[rows, opp]
        )
        & ~def_tapped
    )
    has_response = (
        np.any(response_spell | response_from_hand, axis=1)
        | np.any(response_board, axis=1)
        | np.any(defender_cards, axis=1)
        | azk01_040_combat_clean
    )
    no_kira_redirect = ~np.any(
        (def_zone == self._zone_alley) & (def_defs == self._azk01_034_id),
        axis=1,
    )
    board = (zone_host == self._zone_garden) | (zone_host == self._zone_alley)
    passive_watch = board & np.isin(def_host, self._simple_play_watch_ids)
    passive_watch_ok = ~np.any(
        passive_watch & (def_host != self._stt01_008_id),
        axis=(1, 2),
    )
    return (
        base
        & (active_acts[:, 0] == self._act_attack)
        & attacker_exists
        & defender_exists
        & no_attached
        & no_attack_declaration_triggers
        & has_response
        & (~defender_azk01_040_when_attacked | (azk01_040_combat_clean & no_modifiers))
        & no_kira_redirect
        & passive_watch_ok
    )

  def _attack_azk01_004_leader_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attack)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    attacker_index = active_acts[:, 1]
    defender_index = active_acts[:, 2]
    atk_zone = zone_host[rows, active]
    atk_zpos = zpos_host[rows, active]
    atk_defs = def_host[rows, active]
    def_zone = zone_host[rows, opp]
    def_defs = def_host[rows, opp]

    attacker_match = (atk_zone == self._zone_garden) & (
        atk_zpos == attacker_index[:, None]
    )
    attacker_exists = np.any(attacker_match, axis=1)
    attacker_inst = np.argmax(attacker_match, axis=1)
    attacker_def = atk_defs[rows, attacker_inst]
    safe_attacker_def = np.maximum(attacker_def, 0)

    leader_match = def_zone == self._zone_leader
    defender_exists = np.any(leader_match, axis=1)
    defender_inst = np.argmax(leader_match, axis=1)
    defender_def = def_defs[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)
    defender_is_leader_target = defender_index == self._garden_size

    attached_to = np.asarray(states.attached_to)
    attacker_attached = (atk_zone == self._zone_attached) & (
        attached_to[rows, active] == attacker_inst[:, None]
    )
    defender_attached = (def_zone == self._zone_attached) & (
        attached_to[rows, opp] == defender_inst[:, None]
    )
    no_attached = ~np.any(attacker_attached, axis=1) & ~np.any(
        defender_attached, axis=1
    )

    source_ok = (
        (attacker_def == self._azk01_004_id)
        & self._timing_when_attacking[safe_attacker_def]
        & self._implemented[safe_attacker_def]
        & ~self._timing_after_attacking[safe_attacker_def]
    )

    cur_atk = np.asarray(states.cur_atk)
    cur_hp = np.asarray(states.cur_hp)
    damage = np.maximum(
        cur_atk[rows, active, attacker_inst].astype(np.int16) + 1,
        0,
    )
    defender_atk = cur_atk[rows, opp, defender_inst].astype(np.int16)
    defender_hp = cur_hp[rows, opp, defender_inst].astype(np.int16)
    nonlethal = (damage > 0) & (damage < defender_hp) & (defender_atk == 0)

    no_defender_triggers = ~(
        self._timing_when_attacked[safe_defender_def]
        | self._timing_takes_damage[safe_defender_def]
        | self._timing_deals_damage[safe_defender_def]
        | (
            self._timing_takes_damage[safe_attacker_def]
            & (defender_atk != 0)
        )
        | self._timing_deals_damage[safe_attacker_def]
        | self._timing_when_destroyed[safe_attacker_def]
        | self._timing_when_destroyed[safe_defender_def]
    )

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, opp, defender_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active, attacker_inst]
        & ~np.asarray(states.grant_godmode)[rows, opp, defender_inst]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    in_hand = opp_zone == self._zone_hand
    opp_tapped = np.asarray(states.tapped)[rows, opp]
    token_ready = (
        (opp_zone[:, self._token_instance] == self._zone_token)
        & ~opp_tapped[:, self._token_instance]
    )
    payment_sources = np.sum(
        (opp_zone == self._zone_ikz_area) & ~opp_tapped,
        axis=1,
    )
    payment_sources += np.sum(
        (opp_zone == self._zone_garden)
        & ~opp_tapped
        & self._counts_as_ikz[opp_safe_defs],
        axis=1,
    )
    payment_sources += token_ready.astype(np.int32)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, opp
    ].astype(np.int32, copy=False)
    response_cost = np.maximum(
        self._ikz_cost[opp_safe_defs].astype(np.int32) - next_reduction[:, None],
        0,
    )
    response_spell = (
        in_hand
        & (self._card_type[opp_safe_defs] == self._card_type_spell)
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_from_hand = (
        in_hand
        & self._response_play_from_hand[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_hand = response_spell | response_from_hand
    in_board = (
        (opp_zone == self._zone_garden)
        | (opp_zone == self._zone_alley)
        | (opp_zone == self._zone_leader)
    )
    response_board = (
        in_board
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
    )
    defender_cards = (
        (opp_zone == self._zone_garden)
        & (
            self._inherent_defender[opp_safe_defs]
            | np.asarray(states.grant_defender)[rows, opp]
        )
        & ~opp_tapped
    )
    no_response = ~(
        np.any(response_hand, axis=1)
        | np.any(response_board, axis=1)
        | np.any(defender_cards, axis=1)
    )

    return (
        base
        & (active_acts[:, 0] == self._act_attack)
        & attacker_exists
        & defender_exists
        & defender_is_leader_target
        & source_ok
        & no_attached
        & no_defender_triggers
        & nonlethal
        & no_modifiers
        & no_response
    )

  def _attack_leader_simple_fast_mask(self, acts: np.ndarray, chosen, phase):
    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    opp = (active + 1) % 2
    zone_host = np.asarray(states.zone)
    zpos_host = np.asarray(states.zpos)
    def_host = np.asarray(states.def_id)

    base = (
        (phase == 2)  # Phase.MAIN
        & (chosen == self._act_attack)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (np.asarray(states.combat_attacker) < 0)
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, np.clip(active, 0, 1)]
    attacker_index = active_acts[:, 1]
    defender_index = active_acts[:, 2]
    atk_zone = zone_host[rows, active]
    atk_zpos = zpos_host[rows, active]
    atk_defs = def_host[rows, active]
    def_zone = zone_host[rows, opp]
    def_defs = def_host[rows, opp]

    leader_attacker_match = atk_zone == self._zone_leader
    garden_attacker_match = (atk_zone == self._zone_garden) & (
        atk_zpos == attacker_index[:, None]
    )
    attacker_match = np.where(
        (attacker_index == self._garden_size)[:, None],
        leader_attacker_match,
        garden_attacker_match,
    )
    attacker_exists = np.any(attacker_match, axis=1)
    attacker_inst = np.argmax(attacker_match, axis=1)
    attacker_def = atk_defs[rows, attacker_inst]
    safe_attacker_def = np.maximum(attacker_def, 0)
    azk01_058_confirm = attacker_def == self._azk01_058_id

    leader_match = def_zone == self._zone_leader
    defender_exists = np.any(leader_match, axis=1)
    defender_inst = np.argmax(leader_match, axis=1)
    defender_def = def_defs[rows, defender_inst]
    safe_defender_def = np.maximum(defender_def, 0)
    defender_is_leader_target = defender_index == self._garden_size

    attached = (atk_zone == self._zone_attached) & (
        np.asarray(states.attached_to)[rows, active] == attacker_inst[:, None]
    )
    attached_defs = np.where(attached, atk_defs, -1)
    safe_attached_defs = np.maximum(attached_defs, 0)
    cur_atk = np.asarray(states.cur_atk)
    cur_hp = np.asarray(states.cur_hp)
    damage = cur_atk[rows, active, attacker_inst].astype(np.int16)
    defender_atk = cur_atk[rows, opp, defender_inst].astype(np.int16)
    defender_hp = cur_hp[rows, opp, defender_inst].astype(np.int16)
    no_attacking_triggers = ~(
        self._timing_when_attacking[safe_attacker_def]
        | (self._timing_after_attacking[safe_attacker_def] & ~azk01_058_confirm)
        | np.any(self._timing_when_attacking[safe_attached_defs] & attached, axis=1)
        | np.any(attached & (attached_defs == self._azk01_044_id), axis=1)
    )
    no_defender_triggers = ~(
        self._timing_when_attacked[safe_defender_def]
        | self._timing_takes_damage[safe_defender_def]
        | self._timing_deals_damage[safe_defender_def]
        | (
            self._timing_takes_damage[safe_attacker_def]
            & (defender_atk != 0)
        )
        | self._timing_deals_damage[safe_attacker_def]
    )

    nonlethal = (damage >= 0) & (damage < defender_hp) & (defender_atk == 0)

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, active, attacker_inst] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_perm)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.carapace_eot)[rows, opp, defender_inst] == 0)
        & (np.asarray(states.frozen_dur)[rows, opp, defender_inst] == 0)
        & ~np.asarray(states.grant_godmode)[rows, active, attacker_inst]
        & ~np.asarray(states.grant_godmode)[rows, opp, defender_inst]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    opp_zone = zone_host[rows, opp]
    opp_defs = def_host[rows, opp]
    opp_safe_defs = np.maximum(opp_defs, 0)
    in_hand = opp_zone == self._zone_hand
    opp_tapped = np.asarray(states.tapped)[rows, opp]
    token_ready = (
        (opp_zone[:, self._token_instance] == self._zone_token)
        & ~opp_tapped[:, self._token_instance]
    )
    payment_sources = np.sum(
        (opp_zone == self._zone_ikz_area) & ~opp_tapped,
        axis=1,
    )
    payment_sources += np.sum(
        (opp_zone == self._zone_garden)
        & ~opp_tapped
        & self._counts_as_ikz[opp_safe_defs],
        axis=1,
    )
    payment_sources += token_ready.astype(np.int32)
    next_reduction = np.asarray(states.next_play_cost_reduction)[
        rows, opp
    ].astype(np.int32, copy=False)
    response_cost = np.maximum(
        self._ikz_cost[opp_safe_defs].astype(np.int32) - next_reduction[:, None],
        0,
    )
    response_spell = (
        in_hand
        & (self._card_type[opp_safe_defs] == self._card_type_spell)
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_from_hand = (
        in_hand
        & self._response_play_from_hand[opp_safe_defs]
        & (response_cost <= payment_sources[:, None])
    )
    response_hand = response_spell | response_from_hand
    in_board = (
        (opp_zone == self._zone_garden)
        | (opp_zone == self._zone_alley)
        | (opp_zone == self._zone_leader)
    )
    response_board = (
        in_board
        & self._timing_is_response[opp_safe_defs]
        & self._has_ability[opp_safe_defs]
    )
    defender_cards = (
        (opp_zone == self._zone_garden)
        & (self._inherent_defender[opp_safe_defs] | np.asarray(states.grant_defender)[rows, opp])
        & ~opp_tapped
    )
    no_response = ~(
        np.any(response_hand, axis=1)
        | np.any(response_board, axis=1)
        | np.any(defender_cards, axis=1)
    )

    return (
        base
        & attacker_exists
        & defender_exists
        & defender_is_leader_target
        & no_attacking_triggers
        & no_defender_triggers
        & nonlethal
        & no_modifiers
        & no_response
    )

  def _response_noop_leader_combat_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    defender = np.asarray(states.active_player).astype(np.int32, copy=False)
    attacker_p = (defender + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    combat_attacker = np.asarray(states.combat_attacker).astype(
        np.int32, copy=False
    )
    combat_defender = np.asarray(states.combat_defender).astype(
        np.int32, copy=False
    )

    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_noop)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (combat_attacker >= 0)
        & (combat_defender >= 0)
        & (
            np.asarray(states.combat_defender_player).astype(
                np.int32, copy=False
            )
            == defender
        )
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, defender]
    safe_attacker = np.maximum(combat_attacker, 0)
    safe_defender = np.maximum(combat_defender, 0)

    attacker_zone = zone_host[rows, attacker_p, safe_attacker]
    defender_zone = zone_host[rows, defender, safe_defender]
    attacker_is_leader = np.asarray(states.combat_attacker_is_leader)
    attacker_def = def_host[rows, attacker_p, safe_attacker]
    defender_def = def_host[rows, defender, safe_defender]
    safe_attacker_def = np.maximum(attacker_def, 0)
    safe_defender_def = np.maximum(defender_def, 0)

    attacker_attached = (
        (zone_host[rows, attacker_p] == self._zone_attached)
        & (
            np.asarray(states.attached_to)[rows, attacker_p]
            == safe_attacker[:, None]
        )
    )
    attached_defs = np.where(
        attacker_attached, def_host[rows, attacker_p], -1
    )
    safe_attached_defs = np.maximum(attached_defs, 0)
    no_attacking_triggers = ~(
        self._timing_when_attacking[safe_attacker_def]
        | self._timing_after_attacking[safe_attacker_def]
        | np.any(
            self._timing_when_attacking[safe_attached_defs]
            & attacker_attached,
            axis=1,
        )
        | np.any(
            attacker_attached & (attached_defs == self._azk01_044_id),
            axis=1,
        )
    )
    no_defender_triggers = ~(
        self._timing_when_attacked[safe_defender_def]
        | self._timing_takes_damage[safe_defender_def]
        | self._timing_deals_damage[safe_defender_def]
        | self._timing_takes_damage[safe_attacker_def]
        | self._timing_deals_damage[safe_attacker_def]
    )

    cur_atk = np.asarray(states.cur_atk)
    cur_hp = np.asarray(states.cur_hp)
    damage = cur_atk[rows, attacker_p, safe_attacker].astype(np.int16)
    defender_atk = cur_atk[rows, defender, safe_defender].astype(np.int16)
    defender_hp = cur_hp[rows, defender, safe_defender].astype(np.int16)
    nonlethal = (damage >= 0) & (damage < defender_hp) & (defender_atk == 0)

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.carapace_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.carapace_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.carapace_perm)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.carapace_eot)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.frozen_dur)[rows, defender, safe_defender] == 0)
        & ~np.asarray(states.grant_godmode)[rows, attacker_p, safe_attacker]
        & ~np.asarray(states.grant_godmode)[rows, defender, safe_defender]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    return (
        base
        & (active_acts[:, 0] == self._act_noop)
        & (
            (attacker_zone == self._zone_garden)
            | (attacker_is_leader & (attacker_zone == self._zone_leader))
        )
        & np.asarray(states.tapped)[rows, attacker_p, safe_attacker]
        & (defender_zone == self._zone_leader)
        & no_attacking_triggers
        & no_defender_triggers
        & nonlethal
        & no_modifiers
    )

  def _response_noop_entity_combat_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    defender = np.asarray(states.active_player).astype(np.int32, copy=False)
    attacker_p = (defender + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    combat_attacker = np.asarray(states.combat_attacker).astype(
        np.int32, copy=False
    )
    combat_defender = np.asarray(states.combat_defender).astype(
        np.int32, copy=False
    )

    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_noop)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (combat_attacker >= 0)
        & (combat_defender >= 0)
        & (
            np.asarray(states.combat_defender_player).astype(
                np.int32, copy=False
            )
            == defender
        )
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, defender]
    safe_attacker = np.maximum(combat_attacker, 0)
    safe_defender = np.maximum(combat_defender, 0)
    attacker_zone = zone_host[rows, attacker_p, safe_attacker]
    defender_zone = zone_host[rows, defender, safe_defender]
    attacker_def = def_host[rows, attacker_p, safe_attacker]
    defender_def = def_host[rows, defender, safe_defender]
    safe_attacker_def = np.maximum(attacker_def, 0)
    safe_defender_def = np.maximum(defender_def, 0)

    attached_to = np.asarray(states.attached_to)
    no_attached = (
        ~np.any(
            (zone_host[rows, attacker_p] == self._zone_attached)
            & (attached_to[rows, attacker_p] == safe_attacker[:, None]),
            axis=1,
        )
        & ~np.any(
            (zone_host[rows, defender] == self._zone_attached)
            & (attached_to[rows, defender] == safe_defender[:, None]),
            axis=1,
        )
    )

    cur_atk = np.asarray(states.cur_atk)
    cur_hp = np.asarray(states.cur_hp)
    damage_to_attacker = cur_atk[rows, defender, safe_defender].astype(np.int16)
    damage_to_defender = cur_atk[rows, attacker_p, safe_attacker].astype(np.int16)
    attacker_hp = cur_hp[rows, attacker_p, safe_attacker].astype(np.int16)
    defender_hp = cur_hp[rows, defender, safe_defender].astype(np.int16)
    attacker_after = attacker_hp - damage_to_attacker
    defender_after = defender_hp - damage_to_defender
    attacker_deals_clean_damage = (
        (damage_to_defender > 0)
        & (damage_to_attacker >= 0)
        & (attacker_hp > 0)
        & (defender_hp > 0)
    )
    attacker_dies_after_combat = (
        attacker_deals_clean_damage
        & (attacker_after <= 0)
    )
    defender_dies_after_combat = (
        attacker_deals_clean_damage
        & (defender_after <= 0)
    )
    defender_survives_after_combat = (
        attacker_deals_clean_damage
        & (defender_after > 0)
    )
    no_passive_death_watch = (
        ~np.any(np.asarray(states.passive_observer_registered), axis=(1, 2))
        & ~np.any(
            (zone_host == self._zone_garden)
            & (def_host == self._stt02_012_id),
            axis=(1, 2),
        )
    )
    clean_defender_death = (
        defender_dies_after_combat
        & no_passive_death_watch
    )
    attacker_stt03_006_destroyed = (
        attacker_dies_after_combat
        & (attacker_def == self._stt03_006_id)
        & no_passive_death_watch
    )
    defender_stt03_006_destroyed = (
        defender_dies_after_combat
        & (defender_def == self._stt03_006_id)
        & no_passive_death_watch
    )
    disallowed_attacker_destroy = (
        self._timing_when_destroyed[safe_attacker_def]
        & attacker_dies_after_combat
        & ~attacker_stt03_006_destroyed
    )
    disallowed_defender_destroy = (
        self._timing_when_destroyed[safe_defender_def]
        & defender_dies_after_combat
        & ~defender_stt03_006_destroyed
    )

    attacker_pending_when_attacking = (
        self._timing_when_attacking[safe_attacker_def]
        & (attacker_def != self._azk01_060_id)
    )
    no_triggers = ~(
        attacker_pending_when_attacking
        | self._timing_after_attacking[safe_attacker_def]
        | self._timing_when_attacked[safe_defender_def]
        | self._timing_takes_damage[safe_attacker_def]
        | self._timing_deals_damage[safe_attacker_def]
        | self._timing_takes_damage[safe_defender_def]
        | self._timing_deals_damage[safe_defender_def]
        | disallowed_attacker_destroy
        | disallowed_defender_destroy
    )
    azk01_062_defender_fizzle = (
        (defender_def == self._azk01_062_id)
        & (np.asarray(states.redirect_count) == 0)
        & ~attacker_pending_when_attacking
        & ~self._timing_after_attacking[safe_attacker_def]
        & ~self._timing_when_attacked[safe_defender_def]
        & ~self._timing_takes_damage[safe_attacker_def]
        & ~self._timing_deals_damage[safe_attacker_def]
        & self._timing_takes_damage[safe_defender_def]
        & ~self._timing_deals_damage[safe_defender_def]
        & ~disallowed_attacker_destroy
        & ~disallowed_defender_destroy
    )
    trigger_ok = no_triggers | azk01_062_defender_fizzle

    no_modifiers = (
        (np.asarray(states.cmb_in_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.carapace_perm)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.carapace_eot)[rows, attacker_p, safe_attacker] == 0)
        & (np.asarray(states.cmb_in_perm)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.cmb_in_eot)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.cmb_out_perm)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.cmb_out_eot)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.carapace_perm)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.carapace_eot)[rows, defender, safe_defender] == 0)
        & (np.asarray(states.frozen_dur)[rows, defender, safe_defender] == 0)
        & ~np.asarray(states.grant_godmode)[rows, attacker_p, safe_attacker]
        & ~np.asarray(states.grant_godmode)[rows, defender, safe_defender]
        & ~self._inherent_godmode[safe_attacker_def]
        & ~self._inherent_godmode[safe_defender_def]
    )

    return (
        base
        & (active_acts[:, 0] == self._act_noop)
        & (attacker_zone == self._zone_garden)
        & (defender_zone == self._zone_garden)
        & (self._card_type[safe_attacker_def] == self._card_type_entity)
        & (self._card_type[safe_defender_def] == self._card_type_entity)
        & np.asarray(states.tapped)[rows, attacker_p, safe_attacker]
        & no_attached
        & trigger_ok
        & (defender_survives_after_combat | clean_defender_death)
        & no_modifiers
    )

  def _response_noop_azk01_040_fast_mask(
      self, acts: np.ndarray, chosen, phase
  ):
    states = self._states
    rows = np.arange(self.num_environments)
    defender = np.asarray(states.active_player).astype(np.int32, copy=False)
    defender_safe = np.clip(defender, 0, 1)
    attacker_p = (defender_safe + 1) % 2
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)
    combat_attacker = np.asarray(states.combat_attacker).astype(
        np.int32, copy=False
    )
    combat_defender = np.asarray(states.combat_defender).astype(
        np.int32, copy=False
    )

    base = (
        (phase == 3)  # Phase.RESPONSE_WINDOW
        & (chosen == self._act_noop)
        & (np.asarray(states.ab_phase) == 0)
        & (np.asarray(states.trig_count) == 0)
        & (np.asarray(states.redirect_count) == 0)
        & (combat_attacker >= 0)
        & (combat_defender >= 0)
        & (
            np.asarray(states.combat_defender_player).astype(
                np.int32, copy=False
            )
            == defender
        )
        & (np.asarray(states.winner) == -1)
        & ~np.asarray(states.eot_abilities_queued)
        & (np.asarray(states.passive_queue_count) == 0)
        & ~np.any(np.asarray(states.stt02_012_event_pending), axis=(1, 2))
    )
    if not np.any(base):
      return base

    active_acts = acts[rows, defender_safe]
    safe_attacker = np.maximum(combat_attacker, 0)
    safe_defender = np.maximum(combat_defender, 0)
    attacker_zone = zone_host[rows, attacker_p, safe_attacker]
    defender_zone = zone_host[rows, defender_safe, safe_defender]
    attacker_def = def_host[rows, attacker_p, safe_attacker]
    defender_def = def_host[rows, defender_safe, safe_defender]
    safe_attacker_def = np.maximum(attacker_def, 0)
    safe_defender_def = np.maximum(defender_def, 0)

    attached_to = np.asarray(states.attached_to)
    no_attached = (
        ~np.any(
            (zone_host[rows, attacker_p] == self._zone_attached)
            & (attached_to[rows, attacker_p] == safe_attacker[:, None]),
            axis=1,
        )
        & ~np.any(
            (zone_host[rows, defender_safe] == self._zone_attached)
            & (attached_to[rows, defender_safe] == safe_defender[:, None]),
            axis=1,
        )
    )
    leader_exists = np.any(zone_host == self._zone_leader, axis=(1, 2))

    return (
        base
        & (active_acts[:, 0] == self._act_noop)
        & (attacker_zone == self._zone_garden)
        & (defender_zone == self._zone_garden)
        & (attacker_def >= 0)
        & (defender_def == self._azk01_040_id)
        & (self._card_type[safe_attacker_def] == self._card_type_entity)
        & (self._card_type[safe_defender_def] == self._card_type_entity)
        & self._timing_when_attacked[safe_defender_def]
        & self._implemented[safe_defender_def]
        & np.asarray(states.tapped)[rows, attacker_p, safe_attacker]
        & no_attached
        & leader_exists
    )

  def _main_noop_simple_fast_mask(self, chosen, phase):
    base = self._main_noop_fast_mask(chosen, phase)
    if not np.any(base):
      return base

    states = self._states
    rows = np.arange(self.num_environments)
    active = np.asarray(states.active_player).astype(np.int32, copy=False)
    zone_host = np.asarray(states.zone)
    def_host = np.asarray(states.def_id)

    no_attached = ~np.any(zone_host == self._zone_attached, axis=(1, 2))
    no_sacrifice = ~np.any(np.asarray(states.sacrifice_eot), axis=(1, 2))

    token_zone = zone_host[:, :, self._token_instance]
    token_tapped = np.asarray(states.tapped)[:, :, self._token_instance]
    token_expires = np.asarray(states.ikz_token_expires_eot)
    no_token_cleanup = ~np.any(
        (token_zone == self._zone_token) & (token_tapped | token_expires),
        axis=1,
    )

    eot_fields = (
        states.atk_buff_eot,
        states.hp_buff_eot,
        states.cmb_in_eot,
        states.cmb_out_eot,
        states.carapace_eot,
    )
    no_eot_modifiers = ~np.any(
        np.stack([np.asarray(field) != 0 for field in eot_fields], axis=0),
        axis=(0, 2, 3),
    )

    no_positive_start_status = ~np.any(
        (np.asarray(states.frozen_dur) > 0)
        | (np.asarray(states.effect_immune_dur) > 0)
        | (np.asarray(states.shocked_dur) != 0),
        axis=(1, 2),
    )
    no_timed_grants = ~np.any(
        (np.asarray(states.timed_tag) != 0)
        & (np.asarray(states.timed_phase) != 0),
        axis=(1, 2, 3),
    )

    active_zone = zone_host[rows, active]
    active_defs = def_host[rows, active]
    active_valid = active_defs >= 0
    active_safe_defs = np.maximum(active_defs, 0)
    active_untap_zone = (
        (active_zone == self._zone_garden)
        | (active_zone == self._zone_alley)
        | (active_zone == self._zone_ikz_area)
        | (active_zone == self._zone_leader)
        | (active_zone == self._zone_gate)
    )
    no_force_tapped = ~np.any(
        active_untap_zone
        & active_valid
        & self._force_tapped[active_safe_defs],
        axis=1,
    )

    return (
        base
        & no_attached
        & no_sacrifice
        & no_token_cleanup
        & no_eot_modifiers
        & no_positive_start_status
        & no_timed_grants
        & no_force_tapped
    )

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
  t0 = time.perf_counter()
  env = JaxVecEnv(num_envs, pool, seed=seed)
  print(
      f"JaxVecEnv bench: envs={num_envs} steps={steps}"
      f" split_actions={int(env._split_actions)}",
      flush=True,
  )
  print(f"  construct: {time.perf_counter() - t0:.1f}s", flush=True)
  t0 = time.perf_counter()
  env.async_reset(seed)
  print(f"  async_reset compile+run: {time.perf_counter() - t0:.1f}s", flush=True)
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

  t0 = time.perf_counter()
  env.recv()
  print(f"  first recv host copy: {time.perf_counter() - t0:.1f}s", flush=True)
  t0 = time.perf_counter()
  env.send(pick_actions())
  print(f"  first send compile+run: {time.perf_counter() - t0:.1f}s", flush=True)
  t0 = time.perf_counter()
  env.recv()  # compile + warmup
  print(f"  second recv host copy: {time.perf_counter() - t0:.1f}s", flush=True)
  start = time.perf_counter()
  trace_steps = steps <= 20 or os.getenv("AZK_JAX_BENCH_TRACE", "0") != "0"
  for step_i in range(steps):
    actions = pick_actions()
    if trace_steps:
      active = np.asarray(env._pending[6]).astype(np.int32, copy=False)
      types = actions.reshape(env.num_environments, 2, 4)[
          np.arange(env.num_environments), np.clip(active, 0, 1), 0
      ]
      step_t0 = time.perf_counter()
      unique = ",".join(str(int(x)) for x in np.unique(types))
      print(
          f"  bench step {step_i}: action_types={unique} start",
          flush=True,
      )
    env.send(actions)
    env.recv()
    if trace_steps:
      print(
          f"  bench step {step_i}: action_types={unique}"
          f" elapsed={time.perf_counter() - step_t0:.1f}s",
          flush=True,
      )
  elapsed = time.perf_counter() - start
  sps = steps * env.num_agents / elapsed
  print(f"JaxVecEnv bench: {num_envs} envs, {sps:,.0f} agent-steps/s"
        f" ({elapsed / steps * 1e3:.1f} ms/step)")
  env.close()


if __name__ == "__main__":
  bench(*(int(x) for x in sys.argv[1:]))
