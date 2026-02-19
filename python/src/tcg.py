import ctypes
import os
import numpy as np
from pettingzoo import AECEnv
from action import ACTION_COMPONENT_COUNT, build_action_space
from obs_debug import ObservationDebugTracker
from observation import (
    MAX_PLAYERS_PER_MATCH,
    OBSERVATION_CTYPE,
    OBSERVATION_STRUCT_SIZE,
    build_observation_space,
    observation_to_dict,
)
import binding

def _env_int(name: str, default: int) -> int:
  raw = os.getenv(name)
  if raw is None:
    return default
  try:
    return int(raw)
  except ValueError:
    return default

class AzukiTCG(AECEnv):
  """
  Azuki TCG environment using PettingZoo's AEC (Agent Environment Cycle) API.
  This is the natural fit for turn-based games where agents act sequentially.
  """
  metadata = {
    'render_modes': ['human', 'ansi'],
    'name': 'azuki_tcg_v0',
  }
  def __init__(
    self,
    seed: int | None = None,
  ) -> None:
    super().__init__()
    self.render_mode = "ansi"
    self.np_random = np.random.default_rng(seed)
    self.possible_agents = list(range(MAX_PLAYERS_PER_MATCH))
    self._agent_count = len(self.possible_agents)
    self._action_space = build_action_space()

    # C binding arrays
    self._observations = np.zeros(
      OBSERVATION_STRUCT_SIZE * self._agent_count,
      dtype=np.uint8,
    )
    self._observation_struct = ctypes.cast(
      self._observations.ctypes.data, ctypes.POINTER(OBSERVATION_CTYPE)
    )
    self._actions = np.zeros(
      (self._agent_count, ACTION_COMPONENT_COUNT),
      dtype=np.int32,
    )
    self._rewards = np.zeros(self._agent_count, dtype=np.float32)
    self._terminals = np.zeros(self._agent_count, dtype=np.bool_)
    self._truncations = np.zeros(self._agent_count, dtype=np.bool_)

    # AEC state
    self.agents = []
    self.agent_selection = None
    self._active_player_index = None
    self.rewards = {}
    self._cumulative_rewards = {}
    self.terminations = {}
    self.truncations = {}
    self.infos = {}
    self._obs_debug = ObservationDebugTracker(
      enabled=os.getenv("AZK_OBS_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"},
      sample_every=_env_int("AZK_OBS_DEBUG_SAMPLE_EVERY", 1),
      possible_agents=self.possible_agents,
    )

    self.c_envs = binding.env_init(
      self._observations,
      self._actions,
      self._rewards,
      self._terminals,
      self._truncations,
      int(seed or 0),
    )

  def observation_space(self, agent: str):
    return build_observation_space()

  def action_space(self, agent):
    return self._action_space

  def _player_index(self, agent) -> int:
    if isinstance(agent, str):
      if agent.startswith("player_"):
        agent = agent.split("_")[-1]
      agent = int(agent)
    return int(agent)

  def _raw_observation(self, agent_index: int):
    return self._observation_struct[agent_index]

  def _collect_observations(self):
    return {
      agent: observation_to_dict(self._raw_observation(idx))
      for idx, agent in enumerate(self.possible_agents)
    }

  def _sync_done_flags(self):
    for idx, agent in enumerate(self.possible_agents):
      self.terminations[agent] = bool(self._terminals[idx])
      self.truncations[agent] = bool(self._truncations[idx])

  def _sync_agent_selection(self):
    active_idx = binding.env_active_player(self.c_envs)
    if active_idx < 0 or active_idx >= self._agent_count:
      raise ValueError(f"Active player index {active_idx} is out of range")

    active_agent = self.possible_agents[active_idx]
    if active_agent not in self.agents:
      raise ValueError(f"Active agent {active_agent} not in agents {self.agents}")

    self._active_player_index = active_idx
    self.agent_selection = active_agent

  def _update_terminal_infos(self):
    """Fetch C-side episode stats and fan them out per-agent."""
    log = binding.env_log(self.c_envs)
    if not log:
      return

    per_agent_stats = (
      {
        "win": float(log.get("p0_winrate", 0.0)),
        "azk_started_first_rate": float(log.get("p0_start_rate", 0.0)),
        "leader_health": float(log.get("p0_avg_leader_health", 0.0)),
        "azk_episode_return": float(log.get("p0_episode_return", 0.0)),
        "azk_episode_length": float(log.get("episode_length", 0.0)),
        "azk_timeout_truncation": float(log.get("timeout_truncation_rate", 0.0)),
        "azk_auto_tick_truncation": float(log.get("auto_tick_truncation_rate", 0.0)),
        "azk_gameover_terminal": float(log.get("gameover_terminal_rate", 0.0)),
        "azk_winner_terminal": float(log.get("winner_terminal_rate", 0.0)),
        "azk_curriculum_episode_cap": float(log.get("curriculum_episode_cap", 0.0)),
        "azk_reward_shaping_scale": float(log.get("reward_shaping_scale", 1.0)),
        "azk_completed_episodes": float(log.get("completed_episodes", 0.0)),
        "azk_noop_selected_rate": float(log.get("p0_noop_selected_rate", 0.0)),
        "azk_attack_selected_rate": float(log.get("p0_attack_selected_rate", 0.0)),
        "azk_attach_weapon_from_hand_selected_rate": float(log.get("p0_attach_weapon_from_hand_selected_rate", 0.0)),
        "azk_play_spell_from_hand_selected_rate": float(log.get("p0_play_spell_from_hand_selected_rate", 0.0)),
        "azk_activate_garden_or_leader_ability_selected_rate": float(log.get("p0_activate_garden_or_leader_ability_selected_rate", 0.0)),
        "azk_activate_alley_ability_selected_rate": float(log.get("p0_activate_alley_ability_selected_rate", 0.0)),
        "azk_gate_portal_selected_rate": float(log.get("p0_gate_portal_selected_rate", 0.0)),
        "azk_play_entity_to_alley_selected_rate": float(log.get("p0_play_entity_to_alley_selected_rate", 0.0)),
        "azk_play_entity_to_garden_selected_rate": float(log.get("p0_play_entity_to_garden_selected_rate", 0.0)),
        "azk_play_selected_rate": float(log.get("p0_play_selected_rate", 0.0)),
        "azk_ability_selected_rate": float(log.get("p0_ability_selected_rate", 0.0)),
        "azk_target_selected_rate": float(log.get("p0_target_selected_rate", 0.0)),
      },
      {
        "win": float(log.get("p1_winrate", 0.0)),
        "azk_started_first_rate": float(log.get("p1_start_rate", 0.0)),
        "leader_health": float(log.get("p1_avg_leader_health", 0.0)),
        "azk_episode_return": float(log.get("p1_episode_return", 0.0)),
        "azk_episode_length": float(log.get("episode_length", 0.0)),
        "azk_timeout_truncation": float(log.get("timeout_truncation_rate", 0.0)),
        "azk_auto_tick_truncation": float(log.get("auto_tick_truncation_rate", 0.0)),
        "azk_gameover_terminal": float(log.get("gameover_terminal_rate", 0.0)),
        "azk_winner_terminal": float(log.get("winner_terminal_rate", 0.0)),
        "azk_curriculum_episode_cap": float(log.get("curriculum_episode_cap", 0.0)),
        "azk_reward_shaping_scale": float(log.get("reward_shaping_scale", 1.0)),
        "azk_completed_episodes": float(log.get("completed_episodes", 0.0)),
        "azk_noop_selected_rate": float(log.get("p1_noop_selected_rate", 0.0)),
        "azk_attack_selected_rate": float(log.get("p1_attack_selected_rate", 0.0)),
        "azk_attach_weapon_from_hand_selected_rate": float(log.get("p1_attach_weapon_from_hand_selected_rate", 0.0)),
        "azk_play_spell_from_hand_selected_rate": float(log.get("p1_play_spell_from_hand_selected_rate", 0.0)),
        "azk_activate_garden_or_leader_ability_selected_rate": float(log.get("p1_activate_garden_or_leader_ability_selected_rate", 0.0)),
        "azk_activate_alley_ability_selected_rate": float(log.get("p1_activate_alley_ability_selected_rate", 0.0)),
        "azk_gate_portal_selected_rate": float(log.get("p1_gate_portal_selected_rate", 0.0)),
        "azk_play_entity_to_alley_selected_rate": float(log.get("p1_play_entity_to_alley_selected_rate", 0.0)),
        "azk_play_entity_to_garden_selected_rate": float(log.get("p1_play_entity_to_garden_selected_rate", 0.0)),
        "azk_play_selected_rate": float(log.get("p1_play_selected_rate", 0.0)),
        "azk_ability_selected_rate": float(log.get("p1_ability_selected_rate", 0.0)),
        "azk_target_selected_rate": float(log.get("p1_target_selected_rate", 0.0)),
      },
    )

    for agent, stats in zip(self.possible_agents, per_agent_stats):
      self.infos[agent].update(stats)
    for agent, debug_stats in self._obs_debug.episode_metrics().items():
      if agent in self.infos:
        self.infos[agent].update(debug_stats)

  def observe(self, agent):
    idx = self._player_index(agent)
    return observation_to_dict(self._raw_observation(idx))

  def reset(self, seed=None, options=None):
    if seed is None:
      # Use the local RNG to produce a fresh episode seed; deterministic if ctor seed was fixed.
      seed = int(self.np_random.integers(0, 2**31 - 1))
    else:
      seed = int(seed)

    binding.env_reset(self.c_envs, seed)
    self._actions.fill(0)

    self.agents = self.possible_agents[:]
    self.rewards = {agent: 0.0 for agent in self.possible_agents}
    self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
    self.terminations = {agent: False for agent in self.possible_agents}
    self.truncations = {agent: False for agent in self.possible_agents}
    self.infos = {agent: {} for agent in self.possible_agents}
    self._active_player_index = None
    self._obs_debug.reset()

    self._sync_done_flags()
    self._sync_agent_selection()
    observations = self._collect_observations()
    self._obs_debug.observe_step(observations)
    observation = observations[self.agent_selection]
    return observation, self.infos[self.agent_selection]

  def step(self, action):
    if not self.agents:
      raise RuntimeError("step() called on finished environment")

    if action is None:
      raise ValueError("AzukiTCG received None action while the match is still running")

    acting_agent = self.agent_selection
    if acting_agent is None:
      raise RuntimeError("Environment does not have an active agent to step")
    agent_idx = self._player_index(acting_agent)
    if (
      self._active_player_index is not None
      and agent_idx != self._active_player_index
    ):
      raise RuntimeError(
        f"Attempted to act for agent {acting_agent}, but active player is {self._active_player_index}"
      )

    encoded_action = np.asarray(action, dtype=np.int32)
    if encoded_action.shape != (ACTION_COMPONENT_COUNT,):
      raise ValueError(
        f"AzukiTCG expects {ACTION_COMPONENT_COUNT} integers (type, subaction_1..3); got shape {encoded_action.shape}"
      )

    self._clear_rewards()
    self._actions[agent_idx] = encoded_action
    binding.env_step(self.c_envs)

    for idx, agent in enumerate(self.possible_agents):
      self.rewards[agent] = float(self._rewards[idx])
    reward = self.rewards[acting_agent]
    info = {}

    self.infos[acting_agent] = info
    self._accumulate_rewards()
    self._sync_done_flags()

    all_done = all(
      self.terminations[agent] or self.truncations[agent]
      for agent in self.possible_agents
    )
    if all_done:
      self._update_terminal_infos()
      self.agents = []
      self.agent_selection = None
      self._active_player_index = None
    else:
      self._sync_agent_selection()

    observations = self._collect_observations()
    self._obs_debug.observe_step(observations)
    observation = observations[acting_agent]
    termination = self.terminations[acting_agent]
    truncation = self.truncations[acting_agent]
    info = self.infos[acting_agent]

    return observation, reward, termination, truncation, info

  def render(self, mode="human"):
    if mode not in self.metadata.get("render_modes", []):
      raise ValueError(f"Unsupported render mode {mode}; expected one of {self.metadata.get('render_modes', [])}")

    frame = binding.env_render(self.c_envs)
    if frame is None:
      return None
    if mode == "ansi":
      return frame
    print(frame)
    return None
