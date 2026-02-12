from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from action import ActionType


@dataclass
class _AgentObsStats:
  samples: int = 0
  legal_sum: float = 0.0
  legal_min: int = 1 << 30
  legal_max: int = -1
  legal_zero_count: int = 0
  primary_nonzero_sum: float = 0.0
  noop_only_count: int = 0
  legal_primary_mismatch_count: int = 0
  hand_sum: float = 0.0
  hand_sumsq: float = 0.0
  opp_hand_sum: float = 0.0
  opp_hand_sumsq: float = 0.0
  unique_signatures: set[tuple[int, ...]] = field(default_factory=set)


class ObservationDebugTracker:
  def __init__(self, *, enabled: bool, sample_every: int, possible_agents: list[int]) -> None:
    self.enabled = bool(enabled)
    self.sample_every = max(int(sample_every), 1)
    self.possible_agents = list(possible_agents)
    self._step_count = 0
    self._stats: dict[int, _AgentObsStats] = {}
    self.reset()

  def reset(self) -> None:
    self._step_count = 0
    self._stats = {agent: _AgentObsStats() for agent in self.possible_agents}

  def _should_sample(self) -> bool:
    return (self._step_count % self.sample_every) == 0

  @staticmethod
  def _safe_int(data: dict[str, Any], key: str) -> int:
    return int(data.get(key, 0)) if data is not None else 0

  def observe_step(self, observations: dict[int, dict[str, Any]]) -> None:
    if not self.enabled:
      return

    self._step_count += 1
    if not self._should_sample():
      return

    for agent, obs in observations.items():
      agent_stats = self._stats.get(agent)
      if agent_stats is None:
        continue

      action_mask = obs.get("action_mask", {})
      primary_mask = np.asarray(action_mask.get("primary_action_mask", []), dtype=np.bool_).reshape(-1)
      legal_count = int(action_mask.get("legal_action_count", 0))
      primary_nonzero = int(np.count_nonzero(primary_mask))
      noop_enabled = bool(
        primary_mask.size > int(ActionType.NOOP) and primary_mask[int(ActionType.NOOP)]
      )
      noop_only = primary_nonzero == 1 and noop_enabled
      mismatch = legal_count > 0 and primary_nonzero == 0

      player = obs.get("player", {})
      opponent = obs.get("opponent", {})
      phase = int(obs.get("phase", 0))
      ability_context = obs.get("ability_context", {})
      ability_phase = self._safe_int(ability_context, "phase")
      hand_count = self._safe_int(player, "hand_count")
      opp_hand_count = self._safe_int(opponent, "hand_count")
      player_hp = self._safe_int(player.get("leader", {}), "cur_hp")
      opp_hp = self._safe_int(opponent.get("leader", {}), "cur_hp")

      signature = (
        phase,
        ability_phase,
        hand_count,
        opp_hand_count,
        player_hp,
        opp_hp,
        legal_count,
        primary_nonzero,
      )
      agent_stats.unique_signatures.add(signature)
      if len(agent_stats.unique_signatures) > 2048:
        # Cap memory growth while keeping enough variety signal.
        agent_stats.unique_signatures = set(tuple(item) for item in list(agent_stats.unique_signatures)[:2048])

      agent_stats.samples += 1
      agent_stats.legal_sum += legal_count
      agent_stats.legal_min = min(agent_stats.legal_min, legal_count)
      agent_stats.legal_max = max(agent_stats.legal_max, legal_count)
      agent_stats.legal_zero_count += int(legal_count == 0)
      agent_stats.primary_nonzero_sum += primary_nonzero
      agent_stats.noop_only_count += int(noop_only)
      agent_stats.legal_primary_mismatch_count += int(mismatch)
      agent_stats.hand_sum += hand_count
      agent_stats.hand_sumsq += hand_count * hand_count
      agent_stats.opp_hand_sum += opp_hand_count
      agent_stats.opp_hand_sumsq += opp_hand_count * opp_hand_count

  def episode_metrics(self) -> dict[int, dict[str, float]]:
    if not self.enabled:
      return {}

    result: dict[int, dict[str, float]] = {}
    for agent, stats in self._stats.items():
      if stats.samples <= 0:
        continue

      samples = float(stats.samples)
      hand_mean = stats.hand_sum / samples
      hand_var = max((stats.hand_sumsq / samples) - (hand_mean * hand_mean), 0.0)
      opp_hand_mean = stats.opp_hand_sum / samples
      opp_hand_var = max((stats.opp_hand_sumsq / samples) - (opp_hand_mean * opp_hand_mean), 0.0)
      result[agent] = {
        "azk_dbg_obs_samples": samples,
        "azk_dbg_legal_action_count_mean": stats.legal_sum / samples,
        "azk_dbg_legal_action_count_min": float(stats.legal_min),
        "azk_dbg_legal_action_count_max": float(stats.legal_max),
        "azk_dbg_legal_zero_rate": stats.legal_zero_count / samples,
        "azk_dbg_primary_mask_nonzero_mean": stats.primary_nonzero_sum / samples,
        "azk_dbg_noop_only_rate": stats.noop_only_count / samples,
        "azk_dbg_legal_primary_mismatch_rate": stats.legal_primary_mismatch_count / samples,
        "azk_dbg_obs_signature_unique_ratio": min(len(stats.unique_signatures) / samples, 1.0),
        "azk_dbg_player_hand_count_std": float(np.sqrt(hand_var)),
        "azk_dbg_opponent_hand_count_std": float(np.sqrt(opp_hand_var)),
      }

    return result
