from __future__ import annotations

from tcg import AzukiTCG as AzukiTCGV1
from observation import build_observation_space


class AzukiTCG(AzukiTCGV1):
  """V2 AEC wrapper using the engine-side observation schema directly."""

  def observation_space(self, agent: str):
    return build_observation_space()
