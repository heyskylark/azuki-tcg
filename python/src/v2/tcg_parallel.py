from __future__ import annotations

from tcg_parallel import AzukiTCGParallel as AzukiTCGParallelV1
from observation import build_observation_space


class AzukiTCGParallel(AzukiTCGParallelV1):
  """V2 Parallel wrapper using the engine-side observation schema directly."""

  def observation_space(self, agent):
    return build_observation_space()
