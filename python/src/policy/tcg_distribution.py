from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TCGActionDistribution:
    """Container for the Azuki action distribution before masking is applied."""

    primary_logits: torch.Tensor
    primary_action_mask: torch.Tensor
    legal_actions: torch.Tensor
    legal_action_count: torch.Tensor
    target_matrix: torch.Tensor
    unit1_projection: torch.Tensor
    unit2_projection: torch.Tensor
    bins2_logits: torch.Tensor
    bins3_logits: torch.Tensor
    gate1_table: torch.Tensor
    gate2_table: torch.Tensor

    def batch_size(self) -> int:
        return self.primary_logits.shape[0]

    def primary_actions(self) -> bool:
        return self.primary_action_mask.to(dtype=torch.bool).detach().cpu().sum(dim=-1).tolist()
