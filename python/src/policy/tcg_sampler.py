from __future__ import annotations

from typing import Callable, Tuple

import torch

from policy.tcg_distribution import TCGActionDistribution
from policy.tcg_policy import (
    ACTION_COMPONENT_COUNT,
    ACT_ATTACH_WEAPON_FROM_HAND,
    ACT_ATTACK,
    MAX_INDEX_SIZE,
)

MASK_MIN_VALUE = -1e9
_FALLBACK_SAMPLE_LOGITS: Callable | None = None


def set_fallback_sampler(func: Callable) -> None:
    """Store the default sampler so we can fall back for non-TCG policies."""
    global _FALLBACK_SAMPLE_LOGITS
    _FALLBACK_SAMPLE_LOGITS = func


def tcg_sample_logits(logits, action=None):
    """Custom sampler that masks Azuki actions after the policy forward pass."""
    if not isinstance(logits, TCGActionDistribution):
        if _FALLBACK_SAMPLE_LOGITS is None:
            raise RuntimeError("Fallback sampler is not configured")
        return _FALLBACK_SAMPLE_LOGITS(logits, action=action)

    distribution = logits
    device = distribution.primary_logits.device
    batch = distribution.primary_logits.shape[0]
    target_index_dim = max(distribution.target_matrix.size(1), MAX_INDEX_SIZE)

    original_action_shape = None
    provided_action = None
    if action is not None:
        original_action_shape = action.shape
        provided_action = action.to(device=device, dtype=torch.long).reshape(-1, ACTION_COMPONENT_COUNT)
        if provided_action.shape[0] != batch:
            raise ValueError(
                f"Provided action batch ({provided_action.shape[0]}) "
                f"does not match logits batch ({batch})"
            )

    gate1_table = distribution.gate1_table.to(device)
    gate2_table = distribution.gate2_table.to(device)

    primary_mask = _ensure_valid_mask(distribution.primary_action_mask.to(device))
    primary_choice, primary_logprob, primary_entropy = _sample_stage(
        distribution.primary_logits,
        primary_mask,
        provided_action[:, 0] if provided_action is not None else None,
    )

    sub1_mask = _build_subaction_mask(
        distribution,
        primary_choice,
        None,
        None,
        column=1,
        mask_size=target_index_dim,
    )
    gate1_weights = gate1_table.index_select(0, primary_choice)
    unit1_logits = _compute_unit_logits(
        distribution.target_matrix,
        gate1_weights,
        distribution.unit1_projection,
        index_dim=target_index_dim,
    )
    sub1_choice, sub1_logprob, sub1_entropy = _sample_stage(
        unit1_logits,
        sub1_mask,
        provided_action[:, 1] if provided_action is not None else None,
    )

    sub2_mask = _build_subaction_mask(
        distribution,
        primary_choice,
        sub1_choice,
        None,
        column=2,
        mask_size=target_index_dim,
    )
    gate2_weights = gate2_table.index_select(0, primary_choice)
    unit2_logits = _compute_unit_logits(
        distribution.target_matrix,
        gate2_weights,
        distribution.unit2_projection,
        index_dim=target_index_dim,
    )
    bins2_logits = _pad_or_trim_to_index_dim(distribution.bins2_logits, target_index_dim)
    requires_unit = (primary_choice == ACT_ATTACK) | (primary_choice == ACT_ATTACH_WEAPON_FROM_HAND)
    requires_unit = requires_unit.unsqueeze(-1)
    sub2_logits = torch.where(requires_unit, unit2_logits, bins2_logits)
    sub2_choice, sub2_logprob, sub2_entropy = _sample_stage(
        sub2_logits,
        sub2_mask,
        provided_action[:, 2] if provided_action is not None else None,
    )

    sub3_mask = _build_subaction_mask(
        distribution,
        primary_choice,
        sub1_choice,
        sub2_choice,
        column=3,
        mask_size=target_index_dim,
    )
    bins3_logits = _pad_or_trim_to_index_dim(distribution.bins3_logits, target_index_dim)
    sub3_choice, sub3_logprob, sub3_entropy = _sample_stage(
        bins3_logits,
        sub3_mask,
        provided_action[:, 3] if provided_action is not None else None,
    )

    chosen_actions = torch.stack(
        (primary_choice, sub1_choice, sub2_choice, sub3_choice),
        dim=-1,
    ).to(dtype=torch.long)
    total_logprob = primary_logprob + sub1_logprob + sub2_logprob + sub3_logprob
    total_entropy = primary_entropy + sub1_entropy + sub2_entropy + sub3_entropy

    if provided_action is not None:
        actions_out = provided_action
        if original_action_shape is not None:
            actions_out = actions_out.reshape(original_action_shape)
    else:
        actions_out = chosen_actions

    return actions_out, total_logprob, total_entropy


def _sample_stage(
    logits: torch.Tensor,
    mask: torch.Tensor,
    provided: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_logits = logits.masked_fill(~mask, MASK_MIN_VALUE)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    if provided is None:
        sampled = torch.multinomial(torch.nan_to_num(probs, nan=0.0), 1).squeeze(-1)
    else:
        sampled = provided.long()
    gathered = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, gathered, entropy


def _ensure_valid_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() != 2:
        raise ValueError(f"Mask should be 2D, got shape {mask.shape}")
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    valid = mask.any(dim=-1, keepdim=True)
    if valid.all():
        return mask
    mask = mask.clone()
    mask[~valid.expand_as(mask)] = False
    mask[~valid.squeeze(-1), 0] = True
    return mask


def _build_subaction_mask(
    distribution: TCGActionDistribution,
    primary: torch.Tensor,
    sub1: torch.Tensor | None,
    sub2: torch.Tensor | None,
    *,
    column: int,
    mask_size: int = MAX_INDEX_SIZE,
) -> torch.Tensor:
    legal_actions = distribution.legal_actions
    counts = distribution.legal_action_count
    device = legal_actions.device
    B, K = legal_actions.shape[:2]
    row_indices = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
    valid = row_indices < counts.view(-1, 1)
    valid &= legal_actions[..., 0] == primary.view(-1, 1)
    if sub1 is not None:
        valid &= legal_actions[..., 1] == sub1.view(-1, 1)
    if sub2 is not None:
        valid &= legal_actions[..., 2] == sub2.view(-1, 1)
    batch_indexes = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
    mask = torch.zeros(B, mask_size, device=device, dtype=torch.bool)
    if valid.any():
        target_indexes = legal_actions[..., column].clamp(0, mask_size - 1)
        mask[batch_indexes[valid], target_indexes[valid]] = True
    return _ensure_valid_mask(mask)


def _compute_unit_logits(
    target_matrix: torch.Tensor,
    gate_weights: torch.Tensor,
    projection: torch.Tensor,
    index_dim: int | None = None,
) -> torch.Tensor:
    weighted = target_matrix * gate_weights.unsqueeze(1)
    logits = torch.sum(weighted * projection.unsqueeze(1), dim=-1)
    if index_dim is None:
        index_dim = MAX_INDEX_SIZE
    return _pad_or_trim_to_index_dim(logits, index_dim)


def _pad_or_trim_to_index_dim(tensor: torch.Tensor, index_dim: int = MAX_INDEX_SIZE) -> torch.Tensor:
    current = tensor.size(-1)
    if current == index_dim:
        return tensor
    if current > index_dim:
        return tensor[..., :index_dim]
    pad_size = index_dim - current
    pad_shape = (*tensor.shape[:-1], pad_size)
    pad_value = torch.finfo(tensor.dtype).min
    pad_tensor = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad_tensor], dim=-1)
