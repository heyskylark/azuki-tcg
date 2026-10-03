from __future__ import annotations

import os
from typing import Callable, Tuple

import torch

from policy.tcg_distribution import TCGActionDistribution, TCGLegalActionDistribution
from policy.v2.tcg_policy import (
    ACTION_COMPONENT_COUNT,
    ACT_ATTACH_WEAPON_FROM_HAND,
    ACT_ATTACK,
    MAX_INDEX_SIZE,
)

MASK_MIN_VALUE = -1e9
LOG_EPS = 1e-8
DEFAULT_PRIMARY_TEMPERATURE = 1.0
DEFAULT_SUBACTION_TEMPERATURE = 1.2
DEFAULT_SMOOTHING_EPS = 0.05
DEFAULT_LEGAL_ROW_TEMPERATURE = 1.0
DEFAULT_DECK_PICK_SMOOTHING_EPS = 0.0
DECK_PICK_PRIMARY = 3  # ActionType.DECK_PICK_CARD
_RUNTIME_PRIMARY_TEMPERATURE = DEFAULT_PRIMARY_TEMPERATURE
_RUNTIME_SUBACTION_TEMPERATURE = DEFAULT_SUBACTION_TEMPERATURE
_RUNTIME_SMOOTHING_EPS = DEFAULT_SMOOTHING_EPS
_RUNTIME_LEGAL_ROW_TEMPERATURE = DEFAULT_LEGAL_ROW_TEMPERATURE
_RUNTIME_DECK_PICK_SMOOTHING_EPS = DEFAULT_DECK_PICK_SMOOTHING_EPS
_FALLBACK_SAMPLE_LOGITS: Callable | None = None

def set_fallback_sampler(func: Callable) -> None:
    """Store the default sampler so we can fall back for non-TCG policies."""
    global _FALLBACK_SAMPLE_LOGITS
    _FALLBACK_SAMPLE_LOGITS = func


def set_sampling_params(
    *,
    primary_temperature: float | None = None,
    subaction_temperature: float | None = None,
    smoothing_eps: float | None = None,
    legal_row_temperature: float | None = None,
    deck_pick_smoothing_eps: float | None = None,
) -> None:
    """Update runtime sampling params used by the custom Azuki sampler."""
    global _RUNTIME_PRIMARY_TEMPERATURE
    global _RUNTIME_SUBACTION_TEMPERATURE
    global _RUNTIME_SMOOTHING_EPS
    global _RUNTIME_LEGAL_ROW_TEMPERATURE
    global _RUNTIME_DECK_PICK_SMOOTHING_EPS

    if primary_temperature is not None:
        _RUNTIME_PRIMARY_TEMPERATURE = max(float(primary_temperature), 1e-6)
    if subaction_temperature is not None:
        _RUNTIME_SUBACTION_TEMPERATURE = max(float(subaction_temperature), 1e-6)
    if smoothing_eps is not None:
        _RUNTIME_SMOOTHING_EPS = min(max(float(smoothing_eps), 0.0), 1.0)
    if legal_row_temperature is not None:
        _RUNTIME_LEGAL_ROW_TEMPERATURE = max(float(legal_row_temperature), 1e-6)
    if deck_pick_smoothing_eps is not None:
        _RUNTIME_DECK_PICK_SMOOTHING_EPS = min(max(float(deck_pick_smoothing_eps), 0.0), 1.0)


def get_sampling_params() -> dict[str, float]:
    return {
        "primary_temperature": float(_RUNTIME_PRIMARY_TEMPERATURE),
        "subaction_temperature": float(_RUNTIME_SUBACTION_TEMPERATURE),
        "smoothing_eps": float(_RUNTIME_SMOOTHING_EPS),
        "legal_row_temperature": float(_RUNTIME_LEGAL_ROW_TEMPERATURE),
        "deck_pick_smoothing_eps": float(_RUNTIME_DECK_PICK_SMOOTHING_EPS),
    }


def tcg_sample_logits(logits, action=None):
    """Custom sampler that masks Azuki actions after the policy forward pass."""
    if isinstance(logits, TCGLegalActionDistribution):
        if action is None:
            # CUDA-graphed rollout: sampling was captured into the graph and
            # these tensors are its static outputs, refreshed per replay. The
            # params snapshot keeps anneal configs correct (they fall through
            # to eager sampling on the same static logits).
            presampled = getattr(logits, "_azk_presampled", None)
            if presampled is not None and (
                getattr(logits, "_azk_presampled_params", None) == get_sampling_params()
            ):
                return presampled
        return _sample_legal_action_rows(logits, action=action)

    if not isinstance(logits, TCGActionDistribution):
        raise ValueError("logits is not a TCGActionDistribution")

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

    # legal_counts = distribution.legal_action_count.detach().cpu().tolist()
    # primary_true = primary_mask.detach().cpu().sum(dim=-1).tolist()
    # print(
    #     "[tcg_sample_debug]",
    #     f"batch={batch}",
    #     f"legal_action_count={legal_counts[:4]}",
    #     f"primary_true_counts={primary_true[:4]}",
    # )

    primary_choice, primary_logprob, primary_entropy = _sample_stage(
        distribution.primary_logits,
        primary_mask,
        provided_action[:, 0] if provided_action is not None else None,
        temperature=_RUNTIME_PRIMARY_TEMPERATURE,
        smoothing_eps=0.0,
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
        temperature=_RUNTIME_SUBACTION_TEMPERATURE,
        smoothing_eps=_RUNTIME_SMOOTHING_EPS,
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
        temperature=_RUNTIME_SUBACTION_TEMPERATURE,
        smoothing_eps=_RUNTIME_SMOOTHING_EPS,
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
        temperature=_RUNTIME_SUBACTION_TEMPERATURE,
        smoothing_eps=_RUNTIME_SMOOTHING_EPS,
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


def tcg_argmax_logits(logits) -> torch.Tensor:
    """Select the highest-scoring legal action with stable first-index ties.

    Promotion and release evaluation must be independent of the mutable
    training sampler temperatures and smoothing values. `torch.argmax` returns
    the first maximum, which gives an explicit deterministic tie break.
    """
    if isinstance(logits, TCGLegalActionDistribution):
        device = logits.legal_action_logits.device
        batch, candidate_count = logits.legal_action_logits.shape
        row_indices = torch.arange(candidate_count, device=device).unsqueeze(0).expand(batch, -1)
        row_mask = row_indices < logits.legal_action_count.to(
            device=device, dtype=torch.long
        ).view(-1, 1)
        row_mask = _ensure_valid_mask(row_mask)
        masked = logits.legal_action_logits.masked_fill(~row_mask, MASK_MIN_VALUE)
        choices = torch.argmax(masked, dim=-1)
        return logits.legal_actions.to(device=device, dtype=torch.long)[
            torch.arange(batch, device=device), choices
        ]

    if not isinstance(logits, TCGActionDistribution):
        raise ValueError("logits is not a TCGActionDistribution")

    distribution = logits
    device = distribution.primary_logits.device
    target_index_dim = max(distribution.target_matrix.size(1), MAX_INDEX_SIZE)
    primary = _argmax_stage(
        distribution.primary_logits,
        _ensure_valid_mask(distribution.primary_action_mask.to(device)),
    )

    sub1_mask = _build_subaction_mask(
        distribution, primary, None, None, column=1, mask_size=target_index_dim
    )
    sub1 = _argmax_stage(
        _compute_unit_logits(
            distribution.target_matrix,
            distribution.gate1_table.to(device).index_select(0, primary),
            distribution.unit1_projection,
            index_dim=target_index_dim,
        ),
        sub1_mask,
    )

    sub2_mask = _build_subaction_mask(
        distribution, primary, sub1, None, column=2, mask_size=target_index_dim
    )
    unit2_logits = _compute_unit_logits(
        distribution.target_matrix,
        distribution.gate2_table.to(device).index_select(0, primary),
        distribution.unit2_projection,
        index_dim=target_index_dim,
    )
    bins2_logits = _pad_or_trim_to_index_dim(distribution.bins2_logits, target_index_dim)
    needs_unit = ((primary == ACT_ATTACK) | (primary == ACT_ATTACH_WEAPON_FROM_HAND)).unsqueeze(-1)
    sub2 = _argmax_stage(
        torch.where(needs_unit, unit2_logits, bins2_logits), sub2_mask
    )

    sub3_mask = _build_subaction_mask(
        distribution, primary, sub1, sub2, column=3, mask_size=target_index_dim
    )
    sub3 = _argmax_stage(
        _pad_or_trim_to_index_dim(distribution.bins3_logits, target_index_dim),
        sub3_mask,
    )
    return torch.stack((primary, sub1, sub2, sub3), dim=-1).to(dtype=torch.long)


def _argmax_stage(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_mask = _ensure_valid_mask(mask)
    return torch.argmax(logits.masked_fill(~valid_mask, MASK_MIN_VALUE), dim=-1)


def legal_action_row_distribution(
    distribution: TCGLegalActionDistribution,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized legal-row probabilities, log probabilities, and mask."""
    device = distribution.legal_action_logits.device
    batch, candidate_count = distribution.legal_action_logits.shape
    row_indices = torch.arange(candidate_count, device=device).unsqueeze(0).expand(batch, -1)
    row_mask = row_indices < distribution.legal_action_count.to(
        device=device, dtype=torch.long
    ).view(-1, 1)
    row_mask = _ensure_valid_mask(row_mask)
    masked_logits = distribution.legal_action_logits.float().masked_fill(
        ~row_mask, MASK_MIN_VALUE
    )
    if _RUNTIME_LEGAL_ROW_TEMPERATURE != 1.0:
        masked_logits = masked_logits / _RUNTIME_LEGAL_ROW_TEMPERATURE
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    if _RUNTIME_DECK_PICK_SMOOTHING_EPS > 0.0:
        legal_count = row_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        uniform_log_probs = torch.where(
            row_mask,
            -legal_count.to(dtype=log_probs.dtype).log(),
            torch.full_like(log_probs, -torch.inf),
        )
        is_pick_row = (
            distribution.legal_actions[:, 0, 0].to(device=device, dtype=torch.long)
            == DECK_PICK_PRIMARY
        ).unsqueeze(-1)
        smoothing_eps = torch.as_tensor(
            _RUNTIME_DECK_PICK_SMOOTHING_EPS,
            device=device,
            dtype=log_probs.dtype,
        )
        smoothed_log_probs = torch.logaddexp(
            log_probs + torch.log1p(-smoothing_eps),
            uniform_log_probs + smoothing_eps.log(),
        )
        log_probs = torch.where(is_pick_row, smoothed_log_probs, log_probs)
    probs = log_probs.exp()
    return probs, log_probs, row_mask


def legal_action_kl(
    old_distribution: TCGLegalActionDistribution,
    new_distribution: TCGLegalActionDistribution,
) -> torch.Tensor:
    """Compute exact KL(old || new) over unique environment action tuples."""
    if old_distribution.legal_action_logits.shape != new_distribution.legal_action_logits.shape:
        raise ValueError("Legal-action KL requires matching distribution shapes")
    old_probs, old_log_probs, old_mask = legal_action_row_distribution(old_distribution)
    _, new_log_probs, new_mask = legal_action_row_distribution(new_distribution)
    if old_mask.shape != new_mask.shape:
        raise ValueError("Legal-action KL requires matching legal-row masks")

    old_actions = old_distribution.legal_actions.to(
        device=old_probs.device,
        dtype=torch.long,
    )
    new_actions = new_distribution.legal_actions.to(
        device=old_probs.device,
        dtype=torch.long,
    )
    candidate_count = old_actions.shape[1]
    row_ids = torch.arange(candidate_count, device=old_probs.device)
    earlier_rows = row_ids.view(1, -1) < row_ids.view(-1, 1)
    result = []
    for start in range(0, old_actions.shape[0], 256):
        stop = min(start + 256, old_actions.shape[0])
        chunk_old_actions = old_actions[start:stop]
        chunk_new_actions = new_actions[start:stop]
        chunk_old_mask = old_mask[start:stop]
        chunk_new_mask = new_mask[start:stop]
        old_equivalent = (
            chunk_old_mask.unsqueeze(2)
            & chunk_old_mask.unsqueeze(1)
            & (
                chunk_old_actions.unsqueeze(2)
                == chunk_old_actions.unsqueeze(1)
            ).all(dim=-1)
        )
        old_to_new = (
            chunk_old_mask.unsqueeze(2)
            & chunk_new_mask.unsqueeze(1)
            & (
                chunk_old_actions.unsqueeze(2)
                == chunk_new_actions.unsqueeze(1)
            ).all(dim=-1)
        )
        old_action_log_probs = torch.logsumexp(
            old_log_probs[start:stop].unsqueeze(1).masked_fill(
                ~old_equivalent,
                -torch.inf,
            ),
            dim=-1,
        )
        new_action_log_probs = torch.logsumexp(
            new_log_probs[start:stop].unsqueeze(1).masked_fill(
                ~old_to_new,
                -torch.inf,
            ),
            dim=-1,
        )
        representative = chunk_old_mask & ~(
            old_equivalent & earlier_rows.unsqueeze(0)
        ).any(dim=-1)
        terms = torch.where(
            representative,
            old_action_log_probs.exp()
            * (old_action_log_probs - new_action_log_probs),
            torch.zeros_like(old_action_log_probs),
        )
        result.append(terms.sum(dim=-1))
    return torch.cat(result)


def _legal_action_row_probabilities(
    distribution: TCGLegalActionDistribution,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs, _, row_mask = legal_action_row_distribution(distribution)
    return probs, row_mask


def _sample_legal_action_rows(
    distribution: TCGLegalActionDistribution,
    *,
    action=None,
):
    device = distribution.legal_action_logits.device
    batch = distribution.legal_action_logits.shape[0]
    probs, log_probs, row_mask = legal_action_row_distribution(distribution)
    entropy = -torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs)).sum(dim=-1)
    legal_actions = distribution.legal_actions.to(device=device, dtype=torch.long)

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
        row_choice = _match_provided_legal_rows(
            legal_actions,
            distribution.legal_action_count.to(device=device, dtype=torch.long),
            provided_action,
        )
    else:
        row_choice = torch.multinomial(torch.nan_to_num(probs, nan=0.0), 1).squeeze(-1)

    chosen_actions = legal_actions[
        torch.arange(batch, device=device),
        row_choice,
    ]
    selected_actions = provided_action if provided_action is not None else chosen_actions
    matching_rows = row_mask & (
        legal_actions == selected_actions.unsqueeze(1)
    ).all(dim=-1)
    total_logprob = torch.logsumexp(
        log_probs.masked_fill(~matching_rows, -torch.inf),
        dim=-1,
    )

    if provided_action is not None:
        actions_out = provided_action
        if original_action_shape is not None:
            actions_out = actions_out.reshape(original_action_shape)
    else:
        actions_out = chosen_actions

    return actions_out, total_logprob, entropy


def legal_action_logprob_components(
    distribution: TCGLegalActionDistribution,
    action: torch.Tensor,
) -> torch.Tensor:
    """Decompose one legal-row log probability by action-tuple prefixes."""
    device = distribution.legal_action_logits.device
    batch = distribution.legal_action_logits.shape[0]
    original_shape = action.shape
    provided = action.to(device=device, dtype=torch.long).reshape(
        -1, ACTION_COMPONENT_COUNT
    )
    if provided.shape[0] != batch:
        raise ValueError(
            f"Provided action batch ({provided.shape[0]}) does not match logits batch ({batch})"
        )

    _, log_probs, row_mask = legal_action_row_distribution(distribution)
    legal_actions = distribution.legal_actions.to(device=device, dtype=torch.long)
    selected_matches = row_mask & (
        legal_actions == provided.unsqueeze(1)
    ).all(dim=-1)
    selected_logprob = torch.logsumexp(
        log_probs.masked_fill(~selected_matches, -torch.inf),
        dim=-1,
    )

    prefix_logprobs: list[torch.Tensor] = []
    prefix_match = row_mask
    for component_index in range(ACTION_COMPONENT_COUNT - 1):
        prefix_match = prefix_match & (
            legal_actions[..., component_index]
            == provided[:, component_index].unsqueeze(-1)
        )
        prefix_logprobs.append(
            torch.logsumexp(log_probs.masked_fill(~prefix_match, -torch.inf), dim=-1)
        )

    components = torch.stack(
        (
            prefix_logprobs[0],
            prefix_logprobs[1] - prefix_logprobs[0],
            prefix_logprobs[2] - prefix_logprobs[1],
            selected_logprob - prefix_logprobs[2],
        ),
        dim=-1,
    )
    return components.reshape(*original_shape[:-1], ACTION_COMPONENT_COUNT)


def _sample_stage(
    logits: torch.Tensor,
    mask: torch.Tensor,
    provided: torch.Tensor | None,
    *,
    temperature: float = 1.0,
    smoothing_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_logits = logits.masked_fill(~mask, MASK_MIN_VALUE)
    if temperature != 1.0:
        masked_logits = masked_logits / temperature

    probs = torch.softmax(masked_logits, dim=-1)
    if smoothing_eps > 0.0:
        legal = mask.to(dtype=probs.dtype)
        legal_count = legal.sum(dim=-1, keepdim=True).clamp(min=1.0)
        uniform = legal / legal_count
        probs = (1.0 - smoothing_eps) * probs + smoothing_eps * uniform

    log_probs = torch.log(probs + LOG_EPS)
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
    # Branch-free: rows with no legal entries fall back to "only column 0".
    # (`if valid.all()` would force a GPU->CPU sync per call.)
    first_col_only = torch.zeros_like(mask)
    first_col_only[:, 0] = True
    return torch.where(valid, mask, first_col_only)


def _match_provided_legal_rows(
    legal_actions: torch.Tensor,
    legal_action_count: torch.Tensor,
    provided_action: torch.Tensor,
) -> torch.Tensor:
    batch, candidate_count = legal_actions.shape[:2]
    row_indices = torch.arange(candidate_count, device=legal_actions.device).unsqueeze(0).expand(batch, -1)
    valid_rows = row_indices < legal_action_count.view(-1, 1)
    matches = valid_rows & (legal_actions == provided_action.unsqueeze(1)).all(dim=-1)
    any_match = matches.any(dim=-1)
    first_match = matches.to(dtype=torch.int64).argmax(dim=-1)
    return torch.where(any_match, first_match, torch.zeros_like(first_match))


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
