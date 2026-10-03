"""Per-seat element routing over element-specialist policies (evaluation only)."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from observation import DECKBUILD_OBSERVATION_CTYPE
from policy.tcg_distribution import TCGLegalActionDistribution
from policy.v2.tcg_sampler import tcg_argmax_logits
from specialist import LEARNER_ELEMENT_CODES, element_code_by_def_id

_ELEMENT_BY_CODE = {code: name for name, code in LEARNER_ELEMENT_CODES.items()}


def _gate_offset() -> int:
  dtype = np.dtype(DECKBUILD_OBSERVATION_CTYPE)
  deck_dtype, deck_offset = dtype.fields["deck_context"][:2]
  return int(deck_offset + deck_dtype.fields["gate_card_def_id"][1])


class SpecialistEnsemble(torch.nn.Module):
  """Route every row to the specialist of its own seat's gate element.

  The seat's gate is fixed for the whole episode (assigned at reset, kept
  through draft and battle), so each game's recurrent state stays with one
  member. Elements without a specialist fall back to `fallback`. Only the
  deterministic argmax evaluation path is supported: forward_eval returns a
  one-candidate distribution holding each member's argmax action.
  """

  def __init__(
    self,
    specialists: dict[str, tuple[str, torch.nn.Module]],
    fallback: tuple[str, torch.nn.Module],
  ) -> None:
    super().__init__()
    unknown = set(specialists) - {name for name in LEARNER_ELEMENT_CODES if name != "none"}
    if unknown:
      raise ValueError(f"Unknown specialist elements: {sorted(unknown)}")
    labels: list[str] = []
    modules: list[torch.nn.Module] = []
    for label, module in [fallback, *specialists.values()]:
      if label in labels:
        if modules[labels.index(label)] is not module:
          raise ValueError(f"Label {label!r} maps to two different policies")
        continue
      labels.append(label)
      modules.append(module)
    hidden = {int(module.hidden_size) for module in modules}
    if len(hidden) != 1:
      raise ValueError(f"Ensemble members disagree on hidden_size: {sorted(hidden)}")
    self.hidden_size = hidden.pop()
    self.labels = tuple(labels)
    self.members = torch.nn.ModuleList(modules)
    route = np.zeros(len(LEARNER_ELEMENT_CODES), dtype=np.int64)  # code -> member index
    for element, (label, _) in specialists.items():
      route[LEARNER_ELEMENT_CODES[element]] = labels.index(label)
    self.register_buffer("_route_by_code", torch.as_tensor(route), persistent=False)
    self.register_buffer(
      "_code_by_def_id", torch.as_tensor(element_code_by_def_id().astype(np.int64)), persistent=False
    )
    self._gate_offset = _gate_offset()
    self.routed_rows: Counter[str] = Counter()

  def element_codes(self, observations: torch.Tensor) -> torch.Tensor:
    low = observations[:, self._gate_offset].long()
    high = observations[:, self._gate_offset + 1].long()
    gate = low | (high << 8)
    gate = torch.where(gate >= 32768, gate - 65536, gate)
    valid = (gate >= 0) & (gate < self._code_by_def_id.numel())
    return torch.where(valid, self._code_by_def_id[gate.clamp(min=0, max=self._code_by_def_id.numel() - 1)], 0)

  def forward_eval(self, observations, state):
    codes = self.element_codes(observations)
    members = self._route_by_code[codes]
    batch = int(observations.shape[0])
    actions = torch.zeros(batch, 4, dtype=torch.long, device=observations.device)
    h = state.get("lstm_h")
    c = state.get("lstm_c")
    new_h = None if h is None else h.clone()
    new_c = None if c is None else c.clone()
    mask = state.get("mask")
    for index, member in enumerate(self.members):
      rows = torch.nonzero(members == index, as_tuple=False).flatten()
      if rows.numel() == 0:
        continue
      sub_state = {}
      if mask is not None:
        sub_state["mask"] = mask[rows]
      if h is not None:
        sub_state["lstm_h"] = h[rows]
        sub_state["lstm_c"] = c[rows]
      logits, _ = member.forward_eval(observations[rows], sub_state)
      actions[rows] = tcg_argmax_logits(logits).to(dtype=torch.long)
      if h is not None:
        new_h[rows] = sub_state["lstm_h"]
        new_c[rows] = sub_state["lstm_c"]
      for code, count in zip(*torch.unique(codes[rows], return_counts=True)):
        self.routed_rows[f"{_ELEMENT_BY_CODE[int(code)]}->{self.labels[index]}"] += int(count)
    if h is not None:
      state["lstm_h"] = new_h
      state["lstm_c"] = new_c
    distribution = TCGLegalActionDistribution(
      legal_action_logits=torch.zeros(batch, 1, device=observations.device),
      legal_actions=actions[:, None, :],
      legal_action_count=torch.ones(batch, dtype=torch.long, device=observations.device),
    )
    return distribution, None
