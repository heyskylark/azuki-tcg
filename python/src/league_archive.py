from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping, Sequence

from league_promotion import PromotionGameRecord
from league_state import LeaguePolicyEntry


ARCHIVE_STATE_VERSION = 1


@dataclass
class QualityArchiveEntry:
  policy_id: str
  admitted_epoch: int
  run_id: str
  route: str
  active: bool = True
  retired_epoch: int | None = None


@dataclass
class PanelMember:
  policy_id: str
  role: str
  checkpoint_path: str
  checkpoint_hash: str
  selection_evidence: dict = field(default_factory=dict)


@dataclass
class PanelManifest:
  version: int
  activated_epoch: int
  schedule_seed: int
  members: list[PanelMember]
  retired_epoch: int | None = None


@dataclass
class PayoffStat:
  games: int = 0
  points: float = 0.0
  wins: int = 0
  losses: int = 0
  draws: int = 0

  @property
  def score(self) -> float:
    return self.points / self.games if self.games else 0.5


@dataclass
class PromotionArchiveState:
  version: int = ARCHIVE_STATE_VERSION
  production_anchor_policy_id: str | None = None
  quality_archive: list[QualityArchiveEntry] = field(default_factory=list)
  panels: list[PanelManifest] = field(default_factory=list)
  active_panel_version: int | None = None
  payoff_matrix: dict[str, dict[str, PayoffStat]] = field(default_factory=dict)
  checkpoint_hashes: dict[str, str] = field(default_factory=dict)
  panel_cache: dict[str, str] = field(default_factory=dict)
  reference_cache: dict[str, str] = field(default_factory=dict)
  recorded_run_ids: list[str] = field(default_factory=list)
  history: list[dict] = field(default_factory=list)


def _quality_entry_from_json(payload: Mapping) -> QualityArchiveEntry:
  retired = payload.get("retired_epoch")
  return QualityArchiveEntry(
    policy_id=str(payload["policy_id"]),
    admitted_epoch=int(payload.get("admitted_epoch", 0)),
    run_id=str(payload.get("run_id", "unknown")),
    route=str(payload.get("route", "unknown")),
    active=bool(payload.get("active", True)),
    retired_epoch=None if retired is None else int(retired),
  )


def _panel_from_json(payload: Mapping) -> PanelManifest:
  retired = payload.get("retired_epoch")
  return PanelManifest(
    version=int(payload["version"]),
    activated_epoch=int(payload.get("activated_epoch", 0)),
    schedule_seed=int(payload.get("schedule_seed", 0)),
    members=[
      PanelMember(
        policy_id=str(item["policy_id"]),
        role=str(item.get("role", "unknown")),
        checkpoint_path=str(item.get("checkpoint_path", "")),
        checkpoint_hash=str(item.get("checkpoint_hash", "")),
        selection_evidence=dict(item.get("selection_evidence", {}) or {}),
      )
      for item in payload.get("members", [])
    ],
    retired_epoch=None if retired is None else int(retired),
  )


def load_promotion_archive_state(path: Path) -> PromotionArchiveState:
  if not path.exists():
    return PromotionArchiveState()
  raw = json.loads(path.read_text(encoding="utf-8"))
  matrix: dict[str, dict[str, PayoffStat]] = {}
  for policy_id, opponents in dict(raw.get("payoff_matrix", {}) or {}).items():
    matrix[str(policy_id)] = {
      str(opponent_id): PayoffStat(
        games=int(item.get("games", 0)),
        points=float(item.get("points", 0.0)),
        wins=int(item.get("wins", 0)),
        losses=int(item.get("losses", 0)),
        draws=int(item.get("draws", 0)),
      )
      for opponent_id, item in dict(opponents or {}).items()
    }
  return PromotionArchiveState(
    version=int(raw.get("version", ARCHIVE_STATE_VERSION)),
    production_anchor_policy_id=raw.get("production_anchor_policy_id"),
    quality_archive=[
      _quality_entry_from_json(item) for item in raw.get("quality_archive", [])
    ],
    panels=[_panel_from_json(item) for item in raw.get("panels", [])],
    active_panel_version=raw.get("active_panel_version"),
    payoff_matrix=matrix,
    checkpoint_hashes={
      str(key): str(value) for key, value in dict(raw.get("checkpoint_hashes", {}) or {}).items()
    },
    panel_cache={
      str(key): str(value) for key, value in dict(raw.get("panel_cache", {}) or {}).items()
    },
    reference_cache={
      str(key): str(value) for key, value in dict(raw.get("reference_cache", {}) or {}).items()
    },
    recorded_run_ids=[str(value) for value in raw.get("recorded_run_ids", [])],
    history=list(raw.get("history", [])),
  )


def save_promotion_archive_state(path: Path, state: PromotionArchiveState) -> None:
  payload = asdict(state)
  path.parent.mkdir(parents=True, exist_ok=True)
  encoded = json.dumps(payload, indent=2, sort_keys=True)
  tmp_path = path.with_suffix(path.suffix + ".tmp")
  backup = path.with_suffix(path.suffix + ".bak")
  tmp_path.write_text(encoded, encoding="utf-8")
  if path.exists():
    try:
      os.replace(path, backup)
    except OSError:
      pass
  os.replace(tmp_path, path)


def active_quality_ids(state: PromotionArchiveState) -> list[str]:
  return [entry.policy_id for entry in state.quality_archive if entry.active]


def active_panel(state: PromotionArchiveState) -> PanelManifest | None:
  if state.active_panel_version is None:
    return None
  for panel in state.panels:
    if panel.version == state.active_panel_version:
      return panel
  return None


def ensure_production_anchor(
  state: PromotionArchiveState,
  *,
  policy_id: str,
  epoch: int,
) -> None:
  if state.production_anchor_policy_id is None:
    state.production_anchor_policy_id = str(policy_id)
    state.quality_archive.append(
      QualityArchiveEntry(
        policy_id=str(policy_id),
        admitted_epoch=int(epoch),
        run_id="bootstrap",
        route="bootstrap_anchor",
      )
    )
    state.history.append(
      {
        "event": "production_anchor_bootstrapped",
        "policy_id": str(policy_id),
        "epoch": int(epoch),
        "ts": float(time.time()),
      }
    )


def _created_epoch(entries: Mapping[str, LeaguePolicyEntry], policy_id: str) -> int:
  entry = entries.get(policy_id)
  return int(entry.created_epoch) if entry is not None else -1


def _payoff_vector_distance(
  state: PromotionArchiveState,
  left: str,
  right: str,
) -> tuple[int, float]:
  left_row = state.payoff_matrix.get(left, {})
  right_row = state.payoff_matrix.get(right, {})
  common = sorted(set(left_row).intersection(right_row))
  if not common:
    return 0, 0.0
  squared = sum((left_row[key].score - right_row[key].score) ** 2 for key in common)
  return len(common), float((squared / len(common)) ** 0.5)


def _desired_panel_members(
  state: PromotionArchiveState,
  entries: Mapping[str, LeaguePolicyEntry],
  *,
  panel_size: int,
  exclude_ids: set[str],
) -> list[tuple[str, str, dict]]:
  anchor_id = state.production_anchor_policy_id
  if anchor_id is None or anchor_id not in entries:
    return []

  selected: list[tuple[str, str, dict]] = [
    (anchor_id, "production_anchor", {"source": "production_anchor"})
  ]
  selected_ids = {anchor_id}
  active_ids = [
    entry.policy_id
    for entry in sorted(
      entries.values(), key=lambda item: (item.created_epoch, item.created_ts, item.policy_id)
    )
    if entry.active and entry.policy_id != anchor_id and entry.policy_id not in exclude_ids
  ]
  quality = [
    item
    for item in state.quality_archive
    if item.active and item.policy_id in entries and item.policy_id not in exclude_ids
  ]

  recent_quality = sorted(
    (item for item in quality if item.policy_id not in selected_ids),
    key=lambda item: (item.admitted_epoch, _created_epoch(entries, item.policy_id), item.policy_id),
    reverse=True,
  )
  if recent_quality:
    item = recent_quality[0]
    selected.append(
      (
        item.policy_id,
        "recent_quality",
        {"admitted_epoch": item.admitted_epoch, "route": item.route},
      )
    )
    selected_ids.add(item.policy_id)

  anchor_payoffs = state.payoff_matrix.get(anchor_id, {})
  hardest_candidates: dict[str, tuple[float, int, str]] = {
    policy_id: (
      anchor_payoffs[policy_id].score,
      anchor_payoffs[policy_id].games,
      "live_payoff_matrix",
    )
    for policy_id in active_ids
    if policy_id not in selected_ids
    and policy_id in anchor_payoffs
    and anchor_payoffs[policy_id].games > 0
  }
  current = active_panel(state)
  if current is not None:
    for member in current.members:
      if (
        member.role != "hardest_retained"
        or member.policy_id in selected_ids
        or member.policy_id not in active_ids
        or member.policy_id in hardest_candidates
      ):
        continue
      raw_score = member.selection_evidence.get("anchor_score")
      if not isinstance(raw_score, (int, float)) or not 0.0 <= float(raw_score) <= 1.0:
        continue
      hardest_candidates[member.policy_id] = (
        float(raw_score),
        int(member.selection_evidence.get("games", 0)),
        "frozen_panel_evidence",
      )
  hardest = sorted(
    (score, -games, policy_id, source)
    for policy_id, (score, games, source) in hardest_candidates.items()
  )
  if hardest:
    score, neg_games, policy_id, source = hardest[0]
    selected.append(
      (
        policy_id,
        "hardest_retained",
        {"anchor_score": score, "games": -neg_games, "source": source},
      )
    )
    selected_ids.add(policy_id)

  distinct_candidates: list[tuple[float, int, int, str]] = []
  for item in quality:
    if item.policy_id in selected_ids:
      continue
    distances = [_payoff_vector_distance(state, item.policy_id, member_id) for member_id in selected_ids]
    comparable = sum(count for count, _ in distances)
    min_distance = min((distance for count, distance in distances if count > 0), default=0.0)
    epoch_separation = min(
      abs(_created_epoch(entries, item.policy_id) - _created_epoch(entries, member_id))
      for member_id in selected_ids
    )
    distinct_candidates.append((min_distance, comparable, epoch_separation, item.policy_id))
  if distinct_candidates:
    distance, comparable, separation, policy_id = max(distinct_candidates)
    selected.append(
      (
        policy_id,
        "historically_distinct",
        {
          "payoff_vector_distance": distance,
          "comparable_cells": comparable,
          "epoch_separation": separation,
        },
      )
    )
    selected_ids.add(policy_id)

  # Young archives usually lack three qualified members or a measured payoff
  # matrix. Fill deterministically with the most epoch-separated retained
  # checkpoints, then the oldest retained checkpoint.
  while len(selected) < panel_size:
    candidates = [policy_id for policy_id in active_ids if policy_id not in selected_ids]
    if not candidates:
      break
    candidates.sort(
      key=lambda policy_id: (
        -min(
          abs(_created_epoch(entries, policy_id) - _created_epoch(entries, member_id))
          for member_id in selected_ids
        ),
        _created_epoch(entries, policy_id),
        policy_id,
      )
    )
    policy_id = candidates[0]
    separation = min(
      abs(_created_epoch(entries, policy_id) - _created_epoch(entries, member_id))
      for member_id in selected_ids
    )
    selected.append(
      (
        policy_id,
        "fallback_historical",
        {"epoch_separation": separation, "reason": "archive_or_payoff_matrix_young"},
      )
    )
    selected_ids.add(policy_id)

  return selected[:panel_size]


def ensure_panel_manifest(
  state: PromotionArchiveState,
  entries: Mapping[str, LeaguePolicyEntry],
  *,
  epoch: int,
  panel_size: int,
  refresh_epochs: int,
  base_seed: int,
  checkpoint_hash: Callable[[str], str],
  exclude_ids: set[str] | None = None,
) -> PanelManifest | None:
  desired = _desired_panel_members(
    state,
    entries,
    panel_size=panel_size,
    exclude_ids=set() if exclude_ids is None else set(exclude_ids),
  )
  if len(desired) != panel_size:
    return None

  current = active_panel(state)
  if current is not None and int(epoch) - current.activated_epoch < refresh_epochs:
    return current

  desired_by_id = {policy_id: (role, evidence) for policy_id, role, evidence in desired}
  if current is None:
    next_members = desired
  else:
    current_ids = [member.policy_id for member in current.members]
    anchor_id = state.production_anchor_policy_id
    additions = [item for item in desired if item[0] not in current_ids]
    removable = [
      member for member in reversed(current.members)
      if member.policy_id != anchor_id and member.policy_id not in desired_by_id
    ]
    if not additions or not removable:
      return current
    remove_id = removable[0].policy_id
    addition = additions[0]
    next_members = []
    for member in current.members:
      if member.policy_id == remove_id:
        next_members.append(addition)
      else:
        role, evidence = desired_by_id.get(
          member.policy_id,
          (member.role, dict(member.selection_evidence)),
        )
        next_members.append((member.policy_id, role, evidence))
    current.retired_epoch = int(epoch)

  version = max((panel.version for panel in state.panels), default=0) + 1
  manifest = PanelManifest(
    version=version,
    activated_epoch=int(epoch),
    schedule_seed=int(base_seed) + 1_000_003 * version,
    members=[
      PanelMember(
        policy_id=policy_id,
        role=role,
        checkpoint_path=entries[policy_id].checkpoint_path,
        checkpoint_hash=checkpoint_hash(policy_id),
        selection_evidence=evidence,
      )
      for policy_id, role, evidence in next_members
    ],
  )
  state.panels.append(manifest)
  state.active_panel_version = manifest.version
  state.history.append(
    {
      "event": "panel_activated",
      "version": manifest.version,
      "epoch": int(epoch),
      "members": [member.policy_id for member in manifest.members],
      "ts": float(time.time()),
    }
  )
  return manifest


def record_payoff_results(
  state: PromotionArchiveState,
  *,
  run_id: str,
  candidate_id: str,
  records: Sequence[PromotionGameRecord],
) -> bool:
  if run_id in state.recorded_run_ids:
    return False
  grouped: dict[str, list[PromotionGameRecord]] = {}
  for record in records:
    grouped.setdefault(record.opponent_id, []).append(record)
  for opponent_id, opponent_records in grouped.items():
    if opponent_id.startswith("reference:"):
      continue
    candidate_row = state.payoff_matrix.setdefault(candidate_id, {})
    opponent_row = state.payoff_matrix.setdefault(opponent_id, {})
    forward = candidate_row.setdefault(opponent_id, PayoffStat())
    reverse = opponent_row.setdefault(candidate_id, PayoffStat())
    for record in opponent_records:
      score = record.candidate_score
      forward.games += 1
      forward.points += score
      reverse.games += 1
      reverse.points += 1.0 - score
      if score == 1.0:
        forward.wins += 1
        reverse.losses += 1
      elif score == 0.0:
        forward.losses += 1
        reverse.wins += 1
      else:
        forward.draws += 1
        reverse.draws += 1
  state.recorded_run_ids.append(str(run_id))
  return True


def admit_quality_policy(
  state: PromotionArchiveState,
  *,
  policy_id: str,
  epoch: int,
  run_id: str,
  route: str,
  max_active: int,
) -> None:
  existing = next((item for item in state.quality_archive if item.policy_id == policy_id), None)
  if existing is None:
    state.quality_archive.append(
      QualityArchiveEntry(
        policy_id=policy_id,
        admitted_epoch=int(epoch),
        run_id=str(run_id),
        route=str(route),
      )
    )
  else:
    existing.active = True
    existing.retired_epoch = None
    existing.admitted_epoch = int(epoch)
    existing.run_id = str(run_id)
    existing.route = str(route)

  active = [item for item in state.quality_archive if item.active]
  panel_ids = {member.policy_id for member in (active_panel(state).members if active_panel(state) else [])}
  while len(active) > max_active:
    candidates = [
      item
      for item in active
      if item.policy_id != state.production_anchor_policy_id and item.policy_id not in panel_ids
    ]
    if not candidates:
      candidates = [item for item in active if item.policy_id != state.production_anchor_policy_id]
    if not candidates:
      break
    retire = min(candidates, key=lambda item: (item.admitted_epoch, item.policy_id))
    retire.active = False
    retire.retired_epoch = int(epoch)
    active = [item for item in state.quality_archive if item.active]

  state.history.append(
    {
      "event": "quality_archive_admission",
      "policy_id": policy_id,
      "epoch": int(epoch),
      "run_id": str(run_id),
      "route": str(route),
      "ts": float(time.time()),
    }
  )


def protected_policy_ids(state: PromotionArchiveState) -> set[str]:
  protected = set(active_quality_ids(state))
  if state.production_anchor_policy_id is not None:
    protected.add(state.production_anchor_policy_id)
  panel = active_panel(state)
  if panel is not None:
    protected.update(member.policy_id for member in panel.members)
  return protected
