from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from league_ratings import LeagueRating


@dataclass
class LeaguePolicyEntry:
  policy_id: str
  checkpoint_path: str
  created_epoch: int
  created_ts: float
  source: str
  created_by_learner_id: str | None = None
  active: bool = True
  bucket: str = "recent"
  rating: LeagueRating = field(default_factory=lambda: LeagueRating(policy_id="unknown"))


@dataclass
class LeagueState:
  version: int = 1
  champion_policy_id: str | None = None
  learner_policy_id: str | None = None
  current_candidate_policy_id: str | None = None
  next_policy_index: int = 1
  policies: dict[str, LeaguePolicyEntry] = field(default_factory=dict)
  history: list[dict] = field(default_factory=list)


def _rating_from_json(policy_id: str, payload: dict) -> LeagueRating:
  return LeagueRating(
    policy_id=policy_id,
    elo=float(payload.get("elo", 1000.0)),
    games=int(payload.get("games", 0)),
    wins=int(payload.get("wins", 0)),
    losses=int(payload.get("losses", 0)),
    draws=int(payload.get("draws", 0)),
  )


def load_league_state(path: Path) -> LeagueState:
  if not path.exists():
    return LeagueState()

  primary_err = None
  raw = None
  try:
    raw = json.loads(path.read_text(encoding="utf-8"))
  except Exception as exc:  # pragma: no cover - defensive recovery
    primary_err = exc
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
      raw = json.loads(backup.read_text(encoding="utf-8"))
    else:
      raise

  state = LeagueState(
    version=int(raw.get("version", 1)),
    champion_policy_id=raw.get("champion_policy_id"),
    learner_policy_id=raw.get("learner_policy_id"),
    current_candidate_policy_id=raw.get("current_candidate_policy_id"),
    next_policy_index=int(raw.get("next_policy_index", 1)),
    history=list(raw.get("history", [])),
  )
  policies_raw = raw.get("policies", {})
  for policy_id, item in policies_raw.items():
    rating_item = item.get("rating", {})
    entry = LeaguePolicyEntry(
      policy_id=str(policy_id),
      checkpoint_path=str(item.get("checkpoint_path", "")),
      created_epoch=int(item.get("created_epoch", 0)),
      created_ts=float(item.get("created_ts", 0.0)),
      source=str(item.get("source", "unknown")),
      created_by_learner_id=item.get("created_by_learner_id"),
      active=bool(item.get("active", True)),
      bucket=str(item.get("bucket", "recent")),
      rating=_rating_from_json(str(policy_id), rating_item),
    )
    state.policies[policy_id] = entry
  if primary_err is not None:
    state.history.append(
      {
        "event": "state_recovered_from_backup",
        "ts": float(time.time()),
        "error": str(primary_err),
      }
    )
  return state


def save_league_state(path: Path, state: LeagueState) -> None:
  payload = {
    "version": int(state.version),
    "champion_policy_id": state.champion_policy_id,
    "learner_policy_id": state.learner_policy_id,
    "current_candidate_policy_id": state.current_candidate_policy_id,
    "next_policy_index": int(state.next_policy_index),
    "history": state.history,
    "policies": {
      policy_id: {
        **{k: v for k, v in asdict(entry).items() if k != "rating"},
        "rating": asdict(entry.rating),
      }
      for policy_id, entry in state.policies.items()
    },
  }
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


def register_policy(
  state: LeagueState,
  *,
  checkpoint_path: Path,
  created_epoch: int,
  source: str,
  created_by_learner_id: str | None = None,
) -> LeaguePolicyEntry:
  resolved = str(checkpoint_path.resolve())
  for existing in state.policies.values():
    if existing.checkpoint_path == resolved:
      if created_epoch > existing.created_epoch:
        existing.created_epoch = int(created_epoch)
      if created_by_learner_id is not None:
        existing.created_by_learner_id = str(created_by_learner_id)
      return existing

  policy_id = f"p{state.next_policy_index:06d}"
  state.next_policy_index += 1
  rating = LeagueRating(policy_id=policy_id)
  entry = LeaguePolicyEntry(
    policy_id=policy_id,
    checkpoint_path=resolved,
    created_epoch=int(created_epoch),
    created_ts=float(time.time()),
    source=source,
    created_by_learner_id=str(created_by_learner_id) if created_by_learner_id else None,
    active=True,
    bucket="recent",
    rating=rating,
  )
  state.policies[policy_id] = entry
  if state.champion_policy_id is None:
    state.champion_policy_id = policy_id
  return entry


def classify_and_prune(
  state: LeagueState,
  *,
  keep_recent: int,
  keep_mid: int,
  keep_old: int,
) -> list[str]:
  entries = sorted(state.policies.values(), key=lambda e: (e.created_epoch, e.created_ts))
  if not entries:
    return []

  ids = [entry.policy_id for entry in entries]
  champion_id = state.champion_policy_id
  keep_ids = set()
  if champion_id is not None and champion_id in state.policies:
    keep_ids.add(champion_id)

  old_slice = ids[: max(keep_old, 0)]
  recent_slice = ids[max(len(ids) - max(keep_recent, 0), 0) :]
  keep_ids.update(old_slice)
  keep_ids.update(recent_slice)

  remaining = [policy_id for policy_id in ids if policy_id not in keep_ids]
  if keep_mid > 0 and remaining:
    if len(remaining) <= keep_mid:
      keep_ids.update(remaining)
    else:
      step = max(len(remaining) // keep_mid, 1)
      keep_ids.update(remaining[::step][:keep_mid])

  for entry in state.policies.values():
    entry.active = entry.policy_id in keep_ids
    if not entry.active:
      entry.bucket = "pruned"
      continue
    if entry.policy_id in recent_slice:
      entry.bucket = "recent"
    elif entry.policy_id in old_slice:
      entry.bucket = "old"
    else:
      entry.bucket = "mid"

  return sorted(keep_ids)
