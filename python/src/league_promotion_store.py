from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from league_promotion import PromotionGameRecord, PromotionGameSpec


def checkpoint_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    while chunk := handle.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def schedule_sha256(games: Sequence[PromotionGameSpec]) -> str:
  payload = [game.to_dict() for game in games]
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def deck_sha256(deck) -> str:
  encoded = json.dumps(list(deck), sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _json_value(value):
  if is_dataclass(value):
    return asdict(value)
  if hasattr(value, "to_dict"):
    return value.to_dict()
  return value


def write_immutable_json(path: Path, payload: Mapping) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  encoded = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8")
  if path.exists():
    existing = path.read_bytes()
    if existing == encoded:
      return path
    raise FileExistsError(f"Refusing to replace immutable promotion artifact: {path}")

  fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(fd, "wb") as handle:
      handle.write(encoded)
      handle.flush()
      os.fsync(handle.fileno())
    try:
      os.link(tmp_name, path)
    except FileExistsError:
      existing = path.read_bytes()
      if existing != encoded:
        raise FileExistsError(f"Refusing to replace immutable promotion artifact: {path}")
  finally:
    try:
      os.unlink(tmp_name)
    except FileNotFoundError:
      pass
  return path


def write_promotion_run(
  root: Path,
  *,
  run_id: str,
  candidate_id: str,
  candidate_checkpoint: str,
  candidate_checkpoint_hash: str,
  panel_version: int,
  schedule: Sequence[PromotionGameSpec],
  records: Sequence[PromotionGameRecord],
  opponent_checkpoint_hashes: Mapping[str, str],
  metadata: Mapping,
  screen_result,
  decision,
) -> Path:
  specs = {game.game_id: game for game in schedule}
  if len(specs) != len(schedule):
    raise ValueError("Promotion schedule game ids must be unique")
  record_ids = {record.game_id for record in records}
  if record_ids != set(specs):
    missing = sorted(set(specs) - record_ids)
    extra = sorted(record_ids - set(specs))
    raise ValueError(f"Promotion raw records do not match schedule: missing={missing}, extra={extra}")

  enriched_games = []
  for record in records:
    item = record.to_dict()
    item.update(
      {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_checkpoint_hash": candidate_checkpoint_hash,
        "opponent_checkpoint_hash": opponent_checkpoint_hashes.get(record.opponent_id, ""),
        "panel_version": int(panel_version),
        "policy_action_mode": "legal_argmax_stable_first",
        "opponent_seat": 1 - record.candidate_seat,
        "draw": record.winner_seat not in (0, 1),
        "truncated": not record.completed_normally,
        "timeout": record.end_reason == "timeout",
      }
    )
    enriched_games.append(item)

  payload = {
    "schema_version": 1,
    "run_id": run_id,
    "candidate_id": candidate_id,
    "candidate_checkpoint": candidate_checkpoint,
    "candidate_checkpoint_hash": candidate_checkpoint_hash,
    "panel_version": int(panel_version),
    "schedule_hash": schedule_sha256(schedule),
    "schedule": [game.to_dict() for game in schedule],
    "metadata": dict(metadata),
    "screen_result": None if screen_result is None else _json_value(screen_result),
    "decision": None if decision is None else _json_value(decision),
    "games": enriched_games,
  }
  return write_immutable_json(root / f"{run_id}.json", payload)


def promotion_record_from_dict(payload: Mapping) -> PromotionGameRecord:
  fields = {
    "game_id",
    "block_id",
    "phase",
    "opponent_id",
    "seed",
    "candidate_seat",
    "candidate_gate",
    "opponent_gate",
    "winner_seat",
    "steps",
    "end_reason",
    "schedule_version",
    "reference_seat",
    "reference_deck_index",
    "world_seed",
    "starting_player",
    "evaluator_version",
    "wall_time_seconds",
    "candidate_leader",
    "opponent_leader",
  }
  return PromotionGameRecord(**{key: payload[key] for key in fields if key in payload})


def load_records(path: Path) -> list[PromotionGameRecord]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  games = payload.get("games", [])
  if not isinstance(games, list):
    raise ValueError(f"Promotion artifact has invalid games field: {path}")
  return [promotion_record_from_dict(item) for item in games]
