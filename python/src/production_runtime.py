from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from math import gcd
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping


_FORBIDDEN_PROCESS_ENV = {
  "AZK_RESUME_ALLOW_BINDING_MISMATCH",
  "AZK_RESUME_ALLOW_SOURCE_DRIFT",
  "AZK_RESUME_ALLOW_SCHEDULE_REWIND",
  "AZK_RESUME_KEEP_CURRENT_REWARD_ENV",
  "AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV",
}


def _env_value(value: object) -> str | None:
  if value is None or value == "":
    return None
  if isinstance(value, bool):
    return "1" if value else "0"
  if isinstance(value, (list, tuple)):
    return ",".join(str(item) for item in value)
  return str(value)


def apply_process_environment(config: Mapping[str, object]) -> dict[str, str]:
  """Apply the declarative AZK_* environment section before env construction."""
  section = config.get("process_env")
  if section is None:
    return {}
  if not isinstance(section, Mapping):
    raise TypeError("process_env must be an INI section")

  applied: dict[str, str] = {}
  for raw_name, raw_value in section.items():
    name = str(raw_name).upper()
    if not name.startswith("AZK_"):
      raise ValueError(f"process_env only accepts AZK_* keys: {raw_name}")
    if name in _FORBIDDEN_PROCESS_ENV:
      raise ValueError(f"production config cannot enable resume escape hatch {name}")
    value = _env_value(raw_value)
    if value is None:
      os.environ.pop(name, None)
    else:
      os.environ[name] = value
      applied[name] = value
  return applied


def parse_patterns(raw: object, *, default: tuple[str, ...] = ("*",)) -> tuple[str, ...]:
  if raw is None:
    return default
  if isinstance(raw, (list, tuple)):
    values = tuple(str(item).strip() for item in raw if str(item).strip())
  else:
    values = tuple(item.strip() for item in str(raw).split(",") if item.strip())
  return values or default


def filter_numeric_metrics(logs: Mapping[str, object], patterns: Iterable[str]) -> dict[str, float]:
  selected: dict[str, float] = {}
  pattern_tuple = tuple(patterns)
  for key, value in logs.items():
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      continue
    if not any(fnmatchcase(str(key), pattern) for pattern in pattern_tuple):
      continue
    number = float(value)
    if number == number and number not in (float("inf"), float("-inf")):
      selected[str(key)] = number
  return selected


@dataclass(frozen=True)
class CheckpointPolicy:
  recovery_interval: int
  evaluation_interval: int
  milestone_interval: int
  recovery_keep: int
  evaluation_keep: int

  @classmethod
  def from_config(cls, config: Mapping[str, object]) -> CheckpointPolicy | None:
    section = config.get("artifacts")
    if not isinstance(section, Mapping) or not bool(section.get("tiered_checkpoints", False)):
      return None
    policy = cls(
      recovery_interval=int(section.get("recovery_interval_updates", 100)),
      evaluation_interval=int(section.get("evaluation_interval_updates", 250)),
      milestone_interval=int(section.get("milestone_interval_updates", 1000)),
      recovery_keep=int(section.get("recovery_keep", 3)),
      evaluation_keep=int(section.get("evaluation_keep", 12)),
    )
    policy.validate()
    return policy

  def validate(self) -> None:
    intervals = (self.recovery_interval, self.evaluation_interval, self.milestone_interval)
    if any(value <= 0 for value in intervals):
      raise ValueError("checkpoint intervals must be positive")
    if self.recovery_keep < 1 or self.evaluation_keep < 1:
      raise ValueError("checkpoint retention counts must be positive")

  @property
  def scheduler_interval(self) -> int:
    return gcd(gcd(self.recovery_interval, self.evaluation_interval), self.milestone_interval)

  def due(self, update: int, *, done: bool = False) -> bool:
    if done:
      return True
    return any(
      update % interval == 0
      for interval in (self.recovery_interval, self.evaluation_interval, self.milestone_interval)
    )

  def roles(self, update: int) -> tuple[str, ...]:
    roles = []
    if update % self.recovery_interval == 0:
      roles.append("recovery")
    if update % self.evaluation_interval == 0:
      roles.append("evaluation")
    if update % self.milestone_interval == 0:
      roles.append("milestone")
    return tuple(roles)

  def retained_updates(
    self,
    updates: Iterable[int],
    *,
    protected_updates: Iterable[int] = (),
  ) -> set[int]:
    available = sorted(set(int(update) for update in updates))
    retained = set(int(update) for update in protected_updates)
    milestones = [update for update in available if update % self.milestone_interval == 0]
    evaluations = [update for update in available if update % self.evaluation_interval == 0]
    recoveries = [update for update in available if update % self.recovery_interval == 0]
    retained.update(milestones)
    retained.update(evaluations[-self.evaluation_keep :])
    retained.update(recoveries[-self.recovery_keep :])
    if available:
      retained.add(available[-1])
    return retained


def checkpoint_update(path: Path) -> int | None:
  stem = path.stem
  suffix = stem.rsplit("_", 1)[-1]
  return int(suffix) if suffix.isdigit() else None


def checkpoint_updates(directory: Path) -> list[int]:
  updates = []
  for path in directory.glob("model_*.pt"):
    update = checkpoint_update(path)
    if update is not None:
      updates.append(update)
  return sorted(set(updates))


def active_league_updates(state_path: Path, *, checkpoint_dir: Path) -> set[int]:
  if not state_path.is_file():
    return set()
  payload = json.loads(state_path.read_text(encoding="utf-8"))
  policies = payload.get("policies")
  if not isinstance(policies, Mapping):
    return set()
  protected: set[int] = set()
  root = checkpoint_dir.resolve()
  for raw in policies.values():
    if not isinstance(raw, Mapping) or not bool(raw.get("active", False)):
      continue
    raw_path = raw.get("checkpoint_path")
    if not isinstance(raw_path, str):
      continue
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
      path = Path.cwd() / path
    try:
      if path.resolve().parent != root:
        continue
    except OSError:
      continue
    update = checkpoint_update(path)
    if update is not None:
      protected.add(update)
  return protected


def checkpoint_files(directory: Path, update: int) -> tuple[Path, ...]:
  suffix = f"{update:06d}"
  names = (
    f"model_azuki_local_{suffix}.pt",
    f"model_azuki_local_{suffix}.pt.meta.json",
    f"trainer_state_{suffix}.pt",
    f"league_state_{suffix}.json",
    f"promotion_state_{suffix}.json",
    f"checkpoint_{suffix}.manifest.json",
  )
  return tuple(directory / name for name in names)


def prune_checkpoints(
  directory: Path,
  policy: CheckpointPolicy,
  *,
  protected_updates: Iterable[int] = (),
) -> list[Path]:
  updates = checkpoint_updates(directory)
  retained = policy.retained_updates(updates, protected_updates=protected_updates)
  removed: list[Path] = []
  for update in updates:
    if update in retained:
      continue
    for path in checkpoint_files(directory, update):
      if path.exists():
        path.unlink()
        removed.append(path)
  return removed


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def finalize_checkpoint_state(
  checkpoint_path: Path,
  *,
  league_state_path: Path | None,
  promotion_state_path: Path | None,
  config_path: Path,
  roles: Iterable[str],
) -> Path:
  update = checkpoint_update(checkpoint_path)
  if update is None:
    raise ValueError(f"cannot parse checkpoint update: {checkpoint_path}")
  directory = checkpoint_path.parent
  suffix = f"{update:06d}"
  copied: list[Path] = [checkpoint_path]
  metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
  trainer_path = directory / f"trainer_state_{suffix}.pt"
  copied.extend((metadata_path, trainer_path))

  for source, destination in (
    (league_state_path, directory / f"league_state_{suffix}.json"),
    (promotion_state_path, directory / f"promotion_state_{suffix}.json"),
  ):
    if source is not None and source.is_file():
      shutil.copy2(source, destination)
      copied.append(destination)

  missing = [str(path) for path in copied if not path.is_file()]
  if missing:
    raise FileNotFoundError("checkpoint state is incomplete: " + ", ".join(missing))

  manifest = {
    "schema_version": 1,
    "update": update,
    "roles": sorted(set(str(role) for role in roles)),
    "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
    "artifacts": [
      {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
      for path in copied
    ],
  }
  output = directory / f"checkpoint_{suffix}.manifest.json"
  temporary = output.with_suffix(output.suffix + ".tmp")
  temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  temporary.replace(output)
  return output
