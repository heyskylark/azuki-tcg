"""Element-specialist training helpers: element codes and opponent-lineage guard."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

# Mirrors CardElement in include/generated/card_defs.h; 0 (NORMAL) = disabled.
LEARNER_ELEMENT_CODES: dict[str, int] = {
  "none": 0,
  "lightning": 1,
  "water": 2,
  "earth": 3,
  "fire": 4,
}
SPECIALIST_ELEMENTS: tuple[str, ...] = ("fire", "water", "earth", "lightning")
_METADATA_ELEMENT_NAMES = {"LIGHTNING": "lightning", "WATER": "water", "EARTH": "earth", "FIRE": "fire"}

# August 2026 1B-run checkpoints (corrected_production_1b_lr1500 lineage). They
# won with cheap aggro and no strategy; specialists must never train against or
# resume from them.
LEGACY_CHECKPOINT_SHA256: dict[str, str] = {
  "c04062e8b2f529c518422e8595b22184baf62be11f78ad69b21b525744336e1b": "p021000",
  "e16acde734167ea188d837cad645dffdb72e870d6e4d11d01521c6a6e8389b1c": "p029300",
  "5f8fa51ba75f86a18c478f7485301886c0636bf0d48c329caa06780c8dc2c336": "p044000",
  "912253140a792ae7f7e77ca8cd4b85ca9dc7bcd9b82d144244f0aba216d0d66d": "p052000",
  "3ba834ffc9cce5a9827ee0df38bc5eaecd6b8393ac814cebedd5fb7e22cb3237": "p060000",
  "c38640c04a1b11f337d900d7f551c4df0e3df6e754c997ba3c1d12f61245d026": "p065105",
}
# The 1B run's checkpoints start at 194,576,077 learner steps (p021000); the
# recent lineage (fresh prebuilt80_s43 start -> u9305) stays below 90M.
LEGACY_MIN_GLOBAL_STEP = 150_000_000
_LEGACY_PATH_MARKERS = ("production_1b", "/evaluation_opponents/")
# Specialist runs only ever resume from recent-lineage parents (enforced at
# launch), so their own checkpoints may pass the step heuristic's threshold.
SPECIALIST_RUNS_ROOT = Path(__file__).resolve().parents[2] / "train-specialist" / "runs"


def parse_learner_element(value: object) -> str:
  name = "none" if value is None else str(value).strip().lower()
  if name == "":
    name = "none"
  if name not in LEARNER_ELEMENT_CODES:
    raise ValueError(
      f"learner_element must be one of {sorted(LEARNER_ELEMENT_CODES)}; got {value!r}"
    )
  return name


@lru_cache(maxsize=None)
def element_code_by_def_id() -> np.ndarray:
  """int8 lookup: CardDefId -> LEARNER_ELEMENT_CODES value (0 for NORMAL/unknown)."""
  from deck_building import _load_policy_card_records

  records = _load_policy_card_records()
  table = np.zeros(max(record.card_def_id for record in records) + 1, dtype=np.int8)
  for record in records:
    name = _METADATA_ELEMENT_NAMES.get(str(record.element).upper())
    if name is not None:
      table[record.card_def_id] = LEARNER_ELEMENT_CODES[name]
  return table


def gate_element_codes(gate_def_ids: np.ndarray) -> np.ndarray:
  """Element code per gate def id; invalid ids (<0 or unknown) map to -1."""
  table = element_code_by_def_id()
  ids = np.asarray(gate_def_ids, dtype=np.int64)
  valid = (ids >= 0) & (ids < table.size)
  out = np.full(ids.shape, -1, dtype=np.int8)
  out[valid] = table[ids[valid]]
  return out


def _sha256(path: Path) -> str:
  with path.open("rb") as handle:
    return hashlib.file_digest(handle, "sha256").hexdigest()


def legacy_checkpoint_reason(path: str | Path) -> str | None:
  """Return why `path` belongs to the banned August 1B lineage, else None."""
  checkpoint = Path(path).expanduser()
  text = str(checkpoint.resolve()) if checkpoint.exists() else str(checkpoint)
  for marker in _LEGACY_PATH_MARKERS:
    if marker in text:
      return f"path contains {marker!r}"
  meta_path = Path(str(checkpoint) + ".meta.json")
  if meta_path.is_file():
    meta = json.loads(meta_path.read_text())
    global_step = int(meta.get("global_step", 0))
    in_specialist_runs = SPECIALIST_RUNS_ROOT in Path(text).parents
    if global_step >= LEGACY_MIN_GLOBAL_STEP and not in_specialist_runs:
      return f"global_step {global_step} >= {LEGACY_MIN_GLOBAL_STEP}"
  if checkpoint.is_file():
    digest = _sha256(checkpoint)
    if digest in LEGACY_CHECKPOINT_SHA256:
      return f"sha256 matches legacy {LEGACY_CHECKPOINT_SHA256[digest]}"
  return None


def assert_recent_lineage(paths) -> None:
  """Raise if any checkpoint path is from the banned August 1B run."""
  offenders = []
  for path in paths:
    reason = legacy_checkpoint_reason(path)
    if reason is not None:
      offenders.append(f"{path} ({reason})")
  if offenders:
    raise ValueError(
      "Legacy August 1B-run checkpoints are banned from specialist training/evaluation: "
      + "; ".join(offenders)
    )
