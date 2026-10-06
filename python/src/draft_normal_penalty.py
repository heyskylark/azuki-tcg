"""Leader-conditioned composition cost for complete 50-card draft trajectories.

AZK_DRAFT_NORMAL_PENALTY_CONFIG selects the leader JSON. Its COEF_INITIAL,
COEF_FINAL, ANNEAL_START_ROWS and ANNEAL_END_ROWS environment variables share
the AZK_DRAFT_NORMAL_PENALTY_ prefix and the existing sampled-row reward clock.
An unset configuration and zero coefficients disable this optional objective.

Count copies in all 50 selected main cards, including forced-prefix cards and
the final pick. Subtract the scheduled excess cost only from unforced draft
actor advantages; terminal win-probability targets and battle rewards stay
unchanged. A threshold is a soft allowance, never a card-legality restriction.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_DECK_SIZE = 50


def penalty_config_path(path: str | Path) -> Path:
  resolved = Path(path).expanduser()
  return resolved if resolved.is_absolute() else ROOT / resolved


def penalty_config_sha256(path: str | Path) -> str:
  return hashlib.sha256(penalty_config_path(path).read_bytes()).hexdigest()


def load_leader_normal_penalty(
  path: str | Path,
) -> tuple[dict[int, tuple[float, float]], frozenset[int]]:
  payload = json.loads(penalty_config_path(path).read_text())
  if (
    not isinstance(payload, dict)
    or payload.get("schema_id") != "azuki.leader_normal_penalty"
    or payload.get("schema_version") != 1
    or payload.get("main_deck_size") != MAIN_DECK_SIZE
    or set(payload) - {"schema_id", "schema_version", "main_deck_size", "leaders", "provenance"}
  ):
    raise ValueError("Invalid leader Normal-penalty configuration")
  metadata_bytes = (ROOT / "python/config/policy_card_metadata_v1.json").read_bytes()
  provenance = payload.get("provenance", {})
  if not isinstance(provenance, dict):
    raise ValueError("Normal-penalty provenance must be an object")
  expected_metadata = provenance.get("metadata_sha256")
  if expected_metadata is not None and hashlib.sha256(metadata_bytes).hexdigest() != expected_metadata:
    raise ValueError("Normal-penalty catalog differs from the registered metadata")
  metadata = json.loads(metadata_bytes)
  records = metadata["records"]
  leader_ids = {
    record["card_code"]: int(record["card_def_id"])
    for record in records if record["card_type"] == "LEADER"
  }
  leaders = payload.get("leaders")
  if not isinstance(leaders, dict) or set(leaders) != set(leader_ids):
    raise ValueError("Normal-penalty configuration must explicitly cover every supported leader")
  settings = {}
  for code, entry in leaders.items():
    if not isinstance(entry, dict) or set(entry) != {"max_normal_fraction", "weight"}:
      raise ValueError(f"Invalid Normal-penalty settings for {code}")
    threshold = entry["max_normal_fraction"]
    weight = entry["weight"]
    if (
      isinstance(threshold, bool) or not isinstance(threshold, (int, float))
      or not math.isfinite(threshold) or not 0 <= threshold < 1
      or isinstance(weight, bool) or not isinstance(weight, (int, float))
      or not math.isfinite(weight) or weight < 0
    ):
      raise ValueError(f"Invalid Normal-penalty threshold or weight for {code}")
    settings[leader_ids[code]] = (float(threshold), float(weight))
  normal_ids = frozenset(
    int(record["card_def_id"]) for record in records
    if record["element"] == "NORMAL" and record["card_type"] in ("ENTITY", "SPELL", "WEAPON")
  )
  return settings, normal_ids


def leader_normal_cost(
  normal_count: int,
  leader_id: int,
  settings: dict[int, tuple[float, float]],
) -> float:
  """Bounded quadratic excess above a leader's allowance; count copies, not names."""
  if isinstance(normal_count, bool) or not isinstance(normal_count, int) or not 0 <= normal_count <= MAIN_DECK_SIZE:
    raise ValueError("Normal-card count must describe a complete 50-card main deck")
  threshold, weight = settings[leader_id]
  excess = max(0.0, normal_count / MAIN_DECK_SIZE - threshold) / (1.0 - threshold)
  return weight * excess * excess
