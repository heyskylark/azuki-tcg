from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


SCREEN_PHASE = "screen"
CONFIRMATION_PHASE = "confirmation"
REFERENCE_PHASE = "reference"
SCHEDULE_VERSION = "paired-v1"
UNIFORM_SCHEDULE_VERSION = "paired-context-v2"
REFERENCE_SCHEDULE_VERSION = "reference-v1"
UNIFORM_REFERENCE_SCHEDULE_VERSION = "reference-context-v2"

# The directed cycle gives every gate one seat-0 and one seat-1 appearance for
# each policy. Alternating sibling and cross-element edges preserves both local
# and adaptation checks in only eight paired blocks.
GATE_CODES = (
  "STT01-002",  # Surge
  "AZK01-120",  # Stormchain
  "STT04-002",  # Ragefire
  "AZK01-122",  # Rushfire
  "AZK01-124",  # Gate of Devotion
  "STT03-002",  # Stonehaven
  "AZK01-126",  # Echoed Waves
  "STT02-002",  # Hydromancy
)
CONFIRMATION_GATE_EDGES = tuple(
  (GATE_CODES[index], GATE_CODES[(index + 1) % len(GATE_CODES)])
  for index in range(len(GATE_CODES))
)


@dataclass(frozen=True)
class PromotionGameSpec:
  game_id: str
  block_id: str
  phase: str
  opponent_id: str
  seed: int
  candidate_seat: int
  gate0: int
  gate1: int
  leader0: int = -1
  leader1: int = -1
  schedule_version: str = SCHEDULE_VERSION
  reference_seat: int = -1
  reference_deck_index: int = -1

  def to_dict(self) -> dict:
    return asdict(self)


@dataclass(frozen=True)
class PromotionGameRecord:
  game_id: str
  block_id: str
  phase: str
  opponent_id: str
  seed: int
  candidate_seat: int
  candidate_gate: int
  opponent_gate: int
  winner_seat: int
  steps: int
  end_reason: str
  schedule_version: str = SCHEDULE_VERSION
  reference_seat: int = -1
  reference_deck_index: int = -1
  world_seed: int = -1
  starting_player: int = -1
  evaluator_version: str = "unknown"
  wall_time_seconds: float = 0.0
  candidate_leader: int = -1
  opponent_leader: int = -1

  @property
  def candidate_score(self) -> float:
    if self.winner_seat == self.candidate_seat:
      return 1.0
    if self.winner_seat in (0, 1):
      return 0.0
    return 0.5

  @property
  def completed_normally(self) -> bool:
    return self.end_reason == "gameover"

  def to_dict(self) -> dict:
    payload = asdict(self)
    payload["candidate_score"] = self.candidate_score
    return payload


@dataclass(frozen=True)
class PromotionThresholds:
  screen_pooled_min: float = 0.48
  screen_opponent_floor: float = 0.25
  screen_seat_floor: float = 0.35
  pooled_min: float = 0.52
  opponent_quorum_min: float = 0.45
  opponent_quorum_count: int = 3
  opponent_break_even_count: int = 2
  opponent_floor: float = 0.35
  seat_floor: float = 0.40
  paired_lcb_min: float = 0.48
  bootstrap_confidence: float = 0.80
  bootstrap_samples: int = 10_000
  reference_candidate_floor: float = 0.50
  reference_noninferiority: float = -0.15
  reference_paired_lcb_min: float = -0.18
  reference_leap: float = 0.05
  max_timeout_rate: float = 0.0
  anchored_screen_pooled_floor: float = 0.35
  anchored_screen_opponent_floor: float = 0.15
  anchored_screen_seat_floor: float = 0.25
  anchored_screen_relative_pooled_min: float = -0.08
  anchored_screen_relative_opponent_floor: float = -0.20
  anchored_screen_relative_seat_floor: float = -0.15
  anchored_pooled_floor: float = 0.40
  anchored_opponent_floor: float = 0.20
  anchored_seat_floor: float = 0.35
  anchored_relative_pooled_min: float = -0.02
  anchored_relative_opponent_quorum_min: float = -0.05
  anchored_relative_opponent_quorum_count: int = 3
  anchored_relative_break_even_count: int = 2
  anchored_relative_opponent_floor: float = -0.10
  anchored_relative_seat_floor: float = -0.08
  anchored_relative_paired_lcb_min: float = -0.05
  anchored_leap_relative_pooled_min: float = -0.05
  anchored_max_timeout_rate: float = 0.01
  anchored_max_timeout_delta: float = 0.01


@dataclass(frozen=True)
class PanelSummary:
  games: int
  blocks: int
  pooled_score: float
  paired_lcb: float
  opponent_scores: dict[str, float]
  opponent_games: dict[str, int]
  seat_scores: dict[int, float]
  gate_scores: dict[int, float]
  leader_scores: dict[int, float]
  context_scores: dict[str, float]
  phase_scores: dict[str, float]
  phase_games: dict[str, int]
  draws: int
  end_reason_counts: dict[str, int]
  timeout_rate: float

  def to_dict(self) -> dict:
    return asdict(self)


@dataclass(frozen=True)
class PanelComparison:
  games: int
  blocks: int
  candidate_score: float
  anchor_score: float
  delta: float
  paired_lcb: float
  opponent_deltas: dict[str, float]
  seat_deltas: dict[int, float]
  leader_deltas: dict[int, float]
  context_deltas: dict[str, float]
  candidate_timeout_rate: float
  anchor_timeout_rate: float
  timeout_delta: float

  def to_dict(self) -> dict:
    return asdict(self)


@dataclass(frozen=True)
class ReferenceComparison:
  games: int
  candidate_score: float
  anchor_score: float
  delta: float
  paired_lcb: float

  def to_dict(self) -> dict:
    return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
  admitted: bool
  route: str
  reasons: tuple[str, ...]
  summary: PanelSummary
  reference: ReferenceComparison | None
  panel_comparison: PanelComparison | None = None

  def to_dict(self) -> dict:
    payload = asdict(self)
    payload["summary"] = self.summary.to_dict()
    payload["reference"] = None if self.reference is None else self.reference.to_dict()
    payload["panel_comparison"] = (
      None if self.panel_comparison is None else self.panel_comparison.to_dict()
    )
    return payload


def _derived_seed(base_seed: int, opponent_index: int, phase_index: int, block_index: int) -> int:
  value = (
    int(base_seed)
    + 1_000_003 * int(opponent_index + 1)
    + 65_537 * int(phase_index + 1)
    + 7_919 * int(block_index + 1)
  )
  return int(value % 2_147_483_647)


def _gate_ids(records_by_code: Mapping[str, object]) -> dict[str, int]:
  out: dict[str, int] = {}
  for code in GATE_CODES:
    record = records_by_code.get(code)
    if record is None or not hasattr(record, "card_def_id"):
      raise ValueError(f"Promotion schedule gate {code} is missing from the deck-build catalog")
    out[code] = int(record.card_def_id)
  if len(set(out.values())) != len(GATE_CODES):
    raise ValueError("Promotion schedule gate ids must be unique")
  return out


def _leader_ids_by_gate(
  records_by_code: Mapping[str, object],
  leader_ids_by_element: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, int]]:
  out: dict[str, tuple[int, int]] = {}
  for code in GATE_CODES:
    record = records_by_code.get(code)
    if record is None or not hasattr(record, "element"):
      raise ValueError(f"Promotion schedule gate {code} has no element metadata")
    element = str(record.element)
    leaders = tuple(sorted({int(value) for value in leader_ids_by_element.get(element, ())}))
    if len(leaders) != 2:
      raise ValueError(
        f"Uniform promotion schedule requires two leaders for {element}; got {len(leaders)}"
      )
    out[code] = (leaders[0], leaders[1])
  return out


def build_panel_schedule(
  opponent_ids: Sequence[str],
  records_by_code: Mapping[str, object],
  *,
  base_seed: int,
  include_confirmation: bool,
  leader_ids_by_element: Mapping[str, Sequence[int]] | None = None,
) -> list[PromotionGameSpec]:
  if not opponent_ids:
    return []
  if len(set(opponent_ids)) != len(opponent_ids):
    raise ValueError("Promotion panel opponents must be unique")

  ids = _gate_ids(records_by_code)
  leader_ids = (
    None
    if leader_ids_by_element is None
    else _leader_ids_by_gate(records_by_code, leader_ids_by_element)
  )
  schedule_version = SCHEDULE_VERSION if leader_ids is None else UNIFORM_SCHEDULE_VERSION
  games: list[PromotionGameSpec] = []
  for opponent_index, opponent_id in enumerate(opponent_ids):
    screen_contexts = (
      [(code, -1) for code in GATE_CODES]
      if leader_ids is None
      else [
        (code, leader)
        for code in GATE_CODES
        for leader in leader_ids[code]
      ]
    )
    for block_index, (code, leader) in enumerate(screen_contexts):
      gate = ids[code]
      block_id = f"{opponent_id}:{SCREEN_PHASE}:{block_index:02d}"
      seed = _derived_seed(base_seed, opponent_index, 0, block_index)
      for candidate_seat in (0, 1):
        games.append(
          PromotionGameSpec(
            game_id=f"{block_id}:s{candidate_seat}",
            block_id=block_id,
            phase=SCREEN_PHASE,
            opponent_id=opponent_id,
            seed=seed,
            candidate_seat=candidate_seat,
            gate0=gate,
            gate1=gate,
            leader0=leader,
            leader1=leader,
            schedule_version=schedule_version,
          )
        )

    if not include_confirmation:
      continue
    for block_index, (code0, code1) in enumerate(CONFIRMATION_GATE_EDGES):
      block_id = f"{opponent_id}:{CONFIRMATION_PHASE}:{block_index:02d}"
      seed = _derived_seed(base_seed, opponent_index, 1, block_index)
      if leader_ids is None:
        leader0 = -1
        leader1 = -1
      else:
        variant = block_index % 2
        leader0 = leader_ids[code0][variant]
        leader1 = leader_ids[code1][variant]
      for candidate_seat in (0, 1):
        games.append(
          PromotionGameSpec(
            game_id=f"{block_id}:s{candidate_seat}",
            block_id=block_id,
            phase=CONFIRMATION_PHASE,
            opponent_id=opponent_id,
            seed=seed,
            candidate_seat=candidate_seat,
            gate0=ids[code0],
            gate1=ids[code1],
            leader0=leader0,
            leader1=leader1,
            schedule_version=schedule_version,
          )
        )
  return games


def build_reference_schedule(
  reference_deck_indices: Sequence[int],
  records_by_code: Mapping[str, object],
  *,
  base_seed: int,
  schedule_id: str = "",
  leader_ids_by_element: Mapping[str, Sequence[int]] | None = None,
  leader_assignment_offset: int = 0,
) -> list[PromotionGameSpec]:
  if not reference_deck_indices:
    return []
  if len(set(int(index) for index in reference_deck_indices)) != len(reference_deck_indices):
    raise ValueError("Reference deck indices must be unique")

  ids = _gate_ids(records_by_code)
  leader_ids = (
    None
    if leader_ids_by_element is None
    else _leader_ids_by_gate(records_by_code, leader_ids_by_element)
  )
  normalized_schedule_id = str(schedule_id).strip()
  normalized_leader_offset = int(leader_assignment_offset)
  if normalized_leader_offset not in (0, 1):
    raise ValueError("leader_assignment_offset must be 0 or 1")
  schedule_prefix = f"{normalized_schedule_id}:" if normalized_schedule_id else ""
  schedule_base = (
    REFERENCE_SCHEDULE_VERSION
    if leader_ids is None
    else UNIFORM_REFERENCE_SCHEDULE_VERSION
  )
  schedule_version = (
    f"{schedule_base}:{normalized_schedule_id}"
    if normalized_schedule_id
    else schedule_base
  )
  games: list[PromotionGameSpec] = []
  for deck_position, deck_index_raw in enumerate(reference_deck_indices):
    deck_index = int(deck_index_raw)
    if deck_index < 0:
      raise ValueError("Reference deck indices must be non-negative")
    for candidate_seat in (0, 1):
      for gate_position, code in enumerate(GATE_CODES):
        reference_seat = 1 - candidate_seat
        gate = ids[code]
        if leader_ids is None:
          candidate_leader = -1
          reference_placeholder_leader = -1
        else:
          candidate_leader = leader_ids[code][
            (candidate_seat + normalized_leader_offset) % 2
          ]
          # Native validation runs before the fixed-reference deck override,
          # so both scheduled seats need leaders compatible with this gate.
          reference_placeholder_leader = leader_ids[code][
            ((1 - candidate_seat) + normalized_leader_offset) % 2
          ]
        block_id = (
          f"ref:{schedule_prefix}{deck_index}:seat{candidate_seat}:gate{gate_position:02d}"
        )
        seed = _derived_seed(base_seed, deck_position * 2 + candidate_seat, 2, gate_position)
        games.append(
          PromotionGameSpec(
            game_id=block_id,
            block_id=block_id,
            phase=REFERENCE_PHASE,
            opponent_id=f"reference:{deck_index}",
            seed=seed,
            candidate_seat=candidate_seat,
            gate0=gate,
            gate1=gate,
            leader0=(
              candidate_leader
              if candidate_seat == 0
              else reference_placeholder_leader
            ),
            leader1=(
              candidate_leader
              if candidate_seat == 1
              else reference_placeholder_leader
            ),
            schedule_version=schedule_version,
            reference_seat=reference_seat,
            reference_deck_index=deck_index,
          )
        )
  return games


def _mean_score(records: Iterable[PromotionGameRecord]) -> float:
  values = [record.candidate_score for record in records]
  return float(np.mean(values)) if values else 0.0


def _paired_block_scores(records: Sequence[PromotionGameRecord]) -> dict[tuple[str, str], list[float]]:
  grouped: dict[tuple[str, str, str], list[PromotionGameRecord]] = defaultdict(list)
  for record in records:
    grouped[(record.opponent_id, record.phase, record.block_id)].append(record)

  strata: dict[tuple[str, str], list[float]] = defaultdict(list)
  for (opponent_id, phase, block_id), block_records in grouped.items():
    if len(block_records) != 2:
      raise ValueError(f"Paired block {block_id} has {len(block_records)} games; expected 2")
    seats = {record.candidate_seat for record in block_records}
    if seats != {0, 1}:
      raise ValueError(f"Paired block {block_id} does not contain both candidate seats")
    strata[(opponent_id, phase)].append(_mean_score(block_records))
  return strata


def paired_bootstrap_lower_bound(
  records: Sequence[PromotionGameRecord],
  *,
  confidence: float,
  samples: int,
  seed: int,
) -> float:
  if not 0.5 < confidence < 1.0:
    raise ValueError("bootstrap confidence must be in (0.5, 1.0)")
  if samples < 100:
    raise ValueError("bootstrap samples must be >= 100")
  strata = _paired_block_scores(records)
  if not strata:
    return 0.0

  rng = np.random.default_rng(int(seed))
  boot = np.zeros(int(samples), dtype=np.float64)
  total_blocks = sum(len(values) for values in strata.values())
  for values in strata.values():
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    boot += array[indices].sum(axis=1)
  boot /= float(total_blocks)
  return float(np.quantile(boot, 1.0 - float(confidence)))


def summarize_panel(
  records: Sequence[PromotionGameRecord],
  *,
  thresholds: PromotionThresholds,
  bootstrap_seed: int,
) -> PanelSummary:
  if not records:
    raise ValueError("Cannot summarize an empty promotion evaluation")
  if len({record.game_id for record in records}) != len(records):
    raise ValueError("Promotion game ids must be unique")

  strata = _paired_block_scores(records)
  by_opponent: dict[str, list[PromotionGameRecord]] = defaultdict(list)
  by_seat: dict[int, list[PromotionGameRecord]] = defaultdict(list)
  by_gate: dict[int, list[PromotionGameRecord]] = defaultdict(list)
  by_leader: dict[int, list[PromotionGameRecord]] = defaultdict(list)
  by_context: dict[str, list[PromotionGameRecord]] = defaultdict(list)
  by_phase: dict[str, list[PromotionGameRecord]] = defaultdict(list)
  end_reason_counts: dict[str, int] = defaultdict(int)
  for record in records:
    by_opponent[record.opponent_id].append(record)
    by_seat[record.candidate_seat].append(record)
    by_gate[record.candidate_gate].append(record)
    if record.candidate_leader >= 0:
      by_leader[record.candidate_leader].append(record)
      by_context[f"{record.candidate_gate}:{record.candidate_leader}"].append(record)
    by_phase[record.phase].append(record)
    end_reason_counts[record.end_reason] += 1

  return PanelSummary(
    games=len(records),
    blocks=sum(len(values) for values in strata.values()),
    pooled_score=_mean_score(records),
    paired_lcb=paired_bootstrap_lower_bound(
      records,
      confidence=thresholds.bootstrap_confidence,
      samples=thresholds.bootstrap_samples,
      seed=bootstrap_seed,
    ),
    opponent_scores={key: _mean_score(value) for key, value in sorted(by_opponent.items())},
    opponent_games={key: len(value) for key, value in sorted(by_opponent.items())},
    seat_scores={key: _mean_score(value) for key, value in sorted(by_seat.items())},
    gate_scores={key: _mean_score(value) for key, value in sorted(by_gate.items())},
    leader_scores={key: _mean_score(value) for key, value in sorted(by_leader.items())},
    context_scores={key: _mean_score(value) for key, value in sorted(by_context.items())},
    phase_scores={key: _mean_score(value) for key, value in sorted(by_phase.items())},
    phase_games={key: len(value) for key, value in sorted(by_phase.items())},
    draws=sum(record.winner_seat not in (0, 1) for record in records),
    end_reason_counts=dict(sorted(end_reason_counts.items())),
    timeout_rate=float(
      sum(not record.completed_normally for record in records) / len(records)
    ),
  )


def _panel_match_key(record: PromotionGameRecord) -> tuple:
  uniform_context = record.schedule_version.startswith(UNIFORM_SCHEDULE_VERSION)
  return (
    record.opponent_id,
    record.phase,
    record.seed,
    record.candidate_seat,
    record.candidate_gate,
    record.opponent_gate,
    record.candidate_leader if uniform_context else -1,
    record.opponent_leader if uniform_context else -1,
    record.schedule_version,
  )


def compare_panel_records(
  candidate_records: Sequence[PromotionGameRecord],
  anchor_records: Sequence[PromotionGameRecord],
  *,
  confidence: float,
  samples: int,
  seed: int,
) -> PanelComparison:
  if not 0.5 < confidence < 1.0:
    raise ValueError("bootstrap confidence must be in (0.5, 1.0)")
  if samples < 100:
    raise ValueError("bootstrap samples must be >= 100")

  candidate = {_panel_match_key(record): record for record in candidate_records}
  anchor = {_panel_match_key(record): record for record in anchor_records}
  if len(candidate) != len(candidate_records) or len(anchor) != len(anchor_records):
    raise ValueError("Panel comparison records must have unique schedule coordinates")
  if candidate.keys() != anchor.keys() or not candidate:
    raise ValueError("Candidate and anchor panel records must use identical schedules")

  game_deltas: dict[tuple, float] = {
    key: candidate[key].candidate_score - anchor[key].candidate_score
    for key in candidate
  }
  blocks: dict[tuple[str, str, int], list[float]] = defaultdict(list)
  strata: dict[tuple[str, str], list[float]] = defaultdict(list)
  for key, delta in game_deltas.items():
    opponent_id, phase, game_seed = key[0], key[1], key[2]
    blocks[(opponent_id, phase, game_seed)].append(delta)
  for (opponent_id, phase, game_seed), values in blocks.items():
    if len(values) != 2:
      raise ValueError(
        f"Panel comparison block {opponent_id}/{phase}/{game_seed} has "
        f"{len(values)} games; expected 2"
      )
    strata[(opponent_id, phase)].append(float(np.mean(values)))

  rng = np.random.default_rng(int(seed))
  bootstrap = np.zeros(int(samples), dtype=np.float64)
  total_blocks = sum(len(values) for values in strata.values())
  for values in strata.values():
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    bootstrap += array[indices].sum(axis=1)
  bootstrap /= float(total_blocks)

  opponent_deltas = {
    opponent_id: float(np.mean([
      delta for key, delta in game_deltas.items() if key[0] == opponent_id
    ]))
    for opponent_id in sorted({key[0] for key in game_deltas})
  }
  seat_deltas = {
    seat: float(np.mean([
      delta for key, delta in game_deltas.items() if key[3] == seat
    ]))
    for seat in sorted({int(key[3]) for key in game_deltas})
  }
  uniform_context = all(
    record.schedule_version.startswith(UNIFORM_SCHEDULE_VERSION)
    for record in candidate_records
  )
  leader_deltas = {}
  context_deltas = {}
  if uniform_context:
    leader_deltas = {
      leader: float(np.mean([
        delta for key, delta in game_deltas.items() if key[6] == leader
      ]))
      for leader in sorted({int(key[6]) for key in game_deltas})
    }
    context_deltas = {
      f"{gate}:{leader}": float(np.mean([
        delta
        for key, delta in game_deltas.items()
        if key[4] == gate and key[6] == leader
      ]))
      for gate, leader in sorted({(int(key[4]), int(key[6])) for key in game_deltas})
    }
  candidate_timeout_rate = float(
    sum(not record.completed_normally for record in candidate_records) / len(candidate_records)
  )
  anchor_timeout_rate = float(
    sum(not record.completed_normally for record in anchor_records) / len(anchor_records)
  )
  return PanelComparison(
    games=len(game_deltas),
    blocks=total_blocks,
    candidate_score=_mean_score(candidate_records),
    anchor_score=_mean_score(anchor_records),
    delta=float(np.mean(list(game_deltas.values()))),
    paired_lcb=float(np.quantile(bootstrap, 1.0 - float(confidence))),
    opponent_deltas=opponent_deltas,
    seat_deltas=seat_deltas,
    leader_deltas=leader_deltas,
    context_deltas=context_deltas,
    candidate_timeout_rate=candidate_timeout_rate,
    anchor_timeout_rate=anchor_timeout_rate,
    timeout_delta=candidate_timeout_rate - anchor_timeout_rate,
  )


def screen_passes(
  summary: PanelSummary,
  thresholds: PromotionThresholds,
  panel_comparison: PanelComparison | None = None,
) -> tuple[bool, tuple[str, ...]]:
  reasons: list[str] = []
  if panel_comparison is not None:
    if summary.pooled_score < thresholds.anchored_screen_pooled_floor:
      reasons.append("screen_pooled_catastrophic_floor_failed")
    if (
      summary.opponent_scores
      and min(summary.opponent_scores.values()) < thresholds.anchored_screen_opponent_floor
    ):
      reasons.append("screen_opponent_catastrophic_floor_failed")
    if summary.seat_scores and min(summary.seat_scores.values()) < thresholds.anchored_screen_seat_floor:
      reasons.append("screen_seat_catastrophic_floor_failed")
    if panel_comparison.delta < thresholds.anchored_screen_relative_pooled_min:
      reasons.append("screen_panel_relative_pooled_failed")
    if (
      panel_comparison.opponent_deltas
      and min(panel_comparison.opponent_deltas.values())
      < thresholds.anchored_screen_relative_opponent_floor
    ):
      reasons.append("screen_panel_relative_opponent_floor_failed")
    if (
      panel_comparison.seat_deltas
      and min(panel_comparison.seat_deltas.values())
      < thresholds.anchored_screen_relative_seat_floor
    ):
      reasons.append("screen_panel_relative_seat_floor_failed")
    if (
      panel_comparison.candidate_timeout_rate > thresholds.anchored_max_timeout_rate
      or panel_comparison.timeout_delta > thresholds.anchored_max_timeout_delta
    ):
      reasons.append("screen_timeout_rate_regressed")
    return not reasons, tuple(reasons)

  if summary.pooled_score < thresholds.screen_pooled_min:
    reasons.append("screen_pooled_score_too_low")
  if summary.opponent_scores and min(summary.opponent_scores.values()) < thresholds.screen_opponent_floor:
    reasons.append("screen_opponent_floor_failed")
  if summary.seat_scores and min(summary.seat_scores.values()) < thresholds.screen_seat_floor:
    reasons.append("screen_seat_floor_failed")
  if summary.timeout_rate > thresholds.max_timeout_rate:
    reasons.append("screen_timeout_rate_regressed")
  return not reasons, tuple(reasons)


def compare_reference_records(
  candidate_records: Sequence[PromotionGameRecord],
  anchor_records: Sequence[PromotionGameRecord],
  *,
  confidence: float,
  samples: int,
  seed: int,
) -> ReferenceComparison:
  candidate = {record.game_id: record for record in candidate_records}
  anchor = {record.game_id: record for record in anchor_records}
  if candidate.keys() != anchor.keys() or not candidate:
    raise ValueError("Candidate and anchor reference schedules must contain identical game ids")

  game_ids = sorted(candidate)
  deltas = np.asarray(
    [candidate[game_id].candidate_score - anchor[game_id].candidate_score for game_id in game_ids],
    dtype=np.float64,
  )
  rng = np.random.default_rng(int(seed))
  indices = rng.integers(0, deltas.size, size=(int(samples), deltas.size))
  bootstrap = deltas[indices].mean(axis=1)
  return ReferenceComparison(
    games=len(game_ids),
    candidate_score=_mean_score(candidate_records),
    anchor_score=_mean_score(anchor_records),
    delta=float(deltas.mean()),
    paired_lcb=float(np.quantile(bootstrap, 1.0 - float(confidence))),
  )


def decide_promotion(
  records: Sequence[PromotionGameRecord],
  *,
  thresholds: PromotionThresholds,
  reference: ReferenceComparison | None,
  require_reference: bool,
  bootstrap_seed: int,
  panel_comparison: PanelComparison | None = None,
) -> PromotionDecision:
  summary = summarize_panel(records, thresholds=thresholds, bootstrap_seed=bootstrap_seed)

  if panel_comparison is not None:
    if panel_comparison.games != summary.games or not np.isclose(
      panel_comparison.candidate_score, summary.pooled_score
    ):
      raise ValueError("Panel comparison does not describe the candidate records")
    reasons: list[str] = []
    if summary.pooled_score < thresholds.anchored_pooled_floor:
      reasons.append("pooled_catastrophic_floor_failed")
    if (
      summary.opponent_scores
      and min(summary.opponent_scores.values()) < thresholds.anchored_opponent_floor
    ):
      reasons.append("opponent_catastrophic_floor_failed")
    if summary.seat_scores and min(summary.seat_scores.values()) < thresholds.anchored_seat_floor:
      reasons.append("seat_catastrophic_floor_failed")
    if panel_comparison.delta < thresholds.anchored_relative_pooled_min:
      reasons.append("panel_relative_pooled_failed")
    relative_quorum = sum(
      delta >= thresholds.anchored_relative_opponent_quorum_min
      for delta in panel_comparison.opponent_deltas.values()
    )
    if relative_quorum < thresholds.anchored_relative_opponent_quorum_count:
      reasons.append("panel_relative_opponent_quorum_failed")
    relative_break_even = sum(
      delta >= 0.0 for delta in panel_comparison.opponent_deltas.values()
    )
    if relative_break_even < thresholds.anchored_relative_break_even_count:
      reasons.append("panel_relative_break_even_failed")
    if (
      panel_comparison.opponent_deltas
      and min(panel_comparison.opponent_deltas.values())
      < thresholds.anchored_relative_opponent_floor
    ):
      reasons.append("panel_relative_opponent_floor_failed")
    if (
      panel_comparison.seat_deltas
      and min(panel_comparison.seat_deltas.values()) < thresholds.anchored_relative_seat_floor
    ):
      reasons.append("panel_relative_seat_floor_failed")
    if panel_comparison.paired_lcb < thresholds.anchored_relative_paired_lcb_min:
      reasons.append("panel_relative_confidence_failed")
    if (
      panel_comparison.candidate_timeout_rate > thresholds.anchored_max_timeout_rate
      or panel_comparison.timeout_delta > thresholds.anchored_max_timeout_delta
    ):
      reasons.append("timeout_rate_regressed")

    if require_reference and reference is None:
      reasons.append("reference_result_missing")
    elif reference is not None:
      if reference.candidate_score < thresholds.reference_candidate_floor:
        reasons.append("reference_candidate_floor_failed")
      if reference.delta < thresholds.reference_noninferiority:
        reasons.append("reference_noninferiority_failed")
      if reference.paired_lcb < thresholds.reference_paired_lcb_min:
        reasons.append("reference_confidence_failed")

    if not reasons:
      return PromotionDecision(
        admitted=True,
        route="standard",
        reasons=(),
        summary=summary,
        reference=reference,
        panel_comparison=panel_comparison,
      )

    leap_waivable = {
      "panel_relative_pooled_failed",
      "panel_relative_opponent_quorum_failed",
      "panel_relative_break_even_failed",
      "panel_relative_confidence_failed",
      "reference_noninferiority_failed",
      "reference_confidence_failed",
    }
    non_waivable = [reason for reason in reasons if reason not in leap_waivable]
    leap_ok = (
      reference is not None
      and reference.delta >= thresholds.reference_leap
      and reference.paired_lcb > 0.0
      and panel_comparison.delta >= thresholds.anchored_leap_relative_pooled_min
      and not non_waivable
    )
    if leap_ok:
      return PromotionDecision(
        admitted=True,
        route="external_leap",
        reasons=(),
        summary=summary,
        reference=reference,
        panel_comparison=panel_comparison,
      )

    return PromotionDecision(
      admitted=False,
      route="rejected",
      reasons=tuple(reasons),
      summary=summary,
      reference=reference,
      panel_comparison=panel_comparison,
    )

  reasons: list[str] = []

  if summary.pooled_score < thresholds.pooled_min:
    reasons.append("pooled_score_too_low")
  quorum = sum(score >= thresholds.opponent_quorum_min for score in summary.opponent_scores.values())
  if quorum < thresholds.opponent_quorum_count:
    reasons.append("opponent_quorum_failed")
  break_even = sum(score >= 0.5 for score in summary.opponent_scores.values())
  if break_even < thresholds.opponent_break_even_count:
    reasons.append("opponent_break_even_failed")
  if summary.opponent_scores and min(summary.opponent_scores.values()) < thresholds.opponent_floor:
    reasons.append("opponent_floor_failed")
  if summary.seat_scores and min(summary.seat_scores.values()) < thresholds.seat_floor:
    reasons.append("seat_floor_failed")
  if summary.paired_lcb < thresholds.paired_lcb_min:
    reasons.append("paired_confidence_failed")
  if summary.timeout_rate > thresholds.max_timeout_rate:
    reasons.append("timeout_rate_regressed")

  if require_reference and reference is None:
    reasons.append("reference_result_missing")
  elif reference is not None:
    if reference.candidate_score < thresholds.reference_candidate_floor:
      reasons.append("reference_candidate_floor_failed")
    if reference.delta < thresholds.reference_noninferiority:
      reasons.append("reference_noninferiority_failed")
    if reference.paired_lcb < thresholds.reference_paired_lcb_min:
      reasons.append("reference_confidence_failed")

  if not reasons:
    return PromotionDecision(
      admitted=True,
      route="standard",
      reasons=(),
      summary=summary,
      reference=reference,
    )

  leap_reasons = {
    "pooled_score_too_low",
    "opponent_quorum_failed",
    "opponent_break_even_failed",
    "paired_confidence_failed",
    "reference_noninferiority_failed",
    "reference_confidence_failed",
  }
  non_waivable = [reason for reason in reasons if reason not in leap_reasons]
  leap_ok = (
    reference is not None
    and reference.delta >= thresholds.reference_leap
    and reference.paired_lcb > 0.0
    and summary.pooled_score >= thresholds.screen_pooled_min
    and not non_waivable
  )
  if leap_ok:
    return PromotionDecision(
      admitted=True,
      route="external_leap",
      reasons=(),
      summary=summary,
      reference=reference,
    )

  return PromotionDecision(
    admitted=False,
    route="rejected",
    reasons=tuple(reasons),
    summary=summary,
    reference=reference,
  )
