from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


ARTIFACT_SCHEMA_VERSION = 1
MAX_MAIN_DECK_SIZE = 50


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def prefix_quartile(main_count: int) -> str:
  count = int(main_count)
  if not 0 <= count <= MAX_MAIN_DECK_SIZE:
    raise ValueError("main_count must be in [0, 50]")
  if count <= 12:
    return "q1_00_12"
  if count <= 25:
    return "q2_13_25"
  if count <= 37:
    return "q3_26_37"
  return "q4_38_50"


class FrozenDraftPrefixPredictor:
  """Small NumPy inference runtime for a frozen draft-prefix outcome model."""

  def __init__(self, artifact_path: Path, *, expected_sha256: str | None = None):
    path = Path(artifact_path).resolve()
    if not path.is_file():
      raise FileNotFoundError(f"draft prefix outcome model not found: {path}")
    actual_sha256 = file_sha256(path)
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
      raise ValueError(
        "draft prefix outcome model hash mismatch: "
        f"expected {expected_sha256.lower()}, got {actual_sha256}"
      )

    with np.load(path, allow_pickle=False) as artifact:
      required = {
        "schema_version",
        "gate_ids",
        "leader_ids",
        "card_embedding",
        "gate_embedding",
        "leader_embedding",
        "seat_embedding",
        "fc1_weight",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
      }
      missing = sorted(required.difference(artifact.files))
      if missing:
        raise ValueError(f"draft prefix outcome artifact missing arrays: {missing}")
      schema_version = int(np.asarray(artifact["schema_version"]).reshape(-1)[0])
      if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
          f"unsupported draft prefix outcome schema {schema_version}; "
          f"expected {ARTIFACT_SCHEMA_VERSION}"
        )
      self.gate_ids = np.asarray(artifact["gate_ids"], dtype=np.int16).copy()
      self.leader_ids = np.asarray(artifact["leader_ids"], dtype=np.int16).copy()
      self.card_embedding = np.asarray(
        artifact["card_embedding"], dtype=np.float32
      ).copy()
      self.gate_embedding = np.asarray(
        artifact["gate_embedding"], dtype=np.float32
      ).copy()
      self.leader_embedding = np.asarray(
        artifact["leader_embedding"], dtype=np.float32
      ).copy()
      self.seat_embedding = np.asarray(
        artifact["seat_embedding"], dtype=np.float32
      ).copy()
      self.fc1_weight = np.asarray(artifact["fc1_weight"], dtype=np.float32).copy()
      self.fc1_bias = np.asarray(artifact["fc1_bias"], dtype=np.float32).copy()
      self.fc2_weight = np.asarray(artifact["fc2_weight"], dtype=np.float32).copy()
      self.fc2_bias = np.asarray(artifact["fc2_bias"], dtype=np.float32).copy()

    self.path = path
    self.sha256 = actual_sha256
    self._validate_shapes()
    self._gate_lookup = np.full(int(self.gate_ids.max()) + 1, -1, dtype=np.int16)
    self._gate_lookup[self.gate_ids.astype(np.int64)] = np.arange(
      self.gate_ids.size, dtype=np.int16
    )
    self._leader_lookup = np.full(int(self.leader_ids.max()) + 1, -1, dtype=np.int16)
    self._leader_lookup[self.leader_ids.astype(np.int64)] = np.arange(
      self.leader_ids.size, dtype=np.int16
    )
    self._positions = np.arange(MAX_MAIN_DECK_SIZE, dtype=np.int64)[None, :]

  def _validate_shapes(self) -> None:
    if self.gate_ids.ndim != 1 or self.gate_ids.size < 1:
      raise ValueError("gate_ids must be a nonempty vector")
    if self.leader_ids.ndim != 1 or self.leader_ids.size < 1:
      raise ValueError("leader_ids must be a nonempty vector")
    if len(set(self.gate_ids.tolist())) != self.gate_ids.size:
      raise ValueError("gate_ids must be unique")
    if len(set(self.leader_ids.tolist())) != self.leader_ids.size:
      raise ValueError("leader_ids must be unique")
    if self.gate_embedding.shape[0] != self.gate_ids.size:
      raise ValueError("gate embedding row count does not match gate_ids")
    if self.leader_embedding.shape[0] != self.leader_ids.size:
      raise ValueError("leader embedding row count does not match leader_ids")
    if self.seat_embedding.shape[0] != 2:
      raise ValueError("seat embedding must contain exactly two seats")
    if self.card_embedding.ndim != 2 or self.card_embedding.shape[0] < 1:
      raise ValueError("card_embedding must be a nonempty matrix")
    feature_width = (
      self.card_embedding.shape[1]
      + self.gate_embedding.shape[1]
      + self.leader_embedding.shape[1]
      + self.seat_embedding.shape[1]
      + 4
    )
    if self.fc1_weight.ndim != 2 or self.fc1_weight.shape[1] != feature_width:
      raise ValueError("fc1_weight has an invalid input width")
    if self.fc1_bias.shape != (self.fc1_weight.shape[0],):
      raise ValueError("fc1_bias has an invalid shape")
    if self.fc2_weight.shape != (1, self.fc1_weight.shape[0]):
      raise ValueError("fc2_weight has an invalid shape")
    if self.fc2_bias.shape != (1,):
      raise ValueError("fc2_bias has an invalid shape")

  @staticmethod
  def _indices(values: np.ndarray, lookup: np.ndarray, label: str) -> np.ndarray:
    valid_range = (values >= 0) & (values < lookup.size)
    safe_values = np.where(valid_range, values, 0)
    indices = lookup[safe_values].astype(np.int64, copy=False)
    valid = valid_range & (indices >= 0)
    if not bool(valid.all()):
      raise ValueError(f"unknown {label} id: {int(values[~valid][0])}")
    return indices

  def _predict_from_card_features(
    self,
    *,
    gate_values: np.ndarray,
    leader_values: np.ndarray,
    seat_values: np.ndarray,
    count_values: np.ndarray,
    card_features: np.ndarray,
  ) -> np.ndarray:
    gate_indices = self._indices(gate_values, self._gate_lookup, "gate")
    leader_indices = self._indices(
      leader_values, self._leader_lookup, "leader"
    )
    fraction = count_values.astype(np.float32) / float(MAX_MAIN_DECK_SIZE)
    position_features = np.stack(
      (
        fraction,
        fraction * fraction,
        np.sqrt(fraction),
        (count_values == MAX_MAIN_DECK_SIZE).astype(np.float32),
      ),
      axis=1,
    )
    features = np.concatenate(
      (
        card_features,
        self.gate_embedding[gate_indices],
        self.leader_embedding[leader_indices],
        self.seat_embedding[seat_values],
        position_features,
      ),
      axis=1,
      dtype=np.float32,
    )
    hidden = np.maximum(features @ self.fc1_weight.T + self.fc1_bias, 0.0)
    logits = (hidden @ self.fc2_weight.T + self.fc2_bias).reshape(-1)
    logits = np.clip(logits, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)

  def predict_probability_from_card_sums(
    self,
    *,
    gate_ids: np.ndarray,
    leader_ids: np.ndarray,
    seats: np.ndarray,
    card_embedding_sums: np.ndarray,
    main_counts: np.ndarray,
  ) -> np.ndarray:
    gate_values = np.asarray(gate_ids, dtype=np.int16).reshape(-1)
    leader_values = np.asarray(leader_ids, dtype=np.int16).reshape(-1)
    seat_values = np.asarray(seats, dtype=np.int64).reshape(-1)
    count_values = np.asarray(main_counts, dtype=np.int64).reshape(-1)
    card_sums = np.asarray(card_embedding_sums, dtype=np.float32)
    batch_size = gate_values.size
    if card_sums.shape != (batch_size, self.card_embedding.shape[1]):
      raise ValueError("card_embedding_sums has an invalid shape")
    if not (
      leader_values.size == batch_size
      and seat_values.size == batch_size
      and count_values.size == batch_size
    ):
      raise ValueError("predictor inputs must have matching batch dimensions")
    if np.any((seat_values < 0) | (seat_values > 1)):
      raise ValueError("seat ids must be 0 or 1")
    if np.any((count_values < 0) | (count_values > MAX_MAIN_DECK_SIZE)):
      raise ValueError("main counts must be in [0, 50]")
    return self._predict_from_card_features(
      gate_values=gate_values,
      leader_values=leader_values,
      seat_values=seat_values,
      count_values=count_values,
      card_features=card_sums / float(MAX_MAIN_DECK_SIZE),
    )

  def predict_probability(
    self,
    *,
    gate_ids: np.ndarray,
    leader_ids: np.ndarray,
    seats: np.ndarray,
    main_card_ids: np.ndarray,
    main_counts: np.ndarray,
  ) -> np.ndarray:
    gate_values = np.asarray(gate_ids, dtype=np.int16).reshape(-1)
    leader_values = np.asarray(leader_ids, dtype=np.int16).reshape(-1)
    seat_values = np.asarray(seats, dtype=np.int64).reshape(-1)
    count_values = np.asarray(main_counts, dtype=np.int64).reshape(-1)
    cards = np.asarray(main_card_ids, dtype=np.int64)
    batch_size = gate_values.size
    if cards.shape != (batch_size, MAX_MAIN_DECK_SIZE):
      raise ValueError(
        f"main_card_ids must have shape ({batch_size}, {MAX_MAIN_DECK_SIZE})"
      )
    if not (
      leader_values.size == batch_size
      and seat_values.size == batch_size
      and count_values.size == batch_size
    ):
      raise ValueError("predictor inputs must have matching batch dimensions")
    if np.any((seat_values < 0) | (seat_values > 1)):
      raise ValueError("seat ids must be 0 or 1")
    if np.any((count_values < 0) | (count_values > MAX_MAIN_DECK_SIZE)):
      raise ValueError("main counts must be in [0, 50]")

    selected = self._positions < count_values[:, None]
    selected_cards = cards[selected]
    if selected_cards.size and (
      selected_cards.min() < 0 or selected_cards.max() >= self.card_embedding.shape[0]
    ):
      raise ValueError("selected prefix contains an unknown card id")
    safe_cards = np.where(selected, cards, 0)
    card_features = self.card_embedding[safe_cards]
    card_features *= selected[:, :, None]
    card_features = card_features.sum(axis=1, dtype=np.float32) / float(
      MAX_MAIN_DECK_SIZE
    )

    return self._predict_from_card_features(
      gate_values=gate_values,
      leader_values=leader_values,
      seat_values=seat_values,
      count_values=count_values,
      card_features=card_features,
    )

  def predict_potential(self, **kwargs) -> np.ndarray:
    return 2.0 * self.predict_probability(**kwargs) - 1.0


class PrefixOutcomeRedistributor:
  """Stateful, exactly telescoping potential redistribution for live rollouts."""

  def __init__(
    self,
    predictor: FrozenDraftPrefixPredictor,
    *,
    total_agents: int,
    coefficient: float = 1.0,
  ):
    if total_agents < 1:
      raise ValueError("total_agents must be positive")
    if coefficient < 0.0 or not np.isfinite(coefficient):
      raise ValueError("prefix outcome coefficient must be finite and nonnegative")
    self.predictor = predictor
    self.total_agents = int(total_agents)
    self.coefficient = float(coefficient)
    self.active = np.zeros(total_agents, dtype=np.bool_)
    self.episode_ids = np.full(total_agents, -1, dtype=np.int64)
    self.ignored_episode_ids = np.full(total_agents, -1, dtype=np.int64)
    self.q0 = np.zeros(total_agents, dtype=np.float32)
    self.qprev = np.zeros(total_agents, dtype=np.float32)
    self.main_counts = np.full(total_agents, -1, dtype=np.int16)
    self.gate_ids = np.full(total_agents, -1, dtype=np.int16)
    self.leader_ids = np.full(total_agents, -1, dtype=np.int16)
    self.card_embedding_sums = np.zeros(
      (total_agents, predictor.card_embedding.shape[1]), dtype=np.float32
    )
    self.running_sum = np.zeros(total_agents, dtype=np.float64)
    self.reset_window_metrics()

  def reset_window_metrics(self) -> None:
    self.delta_count = 0
    self.delta_sum = 0.0
    self.delta_abs_sum = 0.0
    self.delta_abs_max = 0.0
    self.residual_count = 0
    self.residual_sum = 0.0
    self.residual_abs_sum = 0.0
    self.telescope_count = 0
    self.telescope_abs_sum = 0.0
    self.telescope_abs_max = 0.0
    self.completed = 0
    self.truncated = 0
    self.unsynchronized = 0
    self.prediction_count = 0
    self.prediction_sum = 0.0
    self.prediction_square_sum = 0.0
    self.quartile_deltas = {name: 0 for name in (
      "q1_00_12",
      "q2_13_25",
      "q3_26_37",
      "q4_38_50",
    )}

  def _predict(
    self,
    *,
    gate_ids: np.ndarray,
    leader_ids: np.ndarray,
    seats: np.ndarray,
    cards: np.ndarray,
    counts: np.ndarray,
  ) -> np.ndarray:
    predictions = self.predictor.predict_potential(
      gate_ids=gate_ids,
      leader_ids=leader_ids,
      seats=seats,
      main_card_ids=cards,
      main_counts=counts,
    ).astype(np.float32, copy=False)
    self.prediction_count += int(predictions.size)
    self.prediction_sum += float(predictions.sum(dtype=np.float64))
    self.prediction_square_sum += float(
      np.square(predictions, dtype=np.float64).sum(dtype=np.float64)
    )
    return predictions

  def _predict_from_sums(
    self,
    *,
    gate_ids: np.ndarray,
    leader_ids: np.ndarray,
    seats: np.ndarray,
    card_embedding_sums: np.ndarray,
    counts: np.ndarray,
  ) -> np.ndarray:
    predictions = 2.0 * self.predictor.predict_probability_from_card_sums(
      gate_ids=gate_ids,
      leader_ids=leader_ids,
      seats=seats,
      card_embedding_sums=card_embedding_sums,
      main_counts=counts,
    ) - 1.0
    predictions = predictions.astype(np.float32, copy=False)
    self.prediction_count += int(predictions.size)
    self.prediction_sum += float(predictions.sum(dtype=np.float64))
    self.prediction_square_sum += float(
      np.square(predictions, dtype=np.float64).sum(dtype=np.float64)
    )
    return predictions

  def step(
    self,
    *,
    agent_ids: np.ndarray,
    episode_ids: np.ndarray,
    trainable: np.ndarray,
    done: np.ndarray,
    terminal: np.ndarray,
    modes: np.ndarray,
    gate_ids: np.ndarray,
    leader_ids: np.ndarray,
    main_card_ids: np.ndarray,
    main_counts: np.ndarray,
  ) -> np.ndarray:
    ids = np.asarray(agent_ids, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    trainable_rows = np.asarray(trainable, dtype=np.bool_).reshape(-1)
    done_rows = np.asarray(done, dtype=np.bool_).reshape(-1)
    terminal_rows = np.asarray(terminal, dtype=np.bool_).reshape(-1)
    mode_values = np.asarray(modes, dtype=np.int32).reshape(-1)
    gate_values = np.asarray(gate_ids, dtype=np.int16).reshape(-1)
    leader_values = np.asarray(leader_ids, dtype=np.int16).reshape(-1)
    cards = np.asarray(main_card_ids, dtype=np.int16)
    counts = np.asarray(main_counts, dtype=np.int16).reshape(-1)
    size = ids.size
    if any(
      values.size != size
      for values in (
        episodes,
        trainable_rows,
        done_rows,
        terminal_rows,
        mode_values,
        gate_values,
        leader_values,
        counts,
      )
    ) or cards.shape != (size, MAX_MAIN_DECK_SIZE):
      raise ValueError("prefix redistribution inputs have incompatible shapes")
    if np.any((ids < 0) | (ids >= self.total_agents)):
      raise ValueError("agent id is outside the redistributor state range")
    output = np.zeros(size, dtype=np.float32)

    done_ids = ids[done_rows]
    if done_ids.size:
      self.ignored_episode_ids[done_ids] = -1

    state_active = self.active[ids]
    same_episode = self.episode_ids[ids] == episodes
    finishing = done_rows & state_active & same_episode
    finish_positions = np.nonzero(finishing)[0]
    if finish_positions.size:
      finish_ids = ids[finish_positions]
      residuals = self.coefficient * (
        self.q0[finish_ids] - self.qprev[finish_ids]
      )
      output[finish_positions] = residuals
      telescope = self.running_sum[finish_ids] + residuals.astype(np.float64)
      self.residual_count += int(residuals.size)
      self.residual_sum += float(residuals.sum(dtype=np.float64))
      self.residual_abs_sum += float(np.abs(residuals).sum(dtype=np.float64))
      self.telescope_count += int(telescope.size)
      self.telescope_abs_sum += float(np.abs(telescope).sum(dtype=np.float64))
      self.telescope_abs_max = max(
        self.telescope_abs_max, float(np.abs(telescope).max(initial=0.0))
      )
      self.completed += int(terminal_rows[finish_positions].sum())
      self.truncated += int((~terminal_rows[finish_positions]).sum())
      self.active[finish_ids] = False
      self.episode_ids[finish_ids] = -1
      self.main_counts[finish_ids] = -1
      self.running_sum[finish_ids] = 0.0

    live = ~done_rows & trainable_rows
    live_positions = np.nonzero(live)[0]
    if not live_positions.size:
      return output
    live_ids = ids[live_positions]
    live_episodes = episodes[live_positions]
    new_episode = (~self.active[live_ids]) | (
      self.episode_ids[live_ids] != live_episodes
    )
    new_episode &= self.ignored_episode_ids[live_ids] != live_episodes
    new_positions = live_positions[new_episode]
    if new_positions.size:
      synchronized = (mode_values[new_positions] == 2) & (counts[new_positions] == 0)
      synchronized_positions = new_positions[synchronized]
      unsynchronized_positions = new_positions[~synchronized]
      self.unsynchronized += int(unsynchronized_positions.size)
      if unsynchronized_positions.size:
        self.ignored_episode_ids[ids[unsynchronized_positions]] = episodes[
          unsynchronized_positions
        ]
      if synchronized_positions.size:
        synchronized_ids = ids[synchronized_positions]
        seats = synchronized_ids % 2
        initial = self._predict_from_sums(
          gate_ids=gate_values[synchronized_positions],
          leader_ids=leader_values[synchronized_positions],
          seats=seats,
          card_embedding_sums=np.zeros(
            (synchronized_positions.size, self.predictor.card_embedding.shape[1]),
            dtype=np.float32,
          ),
          counts=np.zeros(synchronized_positions.size, dtype=np.int16),
        )
        self.active[synchronized_ids] = True
        self.episode_ids[synchronized_ids] = episodes[synchronized_positions]
        self.q0[synchronized_ids] = initial
        self.qprev[synchronized_ids] = initial
        self.main_counts[synchronized_ids] = 0
        self.gate_ids[synchronized_ids] = gate_values[synchronized_positions]
        self.leader_ids[synchronized_ids] = leader_values[synchronized_positions]
        self.card_embedding_sums[synchronized_ids] = 0.0
        self.running_sum[synchronized_ids] = 0.0

    state_active = self.active[ids]
    same_episode = self.episode_ids[ids] == episodes
    progressed = (
      ~done_rows
      & trainable_rows
      & state_active
      & same_episode
      & (counts > self.main_counts[ids])
      & (counts <= MAX_MAIN_DECK_SIZE)
    )
    progress_positions = np.nonzero(progressed)[0]
    if not progress_positions.size:
      return output
    progress_ids = ids[progress_positions]
    previous_counts = self.main_counts[progress_ids].astype(np.int64)
    next_counts = counts[progress_positions].astype(np.int64)
    increments = next_counts - previous_counts
    if np.any(increments <= 0):
      raise RuntimeError("draft prefix progression must add at least one card")
    single = increments == 1
    if bool(single.any()):
      single_ids = progress_ids[single]
      new_card_ids = cards[progress_positions[single], next_counts[single] - 1].astype(
        np.int64
      )
      if np.any(
        (new_card_ids < 0)
        | (new_card_ids >= self.predictor.card_embedding.shape[0])
      ):
        raise ValueError("draft prefix progression contains an unknown card id")
      self.card_embedding_sums[single_ids] += self.predictor.card_embedding[
        new_card_ids
      ]
    for local_index in np.nonzero(~single)[0].tolist():
      agent_id = int(progress_ids[local_index])
      start = int(previous_counts[local_index])
      stop = int(next_counts[local_index])
      added_cards = cards[progress_positions[local_index], start:stop].astype(np.int64)
      if added_cards.size != stop - start or np.any(
        (added_cards < 0) | (added_cards >= self.predictor.card_embedding.shape[0])
      ):
        raise ValueError("draft prefix progression contains an unknown card id")
      self.card_embedding_sums[agent_id] += self.predictor.card_embedding[
        added_cards
      ].sum(axis=0, dtype=np.float32)
    potentials = self._predict_from_sums(
      gate_ids=self.gate_ids[progress_ids],
      leader_ids=self.leader_ids[progress_ids],
      seats=progress_ids % 2,
      card_embedding_sums=self.card_embedding_sums[progress_ids],
      counts=counts[progress_positions],
    )
    deltas = self.coefficient * (potentials - self.qprev[progress_ids])
    output[progress_positions] = deltas
    self.qprev[progress_ids] = potentials
    self.main_counts[progress_ids] = counts[progress_positions]
    self.running_sum[progress_ids] += deltas.astype(np.float64)
    self.delta_count += int(deltas.size)
    self.delta_sum += float(deltas.sum(dtype=np.float64))
    self.delta_abs_sum += float(np.abs(deltas).sum(dtype=np.float64))
    self.delta_abs_max = max(
      self.delta_abs_max, float(np.abs(deltas).max(initial=0.0))
    )
    for count in counts[progress_positions].tolist():
      self.quartile_deltas[prefix_quartile(int(count))] += 1
    return output

  def window_metrics(self) -> dict[str, float]:
    prediction_mean = self.prediction_sum / max(self.prediction_count, 1)
    prediction_variance = max(
      0.0,
      self.prediction_square_sum / max(self.prediction_count, 1)
      - prediction_mean * prediction_mean,
    )
    metrics = {
      "delta_count": float(self.delta_count),
      "delta_mean": self.delta_sum / max(self.delta_count, 1),
      "delta_abs_mean": self.delta_abs_sum / max(self.delta_count, 1),
      "delta_abs_max": self.delta_abs_max,
      "residual_count": float(self.residual_count),
      "residual_mean": self.residual_sum / max(self.residual_count, 1),
      "residual_abs_mean": self.residual_abs_sum / max(self.residual_count, 1),
      "telescope_count": float(self.telescope_count),
      "telescope_abs_mean": self.telescope_abs_sum / max(self.telescope_count, 1),
      "telescope_abs_max": self.telescope_abs_max,
      "completed": float(self.completed),
      "truncated": float(self.truncated),
      "unsynchronized": float(self.unsynchronized),
      "prediction_count": float(self.prediction_count),
      "prediction_mean": prediction_mean,
      "prediction_std": math.sqrt(prediction_variance),
    }
    metrics.update(
      {f"{quartile}_deltas": float(value) for quartile, value in self.quartile_deltas.items()}
    )
    return metrics
