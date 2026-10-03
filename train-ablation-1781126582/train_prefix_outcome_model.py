#!/usr/bin/env python3
"""Train and validate the frozen Stage 2 draft-prefix outcome predictor."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from deck_building import build_deck_build_catalog
from draft_prefix_outcome import (
  ARTIFACT_SCHEMA_VERSION,
  FrozenDraftPrefixPredictor,
  file_sha256,
  prefix_quartile,
)
from training_deck_pool import load_training_deck_pool


def _stable_bucket(value: str, buckets: int) -> int:
  digest = hashlib.sha256(value.encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "little") % buckets


def _read_trajectories(paths: list[Path]) -> list[dict]:
  rows: list[dict] = []
  seen: set[str] = set()
  for path in paths:
    with path.open(encoding="utf-8") as handle:
      for line_number, line in enumerate(handle, start=1):
        if not line.strip():
          continue
        row = json.loads(line)
        trajectory_id = str(row.get("trajectory_id", ""))
        if not trajectory_id or trajectory_id in seen:
          raise ValueError(f"duplicate or missing trajectory id at {path}:{line_number}")
        seen.add(trajectory_id)
        cards = row.get("main_card_ids")
        if not isinstance(cards, list) or len(cards) != 50:
          raise ValueError(f"trajectory {trajectory_id} must contain exactly 50 cards")
        target = float(row.get("target", math.nan))
        if not 0.0 <= target <= 1.0:
          raise ValueError(f"trajectory {trajectory_id} has an invalid target")
        rows.append(row)
  if not rows:
    raise ValueError("prefix outcome dataset is empty")
  return rows


class PrefixOutcomeNet(nn.Module):
  def __init__(self, *, cards: int, gates: int, leaders: int):
    super().__init__()
    self.card_embedding = nn.Embedding(cards, 16)
    self.gate_embedding = nn.Embedding(gates, 8)
    self.leader_embedding = nn.Embedding(leaders, 8)
    self.seat_embedding = nn.Embedding(2, 4)
    self.fc1 = nn.Linear(40, 32)
    self.fc2 = nn.Linear(32, 1)
    nn.init.zeros_(self.card_embedding.weight)

  def forward(
    self,
    gates: torch.Tensor,
    leaders: torch.Tensor,
    seats: torch.Tensor,
    cards: torch.Tensor,
    counts: torch.Tensor,
  ) -> torch.Tensor:
    positions = torch.arange(50, device=cards.device)[None, :]
    selected = positions < counts[:, None]
    card_features = self.card_embedding(cards.clamp(min=0))
    card_features = (card_features * selected[:, :, None]).sum(dim=1) / 50.0
    fraction = counts.float() / 50.0
    position_features = torch.stack(
      (
        fraction,
        fraction.square(),
        torch.sqrt(fraction),
        (counts == 50).float(),
      ),
      dim=1,
    )
    features = torch.cat(
      (
        card_features,
        self.gate_embedding(gates),
        self.leader_embedding(leaders),
        self.seat_embedding(seats),
        position_features,
      ),
      dim=1,
    )
    return self.fc2(torch.relu(self.fc1(features))).flatten()


def _trajectory_arrays(
  rows: list[dict], gate_ids: np.ndarray, leader_ids: np.ndarray
) -> dict[str, np.ndarray]:
  gate_lookup = {int(value): index for index, value in enumerate(gate_ids.tolist())}
  leader_lookup = {
    int(value): index for index, value in enumerate(leader_ids.tolist())
  }
  return {
    "gate": np.asarray([gate_lookup[int(row["gate_id"])] for row in rows], dtype=np.int64),
    "leader": np.asarray(
      [leader_lookup[int(row["leader_id"])] for row in rows], dtype=np.int64
    ),
    "seat": np.asarray([int(row["seat"]) for row in rows], dtype=np.int64),
    "cards": np.asarray([row["main_card_ids"] for row in rows], dtype=np.int64),
    "target": np.asarray([float(row["target"]) for row in rows], dtype=np.float32),
  }


def _expand_prefixes(arrays: dict[str, np.ndarray], trajectory_indices: np.ndarray):
  prefix_counts = np.arange(51, dtype=np.int64)
  repeated = np.repeat(trajectory_indices, 51)
  return {
    "gate": arrays["gate"][repeated],
    "leader": arrays["leader"][repeated],
    "seat": arrays["seat"][repeated],
    "cards": arrays["cards"][repeated],
    "count": np.tile(prefix_counts, trajectory_indices.size),
    "target": arrays["target"][repeated],
    "trajectory": repeated,
  }


def _tensor_batch(expanded: dict[str, np.ndarray], indices: np.ndarray, device: str):
  return (
    torch.as_tensor(expanded["gate"][indices], device=device),
    torch.as_tensor(expanded["leader"][indices], device=device),
    torch.as_tensor(expanded["seat"][indices], device=device),
    torch.as_tensor(expanded["cards"][indices], device=device),
    torch.as_tensor(expanded["count"][indices], device=device),
    torch.as_tensor(expanded["target"][indices], device=device),
  )


def _predict(
  model: PrefixOutcomeNet,
  expanded: dict[str, np.ndarray],
  *,
  device: str,
  batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
  predictions: list[np.ndarray] = []
  logits_out: list[np.ndarray] = []
  model.eval()
  with torch.inference_mode():
    for start in range(0, expanded["target"].size, batch_size):
      indices = np.arange(start, min(start + batch_size, expanded["target"].size))
      gates, leaders, seats, cards, counts, _ = _tensor_batch(
        expanded, indices, device
      )
      logits = model(gates, leaders, seats, cards, counts)
      logits_out.append(logits.float().cpu().numpy())
      predictions.append(torch.sigmoid(logits).float().cpu().numpy())
  return np.concatenate(predictions), np.concatenate(logits_out)


def _fit_calibration(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
  logits_t = torch.as_tensor(logits, dtype=torch.float64)
  targets_t = torch.as_tensor(targets, dtype=torch.float64)
  log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
  bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
  optimizer = torch.optim.LBFGS(
    [log_temperature, bias], lr=0.25, max_iter=80, line_search_fn="strong_wolfe"
  )

  def closure():
    optimizer.zero_grad()
    calibrated = logits_t / log_temperature.exp().clamp(min=0.05, max=20.0) + bias
    loss = nn.functional.binary_cross_entropy_with_logits(calibrated, targets_t)
    loss.backward()
    return loss

  optimizer.step(closure)
  temperature = float(log_temperature.detach().exp().clamp(min=0.05, max=20.0))
  return temperature, float(bias.detach())


def _apply_calibration(model: PrefixOutcomeNet, temperature: float, bias: float) -> None:
  with torch.no_grad():
    model.fc2.weight.div_(temperature)
    model.fc2.bias.div_(temperature).add_(bias)


def _train_model(
  arrays: dict[str, np.ndarray],
  rows: list[dict],
  train_trajectories: np.ndarray,
  *,
  cards: int,
  gates: int,
  leaders: int,
  seed: int,
  device: str,
  epochs: int,
  batch_size: int,
  pairwise_coefficient: float,
) -> tuple[PrefixOutcomeNet, dict[str, float]]:
  internal_validation = np.asarray(
    [
      index
      for index in train_trajectories.tolist()
      if _stable_bucket(
        f"internal:{rows[index].get('source_trajectory_id', rows[index]['trajectory_id'])}",
        10,
      ) == 0
    ],
    dtype=np.int64,
  )
  internal_set = set(internal_validation.tolist())
  fitting = np.asarray(
    [index for index in train_trajectories.tolist() if index not in internal_set],
    dtype=np.int64,
  )
  if fitting.size < 64 or internal_validation.size < 16:
    raise ValueError("training split is too small for grouped internal validation")
  train_rows = _expand_prefixes(arrays, fitting)
  validation_rows = _expand_prefixes(arrays, internal_validation)

  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  model = PrefixOutcomeNet(cards=cards, gates=gates, leaders=leaders).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
  generator = np.random.default_rng(seed)
  fitting_set = set(fitting.tolist())
  pair_groups: dict[str, list[int]] = defaultdict(list)
  for index in fitting.tolist():
    source_id = rows[index].get("source_trajectory_id")
    if source_id is not None:
      pair_groups[str(source_id)].append(index)
  pairs = np.asarray(
    [values for values in pair_groups.values() if len(values) == 2], dtype=np.int64
  )
  if pairwise_coefficient > 0.0 and not pairs.size:
    raise ValueError("pairwise predictor training requires paired dataset variants")
  if pairs.size and any(
    int(index) not in fitting_set for index in pairs.reshape(-1).tolist()
  ):
    raise RuntimeError("pairwise training crossed the grouped calibration split")
  best_loss = math.inf
  best_state: dict[str, torch.Tensor] | None = None
  stale = 0
  epochs_run = 0
  for epoch in range(epochs):
    model.train()
    order = generator.permutation(train_rows["target"].size)
    for start in range(0, order.size, batch_size):
      batch_indices = order[start:start + batch_size]
      gates_t, leaders_t, seats_t, cards_t, counts_t, targets_t = _tensor_batch(
        train_rows, batch_indices, device
      )
      optimizer.zero_grad(set_to_none=True)
      logits = model(gates_t, leaders_t, seats_t, cards_t, counts_t)
      loss = nn.functional.binary_cross_entropy_with_logits(logits, targets_t)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
    if pairwise_coefficient > 0.0:
      pair_order = pairs[generator.permutation(pairs.shape[0])]
      for start in range(0, pair_order.shape[0], max(1, batch_size // 2)):
        pair_batch = pair_order[start:start + max(1, batch_size // 2)]
        left = pair_batch[:, 0]
        right = pair_batch[:, 1]
        counts = torch.full((left.size,), 50, device=device, dtype=torch.long)
        left_logits = model(
          torch.as_tensor(arrays["gate"][left], device=device),
          torch.as_tensor(arrays["leader"][left], device=device),
          torch.as_tensor(arrays["seat"][left], device=device),
          torch.as_tensor(arrays["cards"][left], device=device),
          counts,
        )
        right_logits = model(
          torch.as_tensor(arrays["gate"][right], device=device),
          torch.as_tensor(arrays["leader"][right], device=device),
          torch.as_tensor(arrays["seat"][right], device=device),
          torch.as_tensor(arrays["cards"][right], device=device),
          counts,
        )
        target_difference = torch.as_tensor(
          arrays["target"][right] - arrays["target"][left], device=device
        )
        predicted_difference = torch.sigmoid(right_logits) - torch.sigmoid(left_logits)
        pair_weights = torch.where(target_difference != 0.0, 5.0, 1.0)
        pair_loss = (
          pairwise_coefficient
          * pair_weights
          * (predicted_difference - target_difference).square()
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        pair_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    validation_predictions, validation_logits = _predict(
      model, validation_rows, device=device, batch_size=batch_size
    )
    clipped = np.clip(validation_predictions, 1e-7, 1.0 - 1e-7)
    validation_loss = float(
      -np.mean(
        validation_rows["target"] * np.log(clipped)
        + (1.0 - validation_rows["target"]) * np.log(1.0 - clipped)
      )
    )
    epochs_run = epoch + 1
    if validation_loss < best_loss - 1e-5:
      best_loss = validation_loss
      best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
      }
      stale = 0
    else:
      stale += 1
      if stale >= 8:
        break
  if best_state is None:
    raise RuntimeError("prefix predictor training did not produce a checkpoint")
  model.load_state_dict(best_state)
  _, validation_logits = _predict(
    model, validation_rows, device=device, batch_size=batch_size
  )
  temperature, bias = _fit_calibration(
    validation_logits, validation_rows["target"]
  )
  _apply_calibration(model, temperature, bias)
  return model, {
    "epochs": float(epochs_run),
    "best_internal_bce": best_loss,
    "calibration_temperature": temperature,
    "calibration_bias": bias,
    "fit_trajectories": float(fitting.size),
    "calibration_trajectories": float(internal_validation.size),
    "paired_training_coordinates": float(pairs.shape[0] if pairs.ndim == 2 else 0),
    "pairwise_coefficient": float(pairwise_coefficient),
  }


def _auc(targets: np.ndarray, predictions: np.ndarray) -> float:
  decisive = np.logical_or(targets == 0.0, targets == 1.0)
  y = targets[decisive].astype(np.int8)
  p = predictions[decisive]
  positives = int(y.sum())
  negatives = int(y.size - positives)
  if positives == 0 or negatives == 0:
    return math.nan
  order = np.argsort(p, kind="stable")
  ranks = np.empty(order.size, dtype=np.float64)
  start = 0
  while start < order.size:
    end = start + 1
    while end < order.size and p[order[end]] == p[order[start]]:
      end += 1
    ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
    start = end
  rank_sum = float(ranks[y == 1].sum())
  return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
  clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
  bce = float(
    -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
  )
  ece = 0.0
  for lower in np.linspace(0.0, 0.9, 10):
    upper = lower + 0.1
    selected = (predictions >= lower) & (
      predictions <= upper if upper >= 1.0 else predictions < upper
    )
    if selected.any():
      ece += float(selected.mean()) * abs(
        float(predictions[selected].mean()) - float(targets[selected].mean())
      )
  return {
    "rows": float(targets.size),
    "target_mean": float(targets.mean()),
    "prediction_mean": float(predictions.mean()),
    "bce": bce,
    "brier": float(np.mean((predictions - targets) ** 2)),
    "ece_10": ece,
    "auc_decisive": _auc(targets, predictions),
  }


def _stratified_metrics(
  expanded: dict[str, np.ndarray],
  predictions: np.ndarray,
  rows: list[dict],
) -> dict[str, object]:
  result: dict[str, object] = {"overall": _metrics(expanded["target"], predictions)}
  trajectory_indices = expanded["trajectory"]
  group_values = {
    "gate": np.asarray([str(rows[i]["gate_id"]) for i in trajectory_indices]),
    "leader": np.asarray([str(rows[i]["leader_id"]) for i in trajectory_indices]),
    "seat": np.asarray([str(rows[i]["seat"]) for i in trajectory_indices]),
    "quartile": np.asarray([prefix_quartile(value) for value in expanded["count"]]),
  }
  for group_name, values in group_values.items():
    result[f"by_{group_name}"] = {
      value: _metrics(expanded["target"][values == value], predictions[values == value])
      for value in sorted(set(values.tolist()))
    }
  if predictions.size % 51:
    raise ValueError("expanded prefix predictions are not trajectory-aligned")
  prediction_matrix = predictions.reshape(-1, 51)
  trajectory_targets = expanded["target"].reshape(-1, 51)[:, 0]
  incremental: dict[str, dict[str, float]] = {}
  for count in (12, 25, 37, 50):
    deltas = prediction_matrix[:, count] - prediction_matrix[:, 0]
    target_std = float(trajectory_targets.std())
    delta_std = float(deltas.std())
    correlation = (
      float(np.corrcoef(trajectory_targets, deltas)[0, 1])
      if target_std > 0.0 and delta_std > 0.0
      else math.nan
    )
    incremental[f"pick_{count:02d}"] = {
      "trajectories": float(deltas.size),
      "delta_mean": float(deltas.mean()),
      "delta_std": delta_std,
      "delta_abs_mean": float(np.abs(deltas).mean()),
      "outcome_correlation": correlation,
      "auc_decisive": _auc(trajectory_targets, deltas),
    }
  result["incremental_from_empty"] = incremental
  local_trajectories = expanded["trajectory"].reshape(-1, 51)[:, 0]
  local_prediction_matrix = predictions.reshape(-1, 51)
  local_by_source: dict[str, list[int]] = defaultdict(list)
  for local_index, trajectory_index in enumerate(local_trajectories.tolist()):
    source_id = rows[trajectory_index].get("source_trajectory_id")
    if source_id is not None:
      local_by_source[str(source_id)].append(local_index)
  paired = [values for values in local_by_source.values() if len(values) == 2]
  if paired:
    left = np.asarray([values[0] for values in paired], dtype=np.int64)
    right = np.asarray([values[1] for values in paired], dtype=np.int64)
    pair_targets = trajectory_targets[right] - trajectory_targets[left]
    pair_predictions = (
      local_prediction_matrix[right, 50] - local_prediction_matrix[left, 50]
    )
    changed = pair_targets != 0.0
    result["paired_counterfactual"] = {
      "pairs": float(len(paired)),
      "changed_outcome_pairs": float(changed.sum()),
      "target_difference_mean": float(pair_targets.mean()),
      "prediction_difference_mean": float(pair_predictions.mean()),
      "prediction_difference_std": float(pair_predictions.std()),
      "mse": float(np.mean((pair_predictions - pair_targets) ** 2)),
      "changed_sign_accuracy": (
        float(
          (np.sign(pair_predictions[changed]) == np.sign(pair_targets[changed])).mean()
        )
        if changed.any()
        else math.nan
      ),
      "difference_correlation": (
        float(np.corrcoef(pair_targets, pair_predictions)[0, 1])
        if pair_targets.std() > 0.0 and pair_predictions.std() > 0.0
        else math.nan
      ),
    }
  return result


def _export_model(
  model: PrefixOutcomeNet,
  path: Path,
  *,
  gate_ids: np.ndarray,
  leader_ids: np.ndarray,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  state = model.to("cpu").state_dict()
  np.savez(
    path,
    schema_version=np.asarray([ARTIFACT_SCHEMA_VERSION], dtype=np.int32),
    gate_ids=gate_ids.astype(np.int16),
    leader_ids=leader_ids.astype(np.int16),
    card_embedding=state["card_embedding.weight"].numpy().astype(np.float32),
    gate_embedding=state["gate_embedding.weight"].numpy().astype(np.float32),
    leader_embedding=state["leader_embedding.weight"].numpy().astype(np.float32),
    seat_embedding=state["seat_embedding.weight"].numpy().astype(np.float32),
    fc1_weight=state["fc1.weight"].numpy().astype(np.float32),
    fc1_bias=state["fc1.bias"].numpy().astype(np.float32),
    fc2_weight=state["fc2.weight"].numpy().astype(np.float32),
    fc2_bias=state["fc2.bias"].numpy().astype(np.float32),
  )


def _markdown(report: dict) -> str:
  lines = [
    "# Frozen Draft-Prefix Outcome Predictor",
    "",
    f"Dataset: **{report['dataset']['trajectories']} complete trajectories**, "
    f"{report['dataset']['prefix_rows']} exact prefixes.",
    "",
    f"Frozen artifact: `{report['artifact']['path']}`",
    "",
    f"SHA-256: `{report['artifact']['sha256']}`",
    "",
    "| Holdout axis | Trajectories | Brier | BCE | ECE | Decisive AUC |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  for axis, values in report["validation"].items():
    overall = values["metrics"]["overall"]
    lines.append(
      f"| {axis} | {values['holdout_trajectories']} | {overall['brier']:.4f} | "
      f"{overall['bce']:.4f} | {overall['ece_10']:.4f} | "
      f"{overall['auc_decisive']:.4f} |"
    )
  lines.extend(
    [
      "",
      f"Validation completeness: **{'PASS' if report['validation_complete'] else 'FAIL'}**.",
      "",
      "The artifact is frozen during policy training. Prefix rewards use successive "
      "potential differences and a terminal residual; they do not add net episode return.",
      "",
    ]
  )
  return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, action="append", required=True)
  parser.add_argument("--heldout-generation", required=True)
  parser.add_argument("--heldout-lineage", required=True)
  parser.add_argument("--artifact", type=Path, required=True)
  parser.add_argument("--report-json", type=Path, required=True)
  parser.add_argument("--report-md", type=Path, required=True)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--seed", type=int, default=420054)
  parser.add_argument("--epochs", type=int, default=60)
  parser.add_argument("--batch-size", type=int, default=8192)
  parser.add_argument("--pairwise-coef", type=float, default=0.0)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.epochs < 1 or args.batch_size < 1:
    raise ValueError("epochs and batch_size must be positive")
  if args.pairwise_coef < 0.0:
    raise ValueError("pairwise coefficient must be nonnegative")
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  rows = _read_trajectories(args.input)
  gate_ids = np.asarray(sorted({int(row["gate_id"]) for row in rows}), dtype=np.int16)
  leader_ids = np.asarray(
    sorted({int(row["leader_id"]) for row in rows}), dtype=np.int16
  )
  max_observed_card_id = max(
    max(int(card_id) for card_id in row["main_card_ids"]) for row in rows
  )
  catalog = build_deck_build_catalog(load_training_deck_pool())
  max_card_id = max(int(card_id) for card_id in catalog.records_by_def_id)
  if max_observed_card_id > max_card_id:
    raise ValueError("prefix dataset contains a card outside the current catalog")
  arrays = _trajectory_arrays(rows, gate_ids, leader_ids)
  all_indices = np.arange(len(rows), dtype=np.int64)

  holdout_masks = {
    "seed": np.asarray(
      [_stable_bucket(f"seed:{int(row['seed'])}", 5) == 0 for row in rows],
      dtype=np.bool_,
    ),
    "opponent_lineage": np.asarray(
      [str(row["opponent_lineage"]) == args.heldout_lineage for row in rows],
      dtype=np.bool_,
    ),
    "policy_generation": np.asarray(
      [str(row["policy_generation"]) == args.heldout_generation for row in rows],
      dtype=np.bool_,
    ),
  }
  validation: dict[str, object] = {}
  for axis_index, (axis, holdout_mask) in enumerate(holdout_masks.items()):
    train_indices = all_indices[~holdout_mask]
    holdout_indices = all_indices[holdout_mask]
    if train_indices.size < 128 or holdout_indices.size < 64:
      raise ValueError(
        f"{axis} split is too small: train={train_indices.size}, "
        f"holdout={holdout_indices.size}"
      )
    model, training = _train_model(
      arrays,
      rows,
      train_indices,
      cards=max_card_id + 1,
      gates=gate_ids.size,
      leaders=leader_ids.size,
      seed=args.seed + axis_index,
      device=args.device,
      epochs=args.epochs,
      batch_size=args.batch_size,
      pairwise_coefficient=args.pairwise_coef,
    )
    expanded = _expand_prefixes(arrays, holdout_indices)
    predictions, _ = _predict(
      model, expanded, device=args.device, batch_size=args.batch_size
    )
    validation[axis] = {
      "train_trajectories": int(train_indices.size),
      "holdout_trajectories": int(holdout_indices.size),
      "training": training,
      "metrics": _stratified_metrics(expanded, predictions, rows),
    }

  final_model, final_training = _train_model(
    arrays,
    rows,
    all_indices,
    cards=max_card_id + 1,
    gates=gate_ids.size,
    leaders=leader_ids.size,
    seed=args.seed + 100,
    device=args.device,
    epochs=args.epochs,
    batch_size=args.batch_size,
    pairwise_coefficient=args.pairwise_coef,
  )
  _export_model(final_model, args.artifact, gate_ids=gate_ids, leader_ids=leader_ids)

  predictor = FrozenDraftPrefixPredictor(args.artifact)
  parity_indices = all_indices[: min(128, all_indices.size)]
  parity_rows = _expand_prefixes(arrays, parity_indices)
  torch_predictions, _ = _predict(
    final_model, parity_rows, device="cpu", batch_size=args.batch_size
  )
  numpy_predictions = predictor.predict_probability(
    gate_ids=gate_ids[parity_rows["gate"]],
    leader_ids=leader_ids[parity_rows["leader"]],
    seats=parity_rows["seat"],
    main_card_ids=parity_rows["cards"],
    main_counts=parity_rows["count"],
  )
  parity_max_abs = float(np.max(np.abs(torch_predictions - numpy_predictions)))
  if parity_max_abs > 2e-5:
    raise RuntimeError(f"NumPy artifact parity failed: max abs diff {parity_max_abs}")

  dataset_digest = hashlib.sha256()
  for path in sorted(args.input):
    dataset_digest.update(path.resolve().as_posix().encode("utf-8"))
    dataset_digest.update(file_sha256(path).encode("ascii"))
  required_groups = {"gate": 8, "leader": 8, "seat": 2, "quartile": 4}
  validation_complete = True
  for axis_values in validation.values():
    groups = axis_values["metrics"]
    for group, expected in required_groups.items():
      validation_complete &= len(groups[f"by_{group}"]) == expected
    overall = groups["overall"]
    validation_complete &= all(
      math.isfinite(float(overall[key]))
      for key in ("brier", "bce", "ece_10", "auc_decisive")
    )

  report = {
    "schema_version": 1,
    "dataset": {
      "inputs": [str(path.resolve()) for path in args.input],
      "sha256": dataset_digest.hexdigest(),
      "trajectories": len(rows),
      "prefix_rows": len(rows) * 51,
      "policy_generations": sorted({str(row["policy_generation"]) for row in rows}),
      "opponent_lineages": sorted({str(row["opponent_lineage"]) for row in rows}),
      "gates": gate_ids.astype(int).tolist(),
      "leaders": leader_ids.astype(int).tolist(),
      "unique_cards": len(
        {int(card_id) for row in rows for card_id in row["main_card_ids"]}
      ),
      "max_observed_card_id": max_observed_card_id,
      "max_card_id": max_card_id,
    },
    "split_contract": {
      "heldout_generation": args.heldout_generation,
      "heldout_lineage": args.heldout_lineage,
      "seed_rule": "sha256('seed:' + seed) modulo 5 equals 0",
      "prefixes_from_same_trajectory_never_cross_a_split": True,
    },
    "validation": validation,
    "validation_complete": bool(validation_complete),
    "final_training": final_training,
    "artifact": {
      "path": str(args.artifact.resolve()),
      "sha256": predictor.sha256,
      "schema_version": ARTIFACT_SCHEMA_VERSION,
      "numpy_torch_parity_max_abs": parity_max_abs,
      "parameters": int(sum(parameter.numel() for parameter in final_model.parameters())),
    },
  }
  args.report_json.parent.mkdir(parents=True, exist_ok=True)
  args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  args.report_md.parent.mkdir(parents=True, exist_ok=True)
  args.report_md.write_text(_markdown(report), encoding="utf-8")
  print(
    f"[prefix-outcome-train] trajectories={len(rows)} "
    f"artifact={args.artifact} sha256={predictor.sha256} "
    f"validation_complete={validation_complete}",
    flush=True,
  )


if __name__ == "__main__":
  main()
