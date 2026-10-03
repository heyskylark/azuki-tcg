from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
  import psutil
except ImportError:  # pragma: no cover - optional dependency in some runtimes
  psutil = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "python" / "config" / "azuki_speed_3090_parallel.ini"
DEFAULT_VARIANT_FILE = REPO_ROOT / "python" / "config" / "sps_autoresearch_3090.json"

EPOCH_LOG_PATTERN = re.compile(r"^\[epoch\s+(?P<epoch>\d+)\]\s+(?P<payload>\{.*\})\s*$")
ENV_PROFILE_PATTERN = re.compile(
  r"^\[EnvProfile\]\s+steps=(?P<steps>\d+)\s+avg_step_us=(?P<avg_step_us>[-+0-9.]+)"
  r"\s+avg_tick_us=(?P<avg_tick_us>[-+0-9.]+)\s+avg_refresh_us=(?P<avg_refresh_us>[-+0-9.]+)"
  r"\s+avg_auto_ticks=(?P<avg_auto_ticks>[-+0-9.]+)\s+tick_share=(?P<tick_share>[-+0-9.]+)"
  r"\s+refresh_share=(?P<refresh_share>[-+0-9.]+)\s*$"
)


CONTROL_OVERRIDES: dict[str, Any] = {
  "vec.num_envs": 120,
  "vec.num_workers": 4,
  "vec.batch_size": 120,
  "vec.zero_copy": True,
  "env.direct_parallel": True,
  "train.batch_size": 3840,
  "train.minibatch_size": 960,
  "train.max_minibatch_size": 960,
  "train.precision": "bfloat16",
  "train.compile": False,
  "league.enable": False,
  "wandb": False,
  "neptune": False,
}


SAFE_VARIANTS: list[dict[str, Any]] = [
  {
    "label": "control",
    "description": "Stable local 3090 control branch from the v2 generalization TODO.",
  },
  {
    "label": "envs_96",
    "description": "Reduce env concurrency while keeping the learner batch shape fixed.",
    "overrides": {
      "vec.num_envs": 96,
      "vec.batch_size": 96,
    },
  },
  {
    "label": "workers_2",
    "description": "Reduce worker fanout to test whether the run is coordination-bound.",
    "overrides": {
      "vec.num_workers": 2,
    },
  },
  {
    "label": "workers_6",
    "description": "Increase worker fanout to test whether env throughput improves.",
    "overrides": {
      "vec.num_workers": 6,
    },
  },
  {
    "label": "zero_copy_off",
    "description": "Disable zero-copy to measure copy overhead sensitivity.",
    "overrides": {
      "vec.zero_copy": False,
    },
  },
  {
    "label": "direct_parallel_off",
    "description": "Turn off direct-parallel env wrapping to isolate env wrapper overhead.",
    "overrides": {
      "env.direct_parallel": False,
    },
  },
  {
    "label": "precision_float32",
    "description": "Use float32 to measure precision-driven learner throughput changes.",
    "overrides": {
      "train.precision": "float32",
    },
  },
  {
    "label": "compile_on",
    "description": "Enable torch.compile to test whether learner kernels amortize on short probes.",
    "overrides": {
      "train.compile": True,
    },
  },
  {
    "label": "league_on",
    "description": "Measure the end-to-end league overhead versus pure self-play.",
    "overrides": {
      "league.enable": True,
    },
  },
]


ADAPTIVE_SEARCH_SPACE: dict[str, list[Any]] = {
  "vec.num_envs": [72, 96, 120, 144],
  "vec.num_workers": [2, 4, 6, 8],
  "vec.zero_copy": [True, False],
  "env.direct_parallel": [True, False],
  "policy.legal_action_scorer_use_references": [True, False],
  "train.precision": ["bfloat16", "float32"],
  "train.compile": [False, True],
  "league.enable": [False, True],
}


BOUND_PRIORITY: dict[str, list[str]] = {
  "env_bound": [
    "vec.num_workers",
    "vec.num_envs",
    "env.direct_parallel",
    "vec.zero_copy",
    "train.compile",
    "train.precision",
    "league.enable",
  ],
  "copy_bound": [
    "vec.zero_copy",
    "env.direct_parallel",
    "train.precision",
    "train.compile",
    "vec.num_workers",
    "vec.num_envs",
    "league.enable",
  ],
  "learner_bound": [
    "train.compile",
    "train.precision",
    "policy.legal_action_scorer_use_references",
    "vec.num_envs",
    "vec.num_workers",
    "vec.zero_copy",
    "env.direct_parallel",
    "league.enable",
  ],
  "misc_bound": [
    "train.compile",
    "train.precision",
    "vec.num_workers",
    "vec.num_envs",
    "vec.zero_copy",
    "env.direct_parallel",
    "league.enable",
  ],
}


DEFAULT_TARGET_METRIC = "tail_mean.SPS"
DEFAULT_TARGET_MODE = "at_least"


@dataclass(frozen=True)
class ProbeVariant:
  label: str
  description: str
  overrides: dict[str, Any]
  env: dict[str, str]


def parse_epoch_log_line(line: str) -> dict[str, Any] | None:
  match = EPOCH_LOG_PATTERN.match(line.strip())
  if match is None:
    return None

  # NumPy 2.x reprs scalar metrics as np.float64(...), which is not a Python
  # literal. Training logs contain only scalar wrappers here, so unwrap them
  # before using the deliberately strict literal parser.
  raw_payload = re.sub(
    r"np\.(?:float16|float32|float64)\(([^()]*)\)", r"\1",
    match.group("payload"),
  )
  payload = ast.literal_eval(raw_payload)
  if not isinstance(payload, dict):
    return None

  parsed = {"epoch_index": int(match.group("epoch"))}
  parsed.update(payload)
  return parsed


def parse_env_profile_line(line: str) -> dict[str, float] | None:
  match = ENV_PROFILE_PATTERN.match(line.strip())
  if match is None:
    return None

  parsed: dict[str, float] = {}
  for key, value in match.groupdict().items():
    if key == "steps":
      parsed[key] = float(int(value))
    else:
      parsed[key] = float(value)
  return parsed


def evaluate_action_sanity(
  metrics: dict[str, float],
  *,
  truncation_threshold: float = 1e-6,
  noop_min: float = 0.15,
  noop_max: float = 0.55,
  active_mass_min: float = 0.35,
) -> dict[str, Any]:
  reasons: list[str] = []
  seats: list[dict[str, float]] = []
  availability: list[bool] = []

  for seat in (0, 1):
    truncation_key = f"environment/{seat}/azk_zero_legal_action_truncation"
    noop_key = f"environment/{seat}/azk_noop_selected_rate"
    attack_key = f"environment/{seat}/azk_attack_selected_rate"
    play_key = f"environment/{seat}/azk_play_selected_rate"
    ability_key = f"environment/{seat}/azk_ability_selected_rate"
    target_key = f"environment/{seat}/azk_target_selected_rate"
    keys = (truncation_key, noop_key, attack_key, play_key, ability_key, target_key)
    available = any(key in metrics for key in keys)
    availability.append(available)

    truncation = float(metrics.get(truncation_key, 0.0))
    noop = float(metrics.get(noop_key, 0.0))
    attack = float(metrics.get(attack_key, 0.0))
    play = float(metrics.get(play_key, 0.0))
    ability = float(metrics.get(ability_key, 0.0))
    target = float(metrics.get(target_key, 0.0))
    active_mass = attack + play + ability + target
    seats.append(
      {
        "available": 1.0 if available else 0.0,
        "seat": float(seat),
        "truncation": truncation,
        "noop": noop,
        "attack": attack,
        "play": play,
        "ability": ability,
        "target": target,
        "active_mass": active_mass,
      }
    )

    if not available:
      continue
    if truncation > truncation_threshold:
      reasons.append(f"seat {seat} truncation {truncation:.6f} exceeded {truncation_threshold:.6f}")
    if noop < noop_min or noop > noop_max:
      reasons.append(f"seat {seat} noop {noop:.4f} fell outside [{noop_min:.2f}, {noop_max:.2f}]")
    if active_mass < active_mass_min:
      reasons.append(f"seat {seat} non-noop mass {active_mass:.4f} fell below {active_mass_min:.2f}")

  available = any(availability)
  passed: bool | None
  if not available:
    passed = None
  else:
    passed = len(reasons) == 0

  return {
    "available": available,
    "pass": passed,
    "reasons": reasons,
    "seats": seats,
    "config": {
      "truncation_threshold": truncation_threshold,
      "noop_min": noop_min,
      "noop_max": noop_max,
      "active_mass_min": active_mass_min,
    },
  }


def _safe_mean(values: list[float]) -> float | None:
  finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
  if not finite:
    return None
  return sum(finite) / len(finite)


def _safe_max(values: list[float]) -> float | None:
  finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
  if not finite:
    return None
  return max(finite)


def _mean_numeric_records(records: list[dict[str, Any]]) -> dict[str, float]:
  collected: dict[str, list[float]] = {}
  for record in records:
    for key, value in record.items():
      if isinstance(value, bool):
        continue
      if isinstance(value, (int, float)) and math.isfinite(float(value)):
        collected.setdefault(key, []).append(float(value))
  return {
    key: sum(values) / len(values)
    for key, values in collected.items()
    if values
  }


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_safe(inner) for key, inner in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(inner) for inner in value]
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, float):
    if math.isnan(value) or math.isinf(value):
      return None
    return value
  return value


def _build_default_variant_spec(total_timesteps: int) -> dict[str, Any]:
  control = dict(CONTROL_OVERRIDES)
  control["train.total_timesteps"] = int(total_timesteps)
  return {
    "description": "Safe first-pass local 3090 SPS autoresearch sweep.",
    "control_overrides": control,
    "variants": SAFE_VARIANTS,
  }


def _load_variant_spec(path: Path | None, total_timesteps: int) -> dict[str, Any]:
  if path is None:
    return _build_default_variant_spec(total_timesteps)

  payload = json.loads(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"Variant spec must be a JSON object: {path}")

  control = payload.get("control_overrides")
  if not isinstance(control, dict):
    raise ValueError(f"Variant spec is missing control_overrides: {path}")
  control = dict(control)
  control.setdefault("train.total_timesteps", int(total_timesteps))

  variants = payload.get("variants")
  if not isinstance(variants, list) or not variants:
    raise ValueError(f"Variant spec must define a non-empty variants array: {path}")

  return {
    "description": payload.get("description"),
    "control_overrides": control,
    "variants": variants,
  }


def _materialize_variants(spec: dict[str, Any]) -> tuple[dict[str, Any], list[ProbeVariant]]:
  control_overrides = spec["control_overrides"]
  variants: list[ProbeVariant] = []

  for index, raw in enumerate(spec["variants"]):
    if not isinstance(raw, dict):
      raise ValueError(f"Variant #{index} must be an object")
    label = raw.get("label")
    if not isinstance(label, str) or not label:
      raise ValueError(f"Variant #{index} is missing a non-empty label")
    description = raw.get("description")
    if not isinstance(description, str) or not description:
      description = label
    overrides = raw.get("overrides", {})
    env = raw.get("env", {})
    if not isinstance(overrides, dict):
      raise ValueError(f"Variant '{label}' overrides must be an object")
    if not isinstance(env, dict):
      raise ValueError(f"Variant '{label}' env must be an object")
    variants.append(
      ProbeVariant(
        label=label,
        description=description,
        overrides=dict(overrides),
        env={str(key): str(value) for key, value in env.items()},
      )
    )

  return control_overrides, variants


def _default_python_executable() -> Path:
  codex_python = REPO_ROOT / "python" / ".venv-codex" / "bin" / "python"
  if codex_python.exists():
    return codex_python
  return Path(sys.executable)


def _default_output_dir() -> Path:
  timestamp = int(time.time())
  return REPO_ROOT / "experiments" / f"sps_autoresearch_{timestamp}"


def _path_with_pythonpath(existing: str | None) -> str:
  required = [str(REPO_ROOT / "python" / "src"), str(REPO_ROOT / "build" / "python" / "src")]
  current = [] if not existing else [entry for entry in existing.split(os.pathsep) if entry]
  ordered: list[str] = []
  for entry in required + current:
    if entry not in ordered:
      ordered.append(entry)
  return os.pathsep.join(ordered)


def _value_to_cli(value: Any) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  return str(value)


def _normalize_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
  normalized = dict(overrides)
  env_count = normalized.get("vec.num_envs")
  batch_size = normalized.get("vec.batch_size")
  if env_count is not None and batch_size is None:
    normalized["vec.batch_size"] = env_count
  return normalized


def _signature_for_overrides(overrides: dict[str, Any]) -> str:
  normalized = _normalize_overrides(overrides)
  return json.dumps(_json_safe(normalized), sort_keys=True, separators=(",", ":"))


def _merged_overrides(control: dict[str, Any], variant: ProbeVariant) -> dict[str, Any]:
  merged = dict(control)
  merged.update(variant.overrides)
  return _normalize_overrides(merged)


def _build_train_command(
  python_executable: Path,
  config_path: Path,
  overrides: dict[str, Any],
) -> list[str]:
  command = [
    str(python_executable),
    "python/src/train.py",
    "--config",
    str(config_path),
  ]
  for key, value in overrides.items():
    command.append(f"--{key}")
    command.append(_value_to_cli(value))
  return command


def _extract_metric_value(payload: dict[str, Any], metric_path: str) -> float | None:
  current: Any = payload
  for part in metric_path.split("."):
    if not isinstance(current, dict) or part not in current:
      return None
    current = current[part]
  if isinstance(current, bool):
    return None
  if not isinstance(current, (int, float)):
    return None
  numeric = float(current)
  if not math.isfinite(numeric):
    return None
  return numeric


def _result_guardrail_pass(result: dict[str, Any]) -> bool:
  if result.get("status") != "ok":
    return False
  return bool(result.get("action_sanity", {}).get("pass") is True)


def _metric_satisfies_target(value: float | None, *, target_value: float, target_mode: str) -> bool:
  if value is None:
    return False
  if target_mode == "at_least":
    return value >= target_value
  if target_mode == "at_most":
    return value <= target_value
  raise ValueError(f"Unsupported target_mode '{target_mode}'")


def _result_meets_target(
  result: dict[str, Any],
  *,
  target_metric: str,
  target_value: float,
  target_mode: str,
  require_guardrails: bool,
) -> bool:
  if result.get("status") != "ok":
    return False
  if require_guardrails and not _result_guardrail_pass(result):
    return False
  value = _extract_metric_value(result, target_metric)
  return _metric_satisfies_target(value, target_value=target_value, target_mode=target_mode)


def _objective_sort_key(
  result: dict[str, Any],
  *,
  target_metric: str,
  target_mode: str,
  require_guardrails: bool,
) -> tuple[int, int, int, float, int, str]:
  metric_value = _extract_metric_value(result, target_metric)
  status_ok = result.get("status") == "ok"
  guardrail_pass = _result_guardrail_pass(result)
  if target_mode == "at_least":
    objective = -(metric_value if metric_value is not None else -1e18)
  elif target_mode == "at_most":
    objective = metric_value if metric_value is not None else 1e18
  else:
    raise ValueError(f"Unsupported target_mode '{target_mode}'")
  if require_guardrails:
    return (
      0 if guardrail_pass else 1,
      0 if status_ok else 1,
      0 if metric_value is not None else 1,
      float(objective),
      0 if guardrail_pass else 1,
      str(result.get("label", "")),
    )
  return (
    0 if status_ok else 1,
    0 if metric_value is not None else 1,
    0,
    float(objective),
    0 if guardrail_pass else 1,
    str(result.get("label", "")),
  )


def _best_result_for_target(
  results: list[dict[str, Any]],
  *,
  target_metric: str,
  target_mode: str,
  require_guardrails: bool,
) -> dict[str, Any] | None:
  if not results:
    return None
  ordered = sorted(
    results,
    key=lambda result: _objective_sort_key(
      result,
      target_metric=target_metric,
      target_mode=target_mode,
      require_guardrails=require_guardrails,
    ),
  )
  return ordered[0]


def _best_target_hit(
  results: list[dict[str, Any]],
  *,
  target_metric: str,
  target_value: float,
  target_mode: str,
  require_guardrails: bool,
) -> dict[str, Any] | None:
  matching = [
    result
    for result in results
    if _result_meets_target(
      result,
      target_metric=target_metric,
      target_value=target_value,
      target_mode=target_mode,
      require_guardrails=require_guardrails,
    )
  ]
  return _best_result_for_target(
    matching,
    target_metric=target_metric,
    target_mode=target_mode,
    require_guardrails=require_guardrails,
  )


def _query_nvidia_smi(gpu_index: int) -> dict[str, float] | None:
  try:
    result = subprocess.run(
      [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
      ],
      check=False,
      capture_output=True,
      text=True,
      timeout=2.0,
    )
  except (FileNotFoundError, subprocess.TimeoutExpired):  # pragma: no cover - environment-dependent
    return None

  if result.returncode != 0:
    return None

  for raw_line in result.stdout.splitlines():
    parts = [part.strip() for part in raw_line.split(",")]
    if len(parts) != 5:
      continue
    try:
      index = int(parts[0])
    except ValueError:
      continue
    if index != gpu_index:
      continue
    try:
      util = float(parts[2])
      mem_used = float(parts[3])
      mem_total = float(parts[4])
    except ValueError:
      return None
    return {
      "gpu_index": float(index),
      "gpu_util_percent": util,
      "gpu_mem_used_mb": mem_used,
      "gpu_mem_total_mb": mem_total,
      "gpu_mem_percent": (100.0 * mem_used / mem_total) if mem_total > 0 else 0.0,
    }
  return None


class ResourceMonitor(threading.Thread):
  def __init__(self, pid: int, *, interval_s: float, gpu_index: int):
    super().__init__(daemon=True)
    self.pid = int(pid)
    self.interval_s = float(interval_s)
    self.gpu_index = int(gpu_index)
    self.samples: list[dict[str, float]] = []
    self._stop_event = threading.Event()
    self._process = None if psutil is None else psutil.Process(self.pid)

  def stop(self) -> None:
    self._stop_event.set()

  def _process_tree_rss_mb(self) -> float | None:
    if self._process is None:
      return None
    try:
      processes = [self._process] + self._process.children(recursive=True)
    except Exception:  # pragma: no cover - psutil races
      return None

    rss = 0
    found = False
    for process in processes:
      try:
        rss += int(process.memory_info().rss)
        found = True
      except Exception:  # pragma: no cover - psutil races
        continue
    if not found:
      return None
    return rss / (1024.0 * 1024.0)

  def run(self) -> None:  # pragma: no cover - timing-heavy integration behavior
    if psutil is not None:
      psutil.cpu_percent(interval=None)

    while not self._stop_event.is_set():
      sample: dict[str, float] = {"timestamp": time.time()}
      if psutil is not None:
        try:
          sample["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
          sample["system_ram_percent"] = float(psutil.virtual_memory().percent)
        except Exception:
          pass
        rss_mb = self._process_tree_rss_mb()
        if rss_mb is not None:
          sample["process_tree_rss_mb"] = float(rss_mb)

      gpu = _query_nvidia_smi(self.gpu_index)
      if gpu is not None:
        sample.update(gpu)
      self.samples.append(sample)
      self._stop_event.wait(self.interval_s)


def _summarize_resource_samples(samples: list[dict[str, float]]) -> dict[str, float]:
  keys = (
    "system_cpu_percent",
    "system_ram_percent",
    "process_tree_rss_mb",
    "gpu_util_percent",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_mem_percent",
  )
  summary: dict[str, float] = {}
  for key in keys:
    values = [sample[key] for sample in samples if key in sample]
    mean = _safe_mean(values)
    peak = _safe_max(values)
    if mean is not None:
      summary[f"{key}_mean"] = mean
    if peak is not None:
      summary[f"{key}_max"] = peak
  return summary


def _write_json(path: Path, payload: Any) -> None:
  path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for record in records:
      handle.write(json.dumps(_json_safe(record), sort_keys=True))
      handle.write("\n")


def _parse_log_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, float]], list[str]]:
  epoch_records: list[dict[str, Any]] = []
  env_profiles: list[dict[str, float]] = []
  lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  for line in lines:
    epoch = parse_epoch_log_line(line)
    if epoch is not None:
      epoch_records.append(epoch)
      continue
    env_profile = parse_env_profile_line(line)
    if env_profile is not None:
      env_profiles.append(env_profile)
  return epoch_records, env_profiles, lines


def _tail_records(records: list[dict[str, Any]], tail_epochs: int) -> list[dict[str, Any]]:
  if tail_epochs <= 0 or len(records) <= tail_epochs:
    return list(records)
  return records[-tail_epochs:]


def _phase_breakdown(metrics: dict[str, float]) -> dict[str, Any]:
  env_time = float(metrics.get("performance/env", 0.0))
  copy_time = float(metrics.get("performance/eval_copy", 0.0)) + float(metrics.get("performance/train_copy", 0.0))
  forward_time = float(metrics.get("performance/eval_forward", 0.0)) + float(metrics.get("performance/train_forward", 0.0))
  learn_time = float(metrics.get("performance/learn", 0.0))
  misc_time = float(metrics.get("performance/eval_misc", 0.0)) + float(metrics.get("performance/train_misc", 0.0))
  total = env_time + copy_time + forward_time + learn_time + misc_time
  if total <= 0.0:
    return {
      "dominant_component": None,
      "bound_classification": None,
      "components": {},
    }

  components = {
    "env": env_time,
    "copy": copy_time,
    "forward": forward_time,
    "learn": learn_time,
    "misc": misc_time,
  }
  dominant_component = max(components, key=components.get)
  if dominant_component == "env":
    bound_classification = "env_bound"
  elif dominant_component == "copy":
    bound_classification = "copy_bound"
  elif dominant_component in {"forward", "learn"}:
    bound_classification = "learner_bound"
  else:
    bound_classification = "misc_bound"

  component_shares = {
    key: {
      "seconds": value,
      "share": value / total,
    }
    for key, value in components.items()
  }
  return {
    "dominant_component": dominant_component,
    "bound_classification": bound_classification,
    "components": component_shares,
    "total_seconds": total,
  }


def _record_has_nonzero_phase_metrics(record: dict[str, Any]) -> bool:
  phase_keys = (
    "performance/env",
    "performance/eval_copy",
    "performance/train_copy",
    "performance/eval_forward",
    "performance/train_forward",
    "performance/learn",
    "performance/eval_misc",
    "performance/train_misc",
  )
  for key in phase_keys:
    value = record.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0:
      return True
  return False


def _detect_failure_kind(exit_code: int, lines: list[str]) -> str | None:
  if exit_code == 0:
    return None
  lower_text = "\n".join(lines).lower()
  if "out of memory" in lower_text:
    return "oom"
  if "keyboardinterrupt" in lower_text:
    return "interrupted"
  if "traceback" in lower_text:
    return "python_exception"
  return "nonzero_exit"


def _load_adaptive_search_space(spec: dict[str, Any]) -> dict[str, list[Any]]:
  raw = spec.get("adaptive_search_space")
  if raw is None:
    return dict(ADAPTIVE_SEARCH_SPACE)
  if not isinstance(raw, dict):
    raise ValueError("adaptive_search_space must be a JSON object when provided")
  normalized: dict[str, list[Any]] = {}
  for key, values in raw.items():
    if not isinstance(key, str):
      raise ValueError("adaptive_search_space keys must be strings")
    if not isinstance(values, list) or not values:
      raise ValueError(f"adaptive_search_space['{key}'] must be a non-empty array")
    normalized[key] = list(values)
  return normalized


def _candidate_values(current: Any, values: list[Any]) -> list[Any]:
  alternatives = [value for value in values if value != current]
  if isinstance(current, (int, float)) and not isinstance(current, bool):
    return sorted(alternatives, key=lambda value: abs(float(value) - float(current)))
  return alternatives


def _sanitize_label_component(value: Any) -> str:
  text = str(value).lower()
  sanitized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
  return sanitized or "value"


def _adaptive_label(knob: str, value: Any) -> str:
  knob_label = knob.replace(".", "_")
  return f"{knob_label}_{_sanitize_label_component(value)}"


def _propose_adaptive_variants(
  incumbent_overrides: dict[str, Any],
  *,
  tried_signatures: set[str],
  search_space: dict[str, list[Any]],
  bound_classification: str | None,
  batch_size: int,
) -> list[ProbeVariant]:
  priority = BOUND_PRIORITY.get(bound_classification or "", BOUND_PRIORITY["misc_bound"])
  proposals: list[ProbeVariant] = []
  seen_labels: set[str] = set()

  for knob in priority:
    values = search_space.get(knob)
    if not values:
      continue
    current_value = incumbent_overrides.get(knob, CONTROL_OVERRIDES.get(knob))
    for candidate_value in _candidate_values(current_value, values):
      overrides = {knob: candidate_value}
      if knob == "vec.num_envs":
        overrides["vec.batch_size"] = candidate_value
      merged = dict(incumbent_overrides)
      merged.update(overrides)
      signature = _signature_for_overrides(merged)
      if signature in tried_signatures:
        continue
      label = _adaptive_label(knob, candidate_value)
      if label in seen_labels:
        continue
      description = f"Adaptive probe: set {knob}={candidate_value}"
      proposals.append(
        ProbeVariant(
          label=label,
          description=description,
          overrides=overrides,
          env={},
        )
      )
      seen_labels.add(label)
      if len(proposals) >= batch_size:
        return proposals
  return proposals


def _run_variant(
  *,
  variant: ProbeVariant,
  control_overrides: dict[str, Any],
  args: argparse.Namespace,
  run_name: str | None = None,
) -> dict[str, Any]:
  merged_overrides = _merged_overrides(control_overrides, variant)
  resolved_run_name = run_name or variant.label
  run_dir = args.output_dir / resolved_run_name
  run_dir.mkdir(parents=True, exist_ok=True)

  log_path = run_dir / "train.log"
  epochs_path = run_dir / "epochs.jsonl"
  env_profile_path = run_dir / "env_profile.jsonl"
  resource_path = run_dir / "resource_samples.jsonl"
  summary_path = run_dir / "summary.json"

  command = _build_train_command(args.python_executable, args.config, merged_overrides)
  env = os.environ.copy()
  env["PYTHONPATH"] = _path_with_pythonpath(env.get("PYTHONPATH"))
  env["PYTHONUNBUFFERED"] = "1"
  env["RICH_NO_COLOR"] = "1"
  env.setdefault("WANDB_MODE", "disabled")
  env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
  env["AZK_ENV_PROFILE"] = "1"
  env["AZK_ENV_PROFILE_EVERY"] = str(args.env_profile_every)
  env.update(variant.env)

  started_at = time.time()
  with log_path.open("w", encoding="utf-8") as log_handle:
    process = subprocess.Popen(
      command,
      cwd=REPO_ROOT,
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
    )
    monitor = ResourceMonitor(process.pid, interval_s=args.monitor_interval, gpu_index=args.gpu_index)
    monitor.start()
    try:
      assert process.stdout is not None
      for line in process.stdout:
        log_handle.write(line)
    finally:
      if process.stdout is not None:
        process.stdout.close()
      exit_code = process.wait()
      monitor.stop()
      monitor.join(timeout=max(1.0, args.monitor_interval * 4.0))

  finished_at = time.time()
  epoch_records, env_profiles, lines = _parse_log_file(log_path)
  _write_jsonl(epochs_path, epoch_records)
  _write_jsonl(env_profile_path, env_profiles)
  _write_jsonl(resource_path, monitor.samples)

  tail = _tail_records(epoch_records, args.tail_epochs)
  tail_mean = _mean_numeric_records(tail)
  env_profile_mean = _mean_numeric_records(env_profiles)
  resource_summary = _summarize_resource_samples(monitor.samples)
  action_sanity = evaluate_action_sanity(tail_mean)
  phase_records = [record for record in tail if _record_has_nonzero_phase_metrics(record)]
  phase_metrics = _mean_numeric_records(phase_records)
  phase_breakdown = _phase_breakdown(phase_metrics)
  failure_kind = _detect_failure_kind(exit_code, lines)

  summary = {
    "run_name": resolved_run_name,
    "label": variant.label,
    "description": variant.description,
    "status": "ok" if exit_code == 0 else "failed",
    "exit_code": int(exit_code),
    "failure_kind": failure_kind,
    "log_path": str(log_path.resolve()),
    "epochs_path": str(epochs_path.resolve()),
    "env_profile_path": str(env_profile_path.resolve()),
    "resource_samples_path": str(resource_path.resolve()),
    "command": command,
    "command_str": " ".join(shlex.quote(part) for part in command),
    "overrides": merged_overrides,
    "runtime_seconds": finished_at - started_at,
    "epoch_count": len(epoch_records),
    "tail_epoch_count": len(tail),
    "tail_mean": tail_mean,
    "last_epoch": epoch_records[-1] if epoch_records else None,
    "env_profile_count": len(env_profiles),
    "env_profile_mean": env_profile_mean,
    "resource_summary": resource_summary,
    "action_sanity": action_sanity,
    "phase_record_count": len(phase_records),
    "phase_breakdown": phase_breakdown,
    "summary_path": str(summary_path.resolve()),
  }
  _write_json(summary_path, summary)
  return summary


def _rank_variants(
  results: list[dict[str, Any]],
  *,
  target_metric: str,
  target_mode: str,
  require_guardrails: bool,
) -> list[dict[str, Any]]:
  control_metric_value = None
  for result in results:
    if result.get("label") == "control":
      control_metric_value = _extract_metric_value(result, target_metric)
      break

  ranking = []
  for result in results:
    metric_value = _extract_metric_value(result, target_metric)
    action_sanity = result.get("action_sanity", {})
    guardrail_pass = bool(result.get("status") == "ok" and action_sanity.get("pass") is True and metric_value is not None)
    delta_vs_control = None
    if control_metric_value is not None and metric_value is not None:
      delta_vs_control = float(metric_value) - float(control_metric_value)
    relative_delta_vs_control = None
    if control_metric_value not in (None, 0) and metric_value is not None:
      relative_delta_vs_control = (float(metric_value) / float(control_metric_value)) - 1.0
    ranking.append(
      {
        "label": result.get("label"),
        "run_name": result.get("run_name"),
        "status": result.get("status"),
        "guardrail_pass": guardrail_pass,
        "metric_path": target_metric,
        "metric_value": metric_value,
        "delta_vs_control": delta_vs_control,
        "relative_delta_vs_control": relative_delta_vs_control,
        "bound_classification": result.get("phase_breakdown", {}).get("bound_classification"),
        "dominant_component": result.get("phase_breakdown", {}).get("dominant_component"),
        "failure_kind": result.get("failure_kind"),
        "summary_path": result.get("summary_path"),
      }
    )

  ranking.sort(
    key=lambda item: _objective_sort_key(
      {
        "label": item["label"],
        "status": item["status"],
        "action_sanity": {"pass": item["guardrail_pass"]},
        "metric_wrapper": {"value": item["metric_value"]},
      },
      target_metric="metric_wrapper.value",
      target_mode=target_mode,
      require_guardrails=require_guardrails,
    )
  )
  return ranking


def _print_variant_plan(control_overrides: dict[str, Any], variants: list[ProbeVariant]) -> None:
  print("Control overrides:")
  print(json.dumps(_json_safe(control_overrides), indent=2, sort_keys=True))
  print("Variants:")
  for variant in variants:
    print(f"- {variant.label}: {variant.description}")
    if variant.overrides:
      print(f"  overrides={json.dumps(_json_safe(variant.overrides), sort_keys=True)}")
    if variant.env:
      print(f"  env={json.dumps(_json_safe(variant.env), sort_keys=True)}")


def _run_fixed_search(
  *,
  args: argparse.Namespace,
  control_overrides: dict[str, Any],
  variants: list[ProbeVariant],
  variant_file: Path | None,
) -> dict[str, Any]:
  planned_runs_path = args.output_dir / "planned_runs.json"
  _write_json(
    planned_runs_path,
    {
      "mode": "fixed",
      "config": str(args.config.resolve()),
      "variant_file": str(variant_file.resolve()) if variant_file is not None else None,
      "control_overrides": control_overrides,
      "variants": [
        {
          "label": variant.label,
          "description": variant.description,
          "overrides": variant.overrides,
          "env": variant.env,
        }
        for variant in variants
      ],
    },
  )

  results = []
  for variant in variants:
    print(f"[autoresearch] running {variant.label}")
    result = _run_variant(variant=variant, control_overrides=control_overrides, args=args)
    results.append(result)
    metric_value = _extract_metric_value(result, args.target_metric)
    action_sanity = result.get("action_sanity", {})
    guardrail_state = action_sanity.get("pass")
    if guardrail_state is True:
      guardrail_label = "pass"
    elif guardrail_state is False:
      guardrail_label = "fail"
    else:
      guardrail_label = "unavailable"
    print(
      "[autoresearch] finished "
      f"{variant.label}: status={result['status']}, "
      f"target_metric={None if metric_value is None else round(float(metric_value), 4)}, "
      f"guardrails={guardrail_label}"
    )

  ranking = _rank_variants(
    results,
    target_metric=args.target_metric,
    target_mode=args.target_mode,
    require_guardrails=bool(args.require_guardrails),
  )
  return {
    "mode": "fixed",
    "config": str(args.config.resolve()),
    "variant_file": str(variant_file.resolve()) if variant_file is not None else None,
    "output_dir": str(args.output_dir.resolve()),
    "control_overrides": control_overrides,
    "results": results,
    "ranking": ranking,
  }


def _run_adaptive_search(
  *,
  args: argparse.Namespace,
  spec: dict[str, Any],
  control_overrides: dict[str, Any],
  variants: list[ProbeVariant],
  variant_file: Path | None,
) -> dict[str, Any]:
  search_space = _load_adaptive_search_space(spec)
  planned_runs_path = args.output_dir / "planned_runs.json"
  _write_json(
    planned_runs_path,
    {
      "mode": "adaptive",
      "config": str(args.config.resolve()),
      "variant_file": str(variant_file.resolve()) if variant_file is not None else None,
      "control_overrides": control_overrides,
      "adaptive_search_space": search_space,
      "target_metric": args.target_metric,
      "target_mode": args.target_mode,
      "target_value": args.target_value,
      "adaptive_max_rounds": args.adaptive_max_rounds,
      "adaptive_max_runs": args.adaptive_max_runs,
      "adaptive_batch_size": args.adaptive_batch_size,
      "adaptive_start_with_variant_file": args.adaptive_start_with_variant_file,
      "variants": [
        {
          "label": variant.label,
          "description": variant.description,
          "overrides": variant.overrides,
          "env": variant.env,
        }
        for variant in variants
      ],
    },
  )

  all_results: list[dict[str, Any]] = []
  rounds: list[dict[str, Any]] = []
  tried_signatures: set[str] = set()
  stop_reason = "max_rounds_reached"

  control_variant = ProbeVariant(
    label="control",
    description="Adaptive control baseline.",
    overrides={},
    env={},
  )
  control_result = _run_variant(
    variant=control_variant,
    control_overrides=control_overrides,
    args=args,
    run_name="round_00_control",
  )
  tried_signatures.add(_signature_for_overrides(_merged_overrides(control_overrides, control_variant)))
  all_results.append(control_result)
  rounds.append(
    {
      "round_index": 0,
      "strategy": "control",
      "variant_labels": ["control"],
      "result_run_names": [control_result["run_name"]],
      "incumbent_run_name": control_result["run_name"],
    }
  )

  incumbent = _best_result_for_target(
    all_results,
    target_metric=args.target_metric,
    target_mode=args.target_mode,
    require_guardrails=bool(args.require_guardrails),
  )
  if incumbent is not None and _result_meets_target(
    incumbent,
    target_metric=args.target_metric,
    target_value=float(args.target_value),
    target_mode=args.target_mode,
    require_guardrails=bool(args.require_guardrails),
  ):
    stop_reason = "target_reached"
  else:
    next_variants: list[ProbeVariant] = []
    if args.adaptive_start_with_variant_file:
      for variant in variants:
        merged = _merged_overrides(control_overrides, variant)
        signature = _signature_for_overrides(merged)
        if signature in tried_signatures:
          continue
        next_variants.append(variant)

    round_index = 1
    while round_index <= args.adaptive_max_rounds and len(all_results) < args.adaptive_max_runs:
      if not next_variants:
        if incumbent is None:
          stop_reason = "no_incumbent"
          break
        incumbent_overrides = _merged_overrides(control_overrides, ProbeVariant(
          label=str(incumbent["label"]),
          description=str(incumbent.get("description", incumbent["label"])),
          overrides=incumbent.get("overrides", {}),
          env={},
        ))
        next_variants = _propose_adaptive_variants(
          incumbent_overrides,
          tried_signatures=tried_signatures,
          search_space=search_space,
          bound_classification=incumbent.get("phase_breakdown", {}).get("bound_classification"),
          batch_size=args.adaptive_batch_size,
        )
      if not next_variants:
        stop_reason = "search_space_exhausted"
        break

      remaining_budget = max(0, args.adaptive_max_runs - len(all_results))
      batch = []
      for variant in next_variants[:remaining_budget]:
        signature = _signature_for_overrides(_merged_overrides(control_overrides, variant))
        if signature in tried_signatures:
          continue
        batch.append(variant)
      next_variants = []
      if not batch:
        stop_reason = "search_space_exhausted"
        break

      round_results: list[dict[str, Any]] = []
      round_labels: list[str] = []
      for variant in batch:
        merged = _merged_overrides(control_overrides, variant)
        signature = _signature_for_overrides(merged)
        if signature in tried_signatures:
          continue
        run_name = f"round_{round_index:02d}_{variant.label}"
        print(f"[autoresearch] running {run_name}")
        result = _run_variant(
          variant=variant,
          control_overrides=control_overrides,
          args=args,
          run_name=run_name,
        )
        tried_signatures.add(signature)
        all_results.append(result)
        round_results.append(result)
        round_labels.append(variant.label)
        metric_value = _extract_metric_value(result, args.target_metric)
        action_sanity = result.get("action_sanity", {})
        guardrail_state = action_sanity.get("pass")
        if guardrail_state is True:
          guardrail_label = "pass"
        elif guardrail_state is False:
          guardrail_label = "fail"
        else:
          guardrail_label = "unavailable"
        print(
          "[autoresearch] finished "
          f"{run_name}: status={result['status']}, "
          f"target_metric={None if metric_value is None else round(float(metric_value), 4)}, "
          f"guardrails={guardrail_label}"
        )
        if _result_meets_target(
          result,
          target_metric=args.target_metric,
          target_value=float(args.target_value),
          target_mode=args.target_mode,
          require_guardrails=bool(args.require_guardrails),
        ):
          incumbent = _best_result_for_target(
            all_results,
            target_metric=args.target_metric,
            target_mode=args.target_mode,
            require_guardrails=bool(args.require_guardrails),
          )
          stop_reason = "target_reached"
          break

      incumbent = _best_result_for_target(
        all_results,
        target_metric=args.target_metric,
        target_mode=args.target_mode,
        require_guardrails=bool(args.require_guardrails),
      )
      rounds.append(
        {
          "round_index": round_index,
          "strategy": "variant_file_seed" if round_index == 1 and args.adaptive_start_with_variant_file else "adaptive",
          "variant_labels": round_labels,
          "result_run_names": [item["run_name"] for item in round_results],
          "incumbent_run_name": incumbent["run_name"] if incumbent is not None else None,
        }
      )
      if stop_reason == "target_reached":
        break

      if len(all_results) >= args.adaptive_max_runs:
        stop_reason = "run_budget_exhausted"
        break

      if incumbent is None:
        stop_reason = "no_incumbent"
        break

      incumbent_variant = ProbeVariant(
        label=str(incumbent["label"]),
        description=str(incumbent.get("description", incumbent["label"])),
        overrides=incumbent.get("overrides", {}),
        env={},
      )
      next_variants = _propose_adaptive_variants(
        _merged_overrides(control_overrides, incumbent_variant),
        tried_signatures=tried_signatures,
        search_space=search_space,
        bound_classification=incumbent.get("phase_breakdown", {}).get("bound_classification"),
        batch_size=args.adaptive_batch_size,
      )
      round_index += 1

  ranking = _rank_variants(
    all_results,
    target_metric=args.target_metric,
    target_mode=args.target_mode,
    require_guardrails=bool(args.require_guardrails),
  )
  incumbent = _best_result_for_target(
    all_results,
    target_metric=args.target_metric,
    target_mode=args.target_mode,
    require_guardrails=bool(args.require_guardrails),
  )
  target_result = _best_target_hit(
    all_results,
    target_metric=args.target_metric,
    target_value=float(args.target_value),
    target_mode=args.target_mode,
    require_guardrails=bool(args.require_guardrails),
  )
  return {
    "mode": "adaptive",
    "config": str(args.config.resolve()),
    "variant_file": str(variant_file.resolve()) if variant_file is not None else None,
    "output_dir": str(args.output_dir.resolve()),
    "control_overrides": control_overrides,
    "adaptive_search_space": search_space,
    "target_metric": args.target_metric,
    "target_mode": args.target_mode,
    "target_value": args.target_value,
    "require_guardrails": bool(args.require_guardrails),
    "target_reached": target_result is not None,
    "incumbent_run_name": incumbent["run_name"] if incumbent is not None else None,
    "target_result_run_name": target_result["run_name"] if target_result is not None else None,
    "stop_reason": stop_reason,
    "rounds": rounds,
    "results": all_results,
    "ranking": ranking,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run short matched Azuki SPS probes on the local 3090.")
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
  parser.add_argument("--variant-file", type=Path, default=DEFAULT_VARIANT_FILE)
  parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
  parser.add_argument("--python-executable", type=Path, default=_default_python_executable())
  parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
  parser.add_argument("--total-timesteps", type=int, default=30720)
  parser.add_argument("--tail-epochs", type=int, default=5)
  parser.add_argument("--monitor-interval", type=float, default=1.0)
  parser.add_argument("--env-profile-every", type=int, default=4000)
  parser.add_argument("--gpu-index", type=int, default=0)
  parser.add_argument("--target-metric", type=str, default=DEFAULT_TARGET_METRIC)
  parser.add_argument("--target-mode", choices=("at_least", "at_most"), default=DEFAULT_TARGET_MODE)
  parser.add_argument("--target-value", type=float)
  parser.add_argument("--require-guardrails", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--adaptive-max-rounds", type=int, default=4)
  parser.add_argument("--adaptive-max-runs", type=int, default=16)
  parser.add_argument("--adaptive-batch-size", type=int, default=4)
  parser.add_argument("--adaptive-start-with-variant-file", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--control-only", action="store_true")
  parser.add_argument("--list-variants", action="store_true")
  args = parser.parse_args()
  if args.mode == "adaptive" and args.target_value is None:
    parser.error("--target-value is required when --mode adaptive")
  if args.adaptive_batch_size <= 0:
    parser.error("--adaptive-batch-size must be > 0")
  if args.adaptive_max_rounds <= 0:
    parser.error("--adaptive-max-rounds must be > 0")
  if args.adaptive_max_runs <= 0:
    parser.error("--adaptive-max-runs must be > 0")
  return args


def main() -> None:
  args = parse_args()
  variant_file = args.variant_file
  if variant_file is not None and not variant_file.exists():
    variant_file = None

  spec = _load_variant_spec(variant_file, args.total_timesteps)
  control_overrides, variants = _materialize_variants(spec)
  if args.control_only:
    variants = [variant for variant in variants if variant.label == "control"]
    if not variants:
      raise ValueError("control-only requested, but no 'control' variant exists in the spec")

  if args.list_variants:
    _print_variant_plan(control_overrides, variants)
    return

  args.output_dir.mkdir(parents=True, exist_ok=True)
  if args.mode == "adaptive":
    summary = _run_adaptive_search(
      args=args,
      spec=spec,
      control_overrides=control_overrides,
      variants=variants,
      variant_file=variant_file,
    )
  else:
    summary = _run_fixed_search(
      args=args,
      control_overrides=control_overrides,
      variants=variants,
      variant_file=variant_file,
    )
  summary_path = args.output_dir / "summary.json"
  _write_json(summary_path, summary)

  print(f"[autoresearch] wrote summary to {summary_path}")
  ranking = summary.get("ranking", [])
  if ranking:
    best = ranking[0]
    print(
      "[autoresearch] top result: "
      f"{best['label']} guardrail_pass={best['guardrail_pass']} "
      f"metric={best['metric_value']} bound={best['bound_classification']}"
    )


if __name__ == "__main__":
  main()
