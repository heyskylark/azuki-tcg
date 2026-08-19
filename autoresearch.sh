#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/python/.venv-codex/bin/python"
CANDIDATE="$ROOT/python/config/autoresearch_training_tuning.json"
OUTPUT_DIR="/tmp/azuki-autoresearch-training-sps"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Required benchmark interpreter is missing: %s\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$ROOT/build/CMakeCache.txt" ]]; then
  printf 'Configured native build directory is missing: %s\n' "$ROOT/build" >&2
  exit 1
fi
if [[ ! -f "$CANDIDATE" ]]; then
  printf 'Tuning candidate is missing: %s\n' "$CANDIDATE" >&2
  exit 1
fi

cmake --build "$ROOT/build" \
  --target azuki_puffer_env \
  --parallel "$(nproc)"

export AZK_BUILD_PYTHON_DIR="$ROOT/build/python/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=42
export PYTHONPATH="$ROOT/python/src:$ROOT/build/python/src"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export NEPTUNE_MODE=offline

"$PYTHON" -c '
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if set(payload) != {"description", "control_overrides", "variants"}:
    raise SystemExit("candidate must contain only description, control_overrides, and variants")
variants = payload["variants"]
if not isinstance(variants, list) or len(variants) != 2:
    raise SystemExit("candidate must define matched control and candidate variants")
if variants[0] != {
    "label": "control",
    "description": "Fixed post-core-optimization control.",
}:
    raise SystemExit("the fixed control variant changed")
candidate_variant = variants[1]
if set(candidate_variant) != {"label", "description", "overrides"}:
    raise SystemExit("candidate variant must contain only label, description, and overrides")
if candidate_variant["label"] != "candidate":
    raise SystemExit("candidate variant label changed")
baseline = {
    "env.direct_parallel": True,
    "league.enable": False,
    "neptune": False,
    "policy.legal_action_scorer_use_references": True,
    "train.batch_size": 3840,
    "train.compile": False,
    "train.max_minibatch_size": 960,
    "train.minibatch_size": 960,
    "train.precision": "bfloat16",
    "train.seed": 42,
    "train.seed_process_rngs": True,
    "vec.batch_size": 120,
    "vec.num_envs": 120,
    "vec.num_workers": 4,
    "vec.seed": 42,
    "vec.zero_copy": True,
    "wandb": False,
}
controls = payload["control_overrides"]
if controls != baseline or any(type(controls[key]) is not type(value) for key, value in baseline.items()):
    raise SystemExit("fixed control configuration changed")
candidate = candidate_variant["overrides"]
tunable_types = {
    "policy.legal_action_scorer_use_references": bool,
    "train.compile": bool,
    "train.max_minibatch_size": int,
    "train.minibatch_size": int,
    "train.precision": str,
    "vec.batch_size": int,
    "vec.num_envs": int,
    "vec.num_workers": int,
    "vec.zero_copy": bool,
}
if set(candidate) != set(tunable_types):
    raise SystemExit(
        f"candidate keys differ: missing={sorted(set(tunable_types) - set(candidate))}, "
        f"extra={sorted(set(candidate) - set(tunable_types))}"
    )
for key, expected_type in tunable_types.items():
    if type(candidate[key]) is not expected_type:
        raise SystemExit(f"{key} must be {expected_type.__name__}")
num_envs = candidate["vec.num_envs"]
num_workers = candidate["vec.num_workers"]
vector_batch = candidate["vec.batch_size"]
minibatch = candidate["train.minibatch_size"]
if not 24 <= num_envs <= 720:
    raise SystemExit("vec.num_envs must be in [24, 720]")
if not 1 <= num_workers <= 12 or num_envs % num_workers:
    raise SystemExit("vec.num_workers must be in [1, 12] and divide vec.num_envs")
envs_per_worker = num_envs // num_workers
if not envs_per_worker <= vector_batch <= num_envs:
    raise SystemExit("vec.batch_size must be between envs-per-worker and vec.num_envs")
if vector_batch % envs_per_worker:
    raise SystemExit("vec.batch_size must contain whole worker blocks")
if candidate["vec.zero_copy"] and num_envs % vector_batch:
    raise SystemExit("zero-copy requires vec.num_envs divisible by vec.batch_size")
if not 16 <= minibatch <= 3840 or minibatch % 16 or 3840 % minibatch:
    raise SystemExit("train.minibatch_size must be a BPTT-aligned divisor of 3840")
if candidate["train.max_minibatch_size"] != minibatch:
    raise SystemExit("train.max_minibatch_size must equal train.minibatch_size")
if candidate["train.precision"] not in {"bfloat16", "float32"}:
    raise SystemExit("train.precision must be bfloat16 or float32")
' "$CANDIDATE"

rm -rf -- "$OUTPUT_DIR"
"$PYTHON" "$ROOT/python/src/sps_autoresearch.py" \
  --config "$ROOT/python/config/azuki_speed_3090_parallel.ini" \
  --variant-file "$CANDIDATE" \
  --output-dir "$OUTPUT_DIR" \
  --python-executable "$PYTHON" \
  --mode fixed \
  --total-timesteps 30720 \
  --tail-epochs 5 \
  --require-guardrails

"$PYTHON" -c '
import json
import math
import statistics
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
results = summary.get("results")
if not isinstance(results, list) or len(results) != 2:
    raise SystemExit("benchmark must produce matched control and candidate results")
by_label = {result.get("label"): result for result in results}
if set(by_label) != {"control", "candidate"}:
    raise SystemExit("benchmark result labels changed")

def validate_result(label):
    result = by_label[label]
    if result.get("status") != "ok" or result.get("exit_code") != 0:
        raise SystemExit(
            "{} training failed: {}".format(label, result.get("failure_kind"))
        )
    if result.get("epoch_count") != 8 or result.get("tail_epoch_count") != 5:
        raise SystemExit(f"{label} must produce eight epochs and a five-epoch tail")
    if result.get("action_sanity", {}).get("pass") is not True:
        raise SystemExit(
            "{} action-sanity failed: {}".format(label, result.get("action_sanity"))
        )
    with open(result["epochs_path"], encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if [record.get("epoch_index") for record in records] != list(range(1, 9)):
        raise SystemExit(f"{label} epoch sequence is incomplete or out of order")
    losses = [
        float(value)
        for record in records
        for key, value in record.items()
        if key.startswith("losses/")
    ]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise SystemExit(f"{label} loss metrics are missing or non-finite")
    return result, records, losses

def mean_metric(records, key):
    values = [record.get(key) for record in records[-5:]]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        raise SystemExit(f"tail metric is missing or non-finite: {key}={values!r}")
    return statistics.fmean(float(value) for value in values)

def relative_delta(candidate, control, floor):
    return abs(candidate - control) / max(abs(control), floor)

control_result, control_records, control_losses = validate_result("control")
candidate_result, candidate_records, candidate_losses = validate_result("candidate")
behavior_deltas = []
candidate_episode_lengths = []
candidate_completed_episodes = []
action_keys = (
    "azk_noop_selected_rate",
    "azk_attack_selected_rate",
    "azk_play_selected_rate",
    "azk_ability_selected_rate",
    "azk_target_selected_rate",
)
for seat in (0, 1):
    prefix = f"environment/{seat}/"
    control_length = mean_metric(control_records, prefix + "azk_episode_length")
    candidate_length = mean_metric(candidate_records, prefix + "azk_episode_length")
    length_delta = relative_delta(candidate_length, control_length, 1.0)
    if length_delta > 0.20:
        raise SystemExit(f"seat {seat} episode-length delta {length_delta:.4f} exceeded 0.20")

    control_return = mean_metric(control_records, prefix + "azk_episode_return")
    candidate_return = mean_metric(candidate_records, prefix + "azk_episode_return")
    return_delta = relative_delta(candidate_return, control_return, 0.25)
    if return_delta > 0.50:
        raise SystemExit(f"seat {seat} episode-return delta {return_delta:.4f} exceeded 0.50")

    control_completed = mean_metric(control_records, prefix + "azk_completed_episodes")
    candidate_completed = mean_metric(candidate_records, prefix + "azk_completed_episodes")
    completion_delta = relative_delta(candidate_completed, control_completed, 1.0)
    if control_completed <= 0.0 or candidate_completed <= 0.0:
        raise SystemExit(f"seat {seat} completed no episodes")
    if completion_delta > 0.35:
        raise SystemExit(f"seat {seat} completion delta {completion_delta:.4f} exceeded 0.35")

    control_mix = [mean_metric(control_records, prefix + key) for key in action_keys]
    candidate_mix = [mean_metric(candidate_records, prefix + key) for key in action_keys]
    mix_delta = sum(
        abs(candidate - control) for candidate, control in zip(candidate_mix, control_mix)
    ) / max(sum(abs(value) for value in control_mix), 1e-6)
    if mix_delta > 0.20:
        raise SystemExit(f"seat {seat} action-mix delta {mix_delta:.4f} exceeded 0.20")

    timeout = mean_metric(candidate_records, prefix + "azk_timeout_truncation")
    auto_tick = mean_metric(candidate_records, prefix + "azk_auto_tick_truncation")
    if timeout > 0.05 or auto_tick > 0.05:
        raise SystemExit(
            f"seat {seat} truncation behavior changed: timeout={timeout}, auto_tick={auto_tick}"
        )
    behavior_deltas.extend((length_delta, return_delta, completion_delta, mix_delta))
    candidate_episode_lengths.append(candidate_length)
    candidate_completed_episodes.append(candidate_completed)

control_sps = mean_metric(control_records, "SPS")
training_sps = mean_metric(candidate_records, "SPS")
tail_sps = [float(record["SPS"]) for record in candidate_records[-5:]]
print(f"METRIC training_sps={training_sps:.6f}")
print(f"METRIC control_sps={control_sps:.6f}")
print(f"METRIC sps_relative_improvement={(training_sps / control_sps - 1.0):.6f}")
print(f"METRIC tail_sps_stddev={statistics.pstdev(tail_sps):.6f}")
print(
    "METRIC benchmark_runtime_seconds={:.6f}".format(
        float(control_result["runtime_seconds"]) + float(candidate_result["runtime_seconds"])
    )
)
print(
    f"METRIC rollout_episode_length={statistics.fmean(candidate_episode_lengths):.6f}"
)
print(
    f"METRIC rollout_completed_episodes={statistics.fmean(candidate_completed_episodes):.6f}"
)
print(f"METRIC rollout_relative_delta_max={max(behavior_deltas):.6f}")
print(f"METRIC finite_loss_values={len(candidate_losses)}")
print("METRIC action_sanity_pass=1")
print("METRIC rollout_behavior_pass=1")
print("METRIC finite_losses_pass=1")
print("METRIC benchmark_epochs=8")
' "$OUTPUT_DIR/summary.json"
