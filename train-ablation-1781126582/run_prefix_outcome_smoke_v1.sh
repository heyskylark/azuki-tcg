#!/usr/bin/env bash
# Stage 2 fallback integrity and throughput gate for frozen outcome redistribution.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${PREFIX_OUTCOME_SMOKE_CAMPAIGN:-prefix_outcome_smoke_v1}
PARENT_MANIFEST=${PREFIX_OUTCOME_PARENT_MANIFEST:-train-ablation-1781126582/results/next_ablation_v1/stage2/parent_manifest.json}
MODEL_ROOT=${PREFIX_OUTCOME_MODEL_ROOT:-train-ablation-1781126582/results/next_ablation_v1/stage2/prefix_outcome_model_v1}
MODEL_FILENAME=${PREFIX_OUTCOME_MODEL_FILENAME:-frozen_prefix_outcome_v1.npz}
PREDICTOR="$MODEL_ROOT/$MODEL_FILENAME"
PREDICTOR_REPORT="$MODEL_ROOT/predictor_report.json"
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
UPDATES=${PREFIX_OUTCOME_SMOKE_UPDATES:-48}
STEADY_UPDATES=${PREFIX_OUTCOME_SMOKE_STEADY_UPDATES:-24}
PEAK_LR=${PREFIX_OUTCOME_SMOKE_PEAK_LR:-0.000003}
COEFFICIENT=${PREFIX_OUTCOME_COEFFICIENT:-1.0}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235

if [[ ! -f "$PARENT_MANIFEST" || ! -f "$PREDICTOR" || ! -f "$PREDICTOR_REPORT" ]]; then
  echo "[$CAMPAIGN] missing parent or frozen predictor artifact" >&2
  exit 1
fi
if ! jq -e '.validation_complete == true' "$PREDICTOR_REPORT" >/dev/null; then
  echo "[$CAMPAIGN] predictor validation is incomplete" >&2
  exit 1
fi
PREDICTOR_SHA=$(jq -er '.artifact.sha256' "$PREDICTOR_REPORT")
if [[ "$(sha256sum "$PREDICTOR" | awk '{print $1}')" != "$PREDICTOR_SHA" ]]; then
  echo "[$CAMPAIGN] predictor hash mismatch" >&2
  exit 1
fi

manifest_value() { jq -er "$1" "$PARENT_MANIFEST"; }
PARENT_MODEL=$(manifest_value '.model.path')
PARENT_TRAINER=$(manifest_value '.trainer.path')
PARENT_METADATA=$(manifest_value '.metadata.path')
PARENT_LEAGUE=$(manifest_value '.league.path')
PARENT_PROMOTION=$(manifest_value '.promotion.path')
PARENT_EPOCH=$(manifest_value '.epoch')
EXPECTED_PARENT_EPISODES=$(manifest_value '.completed_episodes')
TARGET_EPOCH=$((PARENT_EPOCH + UPDATES))
TOTAL_TIMESTEPS=$((TARGET_EPOCH * 15360))
STEADY_EPOCH_START=$((TARGET_EPOCH - STEADY_UPDATES + 1))

if (( UPDATES < 8 || STEADY_UPDATES < 4 || STEADY_UPDATES >= UPDATES )); then
  echo "[$CAMPAIGN] invalid update windows" >&2
  exit 1
fi

require_manifest_hash() {
  local key=$1 path=$2 label=$3 expected actual
  expected=$(manifest_value ".$key.sha256")
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$CAMPAIGN] $label hash mismatch" >&2
    exit 1
  fi
}

latest_jsonl() {
  find experiments/runlogs -maxdepth 1 -type f -name "$1_*.jsonl" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

latest_run_dir() {
  find experiments -maxdepth 1 -type d -name "azuki_local_$1_*" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

cleanup_training_processes() {
  local tag=$1
  local pids=()
  mapfile -t pids < <(
    pgrep -f -- "python/src/train.py.*--tag ${tag}([[:space:]]|$)" 2>/dev/null || true
  )
  if (( ${#pids[@]} )); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
  fi
  local pid
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

run_arm() {
  local arm=$1 model_path=$2 model_sha=$3
  local arm_results="$RESULT_ROOT/$arm"
  local arm_league="$LEAGUE_ROOT/$arm"
  local snapshot_dir="experiments/abl_snapshots/${CAMPAIGN}_${arm}"
  local tag="${CAMPAIGN}_${arm}"
  local live_log="$arm_results/train.live.log"
  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] reuse completed $arm"
    return
  fi
  if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null ||
     [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
    echo "[$CAMPAIGN] existing incomplete state for $arm; refusing overwrite" >&2
    exit 1
  fi
  mkdir -p "$arm_results" "$arm_league/opponents" "$snapshot_dir"
  cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
  cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"

  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm"
  set +e
  env \
    AZK_RESUME_KEEP_CURRENT_REWARD_ENV=1 \
    AZK_REWARD_LEADER_DELTA_WEIGHT=1.25 \
    AZK_REWARD_BOARD_DELTA_WEIGHT=0.35 \
    AZK_REWARD_NOOP_PENALTY=0.02 \
    AZK_TRUNCATION_BOARD_EDGE_WEIGHT=0.45 \
    AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0.15 \
    AZK_REWARD_SHAPING_ANNEAL=1 \
    AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
    AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 \
    AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
    AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 \
    AZK_TRAINER_SHAPED_REWARD_ANNEAL=0 \
    AZK_TRAINER_SHAPED_REWARD_ANNEAL_START_EPOCH=0 \
    AZK_TRAINER_SHAPED_REWARD_ANNEAL_END_EPOCH=0 \
    AZK_PORTAL_GP_BONUS=0.3 \
    AZK_PORTAL_OUTCOME_BONUS=0 \
    AZK_XGATE_MASK=1 \
    AZK_PFSP=1 \
    AZK_PFSP_POWER=2.0 \
    AZK_EARLY_TEMPO_BONUS=0.1 \
    AZK_EARLY_TEMPO_CAP=4 \
    AZK_EARLY_TEMPO_TURNS=2 \
    AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES=1 \
    AZK_DMG_MITIGATION_BONUS=0.15 \
    AZK_DMG_MITIGATION_CAP=10 \
    AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0 \
    AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP=6 \
    AZK_GENERATED_IKZ_CONVERSION_BONUS=0 \
    AZK_GENERATED_IKZ_CONVERSION_STEP_CAP=4 \
    AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
    AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
    AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4 \
    AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0 \
    AZK_DRAFT_VBOOT_COEF=0 \
    AZK_DRAFT_SIBDIFF_COEF=0 \
    AZK_DRAFT_SIBDIFF_CAP=0.05 \
    AZK_DRAFT_AUX_ANNEAL=1 \
    AZK_DRAFT_REF_SEAT_PROB=0 \
    AZK_LEADER_TERMINAL_CREDIT_COEF=0 \
    AZK_DRAFT_TERMINAL_CREDIT_COEF=0 \
    AZK_DRAFT_EPISODE_CREDIT_COEF=0 \
    AZK_DRAFT_PREFIX_OUTCOME_MODEL="$model_path" \
    AZK_DRAFT_PREFIX_OUTCOME_SHA256="$model_sha" \
    AZK_DRAFT_PREFIX_OUTCOME_COEF="$COEFFICIENT" \
    AZK_RESUME_ALLOW_BINDING_MISMATCH=1 \
    AZK_RESUME_ALLOW_SOURCE_DRIFT=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/train.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --resume-checkpoint "$PARENT_MODEL" \
      --resume-load-optimizer --resume-restart-lr-schedule \
      --no-resume-auto-reset-critic \
      --jsonl-log experiments/runlogs --tag "$tag" \
      --train.seed 42 --train.learning_rate "$PEAK_LR" \
      --train.ent_coef 0.002 \
      --train.ent_coef_anneal_initial 0.002 \
      --train.ent_coef_anneal_final 0.002 \
      --train.total_timesteps "$TOTAL_TIMESTEPS" \
      --train.checkpoint_interval "$UPDATES" \
      --env.draft_uniform_assignment true \
      --env.deck_snapshot_every 100000 \
      --env.deck_snapshot_dir "$snapshot_dir" \
      --league.state_path "$arm_league/league_state.json" \
      --league.opponent_dir "$arm_league/opponents" \
      --league.eval_interval 100000 \
      --league.quick_eval_interval 100000 \
      --league.full_eval_interval 100000 \
      --league.promotion_shadow_mode true \
      --league.promotion_archive_affects_training_pool false \
      --league.promotion_panel_refresh_epochs 100000 \
      --league.production_anchor_checkpoint \
        experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt \
      > >(tee "$live_log") 2>&1
  local status=$?
  set -e
  cleanup_training_processes "$tag"
  if (( status != 0 )); then
    echo "[$CAMPAIGN] $arm failed with status $status" >&2
    exit "$status"
  fi

  local run_dir jsonl checkpoint trainer_state
  run_dir=$(latest_run_dir "$tag")
  jsonl=$(latest_jsonl "$tag")
  checkpoint=$(printf '%s/model_azuki_local_%06d.pt' "$run_dir" "$TARGET_EPOCH")
  trainer_state=$(printf '%s/trainer_state_%06d.pt' "$run_dir" "$TARGET_EPOCH")
  if [[ -z "$run_dir" || -z "$jsonl" || ! -f "$checkpoint" || ! -f "$trainer_state" ]]; then
    echo "[$CAMPAIGN] $arm lacks atomic p$TARGET_EPOCH output" >&2
    exit 1
  fi
  if ! grep -q 'optimizer_restored=True' "$live_log" ||
     ! grep -q "remaining_epochs=$UPDATES" "$live_log" ||
     ! grep -q "total_epochs=$TARGET_EPOCH" "$live_log" ||
     ! grep -q 'uniform_assignment=True' "$live_log"; then
    echo "[$CAMPAIGN] $arm resume/lifecycle invariants failed" >&2
    exit 1
  fi
  if [[ "$arm" == prefix_outcome ]] &&
     ! grep -q "\[draft-prefix-outcome\] enabled:.*sha256=$PREDICTOR_SHA" "$live_log"; then
    echo "[$CAMPAIGN] candidate did not load the frozen predictor" >&2
    exit 1
  fi
  cp -a "$jsonl" "$arm_results/train.jsonl"
  printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
  sha256sum "$checkpoint" "$trainer_state" > "$arm_results/checkpoint_sha256.txt"
  touch "$arm_results/TRAIN_DONE"
}

require_manifest_hash model "$PARENT_MODEL" parent-model
require_manifest_hash trainer "$PARENT_TRAINER" parent-trainer
require_manifest_hash metadata "$PARENT_METADATA" parent-metadata
require_manifest_hash league "$PARENT_LEAGUE" parent-league
require_manifest_hash promotion "$PARENT_PROMOTION" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
cp -a "$PARENT_MANIFEST" "$RESULT_ROOT/parent_manifest.json"
sha256sum \
  "$PREDICTOR" "$PREDICTOR_REPORT" \
  python/src/draft_prefix_outcome.py python/src/league_training.py python/src/train.py \
  python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
  build/python/src/binding*.so \
  train-ablation-1781126582/run_prefix_outcome_smoke_v1.sh \
  > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\ncoefficient=%s\npredictor_sha256=%s\n' \
  "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$COEFFICIENT" "$PREDICTOR_SHA" \
  > "$RESULT_ROOT/campaign_config.txt"

run_arm control "" ""
run_arm prefix_outcome "$PREDICTOR" "$PREDICTOR_SHA"

.venv/bin/python - "$RESULT_ROOT" "$PARENT_EPOCH" "$TARGET_EPOCH" \
  "$STEADY_EPOCH_START" "$SPS_RELATIVE_FLOOR" "$SPS_HARD_FLOOR" "$UPDATES" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
parent_epoch = int(sys.argv[2])
target_epoch = int(sys.argv[3])
steady_start = int(sys.argv[4])
relative_floor = float(sys.argv[5])
hard_floor = float(sys.argv[6])
updates = int(sys.argv[7])
integration_only = updates < 16

def summarize(arm):
  rows = {}
  for line in (root / arm / "train.jsonl").read_text(encoding="utf-8").splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    epoch = row.get("epoch")
    if isinstance(epoch, (int, float)) and parent_epoch < int(epoch) <= target_epoch:
      rows[int(epoch)] = row
  expected = set(range(parent_epoch + 1, target_epoch + 1))
  if set(rows) != expected:
    raise RuntimeError(f"{arm} epoch mismatch: missing={sorted(expected - set(rows))}")
  ordered = [rows[epoch] for epoch in sorted(rows)]
  intervals = []
  for left, right in zip(ordered, ordered[1:]):
    elapsed = float(right["uptime"]) - float(left["uptime"])
    steps = float(right["agent_steps"]) - float(left["agent_steps"])
    if elapsed > 0 and steps > 0:
      intervals.append((int(right["epoch"]), steps / elapsed))
  steady = [value for epoch, value in intervals if epoch >= steady_start]
  def vals(key):
    return [float(row[key]) for row in ordered if isinstance(row.get(key), (int, float))]
  prefix = "environment/draft_prefix_outcome/"
  draft_picks = vals("environment/deckbuild/picks")
  return {
    "epochs": len(ordered),
    "steady_interval_sps_median": statistics.median(steady),
    "steady_interval_sps_p10": sorted(steady)[max(0, int(0.1 * (len(steady) - 1)))],
    "timeout_rate_max": max(vals("environment/timeout_truncation_rate"), default=0.0),
    "draft_picks_mean": statistics.mean(draft_picks) if draft_picks else None,
    "delta_count": sum(vals(prefix + "delta_count")),
    "residual_count": sum(vals(prefix + "residual_count")),
    "completed": sum(vals(prefix + "completed")),
    "truncated": sum(vals(prefix + "truncated")),
    "unsynchronized": sum(vals(prefix + "unsynchronized")),
    "telescope_abs_max": max(vals(prefix + "telescope_abs_max"), default=0.0),
    "delta_abs_max": max(vals(prefix + "delta_abs_max"), default=0.0),
    "inference_seconds": sum(vals(prefix + "inference_seconds")),
    "prediction_std_mean": statistics.mean(vals(prefix + "prediction_std")) if vals(prefix + "prediction_std") else 0.0,
    "quartile_deltas": {
      quartile: sum(vals(prefix + quartile + "_deltas"))
      for quartile in ("q1_00_12", "q2_13_25", "q3_26_37", "q4_38_50")
    },
    "telemetry_rows": len(vals(prefix + "delta_count")),
  }

control = summarize("control")
candidate = summarize("prefix_outcome")
ratio = candidate["steady_interval_sps_median"] / control["steady_interval_sps_median"]
integrity = (
  candidate["telemetry_rows"] == candidate["epochs"]
  and candidate["delta_count"] > 0
  and (integration_only or candidate["residual_count"] > 0)
  and (integration_only or candidate["completed"] > 0)
  and candidate["truncated"] == 0
  and candidate["unsynchronized"] == 0
  and candidate["telescope_abs_max"] <= 1e-5
  and candidate["delta_abs_max"] > 0
  and candidate["prediction_std_mean"] > 0
  and candidate["inference_seconds"] > 0
  and all(value > 0 for value in candidate["quartile_deltas"].values())
  and candidate["timeout_rate_max"] == 0
  and (
    integration_only
    or (
      candidate["draft_picks_mean"] is not None
      and abs(candidate["draft_picks_mean"] - 50.0) < 1e-6
    )
  )
  and control["telemetry_rows"] == 0
)
performance = ratio >= relative_floor and candidate["steady_interval_sps_median"] >= hard_floor
payload = {
  "schema_version": 1,
  "integration_only": integration_only,
  "status": "pass" if integrity and performance else "fail",
  "integrity_pass": integrity,
  "performance_pass": performance,
  "sps_ratio": ratio,
  "relative_floor": relative_floor,
  "hard_floor": hard_floor,
  "control": control,
  "candidate": candidate,
}
(root / "smoke_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if not integrity or not performance:
  raise SystemExit(2)
PY

touch "$RESULT_ROOT/SMOKE_DONE"
echo "[$CAMPAIGN] smoke complete"
