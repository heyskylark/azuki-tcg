#!/usr/bin/env bash
# Matched p4870->p7800 validation of a fixed shaping floor versus a zero tail.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${SHAPED_REWARD_TAIL_CAMPAIGN:-shaped_reward_tail45_v1}
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
PARENT_DIR=experiments/azuki_local_strategy_recovery_leader15_v1_control_178447770084
PARENT_MODEL="$PARENT_DIR/model_azuki_local_004870.pt"
PARENT_TRAINER="$PARENT_DIR/trainer_state_004870.pt"
PARENT_METADATA="$PARENT_MODEL.meta.json"
PARENT_LEAGUE=experiments/league/strategy_recovery_leader15_v1/control/league_state.json
PARENT_PROMOTION=experiments/league/strategy_recovery_leader15_v1/control/league_state_promotion.json
EXPECTED_MODEL_SHA=def350888ecbf9014e590fb8540080cb26b6c14852882caa85def3f6db8a4f76
EXPECTED_TRAINER_SHA=ce1aa27a1ab58e4c9060f6de99b86cdfc6e8fe815803d9b3ba04155848aa3486
EXPECTED_LEAGUE_SHA=22b247ba70c6d6a72f92ab4af1e805e874c2aa5337a4bba5ea165145f3e236de
EXPECTED_PROMOTION_SHA=9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a
EXPECTED_PARENT_EPISODES=404
PARENT_EPOCH=4870
ANNEAL_START_EPOCH=6000
ANNEAL_END_EPOCH=6900
TARGET_EPOCH=7800
TOTAL_TIMESTEPS=119808000
PEAK_LR=${SHAPED_REWARD_TAIL_PEAK_LR:-0.00003}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235

arms=(fixed_floor zero_tail)
segment_names=(to6000 to6900 to7800)
segment_ends=(6000 6900 7800)

persisted_env_vars=(
  AZK_MAX_AUTO_TICKS_PER_STEP
  AZK_MAX_TICKS_PER_EPISODE
  AZK_MAX_TICKS_CURRICULUM
  AZK_MAX_TICKS_CURRICULUM_INITIAL
  AZK_MAX_TICKS_CURRICULUM_FINAL
  AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES
  AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES
  AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_EVERY
  AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_CAP
  AZK_REWARD_SHAPING_ANNEAL
  AZK_REWARD_SHAPING_ANNEAL_INITIAL
  AZK_REWARD_SHAPING_ANNEAL_FINAL
  AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES
  AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES
  AZK_TRAINER_SHAPED_REWARD_ANNEAL
  AZK_TRAINER_SHAPED_REWARD_ANNEAL_START_EPOCH
  AZK_TRAINER_SHAPED_REWARD_ANNEAL_END_EPOCH
  AZK_REWARD_LEADER_DELTA_WEIGHT
  AZK_REWARD_BOARD_DELTA_WEIGHT
  AZK_REWARD_NOOP_PENALTY
  AZK_TRUNCATION_BOARD_EDGE_WEIGHT
  AZK_REWARD_UNTAPPED_IKZ_WEIGHT
  AZK_PORTAL_GP_BONUS
  AZK_PORTAL_OUTCOME_BONUS
  AZK_EARLY_TEMPO_BONUS
  AZK_EARLY_TEMPO_CAP
  AZK_EARLY_TEMPO_TURNS
  AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES
  AZK_DMG_MITIGATION_BONUS
  AZK_DMG_MITIGATION_CAP
  AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP
  AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP
  AZK_GENERATED_IKZ_CONVERSION_BONUS
  AZK_GENERATED_IKZ_CONVERSION_STEP_CAP
  AZK_TEMP_CHARGE_REALIZATION_BONUS
  AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE
  AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP
  AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS
  AZK_DRAFT_VBOOT_COEF
  AZK_DRAFT_SIBDIFF_COEF
  AZK_DRAFT_SIBDIFF_CAP
  AZK_DRAFT_AUX_ANNEAL
  AZK_LEADER_TERMINAL_CREDIT_COEF
  AZK_LEADER_TERMINAL_CREDIT_CLIP
  AZK_LEADER_TERMINAL_CREDIT_GATE_CODES
  AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL
  AZK_LEADER_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS
)

require_hash() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$CAMPAIGN] $label hash mismatch: $actual" >&2
    exit 1
  fi
}

latest_jsonl() {
  local tag=$1
  find experiments/runlogs -maxdepth 1 -type f -name "${tag}_*.jsonl" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

cleanup_training_processes() {
  local tag=$1
  local pids=()
  mapfile -t pids < <(
    pgrep -f -- "python/src/train.py.*--tag ${tag}([[:space:]]|$)" 2>/dev/null || true
  )
  if (( ${#pids[@]} == 0 )); then
    return
  fi
  echo "[$CAMPAIGN] cleaning residual $tag processes: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 2
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

run_segment() {
  local arm=$1 segment=$2 resume_model=$3 tag=$4 log=$5
  local start_epoch=$6 expected_episodes=$7 stop_epoch=$8 initial=$9
  local arm_league=${10} snapshot_dir=${11}
  local trainer_anneal=0
  if [[ "$arm" == zero_tail ]]; then
    trainer_anneal=1
  fi

  local env_cmd=(env)
  local name
  for name in "${persisted_env_vars[@]}"; do
    env_cmd+=(-u "$name")
  done
  env_cmd+=(
    -u AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV
    -u AZK_RESUME_KEEP_CURRENT_REWARD_ENV
    -u AZK_RESUME_COMPLETED_EPISODES
    -u AZK_RESUME_ALLOW_SCHEDULE_REWIND
    AZK_XGATE_MASK=1
    AZK_PFSP=1
    AZK_DRAFT_REF_SEAT_PROB=0
    AZK_RESUME_ALLOW_BINDING_MISMATCH=1
    AZK_RESUME_ALLOW_SOURCE_DRIFT=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONUNBUFFERED=1
    PYTHONPATH=build/python/src:python/src
  )
  if (( initial )); then
    env_cmd+=(
      AZK_REWARD_LEADER_DELTA_WEIGHT=1.25
      AZK_REWARD_BOARD_DELTA_WEIGHT=0.35
      AZK_REWARD_NOOP_PENALTY=0.02
      AZK_TRUNCATION_BOARD_EDGE_WEIGHT=0.45
      AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0.15
      AZK_REWARD_SHAPING_ANNEAL=1
      AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0
      AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15
      AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12
      AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40
      AZK_TRAINER_SHAPED_REWARD_ANNEAL="$trainer_anneal"
      AZK_TRAINER_SHAPED_REWARD_ANNEAL_START_EPOCH="$ANNEAL_START_EPOCH"
      AZK_TRAINER_SHAPED_REWARD_ANNEAL_END_EPOCH="$ANNEAL_END_EPOCH"
      AZK_PORTAL_GP_BONUS=0.3
      AZK_PORTAL_OUTCOME_BONUS=0
      AZK_EARLY_TEMPO_BONUS=0.1
      AZK_EARLY_TEMPO_CAP=4
      AZK_EARLY_TEMPO_TURNS=2
      AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES=1
      AZK_DMG_MITIGATION_BONUS=0.15
      AZK_DMG_MITIGATION_CAP=10
      AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0
      AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP=6
      AZK_GENERATED_IKZ_CONVERSION_BONUS=0
      AZK_GENERATED_IKZ_CONVERSION_STEP_CAP=4
      AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08
      AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025
      AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4
      AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0
      AZK_DRAFT_VBOOT_COEF=0
      AZK_DRAFT_SIBDIFF_COEF=0
      AZK_DRAFT_SIBDIFF_CAP=0.05
      AZK_DRAFT_AUX_ANNEAL=1
      AZK_LEADER_TERMINAL_CREDIT_COEF=0
      AZK_LEADER_TERMINAL_CREDIT_CLIP=0.2
      "AZK_LEADER_TERMINAL_CREDIT_GATE_CODES=AZK01-122,STT04-002"
      AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL=4
      AZK_LEADER_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS=10
    )
  fi

  local restart_args=()
  if (( initial )); then
    restart_args+=(--resume-restart-lr-schedule)
  fi
  touch "$log"
  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm/$segment from p$start_epoch"
  "${env_cmd[@]}" \
  .venv/bin/python python/src/train.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --resume-checkpoint "$resume_model" \
    --resume-load-optimizer "${restart_args[@]}" \
    --no-resume-auto-reset-critic \
    --jsonl-log experiments/runlogs --tag "$tag" \
    --train.seed 42 --train.learning_rate "$PEAK_LR" \
    --train.ent_coef 0.002 \
    --train.ent_coef_anneal_initial 0.002 \
    --train.ent_coef_anneal_final 0.002 \
    --train.total_timesteps "$TOTAL_TIMESTEPS" \
    --train.checkpoint_interval 100 \
    --env.deck_snapshot_every 100 \
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
    > >(tee "$log") 2>&1 &
  local train_pid=$!

  local startup_checked=0 startup_failed=0 split_stopped=0 guard_failed=0
  local run_dir="" last_guard_count=0 tail_label_checked=0
  while kill -0 "$train_pid" 2>/dev/null; do
    if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$log"; then
      local schedule_ok=1
      if (( initial )); then
        grep -q 'remaining_epochs=2930' "$log" || schedule_ok=0
      else
        grep -q '\[resume\] applied saved schedule env vars' "$log" || schedule_ok=0
        grep -q '\[resume\] applied saved reward env vars' "$log" || schedule_ok=0
        if grep -q 'restarted LR schedule' "$log"; then
          schedule_ok=0
        fi
      fi
      if ! grep -q "total_epochs=$TARGET_EPOCH" "$log" ||
         ! grep -q "AZK_RESUME_COMPLETED_EPISODES=$expected_episodes" "$log" ||
         ! grep -q "epoch=$start_epoch" "$log" ||
         (( ! schedule_ok )); then
        echo "[$CAMPAIGN] $arm/$segment startup invariant failure" \
          | tee "$RESULT_ROOT/$arm/${segment}_STARTUP_FAILED"
        startup_failed=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
      if [[ "$arm" == zero_tail ]] &&
         ! grep -q "\[trainer-shaped-reward\] enabled: start_epoch=$ANNEAL_START_EPOCH, end_epoch=$ANNEAL_END_EPOCH" "$log"; then
        echo "[$CAMPAIGN] $arm/$segment trainer anneal was not activated" \
          | tee "$RESULT_ROOT/$arm/${segment}_STARTUP_FAILED"
        startup_failed=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
      startup_checked=1
      touch "$RESULT_ROOT/$arm/${segment}_STARTUP_OK"
      echo "[$CAMPAIGN] $arm/$segment startup invariants passed"
    fi

    local jsonl
    jsonl=$(latest_jsonl "$tag")
    if [[ -n "$jsonl" ]]; then
      local health
      if ! health=$(.venv/bin/python - "$jsonl" "$arm" \
          "$ANNEAL_START_EPOCH" "$ANNEAL_END_EPOCH" <<'PY'
import json
import statistics
import sys

path, arm = sys.argv[1], sys.argv[2]
start, end = int(sys.argv[3]), int(sys.argv[4])
rows = []
with open(path, encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("epoch"), (int, float)) and isinstance(row.get("SPS"), (int, float)):
            rows.append(row)

def multiplier(epoch):
    if arm != "zero_tail" or epoch <= start:
        return 1.0
    if epoch >= end:
        return 0.0
    return (end - epoch) / (end - start)

mismatches = 0
for row in rows:
    epoch = int(row["epoch"])
    actual = row.get("environment/trainer_shaped_reward_multiplier")
    update = row.get("environment/trainer_shaped_reward_update")
    if not isinstance(actual, (int, float)) or abs(float(actual) - multiplier(epoch)) > 1e-8:
        mismatches += 1
    if not isinstance(update, (int, float)) or int(update) != epoch:
        mismatches += 1
    native = row.get("environment/reward_shaping_scale")
    if isinstance(native, (int, float)) and abs(float(native) - 0.15) > 1e-6:
        mismatches += 1
    effective = row.get("environment/effective_reward_shaping_scale")
    if isinstance(effective, (int, float)) and abs(float(effective) - 0.15 * multiplier(epoch)) > 1e-6:
        mismatches += 1

values = [float(row["SPS"]) for row in rows]
previous = statistics.median(values[-40:-20]) if len(values) >= 40 else 0.0
current = statistics.median(values[-20:]) if len(values) >= 40 else 0.0
tail_labels = max(
    (float(row.get("losses/win_prob_aux_labeled_rows", 0.0)) for row in rows if int(row["epoch"]) >= end),
    default=0.0,
)
leader_examples = sum(float(row.get("losses/leader_credit_examples", 0.0)) for row in rows)
last_epoch = int(rows[-1]["epoch"]) if rows else 0
print(len(rows), last_epoch, previous, current, mismatches, tail_labels, leader_examples)
PY
      ); then
        echo "[$CAMPAIGN] $arm/$segment health parser failed" \
          | tee "$RESULT_ROOT/$arm/${segment}_HEALTH_FAILED"
        guard_failed=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
      local row_count last_epoch previous_median current_median mismatches tail_labels leader_examples
      read -r row_count last_epoch previous_median current_median mismatches tail_labels leader_examples <<< "$health"
      if (( mismatches > 0 )) || awk -v n="$leader_examples" 'BEGIN { exit !(n > 0) }'; then
        printf 'row_count=%s\nlast_epoch=%s\nschedule_mismatches=%s\nleader_examples=%s\n' \
          "$row_count" "$last_epoch" "$mismatches" "$leader_examples" \
          | tee "$RESULT_ROOT/$arm/${segment}_SIGNAL_GUARD_FAILED"
        guard_failed=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
      if (( row_count >= 40 && row_count > last_guard_count )); then
        last_guard_count=$row_count
        if awk -v a="$previous_median" -v b="$current_median" -v t="$SPS_HARD_FLOOR" \
          'BEGIN { exit !((a < t) && (b < t)) }'; then
          printf 'previous_median=%s\ncurrent_median=%s\nthreshold=%s\n' \
            "$previous_median" "$current_median" "$SPS_HARD_FLOOR" \
            | tee "$RESULT_ROOT/$arm/${segment}_SPS_GUARD_FAILED"
          guard_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
      fi
      if [[ "$arm" == zero_tail ]] && (( start_epoch >= ANNEAL_END_EPOCH )) &&
         (( row_count >= 60 && ! tail_label_checked )); then
        if ! awk -v n="$tail_labels" 'BEGIN { exit !(n > 0) }'; then
          echo "[$CAMPAIGN] $arm/$segment terminal labels stayed empty for 60 zero-tail updates" \
            | tee "$RESULT_ROOT/$arm/${segment}_TERMINAL_LABEL_GUARD_FAILED"
          guard_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
        tail_label_checked=1
        touch "$RESULT_ROOT/$arm/${segment}_TERMINAL_LABEL_OK"
      fi
    fi

    if [[ "$stop_epoch" -lt "$TARGET_EPOCH" ]]; then
      if [[ -z "$run_dir" ]]; then
        run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
      fi
      local split_model=""
      if [[ -n "$run_dir" ]]; then
        split_model="$run_dir/model_azuki_local_$(printf '%06d' "$stop_epoch").pt"
      fi
      if [[ -n "$split_model" ]] && [[ -f "$split_model" ]] &&
         [[ -f "$run_dir/trainer_state_$(printf '%06d' "$stop_epoch").pt" ]] &&
         [[ -f "$split_model.meta.json" ]] &&
         jq -e ".history[-1].epoch == $stop_epoch and .history[-1].event == \"checkpoint_ingested\"" \
           "$arm_league/league_state.json" >/dev/null; then
        split_stopped=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
    fi
    sleep 15
  done

  set +e
  wait "$train_pid"
  local train_status=$?
  set -e
  cleanup_training_processes "$tag"
  if (( guard_failed )); then
    return 2
  fi
  if (( startup_failed || ! startup_checked )); then
    return 1
  fi
  if [[ "$stop_epoch" -lt "$TARGET_EPOCH" ]]; then
    if (( ! split_stopped )); then
      echo "[$CAMPAIGN] $arm/$segment exited before atomic p$stop_epoch" >&2
      return 1
    fi
  elif (( train_status != 0 )); then
    echo "[$CAMPAIGN] $arm/$segment exited with status $train_status" >&2
    return "$train_status"
  fi

  if [[ -z "$run_dir" ]]; then
    run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
  fi
  local checkpoint trainer_state jsonl
  checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$stop_epoch").pt"
  trainer_state="$run_dir/trainer_state_$(printf '%06d' "$stop_epoch").pt"
  jsonl=$(latest_jsonl "$tag")
  if [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]] || [[ -z "$jsonl" ]]; then
    echo "[$CAMPAIGN] $arm/$segment lacks its atomic output" >&2
    return 1
  fi
  SEGMENT_RUN_DIR=$run_dir
  SEGMENT_JSONL=$jsonl
}

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
if [[ ! -f "$RESULT_ROOT/runtime_sha256.txt" ]]; then
  sha256sum \
    "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_LEAGUE" "$PARENT_PROMOTION" \
    python/src/azk_puffer/trainer.py python/src/league_training.py python/src/train.py \
    python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
    python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
    build/python/src/binding*.so \
    train-ablation-1781126582/run_shaped_reward_tail45_v1.sh \
    > "$RESULT_ROOT/runtime_sha256.txt"
  git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
  printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\ntotal_timesteps=%s\npeak_lr=%s\nanneal_start=%s\nanneal_end=%s\n' \
    "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$TOTAL_TIMESTEPS" \
    "$PEAK_LR" "$ANNEAL_START_EPOCH" "$ANNEAL_END_EPOCH" \
    > "$RESULT_ROOT/campaign_config.txt"
fi

for arm in "${arms[@]}"; do
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/${CAMPAIGN}_${arm}"
  mkdir -p "$arm_results"
  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already complete"
    continue
  fi
  existing=0
  for segment in "${segment_names[@]}"; do
    tag="${CAMPAIGN}_${arm}_${segment}"
    if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null; then
      existing=1
    fi
  done
  if (( existing )) || [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
    echo "[$CAMPAIGN] existing incomplete state for $arm; refusing overwrite" >&2
    exit 1
  fi
  mkdir -p "$arm_league/opponents" "$snapshot_dir"
  cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
  cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"

  resume_model=$PARENT_MODEL
  start_epoch=$PARENT_EPOCH
  expected_episodes=$EXPECTED_PARENT_EPISODES
  for index in "${!segment_names[@]}"; do
    segment=${segment_names[$index]}
    stop_epoch=${segment_ends[$index]}
    tag="${CAMPAIGN}_${arm}_${segment}"
    initial=0
    if (( index == 0 )); then
      initial=1
    fi
    run_segment "$arm" "$segment" "$resume_model" "$tag" \
      "$arm_results/train.${segment}.live.log" "$start_epoch" \
      "$expected_episodes" "$stop_epoch" "$initial" "$arm_league" "$snapshot_dir"
    run_dir=$SEGMENT_RUN_DIR
    jsonl=$SEGMENT_JSONL
    cp -a "$jsonl" "$arm_results/train.${segment}.jsonl"
    printf '%s\n' "$run_dir" > "$arm_results/${segment}_run_dir.txt"
    resume_model="$run_dir/model_azuki_local_$(printf '%06d' "$stop_epoch").pt"
    metadata="$resume_model.meta.json"
    expected_episodes=$(jq -r '.env_completed_episodes' "$metadata")
    if ! [[ "$expected_episodes" =~ ^[0-9]+$ ]] ||
       (( expected_episodes < EXPECTED_PARENT_EPISODES )); then
      echo "[$CAMPAIGN] invalid $arm/$segment episode count: $expected_episodes" >&2
      exit 1
    fi
    sha256sum "$resume_model" \
      "$run_dir/trainer_state_$(printf '%06d' "$stop_epoch").pt" "$metadata" \
      "$arm_league/league_state.json" "$arm_league/league_state_promotion.json" \
      "$jsonl" > "$arm_results/${segment}_output_sha256.txt"
    start_epoch=$stop_epoch
    touch "$arm_results/${segment}_DONE"
  done

  .venv/bin/python - \
    "$arm" "$arm_results" "$PEAK_LR" "$ANNEAL_START_EPOCH" \
    "$ANNEAL_END_EPOCH" "$PARENT_EPOCH" "$TARGET_EPOCH" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
root = Path(sys.argv[2])
peak_lr = float(sys.argv[3])
start = int(sys.argv[4])
end = int(sys.argv[5])
parent = int(sys.argv[6])
target = int(sys.argv[7])
enabled = arm == "zero_tail"
segments = (
    ("to6000", parent + 1, 6000),
    ("to6900", 6001, 6900),
    ("to7800", 6901, target),
)

rows = {}
for name, lower, upper in segments:
    path = root / f"train.{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = row.get("epoch")
            if isinstance(epoch, (int, float)) and lower <= int(epoch) <= upper:
                rows[int(epoch)] = row
expected_epochs = set(range(parent + 1, target + 1))
if set(rows) != expected_epochs:
    raise RuntimeError(
        f"epoch mismatch: missing={sorted(expected_epochs - set(rows))[:10]} "
        f"extra={sorted(set(rows) - expected_epochs)[:10]}"
    )

def multiplier(epoch: int) -> float:
    if not enabled or epoch <= start:
        return 1.0
    if epoch >= end:
        return 0.0
    return (end - epoch) / (end - start)

native_scale_samples = 0
effective_scale_samples = 0
max_lr_error = 0.0
for epoch, row in rows.items():
    expected_multiplier = multiplier(epoch)
    actual = row.get("environment/trainer_shaped_reward_multiplier")
    update = row.get("environment/trainer_shaped_reward_update")
    if not isinstance(actual, (int, float)) or abs(float(actual) - expected_multiplier) > 1e-8:
        raise RuntimeError(f"multiplier mismatch at p{epoch}: {actual} != {expected_multiplier}")
    if not isinstance(update, (int, float)) or int(update) != epoch:
        raise RuntimeError(f"absolute update mismatch at p{epoch}: {update}")
    native = row.get("environment/reward_shaping_scale")
    if isinstance(native, (int, float)):
        native_scale_samples += 1
        if abs(float(native) - 0.15) > 1e-6:
            raise RuntimeError(f"native shaping scale mismatch at p{epoch}: {native}")
    effective = row.get("environment/effective_reward_shaping_scale")
    if isinstance(effective, (int, float)):
        effective_scale_samples += 1
        expected_effective = 0.15 * expected_multiplier
        if abs(float(effective) - expected_effective) > 1e-6:
            raise RuntimeError(
                f"effective shaping scale mismatch at p{epoch}: {effective} != {expected_effective}"
            )
    observed_lr = float(row["learning_rate"])
    progress = epoch - parent
    expected_lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress / (target - parent)))
    max_lr_error = max(max_lr_error, abs(observed_lr - expected_lr))
if max_lr_error > 1e-11:
    raise RuntimeError(f"LR schedule mismatch: max_abs_error={max_lr_error}")

run_dirs = {
    name: Path((root / f"{name}_run_dir.txt").read_text(encoding="utf-8").strip())
    for name, _, _ in segments
}
checkpoint_locations = {
    6000: run_dirs["to6000"],
    6900: run_dirs["to6900"],
    7400: run_dirs["to7800"],
    7800: run_dirs["to7800"],
}
checkpoint_multipliers = {}
checkpoint_manifest = {}
for epoch, directory in checkpoint_locations.items():
    model = directory / f"model_azuki_local_{epoch:06d}.pt"
    trainer = directory / f"trainer_state_{epoch:06d}.pt"
    metadata_path = model.with_suffix(model.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = metadata.get("trainer_shaped_reward_schedule")
    if not isinstance(state, dict):
        raise RuntimeError(f"missing shaped-reward schedule metadata at p{epoch}")
    if bool(state.get("enabled")) != enabled:
        raise RuntimeError(f"schedule enabled mismatch at p{epoch}: {state}")
    if int(state.get("absolute_update", -1)) != epoch:
        raise RuntimeError(f"schedule update mismatch at p{epoch}: {state}")
    expected_multiplier = multiplier(epoch)
    if abs(float(state.get("multiplier", -1.0)) - expected_multiplier) > 1e-8:
        raise RuntimeError(f"schedule multiplier mismatch at p{epoch}: {state}")
    checkpoint_multipliers[str(epoch)] = float(state["multiplier"])
    checkpoint_manifest[str(epoch)] = {
        "model": str(model),
        "trainer_state": str(trainer),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "trainer_state_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
    }

final_metadata = json.loads(
    (checkpoint_locations[7800] / "model_azuki_local_007800.pt.meta.json").read_text(encoding="utf-8")
)
reward_env = final_metadata["resume_config_fingerprint"]["reward_env"]
required_reward_env = {
    "AZK_EARLY_TEMPO_BONUS": "0.1",
    "AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES": "1",
    "AZK_DMG_MITIGATION_BONUS": "0.15",
    "AZK_TEMP_CHARGE_REALIZATION_BONUS": "0.08",
    "AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE": "0.025",
    "AZK_LEADER_TERMINAL_CREDIT_COEF": "0",
}
for key, value in required_reward_env.items():
    if reward_env.get(key) != value:
        raise RuntimeError(f"reward fingerprint mismatch: {key}={reward_env.get(key)!r}, expected={value!r}")

tail_rows = [row for epoch, row in rows.items() if epoch >= end]
tail_labels_max = max(
    (float(row.get("losses/win_prob_aux_labeled_rows", 0.0)) for row in tail_rows),
    default=0.0,
)
if tail_labels_max <= 0.0:
    raise RuntimeError("terminal labels were absent throughout the zero-tail range")
leader_examples = sum(float(row.get("losses/leader_credit_examples", 0.0)) for row in rows.values())
if leader_examples != 0.0:
    raise RuntimeError(f"leader credit was unexpectedly active: {leader_examples}")

excluded = set()
for _, lower, _ in segments:
    excluded.update(range(lower, lower + 20))
steady_rows = [row for epoch, row in sorted(rows.items()) if epoch not in excluded]
sps = [float(row["SPS"]) for row in steady_rows]
kl = [float(row["losses/approx_kl"]) for row in rows.values() if isinstance(row.get("losses/approx_kl"), (int, float))]
summary = {
    "arm": arm,
    "metric_rows": len(rows),
    "steady_sps_count": len(sps),
    "steady_sps_median": statistics.median(sps),
    "steady_sps_p10": sorted(sps)[max(0, int(0.10 * len(sps)) - 1)],
    "steady_sps_last100_median": statistics.median(sps[-100:]),
    "approx_kl_median": statistics.median(kl),
    "approx_kl_p90": sorted(kl)[max(0, int(0.90 * len(kl)) - 1)],
    "max_lr_error": max_lr_error,
    "native_scale_samples": native_scale_samples,
    "effective_scale_samples": effective_scale_samples,
    "checkpoint_multipliers": checkpoint_multipliers,
    "tail_terminal_labels_max": tail_labels_max,
    "leader_credit_examples": leader_examples,
    "checkpoints": checkpoint_manifest,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
(root / "checkpoints.json").write_text(json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
  touch "$arm_results/TRAIN_DONE"
done

.venv/bin/python - "$RESULT_ROOT" "$SPS_RELATIVE_FLOOR" "$SPS_HARD_FLOOR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
relative_floor = float(sys.argv[2])
hard_floor = float(sys.argv[3])
control = json.loads((root / "fixed_floor" / "summary.json").read_text(encoding="utf-8"))
treatment = json.loads((root / "zero_tail" / "summary.json").read_text(encoding="utf-8"))
ratio = treatment["steady_sps_median"] / control["steady_sps_median"]
p6000_identical = (
    control["checkpoints"]["6000"]["model_sha256"]
    == treatment["checkpoints"]["6000"]["model_sha256"]
)
passed = (
    ratio >= relative_floor
    and control["steady_sps_median"] >= hard_floor
    and treatment["steady_sps_median"] >= hard_floor
    and treatment["checkpoint_multipliers"] == {
        "6000": 1.0,
        "6900": 0.0,
        "7400": 0.0,
        "7800": 0.0,
    }
    and treatment["tail_terminal_labels_max"] > 0.0
)
payload = {
    "status": "pass" if passed else "fail",
    "fixed_floor_sps_median": control["steady_sps_median"],
    "zero_tail_sps_median": treatment["steady_sps_median"],
    "sps_ratio": ratio,
    "relative_floor": relative_floor,
    "hard_floor": hard_floor,
    "p6000_models_identical": p6000_identical,
    "zero_tail_checkpoint_multipliers": treatment["checkpoint_multipliers"],
    "zero_tail_terminal_labels_max": treatment["tail_terminal_labels_max"],
}
(root / "training_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if not passed:
    raise SystemExit(2)
PY

touch "$RESULT_ROOT/TRAINING_DONE"
echo "[$CAMPAIGN] TRAINING_DONE"
