#!/usr/bin/env bash
# Shared-fork p4870->p4910 smoke for late shaped-reward removal and resume.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${SHAPED_REWARD_LATEZERO_SMOKE_CAMPAIGN:-shaped_reward_latezero20_smoke_v1}
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
FORK_EPOCH=4880
SPLIT_EPOCH=4890
TARGET_EPOCH=4910
TOTAL_TIMESTEPS=75417600
ANNEAL_START_EPOCH=4880
ANNEAL_END_EPOCH=4890
PEAK_LR=${SHAPED_REWARD_LATEZERO_SMOKE_PEAK_LR:-0.000003}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235

arms=(fixed_floor late_zero)

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
  AZK_DRAFT_TERMINAL_CREDIT_COEF
  AZK_DRAFT_TERMINAL_CREDIT_CLIP
  AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL
  AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE
  AZK_DRAFT_TERMINAL_CREDIT_SEED
  AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS
  AZK_DRAFT_TERMINAL_CREDIT_GRAD_PROBE
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
  local start_epoch=$6 expected_episodes=$7 stop_epoch=$8 restart_lr=$9
  local arm_league=${10}
  local snapshot_dir=${11}
  local schedule_override=${12}

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
    -u AZK_DRAFT_REF_OPPONENT_ONLY
    AZK_XGATE_MASK=1
    AZK_PFSP=1
    AZK_PFSP_POWER=2.0
    AZK_DRAFT_REF_SEAT_PROB=0
    AZK_RESUME_ALLOW_BINDING_MISMATCH=1
    AZK_RESUME_ALLOW_SOURCE_DRIFT=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONUNBUFFERED=1
    PYTHONPATH=build/python/src:python/src
  )
  if (( restart_lr )); then
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
      AZK_TRAINER_SHAPED_REWARD_ANNEAL=0
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
      AZK_DRAFT_TERMINAL_CREDIT_COEF=0
      AZK_DRAFT_TERMINAL_CREDIT_CLIP=0.2
      AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL=4
      AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE=512
      AZK_DRAFT_TERMINAL_CREDIT_SEED=42
      AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS=10
      AZK_DRAFT_TERMINAL_CREDIT_GRAD_PROBE=0
    )
  elif (( schedule_override )); then
    env_cmd+=(
      AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV=1
      AZK_REWARD_SHAPING_ANNEAL=1
      AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0
      AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15
      AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12
      AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40
      AZK_TRAINER_SHAPED_REWARD_ANNEAL=1
      AZK_TRAINER_SHAPED_REWARD_ANNEAL_START_EPOCH="$ANNEAL_START_EPOCH"
      AZK_TRAINER_SHAPED_REWARD_ANNEAL_END_EPOCH="$ANNEAL_END_EPOCH"
    )
  fi

  local restart_args=()
  if (( restart_lr )); then
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
    --train.checkpoint_interval 5 \
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
    > >(tee "$log") 2>&1 &
  local train_pid=$!

  local startup_checked=0
  local startup_failed=0
  local split_stopped=0
  local run_dir=""
  while kill -0 "$train_pid" 2>/dev/null; do
    if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$log"; then
      local schedule_ok=1
      if (( restart_lr )); then
        grep -q 'remaining_epochs=40' "$log" || schedule_ok=0
        grep -q 'restarted LR schedule' "$log" || schedule_ok=0
      elif (( schedule_override )); then
        grep -q '\[resume\] keeping current schedule env vars per AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV=1' "$log" || schedule_ok=0
        grep -q '\[resume\] applied saved reward env vars' "$log" || schedule_ok=0
        if grep -q 'restarted LR schedule' "$log"; then
          schedule_ok=0
        fi
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
      if [[ "$arm" == late_zero ]] &&
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

    if [[ "$stop_epoch" -lt "$TARGET_EPOCH" ]]; then
      if [[ -z "$run_dir" ]]; then
        run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
      fi
      local split_model=""
      if [[ -n "$run_dir" ]]; then
        split_model="$run_dir/model_azuki_local_$(printf '%06d' "$stop_epoch").pt"
      fi
      if [[ -n "$split_model" ]] &&
         [[ -f "$split_model" ]] &&
         [[ -f "$run_dir/trainer_state_$(printf '%06d' "$stop_epoch").pt" ]] &&
         [[ -f "$split_model.meta.json" ]] &&
         jq -e ".history[-1].epoch == $stop_epoch and .history[-1].event == \"checkpoint_ingested\"" \
           "$arm_league/league_state.json" >/dev/null; then
        split_stopped=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
    fi
    sleep 2
  done

  set +e
  wait "$train_pid"
  local train_status=$?
  set -e
  cleanup_training_processes "$tag"
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
    train-ablation-1781126582/run_shaped_reward_latezero20_smoke_v1.sh \
    > "$RESULT_ROOT/runtime_sha256.txt"
  git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
fi

shared_results="$RESULT_ROOT/shared"
shared_league="$LEAGUE_ROOT/shared"
shared_snapshot="experiments/abl_snapshots/${CAMPAIGN}_shared"
shared_tag="${CAMPAIGN}_shared_to4880"
mkdir -p "$shared_results"
if compgen -G "experiments/azuki_local_${shared_tag}_*" >/dev/null ||
   [[ -e "$shared_league" ]] || [[ -e "$shared_snapshot" ]]; then
  echo "[$CAMPAIGN] existing shared smoke state; refusing overwrite" >&2
  exit 1
fi
mkdir -p "$shared_league/opponents" "$shared_snapshot"
cp -a "$PARENT_LEAGUE" "$shared_league/league_state.json"
cp -a "$PARENT_PROMOTION" "$shared_league/league_state_promotion.json"
run_segment shared to4880 "$PARENT_MODEL" "$shared_tag" \
  "$shared_results/train.live.log" "$PARENT_EPOCH" \
  "$EXPECTED_PARENT_EPISODES" "$FORK_EPOCH" 1 "$shared_league" \
  "$shared_snapshot" 0
shared_run_dir=$SEGMENT_RUN_DIR
shared_jsonl=$SEGMENT_JSONL
shared_model="$shared_run_dir/model_azuki_local_$(printf '%06d' "$FORK_EPOCH").pt"
shared_trainer="$shared_run_dir/trainer_state_$(printf '%06d' "$FORK_EPOCH").pt"
shared_metadata="$shared_model.meta.json"
shared_episodes=$(jq -r '.env_completed_episodes' "$shared_metadata")
cp -a "$shared_jsonl" "$shared_results/train.jsonl"
printf '%s\n' "$shared_run_dir" > "$shared_results/run_dir.txt"

fork_root="$RESULT_ROOT/fork"
for arm in "${arms[@]}"; do
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/${CAMPAIGN}_${arm}"
  mkdir -p "$arm_results" "$fork_root/$arm" "$arm_league" "$snapshot_dir"
  cp -a "$shared_model" "$shared_trainer" "$shared_metadata" "$fork_root/$arm/"
  cp -a "$shared_league/." "$arm_league/"
done
cmp -s "$fork_root/fixed_floor/$(basename "$shared_model")" \
  "$fork_root/late_zero/$(basename "$shared_model")"
cmp -s "$fork_root/fixed_floor/$(basename "$shared_trainer")" \
  "$fork_root/late_zero/$(basename "$shared_trainer")"
diff -qr "$LEAGUE_ROOT/fixed_floor" "$LEAGUE_ROOT/late_zero" >/dev/null
touch "$fork_root/FORK_IDENTICAL"

for arm in "${arms[@]}"; do
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/${CAMPAIGN}_${arm}"
  fork_model="$fork_root/$arm/$(basename "$shared_model")"
  schedule_override=0
  if [[ "$arm" == late_zero ]]; then
    schedule_override=1
  fi
  part1_tag="${CAMPAIGN}_${arm}_to4890"
  run_segment "$arm" to4890 "$fork_model" "$part1_tag" \
    "$arm_results/train.part1.live.log" "$FORK_EPOCH" "$shared_episodes" \
    "$SPLIT_EPOCH" 0 "$arm_league" "$snapshot_dir" "$schedule_override"
  part1_run_dir=$SEGMENT_RUN_DIR
  part1_jsonl=$SEGMENT_JSONL
  split_model="$part1_run_dir/model_azuki_local_$(printf '%06d' "$SPLIT_EPOCH").pt"
  split_episodes=$(jq -r '.env_completed_episodes' "$split_model.meta.json")
  cp -a "$part1_jsonl" "$arm_results/train.part1.jsonl"
  printf '%s\n' "$part1_run_dir" > "$arm_results/part1_run_dir.txt"
done

for arm in "${arms[@]}"; do
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/${CAMPAIGN}_${arm}"
  part1_run_dir=$(<"$arm_results/part1_run_dir.txt")
  split_model="$part1_run_dir/model_azuki_local_$(printf '%06d' "$SPLIT_EPOCH").pt"
  split_episodes=$(jq -r '.env_completed_episodes' "$split_model.meta.json")
  part2_tag="${CAMPAIGN}_${arm}_to4910"
  run_segment "$arm" to4910 "$split_model" "$part2_tag" \
    "$arm_results/train.part2.live.log" "$SPLIT_EPOCH" "$split_episodes" \
    "$TARGET_EPOCH" 0 "$arm_league" "$snapshot_dir" 0
  part2_run_dir=$SEGMENT_RUN_DIR
  part2_jsonl=$SEGMENT_JSONL
  cp -a "$part2_jsonl" "$arm_results/train.part2.jsonl"
  printf '%s\n' "$part2_run_dir" > "$arm_results/part2_run_dir.txt"
done

for arm in "${arms[@]}"; do
  arm_results="$RESULT_ROOT/$arm"
  .venv/bin/python - \
    "$arm" "$arm_results" "$fork_root/$arm" "$PEAK_LR" "$PARENT_EPOCH" \
    "$FORK_EPOCH" "$ANNEAL_START_EPOCH" "$ANNEAL_END_EPOCH" "$TARGET_EPOCH" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
root = Path(sys.argv[2])
fork_root = Path(sys.argv[3])
peak_lr = float(sys.argv[4])
parent = int(sys.argv[5])
fork = int(sys.argv[6])
start = int(sys.argv[7])
end = int(sys.argv[8])
target = int(sys.argv[9])
enabled = arm == "late_zero"
part1_dir = Path((root / "part1_run_dir.txt").read_text(encoding="utf-8").strip())
part2_dir = Path((root / "part2_run_dir.txt").read_text(encoding="utf-8").strip())
rows = {}
for path in (root / "train.part1.jsonl", root / "train.part2.jsonl"):
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = row.get("epoch")
        if isinstance(epoch, (int, float)) and fork < int(epoch) <= target:
            rows[int(epoch)] = row
expected = set(range(fork + 1, target + 1))
if set(rows) != expected:
    raise RuntimeError(f"epoch mismatch: missing={sorted(expected - set(rows))}")

def multiplier(epoch: int) -> float:
    if not enabled or epoch <= start:
        return 1.0
    if epoch >= end:
        return 0.0
    return (end - epoch) / (end - start)

max_lr_error = 0.0
for epoch, row in rows.items():
    actual = row.get("environment/trainer_shaped_reward_multiplier")
    if not isinstance(actual, (int, float)) or abs(float(actual) - multiplier(epoch)) > 1e-8:
        raise RuntimeError(f"{arm} multiplier mismatch at p{epoch}: {actual}")
    if int(row.get("environment/trainer_shaped_reward_update", -1)) != epoch:
        raise RuntimeError(f"{arm} absolute update mismatch at p{epoch}")
    if abs(float(row.get("environment/reward_shaping_scale", 0.15)) - 0.15) > 1e-6:
        raise RuntimeError(f"{arm} native shaping mismatch at p{epoch}")
    expected_effective = 0.15 * multiplier(epoch)
    if abs(float(row.get("environment/effective_reward_shaping_scale", expected_effective)) - expected_effective) > 1e-6:
        raise RuntimeError(f"{arm} effective shaping mismatch at p{epoch}")
    expected_lr = peak_lr * 0.5 * (
        1.0 + math.cos(math.pi * (epoch - parent) / (target - parent))
    )
    max_lr_error = max(max_lr_error, abs(float(row["learning_rate"]) - expected_lr))
if max_lr_error > 1e-11:
    raise RuntimeError(f"{arm} LR mismatch: {max_lr_error}")

checkpoint_samples = {}
checkpoint_locations = {fork: fork_root, 4885: part1_dir, 4890: part1_dir, target: part2_dir}
for epoch, directory in checkpoint_locations.items():
    metadata = directory / f"model_azuki_local_{epoch:06d}.pt.meta.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    state = payload.get("trainer_shaped_reward_schedule")
    if not isinstance(state, dict):
        raise RuntimeError(f"missing schedule metadata at p{epoch}")
    expected_enabled = enabled and epoch > fork
    if bool(state.get("enabled")) != expected_enabled:
        raise RuntimeError(f"enabled mismatch at p{epoch}: {state}")
    if int(state.get("absolute_update", -1)) != epoch:
        raise RuntimeError(f"metadata update mismatch at p{epoch}: {state}")
    if abs(float(state.get("multiplier", -1.0)) - multiplier(epoch)) > 1e-8:
        raise RuntimeError(f"metadata multiplier mismatch at p{epoch}: {state}")
    checkpoint_samples[str(epoch)] = float(state["multiplier"])

post_zero = [row for epoch, row in rows.items() if epoch >= end]
labels_max = max(
    (float(row.get("losses/win_prob_aux_labeled_rows", 0.0)) for row in post_zero),
    default=0.0,
)
if labels_max <= 0.0:
    raise RuntimeError(f"{arm} had no terminal labels after p{end}")
leader_examples = sum(float(row.get("losses/leader_credit_examples", 0.0)) for row in rows.values())
draft_examples = sum(float(row.get("losses/draft_credit_examples", 0.0)) for row in rows.values())
if leader_examples != 0.0 or draft_examples != 0.0:
    raise RuntimeError(f"unexpected terminal draft credit: {leader_examples}, {draft_examples}")
steady_sps = [
    float(row["SPS"])
    for epoch, row in sorted(rows.items())
    if epoch not in range(fork + 1, fork + 6) and epoch not in range(end + 1, end + 6)
]
fork_model = fork_root / f"model_azuki_local_{fork:06d}.pt"
summary = {
    "arm": arm,
    "metric_rows": len(rows),
    "fork_model_sha256": hashlib.sha256(fork_model.read_bytes()).hexdigest(),
    "checkpoint_multipliers": checkpoint_samples,
    "first_post_resume_update": int(rows[end + 1]["environment/trainer_shaped_reward_update"]),
    "first_post_resume_multiplier": float(rows[end + 1]["environment/trainer_shaped_reward_multiplier"]),
    "post_zero_terminal_labels_max": labels_max,
    "steady_sps_count": len(steady_sps),
    "steady_sps_median": statistics.median(steady_sps),
    "max_lr_error": max_lr_error,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
done

.venv/bin/python - "$RESULT_ROOT" "$SPS_RELATIVE_FLOOR" "$SPS_HARD_FLOOR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
relative_floor = float(sys.argv[2])
hard_floor = float(sys.argv[3])
control = json.loads((root / "fixed_floor" / "summary.json").read_text(encoding="utf-8"))
treatment = json.loads((root / "late_zero" / "summary.json").read_text(encoding="utf-8"))
ratio = treatment["steady_sps_median"] / control["steady_sps_median"]
fork_identical = control["fork_model_sha256"] == treatment["fork_model_sha256"]
schedule_ok = treatment["checkpoint_multipliers"] == {
    "4880": 1.0,
    "4885": 0.5,
    "4890": 0.0,
    "4910": 0.0,
}
payload = {
    "status": "pass",
    "fork_identical": fork_identical,
    "fixed_floor_sps_median": control["steady_sps_median"],
    "late_zero_sps_median": treatment["steady_sps_median"],
    "sps_ratio": ratio,
    "relative_floor": relative_floor,
    "hard_floor": hard_floor,
    "schedule_ok": schedule_ok,
    "resume_restored_zero": treatment["first_post_resume_multiplier"] == 0.0,
    "terminal_labels_survived_zero": treatment["post_zero_terminal_labels_max"] > 0.0,
}
if not (
    fork_identical
    and ratio >= relative_floor
    and control["steady_sps_median"] >= hard_floor
    and treatment["steady_sps_median"] >= hard_floor
    and schedule_ok
    and payload["resume_restored_zero"]
    and payload["terminal_labels_survived_zero"]
):
    payload["status"] = "fail"
(root / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if payload["status"] != "pass":
    raise SystemExit(2)
PY

touch "$RESULT_ROOT/SMOKE_DONE"
echo "[$CAMPAIGN] SMOKE_DONE"
