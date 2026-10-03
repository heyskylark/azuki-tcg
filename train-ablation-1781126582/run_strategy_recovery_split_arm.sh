#!/usr/bin/env bash
# Train one strategy-recovery arm with the same planned p3300 process boundary
# used by the recovered control. The LR scheduler and episode-driven reward
# schedule continue from checkpoint state; neither is restarted at p3300.
set -euo pipefail

if (( $# != 5 )); then
  echo "usage: $0 CAMPAIGN ARM TEMPO_DEDUP PORTAL_GP PEAK_LR" >&2
  exit 2
fi

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=$1
ARM=$2
TEMPO_DEDUP=$3
PORTAL_GP=$4
PEAK_LR=$5
case "$ARM" in
  control|tempo_dedup|portal_gp_tail) ;;
  *) echo "unsupported arm: $ARM" >&2; exit 2 ;;
esac
if [[ "$TEMPO_DEDUP" != 0 && "$TEMPO_DEDUP" != 1 ]]; then
  echo "TEMPO_DEDUP must be 0 or 1" >&2
  exit 2
fi

RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
ARM_RESULTS="$RESULT_ROOT/$ARM"
ARM_LEAGUE="experiments/league/$CAMPAIGN/$ARM"
PARENT_DIR=experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308
PARENT_MODEL="$PARENT_DIR/model_azuki_local_002930.pt"
PARENT_TRAINER="$PARENT_DIR/trainer_state_002930.pt"
PARENT_METADATA="$PARENT_MODEL.meta.json"
PARENT_LEAGUE=experiments/league/promotionv2_archive45_final/league_state.json
PARENT_PROMOTION=experiments/league/promotionv2_archive45_final/league_state_promotion.json
EXPECTED_MODEL_SHA=7196734b2250ea1901e0e74e7bd693ebe6cf0918a5e51e5d231a3424b5d045d4
EXPECTED_TRAINER_SHA=0aa5da87f27801ed008c984ef64373874eec843e17558b1af2710d7e9b2bc5e1
EXPECTED_LEAGUE_SHA=7b1c8e5a932694d3370305d542231c63ca28ce8b6ee062563fa80d930675f707
EXPECTED_PROMOTION_SHA=9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a
EXPECTED_PARENT_EPISODES=242
SPLIT_EPOCH=3300
TARGET_EPOCH=3900
TOTAL_TIMESTEPS=59904000
SPS_HARD_FLOOR=1235
BASE_TAG="${CAMPAIGN}_${ARM}"
RESUME_TAG="${BASE_TAG}_resume3300"
SNAPSHOT_DIR="experiments/abl_snapshots/$BASE_TAG"

require_hash() {
  local path=$1 expected=$2 label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$BASE_TAG] $label hash mismatch: $actual" >&2
    exit 1
  fi
}

latest_jsonl() {
  local tag=$1
  find experiments/runlogs -maxdepth 1 -type f -name "${tag}_*.jsonl" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

cleanup_segment_processes() {
  local tag=$1
  local pids=()
  mapfile -t pids < <(
    pgrep -f -- "python/src/train.py.*--tag ${tag}([[:space:]]|$)" 2>/dev/null || true
  )
  if (( ${#pids[@]} == 0 )); then
    return
  fi

  echo "[$BASE_TAG] cleaning residual $tag processes: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 2
  local remaining=()
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      remaining+=("$pid")
    fi
  done
  if (( ${#remaining[@]} > 0 )); then
    kill -KILL "${remaining[@]}" 2>/dev/null || true
  fi
}

run_segment() {
  local segment=$1 resume_model=$2 tag=$3 log=$4
  local start_epoch=$5 expected_episodes=$6 stop_epoch=$7
  local restart_lr=$8
  local restart_args=()
  if [[ "$restart_lr" == 1 ]]; then
    restart_args+=(--resume-restart-lr-schedule)
  fi
  touch "$log"

  echo "[$BASE_TAG] $(date --iso-8601=seconds) $segment start from p$start_epoch"
  env \
  AZK_REWARD_SHAPING_ANNEAL=1 \
  AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
  AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 \
  AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
  AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 \
  AZK_PORTAL_GP_BONUS="$PORTAL_GP" \
  AZK_PORTAL_OUTCOME_BONUS=0 \
  AZK_XGATE_MASK=1 \
  AZK_PFSP=1 \
  AZK_EARLY_TEMPO_BONUS=0.1 \
  AZK_EARLY_TEMPO_CAP=4 \
  AZK_EARLY_TEMPO_TURNS=2 \
  AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES="$TEMPO_DEDUP" \
  AZK_DMG_MITIGATION_BONUS=0.15 \
  AZK_DMG_MITIGATION_CAP=10 \
  AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0 \
  AZK_GENERATED_IKZ_CONVERSION_BONUS=0 \
  AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
  AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
  AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4 \
  AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0 \
  AZK_DRAFT_REF_SEAT_PROB=0 \
  AZK_RESUME_ALLOW_BINDING_MISMATCH=1 \
  AZK_RESUME_ALLOW_SOURCE_DRIFT=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/train.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --resume-checkpoint "$resume_model" \
    --resume-load-optimizer \
    "${restart_args[@]}" \
    --no-resume-auto-reset-critic \
    --jsonl-log experiments/runlogs \
    --tag "$tag" \
    --train.seed 42 \
    --train.learning_rate "$PEAK_LR" \
    --train.ent_coef 0.002 \
    --train.ent_coef_anneal_initial 0.002 \
    --train.ent_coef_anneal_final 0.002 \
    --train.total_timesteps "$TOTAL_TIMESTEPS" \
    --train.checkpoint_interval 100 \
    --env.deck_snapshot_every 25 \
    --env.deck_snapshot_dir "$SNAPSHOT_DIR" \
    --league.state_path "$ARM_LEAGUE/league_state.json" \
    --league.opponent_dir "$ARM_LEAGUE/opponents" \
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
  local lr_checked=$restart_lr
  local guard_failed=0
  local split_stopped=0
  local last_guard_count=0
  local run_dir=""
  while kill -0 "$train_pid" 2>/dev/null; do
    if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$log"; then
      local restart_ok=1
      if [[ "$restart_lr" == 1 ]]; then
        grep -q 'remaining_epochs=970' "$log" || restart_ok=0
      elif grep -q 'restarted LR schedule' "$log"; then
        restart_ok=0
      fi
      if ! grep -q "total_epochs=$TARGET_EPOCH" "$log" ||
         ! grep -q "AZK_RESUME_COMPLETED_EPISODES=$expected_episodes" "$log" ||
         ! grep -q "epoch=$start_epoch" "$log" ||
         (( ! restart_ok )); then
        echo "[$BASE_TAG] $segment startup invariant failure" \
          | tee "$ARM_RESULTS/${segment}_STARTUP_FAILED"
        startup_failed=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
      startup_checked=1
      touch "$ARM_RESULTS/${segment}_STARTUP_OK"
      echo "[$BASE_TAG] $segment startup invariants passed"
    fi

    local jsonl
    jsonl=$(latest_jsonl "$tag")
    if [[ -n "$jsonl" ]]; then
      local guard
      guard=$(.venv/bin/python - "$jsonl" <<'PY'
import json
import statistics
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("SPS"), (int, float)):
            rows.append(row)
if len(rows) < 40:
    print(len(rows), 0.0, 0.0, rows[0]["learning_rate"] if rows else 0.0)
else:
    print(
        len(rows),
        statistics.median(float(row["SPS"]) for row in rows[-40:-20]),
        statistics.median(float(row["SPS"]) for row in rows[-20:]),
        rows[0]["learning_rate"],
    )
PY
      )
      local guard_count previous_median current_median first_lr
      read -r guard_count previous_median current_median first_lr <<< "$guard"
      if (( ! lr_checked && guard_count >= 1 )); then
        if ! awk -v lr="$first_lr" -v peak="$PEAK_LR" \
          'BEGIN { ratio=lr/peak; exit !((ratio > 0.50) && (ratio < 0.80)) }'; then
          echo "[$BASE_TAG] $segment LR continuity failure: first_lr=$first_lr" \
            | tee "$ARM_RESULTS/${segment}_LR_FAILED"
          startup_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
        lr_checked=1
        printf 'first_lr=%s\npeak_lr=%s\n' "$first_lr" "$PEAK_LR" \
          > "$ARM_RESULTS/${segment}_lr_check.txt"
        echo "[$BASE_TAG] $segment LR continuity passed: first_lr=$first_lr"
      fi
      if (( guard_count >= 40 && guard_count > last_guard_count )); then
        last_guard_count=$guard_count
        if awk -v a="$previous_median" -v b="$current_median" -v t="$SPS_HARD_FLOOR" \
          'BEGIN { exit !((a < t) && (b < t)) }'; then
          printf 'previous_median=%s\ncurrent_median=%s\nthreshold=%s\n' \
            "$previous_median" "$current_median" "$SPS_HARD_FLOOR" \
            | tee "$ARM_RESULTS/${segment}_SPS_GUARD_FAILED"
          guard_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
      fi
    fi

    if [[ "$stop_epoch" -lt "$TARGET_EPOCH" ]]; then
      if [[ -z "$run_dir" ]]; then
        run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
      fi
      if [[ -n "$run_dir" ]] &&
         [[ -f "$run_dir/model_azuki_local_$(printf '%06d' "$stop_epoch").pt" ]] &&
         [[ -f "$run_dir/trainer_state_$(printf '%06d' "$stop_epoch").pt" ]] &&
         [[ -f "$run_dir/model_azuki_local_$(printf '%06d' "$stop_epoch").pt.meta.json" ]] &&
         jq -e ".history[-1].epoch == $stop_epoch and .history[-1].event == \"checkpoint_ingested\"" \
           "$ARM_LEAGUE/league_state.json" >/dev/null; then
        split_stopped=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
    fi
    sleep 5
  done

  set +e
  wait "$train_pid"
  local train_status=$?
  set -e
  # An interrupt at the matched split can leave the multiprocessing
  # forkserver and workers reparented to init. They continue stepping envs
  # and can consume every CPU core during the resumed segment.
  cleanup_segment_processes "$tag"
  if (( guard_failed )); then
    return 2
  fi
  if (( startup_failed || ! startup_checked || ! lr_checked )); then
    return 1
  fi
  if [[ "$stop_epoch" -lt "$TARGET_EPOCH" ]]; then
    if (( ! split_stopped )); then
      echo "[$BASE_TAG] $segment exited before atomic p$stop_epoch" >&2
      return 1
    fi
  elif (( train_status != 0 )); then
    echo "[$BASE_TAG] $segment exited with status $train_status" >&2
    return "$train_status"
  fi

  if [[ -z "$run_dir" ]]; then
    run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
  fi
  local checkpoint trainer_state
  checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$stop_epoch").pt"
  trainer_state="$run_dir/trainer_state_$(printf '%06d' "$stop_epoch").pt"
  local jsonl
  jsonl=$(latest_jsonl "$tag")
  if [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]] || [[ -z "$jsonl" ]]; then
    echo "[$BASE_TAG] $segment lacks its atomic output" >&2
    return 1
  fi
  SEGMENT_RUN_DIR=$run_dir
  SEGMENT_JSONL=$jsonl
}

unset AZK_RESUME_COMPLETED_EPISODES AZK_RESUME_ALLOW_SCHEDULE_REWIND
unset AZK_DRAFT_REF_DECK_INDICES
require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == 2930 and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$ARM_RESULTS" experiments/runlogs
if [[ -f "$ARM_RESULTS/TRAIN_DONE" ]]; then
  echo "[$BASE_TAG] training already complete"
  exit 0
fi
sha256sum \
  python/src/tcg.h python/src/train.py \
  python/config/azuki_deckbuild_native_3090.ini build/python/src/binding*.so \
  train-ablation-1781126582/run_strategy_recovery_split_arm.sh \
  train-ablation-1781126582/run_strategy_recovery_ladder15_v1.sh \
  > "$ARM_RESULTS/split_runtime_sha256.txt"
if compgen -G "experiments/azuki_local_${BASE_TAG}_*" >/dev/null ||
   compgen -G "experiments/azuki_local_${RESUME_TAG}_*" >/dev/null ||
   [[ -e "$ARM_LEAGUE" ]] || [[ -e "$SNAPSHOT_DIR" ]]; then
  echo "[$BASE_TAG] existing incomplete split-arm state; refusing overwrite" >&2
  exit 1
fi

mkdir -p "$ARM_LEAGUE/opponents" "$SNAPSHOT_DIR"
cp -a "$PARENT_LEAGUE" "$ARM_LEAGUE/league_state.json"
cp -a "$PARENT_PROMOTION" "$ARM_LEAGUE/league_state_promotion.json"
run_segment initial "$PARENT_MODEL" "$BASE_TAG" "$ARM_RESULTS/train.part1.live.log" \
  2930 "$EXPECTED_PARENT_EPISODES" "$SPLIT_EPOCH" 1
part1_run_dir=$SEGMENT_RUN_DIR
part1_jsonl=$SEGMENT_JSONL
split_model="$part1_run_dir/model_azuki_local_$(printf '%06d' "$SPLIT_EPOCH").pt"
split_trainer="$part1_run_dir/trainer_state_$(printf '%06d' "$SPLIT_EPOCH").pt"
split_metadata="$split_model.meta.json"
split_episodes=$(jq -r '.env_completed_episodes' "$split_metadata")
if ! [[ "$split_episodes" =~ ^[0-9]+$ ]] || (( split_episodes < EXPECTED_PARENT_EPISODES )); then
  echo "[$BASE_TAG] invalid p3300 episode count: $split_episodes" >&2
  exit 1
fi
sha256sum \
  "$split_model" "$split_trainer" "$split_metadata" \
  "$ARM_LEAGUE/league_state.json" "$ARM_LEAGUE/league_state_promotion.json" \
  "$part1_jsonl" > "$ARM_RESULTS/split_p3300_input_sha256.txt"
printf '%s\n' "$part1_run_dir" > "$ARM_RESULTS/split_run_dir.txt"
printf '%s\n' "$part1_jsonl" > "$ARM_RESULTS/split_jsonl.txt"
touch "$ARM_RESULTS/SPLIT_P3300_DONE"

run_segment resume3300 "$split_model" "$RESUME_TAG" \
  "$ARM_RESULTS/train.part2.live.log" "$SPLIT_EPOCH" "$split_episodes" \
  "$TARGET_EPOCH" 0
part2_run_dir=$SEGMENT_RUN_DIR
part2_jsonl=$SEGMENT_JSONL
checkpoint="$part2_run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
trainer_state="$part2_run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"

cp -a "$part1_jsonl" "$ARM_RESULTS/train.part1_source.jsonl"
cp -a "$part2_jsonl" "$ARM_RESULTS/train.part2_source.jsonl"
.venv/bin/python - "$part1_jsonl" "$part2_jsonl" "$ARM_RESULTS/train.jsonl" <<'PY'
import json
from pathlib import Path
import sys

rows = {}
for path, lower, upper in (
    (Path(sys.argv[1]), 2931, 3300),
    (Path(sys.argv[2]), 3301, 3900),
):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = row.get("epoch")
            if isinstance(epoch, (int, float)) and lower <= int(epoch) <= upper:
                rows[int(epoch)] = row
expected = set(range(2931, 3901))
if rows.keys() != expected:
    raise RuntimeError(
        f"Merged epochs differ: missing={sorted(expected - rows.keys())[:5]} "
        f"extra={sorted(rows.keys() - expected)[:5]}"
    )
with Path(sys.argv[3]).open("w", encoding="utf-8") as handle:
    for epoch in sorted(rows):
        handle.write(json.dumps(rows[epoch], separators=(",", ":")) + "\n")
PY
.venv/bin/python - "$ARM_RESULTS/train.jsonl" > "$ARM_RESULTS/sps_summary.txt" <<'PY'
import json
import statistics
import sys

values = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        values.append(float(row["SPS"]))
steady = values[20:]
print(len(values), f"{statistics.median(steady):.6f}", f"{statistics.median(steady[-100:]):.6f}")
PY
.venv/bin/python - \
  "$ARM_RESULTS/train.part1.live.log" "$ARM_RESULTS/train.part2.live.log" \
  "$ARM_RESULTS/train.log" <<'PY'
from pathlib import Path
import sys

part1 = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
part2 = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
Path(sys.argv[3]).write_text(part1 + "\n\n===== RESUME P3300 =====\n\n" + part2, encoding="utf-8")
PY
printf '%s\n' "$part2_run_dir" > "$ARM_RESULTS/run_dir.txt"
sha256sum "$checkpoint" "$trainer_state" > "$ARM_RESULTS/checkpoint_sha256.txt"
touch "$ARM_RESULTS/TRAIN_DONE"
echo "[$BASE_TAG] split training complete at p$TARGET_EPOCH"
