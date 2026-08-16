#!/usr/bin/env bash
# Matched lifecycle migration from the accepted p4870 atomic parent.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${UNIFORM_ASSIGNMENT_CAMPAIGN:-uniform_assignment_ladder15_v2}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage1/$CAMPAIGN"
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
UPDATES=${UNIFORM_ASSIGNMENT_UPDATES:-970}
TARGET_EPOCH=$((PARENT_EPOCH + UPDATES))
TOTAL_TIMESTEPS=$((TARGET_EPOCH * 15360))
PEAK_LR=${UNIFORM_ASSIGNMENT_PEAK_LR:-0.00003}
SPS_HARD_FLOOR=1235

if (( UPDATES < 100 )); then
  echo "[$CAMPAIGN] matched efficacy run requires at least 100 updates" >&2
  exit 1
fi

arms=(control uniform_assignment)
uniform_flags=(false true)

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
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 2
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

write_sps_summary() {
  local jsonl=$1 output=$2
  .venv/bin/python - "$jsonl" "$output" <<'PY'
import json
import statistics
import sys
from pathlib import Path

rows = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(row.get("SPS"), (int, float))
            and isinstance(row.get("agent_steps"), (int, float))
            and isinstance(row.get("uptime"), (int, float))
        ):
            rows.append(row)
intervals = []
for left, right in zip(rows, rows[1:]):
    elapsed = float(right["uptime"]) - float(left["uptime"])
    steps = float(right["agent_steps"]) - float(left["agent_steps"])
    if elapsed > 0.0 and steps > 0.0:
        intervals.append(steps / elapsed)
steady = intervals[20:] if len(intervals) > 40 else intervals
if not steady:
    raise RuntimeError("training log has no positive step/time intervals")
logged_sps = [float(row["SPS"]) for row in rows]
payload = {
    "metric_rows": len(rows),
    "interval_rows": len(intervals),
    "median_sps": statistics.median(steady),
    "tail100_median_sps": statistics.median(steady[-100:]),
    "minimum_sps": min(steady),
    "maximum_sps": max(steady),
    "logged_sps_final": logged_sps[-1],
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

unset AZK_RESUME_COMPLETED_EPISODES AZK_RESUME_ALLOW_SCHEDULE_REWIND
unset AZK_DRAFT_REF_DECK_INDICES AZK_DRAFT_REF_OPPONENT_ONLY AZK_DRAFT_REF_SEAT_PROB
unset AZK_LEADER_TERMINAL_CREDIT_COEF AZK_LEADER_TERMINAL_CREDIT_GATE_CODES
unset AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL
unset AZK_DRAFT_TERMINAL_CREDIT_COEF AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL
unset AZK_DRAFT_EPISODE_CREDIT_COEF AZK_DRAFT_EPISODE_CREDIT_CLIP
unset AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS
unset AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL
unset AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF
unset AZK_DRAFT_EPISODE_CREDIT_SEED
unset AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS
unset AZK_DRAFT_PREFIX_PROBS AZK_DRAFT_PREFIX_LENGTHS

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
sha256sum \
  "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_LEAGUE" "$PARENT_PROMOTION" \
  python/src/deck_building.py python/src/deckbuild_metrics.py python/src/azk_native.py \
  python/src/binding.c python/src/tcg.h python/src/training_utils.py \
  python/src/train.py python/src/azk_puffer/trainer.py python/src/league_training.py \
  python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
  python/config/azuki_deckbuild_native_3090.ini build/python/src/binding*.so \
  train-ablation-1781126582/run_uniform_assignment_ladder15_v1.sh \
  train-ablation-1781126582/run_uniform_assignment_confirm45_v1.sh \
  > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\nupdates=%s\ntotal_timesteps=%s\npeak_lr=%s\n' \
  "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$UPDATES" \
  "$TOTAL_TIMESTEPS" "$PEAK_LR" \
  > "$RESULT_ROOT/campaign_config.txt"

for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  uniform=${uniform_flags[$index]}
  tag="${CAMPAIGN}_${arm}"
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/$tag"
  train_log="$arm_results/train.live.log"
  mkdir -p "$arm_results"

  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already complete"
    continue
  fi
  if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null; then
    echo "[$CAMPAIGN] existing uncompleted run for $tag; refusing ambiguous resume" >&2
    exit 1
  fi
  if [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
    echo "[$CAMPAIGN] existing uncompleted state for $arm; refusing overwrite" >&2
    exit 1
  fi
  mkdir -p "$arm_league/opponents" "$snapshot_dir"
  cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
  cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"
  touch "$train_log"
  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm uniform=$uniform"

  env \
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
  AZK_LEADER_TERMINAL_CREDIT_CLIP=0.2 \
  AZK_LEADER_TERMINAL_CREDIT_GATE_CODES=AZK01-122,STT04-002 \
  AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL=4 \
  AZK_LEADER_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS=10 \
  AZK_DRAFT_TERMINAL_CREDIT_COEF=0 \
  AZK_DRAFT_TERMINAL_CREDIT_CLIP=0.2 \
  AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL=4 \
  AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE=512 \
  AZK_DRAFT_TERMINAL_CREDIT_SEED=42 \
  AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS=10 \
  AZK_DRAFT_TERMINAL_CREDIT_GRAD_PROBE=0 \
  AZK_DRAFT_EPISODE_CREDIT_COEF=0 \
  AZK_DRAFT_EPISODE_CREDIT_CLIP=0.2 \
  AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS=80 \
  AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL=4 \
  AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF=0.05 \
  AZK_DRAFT_EPISODE_CREDIT_SEED=420052 \
  AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS=10 \
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
    --train.checkpoint_interval 100 \
    --env.draft_uniform_assignment "$uniform" \
    --env.deck_snapshot_every 25 \
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
    > >(tee "$train_log") 2>&1 &
  train_pid=$!

  startup_checked=0
  guard_failed=0
  while kill -0 "$train_pid" 2>/dev/null; do
    if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$train_log"; then
      expected_lifecycle="uniform_assignment=${uniform^}"
      if ! grep -q "total_epochs=$TARGET_EPOCH" "$train_log" ||
         ! grep -q "remaining_epochs=$UPDATES" "$train_log" ||
         ! grep -q "AZK_RESUME_COMPLETED_EPISODES=$EXPECTED_PARENT_EPISODES" "$train_log" ||
         ! grep -q "$expected_lifecycle" "$train_log"; then
        echo "[$CAMPAIGN] $arm startup invariant failure" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      startup_checked=1
      touch "$arm_results/STARTUP_INVARIANTS_OK"
      echo "[$CAMPAIGN] $arm startup invariants passed"
    fi

    jsonl=$(latest_jsonl "$tag")
    if [[ -n "$jsonl" ]]; then
      guard=$(.venv/bin/python - "$jsonl" "$SPS_HARD_FLOOR" <<'PY'
import json
import statistics
import sys

points = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        steps = row.get("agent_steps")
        uptime = row.get("uptime")
        if isinstance(steps, (int, float)) and isinstance(uptime, (int, float)):
            points.append((float(steps), float(uptime)))
intervals = []
for left, right in zip(points, points[1:]):
    elapsed = right[1] - left[1]
    steps = right[0] - left[0]
    if elapsed > 0.0 and steps > 0.0:
        intervals.append(steps / elapsed)
tail = intervals[-30:]
failed = len(intervals) >= 60 and statistics.median(tail) < float(sys.argv[2])
print("1" if failed else "0")
PY
)
      if [[ "$guard" == 1 ]]; then
        echo "[$CAMPAIGN] $arm sustained SPS below $SPS_HARD_FLOOR" | tee "$arm_results/SPS_GUARD_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
    fi
    sleep 5
  done

  set +e
  wait "$train_pid"
  train_status=$?
  set -e
  cleanup_training_processes "$tag"
  if (( train_status != 0 || guard_failed || ! startup_checked )); then
    echo "[$CAMPAIGN] $arm training failed status=$train_status" >&2
    exit 1
  fi

  run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -printf '%T@ %p\n' \
    | sort -nr | awk 'NR==1 {print $2}')
  checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
  trainer_state="$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
  [[ -f "$checkpoint" && -f "$trainer_state" ]]
  jq -e ".update == $TARGET_EPOCH and .resume_config_fingerprint.draft_uniform_assignment == $uniform" \
    "$checkpoint.meta.json" >/dev/null
  printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
  sha256sum "$checkpoint" "$trainer_state" "$checkpoint.meta.json" \
    > "$arm_results/checkpoint_sha256.txt"
  jsonl=$(latest_jsonl "$tag")
  cp "$jsonl" "$arm_results/train.jsonl"
  write_sps_summary "$jsonl" "$arm_results/sps_summary.json"
  touch "$arm_results/TRAIN_DONE"
done

.venv/bin/python - "$RESULT_ROOT/control/sps_summary.json" \
  "$RESULT_ROOT/uniform_assignment/sps_summary.json" "$RESULT_ROOT/sps_comparison.json" <<'PY'
import json
from pathlib import Path
import sys

control = json.loads(Path(sys.argv[1]).read_text())
candidate = json.loads(Path(sys.argv[2]).read_text())
ratio = candidate["median_sps"] / control["median_sps"]
payload = {
    "control_median_sps": control["median_sps"],
    "candidate_median_sps": candidate["median_sps"],
    "candidate_over_control": ratio,
    "passed_95_percent": ratio >= 0.95,
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed_95_percent"] else 2)
PY

touch "$RESULT_ROOT/LADDER_TRAIN_DONE"
echo "[$CAMPAIGN] matched training complete"
