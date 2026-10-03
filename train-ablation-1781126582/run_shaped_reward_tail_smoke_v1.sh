#!/usr/bin/env bash
# Matched p4870->p4910 smoke for absolute-update shaped-reward annealing.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${SHAPED_REWARD_SMOKE_CAMPAIGN:-shaped_reward_tail_smoke_v1}
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
SPLIT_EPOCH=4890
TARGET_EPOCH=4910
TOTAL_TIMESTEPS=75417600
ANNEAL_START_EPOCH=4875
ANNEAL_END_EPOCH=4885
PEAK_LR=${SHAPED_REWARD_SMOKE_PEAK_LR:-0.000003}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235

arms=(fixed_floor zero_tail)

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
  local arm_league=${10}
  local snapshot_dir=${11}
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
      if (( initial )); then
        grep -q 'remaining_epochs=40' "$log" || schedule_ok=0
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
    train-ablation-1781126582/run_shaped_reward_tail_smoke_v1.sh \
    > "$RESULT_ROOT/runtime_sha256.txt"
  git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
fi

for arm in "${arms[@]}"; do
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/${CAMPAIGN}_${arm}"
  initial_tag="${CAMPAIGN}_${arm}_initial"
  resume_tag="${CAMPAIGN}_${arm}_resume4890"
  mkdir -p "$arm_results"
  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already complete"
    continue
  fi
  if compgen -G "experiments/azuki_local_${initial_tag}_*" >/dev/null ||
     compgen -G "experiments/azuki_local_${resume_tag}_*" >/dev/null ||
     [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
    echo "[$CAMPAIGN] existing incomplete state for $arm; refusing overwrite" >&2
    exit 1
  fi
  mkdir -p "$arm_league/opponents" "$snapshot_dir"
  cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
  cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"

  run_segment "$arm" initial "$PARENT_MODEL" "$initial_tag" \
    "$arm_results/train.part1.live.log" "$PARENT_EPOCH" \
    "$EXPECTED_PARENT_EPISODES" "$SPLIT_EPOCH" 1 "$arm_league" "$snapshot_dir"
  part1_run_dir=$SEGMENT_RUN_DIR
  part1_jsonl=$SEGMENT_JSONL
  split_model="$part1_run_dir/model_azuki_local_$(printf '%06d' "$SPLIT_EPOCH").pt"
  split_metadata="$split_model.meta.json"
  split_episodes=$(jq -r '.env_completed_episodes' "$split_metadata")
  if ! [[ "$split_episodes" =~ ^[0-9]+$ ]] ||
     (( split_episodes < EXPECTED_PARENT_EPISODES )); then
    echo "[$CAMPAIGN] invalid $arm p$SPLIT_EPOCH episode count: $split_episodes" >&2
    exit 1
  fi
  sha256sum "$split_model" \
    "$part1_run_dir/trainer_state_$(printf '%06d' "$SPLIT_EPOCH").pt" \
    "$split_metadata" "$arm_league/league_state.json" "$part1_jsonl" \
    > "$arm_results/split_input_sha256.txt"

  run_segment "$arm" resume4890 "$split_model" "$resume_tag" \
    "$arm_results/train.part2.live.log" "$SPLIT_EPOCH" "$split_episodes" \
    "$TARGET_EPOCH" 0 "$arm_league" "$snapshot_dir"
  part2_run_dir=$SEGMENT_RUN_DIR
  part2_jsonl=$SEGMENT_JSONL

  cp -a "$part1_jsonl" "$arm_results/train.part1.jsonl"
  cp -a "$part2_jsonl" "$arm_results/train.part2.jsonl"
  printf '%s\n' "$part1_run_dir" > "$arm_results/part1_run_dir.txt"
  printf '%s\n' "$part2_run_dir" > "$arm_results/run_dir.txt"

  .venv/bin/python - \
    "$arm" "$part1_jsonl" "$part2_jsonl" "$part1_run_dir" "$part2_run_dir" \
    "$arm_results/summary.json" "$ANNEAL_START_EPOCH" "$ANNEAL_END_EPOCH" <<'PY'
import json
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
paths = (Path(sys.argv[2]), Path(sys.argv[3]))
part1_dir = Path(sys.argv[4])
part2_dir = Path(sys.argv[5])
output = Path(sys.argv[6])
start = int(sys.argv[7])
end = int(sys.argv[8])
rows = {}
for path in paths:
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = row.get("epoch")
        if isinstance(epoch, (int, float)) and 4871 <= int(epoch) <= 4910:
            rows[int(epoch)] = row
expected = set(range(4871, 4911))
if set(rows) != expected:
    raise RuntimeError(
        f"epoch mismatch: missing={sorted(expected - set(rows))} "
        f"extra={sorted(set(rows) - expected)}"
    )

enabled = arm == "zero_tail"
def expected_multiplier(epoch: int) -> float:
    if not enabled or epoch <= start:
        return 1.0
    if epoch >= end:
        return 0.0
    return (end - epoch) / (end - start)

for epoch, row in rows.items():
    actual = row.get("environment/trainer_shaped_reward_multiplier")
    update = row.get("environment/trainer_shaped_reward_update")
    if not isinstance(actual, (int, float)) or abs(float(actual) - expected_multiplier(epoch)) > 1e-8:
        raise RuntimeError(
            f"{arm} multiplier mismatch at p{epoch}: actual={actual}, "
            f"expected={expected_multiplier(epoch)}"
        )
    if not isinstance(update, (int, float)) or int(update) != epoch:
        raise RuntimeError(f"{arm} absolute update mismatch at p{epoch}: {update}")

checkpoint_epochs = (4875, 4880, 4885, 4890, 4910)
checkpoint_samples = {}
for epoch in checkpoint_epochs:
    directory = part1_dir if epoch <= 4890 else part2_dir
    metadata = directory / f"model_azuki_local_{epoch:06d}.pt.meta.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    state = payload.get("trainer_shaped_reward_schedule")
    if not isinstance(state, dict):
        raise RuntimeError(f"missing schedule metadata at p{epoch}")
    expected_value = expected_multiplier(epoch)
    if bool(state.get("enabled")) != enabled:
        raise RuntimeError(f"enabled mismatch at p{epoch}: {state}")
    if int(state.get("absolute_update", -1)) != epoch:
        raise RuntimeError(f"metadata update mismatch at p{epoch}: {state}")
    if abs(float(state.get("multiplier", -1.0)) - expected_value) > 1e-8:
        raise RuntimeError(f"metadata multiplier mismatch at p{epoch}: {state}")
    checkpoint_samples[str(epoch)] = float(state["multiplier"])

post_zero = [row for epoch, row in rows.items() if epoch >= end]
labels_max = max(
    (float(row.get("losses/win_prob_aux_labeled_rows", 0.0)) for row in post_zero),
    default=0.0,
)
if labels_max <= 0.0:
    raise RuntimeError(f"{arm} had no terminal labels after p{end}")

# Exclude five compile/warmup rows after each process start.
steady_sps = [
    float(row["SPS"])
    for epoch, row in sorted(rows.items())
    if epoch not in range(4871, 4876) and epoch not in range(4891, 4896)
]
summary = {
    "arm": arm,
    "metric_rows": len(rows),
    "checkpoint_multipliers": checkpoint_samples,
    "first_post_resume_update": int(rows[4891]["environment/trainer_shaped_reward_update"]),
    "first_post_resume_multiplier": float(
        rows[4891]["environment/trainer_shaped_reward_multiplier"]
    ),
    "post_zero_terminal_labels_max": labels_max,
    "steady_sps_count": len(steady_sps),
    "steady_sps_median": statistics.median(steady_sps),
}
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY

  final_model="$part2_run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
  final_trainer="$part2_run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
  sha256sum "$final_model" "$final_trainer" > "$arm_results/checkpoint_sha256.txt"
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
payload = {
    "status": "pass" if ratio >= relative_floor and treatment["steady_sps_median"] >= hard_floor else "fail",
    "fixed_floor_sps_median": control["steady_sps_median"],
    "zero_tail_sps_median": treatment["steady_sps_median"],
    "sps_ratio": ratio,
    "relative_floor": relative_floor,
    "hard_floor": hard_floor,
    "resume_restored_zero": treatment["first_post_resume_multiplier"] == 0.0,
    "terminal_labels_survived_zero": treatment["post_zero_terminal_labels_max"] > 0.0,
}
if not payload["resume_restored_zero"] or not payload["terminal_labels_survived_zero"]:
    payload["status"] = "fail"
(root / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if payload["status"] != "pass":
    raise SystemExit(2)
PY

touch "$RESULT_ROOT/SMOKE_DONE"
echo "[$CAMPAIGN] SMOKE_DONE"
