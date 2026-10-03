#!/usr/bin/env bash
# Matched post-p4870 smoke and gradient calibration for Step 5b fallback.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${DRAFT_CREDIT_SMOKE_CAMPAIGN:-draft_terminal_credit_smoke_v1}
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
TARGET_EPOCH=${DRAFT_CREDIT_SMOKE_TARGET_EPOCH:-4910}
STEADY_UPDATES=${DRAFT_CREDIT_SMOKE_STEADY_UPDATES:-25}
TOTAL_TIMESTEPS=$((TARGET_EPOCH * 15360))
REMAINING_EPOCHS=$((TARGET_EPOCH - PARENT_EPOCH))
STEADY_EPOCH_START=$((TARGET_EPOCH - STEADY_UPDATES + 1))
PEAK_LR=${DRAFT_CREDIT_SMOKE_PEAK_LR:-0.000003}
PROBE_COEF=${DRAFT_CREDIT_PROBE_COEF:-0.05}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235
MAX_CALIBRATED_COEF=4.0

if (( REMAINING_EPOCHS < 1 || STEADY_UPDATES < 1 || STEADY_EPOCH_START <= PARENT_EPOCH + 1 )); then
  echo "[$CAMPAIGN] invalid target/steady window" >&2
  exit 1
fi

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

latest_run_dir() {
  local tag=$1
  find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" \
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

run_arm() {
  local arm=$1 coefficient=$2 grad_probe=$3
  local arm_results="$RESULT_ROOT/$arm"
  local arm_league="$LEAGUE_ROOT/$arm"
  local snapshot_dir="experiments/abl_snapshots/${CAMPAIGN}_${arm}"
  local tag="${CAMPAIGN}_${arm}"
  local live_log="$arm_results/train.live.log"

  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already complete"
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

  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm coef=$coefficient"
  set +e
  env \
    -u AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV \
    -u AZK_RESUME_KEEP_CURRENT_REWARD_ENV \
    -u AZK_RESUME_COMPLETED_EPISODES \
    -u AZK_RESUME_ALLOW_SCHEDULE_REWIND \
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
    AZK_PORTAL_GP_BONUS=0.3 \
    AZK_PORTAL_OUTCOME_BONUS=0 \
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
    AZK_XGATE_MASK=1 \
    AZK_PFSP=1 \
    AZK_DRAFT_REF_SEAT_PROB=0 \
    AZK_LEADER_TERMINAL_CREDIT_COEF=0 \
    AZK_DRAFT_TERMINAL_CREDIT_COEF="$coefficient" \
    AZK_DRAFT_TERMINAL_CREDIT_CLIP=0.2 \
    AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL=4 \
    AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE=512 \
    AZK_DRAFT_TERMINAL_CREDIT_SEED=420051 \
    AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS=10 \
    AZK_DRAFT_TERMINAL_CREDIT_GRAD_PROBE="$grad_probe" \
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
      --train.checkpoint_interval 40 \
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
  if [[ -z "$run_dir" ]] || [[ -z "$jsonl" ]] ||
     [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]]; then
    echo "[$CAMPAIGN] $arm lacks atomic p$TARGET_EPOCH output" >&2
    exit 1
  fi
  if ! grep -q 'optimizer_restored=True' "$live_log" ||
     ! grep -q "remaining_epochs=$REMAINING_EPOCHS" "$live_log" ||
     ! grep -q "total_epochs=$TARGET_EPOCH" "$live_log"; then
    echo "[$CAMPAIGN] $arm resume invariants failed" >&2
    exit 1
  fi
  if [[ "$arm" != control ]] &&
     ! grep -q '\[draft-terminal-credit\] enabled:' "$live_log"; then
    echo "[$CAMPAIGN] $arm did not enable whole-draft credit" >&2
    exit 1
  fi
  cp -a "$jsonl" "$arm_results/train.jsonl"
  printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
  sha256sum "$checkpoint" "$trainer_state" > "$arm_results/checkpoint_sha256.txt"

  .venv/bin/python - "$arm" "$jsonl" "$arm_results/summary.json" \
    "$PARENT_EPOCH" "$TARGET_EPOCH" "$STEADY_EPOCH_START" <<'PY'
import json
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
parent_epoch = int(sys.argv[4])
target_epoch = int(sys.argv[5])
steady_epoch_start = int(sys.argv[6])
rows = {}
for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
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

def values(key):
    return [float(row[key]) for _, row in sorted(rows.items()) if isinstance(row.get(key), (int, float))]

steady_epochs = range(steady_epoch_start, target_epoch + 1)
steady = [float(rows[epoch]["SPS"]) for epoch in steady_epochs]
boundary = rows[steady_epoch_start - 1]
last = rows[target_epoch]
elapsed = float(last["uptime"]) - float(boundary["uptime"])
steps = float(last["agent_steps"]) - float(boundary["agent_steps"])
if elapsed <= 0.0 or steps <= 0.0:
    raise RuntimeError(f"{arm} invalid steady interval: steps={steps} elapsed={elapsed}")
local_sps = []
for epoch in steady_epochs:
    previous = rows[epoch - 1]
    current = rows[epoch]
    local_elapsed = float(current["uptime"]) - float(previous["uptime"])
    local_steps = float(current["agent_steps"]) - float(previous["agent_steps"])
    if local_elapsed <= 0.0 or local_steps <= 0.0:
        raise RuntimeError(f"{arm} invalid local interval at epoch {epoch}")
    local_sps.append(local_steps / local_elapsed)
summary = {
    "arm": arm,
    "metric_rows": len(rows),
    "steady_epoch_start": steady_epoch_start,
    "steady_epoch_end": target_epoch,
    "steady_interval_sps": steps / elapsed,
    "steady_local_sps_median": statistics.median(local_sps),
    "steady_local_sps_p10": sorted(local_sps)[
        max(0, int(0.1 * (len(local_sps) - 1)))
    ],
    "steady_dashboard_sps_median": statistics.median(steady),
    "steady_sps_median": statistics.median(steady),
    "steady_sps_p10": sorted(steady)[max(0, int(0.1 * (len(steady) - 1)))],
    "terminal_labels_max": max(values("losses/win_prob_aux_labeled_rows"), default=0.0),
    "captured_records": sum(values("environment/draft_credit/captured")),
    "labeled_records": sum(values("environment/draft_credit/labeled")),
    "truncated_records": sum(values("environment/draft_credit/truncated")),
    "incomplete_episodes": sum(values("environment/draft_credit/incomplete_episodes")),
    "trained_examples": sum(values("losses/draft_credit_examples")),
    "fixed_batch_rows": sum(values("losses/draft_credit_fixed_batch_rows")),
    "aux_wall_seconds": sum(values("losses/draft_credit_train_seconds")),
    "aux_gpu_seconds": sum(values("losses/draft_credit_gpu_seconds")),
    "control_draft_gradient_norm": max(
        values("losses/draft_credit_control_draft_gradient_norm"), default=0.0
    ),
    "raw_aux_gradient_norm_first": next(
        (value for value in values("losses/draft_credit_raw_gradient_norm") if value > 0.0),
        0.0,
    ),
    "scaled_aux_gradient_norm_first": next(
        (value for value in values("losses/draft_credit_gradient_norm") if value > 0.0),
        0.0,
    ),
    "importance_min": min(values("losses/draft_credit_importance_mean"), default=0.0),
    "importance_max": max(values("losses/draft_credit_importance_mean"), default=0.0),
    "clipfrac_max": max(values("losses/draft_credit_clipfrac"), default=0.0),
    "trainer_shaped_reward_multiplier_min": min(
        values("environment/trainer_shaped_reward_multiplier"), default=1.0
    ),
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
  touch "$arm_results/TRAIN_DONE"
}

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
sha256sum \
  "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_LEAGUE" "$PARENT_PROMOTION" \
  python/src/azk_puffer/trainer.py python/src/league_training.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
  python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
  build/python/src/binding*.so \
  train-ablation-1781126582/run_draft_terminal_credit_smoke_v1.sh \
  > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"

run_arm control 0 0
run_arm gradient_probe "$PROBE_COEF" 1

.venv/bin/python - "$RESULT_ROOT/gradient_probe/summary.json" \
  "$RESULT_ROOT/calibration.json" "$PROBE_COEF" "$MAX_CALIBRATED_COEF" <<'PY'
import json
import math
import sys
from pathlib import Path

probe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
control_grad = float(probe["control_draft_gradient_norm"])
raw_aux_grad = float(probe["raw_aux_gradient_norm_first"])
if not math.isfinite(control_grad) or not math.isfinite(raw_aux_grad):
    raise RuntimeError("non-finite gradient calibration input")
if control_grad <= 0.0 or raw_aux_grad <= 0.0:
    raise RuntimeError(f"missing gradient calibration signal: {probe}")
coefficient = control_grad / raw_aux_grad
max_coefficient = float(sys.argv[4])
if not 0.0 < coefficient <= max_coefficient:
    raise RuntimeError(f"unsafe calibrated coefficient {coefficient}")
payload = {
    "rule": "control_draft_gradient_norm / raw_aux_gradient_norm_first",
    "probe_coefficient": float(sys.argv[3]),
    "control_draft_gradient_norm": control_grad,
    "raw_aux_gradient_norm": raw_aux_grad,
    "calibrated_coefficient": coefficient,
    "maximum_calibrated_coefficient": max_coefficient,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"{coefficient:.12g}")
PY
CALIBRATED_COEF=$(jq -r '.calibrated_coefficient' "$RESULT_ROOT/calibration.json")
# The calibrated arm is the production path. The expensive one-shot probe is
# confined to the disposable calibration arm and must not contaminate SPS.
run_arm calibrated "$CALIBRATED_COEF" 0

.venv/bin/python - "$RESULT_ROOT" "$SPS_RELATIVE_FLOOR" "$SPS_HARD_FLOOR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
relative_floor = float(sys.argv[2])
hard_floor = float(sys.argv[3])
control = json.loads((root / "control" / "summary.json").read_text(encoding="utf-8"))
candidate = json.loads((root / "calibrated" / "summary.json").read_text(encoding="utf-8"))
calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
ratio = candidate["steady_interval_sps"] / control["steady_interval_sps"]
gradient_ratio = (
    candidate["scaled_aux_gradient_norm_first"]
    / calibration["control_draft_gradient_norm"]
)
integrity = (
    candidate["captured_records"] > 0
    and candidate["labeled_records"] > 0
    and candidate["trained_examples"] > 0
    and candidate["trained_examples"] % 5 == 0
    and candidate["fixed_batch_rows"] >= candidate["trained_examples"]
    and candidate["incomplete_episodes"] == 0
    and candidate["terminal_labels_max"] > 0
    and candidate["trainer_shaped_reward_multiplier_min"] == 1.0
    and math.isfinite(gradient_ratio)
    and 0.5 <= gradient_ratio <= 2.0
)
passed = (
    integrity
    and ratio >= relative_floor
    and candidate["steady_interval_sps"] >= hard_floor
)
payload = {
    "status": "pass" if passed else "fail",
    "integrity_pass": integrity,
    "calibrated_coefficient": calibration["calibrated_coefficient"],
    "control_sps_interval": control["steady_interval_sps"],
    "candidate_sps_interval": candidate["steady_interval_sps"],
    "candidate_local_sps_median": candidate["steady_local_sps_median"],
    "candidate_local_sps_p10": candidate["steady_local_sps_p10"],
    "control_sps_median": control["steady_sps_median"],
    "candidate_sps_median": candidate["steady_sps_median"],
    "candidate_sps_p10": candidate["steady_sps_p10"],
    "sps_ratio": ratio,
    "relative_floor": relative_floor,
    "hard_floor": hard_floor,
    "initial_gradient_ratio": gradient_ratio,
    "candidate": candidate,
}
(root / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if not passed:
    raise SystemExit(2)
PY

touch "$RESULT_ROOT/SMOKE_DONE"
echo "[$CAMPAIGN] smoke complete"
