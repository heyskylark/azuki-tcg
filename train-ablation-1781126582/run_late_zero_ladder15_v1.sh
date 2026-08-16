#!/usr/bin/env bash
# Stage 4 matched late-shaping-removal ladder.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${LATE_ZERO_LADDER_CAMPAIGN:-late_zero_ladder15_v1}
PARENT_MANIFEST=${LATE_ZERO_PARENT_MANIFEST:-train-ablation-1781126582/results/next_ablation_v1/stage4/parent_manifest.json}
SMOKE_ROOT=${LATE_ZERO_SMOKE_ROOT:-train-ablation-1781126582/results/next_ablation_v1/stage4/late_zero_smoke_v1}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage4/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
UPDATES=${LATE_ZERO_LADDER_UPDATES:-970}
PEAK_LR=${LATE_ZERO_LADDER_PEAK_LR:-0.00003}
MODE=${LATE_ZERO_MODE:-ladder}
REQUIRE_SMOKE=${LATE_ZERO_REQUIRE_SMOKE:-1}
CHECKPOINT_INTERVAL=${LATE_ZERO_CHECKPOINT_INTERVAL:-25}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235

manifest_value() {
  jq -er "$1" "$PARENT_MANIFEST"
}

if [[ ! -f "$PARENT_MANIFEST" ]]; then
  echo "[$CAMPAIGN] missing parent manifest: $PARENT_MANIFEST" >&2
  exit 1
fi
if (( REQUIRE_SMOKE )) && [[ ! -f "$SMOKE_ROOT/SMOKE_DONE" ]]; then
  echo "[$CAMPAIGN] passing Stage 4 smoke is required: $SMOKE_ROOT" >&2
  exit 1
fi
if (( REQUIRE_SMOKE )); then
  jq -e '.status == "pass" and .integrity_pass == true and .performance_pass == true' \
    "$SMOKE_ROOT/smoke_report.json" >/dev/null
fi

PARENT_MODEL=$(manifest_value '.model.path')
PARENT_TRAINER=$(manifest_value '.trainer.path')
PARENT_METADATA=$(manifest_value '.metadata.path')
PARENT_LEAGUE=$(manifest_value '.league.path')
PARENT_PROMOTION=$(manifest_value '.promotion.path')
EXPECTED_MODEL_SHA=$(manifest_value '.model.sha256')
EXPECTED_TRAINER_SHA=$(manifest_value '.trainer.sha256')
EXPECTED_METADATA_SHA=$(manifest_value '.metadata.sha256')
EXPECTED_LEAGUE_SHA=$(manifest_value '.league.sha256')
EXPECTED_PROMOTION_SHA=$(manifest_value '.promotion.sha256')
PARENT_EPOCH=$(manifest_value '.epoch')
EXPECTED_PARENT_EPISODES=$(manifest_value '.completed_episodes')
TARGET_EPOCH=$((PARENT_EPOCH + UPDATES))
TOTAL_TIMESTEPS=$((TARGET_EPOCH * 15360))
ZERO_UPDATES=$(((UPDATES + 4) / 5))
ANNEAL_START_EPOCH=$PARENT_EPOCH
ANNEAL_END_EPOCH=$((TARGET_EPOCH - ZERO_UPDATES + 1))
PARENT_CREDIT_COEF=$(jq -er '.resume_config_fingerprint.reward_env.AZK_DRAFT_EPISODE_CREDIT_COEF' "$PARENT_METADATA")
PARENT_PREFIX_PROBS=$(jq -r '.resume_config_fingerprint.reward_env.AZK_DRAFT_PREFIX_PROBS // ""' "$PARENT_METADATA")

if [[ "$MODE" == ladder ]] && (( UPDATES < 100 )); then
  echo "[$CAMPAIGN] efficacy ladder requires at least 100 updates" >&2
  exit 1
fi
if (( UPDATES < 32 || ZERO_UPDATES < 6 || ANNEAL_END_EPOCH <= ANNEAL_START_EPOCH )); then
  echo "[$CAMPAIGN] invalid late-zero schedule" >&2
  exit 1
fi
if ! .venv/bin/python - "$PARENT_CREDIT_COEF" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
then
  echo "[$CAMPAIGN] Stage 4 requires an accepted nonzero full-episode credit parent" >&2
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

write_training_summary() {
  local arm=$1 jsonl=$2 output=$3
  .venv/bin/python - "$arm" "$jsonl" "$output" "$PARENT_EPOCH" "$TARGET_EPOCH" \
    "$ANNEAL_START_EPOCH" "$ANNEAL_END_EPOCH" <<'PY'
import json
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
path = Path(sys.argv[2])
output = Path(sys.argv[3])
parent_epoch = int(sys.argv[4])
target_epoch = int(sys.argv[5])
anneal_start = int(sys.argv[6])
anneal_end = int(sys.argv[7])
rows = {}
for line in path.read_text(encoding="utf-8").splitlines():
  try:
    row = json.loads(line)
  except json.JSONDecodeError:
    continue
  epoch = row.get("epoch")
  if isinstance(epoch, (int, float)) and parent_epoch < int(epoch) <= target_epoch:
    rows[int(epoch)] = row
expected = set(range(parent_epoch + 1, target_epoch + 1))
if set(rows) != expected:
  raise RuntimeError(f"{arm} epoch mismatch: missing={sorted(expected - set(rows))[:20]}")
ordered = [rows[epoch] for epoch in sorted(rows)]

intervals = []
for left, right in zip(ordered, ordered[1:]):
  elapsed = float(right["uptime"]) - float(left["uptime"])
  steps = float(right["agent_steps"]) - float(left["agent_steps"])
  if elapsed > 0.0 and steps > 0.0:
    intervals.append(steps / elapsed)
steady = intervals[20:]
if not steady:
  raise RuntimeError(f"{arm} has no sustained SPS window")

def values(key):
  return [
    float(row[key])
    for row in ordered
    if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
  ]

trained = [
  row for row in ordered
  if float(row.get("losses/draft_episode_credit_examples", 0.0)) > 0.0
]

def expected_multiplier(epoch):
  if arm == "fixed_floor":
    return 1.0
  if epoch <= anneal_start:
    return 1.0
  if epoch >= anneal_end:
    return 0.0
  return (anneal_end - epoch) / (anneal_end - anneal_start)

schedule = []
for row in ordered:
  epoch = int(row["epoch"])
  multiplier = float(row["environment/trainer_shaped_reward_multiplier"])
  update = int(row["environment/trainer_shaped_reward_update"])
  effective = float(row["environment/effective_reward_shaping_scale"])
  expected_multiplier_value = expected_multiplier(epoch)
  schedule.append({
    "epoch": epoch,
    "update": update,
    "multiplier": multiplier,
    "expected_multiplier": expected_multiplier_value,
    "effective_scale": effective,
    "credit_examples": float(row.get("losses/draft_episode_credit_examples", 0.0)),
    "credit_labels": float(row.get("environment/draft_episode_credit/labeled", 0.0)),
  })
zero_rows = [row for row in schedule if abs(row["expected_multiplier"]) <= 1e-12]
schedule_integrity = all(
  row["update"] == row["epoch"]
  and abs(row["multiplier"] - row["expected_multiplier"]) <= 1e-8
  and abs(row["effective_scale"] - 0.15 * row["expected_multiplier"]) <= 2e-6
  for row in schedule
)
payload = {
  "arm": arm,
  "epochs": len(ordered),
  "parent_epoch": parent_epoch,
  "target_epoch": target_epoch,
  "median_sps": statistics.median(steady),
  "tail100_median_sps": statistics.median(steady[-100:]),
  "p10_sps": sorted(steady)[max(0, int(0.1 * (len(steady) - 1)))],
  "timeout_rate_max": max(values("environment/timeout_truncation_rate"), default=0.0),
  "draft_picks_mean": statistics.mean(values("environment/deckbuild/picks")),
  "captured_records": sum(values("environment/draft_episode_credit/captured")),
  "labeled_records": sum(values("environment/draft_episode_credit/labeled")),
  "completed_episodes": sum(values("environment/draft_episode_credit/completed_episodes")),
  "truncated_records": sum(values("environment/draft_episode_credit/truncated_records")),
  "incomplete_episodes": sum(values("environment/draft_episode_credit/incomplete_episodes")),
  "trained_examples": sum(values("losses/draft_episode_credit_examples")),
  "training_updates": len(trained),
  "gradient_norm_max": max(values("losses/draft_episode_credit_gradient_norm"), default=0.0),
  "standard_actor_rows": sum(values("losses/draft_episode_credit_standard_actor_rows")),
  "standard_masked_rows": sum(values("losses/draft_episode_credit_standard_masked_rows")),
  "aux_wall_seconds": sum(values("losses/draft_episode_credit_train_seconds")),
  "aux_gpu_seconds": sum(values("losses/draft_episode_credit_gpu_seconds")),
  "importance_mean_median": (
    statistics.median(float(row["losses/draft_episode_credit_importance_mean"]) for row in trained)
    if trained else None
  ),
  "clipfrac_max": max(
    (float(row["losses/draft_episode_credit_clipfrac"]) for row in trained),
    default=None,
  ),
  "quartile_examples": {
    str(index): sum(values(f"losses/draft_episode_credit_q{index}_examples"))
    for index in range(1, 5)
  },
  "schedule_integrity": schedule_integrity,
  "anneal_start_epoch": anneal_start,
  "anneal_end_epoch": anneal_end,
  "zero_update_count": len(zero_rows),
  "first_zero_epoch": zero_rows[0]["epoch"] if zero_rows else None,
  "zero_tail_credit_examples": sum(row["credit_examples"] for row in zero_rows),
  "zero_tail_credit_labels": sum(row["credit_labels"] for row in zero_rows),
  "checkpoint_multipliers": {
    str(row["epoch"]): row["multiplier"]
    for row in schedule
    if row["epoch"] == target_epoch or row["epoch"] % 25 == 0
  },
  "prefix_forced_rows": sum(values("environment/draft_prefix/forced_rows")),
  "schedule": schedule,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_METADATA" "$EXPECTED_METADATA_SHA" parent-metadata
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES and .resume_config_fingerprint.draft_uniform_assignment == true" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
cp -a "$PARENT_MANIFEST" "$RESULT_ROOT/parent_manifest.json"
smoke_hash_input=()
if (( REQUIRE_SMOKE )); then
  smoke_hash_input+=("$SMOKE_ROOT/smoke_report.json")
fi
sha256sum \
  "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_METADATA" \
  "$PARENT_LEAGUE" "$PARENT_PROMOTION" "${smoke_hash_input[@]}" \
  python/src/league_training.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
  python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
  build/python/src/binding*.so \
  train-ablation-1781126582/run_late_zero_ladder15_v1.sh \
  > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
printf 'campaign=%s\nmode=%s\nparent_epoch=%s\ntarget_epoch=%s\nupdates=%s\nzero_updates=%s\nanneal_start_epoch=%s\nanneal_end_epoch=%s\nparent_credit_coef=%s\nparent_prefix_probs=%s\n' \
  "$CAMPAIGN" "$MODE" "$PARENT_EPOCH" "$TARGET_EPOCH" "$UPDATES" \
  "$ZERO_UPDATES" "$ANNEAL_START_EPOCH" "$ANNEAL_END_EPOCH" \
  "$PARENT_CREDIT_COEF" "$PARENT_PREFIX_PROBS" \
  > "$RESULT_ROOT/campaign_config.txt"

arms=(fixed_floor late_zero)
anneal_enabled=(0 1)
for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  schedule_enabled=${anneal_enabled[$index]}
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  tag="${CAMPAIGN}_${arm}"
  snapshot_dir="experiments/abl_snapshots/$tag"
  live_log="$arm_results/train.live.log"

  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already complete"
    continue
  fi
  if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null ||
     [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
    echo "[$CAMPAIGN] existing incomplete state for $arm; refusing overwrite" >&2
    exit 1
  fi
  mkdir -p "$arm_results" "$arm_league/opponents" "$snapshot_dir"
  cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
  cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"
  touch "$live_log"

  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm trainer_anneal=$schedule_enabled"
  env \
    AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV=1 \
    AZK_REWARD_SHAPING_ANNEAL=1 \
    AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
    AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 \
    AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
    AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 \
    AZK_TRAINER_SHAPED_REWARD_ANNEAL="$schedule_enabled" \
    AZK_TRAINER_SHAPED_REWARD_ANNEAL_START_EPOCH="$ANNEAL_START_EPOCH" \
    AZK_TRAINER_SHAPED_REWARD_ANNEAL_END_EPOCH="$ANNEAL_END_EPOCH" \
    AZK_XGATE_MASK=1 \
    AZK_PFSP=1 \
    AZK_PFSP_POWER=2.0 \
    AZK_DRAFT_REF_SEAT_PROB=0 \
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
      --train.checkpoint_interval "$CHECKPOINT_INTERVAL" \
      --env.draft_uniform_assignment true \
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
      > >(tee "$live_log") 2>&1 &
  train_pid=$!

  startup_checked=0
  guard_failed=0
  while kill -0 "$train_pid" 2>/dev/null; do
    if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$live_log"; then
      if ! grep -q "total_epochs=$TARGET_EPOCH" "$live_log" ||
         ! grep -q "remaining_epochs=$UPDATES" "$live_log" ||
         ! grep -q "AZK_RESUME_COMPLETED_EPISODES=$EXPECTED_PARENT_EPISODES" "$live_log" ||
         ! grep -q 'uniform_assignment=True' "$live_log" ||
         ! grep -q '\[resume\] keeping current schedule env vars per AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV=1' "$live_log" ||
         ! grep -q '\[resume\] applied saved reward env vars' "$live_log"; then
        echo "[$CAMPAIGN] $arm startup invariant failure" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if ! grep -q '\[draft-episode-credit\] enabled:' "$live_log"; then
        echo "[$CAMPAIGN] $arm did not retain full-episode credit" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if [[ "$arm" == late_zero ]] &&
         ! grep -q "\[trainer-shaped-reward\] enabled: start_epoch=$ANNEAL_START_EPOCH, end_epoch=$ANNEAL_END_EPOCH" "$live_log"; then
        echo "[$CAMPAIGN] late-zero schedule was not enabled" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if [[ "$arm" == fixed_floor ]] && grep -q '\[trainer-shaped-reward\] enabled:' "$live_log"; then
        echo "[$CAMPAIGN] fixed-floor control unexpectedly enabled trainer annealing" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if [[ -n "$PARENT_PREFIX_PROBS" ]] && ! grep -q '\[draft-prefix\] enabled:' "$live_log"; then
        echo "[$CAMPAIGN] $arm dropped the accepted prefix recipe" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if [[ -z "$PARENT_PREFIX_PROBS" ]] && grep -q '\[draft-prefix\] enabled:' "$live_log"; then
        echo "[$CAMPAIGN] $arm unexpectedly enabled prefixes" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      startup_checked=1
      touch "$arm_results/STARTUP_INVARIANTS_OK"
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
    if isinstance(row.get("agent_steps"), (int, float)) and isinstance(row.get("uptime"), (int, float)):
      points.append((float(row["agent_steps"]), float(row["uptime"])))
intervals = []
for left, right in zip(points, points[1:]):
  elapsed = right[1] - left[1]
  steps = right[0] - left[0]
  if elapsed > 0.0 and steps > 0.0:
    intervals.append(steps / elapsed)
failed = len(intervals) >= 60 and statistics.median(intervals[-30:]) < float(sys.argv[2])
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

  run_dir=$(latest_run_dir "$tag")
  jsonl=$(latest_jsonl "$tag")
  checkpoint=$(printf '%s/model_azuki_local_%06d.pt' "$run_dir" "$TARGET_EPOCH")
  trainer_state=$(printf '%s/trainer_state_%06d.pt' "$run_dir" "$TARGET_EPOCH")
  [[ -f "$checkpoint" && -f "$trainer_state" && -f "$checkpoint.meta.json" ]]
  jq -e ".update == $TARGET_EPOCH and .resume_config_fingerprint.draft_uniform_assignment == true" \
    "$checkpoint.meta.json" >/dev/null
  jq -e ".resume_config_fingerprint.reward_env.AZK_DRAFT_EPISODE_CREDIT_COEF == \"$PARENT_CREDIT_COEF\"" \
    "$checkpoint.meta.json" >/dev/null
  jq -e "(.resume_config_fingerprint.reward_env.AZK_DRAFT_PREFIX_PROBS // \"\") == \"$PARENT_PREFIX_PROBS\"" \
    "$checkpoint.meta.json" >/dev/null
  if [[ "$arm" == late_zero ]]; then
    jq -e ".trainer_shaped_reward_schedule.enabled == true and .trainer_shaped_reward_schedule.start_epoch == $ANNEAL_START_EPOCH and .trainer_shaped_reward_schedule.end_epoch == $ANNEAL_END_EPOCH and .trainer_shaped_reward_schedule.multiplier == 0" \
      "$checkpoint.meta.json" >/dev/null
  else
    jq -e '.trainer_shaped_reward_schedule.enabled == false and .trainer_shaped_reward_schedule.multiplier == 1' \
      "$checkpoint.meta.json" >/dev/null
  fi
  printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
  cp -a "$jsonl" "$arm_results/train.jsonl"
  sha256sum "$checkpoint" "$trainer_state" "$checkpoint.meta.json" \
    > "$arm_results/checkpoint_sha256.txt"
  write_training_summary "$arm" "$jsonl" "$arm_results/training_summary.json"
  touch "$arm_results/TRAIN_DONE"
done

.venv/bin/python - "$RESULT_ROOT" "$SPS_RELATIVE_FLOOR" "$SPS_HARD_FLOOR" \
  "$ZERO_UPDATES" "$PARENT_PREFIX_PROBS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
relative_floor = float(sys.argv[2])
hard_floor = float(sys.argv[3])
expected_zero_updates = int(sys.argv[4])
prefix_enabled = bool(sys.argv[5])
control = json.loads((root / "fixed_floor/training_summary.json").read_text())
candidate = json.loads((root / "late_zero/training_summary.json").read_text())
ratio = candidate["median_sps"] / control["median_sps"]
integrity = (
  control["captured_records"] > 0
  and control["labeled_records"] > 0
  and control["completed_episodes"] > 0
  and control["trained_examples"] > 0
  and control["gradient_norm_max"] > 0
  and candidate["captured_records"] > 0
  and candidate["labeled_records"] > 0
  and candidate["completed_episodes"] > 0
  and candidate["trained_examples"] > 0
  and candidate["gradient_norm_max"] > 0
  and candidate["standard_actor_rows"] > 0
  and candidate["standard_masked_rows"] > 0
  and all(value > 0 for value in candidate["quartile_examples"].values())
  and control["schedule_integrity"]
  and candidate["schedule_integrity"]
  and control["zero_update_count"] == 0
  and candidate["zero_update_count"] == expected_zero_updates
  and candidate["first_zero_epoch"] == candidate["anneal_end_epoch"]
  and candidate["zero_tail_credit_examples"] > 0
  and candidate["zero_tail_credit_labels"] > 0
  and ((control["prefix_forced_rows"] > 0 and candidate["prefix_forced_rows"] > 0)
       if prefix_enabled else
       (control["prefix_forced_rows"] == 0 and candidate["prefix_forced_rows"] == 0))
  and candidate["timeout_rate_max"] == 0
  and control["timeout_rate_max"] == 0
  and abs(candidate["draft_picks_mean"] - 50.0) < 1e-6
  and abs(control["draft_picks_mean"] - 50.0) < 1e-6
)
performance = ratio >= relative_floor and candidate["median_sps"] >= hard_floor
payload = {
  "schema_version": 1,
  "status": "pass" if integrity and performance else "fail",
  "integrity_pass": integrity,
  "performance_pass": performance,
  "candidate_over_control_sps": ratio,
  "relative_floor": relative_floor,
  "hard_floor": hard_floor,
}
(root / "training_gate.json").write_text(json.dumps(payload, indent=2) + "\n")
(root / "smoke_report.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
if not integrity or not performance:
  raise SystemExit(2)
PY

touch "$RESULT_ROOT/LADDER_TRAIN_DONE"
if [[ "$MODE" == smoke ]]; then
  touch "$RESULT_ROOT/SMOKE_DONE"
fi
echo "[$CAMPAIGN] matched training complete"
