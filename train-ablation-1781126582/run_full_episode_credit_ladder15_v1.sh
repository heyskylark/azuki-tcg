#!/usr/bin/env bash
# Stage 2 matched 15M efficacy ladder after the full-episode credit smoke passes.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${FULL_CREDIT_LADDER_CAMPAIGN:-full_episode_credit_ladder15_labelfix_v1}
PARENT_MANIFEST=${FULL_CREDIT_PARENT_MANIFEST:-train-ablation-1781126582/results/next_ablation_v1/stage2/parent_manifest.json}
SMOKE_ROOT=${FULL_CREDIT_SMOKE_ROOT:-train-ablation-1781126582/results/next_ablation_v1/stage2/full_episode_credit_smoke_labelfix_v1}
CREDIT_MODE=${FULL_CREDIT_MODE:-retained_rows}
CANDIDATE_ARM=${FULL_CREDIT_CANDIDATE_ARM:-full_episode_credit}
PREFIX_MODEL=${FULL_CREDIT_PREFIX_MODEL:-}
PREFIX_REPORT=${FULL_CREDIT_PREFIX_REPORT:-}
PREFIX_SHA=${FULL_CREDIT_PREFIX_SHA256:-}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
UPDATES=${FULL_CREDIT_LADDER_UPDATES:-970}
PEAK_LR=${FULL_CREDIT_LADDER_PEAK_LR:-0.00003}
CREDIT_COEF=${FULL_CREDIT_COEF:-1.0}
CREDIT_BATCH_DRAFTS=${FULL_CREDIT_BATCH_DRAFTS:-80}
CREDIT_UPDATE_INTERVAL=${FULL_CREDIT_UPDATE_INTERVAL:-4}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235

manifest_value() {
  jq -er "$1" "$PARENT_MANIFEST"
}

if [[ ! -f "$PARENT_MANIFEST" ]]; then
  echo "[$CAMPAIGN] missing parent manifest: $PARENT_MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$SMOKE_ROOT/SMOKE_DONE" ]]; then
  echo "[$CAMPAIGN] passing Stage 2 smoke is required: $SMOKE_ROOT" >&2
  exit 1
fi
jq -e '
  .status == "pass" and
  .integrity_pass == true and
  .continuation_pass == true and
  .throughput_diagnostic_only == true
' "$SMOKE_ROOT/smoke_report.json" >/dev/null
case "$CREDIT_MODE" in
  retained_rows)
    [[ "$CANDIDATE_ARM" == full_episode_credit ]] || {
      echo "[$CAMPAIGN] retained_rows requires candidate arm full_episode_credit" >&2
      exit 1
    }
    ;;
  prefix_outcome)
    if [[ "$CANDIDATE_ARM" != prefix_outcome || ! -f "$PREFIX_MODEL" ||
          ! -f "$PREFIX_REPORT" || -z "$PREFIX_SHA" ]]; then
      echo "[$CAMPAIGN] prefix_outcome requires its named arm, model, report, and hash" >&2
      exit 1
    fi
    [[ "$(sha256sum "$PREFIX_MODEL" | awk '{print $1}')" == "$PREFIX_SHA" ]]
    jq -e --arg sha "$PREFIX_SHA" \
      '.validation_complete == true and .artifact.sha256 == $sha' "$PREFIX_REPORT" >/dev/null
    ;;
  *)
    echo "[$CAMPAIGN] unknown credit mode: $CREDIT_MODE" >&2
    exit 1
    ;;
esac

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

if (( UPDATES < 100 )); then
  echo "[$CAMPAIGN] efficacy ladder requires at least 100 updates" >&2
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
    "$CREDIT_MODE" <<'PY'
import json
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
path = Path(sys.argv[2])
output = Path(sys.argv[3])
parent_epoch = int(sys.argv[4])
target_epoch = int(sys.argv[5])
credit_mode = sys.argv[6]
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
draft_picks = values("environment/deckbuild/picks")
prefix = "environment/draft_prefix_outcome/"
prefix_quartiles = {
  str(index): sum(values(prefix + key + "_deltas"))
  for index, key in enumerate(
    ("q1_00_12", "q2_13_25", "q3_26_37", "q4_38_50"), start=1
  )
}
payload = {
  "arm": arm,
  "credit_mode": credit_mode,
  "epochs": len(ordered),
  "parent_epoch": parent_epoch,
  "target_epoch": target_epoch,
  "median_sps": statistics.median(steady),
  "tail100_median_sps": statistics.median(steady[-100:]),
  "p10_sps": sorted(steady)[max(0, int(0.1 * (len(steady) - 1)))],
  "timeout_rate_max": max(values("environment/timeout_truncation_rate"), default=0.0),
  "draft_picks_mean": statistics.mean(draft_picks) if draft_picks else None,
  "captured_records": sum(values("environment/draft_episode_credit/captured")),
  "labeled_records": sum(values("environment/draft_episode_credit/labeled")),
  "completed_episodes": sum(values("environment/draft_episode_credit/completed_episodes")),
  "draw_episodes": sum(values("environment/draft_episode_credit/draw_episodes")),
  "decisive_episodes": sum(values("environment/draft_episode_credit/decisive_episodes")),
  "win_episodes": sum(values("environment/draft_episode_credit/win_episodes")),
  "loss_episodes": sum(values("environment/draft_episode_credit/loss_episodes")),
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
  "target_mean_min": min(
    (float(row["losses/draft_episode_credit_target_mean"]) for row in trained),
    default=None,
  ),
  "target_mean_max": max(
    (float(row["losses/draft_episode_credit_target_mean"]) for row in trained),
    default=None,
  ),
  "quartile_examples": {
    str(index): sum(values(f"losses/draft_episode_credit_q{index}_examples"))
    for index in range(1, 5)
  },
  "prefix_outcome": {
    "telemetry_rows": len(values(prefix + "delta_count")),
    "delta_count": sum(values(prefix + "delta_count")),
    "residual_count": sum(values(prefix + "residual_count")),
    "completed": sum(values(prefix + "completed")),
    "truncated": sum(values(prefix + "truncated")),
    "unsynchronized": sum(values(prefix + "unsynchronized")),
    "telescope_abs_max": max(values(prefix + "telescope_abs_max"), default=0.0),
    "delta_abs_max": max(values(prefix + "delta_abs_max"), default=0.0),
    "prediction_std_mean": (
      statistics.mean(values(prefix + "prediction_std"))
      if values(prefix + "prediction_std") else 0.0
    ),
    "inference_seconds": sum(values(prefix + "inference_seconds")),
    "quartile_deltas": prefix_quartiles,
  },
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
runtime_inputs=(
  "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_METADATA" \
  "$PARENT_LEAGUE" "$PARENT_PROMOTION" "$SMOKE_ROOT/smoke_report.json" \
  python/src/league_training.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
  python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
  build/python/src/binding*.so \
  train-ablation-1781126582/run_full_episode_credit_ladder15_v1.sh
)
if [[ "$CREDIT_MODE" == prefix_outcome ]]; then
  runtime_inputs+=(python/src/draft_prefix_outcome.py "$PREFIX_MODEL" "$PREFIX_REPORT")
fi
sha256sum "${runtime_inputs[@]}" > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\nupdates=%s\ncredit_mode=%s\ncandidate_arm=%s\ncredit_coef=%s\nbatch_drafts=%s\nupdate_interval=%s\nprefix_sha256=%s\nthroughput_diagnostic_only=1\nsmoke_report_sha256=%s\n' \
  "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$UPDATES" "$CREDIT_MODE" \
  "$CANDIDATE_ARM" "$CREDIT_COEF" "$CREDIT_BATCH_DRAFTS" \
  "$CREDIT_UPDATE_INTERVAL" "$PREFIX_SHA" \
  "$(sha256sum "$SMOKE_ROOT/smoke_report.json" | awk '{print $1}')" \
  > "$RESULT_ROOT/campaign_config.txt"

arms=(control "$CANDIDATE_ARM")
for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  coefficient=0
  episode_credit_coef=0
  prefix_model=
  prefix_sha=
  if [[ "$arm" == "$CANDIDATE_ARM" ]]; then
    coefficient=$CREDIT_COEF
    if [[ "$CREDIT_MODE" == retained_rows ]]; then
      episode_credit_coef=$coefficient
    else
      prefix_model=$PREFIX_MODEL
      prefix_sha=$PREFIX_SHA
    fi
  fi
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

  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm coef=$coefficient"
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
    AZK_DRAFT_EPISODE_CREDIT_COEF="$episode_credit_coef" \
    AZK_DRAFT_EPISODE_CREDIT_CLIP=0.2 \
    AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS="$CREDIT_BATCH_DRAFTS" \
    AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL="$CREDIT_UPDATE_INTERVAL" \
    AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF=0.05 \
    AZK_DRAFT_EPISODE_CREDIT_SEED=420052 \
    AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS=10 \
    AZK_DRAFT_PREFIX_OUTCOME_MODEL="$prefix_model" \
    AZK_DRAFT_PREFIX_OUTCOME_SHA256="$prefix_sha" \
    AZK_DRAFT_PREFIX_OUTCOME_COEF="$coefficient" \
    AZK_DRAFT_PREFIX_PROBS= \
    AZK_DRAFT_PREFIX_LENGTHS= \
    AZK_DRAFT_PREFIX_SEED= \
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
         ! grep -q 'uniform_assignment=True' "$live_log"; then
        echo "[$CAMPAIGN] $arm startup invariant failure" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if [[ "$arm" == "$CANDIDATE_ARM" ]]; then
        if [[ "$CREDIT_MODE" == retained_rows ]] &&
           ! grep -q '\[draft-episode-credit\] enabled:' "$live_log"; then
          echo "[$CAMPAIGN] retained-row candidate was not enabled" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          kill -INT "$train_pid" 2>/dev/null || true
          guard_failed=1
          break
        elif [[ "$CREDIT_MODE" == prefix_outcome ]] &&
             ! grep -q "\[draft-prefix-outcome\] enabled:.*sha256=$PREFIX_SHA" "$live_log"; then
          echo "[$CAMPAIGN] prefix-outcome candidate was not enabled" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          kill -INT "$train_pid" 2>/dev/null || true
          guard_failed=1
          break
        fi
      fi
      startup_checked=1
      touch "$arm_results/STARTUP_INVARIANTS_OK"
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
  if [[ "$arm" == "$CANDIDATE_ARM" ]]; then
    if [[ "$CREDIT_MODE" == retained_rows ]]; then
      jq -e ".resume_config_fingerprint.reward_env.AZK_DRAFT_EPISODE_CREDIT_COEF == \"$coefficient\"" \
        "$checkpoint.meta.json" >/dev/null
    else
      jq -e --arg sha "$PREFIX_SHA" \
        '.resume_config_fingerprint.reward_env.AZK_DRAFT_PREFIX_OUTCOME_SHA256 == $sha' \
        "$checkpoint.meta.json" >/dev/null
    fi
  fi
  printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
  cp -a "$jsonl" "$arm_results/train.jsonl"
  sha256sum "$checkpoint" "$trainer_state" "$checkpoint.meta.json" \
    > "$arm_results/checkpoint_sha256.txt"
  write_training_summary "$arm" "$jsonl" "$arm_results/training_summary.json"
  touch "$arm_results/TRAIN_DONE"
done

.venv/bin/python - "$RESULT_ROOT" "$SPS_RELATIVE_FLOOR" "$SPS_HARD_FLOOR" \
  "$CANDIDATE_ARM" "$CREDIT_MODE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
relative_floor = float(sys.argv[2])
hard_floor = float(sys.argv[3])
candidate_arm = sys.argv[4]
credit_mode = sys.argv[5]
control = json.loads((root / "control/training_summary.json").read_text())
candidate = json.loads((root / candidate_arm / "training_summary.json").read_text())
ratio = candidate["median_sps"] / control["median_sps"]
if credit_mode == "retained_rows":
  integrity = (
    candidate["captured_records"] > 0
    and candidate["labeled_records"] > 0
    and candidate["completed_episodes"] > 0
    and candidate["decisive_episodes"] > 0
    and candidate["win_episodes"] > 0
    and candidate["loss_episodes"] > 0
    and candidate["draw_episodes"] < candidate["completed_episodes"]
    and candidate["trained_examples"] > 0
    and candidate["gradient_norm_max"] > 0
    and candidate["standard_actor_rows"] > 0
    and candidate["standard_masked_rows"] > 0
    and all(value > 0 for value in candidate["quartile_examples"].values())
  )
else:
  prefix = candidate["prefix_outcome"]
  integrity = (
    prefix["telemetry_rows"] == candidate["epochs"]
    and prefix["delta_count"] > 0
    and prefix["residual_count"] > 0
    and prefix["completed"] > 0
    and prefix["truncated"] == 0
    and prefix["unsynchronized"] == 0
    and prefix["telescope_abs_max"] <= 1e-5
    and prefix["delta_abs_max"] > 0
    and prefix["prediction_std_mean"] > 0
    and prefix["inference_seconds"] > 0
    and all(value > 0 for value in prefix["quartile_deltas"].values())
    and control["prefix_outcome"]["telemetry_rows"] == 0
  )
integrity = (
  integrity
  and candidate["timeout_rate_max"] == 0
  and candidate["draft_picks_mean"] is not None
  and abs(candidate["draft_picks_mean"] - 50.0) < 1e-6
)
relative_sps_pass = ratio >= relative_floor
absolute_sps_pass = candidate["median_sps"] >= hard_floor
performance = relative_sps_pass and absolute_sps_pass
continuation = integrity
payload = {
  "integrity_pass": integrity,
  "credit_mode": credit_mode,
  "candidate_arm": candidate_arm,
  "performance_pass": performance,
  "continuation_pass": continuation,
  "throughput_diagnostic_only": True,
  "candidate_over_control_sps": ratio,
  "relative_floor": relative_floor,
  "hard_floor": hard_floor,
  "relative_sps_pass": relative_sps_pass,
  "absolute_sps_pass": absolute_sps_pass,
  "relative_sps_waived": False,
  "relative_sps_waiver_reason": "",
}
(root / "training_gate.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
if not continuation:
  raise SystemExit(2)
PY

touch "$RESULT_ROOT/LADDER_TRAIN_DONE"
echo "[$CAMPAIGN] matched training complete"
