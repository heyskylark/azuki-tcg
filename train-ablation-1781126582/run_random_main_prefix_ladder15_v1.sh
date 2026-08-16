#!/usr/bin/env bash
# Stage 3 matched 15M efficacy ladder after the random-prefix smoke passes.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${RANDOM_PREFIX_LADDER_CAMPAIGN:-random_main_prefix_ladder15_v1}
PARENT_MANIFEST=${RANDOM_PREFIX_PARENT_MANIFEST:-train-ablation-1781126582/results/next_ablation_v1/stage3/parent_manifest.json}
SMOKE_ROOT=${RANDOM_PREFIX_SMOKE_ROOT:-train-ablation-1781126582/results/next_ablation_v1/stage3/random_main_prefix_smoke_v1}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage3/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
UPDATES=${RANDOM_PREFIX_LADDER_UPDATES:-970}
PEAK_LR=${RANDOM_PREFIX_LADDER_PEAK_LR:-0.00003}
CREDIT_COEF=${RANDOM_PREFIX_CREDIT_COEF:-1.0}
CREDIT_BATCH_DRAFTS=${RANDOM_PREFIX_CREDIT_BATCH_DRAFTS:-80}
CREDIT_UPDATE_INTERVAL=${RANDOM_PREFIX_CREDIT_UPDATE_INTERVAL:-4}
PREFIX_LENGTHS=${RANDOM_PREFIX_LENGTHS:-0,1,2,4}
PREFIX_PROBS=${RANDOM_PREFIX_PROBS:-0.5,0.25,0.125,0.125}
PREFIX_SEED=${RANDOM_PREFIX_SEED:-420053}
INCLUDE_STANDARD_CONTROL=${RANDOM_PREFIX_INCLUDE_STANDARD_CONTROL:-0}
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
  echo "[$CAMPAIGN] passing Stage 3 smoke is required: $SMOKE_ROOT" >&2
  exit 1
fi
jq -e '.status == "pass" and .integrity_pass == true and
  .continuation_pass == true and .throughput_diagnostic_only == true' \
  "$SMOKE_ROOT/smoke_report.json" >/dev/null

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
if [[ "$INCLUDE_STANDARD_CONTROL" != 0 && "$INCLUDE_STANDARD_CONTROL" != 1 ]]; then
  echo "[$CAMPAIGN] RANDOM_PREFIX_INCLUDE_STANDARD_CONTROL must be 0 or 1" >&2
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
  .venv/bin/python - "$arm" "$jsonl" "$output" "$PARENT_EPOCH" "$TARGET_EPOCH" <<'PY'
import json
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
path = Path(sys.argv[2])
output = Path(sys.argv[3])
parent_epoch = int(sys.argv[4])
target_epoch = int(sys.argv[5])
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
  "quartile_examples": {
    str(index): sum(values(f"losses/draft_episode_credit_q{index}_examples"))
    for index in range(1, 5)
  },
  "prefix_episodes": sum(values("environment/draft_prefix/episodes")),
  "prefix_forced_rows": sum(values("environment/draft_prefix/forced_rows")),
  "prefix_mean_length_mean": (
    statistics.mean(values("environment/draft_prefix/mean_length"))
    if values("environment/draft_prefix/mean_length") else 0.0
  ),
  "prefix_length_episodes": {
    str(length): sum(values(f"environment/draft_prefix/length_{length}_episodes"))
    for length in (0, 1, 2, 4)
  },
  "delayed_forced_rows": sum(values("losses/draft_episode_credit_forced_rows")),
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
sha256sum \
  "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_METADATA" \
  "$PARENT_LEAGUE" "$PARENT_PROMOTION" "$SMOKE_ROOT/smoke_report.json" \
  python/src/league_training.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
  python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
  build/python/src/binding*.so \
  train-ablation-1781126582/run_random_main_prefix_ladder15_v1.sh \
  > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\nupdates=%s\ncredit_coef=%s\nbatch_drafts=%s\nupdate_interval=%s\nprefix_lengths=%s\nprefix_probs=%s\nprefix_seed=%s\ninclude_standard_control=%s\n' \
  "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$UPDATES" "$CREDIT_COEF" \
  "$CREDIT_BATCH_DRAFTS" "$CREDIT_UPDATE_INTERVAL" "$PREFIX_LENGTHS" \
  "$PREFIX_PROBS" "$PREFIX_SEED" "$INCLUDE_STANDARD_CONTROL" \
  > "$RESULT_ROOT/campaign_config.txt"

if [[ "$INCLUDE_STANDARD_CONTROL" == 1 ]]; then
  arms=(standard_control control_no_prefix random_main_prefix)
  credit_coefficients=(0 "$CREDIT_COEF" "$CREDIT_COEF")
  prefix_probabilities=("" "" "$PREFIX_PROBS")
else
  arms=(control_no_prefix random_main_prefix)
  credit_coefficients=("$CREDIT_COEF" "$CREDIT_COEF")
  prefix_probabilities=("" "$PREFIX_PROBS")
fi
for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  coefficient=${credit_coefficients[$index]}
  prefix_probs=${prefix_probabilities[$index]}
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

  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm prefix_probs=${prefix_probs:-disabled}"
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
    AZK_DRAFT_EPISODE_CREDIT_COEF="$coefficient" \
    AZK_DRAFT_EPISODE_CREDIT_CLIP=0.2 \
    AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS="$CREDIT_BATCH_DRAFTS" \
    AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL="$CREDIT_UPDATE_INTERVAL" \
    AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF=0.05 \
    AZK_DRAFT_EPISODE_CREDIT_SEED=420052 \
    AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS=10 \
    AZK_DRAFT_PREFIX_PROBS="$prefix_probs" \
    AZK_DRAFT_PREFIX_LENGTHS="$PREFIX_LENGTHS" \
    AZK_DRAFT_PREFIX_SEED="$PREFIX_SEED" \
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
      if [[ "$arm" == standard_control ]]; then
        if grep -q '\[draft-episode-credit\] enabled:' "$live_log"; then
          echo "[$CAMPAIGN] standard control unexpectedly enabled full-episode credit" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          kill -INT "$train_pid" 2>/dev/null || true
          guard_failed=1
          break
        fi
      elif ! grep -q '\[draft-episode-credit\] enabled:' "$live_log"; then
        echo "[$CAMPAIGN] $arm did not retain full-episode credit" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if [[ "$arm" == random_main_prefix ]] &&
         ! grep -q '\[draft-prefix\] enabled:' "$live_log"; then
        echo "[$CAMPAIGN] candidate prefix path was not enabled" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
      fi
      if [[ "$arm" != random_main_prefix ]] &&
         grep -q '\[draft-prefix\] enabled:' "$live_log"; then
        echo "[$CAMPAIGN] $arm unexpectedly enabled prefixes" | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        guard_failed=1
        break
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
  jq -e ".resume_config_fingerprint.reward_env.AZK_DRAFT_EPISODE_CREDIT_COEF == \"$coefficient\"" \
    "$checkpoint.meta.json" >/dev/null
  if [[ "$arm" == random_main_prefix ]]; then
    jq -e ".resume_config_fingerprint.reward_env.AZK_DRAFT_PREFIX_PROBS == \"$PREFIX_PROBS\"" \
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
  "$INCLUDE_STANDARD_CONTROL" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
relative_floor = float(sys.argv[2])
hard_floor = float(sys.argv[3])
include_standard_control = bool(int(sys.argv[4]))
control = json.loads((root / "control_no_prefix/training_summary.json").read_text())
candidate = json.loads((root / "random_main_prefix/training_summary.json").read_text())
ratio = candidate["median_sps"] / control["median_sps"]

def exact_credit_integrity(summary):
  return (
    summary["captured_records"] > 0
    and summary["labeled_records"] > 0
    and summary["completed_episodes"] > 0
    and summary["decisive_episodes"] > 0
    and summary["win_episodes"] > 0
    and summary["loss_episodes"] > 0
    and summary["draw_episodes"] < summary["completed_episodes"]
    and summary["trained_examples"] > 0
    and summary["gradient_norm_max"] > 0
    and summary["standard_actor_rows"] > 0
    and summary["standard_masked_rows"] > 0
    and all(value > 0 for value in summary["quartile_examples"].values())
    and summary["timeout_rate_max"] == 0
    and abs(summary["draft_picks_mean"] - 50.0) < 1e-6
  )

control_integrity = (
  exact_credit_integrity(control)
  and control["prefix_forced_rows"] == 0
  and control["delayed_forced_rows"] == 0
)
candidate_integrity = (
  exact_credit_integrity(candidate)
  and candidate["prefix_episodes"] > 0
  and candidate["prefix_forced_rows"] > 0
  and candidate["delayed_forced_rows"] > 0
  and 0.7 <= candidate["prefix_mean_length_mean"] <= 1.3
  and all(value > 0 for value in candidate["prefix_length_episodes"].values())
)
standard = None
standard_integrity = True
exact_over_standard = None
if include_standard_control:
  standard = json.loads((root / "standard_control/training_summary.json").read_text())
  standard_integrity = (
    standard["captured_records"] == 0
    and standard["labeled_records"] == 0
    and standard["trained_examples"] == 0
    and standard["prefix_forced_rows"] == 0
    and standard["delayed_forced_rows"] == 0
    and standard["timeout_rate_max"] == 0
    and abs(standard["draft_picks_mean"] - 50.0) < 1e-6
  )
  exact_over_standard = control["median_sps"] / standard["median_sps"]
integrity = control_integrity and candidate_integrity and standard_integrity
relative_sps_pass = ratio >= relative_floor
absolute_sps_pass = candidate["median_sps"] >= hard_floor
performance = relative_sps_pass and absolute_sps_pass
payload = {
  "integrity_pass": integrity,
  "standard_control_present": include_standard_control,
  "standard_control_integrity_pass": standard_integrity,
  "exact_credit_integrity_pass": control_integrity,
  "prefix_integrity_pass": candidate_integrity,
  "performance_pass": performance,
  "continuation_pass": integrity,
  "throughput_diagnostic_only": True,
  "candidate_over_control_sps": ratio,
  "prefix_over_exact_sps": ratio,
  "exact_over_standard_sps": exact_over_standard,
  "relative_floor": relative_floor,
  "hard_floor": hard_floor,
  "relative_sps_pass": relative_sps_pass,
  "absolute_sps_pass": absolute_sps_pass,
  "relative_sps_waived": False,
  "absolute_sps_waived": False,
}
(root / "training_gate.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
if not integrity:
  raise SystemExit(2)
PY

touch "$RESULT_ROOT/LADDER_TRAIN_DONE"
echo "[$CAMPAIGN] matched training complete"
