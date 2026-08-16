#!/usr/bin/env bash
# Stage 3 integrity smoke for random main-card prefixes. SPS is diagnostic.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${RANDOM_PREFIX_SMOKE_CAMPAIGN:-random_main_prefix_smoke_v1}
PARENT_MANIFEST=${RANDOM_PREFIX_PARENT_MANIFEST:-train-ablation-1781126582/results/next_ablation_v1/stage3/parent_manifest.json}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage3/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
UPDATES=${RANDOM_PREFIX_SMOKE_UPDATES:-48}
STEADY_UPDATES=${RANDOM_PREFIX_SMOKE_STEADY_UPDATES:-24}
PEAK_LR=${RANDOM_PREFIX_SMOKE_PEAK_LR:-0.000003}
CREDIT_COEF=${RANDOM_PREFIX_CREDIT_COEF:-1.0}
CREDIT_BATCH_DRAFTS=${RANDOM_PREFIX_CREDIT_BATCH_DRAFTS:-80}
CREDIT_UPDATE_INTERVAL=${RANDOM_PREFIX_CREDIT_UPDATE_INTERVAL:-4}
PREFIX_LENGTHS=${RANDOM_PREFIX_LENGTHS:-0,1,2,4}
PREFIX_PROBS=${RANDOM_PREFIX_PROBS:-0.5,0.25,0.125,0.125}
PREFIX_SEED=${RANDOM_PREFIX_SEED:-420053}
SPS_RELATIVE_FLOOR=0.95
SPS_HARD_FLOOR=1235

if [[ ! -f "$PARENT_MANIFEST" ]]; then
  echo "[$CAMPAIGN] missing parent manifest: $PARENT_MANIFEST" >&2
  exit 1
fi

manifest_value() {
  jq -er "$1" "$PARENT_MANIFEST"
}

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
STEADY_EPOCH_START=$((TARGET_EPOCH - STEADY_UPDATES + 1))

if (( UPDATES < 8 || STEADY_UPDATES < 4 || STEADY_UPDATES >= UPDATES )); then
  echo "[$CAMPAIGN] invalid update windows" >&2
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
  local arm=$1 prefix_probs=$2
  local coefficient=$CREDIT_COEF
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

  echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm prefix_probs=${prefix_probs:-disabled}"
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
    AZK_DRAFT_EPISODE_CREDIT_COEF="$coefficient" \
    AZK_DRAFT_EPISODE_CREDIT_CLIP=0.2 \
    AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS="$CREDIT_BATCH_DRAFTS" \
    AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL="$CREDIT_UPDATE_INTERVAL" \
    AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF=0.05 \
    AZK_DRAFT_EPISODE_CREDIT_SEED=420052 \
    AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS=10 \
    AZK_DRAFT_PREFIX_LENGTHS="$PREFIX_LENGTHS" \
    AZK_DRAFT_PREFIX_PROBS="$prefix_probs" \
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
  if [[ -z "$run_dir" ]] || [[ -z "$jsonl" ]] ||
     [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]]; then
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
  if ! grep -q '\[draft-episode-credit\] enabled:' "$live_log"; then
    echo "[$CAMPAIGN] $arm did not retain full-episode credit" >&2
    exit 1
  fi
  if [[ "$arm" == random_main_prefix ]]; then
    grep -q '\[draft-prefix\] enabled:' "$live_log" || {
      echo "[$CAMPAIGN] candidate did not enable random prefixes" >&2
      exit 1
    }
  elif grep -q '\[draft-prefix\] enabled:' "$live_log"; then
    echo "[$CAMPAIGN] control unexpectedly enabled random prefixes" >&2
    exit 1
  fi
  cp -a "$jsonl" "$arm_results/train.jsonl"
  printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
  sha256sum "$checkpoint" "$trainer_state" > "$arm_results/checkpoint_sha256.txt"
  touch "$arm_results/TRAIN_DONE"
}

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_METADATA" "$EXPECTED_METADATA_SHA" parent-metadata
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
cp -a "$PARENT_MANIFEST" "$RESULT_ROOT/parent_manifest.json"
sha256sum \
  "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_METADATA" \
  "$PARENT_LEAGUE" "$PARENT_PROMOTION" \
  python/src/league_training.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
  python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
  build/python/src/binding*.so \
  train-ablation-1781126582/run_random_main_prefix_smoke_v1.sh \
  > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\ncredit_coef=%s\nbatch_drafts=%s\nupdate_interval=%s\nprefix_lengths=%s\nprefix_probs=%s\nprefix_seed=%s\n' \
  "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$CREDIT_COEF" \
  "$CREDIT_BATCH_DRAFTS" "$CREDIT_UPDATE_INTERVAL" "$PREFIX_LENGTHS" \
  "$PREFIX_PROBS" "$PREFIX_SEED" \
  > "$RESULT_ROOT/campaign_config.txt"

run_arm control_no_prefix ""
run_arm random_main_prefix "$PREFIX_PROBS"

.venv/bin/python - "$RESULT_ROOT" "$PARENT_EPOCH" "$TARGET_EPOCH" \
  "$STEADY_EPOCH_START" "$SPS_RELATIVE_FLOOR" "$SPS_HARD_FLOOR" <<'PY'
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
  return {
    "epochs": len(ordered),
    "steady_interval_sps_median": statistics.median(steady),
    "steady_interval_sps_p10": sorted(steady)[max(0, int(0.1 * (len(steady) - 1)))],
    "timeout_rate_max": max(vals("environment/timeout_truncation_rate"), default=0.0),
    "draft_picks_mean": statistics.mean(vals("environment/deckbuild/picks")),
    "captured_records": sum(vals("environment/draft_episode_credit/captured")),
    "labeled_records": sum(vals("environment/draft_episode_credit/labeled")),
    "completed_episodes": sum(vals("environment/draft_episode_credit/completed_episodes")),
    "draw_episodes": sum(vals("environment/draft_episode_credit/draw_episodes")),
    "decisive_episodes": sum(vals("environment/draft_episode_credit/decisive_episodes")),
    "win_episodes": sum(vals("environment/draft_episode_credit/win_episodes")),
    "loss_episodes": sum(vals("environment/draft_episode_credit/loss_episodes")),
    "truncated_records": sum(vals("environment/draft_episode_credit/truncated_records")),
    "incomplete_episodes": sum(vals("environment/draft_episode_credit/incomplete_episodes")),
    "trained_examples": sum(vals("losses/draft_episode_credit_examples")),
    "training_updates": sum(value > 0 for value in vals("losses/draft_episode_credit_examples")),
    "fixed_rows": sum(vals("losses/draft_episode_credit_fixed_rows")),
    "gradient_norm_max": max(vals("losses/draft_episode_credit_gradient_norm"), default=0.0),
    "standard_actor_rows": sum(vals("losses/draft_episode_credit_standard_actor_rows")),
    "standard_masked_draft_rows": sum(vals("losses/draft_episode_credit_standard_masked_rows")),
    "importance_mean_min": min(vals("losses/draft_episode_credit_importance_mean"), default=0.0),
    "importance_mean_max": max(vals("losses/draft_episode_credit_importance_mean"), default=0.0),
    "clipfrac_max": max(vals("losses/draft_episode_credit_clipfrac"), default=0.0),
    "q_examples": {
      q: sum(vals(f"losses/draft_episode_credit_q{q}_examples")) for q in range(1, 5)
    },
    "aux_wall_seconds": sum(vals("losses/draft_episode_credit_train_seconds")),
    "aux_gpu_seconds": sum(vals("losses/draft_episode_credit_gpu_seconds")),
    "prefix_episodes": sum(vals("environment/draft_prefix/episodes")),
    "prefix_forced_rows": sum(vals("environment/draft_prefix/forced_rows")),
    "prefix_mean_length_mean": statistics.mean(
      vals("environment/draft_prefix/mean_length")
    ) if vals("environment/draft_prefix/mean_length") else 0.0,
    "prefix_length_episodes": {
      length: sum(vals(f"environment/draft_prefix/length_{length}_episodes"))
      for length in (0, 1, 2, 4)
    },
    "delayed_forced_rows": sum(vals("losses/draft_episode_credit_forced_rows")),
  }

control = summarize("control_no_prefix")
candidate = summarize("random_main_prefix")
ratio = candidate["steady_interval_sps_median"] / control["steady_interval_sps_median"]
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
  and candidate["standard_masked_draft_rows"] > 0
  and candidate["prefix_episodes"] > 0
  and candidate["prefix_forced_rows"] > 0
  and candidate["delayed_forced_rows"] > 0
  and 0.7 <= candidate["prefix_mean_length_mean"] <= 1.3
  and all(value > 0 for value in candidate["prefix_length_episodes"].values())
  and control["prefix_forced_rows"] == 0
  and control["delayed_forced_rows"] == 0
  and all(value > 0 for value in candidate["q_examples"].values())
  and candidate["incomplete_episodes"] == 0
  and candidate["timeout_rate_max"] == 0
  and abs(candidate["draft_picks_mean"] - 50.0) < 1e-6
  and math.isfinite(candidate["importance_mean_min"])
  and math.isfinite(candidate["importance_mean_max"])
)
relative_sps_pass = ratio >= relative_floor
absolute_sps_pass = candidate["steady_interval_sps_median"] >= hard_floor
performance = relative_sps_pass and absolute_sps_pass
payload = {
  "schema_version": 1,
  "status": "pass" if integrity else "fail",
  "integrity_pass": integrity,
  "performance_pass": performance,
  "continuation_pass": integrity,
  "throughput_diagnostic_only": True,
  "sps_ratio": ratio,
  "relative_floor": relative_floor,
  "hard_floor": hard_floor,
  "relative_sps_pass": relative_sps_pass,
  "absolute_sps_pass": absolute_sps_pass,
  "relative_sps_waived": False,
  "absolute_sps_waived": False,
  "control": control,
  "candidate": candidate,
}
(root / "smoke_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if not integrity:
  raise SystemExit(2)
PY

touch "$RESULT_ROOT/SMOKE_DONE"
echo "[$CAMPAIGN] smoke complete"
