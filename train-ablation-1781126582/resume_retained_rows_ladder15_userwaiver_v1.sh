#!/usr/bin/env bash
# Resume the retained-row candidate from its last atomic checkpoint after the
# historical absolute SPS guard stopped p8330. SPS is diagnostic in this
# user-authorized continuation; integrity and model-process failures still stop.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=retained_rows_ladder15_userwaiver_v1
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$CAMPAIGN"
ARM_RESULTS="$RESULT_ROOT/full_episode_credit"
SOURCE_RUN=experiments/azuki_local_retained_rows_ladder15_userwaiver_v1_full_episode_credit_178483931359
SOURCE_MODEL="$SOURCE_RUN/model_azuki_local_008300.pt"
SOURCE_TRAINER="$SOURCE_RUN/trainer_state_008300.pt"
SOURCE_METADATA="$SOURCE_MODEL.meta.json"
SOURCE_JSONL=experiments/runlogs/retained_rows_ladder15_userwaiver_v1_full_episode_credit_178483931359.jsonl
SOURCE_LEAGUE=experiments/league/retained_rows_ladder15_userwaiver_v1/full_episode_credit
RESUME_LEAGUE=experiments/league/retained_rows_ladder15_userwaiver_v1_resume8300/full_episode_credit
TAG=retained_rows_ladder15_userwaiver_v1_full_episode_credit_resume8300
SNAPSHOT_DIR=experiments/abl_snapshots/$TAG
CONSOLIDATED_RUN=experiments/azuki_local_retained_rows_ladder15_userwaiver_v1_full_episode_credit_consolidated
TRAIN_LOG="$ARM_RESULTS/train.resume8300.live.log"
SPS_LOG="$ARM_RESULTS/sps.resume8300.diagnostic.log"
PARENT_EPOCH=7800
RESUME_EPOCH=8300
TARGET_EPOCH=8770
REMAINING_UPDATES=$((TARGET_EPOCH - RESUME_EPOCH))
TOTAL_TIMESTEPS=$((TARGET_EPOCH * 15360))
EXPECTED_MODEL_SHA=82db0d17361c746c8b8618b9ff4f425e0604f84fec0a354f23813f7271b6eeb0
EXPECTED_TRAINER_SHA=81f6eb6928356b7116004cfc5cddb4c1d12175a318b16e5fde2e88744f57fc3e
EXPECTED_METADATA_SHA=35340723b63162f761f81fadce7fd5af0937245dd0be8e4ca04476d21da41b26
EXPECTED_SOURCE_JSONL_SHA=1b05bffd0e72c97c0a6dcde83820be2accfa32486998ebe9a4366ac1578e59d1
EXPECTED_LEAGUE_SHA=bfb70a8033e774a840fc6f02ed81551181a6deacd9797a78cda4285a80a4e23b
EXPECTED_PROMOTION_SHA=9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a
EXPECTED_FIRST_RESUME_LR=0.000014223046562112473
SPS_WAIVER_REASON="User prioritized retained-row model quality over SPS; optimize throughput after efficacy is measured."

require_hash() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$TAG] $label hash mismatch: $actual" >&2
    exit 1
  fi
}

latest_jsonl() {
  find experiments/runlogs -maxdepth 1 -type f -name "${TAG}_*.jsonl" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

latest_run_dir() {
  find experiments -maxdepth 1 -type d -name "azuki_local_${TAG}_*" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

if [[ -f "$ARM_RESULTS/TRAIN_DONE" && -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]]; then
  echo "[$TAG] recovery is already complete"
  exit 0
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
   grep -q '[0-9]'; then
  echo "[$TAG] GPU is occupied; refusing to overlap training" >&2
  exit 1
fi

require_hash "$SOURCE_MODEL" "$EXPECTED_MODEL_SHA" source-model
require_hash "$SOURCE_TRAINER" "$EXPECTED_TRAINER_SHA" source-trainer
require_hash "$SOURCE_METADATA" "$EXPECTED_METADATA_SHA" source-metadata
require_hash "$SOURCE_JSONL" "$EXPECTED_SOURCE_JSONL_SHA" source-jsonl
require_hash "$SOURCE_LEAGUE/league_state.json" "$EXPECTED_LEAGUE_SHA" source-league
require_hash "$SOURCE_LEAGUE/league_state_promotion.json" \
  "$EXPECTED_PROMOTION_SHA" source-promotion
jq -e \
  ".update == $RESUME_EPOCH and
   .env_completed_episodes == 690 and
   .resume_config_fingerprint.draft_uniform_assignment == true and
   .resume_config_fingerprint.reward_env.AZK_DRAFT_EPISODE_CREDIT_COEF == \"1.0\"" \
  "$SOURCE_METADATA" >/dev/null
jq -e --arg path "$ROOT/$SOURCE_MODEL" \
  ".history[-1].event == \"checkpoint_ingested\" and
   .history[-1].epoch == $RESUME_EPOCH and
   .history[-1].checkpoint_path == \$path" \
  "$SOURCE_LEAGUE/league_state.json" >/dev/null

if compgen -G "experiments/azuki_local_${TAG}_*" >/dev/null ||
   compgen -G "experiments/runlogs/${TAG}_*.jsonl" >/dev/null ||
   [[ -e "$RESUME_LEAGUE" ]] || [[ -e "$SNAPSHOT_DIR" ]] ||
   [[ -e "$CONSOLIDATED_RUN" ]]; then
  echo "[$TAG] existing continuation output; refusing ambiguous resume" >&2
  exit 1
fi

mkdir -p "$(dirname "$RESUME_LEAGUE")" "$SNAPSHOT_DIR"
cp -a "$SOURCE_LEAGUE" "$RESUME_LEAGUE"
sha256sum \
  "$SOURCE_MODEL" "$SOURCE_TRAINER" "$SOURCE_METADATA" "$SOURCE_JSONL" \
  "$SOURCE_LEAGUE/league_state.json" \
  "$SOURCE_LEAGUE/league_state_promotion.json" \
  train-ablation-1781126582/finalize_full_episode_credit_recovery.py \
  train-ablation-1781126582/resume_retained_rows_ladder15_userwaiver_v1.sh \
  > "$ARM_RESULTS/resume_p8300_input_sha256.txt"
printf '%s\n' \
  "resume_epoch=$RESUME_EPOCH" \
  "target_epoch=$TARGET_EPOCH" \
  "remaining_updates=$REMAINING_UPDATES" \
  "sps_policy=diagnostic_only" \
  "waiver_reason=$SPS_WAIVER_REASON" \
  > "$ARM_RESULTS/resume_p8300_config.txt"
printf '%s\n' \
  "The original absolute SPS guard stopped after logging p8330." \
  "The last atomic checkpoint is p8300, so source rows p8301-p8330 are" \
  "discarded and replaced by this exact optimizer/scheduler continuation." \
  "Raw SPS remains reported, but it cannot terminate this continuation." \
  > "$ARM_RESULTS/SPS_GUARD_DIAGNOSTIC_ONLY.txt"
touch "$TRAIN_LOG" "$SPS_LOG"

echo "[$TAG] $(date --iso-8601=seconds) resume p8300 without LR restart"
env \
  AZK_RESUME_KEEP_CURRENT_REWARD_ENV=1 \
  AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV=1 \
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
  AZK_DRAFT_EPISODE_CREDIT_COEF=1.0 \
  AZK_DRAFT_EPISODE_CREDIT_CLIP=0.2 \
  AZK_DRAFT_EPISODE_CREDIT_BATCH_DRAFTS=80 \
  AZK_DRAFT_EPISODE_CREDIT_UPDATE_INTERVAL=4 \
  AZK_DRAFT_EPISODE_CREDIT_BASELINE_COEF=0.05 \
  AZK_DRAFT_EPISODE_CREDIT_SEED=420052 \
  AZK_DRAFT_EPISODE_CREDIT_LABEL_WARMUP_EPOCHS=10 \
  AZK_DRAFT_PREFIX_OUTCOME_MODEL= \
  AZK_DRAFT_PREFIX_OUTCOME_SHA256= \
  AZK_DRAFT_PREFIX_OUTCOME_COEF=1.0 \
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
    --resume-checkpoint "$SOURCE_MODEL" \
    --resume-load-optimizer \
    --no-resume-auto-reset-critic \
    --jsonl-log experiments/runlogs --tag "$TAG" \
    --train.seed 42 --train.learning_rate 0.00003 \
    --train.ent_coef 0.002 \
    --train.ent_coef_anneal_initial 0.002 \
    --train.ent_coef_anneal_final 0.002 \
    --train.total_timesteps "$TOTAL_TIMESTEPS" \
    --train.checkpoint_interval 100 \
    --env.draft_uniform_assignment true \
    --env.deck_snapshot_every 25 \
    --env.deck_snapshot_dir "$SNAPSHOT_DIR" \
    --league.state_path "$RESUME_LEAGUE/league_state.json" \
    --league.opponent_dir "$RESUME_LEAGUE/opponents" \
    --league.eval_interval 100000 \
    --league.quick_eval_interval 100000 \
    --league.full_eval_interval 100000 \
    --league.promotion_shadow_mode true \
    --league.promotion_archive_affects_training_pool false \
    --league.promotion_panel_refresh_epochs 100000 \
    --league.production_anchor_checkpoint \
      experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt \
    > >(tee "$TRAIN_LOG") 2>&1 &
train_pid=$!

startup_checked=0
lr_checked=0
last_diagnostic_count=0
while kill -0 "$train_pid" 2>/dev/null; do
  if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$TRAIN_LOG"; then
    if ! grep -q "total_epochs=$TARGET_EPOCH" "$TRAIN_LOG" ||
       ! grep -q 'AZK_RESUME_COMPLETED_EPISODES=690' "$TRAIN_LOG" ||
       ! grep -q 'uniform_assignment=True' "$TRAIN_LOG" ||
       ! grep -Eq \
         '\[resume\] restored trainer state: .*epoch=8300, optimizer_restored=True, scheduler_restored=True' \
         "$TRAIN_LOG" ||
       ! grep -q '\[draft-episode-credit\] enabled:' "$TRAIN_LOG" ||
       grep -q 'restarted LR schedule' "$TRAIN_LOG"; then
      echo "[$TAG] startup invariant failure" \
        | tee "$ARM_RESULTS/RESUME8300_STARTUP_FAILED"
      kill -INT "$train_pid" 2>/dev/null || true
      break
    fi
    startup_checked=1
    touch "$ARM_RESULTS/RESUME8300_STARTUP_OK"
    echo "[$TAG] startup invariants passed; remaining_updates=$REMAINING_UPDATES"
  fi

  jsonl=$(latest_jsonl)
  if [[ -n "$jsonl" ]]; then
    diagnostic=$(.venv/bin/python - "$jsonl" <<'PY'
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
tail = [float(row["SPS"]) for row in rows[-30:]]
first_epoch = int(rows[0]["epoch"]) if rows else 0
first_lr = float(rows[0]["learning_rate"]) if rows else 0.0
median = statistics.median(tail) if tail else 0.0
print(len(rows), first_epoch, first_lr, median)
PY
)
    read -r row_count first_epoch first_lr tail_median <<< "$diagnostic"
    if (( ! lr_checked && row_count >= 1 )); then
      if (( first_epoch != RESUME_EPOCH + 1 )) ||
         ! awk -v actual="$first_lr" -v expected="$EXPECTED_FIRST_RESUME_LR" \
           'BEGIN {
              tolerance = (expected > 0 ? expected : -expected) * 0.000001;
              if (tolerance < 0.000000000001) tolerance = 0.000000000001;
              delta = actual - expected;
              if (delta < 0) delta = -delta;
              exit !(delta <= tolerance);
            }'; then
        echo "[$TAG] LR continuity failure: epoch=$first_epoch lr=$first_lr" \
          | tee "$ARM_RESULTS/RESUME8300_LR_FAILED"
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
      lr_checked=1
      printf 'first_epoch=%s\nfirst_lr=%s\nexpected_lr=%s\n' \
        "$first_epoch" "$first_lr" "$EXPECTED_FIRST_RESUME_LR" \
        > "$ARM_RESULTS/resume8300_lr_check.txt"
      echo "[$TAG] LR continuity passed: first_lr=$first_lr"
    fi
    if (( row_count >= last_diagnostic_count + 20 )); then
      last_diagnostic_count=$row_count
      printf '%s rows=%s tail30_median_sps=%s\n' \
        "$(date --iso-8601=seconds)" "$row_count" "$tail_median" \
        | tee -a "$SPS_LOG"
    fi
  fi
  sleep 5
done

set +e
wait "$train_pid"
train_status=$?
set -e
if (( train_status != 0 || ! startup_checked || ! lr_checked )); then
  echo "[$TAG] training failed status=$train_status startup=$startup_checked lr=$lr_checked" >&2
  exit 1
fi

run_dir=$(latest_run_dir)
jsonl=$(latest_jsonl)
checkpoint=$(printf '%s/model_azuki_local_%06d.pt' "$run_dir" "$TARGET_EPOCH")
trainer_state=$(printf '%s/trainer_state_%06d.pt' "$run_dir" "$TARGET_EPOCH")
metadata="$checkpoint.meta.json"
[[ -f "$checkpoint" && -f "$trainer_state" && -f "$metadata" && -n "$jsonl" ]]
jq -e \
  ".update == $TARGET_EPOCH and
   .resume_config_fingerprint.draft_uniform_assignment == true and
   .resume_config_fingerprint.reward_env.AZK_DRAFT_EPISODE_CREDIT_COEF == \"1.0\"" \
  "$metadata" >/dev/null

mkdir "$CONSOLIDATED_RUN"
for path in \
  "$SOURCE_RUN"/model_azuki_local_*.pt \
  "$SOURCE_RUN"/model_azuki_local_*.pt.meta.json \
  "$SOURCE_RUN"/trainer_state_[0-9]*.pt; do
  epoch=$(basename "$path" | grep -oE '[0-9]{6}' | tail -1)
  if [[ -n "$epoch" ]] && (( 10#$epoch <= RESUME_EPOCH )); then
    ln "$path" "$CONSOLIDATED_RUN/"
  fi
done
for path in \
  "$run_dir"/model_azuki_local_*.pt \
  "$run_dir"/model_azuki_local_*.pt.meta.json \
  "$run_dir"/trainer_state_[0-9]*.pt; do
  epoch=$(basename "$path" | grep -oE '[0-9]{6}' | tail -1)
  if [[ -n "$epoch" ]] && (( 10#$epoch > RESUME_EPOCH )); then
    ln "$path" "$CONSOLIDATED_RUN/"
  fi
done

cp -a "$SOURCE_JSONL" "$ARM_RESULTS/train.part1_source_with_discarded_tail.jsonl"
cp -a "$jsonl" "$ARM_RESULTS/train.part2_resume8300_source.jsonl"
.venv/bin/python \
  train-ablation-1781126582/finalize_full_episode_credit_recovery.py \
  --root "$RESULT_ROOT" \
  --source-jsonl "$SOURCE_JSONL" --resume-jsonl "$jsonl" \
  --parent-epoch "$PARENT_EPOCH" --resume-epoch "$RESUME_EPOCH" \
  --target-epoch "$TARGET_EPOCH" \
  --waiver-reason "$SPS_WAIVER_REASON"

printf '%s\n' "$CONSOLIDATED_RUN" > "$ARM_RESULTS/run_dir.txt"
printf '%s\n' "$run_dir" > "$ARM_RESULTS/resume_run_dir.txt"
printf '%s\n' "$RESUME_LEAGUE" > "$ARM_RESULTS/resume_league_dir.txt"
sha256sum "$checkpoint" "$trainer_state" "$metadata" \
  > "$ARM_RESULTS/checkpoint_sha256.txt"
touch "$ARM_RESULTS/RECOVERY_DONE" "$ARM_RESULTS/TRAIN_DONE"
touch "$RESULT_ROOT/LADDER_TRAIN_DONE"
echo "[$TAG] retained-row candidate recovery complete"
