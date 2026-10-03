#!/usr/bin/env bash
# Matched 15M continuation ladder from the fully qualified promotion-v2 p2930
# endpoint. Reference decks replace a small share of existing frozen matchups;
# the learner always drafts and the already-loaded frozen PFSP policy pilots the
# fixed deck. Every arm is paused at p3900 with a 45M-compatible LR schedule so
# a selected arm can continue without another LR restart.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=reference_ladder15_v1
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
PARENT_DIR=experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308
PARENT_MODEL="$PARENT_DIR/model_azuki_local_002930.pt"
PARENT_TRAINER="$PARENT_DIR/trainer_state_002930.pt"
PARENT_LEAGUE=experiments/league/promotionv2_archive45_final/league_state.json
PARENT_PROMOTION=experiments/league/promotionv2_archive45_final/league_state_promotion.json
QUALIFICATION=train-ablation-1781126582/results/promotion_p2930_qualification_v1/qualification_metrics.json
EXPECTED_MODEL_SHA=7196734b2250ea1901e0e74e7bd693ebe6cf0918a5e51e5d231a3424b5d045d4
EXPECTED_TRAINER_SHA=0aa5da87f27801ed008c984ef64373874eec843e17558b1af2710d7e9b2bc5e1
EXPECTED_LEAGUE_SHA=7b1c8e5a932694d3370305d542231c63ca28ce8b6ee062563fa80d930675f707
EXPECTED_PROMOTION_SHA=9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17
TARGET_EPOCH=3900
CONTINUATION_TOTAL_TIMESTEPS=90000000
SPS_THRESHOLD=1300

arms=(control ref05 ref10)
probs=(0.00 0.05 0.10)

require_hash() {
  local path=$1 expected=$2 label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$CAMPAIGN] $label hash mismatch: $actual" >&2
    exit 1
  fi
}

if [[ ! -f "$QUALIFICATION" ]] ||
   ! jq -e '.metrics["league/promotion_accepted"] == 1' "$QUALIFICATION" >/dev/null; then
  echo "[$CAMPAIGN] p2930 has not passed the frozen full qualification gate" >&2
  exit 1
fi
require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs

for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  probability=${probs[$index]}
  tag="ref15v1_${arm}"
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/$tag"
  train_log="/tmp/train_${tag}.log"

  mkdir -p "$arm_results"
  if [[ -f "$arm_results/ARM_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already complete; skipping"
    continue
  fi
  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    run_dir=$(<"$arm_results/run_dir.txt")
    checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
    trainer_state="$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
    if [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]]; then
      echo "[$CAMPAIGN] $arm TRAIN_DONE is missing its atomic checkpoint pair" >&2
      exit 1
    fi
    echo "[$CAMPAIGN] resuming $arm after completed training"
  else
    if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null; then
      echo "[$CAMPAIGN] existing uncompleted output for $tag; refusing an ambiguous resume" >&2
      exit 1
    fi
    if [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
      echo "[$CAMPAIGN] existing uncompleted state for $arm; refusing overwrite" >&2
      exit 1
    fi

    mkdir -p "$arm_league/opponents" "$snapshot_dir"
    cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
    cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"

    echo "[$CAMPAIGN] $(date --iso-8601=seconds) starting $arm probability=$probability"
    env \
    AZK_REWARD_SHAPING_ANNEAL=1 \
    AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
    AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 \
    AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
    AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 \
    AZK_PORTAL_GP_BONUS=0.3 \
    AZK_XGATE_MASK=1 \
    AZK_PFSP=1 \
    AZK_EARLY_TEMPO_BONUS=0.1 \
    AZK_EARLY_TEMPO_CAP=4 \
    AZK_DMG_MITIGATION_BONUS=0.15 \
    AZK_DMG_MITIGATION_CAP=10 \
    AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
    AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
    AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4 \
    AZK_DRAFT_REF_OPPONENT_ONLY=1 \
    AZK_DRAFT_REF_SEAT_PROB="$probability" \
    AZK_DRAFT_REF_DECK_INDICES="$TRAIN_INDICES" \
    AZK_RESUME_COMPLETED_EPISODES=242 \
    AZK_RESUME_ALLOW_SOURCE_DRIFT=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/train.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --resume-checkpoint "$PARENT_MODEL" \
      --resume-load-optimizer \
      --resume-restart-lr-schedule \
      --no-resume-auto-reset-critic \
      --jsonl-log experiments/runlogs \
      --tag "$tag" \
      --train.seed 42 \
      --train.learning_rate 0.0003 \
      --train.ent_coef 0.002 \
      --train.ent_coef_anneal_initial 0.002 \
      --train.ent_coef_anneal_final 0.002 \
      --train.total_timesteps "$CONTINUATION_TOTAL_TIMESTEPS" \
      --train.checkpoint_interval 100 \
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
        > "$train_log" 2>&1 &
    train_pid=$!

    run_dir=""
    stopped_at_target=0
    guard_failed=0
    last_guard_count=0
    while kill -0 "$train_pid" 2>/dev/null; do
    if [[ -z "$run_dir" ]]; then
      run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
    fi
    if [[ -n "$run_dir" ]] &&
       [[ -f "$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt" ]] &&
       [[ -f "$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt" ]]; then
      stopped_at_target=1
      kill -INT "$train_pid" 2>/dev/null || true
      break
    fi

    jsonl=$(find experiments/runlogs -maxdepth 1 -type f -name "${tag}_*.jsonl" -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')
    if [[ -n "$jsonl" ]]; then
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
            rows.append(float(row["SPS"]))
count = len(rows)
if count < 40:
    print(count, 0.0, 0.0)
else:
    print(count, statistics.median(rows[-40:-20]), statistics.median(rows[-20:]))
PY
      )
      read -r guard_count previous_median current_median <<< "$guard"
      if (( guard_count >= 40 && guard_count > last_guard_count )); then
        last_guard_count=$guard_count
        if awk -v a="$previous_median" -v b="$current_median" -v t="$SPS_THRESHOLD" \
          'BEGIN { exit !((a < t) && (b < t)) }'; then
          echo "[$CAMPAIGN] $arm SPS guard failed: medians=$previous_median,$current_median" \
            | tee "$arm_results/SPS_GUARD_FAILED"
          guard_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
      fi
    fi
      sleep 5
    done
    wait "$train_pid" || true

    if (( guard_failed )); then
      exit 2
    fi
    if (( ! stopped_at_target )); then
      echo "[$CAMPAIGN] $arm exited before the p$TARGET_EPOCH atomic checkpoint" >&2
      exit 1
    fi

    checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
    trainer_state="$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
    sha256sum "$checkpoint" "$trainer_state" > "$arm_results/checkpoint_sha256.txt"
    cp -a "$train_log" "$arm_results/train.log"
    jsonl=$(find experiments/runlogs -maxdepth 1 -type f -name "${tag}_*.jsonl" \
      -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')
    if [[ -z "$jsonl" ]]; then
      echo "[$CAMPAIGN] $arm completed without a JSONL metrics file" >&2
      exit 1
    fi
    cp -a "$jsonl" "$arm_results/train.jsonl"
    printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
    touch "$arm_results/TRAIN_DONE"
  fi

  if [[ ! -f "$arm_results/deck_report.txt" ]]; then
    PYTHONPATH=python/src .venv/bin/python train-ablation-1781126582/analyze_decks.py \
      "$snapshot_dir" --buckets 5 --top 12 --csv "$arm_results/deck_report.csv" \
      > "$arm_results/deck_report.txt" 2>&1 || true
  fi

  for split in train holdout; do
    indices=$TRAIN_INDICES
    if [[ "$split" == holdout ]]; then
      indices=$HOLDOUT_INDICES
    fi
    if [[ ! -f "$arm_results/draftref_${split}.json" ]]; then
      PYTHONPATH=build/python/src:python/src .venv/bin/python \
        python/src/native_reference_eval.py \
        --config python/config/azuki_deckbuild_native_3090.ini \
        --checkpoint "$checkpoint" --opponent-checkpoint "$PARENT_MODEL" \
        --candidate-label "$arm" --opponent-label p2930_parent \
        --split "$split" --deck-indices "$indices" \
        --training-reference-indices "$TRAIN_INDICES" \
        --holdout-reference-indices "$HOLDOUT_INDICES" \
        --seeds 42009919,52009922 --batch-envs 12 \
        --device cuda --max-steps 600 \
        --json "$arm_results/draftref_${split}.json" \
        > "$arm_results/draftref_${split}.log" 2>&1
    fi
    jq -e '.summary.episodes == 288 and (.games | length) == 288' \
      "$arm_results/draftref_${split}.json" >/dev/null
  done

  if [[ ! -f "$arm_results/h2h_vs_parent.json" ]]; then
    PYTHONPATH=build/python/src:python/src .venv/bin/python python/src/native_policy_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint-a "$checkpoint" --checkpoint-b "$PARENT_MODEL" \
      --label-a "$arm" --label-b p2930_parent --batch-envs 12 \
      --seeds 42001701,52001704,62001707,72001710,82001713,92001716 \
      --max-steps 600 --device cuda \
      --json "$arm_results/h2h_vs_parent.json" \
      > "$arm_results/h2h_vs_parent.log" 2>&1
  fi
  jq -e '.summary.episodes == 192 and (.games | length) == 192' \
    "$arm_results/h2h_vs_parent.json" >/dev/null

  touch "$arm_results/ARM_DONE"
  echo "[$CAMPAIGN] $(date --iso-8601=seconds) completed $arm at p$TARGET_EPOCH"
done

PYTHONPATH=python/src .venv/bin/python \
  train-ablation-1781126582/reference_ladder_report.py "$RESULT_ROOT" \
  --json "$RESULT_ROOT/ladder_report.json" \
  --markdown "$RESULT_ROOT/ladder_report.md"
touch "$RESULT_ROOT/LADDER_DONE"
echo "[$CAMPAIGN] LADDER_DONE"
