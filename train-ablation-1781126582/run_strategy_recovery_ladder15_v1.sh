#!/usr/bin/env bash
# Matched 15M continuation ladder from the qualified promotion-v2 p2930
# endpoint. The control must retain parent strength before either shaping arm
# is allowed to train. All arms use a true p2930->p3900 cosine schedule.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${STRATEGY_LADDER_CAMPAIGN:-strategy_recovery_ladder15_v1}
PEAK_LR=${STRATEGY_LADDER_PEAK_LR:-0.0001}
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
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
TARGET_EPOCH=3900
TOTAL_TIMESTEPS=59904000
# The observed full-league lower bound is about 1300 SPS. Stop only after two
# windows fall more than 5% below it; candidate/control relative SPS is checked
# separately at 0.95.
SPS_HARD_FLOOR=1235
SPS_RELATIVE_FLOOR=0.95
OPPORTUNITY_GAMES=64
OPPORTUNITY_SHARDS=4
OPPORTUNITY_SEED0=810001
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17

arms=(control tempo_dedup portal_gp_tail)
tempo_dedup=(0 1 0)
portal_gp=(0.3 0.3 0.0)

require_hash() {
  local path=$1 expected=$2 label=$3
  local actual
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

sps_summary() {
  local jsonl=$1
  .venv/bin/python - "$jsonl" <<'PY'
import json
import statistics
import sys

values = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get("SPS")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
steady = values[20:] if len(values) > 40 else values[1:]
if not steady:
    steady = values
print(
    len(values),
    f"{statistics.median(steady):.6f}",
    f"{statistics.median(steady[-100:]):.6f}",
)
PY
}

run_h2h() {
  local checkpoint_a=$1 checkpoint_b=$2 label_a=$3 label_b=$4 output=$5
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_policy_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint-a "$checkpoint_a" --checkpoint-b "$checkpoint_b" \
      --label-a "$label_a" --label-b "$label_b" --batch-envs 12 \
      --seeds 42001701,52001704,62001707,72001710,82001713,92001716 \
      --max-steps 600 --device cuda --json "$output" \
      > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '.summary.episodes == 192 and (.games | length) == 192 and .summary.timeout_rate == 0' \
    "$output" >/dev/null
}

run_reference_eval() {
  local checkpoint=$1 arm=$2 split=$3 indices=$4 output=$5
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_reference_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --opponent-checkpoint "$PARENT_MODEL" \
      --candidate-label "$arm" --opponent-label p2930_parent \
      --split "$split" --deck-indices "$indices" \
      --training-reference-indices "$TRAIN_INDICES" \
      --holdout-reference-indices "$HOLDOUT_INDICES" \
      --seeds 42009919,52009922 --batch-envs 12 --device cuda \
      --max-steps 600 --json "$output" \
      > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '.summary.episodes == 288 and (.games | length) == 288 and .summary.timeout_rate == 0' \
    "$output" >/dev/null
}

run_opportunity_eval() {
  local checkpoint=$1 arm=$2 dedup=$3 gp=$4 arm_results=$5
  local output_dir="$arm_results/opportunity"
  local games_per_shard=$((OPPORTUNITY_GAMES / OPPORTUNITY_SHARDS))
  mkdir -p "$output_dir"
  if (( OPPORTUNITY_GAMES % OPPORTUNITY_SHARDS != 0 )); then
    echo "[$CAMPAIGN] opportunity games must divide evenly across shards" >&2
    exit 1
  fi

  local pids=()
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    local log="$output_dir/shard${shard}.log"
    local shard_seed=$((OPPORTUNITY_SEED0 + 7919 * games_per_shard * shard))
    if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]; then
      continue
    fi
    rm -f "$temporary"
    echo "[$CAMPAIGN] opportunity $arm shard=$shard seed=$shard_seed"
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python train-ablation-1781126582/play_selfplay_games.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --games "$games_per_shard" \
      --seed0 "$shard_seed" --device cpu --log-legal-actions \
      --out "$temporary" > "$log" 2>&1 &
    pids+=("$!")
  done
  local opportunity_failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      opportunity_failed=1
    fi
  done
  if (( opportunity_failed )); then
    echo "[$CAMPAIGN] one or more $arm opportunity shards failed" >&2
    exit 1
  fi
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$output"
    fi
    if [[ ! -f "$output" ]] || [[ $(wc -l < "$output") -ne $games_per_shard ]]; then
      echo "[$CAMPAIGN] incomplete opportunity shard: $output" >&2
      exit 1
    fi
  done

  local dedup_arg=()
  if [[ "$dedup" == 1 ]]; then
    dedup_arg+=(--early-tempo-dedup-portal-abilities)
  fi
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_opportunity_rates.py \
    "$output_dir"/shard*.jsonl --label "$arm" \
    --early-tempo-bonus 0.1 --early-tempo-cap 4 \
    --portal-gp-bonus "$gp" --shaping-scale 0.15 \
    "${dedup_arg[@]}" --json "$arm_results/opportunity.json" \
    | tee "$arm_results/opportunity.log"
  jq -e ".n_games == $OPPORTUNITY_GAMES" "$arm_results/opportunity.json" >/dev/null
}

write_control_strength_gate() {
  local input=$1 output=$2
  .venv/bin/python - "$input" "$output" <<'PY'
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
blocks = defaultdict(list)
for game in payload["games"]:
    blocks[str(game["block_id"])].append(float(game["candidate_score"]))
block_scores = np.asarray(
    [np.mean(values) for _, values in sorted(blocks.items())], dtype=np.float64
)
rng = np.random.default_rng(42_904_101)
sampled = block_scores[
    rng.integers(0, len(block_scores), size=(50_000, len(block_scores)))
].mean(axis=1)
score = float(payload["summary"]["score"])
ucb80 = float(np.quantile(sampled, 0.90))
lcb80 = float(np.quantile(sampled, 0.10))
timeout_rate = float(payload["summary"]["timeout_rate"])
passed = score >= 0.40 and ucb80 >= 0.45 and timeout_rate == 0.0
result = {
    "passed": passed,
    "episodes": int(payload["summary"]["episodes"]),
    "paired_blocks": int(len(block_scores)),
    "score": score,
    "lcb80": lcb80,
    "ucb80": ucb80,
    "minimum_score": 0.40,
    "minimum_ucb80": 0.45,
    "timeout_rate": timeout_rate,
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if passed else 2)
PY
}

unset AZK_RESUME_COMPLETED_EPISODES AZK_RESUME_ALLOW_SCHEDULE_REWIND
unset AZK_DRAFT_REF_DECK_INDICES AZK_PORTAL_OUTCOME_BONUS
unset AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP AZK_GENERATED_IKZ_CONVERSION_BONUS
unset AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".env_completed_episodes == $EXPECTED_PARENT_EPISODES and .update == 2930" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
if [[ ! -f "$RESULT_ROOT/runtime_sha256.txt" ]]; then
  sha256sum \
    python/src/tcg.h python/src/train.py \
    python/config/azuki_deckbuild_native_3090.ini \
    build/python/src/binding*.so \
    train-ablation-1781126582/run_strategy_recovery_ladder15_v1.sh \
    > "$RESULT_ROOT/runtime_sha256.txt"
  git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
  printf 'campaign=%s\npeak_lr=%s\ntarget_epoch=%s\ntotal_timesteps=%s\n' \
    "$CAMPAIGN" "$PEAK_LR" "$TARGET_EPOCH" "$TOTAL_TIMESTEPS" \
    > "$RESULT_ROOT/campaign_config.txt"
fi

for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  dedup=${tempo_dedup[$index]}
  gp=${portal_gp[$index]}
  tag="${CAMPAIGN}_${arm}"
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/$tag"
  train_log="$arm_results/train.live.log"

  mkdir -p "$arm_results"
  if [[ -f "$arm_results/ARM_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already complete; skipping"
    continue
  fi

  if [[ ! -f "$arm_results/TRAIN_DONE" ]]; then
    bash train-ablation-1781126582/run_strategy_recovery_split_arm.sh \
      "$CAMPAIGN" "$arm" "$dedup" "$gp" "$PEAK_LR"
  fi

  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    run_dir=$(<"$arm_results/run_dir.txt")
    checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
    trainer_state="$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
    if [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]]; then
      echo "[$CAMPAIGN] $arm TRAIN_DONE lacks its atomic checkpoint pair" >&2
      exit 1
    fi
  else
    if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null; then
      echo "[$CAMPAIGN] existing uncompleted output for $tag; refusing ambiguous resume" >&2
      exit 1
    fi
    if [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
      echo "[$CAMPAIGN] existing uncompleted state for $arm; refusing overwrite" >&2
      exit 1
    fi

    mkdir -p "$arm_league/opponents" "$snapshot_dir"
    cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
    cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"

    echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm dedup=$dedup portal_gp=$gp lr=$PEAK_LR"
    touch "$train_log"
    env \
    AZK_REWARD_SHAPING_ANNEAL=1 \
    AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
    AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 \
    AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
    AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 \
    AZK_PORTAL_GP_BONUS="$gp" \
    AZK_PORTAL_OUTCOME_BONUS=0 \
    AZK_XGATE_MASK=1 \
    AZK_PFSP=1 \
    AZK_EARLY_TEMPO_BONUS=0.1 \
    AZK_EARLY_TEMPO_CAP=4 \
    AZK_EARLY_TEMPO_TURNS=2 \
    AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES="$dedup" \
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
      --resume-checkpoint "$PARENT_MODEL" \
      --resume-load-optimizer \
      --resume-restart-lr-schedule \
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
    startup_failed=0
    guard_failed=0
    last_guard_count=0
    while kill -0 "$train_pid" 2>/dev/null; do
      if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$train_log"; then
        if ! grep -q "total_epochs=$TARGET_EPOCH" "$train_log" ||
           ! grep -q "AZK_RESUME_COMPLETED_EPISODES=$EXPECTED_PARENT_EPISODES" "$train_log" ||
           ! grep -q 'remaining_epochs=970' "$train_log"; then
          echo "[$CAMPAIGN] $arm startup invariant failure" \
            | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          startup_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
        startup_checked=1
        touch "$arm_results/STARTUP_INVARIANTS_OK"
        echo "[$CAMPAIGN] $arm startup invariants passed"
      fi

      jsonl=$(latest_jsonl "$tag")
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
if len(rows) < 40:
    print(len(rows), 0.0, 0.0)
else:
    print(len(rows), statistics.median(rows[-40:-20]), statistics.median(rows[-20:]))
PY
        )
        read -r guard_count previous_median current_median <<< "$guard"
        if (( guard_count >= 40 && guard_count > last_guard_count )); then
          last_guard_count=$guard_count
          if awk -v a="$previous_median" -v b="$current_median" -v t="$SPS_HARD_FLOOR" \
            'BEGIN { exit !((a < t) && (b < t)) }'; then
            printf 'previous_median=%s\ncurrent_median=%s\nthreshold=%s\n' \
              "$previous_median" "$current_median" "$SPS_HARD_FLOOR" \
              | tee "$arm_results/SPS_GUARD_FAILED"
            guard_failed=1
            kill -INT "$train_pid" 2>/dev/null || true
            break
          fi
        fi
      fi
      sleep 5
    done

    set +e
    wait "$train_pid"
    train_status=$?
    set -e
    if (( guard_failed )); then
      exit 2
    fi
    if (( startup_failed )); then
      exit 1
    fi
    if (( train_status != 0 )); then
      echo "[$CAMPAIGN] $arm training exited with status $train_status" >&2
      exit "$train_status"
    fi
    if (( ! startup_checked )); then
      echo "[$CAMPAIGN] $arm exited before startup invariants were verified" >&2
      exit 1
    fi

    run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
    checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
    trainer_state="$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
    if [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]]; then
      echo "[$CAMPAIGN] $arm ended without the p$TARGET_EPOCH atomic checkpoint" >&2
      exit 1
    fi
    jsonl=$(latest_jsonl "$tag")
    if [[ -z "$jsonl" ]]; then
      echo "[$CAMPAIGN] $arm completed without JSONL metrics" >&2
      exit 1
    fi
    cp -a "$jsonl" "$arm_results/train.jsonl"
    cp -a "$train_log" "$arm_results/train.log"
    printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
    sha256sum "$checkpoint" "$trainer_state" > "$arm_results/checkpoint_sha256.txt"
    sps_summary "$arm_results/train.jsonl" > "$arm_results/sps_summary.txt"
    touch "$arm_results/TRAIN_DONE"
  fi

  run_h2h "$checkpoint" "$PARENT_MODEL" "$arm" p2930_parent \
    "$arm_results/h2h_vs_parent.json"

  if [[ "$arm" == control ]]; then
    if ! write_control_strength_gate \
      "$arm_results/h2h_vs_parent.json" "$arm_results/control_strength_gate.json"; then
      touch "$RESULT_ROOT/CONTROL_STRENGTH_FAILED"
      echo "[$CAMPAIGN] control strength gate failed; candidate arms will not train" >&2
      exit 3
    fi
  else
    control_run_dir=$(<"$RESULT_ROOT/control/run_dir.txt")
    control_checkpoint="$control_run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
    run_h2h "$checkpoint" "$control_checkpoint" "$arm" control \
      "$arm_results/h2h_vs_control.json"
  fi

  run_reference_eval "$checkpoint" "$arm" train "$TRAIN_INDICES" \
    "$arm_results/draftref_train.json"
  run_reference_eval "$checkpoint" "$arm" holdout "$HOLDOUT_INDICES" \
    "$arm_results/draftref_holdout.json"

  if [[ ! -f "$arm_results/decks.json" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python train-ablation-1781126582/dump_gate_decks.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --episodes 24 --device cuda \
      --json "$arm_results/decks.json" \
      > >(tee "$arm_results/decks.log") 2>&1
  fi
  if [[ ! -f "$arm_results/gate_kl.json" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --episodes 24 --device cuda \
      --json "$arm_results/gate_kl.json" \
      > >(tee "$arm_results/gate_kl.log") 2>&1
  fi
  run_opportunity_eval "$checkpoint" "$arm" "$dedup" "$gp" "$arm_results"

  if [[ "$arm" != control ]]; then
    read -r _ arm_sps _ < "$arm_results/sps_summary.txt"
    read -r _ control_sps _ < "$RESULT_ROOT/control/sps_summary.txt"
    ratio=$(awk -v a="$arm_sps" -v c="$control_sps" 'BEGIN { printf "%.6f", a / c }')
    printf 'arm_median=%s\ncontrol_median=%s\nratio=%s\nthreshold=%s\n' \
      "$arm_sps" "$control_sps" "$ratio" "$SPS_RELATIVE_FLOOR" \
      > "$arm_results/sps_relative_to_control.txt"
    if awk -v ratio="$ratio" -v floor="$SPS_RELATIVE_FLOOR" \
      'BEGIN { exit !(ratio < floor) }'; then
      touch "$arm_results/SPS_RELATIVE_GUARD_FAILED"
      echo "[$CAMPAIGN] $arm median SPS ratio $ratio is below $SPS_RELATIVE_FLOOR" >&2
      exit 4
    fi
  fi

  touch "$arm_results/ARM_DONE"
  echo "[$CAMPAIGN] $(date --iso-8601=seconds) completed $arm at p$TARGET_EPOCH"
done

PYTHONPATH=python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/strategy_recovery_ladder_report.py \
  "$RESULT_ROOT" --json "$RESULT_ROOT/ladder_report.json" \
  --markdown "$RESULT_ROOT/ladder_report.md"
touch "$RESULT_ROOT/LADDER_DONE"
echo "[$CAMPAIGN] LADDER_DONE"
