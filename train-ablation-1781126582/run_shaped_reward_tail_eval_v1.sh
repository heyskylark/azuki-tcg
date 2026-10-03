#!/usr/bin/env bash
# Post-training trajectory evaluation for shaped_reward_tail45_v1.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${SHAPED_REWARD_TAIL_CAMPAIGN:-shaped_reward_tail45_v1}
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
EVAL_ROOT="$RESULT_ROOT/eval"
PARENT_MODEL=experiments/azuki_local_strategy_recovery_leader15_v1_control_178447770084/model_azuki_local_004870.pt
EXPECTED_PARENT_SHA=def350888ecbf9014e590fb8540080cb26b6c14852882caa85def3f6db8a4f76
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17
OPPORTUNITY_GAMES=64
OPPORTUNITY_SHARDS=4
OPPORTUNITY_SEED0=930001

arms=(fixed_floor zero_tail)
epochs=(6000 6900 7400 7800)

require_hash() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$CAMPAIGN-eval] $label hash mismatch: $actual" >&2
    exit 1
  fi
}

checkpoint_for() {
  local arm=$1 epoch=$2 segment
  if (( epoch <= 6000 )); then
    segment=to6000
  elif (( epoch <= 6900 )); then
    segment=to6900
  else
    segment=to7800
  fi
  local run_dir
  run_dir=$(<"$RESULT_ROOT/$arm/${segment}_run_dir.txt")
  printf '%s/model_azuki_local_%06d.pt\n' "$run_dir" "$epoch"
}

run_h2h() {
  local checkpoint_a=$1 checkpoint_b=$2 label_a=$3 label_b=$4 output=$5
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_policy_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint-a "$checkpoint_a" --checkpoint-b "$checkpoint_b" \
      --label-a "$label_a" --label-b "$label_b" --batch-envs 12 \
      --seeds 43001701,53001704,63001707,73001710,83001713,93001716 \
      --max-steps 600 --device cuda --json "$output" \
      > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '.summary.episodes == 192 and (.games | length) == 192 and .summary.timeout_rate == 0' \
    "$output" >/dev/null
}

run_reference_eval() {
  local checkpoint=$1 arm=$2 epoch=$3 split=$4 indices=$5 output=$6
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_reference_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --opponent-checkpoint "$PARENT_MODEL" \
      --candidate-label "${arm}_p${epoch}" --opponent-label p4870_parent \
      --split "$split" --deck-indices "$indices" \
      --training-reference-indices "$TRAIN_INDICES" \
      --holdout-reference-indices "$HOLDOUT_INDICES" \
      --seeds 43009919,53009922 --batch-envs 12 --device cuda \
      --max-steps 600 --json "$output" \
      > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '.summary.episodes == 288 and (.games | length) == 288 and .summary.timeout_rate == 0' \
    "$output" >/dev/null
}

run_opportunity_eval() {
  local checkpoint=$1 arm=$2 epoch=$3 output_dir=$4
  local games_per_shard=$((OPPORTUNITY_GAMES / OPPORTUNITY_SHARDS))
  mkdir -p "$output_dir"
  local pids=()
  local shard
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    local log="$output_dir/shard${shard}.log"
    local shard_seed=$((OPPORTUNITY_SEED0 + 7919 * games_per_shard * shard))
    if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]; then
      continue
    fi
    rm -f "$temporary"
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python train-ablation-1781126582/play_selfplay_games.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --games "$games_per_shard" \
      --seed0 "$shard_seed" --device cpu --log-legal-actions \
      --out "$temporary" > "$log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  (( failed == 0 ))
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$output"
    fi
    [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]
  done

  local effective_scale=0.15
  if [[ "$arm" == zero_tail ]] && (( epoch >= 6900 )); then
    effective_scale=0
  fi
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_opportunity_rates.py \
    "$output_dir"/shard*.jsonl --label "${arm}_p${epoch}" \
    --early-tempo-bonus 0.1 --early-tempo-cap 4 \
    --portal-gp-bonus 0.3 --shaping-scale "$effective_scale" \
    --early-tempo-dedup-portal-abilities \
    --json "$output_dir/opportunity.json" \
    > >(tee "$output_dir/opportunity.log") 2>&1
  jq -e ".n_games == $OPPORTUNITY_GAMES" "$output_dir/opportunity.json" >/dev/null
}

[[ -f "$RESULT_ROOT/TRAINING_DONE" ]] || {
  echo "[$CAMPAIGN-eval] training is not complete" >&2
  exit 1
}
require_hash "$PARENT_MODEL" "$EXPECTED_PARENT_SHA" parent-model
mkdir -p "$EVAL_ROOT"
if [[ ! -f "$EVAL_ROOT/runtime_sha256.txt" ]]; then
  sha256sum \
    "$PARENT_MODEL" python/src/native_policy_eval.py python/src/native_reference_eval.py \
    train-ablation-1781126582/play_selfplay_games.py \
    train-ablation-1781126582/analyze_opportunity_rates.py \
    train-ablation-1781126582/dump_gate_decks.py \
    train-ablation-1781126582/probe_gate_kl.py \
    train-ablation-1781126582/shaped_reward_tail_report.py \
    train-ablation-1781126582/run_shaped_reward_tail_eval_v1.sh \
    > "$EVAL_ROOT/runtime_sha256.txt"
fi

for epoch in "${epochs[@]}"; do
  fixed_checkpoint=$(checkpoint_for fixed_floor "$epoch")
  tail_checkpoint=$(checkpoint_for zero_tail "$epoch")
  [[ -f "$fixed_checkpoint" && -f "$tail_checkpoint" ]]
  epoch_root="$EVAL_ROOT/p$epoch"
  mkdir -p "$epoch_root"

  run_h2h "$tail_checkpoint" "$fixed_checkpoint" \
    "zero_tail_p${epoch}" "fixed_floor_p${epoch}" "$epoch_root/h2h_tail_vs_floor.json"
  for arm in "${arms[@]}"; do
    checkpoint=$(checkpoint_for "$arm" "$epoch")
    arm_root="$epoch_root/$arm"
    mkdir -p "$arm_root"
    run_h2h "$checkpoint" "$PARENT_MODEL" "${arm}_p${epoch}" p4870_parent \
      "$arm_root/h2h_vs_parent.json"
    run_reference_eval "$checkpoint" "$arm" "$epoch" holdout "$HOLDOUT_INDICES" \
      "$arm_root/draftref_holdout.json"
    if (( epoch == 7800 )); then
      run_reference_eval "$checkpoint" "$arm" "$epoch" train "$TRAIN_INDICES" \
        "$arm_root/draftref_train.json"
    fi
    if [[ ! -f "$arm_root/decks.json" ]]; then
      PYTHONPATH=build/python/src:python/src \
      .venv/bin/python train-ablation-1781126582/dump_gate_decks.py \
        --config python/config/azuki_deckbuild_3090.ini \
        --checkpoint "$checkpoint" --episodes 48 --device cuda \
        --json "$arm_root/decks.json" \
        > >(tee "$arm_root/decks.log") 2>&1
    fi
    if [[ ! -f "$arm_root/gate_kl.json" ]]; then
      PYTHONPATH=build/python/src:python/src \
      .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
        --config python/config/azuki_deckbuild_3090.ini \
        --checkpoint "$checkpoint" --episodes 48 --device cuda \
        --json "$arm_root/gate_kl.json" \
        > >(tee "$arm_root/gate_kl.log") 2>&1
    fi
  done
done

# CPU opportunity traces run after GPU panels so they cannot perturb inference.
for epoch in "${epochs[@]}"; do
  for arm in "${arms[@]}"; do
    checkpoint=$(checkpoint_for "$arm" "$epoch")
    run_opportunity_eval "$checkpoint" "$arm" "$epoch" \
      "$EVAL_ROOT/p$epoch/$arm/opportunity"
  done
done

PYTHONPATH=python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/shaped_reward_tail_report.py \
  "$RESULT_ROOT" --json "$RESULT_ROOT/trajectory_report.json" \
  --markdown "$RESULT_ROOT/trajectory_report.md" \
  --decks-markdown "$RESULT_ROOT/deck_compositions.md"

touch "$RESULT_ROOT/EVAL_DONE"
echo "[$CAMPAIGN-eval] EVAL_DONE"
