#!/usr/bin/env bash
# Windowed strength, drafting, playstyle, and causal deck evaluation for Step 5b.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${DRAFT_CREDIT_LADDER_CAMPAIGN:-draft_terminal_credit_ladder15_v1}
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
PARTIAL_MODE=${DRAFT_CREDIT_PARTIAL_MODE:-0}
if [[ "$PARTIAL_MODE" == 1 ]]; then
  TRAJECTORY_ROOT="$RESULT_ROOT/partial_trajectory"
  EVAL_EPOCHS=(5200 5500)
  FINAL_EVAL_EPOCH=5500
else
  TRAJECTORY_ROOT="$RESULT_ROOT/trajectory"
  EVAL_EPOCHS=(5200 5500 5847)
  FINAL_EVAL_EPOCH=5847
fi
PARENT_MODEL=experiments/azuki_local_strategy_recovery_leader15_v1_control_178447770084/model_azuki_local_004870.pt
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17
HYBRID_INDICES=1,5,9,13,17
OPPORTUNITY_GAMES=64
OPPORTUNITY_SHARDS=4
HYBRID_SHARDS=4

if [[ "$PARTIAL_MODE" == 1 ]]; then
  if [[ ! -f "$RESULT_ROOT/draft_terminal_long/SPS_GUARD_FAILED" ]]; then
    echo "[$CAMPAIGN] partial evaluation requires an SPS-guard stop" >&2
    exit 1
  fi
elif [[ ! -f "$RESULT_ROOT/TRAINING_DONE" ]]; then
  echo "[$CAMPAIGN] training must finish before trajectory evaluation" >&2
  exit 1
fi

control_run=$(<"$RESULT_ROOT/control/run_dir.txt")
if [[ "$PARTIAL_MODE" == 1 ]]; then
  candidate_runs=()
  for path in experiments/"azuki_local_${CAMPAIGN}_draft_terminal_long_"*; do
    [[ -d "$path" ]] && candidate_runs+=("$path")
  done
  if (( ${#candidate_runs[@]} != 1 )); then
    echo "[$CAMPAIGN] expected one interrupted candidate run, found ${#candidate_runs[@]}" >&2
    exit 1
  fi
  candidate_run=${candidate_runs[0]}
  printf '%s\n' "$candidate_run" > "$RESULT_ROOT/draft_terminal_long/partial_run_dir.txt"
  candidate_jsonls=()
  for path in experiments/runlogs/"${CAMPAIGN}_draft_terminal_long_"*.jsonl; do
    [[ -f "$path" ]] && candidate_jsonls+=("$path")
  done
  if (( ${#candidate_jsonls[@]} != 1 )); then
    echo "[$CAMPAIGN] expected one interrupted candidate metric log" >&2
    exit 1
  fi
  cp -a "${candidate_jsonls[0]}" \
    "$RESULT_ROOT/draft_terminal_long/partial_train.jsonl"
  sha256sum "$RESULT_ROOT/draft_terminal_long/partial_train.jsonl" \
    > "$RESULT_ROOT/draft_terminal_long/partial_train_sha256.txt"
else
  candidate_run=$(<"$RESULT_ROOT/draft_terminal_long/run_dir.txt")
fi

checkpoint_for() {
  local arm=$1 epoch=$2
  if [[ "$arm" == parent ]]; then
    printf '%s\n' "$PARENT_MODEL"
  elif [[ "$arm" == control ]]; then
    printf '%s/model_azuki_local_%06d.pt\n' "$control_run" "$epoch"
  else
    printf '%s/model_azuki_local_%06d.pt\n' "$candidate_run" "$epoch"
  fi
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
  local checkpoint=$1 label=$2 split=$3 indices=$4 output=$5
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_reference_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --opponent-checkpoint "$PARENT_MODEL" \
      --candidate-label "$label" --opponent-label p4870_parent \
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

run_deck_diagnostics() {
  local checkpoint=$1 output_dir=$2
  if [[ ! -f "$output_dir/decks.json" ]]; then
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/dump_gate_decks.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --episodes 32 --device cuda \
      --json "$output_dir/decks.json" \
      > >(tee "$output_dir/decks.log") 2>&1
  fi
  jq -e '.sampled_episodes == 32 and ([.gates[].sampled.decks | length] | all(. == 32))' \
    "$output_dir/decks.json" >/dev/null

  if [[ ! -f "$output_dir/draft_conditioning.json" ]]; then
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/probe_draft_conditioning.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --episodes 8 --device cuda \
      --json "$output_dir/draft_conditioning.json" \
      > >(tee "$output_dir/draft_conditioning.log") 2>&1
  fi
  jq -e '
    .leader_row_excluded == true and .replayed_action_history_fixed == true and
    ([.sibling_gate[].main_pick_steps] | all(. == 400)) and
    ([.sibling_gate[].control_max_kl] | max) < 1e-6
  ' "$output_dir/draft_conditioning.json" >/dev/null
}

run_opportunity_eval() {
  local checkpoint=$1 label=$2 output_dir=$3
  local games_per_shard=$((OPPORTUNITY_GAMES / OPPORTUNITY_SHARDS))
  mkdir -p "$output_dir/opportunity"
  local pids=()
  local shard
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/opportunity/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    local log="$output_dir/opportunity/shard${shard}.log"
    local seed=$((8300001 + 7919 * games_per_shard * shard))
    if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]; then
      continue
    fi
    rm -f "$temporary"
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python train-ablation-1781126582/play_selfplay_games.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --games "$games_per_shard" \
      --seed0 "$seed" --device cpu --log-legal-actions \
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
    local output="$output_dir/opportunity/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$output"
    fi
    [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]
  done
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_opportunity_rates.py \
    "$output_dir"/opportunity/shard*.jsonl --label "$label" \
    --early-tempo-bonus 0.1 --early-tempo-cap 4 \
    --portal-gp-bonus 0.3 --shaping-scale 0.15 \
    --early-tempo-dedup-portal-abilities \
    --json "$output_dir/opportunity.json" \
    | tee "$output_dir/opportunity.log"
  jq -e ".n_games == $OPPORTUNITY_GAMES" "$output_dir/opportunity.json" >/dev/null
}

run_hybrid_eval() {
  local checkpoint=$1 label=$2 output_dir=$3
  local arms="$output_dir/hybrid_arms.json"
  if [[ ! -f "$arms" ]]; then
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/build_draft_hybrid_arms.py \
      --decks "$output_dir/decks.json" --json "$arms" \
      | tee "$output_dir/hybrid_arms.log"
  fi
  mkdir -p "$output_dir/hybrid"
  local pids=()
  local shard
  for ((shard = 0; shard < HYBRID_SHARDS; shard++)); do
    local output="$output_dir/hybrid/shard${shard}.jsonl"
    local expected=80
    if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $expected ]]; then
      continue
    fi
    rm -f "$output"
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python train-ablation-1781126582/run_fixed_deck_counterfactual.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --deck-arms "$arms" \
      --reference-indices "$HYBRID_INDICES" --device cpu \
      --shards "$HYBRID_SHARDS" --shard-index "$shard" \
      --out "$output" > "$output_dir/hybrid/shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  (( failed == 0 ))
  for ((shard = 0; shard < HYBRID_SHARDS; shard++)); do
    local output="$output_dir/hybrid/shard${shard}.jsonl"
    [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq 80 ]]
  done
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_draft_hybrids.py \
    "$output_dir"/hybrid/shard*.jsonl --label "$label" \
    --json "$output_dir/hybrid.json" | tee "$output_dir/hybrid.log"
  jq -e '.games == 320 and .paired_schedule == true' "$output_dir/hybrid.json" >/dev/null
}

mkdir -p "$TRAJECTORY_ROOT"
sha256sum \
  train-ablation-1781126582/run_draft_terminal_credit_eval_v1.sh \
  train-ablation-1781126582/dump_gate_decks.py \
  train-ablation-1781126582/probe_gate_kl.py \
  train-ablation-1781126582/probe_draft_conditioning.py \
  train-ablation-1781126582/build_draft_hybrid_arms.py \
  train-ablation-1781126582/run_fixed_deck_counterfactual.py \
  train-ablation-1781126582/analyze_draft_hybrids.py \
  train-ablation-1781126582/draft_terminal_credit_ladder_report.py \
  train-ablation-1781126582/draft_terminal_credit_partial_report.py \
  train-ablation-1781126582/play_selfplay_games.py \
  train-ablation-1781126582/analyze_opportunity_rates.py \
  python/src/native_policy_eval.py python/src/native_reference_eval.py \
  > "$TRAJECTORY_ROOT/evaluator_sha256.txt"

if [[ "$PARTIAL_MODE" == 1 ]]; then
  labels=()
  arms=()
  epochs=()
else
  labels=(parent_p4870)
  arms=(parent)
  epochs=(4870)
fi
for epoch in "${EVAL_EPOCHS[@]}"; do
  labels+=("control_p${epoch}" "draft_terminal_long_p${epoch}")
  arms+=(control draft_terminal_long)
  epochs+=("$epoch" "$epoch")
done

for index in "${!labels[@]}"; do
  label=${labels[$index]}
  arm=${arms[$index]}
  epoch=${epochs[$index]}
  checkpoint=$(checkpoint_for "$arm" "$epoch")
  output_dir="$TRAJECTORY_ROOT/$label"
  if [[ ! -f "$checkpoint" ]]; then
    echo "[$CAMPAIGN] missing trajectory checkpoint: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  printf '%s\n' "$checkpoint" > "$output_dir/checkpoint.txt"
  sha256sum "$checkpoint" > "$output_dir/checkpoint_sha256.txt"

  if [[ "$arm" != parent ]]; then
    run_h2h "$checkpoint" "$PARENT_MODEL" "$label" p4870_parent \
      "$output_dir/h2h_vs_parent.json"
  fi
  run_reference_eval "$checkpoint" "$label" holdout "$HOLDOUT_INDICES" \
    "$output_dir/draftref_holdout.json"
  if [[ "$epoch" == "$FINAL_EVAL_EPOCH" || "$arm" == parent ]]; then
    run_reference_eval "$checkpoint" "$label" train "$TRAIN_INDICES" \
      "$output_dir/draftref_train.json"
  fi
  run_deck_diagnostics "$checkpoint" "$output_dir"
done

for epoch in "${EVAL_EPOCHS[@]}"; do
  candidate_dir="$TRAJECTORY_ROOT/draft_terminal_long_p${epoch}"
  run_h2h \
    "$(checkpoint_for draft_terminal_long "$epoch")" \
    "$(checkpoint_for control "$epoch")" \
    "draft_terminal_long_p${epoch}" "control_p${epoch}" \
    "$candidate_dir/h2h_vs_control.json"
done

# CPU traces run only after all GPU panels are complete, so their twelve worker
# threads cannot perturb measured training SPS or GPU evaluation throughput.
for index in "${!labels[@]}"; do
  label=${labels[$index]}
  arm=${arms[$index]}
  epoch=${epochs[$index]}
  checkpoint=$(checkpoint_for "$arm" "$epoch")
  output_dir="$TRAJECTORY_ROOT/$label"
  run_opportunity_eval "$checkpoint" "$label" "$output_dir"
  run_hybrid_eval "$checkpoint" "$label" "$output_dir"
  touch "$output_dir/WINDOW_DONE"
done

if [[ "$PARTIAL_MODE" == 1 ]]; then
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python \
    train-ablation-1781126582/draft_terminal_credit_partial_report.py \
    "$RESULT_ROOT" --epochs "${EVAL_EPOCHS[@]}" \
    --json "$RESULT_ROOT/partial_ladder_report.json" \
    --markdown "$RESULT_ROOT/partial_ladder_report.md"
  touch "$RESULT_ROOT/PARTIAL_EVAL_DONE"
  echo "[$CAMPAIGN] PARTIAL_EVAL_DONE"
else
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/draft_terminal_credit_ladder_report.py \
    "$RESULT_ROOT" --json "$RESULT_ROOT/ladder_report.json" \
    --markdown "$RESULT_ROOT/ladder_report.md"
  touch "$RESULT_ROOT/EVAL_DONE"
  echo "[$CAMPAIGN] EVAL_DONE"
fi
