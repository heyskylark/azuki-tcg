#!/usr/bin/env bash
# Windowed Stage 1 evaluation under the permanent uniform context lifecycle.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${UNIFORM_ASSIGNMENT_CAMPAIGN:-uniform_assignment_ladder15_v2}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage1/$CAMPAIGN"
EVAL_ROOT="$RESULT_ROOT/evaluation"
PARENT_MODEL=experiments/azuki_local_strategy_recovery_leader15_v1_control_178447770084/model_azuki_local_004870.pt
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17
HYBRID_INDICES=1,9,17
HYBRID_GAMES=384
HYBRID_SHARDS=${UNIFORM_ASSIGNMENT_HYBRID_SHARDS:-12}
REFERENCE_MAX_STEPS=${UNIFORM_ASSIGNMENT_REFERENCE_MAX_STEPS:-1200}
if [[ ! "$REFERENCE_MAX_STEPS" =~ ^[0-9]+$ ]] || (( REFERENCE_MAX_STEPS < 600 )); then
  echo "[$CAMPAIGN-eval] reference max steps must be an integer at least 600" >&2
  exit 1
fi
if (( HYBRID_GAMES % HYBRID_SHARDS != 0 )); then
  echo "[$CAMPAIGN-eval] hybrid games must divide evenly across shards" >&2
  exit 1
fi
HYBRID_LINES_PER_SHARD=$((HYBRID_GAMES / HYBRID_SHARDS))
read -r -a EVAL_EPOCHS <<< \
  "${UNIFORM_ASSIGNMENT_EVAL_EPOCHS:-5000 5200 5500 5840}"
read -r -a HYBRID_EPOCHS <<< \
  "${UNIFORM_ASSIGNMENT_HYBRID_EPOCHS:-5200 5500 5840}"
if (( ${#EVAL_EPOCHS[@]} < 2 || ${#HYBRID_EPOCHS[@]} < 1 )); then
  echo "[$CAMPAIGN-eval] invalid evaluation epoch lists" >&2
  exit 1
fi
for epoch in "${EVAL_EPOCHS[@]}" "${HYBRID_EPOCHS[@]}"; do
  if [[ ! "$epoch" =~ ^[0-9]+$ ]]; then
    echo "[$CAMPAIGN-eval] invalid epoch: $epoch" >&2
    exit 1
  fi
done
for epoch in "${HYBRID_EPOCHS[@]}"; do
  if [[ " ${EVAL_EPOCHS[*]} " != *" $epoch "* ]]; then
    echo "[$CAMPAIGN-eval] hybrid epoch $epoch has no full evaluation window" >&2
    exit 1
  fi
done

if [[ ! -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]]; then
  echo "[$CAMPAIGN-eval] waiting for matched training"
  while [[ ! -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]]; do
    if [[ -f "$RESULT_ROOT/control/SPS_GUARD_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/uniform_assignment/SPS_GUARD_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/control/STARTUP_INVARIANTS_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/uniform_assignment/STARTUP_INVARIANTS_FAILED" ]]; then
      echo "[$CAMPAIGN-eval] training guard failed; refusing efficacy evaluation" >&2
      exit 1
    fi
    sleep 30
  done
fi

control_run=$(<"$RESULT_ROOT/control/run_dir.txt")
candidate_run=$(<"$RESULT_ROOT/uniform_assignment/run_dir.txt")

checkpoint_for() {
  local arm=$1 epoch=$2
  case "$arm" in
    parent) printf '%s\n' "$PARENT_MODEL" ;;
    control) printf '%s/model_azuki_local_%06d.pt\n' "$control_run" "$epoch" ;;
    uniform_assignment) printf '%s/model_azuki_local_%06d.pt\n' "$candidate_run" "$epoch" ;;
    *) echo "unknown arm: $arm" >&2; return 1 ;;
  esac
}

run_uniform_panel() {
  local checkpoint_a=$1 checkpoint_b=$2 label_a=$3 label_b=$4 output=$5
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/uniform_context_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint-a "$checkpoint_a" --checkpoint-b "$checkpoint_b" \
      --label-a "$label_a" --label-b "$label_b" \
      --games-per-context 24 --batch-envs 48 --max-steps 600 --device cuda \
      --json "$output" --md "${output%.json}.md" \
      > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '
    .summary.episodes == 384 and .summary.timeout_rate == 0 and
    (.games | length) == 384 and (.control_games | length) == 384 and
    (.summary.by_context | length) == 16 and
    (.summary.policy_b.by_context | length) == 16
  ' "$output" >/dev/null
}

run_reference_panel() {
  local checkpoint=$1 label=$2 output=$3
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_reference_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --opponent-checkpoint "$PARENT_MODEL" \
      --candidate-label "$label" --opponent-label p4870_parent \
      --split heldout --deck-indices "$HOLDOUT_INDICES" \
      --training-reference-indices "$TRAIN_INDICES" \
      --holdout-reference-indices "$HOLDOUT_INDICES" \
      --seeds 42009919,52009922 --batch-envs 12 \
      --max-steps "$REFERENCE_MAX_STEPS" --device cuda \
      --uniform-assignment \
      --json "$output" > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '
    .summary.episodes == 288 and .summary.timeout_rate == 0 and
    .assignment_contract == "uniform_gate_same_element_leader" and
    (.summary.by_candidate_leader | length) == 8 and
    (.summary.by_candidate_context | length) == 16 and
    (.games | length) == 288
  ' "$output" >/dev/null
}

run_context_decks() {
  local checkpoint=$1 output_dir=$2
  if [[ ! -f "$output_dir/context_decks.json" ]]; then
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/dump_context_decks_native.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --drafts-per-context 24 \
      --temperature 1.2 --smoothing-eps 0.05 --batch-envs 48 --device cuda \
      --json "$output_dir/context_decks.json" \
      --md "$output_dir/context_decks.md" \
      > >(tee "$output_dir/context_decks.log") 2>&1
  fi
  jq -e '
    .stochastic_drafts_per_context == 24 and (.contexts | length) == 16 and
    .evaluator.version == "native-batched-draft-v1" and
    ([.contexts[].greedy.summary.main_total] | all(. == 50)) and
    ([.contexts[].stochastic.decks | length] | all(. == 24)) and
    ([.contexts[].stochastic.decks[].summary.main_total] | all(. == 50))
  ' "$output_dir/context_decks.json" >/dev/null
}

run_context_kl() {
  local checkpoint=$1 output_dir=$2
  if [[ ! -f "$output_dir/context_kl.json" ]]; then
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/probe_context_kl_native.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --histories 4 \
      --temperature 1.2 --smoothing-eps 0.05 --batch-envs 128 --device cuda \
      --json "$output_dir/context_kl.json" --md "$output_dir/context_kl.md" \
      > >(tee "$output_dir/context_kl.log") 2>&1
  fi
  jq -e '
    .histories_per_direction == 4 and
    .evaluator.version == "native-batched-context-replay-v1" and
    .aggregate.gate_given_leader.pick_rows == 3200 and
    .aggregate.leader_given_gate.pick_rows == 3200 and
    ([.aggregate[].determinism_kl_max] | max) < 1e-7
  ' "$output_dir/context_kl.json" >/dev/null
}

mkdir -p "$EVAL_ROOT"
sha256sum \
  train-ablation-1781126582/run_uniform_assignment_eval_v1.sh \
  train-ablation-1781126582/run_uniform_assignment_confirm45_eval_v1.sh \
  train-ablation-1781126582/uniform_context_eval.py \
  train-ablation-1781126582/dump_context_decks.py \
  train-ablation-1781126582/dump_context_decks_native.py \
  train-ablation-1781126582/probe_gate_kl.py \
  train-ablation-1781126582/probe_context_kl.py \
  train-ablation-1781126582/probe_context_kl_native.py \
  train-ablation-1781126582/build_context_hybrid_arms.py \
  train-ablation-1781126582/analyze_context_hybrids.py \
  train-ablation-1781126582/run_fixed_deck_counterfactual.py \
  train-ablation-1781126582/uniform_assignment_ladder_report.py \
  python/src/native_reference_eval.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py build/python/src/binding*.so \
  > "$EVAL_ROOT/runtime_sha256.txt"

for epoch in "${EVAL_EPOCHS[@]}"; do
  epoch_dir="$EVAL_ROOT/p$epoch"
  mkdir -p "$epoch_dir/control" "$epoch_dir/uniform_assignment"
  control_checkpoint=$(checkpoint_for control "$epoch")
  candidate_checkpoint=$(checkpoint_for uniform_assignment "$epoch")
  [[ -f "$control_checkpoint" && -f "$candidate_checkpoint" ]]

  run_uniform_panel \
    "$candidate_checkpoint" "$control_checkpoint" \
    "uniform_p$epoch" "control_p$epoch" \
    "$epoch_dir/uniform_vs_control.json"
  run_uniform_panel \
    "$candidate_checkpoint" "$PARENT_MODEL" \
    "uniform_p$epoch" p4870_parent \
    "$epoch_dir/uniform_vs_parent.json"
  run_uniform_panel \
    "$control_checkpoint" "$PARENT_MODEL" \
    "control_p$epoch" p4870_parent \
    "$epoch_dir/control_vs_parent.json"

  run_reference_panel \
    "$candidate_checkpoint" "uniform_p$epoch" \
    "$epoch_dir/uniform_assignment/heldout.json"
  run_reference_panel \
    "$control_checkpoint" "control_p$epoch" \
    "$epoch_dir/control/heldout.json"

  run_context_decks "$candidate_checkpoint" "$epoch_dir/uniform_assignment"
  run_context_decks "$control_checkpoint" "$epoch_dir/control"
  run_context_kl "$candidate_checkpoint" "$epoch_dir/uniform_assignment"
  run_context_kl "$control_checkpoint" "$epoch_dir/control"
done

run_context_hybrids() {
  local checkpoint=$1 label=$2 output_dir=$3
  local arms="$output_dir/context_hybrid_arms.json"
  if [[ ! -f "$arms" ]]; then
    PYTHONPATH=python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/build_context_hybrid_arms.py \
      --decks "$output_dir/context_decks.json" --json "$arms" \
      > >(tee "$output_dir/context_hybrid_arms.log") 2>&1
  fi
  mkdir -p "$output_dir/context_hybrid"
  local pids=()
  local shard
  for ((shard = 0; shard < HYBRID_SHARDS; shard++)); do
    local output="$output_dir/context_hybrid/shard${shard}.jsonl"
    local log="$output_dir/context_hybrid/shard${shard}.log"
    if [[ -f "$output" ]] &&
       [[ $(wc -l < "$output") -eq "$HYBRID_LINES_PER_SHARD" ]]; then
      continue
    fi
    rm -f "$output"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python train-ablation-1781126582/run_fixed_deck_counterfactual.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --deck-arms "$arms" \
      --reference-indices "$HYBRID_INDICES" --device cpu \
      --shards "$HYBRID_SHARDS" --shard-index "$shard" \
      --out "$output" > "$log" 2>&1 &
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
    [[ $(wc -l < "$output_dir/context_hybrid/shard${shard}.jsonl") \
      -eq "$HYBRID_LINES_PER_SHARD" ]]
  done
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_context_hybrids.py \
    "$output_dir"/context_hybrid/shard*.jsonl --label "$label" \
    --json "$output_dir/context_hybrid.json" \
    > >(tee "$output_dir/context_hybrid.log") 2>&1
  jq -e '.games == 384 and .paired_schedule == true and (.contexts | length) == 16' \
    "$output_dir/context_hybrid.json" >/dev/null
}

# Fixed-deck causal games use CPU only and start after every GPU panel is done.
for epoch in "${HYBRID_EPOCHS[@]}"; do
  for arm in control uniform_assignment; do
    checkpoint=$(checkpoint_for "$arm" "$epoch")
    run_context_hybrids "$checkpoint" "${arm}_p${epoch}" \
      "$EVAL_ROOT/p$epoch/$arm"
  done
done

PYTHONPATH=python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/uniform_assignment_ladder_report.py \
  --root "$RESULT_ROOT" --json "$RESULT_ROOT/ladder_report.json" \
  --md "$RESULT_ROOT/ladder_report.md" \
  --epochs "$(IFS=,; echo "${EVAL_EPOCHS[*]}")" \
  --hybrid-epochs "$(IFS=,; echo "${HYBRID_EPOCHS[*]}")"

touch "$RESULT_ROOT/EVALUATION_DONE"
echo "[$CAMPAIGN-eval] complete"
