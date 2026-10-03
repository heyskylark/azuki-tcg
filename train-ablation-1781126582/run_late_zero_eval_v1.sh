#!/usr/bin/env bash
# Stage 4 late-zero strength, mechanics, deck, and causal evaluation.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${LATE_ZERO_LADDER_CAMPAIGN:-late_zero_ladder15_v1}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage4/$CAMPAIGN"
EVAL_ROOT="$RESULT_ROOT/evaluation"
PARENT_MANIFEST="$RESULT_ROOT/parent_manifest.json"
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17
HYBRID_INDICES=1,9,17
HYBRID_GAMES=384
HYBRID_SHARDS=${LATE_ZERO_HYBRID_SHARDS:-12}
REFERENCE_MAX_STEPS=${LATE_ZERO_REFERENCE_MAX_STEPS:-1200}
if (( HYBRID_GAMES % HYBRID_SHARDS != 0 )); then
  echo "[$CAMPAIGN-eval] hybrid games must divide evenly across shards" >&2
  exit 1
fi
HYBRID_LINES_PER_SHARD=$((HYBRID_GAMES / HYBRID_SHARDS))
OPPORTUNITY_GAMES=96
OPPORTUNITY_SHARDS=${LATE_ZERO_OPPORTUNITY_SHARDS:-12}
if (( OPPORTUNITY_GAMES % OPPORTUNITY_SHARDS != 0 )); then
  echo "[$CAMPAIGN-eval] opportunity games must divide evenly across shards" >&2
  exit 1
fi
OPPORTUNITY_SEED0=63005301

if [[ ! -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]]; then
  echo "[$CAMPAIGN-eval] waiting for matched training"
  while [[ ! -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]]; do
    if [[ -f "$RESULT_ROOT/fixed_floor/SPS_GUARD_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/late_zero/SPS_GUARD_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/fixed_floor/STARTUP_INVARIANTS_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/late_zero/STARTUP_INVARIANTS_FAILED" ]]; then
      echo "[$CAMPAIGN-eval] training guard failed; refusing efficacy evaluation" >&2
      exit 1
    fi
    sleep 30
  done
fi

PARENT_MODEL=$(jq -er '.model.path' "$PARENT_MANIFEST")
PARENT_EPOCH=$(jq -er '.epoch' "$PARENT_MANIFEST")
TARGET_EPOCH=$(awk -F= '$1=="target_epoch" {print $2}' "$RESULT_ROOT/campaign_config.txt")
ANNEAL_END_EPOCH=$(awk -F= '$1=="anneal_end_epoch" {print $2}' "$RESULT_ROOT/campaign_config.txt")
fixed_floor_run=$(<"$RESULT_ROOT/fixed_floor/run_dir.txt")
candidate_run=$(<"$RESULT_ROOT/late_zero/run_dir.txt")

checkpoint_for() {
  local arm=$1 epoch=$2
  case "$arm" in
    parent) printf '%s\n' "$PARENT_MODEL" ;;
    fixed_floor) printf '%s/model_azuki_local_%06d.pt\n' "$fixed_floor_run" "$epoch" ;;
    late_zero) printf '%s/model_azuki_local_%06d.pt\n' "$candidate_run" "$epoch" ;;
    *) echo "unknown arm: $arm" >&2; return 1 ;;
  esac
}

mkdir -p "$EVAL_ROOT"
mapfile -t EVAL_EPOCHS < <(
  .venv/bin/python - "$fixed_floor_run" "$candidate_run" "$PARENT_EPOCH" "$TARGET_EPOCH" \
    "$ANNEAL_END_EPOCH" \
    "$EVAL_ROOT/evaluation_manifest.json" <<'PY'
import json
from pathlib import Path
import re
import sys

fixed_floor = Path(sys.argv[1])
candidate = Path(sys.argv[2])
parent = int(sys.argv[3])
target = int(sys.argv[4])
anneal_end = int(sys.argv[5])
output = Path(sys.argv[6])
pattern = re.compile(r"model_azuki_local_(\d{6})\.pt$")

def epochs(path):
  out = set()
  for checkpoint in path.glob("model_azuki_local_*.pt"):
    match = pattern.search(checkpoint.name)
    if match:
      value = int(match.group(1))
      if parent < value <= target:
        out.add(value)
  return out

available = sorted(epochs(fixed_floor).intersection(epochs(candidate)))
if target not in available:
  raise RuntimeError(f"Matched endpoint p{target} is unavailable")
pre_available = [epoch for epoch in available if epoch < anneal_end]
zero_available = [epoch for epoch in available if epoch >= anneal_end]
if len(pre_available) < 3 or len(zero_available) < 3:
  raise RuntimeError(
    f"Need at least three pre-zero and zero checkpoints; pre={pre_available}, "
    f"zero={zero_available}"
  )
pre_desired = [
  parent + round((anneal_end - parent) * fraction)
  for fraction in (0.20, 0.50, 0.80)
]
pre_selected = []
for value in pre_desired:
  epoch = min(pre_available, key=lambda item: (abs(item - value), item))
  if epoch not in pre_selected:
    pre_selected.append(epoch)
zero_desired = [anneal_end, anneal_end + (target - anneal_end) // 2, target]
zero_selected = []
for value in zero_desired:
  epoch = min(zero_available, key=lambda item: (abs(item - value), item))
  if epoch not in zero_selected:
    zero_selected.append(epoch)
if target not in zero_selected:
  zero_selected.append(target)
if len(pre_selected) < 3 or len(zero_selected) < 3:
  raise RuntimeError(
    f"Need three distinct pre-zero and zero windows; pre={pre_selected}, "
    f"zero={zero_selected}"
  )
selected = sorted(pre_selected + zero_selected)
payload = {
  "schema_version": 1,
  "parent_epoch": parent,
  "target_epoch": target,
  "anneal_end_epoch": anneal_end,
  "available_checkpoint_epochs": available,
  "evaluation_epochs": selected,
  "anchor_epochs": sorted(zero_selected),
  "causal_hybrid_epochs": sorted(zero_selected),
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("\n".join(str(value) for value in selected))
PY
)
mapfile -t HYBRID_EPOCHS < <(jq -r '.causal_hybrid_epochs[]' "$EVAL_ROOT/evaluation_manifest.json")
mapfile -t ANCHOR_EPOCHS < <(jq -r '.anchor_epochs[]' "$EVAL_ROOT/evaluation_manifest.json")

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
      --candidate-label "$label" --opponent-label stage4_parent \
      --split heldout --deck-indices "$HOLDOUT_INDICES" \
      --training-reference-indices "$TRAIN_INDICES" \
      --holdout-reference-indices "$HOLDOUT_INDICES" \
      --seeds 44009919,54009922 --batch-envs 12 \
      --max-steps "$REFERENCE_MAX_STEPS" --device cuda \
      --uniform-assignment \
      --json "$output" > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '.summary.episodes == 288 and .summary.timeout_rate == 0 and
    .assignment_contract == "uniform_gate_same_element_leader" and
    (.summary.by_candidate_leader | length) == 8 and
    (.summary.by_candidate_context | length) == 16 and
    (.games | length) == 288' \
    "$output" >/dev/null
}

run_context_decks() {
  local checkpoint=$1 output_dir=$2
  if [[ ! -f "$output_dir/context_decks.json" ]]; then
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/dump_context_decks_native.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --drafts-per-context 24 \
      --temperature 1.2 --smoothing-eps 0.05 --batch-envs 48 --device cuda \
      --json "$output_dir/context_decks.json" --md "$output_dir/context_decks.md" \
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

sha256sum \
  train-ablation-1781126582/run_late_zero_eval_v1.sh \
  train-ablation-1781126582/uniform_context_eval.py \
  train-ablation-1781126582/dump_context_decks.py \
  train-ablation-1781126582/dump_context_decks_native.py \
  train-ablation-1781126582/probe_gate_kl.py \
  train-ablation-1781126582/probe_context_kl.py \
  train-ablation-1781126582/probe_context_kl_native.py \
  train-ablation-1781126582/build_context_hybrid_arms.py \
  train-ablation-1781126582/analyze_context_hybrids.py \
  train-ablation-1781126582/play_selfplay_games.py \
  train-ablation-1781126582/analyze_opportunity_rates.py \
  train-ablation-1781126582/analyze_card_funnels.py \
  train-ablation-1781126582/run_fixed_deck_counterfactual.py \
  train-ablation-1781126582/late_zero_ladder_report.py \
  python/src/native_reference_eval.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py build/python/src/binding*.so \
  > "$EVAL_ROOT/runtime_sha256.txt"

# Finish all GPU panels before starting CPU trace and fixed-deck workers.
for epoch in "${EVAL_EPOCHS[@]}"; do
  epoch_dir="$EVAL_ROOT/p$epoch"
  mkdir -p "$epoch_dir/fixed_floor" "$epoch_dir/late_zero"
  fixed_floor_checkpoint=$(checkpoint_for fixed_floor "$epoch")
  candidate_checkpoint=$(checkpoint_for late_zero "$epoch")
  [[ -f "$fixed_floor_checkpoint" && -f "$candidate_checkpoint" ]]

  run_uniform_panel "$candidate_checkpoint" "$fixed_floor_checkpoint" \
    "late_zero_p$epoch" "fixed_floor_p$epoch" "$epoch_dir/late_zero_vs_fixed_floor.json"
done

for epoch in "${ANCHOR_EPOCHS[@]}"; do
  epoch_dir="$EVAL_ROOT/p$epoch"
  fixed_floor_checkpoint=$(checkpoint_for fixed_floor "$epoch")
  candidate_checkpoint=$(checkpoint_for late_zero "$epoch")
  run_uniform_panel "$candidate_checkpoint" "$PARENT_MODEL" \
    "late_zero_p$epoch" stage4_parent "$epoch_dir/late_zero_vs_parent.json"
  run_uniform_panel "$fixed_floor_checkpoint" "$PARENT_MODEL" \
    "fixed_floor_p$epoch" stage4_parent "$epoch_dir/fixed_floor_vs_parent.json"
  run_reference_panel "$candidate_checkpoint" "late_zero_p$epoch" \
    "$epoch_dir/late_zero/heldout.json"
  run_reference_panel "$fixed_floor_checkpoint" "fixed_floor_p$epoch" \
    "$epoch_dir/fixed_floor/heldout.json"
  run_context_decks "$candidate_checkpoint" "$epoch_dir/late_zero"
  run_context_decks "$fixed_floor_checkpoint" "$epoch_dir/fixed_floor"
  run_context_kl "$candidate_checkpoint" "$epoch_dir/late_zero"
  run_context_kl "$fixed_floor_checkpoint" "$epoch_dir/fixed_floor"
done

run_opportunity_eval() {
  local checkpoint=$1 arm=$2 output_dir=$3
  local games_per_shard=$((OPPORTUNITY_GAMES / OPPORTUNITY_SHARDS))
  mkdir -p "$output_dir/opportunity"
  (( OPPORTUNITY_GAMES % OPPORTUNITY_SHARDS == 0 ))
  local pids=()
  local shard
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/opportunity/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    local log="$output_dir/opportunity/shard${shard}.log"
    local shard_seed=$((OPPORTUNITY_SEED0 + 7919 * games_per_shard * shard))
    if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]; then
      continue
    fi
    rm -f "$temporary"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python train-ablation-1781126582/play_selfplay_games.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --games "$games_per_shard" \
      --seed0 "$shard_seed" --device cpu --log-legal-actions \
      --uniform-assignment --out "$temporary" > "$log" 2>&1 &
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
    [[ -f "$output" && $(wc -l < "$output") -eq $games_per_shard ]]
  done
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_opportunity_rates.py \
    "$output_dir"/opportunity/shard*.jsonl --label "$arm" \
    --early-tempo-bonus 0.1 --early-tempo-cap 4 --portal-gp-bonus 0.3 \
    --shaping-scale 0.15 --early-tempo-dedup-portal-abilities \
    --json "$output_dir/opportunity.json" \
    > >(tee "$output_dir/opportunity.log") 2>&1
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_card_funnels.py \
    "$output_dir"/opportunity/shard*.jsonl --label "$arm" \
    --json "$output_dir/card_funnels.json" \
    > >(tee "$output_dir/card_funnels.log") 2>&1
  jq -e ".n_games == $OPPORTUNITY_GAMES" "$output_dir/opportunity.json" >/dev/null
  jq -e ".games == $OPPORTUNITY_GAMES and .seat_games == (2 * $OPPORTUNITY_GAMES)" \
    "$output_dir/card_funnels.json" >/dev/null
}

for arm in fixed_floor late_zero; do
  checkpoint=$(checkpoint_for "$arm" "$TARGET_EPOCH")
  run_opportunity_eval "$checkpoint" "$arm" "$EVAL_ROOT/p$TARGET_EPOCH/$arm"
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
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
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

for epoch in "${HYBRID_EPOCHS[@]}"; do
  for arm in fixed_floor late_zero; do
    checkpoint=$(checkpoint_for "$arm" "$epoch")
    run_context_hybrids "$checkpoint" "${arm}_p${epoch}" "$EVAL_ROOT/p$epoch/$arm"
  done
done

PYTHONPATH=python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/late_zero_ladder_report.py \
  --root "$RESULT_ROOT" --json "$RESULT_ROOT/ladder_report.json" \
  --md "$RESULT_ROOT/ladder_report.md"

touch "$RESULT_ROOT/EVALUATION_DONE"
echo "[$CAMPAIGN-eval] complete"
