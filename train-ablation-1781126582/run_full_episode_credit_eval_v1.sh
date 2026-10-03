#!/usr/bin/env bash
# Stage 2 strength, context conditioning, deck composition, and causal hybrid evaluation.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${FULL_CREDIT_LADDER_CAMPAIGN:-full_episode_credit_ladder15_v1}
CANDIDATE_ARM=${FULL_CREDIT_CANDIDATE_ARM:-full_episode_credit}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$CAMPAIGN"
EVAL_ROOT="$RESULT_ROOT/evaluation"
PARENT_MANIFEST="$RESULT_ROOT/parent_manifest.json"
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17
HYBRID_INDICES=1,9,17
HYBRID_GAMES=384
HYBRID_SHARDS=${FULL_CREDIT_HYBRID_SHARDS:-12}
REFERENCE_MAX_STEPS=${FULL_CREDIT_REFERENCE_MAX_STEPS:-1200}
if (( HYBRID_GAMES % HYBRID_SHARDS != 0 )); then
  echo "[$CAMPAIGN-eval] hybrid games must divide evenly across shards" >&2
  exit 1
fi
HYBRID_LINES_PER_SHARD=$((HYBRID_GAMES / HYBRID_SHARDS))

if [[ ! -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]]; then
  echo "[$CAMPAIGN-eval] waiting for matched training"
  while [[ ! -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]]; do
    if [[ -f "$RESULT_ROOT/control/SPS_GUARD_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/$CANDIDATE_ARM/SPS_GUARD_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/control/STARTUP_INVARIANTS_FAILED" ]] ||
       [[ -f "$RESULT_ROOT/$CANDIDATE_ARM/STARTUP_INVARIANTS_FAILED" ]]; then
      echo "[$CAMPAIGN-eval] training guard failed; refusing efficacy evaluation" >&2
      exit 1
    fi
    sleep 30
  done
fi

PARENT_MODEL=$(jq -er '.model.path' "$PARENT_MANIFEST")
PARENT_EPOCH=$(jq -er '.epoch' "$PARENT_MANIFEST")
TARGET_EPOCH=$(awk -F= '$1=="target_epoch" {print $2}' "$RESULT_ROOT/campaign_config.txt")
control_run=$(<"$RESULT_ROOT/control/run_dir.txt")
candidate_run=$(<"$RESULT_ROOT/$CANDIDATE_ARM/run_dir.txt")

checkpoint_for() {
  local arm=$1 epoch=$2
  case "$arm" in
    parent) printf '%s\n' "$PARENT_MODEL" ;;
    control) printf '%s/model_azuki_local_%06d.pt\n' "$control_run" "$epoch" ;;
    "$CANDIDATE_ARM") printf '%s/model_azuki_local_%06d.pt\n' "$candidate_run" "$epoch" ;;
    *) echo "unknown arm: $arm" >&2; return 1 ;;
  esac
}

mkdir -p "$EVAL_ROOT"
mapfile -t EVAL_EPOCHS < <(
  .venv/bin/python - "$control_run" "$candidate_run" "$PARENT_EPOCH" "$TARGET_EPOCH" \
    "$EVAL_ROOT/evaluation_manifest.json" <<'PY'
import json
from pathlib import Path
import re
import sys

control = Path(sys.argv[1])
candidate = Path(sys.argv[2])
parent = int(sys.argv[3])
target = int(sys.argv[4])
output = Path(sys.argv[5])
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

available = sorted(epochs(control).intersection(epochs(candidate)))
if target not in available:
  raise RuntimeError(f"Matched endpoint p{target} is unavailable")
desired = [parent + round((target - parent) * fraction) for fraction in (0.15, 0.35, 0.65)]
selected = []
for value in desired:
  epoch = min(available, key=lambda item: (abs(item - value), item))
  if epoch not in selected:
    selected.append(epoch)
if target not in selected:
  selected.append(target)
if len(selected) < 4:
  extras = [epoch for epoch in available if epoch not in selected]
  while len(selected) < 4 and extras:
    selected.append(extras.pop(len(extras) // 2))
selected = sorted(selected)
if len(selected) < 4:
  raise RuntimeError(f"Need four matched evaluation windows, got {selected}")
payload = {
  "schema_version": 1,
  "parent_epoch": parent,
  "target_epoch": target,
  "available_checkpoint_epochs": available,
  "evaluation_epochs": selected,
  "causal_hybrid_epochs": selected[-3:],
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("\n".join(str(value) for value in selected))
PY
)
mapfile -t HYBRID_EPOCHS < <(jq -r '.causal_hybrid_epochs[]' "$EVAL_ROOT/evaluation_manifest.json")

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
      --candidate-label "$label" --opponent-label stage2_parent \
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
  train-ablation-1781126582/run_full_episode_credit_eval_v1.sh \
  train-ablation-1781126582/uniform_context_eval.py \
  train-ablation-1781126582/dump_context_decks.py \
  train-ablation-1781126582/dump_context_decks_native.py \
  train-ablation-1781126582/probe_gate_kl.py \
  train-ablation-1781126582/probe_context_kl.py \
  train-ablation-1781126582/probe_context_kl_native.py \
  train-ablation-1781126582/build_context_hybrid_arms.py \
  train-ablation-1781126582/analyze_context_hybrids.py \
  train-ablation-1781126582/run_fixed_deck_counterfactual.py \
  train-ablation-1781126582/full_episode_credit_ladder_report.py \
  python/src/native_reference_eval.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py build/python/src/binding*.so \
  > "$EVAL_ROOT/runtime_sha256.txt"

# Finish all GPU panels before starting CPU fixed-deck workers.
for epoch in "${EVAL_EPOCHS[@]}"; do
  epoch_dir="$EVAL_ROOT/p$epoch"
  mkdir -p "$epoch_dir/control" "$epoch_dir/$CANDIDATE_ARM"
  control_checkpoint=$(checkpoint_for control "$epoch")
  candidate_checkpoint=$(checkpoint_for "$CANDIDATE_ARM" "$epoch")
  [[ -f "$control_checkpoint" && -f "$candidate_checkpoint" ]]

  run_uniform_panel "$candidate_checkpoint" "$control_checkpoint" \
    "${CANDIDATE_ARM}_p$epoch" "control_p$epoch" "$epoch_dir/credit_vs_control.json"
  run_uniform_panel "$candidate_checkpoint" "$PARENT_MODEL" \
    "${CANDIDATE_ARM}_p$epoch" stage2_parent "$epoch_dir/credit_vs_parent.json"
  run_uniform_panel "$control_checkpoint" "$PARENT_MODEL" \
    "control_p$epoch" stage2_parent "$epoch_dir/control_vs_parent.json"
  run_reference_panel "$candidate_checkpoint" "${CANDIDATE_ARM}_p$epoch" \
    "$epoch_dir/$CANDIDATE_ARM/heldout.json"
  run_reference_panel "$control_checkpoint" "control_p$epoch" \
    "$epoch_dir/control/heldout.json"
  run_context_decks "$candidate_checkpoint" "$epoch_dir/$CANDIDATE_ARM"
  run_context_decks "$control_checkpoint" "$epoch_dir/control"
  run_context_kl "$candidate_checkpoint" "$epoch_dir/$CANDIDATE_ARM"
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
  for arm in control "$CANDIDATE_ARM"; do
    checkpoint=$(checkpoint_for "$arm" "$epoch")
    run_context_hybrids "$checkpoint" "${arm}_p${epoch}" "$EVAL_ROOT/p$epoch/$arm"
  done
done

PYTHONPATH=python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/full_episode_credit_ladder_report.py \
  --root "$RESULT_ROOT" --json "$RESULT_ROOT/ladder_report.json" \
  --md "$RESULT_ROOT/ladder_report.md" --candidate-arm "$CANDIDATE_ARM"

touch "$RESULT_ROOT/EVALUATION_DONE"
echo "[$CAMPAIGN-eval] complete"
