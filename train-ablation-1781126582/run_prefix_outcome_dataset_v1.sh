#!/usr/bin/env bash
# Build and freeze the Stage 2 prefix-outcome model from complete native games.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${PREFIX_OUTCOME_DATASET_CAMPAIGN:-prefix_outcome_model_v1}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$CAMPAIGN"
SNAPSHOT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage1/uniform_assignment_confirm45_v1/state_snapshots"
PARENT_MODEL="experiments/azuki_local_strategy_recovery_leader15_v1_control_178447770084/model_azuki_local_004870.pt"
GAMES_PER_CONTEXT=${PREFIX_OUTCOME_GAMES_PER_CONTEXT:-12}
BATCH_ENVS=${PREFIX_OUTCOME_BATCH_ENVS:-48}
MAX_STEPS=${PREFIX_OUTCOME_MAX_STEPS:-1200}
EPOCHS=${PREFIX_OUTCOME_EPOCHS:-60}
RANDOM_PREFIX_LENGTHS=${PREFIX_OUTCOME_RANDOM_PREFIX_LENGTHS:-0}
RANDOM_PREFIX_PROBABILITIES=${PREFIX_OUTCOME_RANDOM_PREFIX_PROBABILITIES:-1}
RANDOM_PREFIX_SEED=${PREFIX_OUTCOME_RANDOM_PREFIX_SEED:-420053}
ARTIFACT_BASENAME=${PREFIX_OUTCOME_ARTIFACT_BASENAME:-frozen_prefix_outcome_v1.npz}
ARTIFACT="$RESULT_ROOT/$ARTIFACT_BASENAME"

mkdir -p "$RESULT_ROOT/trajectories"

if (( GAMES_PER_CONTEXT < 2 || GAMES_PER_CONTEXT % 2 != 0 )); then
  echo "[$CAMPAIGN] games per context must be a positive even integer" >&2
  exit 1
fi
if [[ ! -f "$PARENT_MODEL" ]]; then
  echo "[$CAMPAIGN] missing accepted historical parent: $PARENT_MODEL" >&2
  exit 1
fi

collect_pair() {
  local generation=$1
  local lineage=$2
  local checkpoint=$3
  local opponent=$4
  local base_seed=$5
  local stem="${generation}_${lineage}"
  local output="$RESULT_ROOT/trajectories/${stem}.jsonl"
  local summary="$RESULT_ROOT/trajectories/${stem}.summary.json"
  if [[ -f "$output" && -f "$summary" ]]; then
    echo "[$CAMPAIGN] reuse $stem"
    return
  fi
  echo "[$CAMPAIGN] collect $stem"
  PYTHONPATH=build/python/src:python/src \
  OMP_NUM_THREADS=12 \
  .venv/bin/python train-ablation-1781126582/collect_prefix_outcomes.py \
    --checkpoint "$checkpoint" \
    --opponent-checkpoint "$opponent" \
    --policy-generation "$generation" \
    --opponent-lineage "$lineage" \
    --games-per-context "$GAMES_PER_CONTEXT" \
    --base-seed "$base_seed" \
    --batch-envs "$BATCH_ENVS" \
    --max-steps "$MAX_STEPS" \
    --device cuda \
    --random-prefix-lengths "$RANDOM_PREFIX_LENGTHS" \
    --random-prefix-probabilities "$RANDOM_PREFIX_PROBABILITIES" \
    --random-prefix-seed "$RANDOM_PREFIX_SEED" \
    --jsonl "$output" \
    --summary-json "$summary"
}

for epoch in 5200 5800 6800 7800; do
  generation="p${epoch}"
  candidate="$SNAPSHOT_ROOT/uniform_assignment/p$(printf '%06d' "$epoch")/model_azuki_local_$(printf '%06d' "$epoch").pt"
  control="$SNAPSHOT_ROOT/control/p$(printf '%06d' "$epoch")/model_azuki_local_$(printf '%06d' "$epoch").pt"
  if [[ ! -f "$candidate" || ! -f "$control" ]]; then
    echo "[$CAMPAIGN] missing Stage 1 snapshot for $generation" >&2
    exit 1
  fi
  collect_pair "$generation" control "$candidate" "$control" 41000019
  collect_pair "$generation" accepted_parent "$candidate" "$PARENT_MODEL" 61000021
done

mapfile -t inputs < <(find "$RESULT_ROOT/trajectories" -maxdepth 1 -type f -name '*.jsonl' | sort)
if (( ${#inputs[@]} != 8 )); then
  echo "[$CAMPAIGN] expected eight trajectory shards, found ${#inputs[@]}" >&2
  exit 1
fi
input_args=()
for input in "${inputs[@]}"; do
  input_args+=(--input "$input")
done

PYTHONPATH=build/python/src:python/src \
OMP_NUM_THREADS=12 \
.venv/bin/python train-ablation-1781126582/train_prefix_outcome_model.py \
  "${input_args[@]}" \
  --heldout-generation p7800 \
  --heldout-lineage accepted_parent \
  --artifact "$ARTIFACT" \
  --report-json "$RESULT_ROOT/predictor_report.json" \
  --report-md "$RESULT_ROOT/predictor_report.md" \
  --device cuda \
  --epochs "$EPOCHS" \
  --batch-size 8192

sha256sum \
  "$ARTIFACT" \
  "$RESULT_ROOT/predictor_report.json" \
  > "$RESULT_ROOT/artifact_sha256.txt"
touch "$RESULT_ROOT/DATASET_DONE"
echo "[$CAMPAIGN] complete"
