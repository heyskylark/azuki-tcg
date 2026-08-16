#!/usr/bin/env bash
# Fit the frozen predictor on paired policy-prefix and randomized-prefix games.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

STAGE2=train-ablation-1781126582/results/next_ablation_v1/stage2
ARGMAX_ROOT="$STAGE2/prefix_outcome_model_v1"
RANDOM_ROOT="$STAGE2/prefix_outcome_model_v2"
CAMPAIGN=${PREFIX_OUTCOME_COMBINED_CAMPAIGN:-prefix_outcome_model_v3}
RESULT_ROOT="$STAGE2/$CAMPAIGN"
EPOCHS=${PREFIX_OUTCOME_EPOCHS:-60}
PAIRWISE_COEF=${PREFIX_OUTCOME_PAIRWISE_COEF:-0}
ARTIFACT_BASENAME=${PREFIX_OUTCOME_ARTIFACT_BASENAME:-frozen_prefix_outcome_v3.npz}

for source in "$ARGMAX_ROOT" "$RANDOM_ROOT"; do
  if [[ ! -f "$source/DATASET_DONE" ]]; then
    echo "[$CAMPAIGN] incomplete source dataset: $source" >&2
    exit 1
  fi
done
mkdir -p "$RESULT_ROOT"

PYTHONPATH=build/python/src:python/src \
.venv/bin/python train-ablation-1781126582/merge_prefix_outcome_datasets.py \
  --dataset "policy_argmax=$ARGMAX_ROOT/trajectories" \
  --dataset "random_prefix=$RANDOM_ROOT/trajectories" \
  --output "$RESULT_ROOT/combined_trajectories.jsonl" \
  --summary-json "$RESULT_ROOT/merge_report.json"

PYTHONPATH=build/python/src:python/src \
OMP_NUM_THREADS=12 \
.venv/bin/python train-ablation-1781126582/train_prefix_outcome_model.py \
  --input "$RESULT_ROOT/combined_trajectories.jsonl" \
  --heldout-generation p7800 \
  --heldout-lineage accepted_parent \
  --artifact "$RESULT_ROOT/$ARTIFACT_BASENAME" \
  --report-json "$RESULT_ROOT/predictor_report.json" \
  --report-md "$RESULT_ROOT/predictor_report.md" \
  --device cuda \
  --epochs "$EPOCHS" \
  --batch-size 8192 \
  --pairwise-coef "$PAIRWISE_COEF"

sha256sum \
  "$RESULT_ROOT/combined_trajectories.jsonl" \
  "$RESULT_ROOT/$ARTIFACT_BASENAME" \
  "$RESULT_ROOT/predictor_report.json" \
  > "$RESULT_ROOT/artifact_sha256.txt"
touch "$RESULT_ROOT/DATASET_DONE"
echo "[$CAMPAIGN] complete"
