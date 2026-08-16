#!/usr/bin/env bash
# Evaluate the fresh Stage 2 45M confirmation with final acceptance semantics.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${FULL_CREDIT_CONFIRM_CAMPAIGN:-full_episode_credit_confirm45_v1}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$CAMPAIGN"

[[ -f "$RESULT_ROOT/LADDER_TRAIN_DONE" ]] || {
  echo "[$CAMPAIGN-eval] matched confirmation training is incomplete" >&2
  exit 1
}

export FULL_CREDIT_LADDER_CAMPAIGN="$CAMPAIGN"
./train-ablation-1781126582/run_full_episode_credit_eval_v1.sh

# Re-synthesize the same immutable evidence using final confirmation semantics.
PYTHONPATH=python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/full_episode_credit_ladder_report.py \
  --root "$RESULT_ROOT" --json "$RESULT_ROOT/confirmation_report.json" \
  --md "$RESULT_ROOT/confirmation_report.md" --confirmation

jq -e '.confirmation == true' "$RESULT_ROOT/confirmation_report.json" >/dev/null
touch "$RESULT_ROOT/CONFIRMATION_EVALUATION_DONE"
