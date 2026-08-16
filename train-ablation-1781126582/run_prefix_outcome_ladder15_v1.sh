#!/usr/bin/env bash
# Stage 2 fallback: matched 15M efficacy ladder for frozen prefix redistribution.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

MODEL_ROOT=${PREFIX_OUTCOME_MODEL_ROOT:-train-ablation-1781126582/results/next_ablation_v1/stage2/prefix_outcome_model_v4}
MODEL_FILENAME=${PREFIX_OUTCOME_MODEL_FILENAME:-frozen_prefix_outcome_v4.npz}
MODEL="$MODEL_ROOT/$MODEL_FILENAME"
REPORT="$MODEL_ROOT/predictor_report.json"
SHA=$(jq -er '.artifact.sha256' "$REPORT")

export FULL_CREDIT_LADDER_CAMPAIGN=${PREFIX_OUTCOME_LADDER_CAMPAIGN:-prefix_outcome_ladder15_v1}
export FULL_CREDIT_SMOKE_ROOT=${PREFIX_OUTCOME_SMOKE_ROOT:-train-ablation-1781126582/results/next_ablation_v1/stage2/prefix_outcome_smoke_v1}
export FULL_CREDIT_MODE=prefix_outcome
export FULL_CREDIT_CANDIDATE_ARM=prefix_outcome
export FULL_CREDIT_PREFIX_MODEL="$MODEL"
export FULL_CREDIT_PREFIX_REPORT="$REPORT"
export FULL_CREDIT_PREFIX_SHA256="$SHA"
export FULL_CREDIT_COEF=${PREFIX_OUTCOME_COEFFICIENT:-1.0}

exec train-ablation-1781126582/run_full_episode_credit_ladder15_v1.sh
