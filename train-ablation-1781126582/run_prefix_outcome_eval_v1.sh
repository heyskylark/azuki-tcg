#!/usr/bin/env bash
# Stage 2 fallback evaluation wrapper for the frozen prefix-outcome ladder.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

export FULL_CREDIT_LADDER_CAMPAIGN=${PREFIX_OUTCOME_LADDER_CAMPAIGN:-prefix_outcome_ladder15_v1}
export FULL_CREDIT_CANDIDATE_ARM=prefix_outcome

exec train-ablation-1781126582/run_full_episode_credit_eval_v1.sh
