#!/usr/bin/env bash
# Four-window evaluation for the fresh matched 45M Stage 1 confirmation.
set -euo pipefail

export UNIFORM_ASSIGNMENT_CAMPAIGN=${UNIFORM_ASSIGNMENT_CONFIRM45_CAMPAIGN:-uniform_assignment_confirm45_v1}
export UNIFORM_ASSIGNMENT_EVAL_EPOCHS="5000 5800 6800 7800"
export UNIFORM_ASSIGNMENT_HYBRID_EPOCHS="5800 6800 7800"

exec ./train-ablation-1781126582/run_uniform_assignment_eval_v1.sh
