#!/usr/bin/env bash
# Fresh matched 45M adaptation test from the accepted p4870 atomic parent.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

export UNIFORM_ASSIGNMENT_CAMPAIGN=${UNIFORM_ASSIGNMENT_CONFIRM45_CAMPAIGN:-uniform_assignment_confirm45_v1}
export UNIFORM_ASSIGNMENT_UPDATES=2930

exec ./train-ablation-1781126582/run_uniform_assignment_ladder15_v1.sh
