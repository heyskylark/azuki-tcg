#!/usr/bin/env bash
# Stage 4 short schedule, terminal-credit, and throughput integrity gate.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

export LATE_ZERO_LADDER_CAMPAIGN=${LATE_ZERO_SMOKE_CAMPAIGN:-late_zero_smoke_v1}
export LATE_ZERO_LADDER_UPDATES=${LATE_ZERO_SMOKE_UPDATES:-48}
export LATE_ZERO_LADDER_PEAK_LR=${LATE_ZERO_SMOKE_PEAK_LR:-0.000003}
export LATE_ZERO_MODE=smoke
export LATE_ZERO_REQUIRE_SMOKE=0
export LATE_ZERO_CHECKPOINT_INTERVAL=${LATE_ZERO_SMOKE_UPDATES:-48}

exec ./train-ablation-1781126582/run_late_zero_ladder15_v1.sh
