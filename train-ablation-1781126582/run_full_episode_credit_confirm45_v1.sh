#!/usr/bin/env bash
# Fresh matched 45M confirmation, gated by the Stage 2 15M decision.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

SOURCE_CAMPAIGN=${FULL_CREDIT_SOURCE_CAMPAIGN:-full_episode_credit_ladder15_v1}
SOURCE_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$SOURCE_CAMPAIGN"
CAMPAIGN=${FULL_CREDIT_CONFIRM_CAMPAIGN:-full_episode_credit_confirm45_v1}

[[ -f "$SOURCE_ROOT/EVALUATION_DONE" ]] || {
  echo "[$CAMPAIGN] Stage 2 15M evaluation is incomplete: $SOURCE_ROOT" >&2
  exit 1
}
jq -e '
  .decision.verdict == "advance_to_45m_confirmation" and
  .decision.advance == true and
  .decision.integrity_pass == true and
  .decision.throughput_pass == true and
  .decision.external_safety_pass == true and
  .decision.mechanics_pass == true
' "$SOURCE_ROOT/ladder_report.json" >/dev/null

RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage2/$CAMPAIGN"
mkdir -p "$RESULT_ROOT"
sha256sum \
  "$SOURCE_ROOT/ladder_report.json" \
  train-ablation-1781126582/run_full_episode_credit_confirm45_v1.sh \
  > "$RESULT_ROOT/confirmation_gate_sha256.txt"

export FULL_CREDIT_LADDER_CAMPAIGN="$CAMPAIGN"
export FULL_CREDIT_LADDER_UPDATES=2930
exec ./train-ablation-1781126582/run_full_episode_credit_ladder15_v1.sh
