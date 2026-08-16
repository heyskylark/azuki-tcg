#!/usr/bin/env bash
# Three-arm 45M confirmation for standard credit, exact credit, and exact credit plus prefix.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${CREDIT_PREFIX_CONFIRM_CAMPAIGN:-credit_prefix_confirm45_v1}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage3/$CAMPAIGN"
CREDIT_SOURCE="train-ablation-1781126582/results/next_ablation_v1/stage2/full_episode_credit_ladder15_labelfix_v1"
PREFIX_SOURCE="train-ablation-1781126582/results/next_ablation_v1/stage3/random_main_prefix_ladder15_v1"
BEHAVIOR_REPORT="train-ablation-1781126582/results/next_ablation_v1/credit_prefix_15m_behavior_analysis.md"

for required in \
  "$CREDIT_SOURCE/EVALUATION_DONE" \
  "$CREDIT_SOURCE/ladder_report.json" \
  "$PREFIX_SOURCE/EVALUATION_DONE" \
  "$PREFIX_SOURCE/ladder_report.json" \
  "$BEHAVIOR_REPORT"; do
  if [[ ! -f "$required" ]]; then
    echo "[$CAMPAIGN] missing 15M qualification artifact: $required" >&2
    exit 1
  fi
done

jq -e '
  .decision.integrity_pass == true and
  .decision.external_safety_pass == true and
  .decision.mechanics_pass == true
' "$CREDIT_SOURCE/ladder_report.json" >/dev/null
jq -e '
  .decision.integrity_pass == true and
  .decision.external_safety_pass == true and
  .throughput.relative_pass == true and
  .throughput.absolute_pass == true
' "$PREFIX_SOURCE/ladder_report.json" >/dev/null

if [[ "${CREDIT_PREFIX_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "[$CAMPAIGN] preflight passed"
  exit 0
fi

mkdir -p "$RESULT_ROOT"
sha256sum \
  "$CREDIT_SOURCE/ladder_report.json" \
  "$PREFIX_SOURCE/ladder_report.json" \
  "$BEHAVIOR_REPORT" \
  train-ablation-1781126582/run_credit_prefix_confirm45_v1.sh \
  train-ablation-1781126582/run_random_main_prefix_ladder15_v1.sh \
  > "$RESULT_ROOT/confirmation_gate_sha256.txt"
cat > "$RESULT_ROOT/confirmation_override.txt" <<'EOF'
The registered 15M adoption gates remain unchanged. This 45M confirmation is
authorized from the combined behavioral analysis because both treatments caused
material within-run policy changes that had not converged at p8770.
EOF

export RANDOM_PREFIX_LADDER_CAMPAIGN="$CAMPAIGN"
export RANDOM_PREFIX_LADDER_UPDATES=2930
export RANDOM_PREFIX_INCLUDE_STANDARD_CONTROL=1
exec ./train-ablation-1781126582/run_random_main_prefix_ladder15_v1.sh
