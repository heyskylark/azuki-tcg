#!/usr/bin/env bash
# Qualified set fixed by the 15M external/action/stability rule. Task 4 is
# intentionally absent after its 39/96 endpoint and 118/288 trajectory.
set -uo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT" || exit 1

DRIVER=train-ablation-1781126582/run_competitive_reward_45m.sh
RESULTS=train-ablation-1781126582/results/reward_45m
ARMS=(rs1entity45 rs2ikzconv45 rs3tempreal45)

mkdir -p "$RESULTS"
for arm in "${ARMS[@]}"; do
  bash "$DRIVER" "$arm" || exit $?
done

touch "$RESULTS/CONFIRMATIONS_DONE"
echo "[reward-45m] $(date +%F_%T) qualified confirmation chain complete"
