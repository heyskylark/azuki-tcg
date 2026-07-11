#!/bin/bash
# S2 smoke: portalgp base with OUTCOME-graded portal bonus (replaces flat GP).
# Waits for the running s1auxann45 trainer before taking the GPU.
set -u
cd /home/skylark/git/azuki-tcg
echo "[s2] $(date +%F_%T) waiting for s1auxann45 trainer"
while pgrep -f "python python/src/train[.]py .*--tag s1auxann45" > /dev/null; do sleep 120; done
echo "[s2] $(date +%F_%T) GPU free; launching s2outcome"
sed 's/run_arm s1auxann 0.05 2.0/run_arm s2outcome 0 0/; s/export AZK_DRAFT_AUX_ANNEAL=1//; s/AZK_PORTAL_GP_BONUS=0.3/AZK_PORTAL_OUTCOME_BONUS=0.3/' \
  train-ablation-1781126582/run_s1.sh > /tmp/run_s2_inner.sh
bash /tmp/run_s2_inner.sh
