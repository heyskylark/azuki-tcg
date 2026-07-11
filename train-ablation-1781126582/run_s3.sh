#!/bin/bash
# S3 smoke: portalgp base + cross-gate replay 0.15 + pick masking.
# Waits for the s2outcome trainer (which itself waits for s1auxann45).
set -u
cd /home/skylark/git/azuki-tcg
echo "[s3] $(date +%F_%T) waiting for s2outcome trainer to exist and finish"
until pgrep -f "python python/src/train[.]py .*--tag s2outcome" > /dev/null; do sleep 120; done
while pgrep -f "python python/src/train[.]py .*--tag s2outcome" > /dev/null; do sleep 120; done
echo "[s3] $(date +%F_%T) GPU free; launching s3xgate"
sed 's/run_arm s1auxann 0.05 2.0/export AZK_XGATE_MASK=1\nrun_arm s3xgate 0 0/; s/export AZK_DRAFT_AUX_ANNEAL=1//; s/--env.draft_same_element_matchup_prob 0.35 \\/--env.draft_same_element_matchup_prob 0.35 \\\n    --env.draft_cross_gate_replay_prob 0.15 \\/' \
  train-ablation-1781126582/run_s1.sh > /tmp/run_s3_inner.sh
bash /tmp/run_s3_inner.sh
