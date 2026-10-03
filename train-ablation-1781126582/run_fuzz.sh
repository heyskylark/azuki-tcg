#!/bin/bash
# Parallel mask-consistency fuzz: 8 workers x 250k vec-steps x 16 envs
# (~32M agent-steps of random play). Desync hits print
# "Invalid-action truncation:" with repro seeds into the worker logs.
set -u
cd /home/skylark/git/azuki-tcg
pids=()
for S in 1 2 3 4 5 6 7 8; do
  OMP_NUM_THREADS=1 PYTHONPATH=build/python/src:python/src \
  .venv/bin/python train-ablation-1781126582/fuzz_mask_consistency.py \
    --steps 250000 --envs 16 --seed "$S" \
    > "/tmp/fuzz_mask_s${S}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "FUZZ DONE rc=$rc"
grep -l "Invalid-action truncation" /tmp/fuzz_mask_s*.log 2>/dev/null || echo "NO DESYNC HITS"
grep -h "Invalid-action truncation" /tmp/fuzz_mask_s*.log 2>/dev/null | head -20
