#!/bin/bash
# Critic gate-sensitivity probe across round-2 arms (go/no-go for 45M runs).
# Runs all arms concurrently on CPU with capped threads.
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
EPISODES=${1:-12}

pids=()
for TAG in ctrl2 anneal1 gateid1 combo1; do
  CKPT=$(ls experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then echo "[$TAG] no checkpoint"; continue; fi
  echo "[$TAG] probing $CKPT"
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/probe_critic_gate.py \
    --checkpoint "$CKPT" --episodes "$EPISODES" --device cpu \
    --json "$RESULTS/critic_probe_${TAG}.json" \
    > "/tmp/critic_probe_${TAG}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL PROBES DONE rc=$rc"
for TAG in ctrl2 anneal1 gateid1 combo1; do
  echo "=== $TAG ==="
  grep -E "^\[|win-prob|tertiles" "/tmp/critic_probe_${TAG}.log" | grep -v resume || true
done
