#!/bin/bash
# Full achievable-gap matrix, sequenced to avoid CPU oversubscription:
#   1. wait for any running LIGHTNING shards
#   2. sibling pairs: WATER, FIRE, EARTH (run_gate_gap.sh each)
#   3. cross-element gate-ability ladder: each non-WATER-ref gate vs STT02-002
#      (Hydromancy) on all-NORMAL mirror decks, same leader both sides
# Detach with: setsid nohup bash train-ablation-1781126582/run_gate_gap_all.sh \
#   > /tmp/gate_gap_all.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results/gate_gap
mkdir -p "$RESULTS"
CKPT=$(ls experiments/azuki_local_combo45b_*/model_azuki_local_002930.pt | head -1)
REF=STT02-002

echo "[gapall] $(date +%F_%T) waiting for running gap shards"
while pgrep -f "probe_gate_gap.py" > /dev/null; do sleep 60; done

for EL in WATER FIRE EARTH; do
  echo "[gapall] $(date +%F_%T) sibling pair $EL"
  bash train-ablation-1781126582/run_gate_gap.sh "$EL" 6 43 >> /tmp/gate_gap_all_elements.log 2>&1
done

# cross ladder: modest n (4 decks x 2 orders x 40 seeds x 2 shards = 640/mode)
for GATE in STT01-002 AZK01-120 AZK01-126 AZK01-122 STT04-002 AZK01-124 STT03-002; do
  for MODE in policy forced blocked; do
    pids=()
    for SHARD in 0 1; do
      OUT="$RESULTS/cross_${GATE}_${MODE}_s${SHARD}.json"
      [ -f "$OUT" ] && continue
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
      .venv/bin/python train-ablation-1781126582/probe_gate_gap.py \
        --checkpoint "$CKPT" --pair "${GATE},${REF}" --neutral-decks 4 \
        --mode "$MODE" --seeds 40 --seed-offset $((SHARD * 40)) \
        --json "$OUT" > "/tmp/gate_gap_cross_${GATE}_${MODE}_s${SHARD}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
  done
  echo "[gapall] $(date +%F_%T) cross ladder done: $GATE vs $REF"
done
echo "[gapall] $(date +%F_%T) ALL DONE"
