#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

OUT=train-ablation-1781126582/results/promotionv2_p2930_analysis/frozen_deck_counterfactual/fire_hybrid
CHECKPOINT=experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308/model_azuki_local_002930.pt
ARMS="$OUT/deck_arms.json"
SHARDS=12
EXPECTED_PER_SHARD=12
mkdir -p "$OUT"

PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/build_fire_hybrid_arms.py \
  --json "$ARMS"

run_shard() {
  local shard=$1
  local output="$OUT/games_shard${shard}.jsonl"
  local temporary="${output}.tmp"
  local log="$OUT/shard${shard}.log"
  if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $EXPECTED_PER_SHARD ]]; then
    echo "[fire-hybrid] keep complete $output"
    return
  fi
  rm -f "$temporary"
  echo "[fire-hybrid] start shard=$shard"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python \
    train-ablation-1781126582/run_fixed_deck_counterfactual.py \
    --checkpoint "$CHECKPOINT" \
    --deck-arms "$ARMS" \
    --device cpu \
    --shards "$SHARDS" \
    --shard-index "$shard" \
    --out "$temporary" > "$log" 2>&1
  if [[ $(wc -l < "$temporary") -ne $EXPECTED_PER_SHARD ]]; then
    echo "[fire-hybrid] shard=$shard wrote an incomplete result" >&2
    return 1
  fi
  mv "$temporary" "$output"
  echo "[fire-hybrid] done shard=$shard"
}

pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
  run_shard "$shard" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  exit "$status"
fi

PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_frozen_deck_counterfactual.py \
  "$OUT"/games_shard*.jsonl \
  --deck-arms "$ARMS" \
  --expected-games-per-arm 36 \
  --skip-identical-control-validation \
  --json "$OUT/summary.json"

echo "[fire-hybrid] complete: $OUT"
