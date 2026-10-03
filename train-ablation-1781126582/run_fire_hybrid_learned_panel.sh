#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

BASE=train-ablation-1781126582/results/promotionv2_p2930_analysis/frozen_deck_counterfactual
OUT="$BASE/fire_hybrid/learned_panel"
CHECKPOINT=experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308/model_azuki_local_002930.pt
CANDIDATE_ARMS="$BASE/fire_hybrid/deck_arms.json"
OPPONENT_ARMS="$BASE/deck_arms.json"
SHARDS=8
EXPECTED_PER_SHARD=16
mkdir -p "$OUT"

run_shard() {
  local shard=$1
  local output="$OUT/games_shard${shard}.jsonl"
  local temporary="${output}.tmp"
  local log="$OUT/shard${shard}.log"
  if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $EXPECTED_PER_SHARD ]]; then
    echo "[fire-learned] keep complete $output"
    return
  fi
  rm -f "$temporary"
  echo "[fire-learned] start shard=$shard"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python \
    train-ablation-1781126582/run_fixed_deck_counterfactual.py \
    --checkpoint "$CHECKPOINT" \
    --deck-arms "$CANDIDATE_ARMS" \
    --opponent-deck-arms "$OPPONENT_ARMS" \
    --opponent-arm native_p2930 \
    --device cpu \
    --shards "$SHARDS" \
    --shard-index "$shard" \
    --out "$temporary" > "$log" 2>&1
  if [[ $(wc -l < "$temporary") -ne $EXPECTED_PER_SHARD ]]; then
    echo "[fire-learned] shard=$shard wrote an incomplete result" >&2
    return 1
  fi
  mv "$temporary" "$output"
  echo "[fire-learned] done shard=$shard"
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
  --deck-arms "$CANDIDATE_ARMS" \
  --expected-games-per-arm 32 \
  --skip-identical-control-validation \
  --json "$OUT/summary.json"

echo "[fire-learned] complete: $OUT"
