#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

OUT=train-ablation-1781126582/results/promotionv2_p2930_analysis/opportunity_lineage
GAMES_PER_CHECKPOINT=64
SHARDS=4
PARALLEL_JOBS=12
SEED0=710001
mkdir -p "$OUT"

LABELS=(
  s14_p2000
  task3_parent_p2930
  promotion_p0100
  promotion_p0300
  promotion_p0977
  promotion_p1500
  promotion_p2930
)
CHECKPOINTS=(
  experiments/azuki_local_s14prod45_178404911119/model_azuki_local_002000.pt
  experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt
  experiments/azuki_local_promotionv2_shadow15_final_178431780213/model_azuki_local_000100.pt
  experiments/azuki_local_promotionv2_shadow15_final_178431780213/model_azuki_local_000300.pt
  experiments/azuki_local_promotionv2_shadow15_final_resume300_contiguous_178432002673/model_azuki_local_000977.pt
  experiments/azuki_local_promotionv2_archive45_final_resume977_178433179898/model_azuki_local_001500.pt
  experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308/model_azuki_local_002930.pt
)

if (( GAMES_PER_CHECKPOINT % SHARDS != 0 )); then
  echo "GAMES_PER_CHECKPOINT must be divisible by SHARDS" >&2
  exit 1
fi
GAMES_PER_SHARD=$((GAMES_PER_CHECKPOINT / SHARDS))

run_shard() {
  local label=$1
  local checkpoint=$2
  local shard=$3
  local shard_seed=$((SEED0 + 7919 * GAMES_PER_SHARD * shard))
  local output="$OUT/${label}_shard${shard}.jsonl"
  local temporary="${output}.tmp"
  local log="$OUT/${label}_shard${shard}.log"

  if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $GAMES_PER_SHARD ]]; then
    echo "[opportunity] keep complete $output"
    return
  fi
  rm -f "$temporary"
  echo "[opportunity] start $label shard=$shard seed=$shard_seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  nice -n 3 .venv/bin/python train-ablation-1781126582/play_selfplay_games.py \
    --checkpoint "$checkpoint" \
    --games "$GAMES_PER_SHARD" \
    --seed0 "$shard_seed" \
    --device cpu \
    --log-legal-actions \
    --out "$temporary" > "$log" 2>&1
  mv "$temporary" "$output"
  echo "[opportunity] done $label shard=$shard"
}

running=0
for index in "${!LABELS[@]}"; do
  for ((shard = 0; shard < SHARDS; shard++)); do
    run_shard "${LABELS[$index]}" "${CHECKPOINTS[$index]}" "$shard" &
    running=$((running + 1))
    if (( running >= PARALLEL_JOBS )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

for label in "${LABELS[@]}"; do
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_opportunity_rates.py \
    "$OUT/${label}_shard"*.jsonl \
    --label "$label" \
    --json "$OUT/${label}.json"
done

echo "[opportunity] complete: $OUT"
