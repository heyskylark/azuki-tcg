#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

BASE=train-ablation-1781126582/results/strategy_recovery_ladder15_v1/tempo_dedup
OUT="$BASE/fire_leader_counterfactual"
CHECKPOINT=experiments/azuki_local_strategy_recovery_ladder15_v1_tempo_dedup_resume3300_178445644982/model_azuki_local_003900.pt
DECK_DUMP="$BASE/decks.json"
CANDIDATE_ARMS="$OUT/deck_arms.json"
OPPONENT_ARMS="$OUT/all_gate_deck_arms.json"

mkdir -p "$OUT/fixed_panel" "$OUT/learned_panel"

PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/build_fire_hybrid_arms.py \
  --p2930 "$DECK_DUMP" \
  --json "$CANDIDATE_ARMS"

PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/build_frozen_deck_arms.py \
  --p2930 "$DECK_DUMP" \
  --json "$OPPONENT_ARMS"

run_fixed_shard() {
  local shard=$1
  local output="$OUT/fixed_panel/games_shard${shard}.jsonl"
  local temporary="${output}.tmp"
  local log="$OUT/fixed_panel/shard${shard}.log"
  if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq 12 ]]; then
    echo "[fire-p3900-fixed] keep complete shard=$shard"
    return
  fi
  rm -f "$temporary"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python \
    train-ablation-1781126582/run_fixed_deck_counterfactual.py \
    --checkpoint "$CHECKPOINT" \
    --deck-arms "$CANDIDATE_ARMS" \
    --device cpu \
    --shards 12 \
    --shard-index "$shard" \
    --out "$temporary" > "$log" 2>&1
  [[ $(wc -l < "$temporary") -eq 12 ]]
  mv "$temporary" "$output"
  echo "[fire-p3900-fixed] done shard=$shard"
}

fixed_pids=()
for shard in {0..11}; do
  run_fixed_shard "$shard" &
  fixed_pids+=("$!")
done
for pid in "${fixed_pids[@]}"; do
  wait "$pid"
done

PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_frozen_deck_counterfactual.py \
  "$OUT"/fixed_panel/games_shard*.jsonl \
  --deck-arms "$CANDIDATE_ARMS" \
  --expected-games-per-arm 36 \
  --skip-identical-control-validation \
  --json "$OUT/fixed_panel/summary.json"

run_learned_shard() {
  local shard=$1
  local output="$OUT/learned_panel/games_shard${shard}.jsonl"
  local temporary="${output}.tmp"
  local log="$OUT/learned_panel/shard${shard}.log"
  if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq 16 ]]; then
    echo "[fire-p3900-learned] keep complete shard=$shard"
    return
  fi
  rm -f "$temporary"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python \
    train-ablation-1781126582/run_fixed_deck_counterfactual.py \
    --checkpoint "$CHECKPOINT" \
    --deck-arms "$CANDIDATE_ARMS" \
    --opponent-deck-arms "$OPPONENT_ARMS" \
    --opponent-arm native_p2930 \
    --device cpu \
    --shards 8 \
    --shard-index "$shard" \
    --out "$temporary" > "$log" 2>&1
  [[ $(wc -l < "$temporary") -eq 16 ]]
  mv "$temporary" "$output"
  echo "[fire-p3900-learned] done shard=$shard"
}

learned_pids=()
for shard in {0..7}; do
  run_learned_shard "$shard" &
  learned_pids+=("$!")
done
for pid in "${learned_pids[@]}"; do
  wait "$pid"
done

PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_frozen_deck_counterfactual.py \
  "$OUT"/learned_panel/games_shard*.jsonl \
  --deck-arms "$CANDIDATE_ARMS" \
  --expected-games-per-arm 32 \
  --skip-identical-control-validation \
  --json "$OUT/learned_panel/summary.json"

sha256sum "$CHECKPOINT" "$DECK_DUMP" "$CANDIDATE_ARMS" "$OPPONENT_ARMS" \
  > "$OUT/input_sha256.txt"
touch "$OUT/DONE"
echo "[fire-p3900] complete: $OUT"
