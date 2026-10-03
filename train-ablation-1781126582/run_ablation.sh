#!/bin/bash
# Ablation runner: trains with overrides, snapshots decks, evals, and indexes results.
# Usage: run_ablation.sh NAME TOTAL_STEPS [extra train.py args...]
# Code-level ablations should be run from their own git branch; this script
# records the active branch+commit alongside results.
set -u
NAME="$1"; shift
TOTAL="$1"; shift
ROOT=/home/skylark/git/azuki-tcg
OUT="$ROOT/train-ablation-1781126582/results"
SNAP="$ROOT/experiments/abl_snapshots/$NAME"
mkdir -p "$OUT" "$SNAP"
BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD)
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cd "$ROOT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
AZK_DECKBUILD_SNAPSHOT_DIR="$SNAP" \
AZK_DECKBUILD_SNAPSHOT_EVERY=25 \
PYTHONPATH=build/python/src:python/src \
.venv/bin/python python/src/train.py \
  --config python/config/azuki_deckbuild_3090.ini \
  --jsonl-log experiments/runlogs \
  --tag "$NAME" \
  --train.total_timesteps "$TOTAL" \
  "$@" > "/tmp/abl_${NAME}.log" 2>&1
EXIT=$?

RUNLOG=$(ls -t "$ROOT"/experiments/runlogs/${NAME}_*.jsonl 2>/dev/null | head -1)
CKPT_DIR=$(ls -dt "$ROOT"/experiments/* 2>/dev/null | grep -E "azuki_local" | head -1)

echo "{\"name\":\"$NAME\",\"branch\":\"$BRANCH\",\"commit\":\"$COMMIT\",\"start\":\"$START_TS\",\"exit\":$EXIT,\"total\":$TOTAL,\"runlog\":\"$RUNLOG\",\"args\":\"$*\"}" >> "$OUT/ablation_index.jsonl"
echo "ablation $NAME finished exit=$EXIT runlog=$RUNLOG"
