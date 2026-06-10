#!/bin/bash
# Detached training launcher: survives the launching shell.
# Usage: launch_deckbuild_run.sh TAG TOTAL_STEPS [extra train.py args...]
set -u
TAG="$1"; shift
TOTAL="$1"; shift
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
mkdir -p "experiments/abl_snapshots/$TAG" experiments/runlogs
setsid nohup env \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  AZK_DECKBUILD_SNAPSHOT_DIR="experiments/abl_snapshots/$TAG" \
  AZK_DECKBUILD_SNAPSHOT_EVERY=25 \
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/train.py \
    --config python/config/azuki_deckbuild_3090.ini \
    --jsonl-log experiments/runlogs \
    --tag "$TAG" \
    --train.total_timesteps "$TOTAL" \
    "$@" \
  > "/tmp/train_${TAG}.log" 2>&1 < /dev/null &
echo "launched $TAG pid=$!"
