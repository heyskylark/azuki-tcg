#!/bin/bash
# One-shot evidence analysis for a round-2 arm. Usage: analyze_arm.sh TAG [DEVICE]
# Produces: snapshot buckets, identity probe, synergy lifts (+permutation p),
# gate-KL probe + deck-behavior probe on the final checkpoint.
set -u
TAG=$1
DEVICE=${2:-cpu}
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
DIR=train-ablation-1781126582
OUT="$DIR/results/round2_${TAG}"
mkdir -p "$OUT"
SNAP="experiments/abl_snapshots/$TAG"
CKPT=$(ls -t experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | head -1)
echo "=== $TAG ckpt=$CKPT ==="

.venv/bin/python "$DIR/analyze_decks.py" "$SNAP" --buckets 4 --top 8 \
  --csv "$OUT/snapshot_buckets.csv" > "$OUT/snapshot_buckets.txt" 2>&1
.venv/bin/python "$DIR/gate_identity_probe.py" "$SNAP" --last-frac 0.25 \
  > "$OUT/identity_probe.txt" 2>&1
.venv/bin/python "$DIR/synergy_lift.py" "$SNAP" --last-frac 0.25 --top 8 --per-gate \
  --permutations 300 > "$OUT/synergy_lift.txt" 2>&1

if [ -n "$CKPT" ]; then
  PYTHONPATH=build/python/src:python/src .venv/bin/python "$DIR/probe_gate_kl.py" \
    --checkpoint "$CKPT" --episodes 6 --device "$DEVICE" \
    --json "$OUT/gate_kl.json" > "$OUT/gate_kl.txt" 2>&1
  PYTHONPATH=build/python/src:python/src .venv/bin/python "$DIR/probe_deck_behavior.py" \
    --checkpoint "$CKPT" --episodes 24 --device "$DEVICE" \
    --json "$OUT/deck_behavior.json" > "$OUT/deck_behavior.txt" 2>&1
fi
grep -E "cross-gate|excess|conditioning" "$OUT"/*.txt | head -30
echo "=== $TAG analysis written to $OUT ==="
