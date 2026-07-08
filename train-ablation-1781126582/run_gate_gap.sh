#!/bin/bash
# Achievable-gap probe driver: 3 play modes x 2 seed-shards per element,
# ~1030 episodes/mode. Usage: run_gate_gap.sh ELEMENT [DECKS] [SEEDS_PER_SHARD]
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results/gate_gap
mkdir -p "$RESULTS"
ELEMENT=${1:?element}
DECKS=${2:-6}
SEEDS=${3:-43}
CKPT=$(ls experiments/azuki_local_combo45b_*/model_azuki_local_002930.pt | head -1)

pids=()
for MODE in policy forced blocked; do
  for SHARD in 0 1; do
    OUT="$RESULTS/${ELEMENT,,}_${MODE}_s${SHARD}.json"
    [ -f "$OUT" ] && continue
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python train-ablation-1781126582/probe_gate_gap.py \
      --checkpoint "$CKPT" --element "$ELEMENT" --mode "$MODE" \
      --decks "$DECKS" --seeds "$SEEDS" --seed-offset $((SHARD * SEEDS)) \
      --json "$OUT" > "/tmp/gate_gap_${ELEMENT,,}_${MODE}_s${SHARD}.log" 2>&1 &
    pids+=($!)
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "SHARDS DONE rc=$rc"

.venv/bin/python - "$RESULTS" "$ELEMENT" <<'EOF'
import json, sys
from pathlib import Path
res, element = Path(sys.argv[1]), sys.argv[2].lower()
print(f"=== {element} achievable-gap summary ===")
for mode in ("policy", "forced", "blocked"):
    w = n = 0.0
    pa = pb = 0
    for p in sorted(res.glob(f"{element}_{mode}_s*.json")):
        d = json.loads(p.read_text())
        w += d["winrate_a"] * d["n"]; n += d["n"]
        pa += d["portal_steps_a"]; pb += d["portal_steps_b"]
    if n:
        wr = w / n
        se = (wr * (1 - wr) / n) ** 0.5
        print(f"  {mode:>7}: winrate(gate_a)={wr:.4f} +/- {2*se:.4f} (n={int(n)}) portals/ep a={pa/n:.2f} b={pb/n:.2f}")
EOF
