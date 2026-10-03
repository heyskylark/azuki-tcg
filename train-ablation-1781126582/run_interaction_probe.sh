#!/bin/bash
# Deck-composition x gate interaction matrix: for each sibling pair, archetype
# vs contrast decks under gate A and under gate B (mirror-gate design).
# interaction = WR(arch|A) - WR(arch|B). 2 shards/cell, n=500/cell.
# Detach with: setsid nohup bash train-ablation-1781126582/run_interaction_probe.sh \
#   > /tmp/interaction_probe.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results/gate_ix
mkdir -p "$RESULTS"
CKPT=$(ls experiments/azuki_local_combo45b_*/model_azuki_local_002930.pt | head -1)

declare -A GATES=(
  [LIGHTNING]="STT01-002 AZK01-120"
  [WATER]="STT02-002 AZK01-126"
  [FIRE]="AZK01-122 STT04-002"
  [EARTH]="AZK01-124 STT03-002"
)

for EL in LIGHTNING WATER FIRE EARTH; do
  read -r GA GB <<< "${GATES[$EL]}"
  for GATE in "$GA" "$GB"; do
    pids=()
    for SHARD in 0 1; do
      OUT="$RESULTS/${EL,,}_${GATE}_s${SHARD}.json"
      [ -f "$OUT" ] && continue
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
      .venv/bin/python train-ablation-1781126582/probe_deck_gate_interaction.py \
        --checkpoint "$CKPT" --element "$EL" --gate "$GATE" \
        --seeds 125 --seed-offset $((SHARD * 125)) \
        --json "$OUT" > "/tmp/gate_ix_${EL,,}_${GATE}_s${SHARD}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    echo "[ix] $(date +%F_%T) done $EL $GATE"
  done
done
echo "[ix] $(date +%F_%T) ALL DONE"

.venv/bin/python - "$RESULTS" <<'EOF'
import json, sys, glob
from pathlib import Path
from collections import defaultdict
res = Path(sys.argv[1])
cells = defaultdict(lambda: [0.0, 0])
for p in res.glob("*_s*.json"):
    d = json.loads(p.read_text())
    key = (d["element"], d["gate"])
    cells[key][0] += d["winrate_archetype"] * d["n"]; cells[key][1] += d["n"]
by_el = defaultdict(dict)
for (el, gate), (w, n) in cells.items():
    by_el[el][gate] = (w / n, n)
print("=== composition x gate interaction ===")
for el, gates in sorted(by_el.items()):
    if len(gates) == 2:
        (g1, (w1, n1)), (g2, (w2, n2)) = sorted(gates.items())
        se = (w1*(1-w1)/n1 + w2*(1-w2)/n2) ** 0.5
        print(f"  {el}: WR(arch|{g1})={w1:.3f} WR(arch|{g2})={w2:.3f} interaction={w1-w2:+.3f} +/- {2*se:.3f}")
EOF
