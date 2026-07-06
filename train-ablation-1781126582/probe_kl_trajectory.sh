#!/bin/bash
# Gate-swap KL across an arm's checkpoints -> does gate conditioning GROW?
# Usage: probe_kl_trajectory.sh TAG [DEVICE] [EPISODES]
set -u
TAG=$1
DEVICE=${2:-cpu}
EPISODES=${3:-4}
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
OUT="train-ablation-1781126582/results/round2_${TAG}"
mkdir -p "$OUT"
for CKPT in $(ls experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | sort); do
  EP=$(basename "$CKPT" | grep -oE '[0-9]+' | tail -1)
  JSON="$OUT/gate_kl_ep${EP}.json"
  [ -f "$JSON" ] && continue
  echo "=== $TAG ep$EP ==="
  PYTHONPATH=build/python/src:python/src .venv/bin/python \
    train-ablation-1781126582/probe_gate_kl.py \
    --checkpoint "$CKPT" --episodes "$EPISODES" --device "$DEVICE" \
    --json "$JSON" 2>&1 | grep -E "^\["
done
echo "--- KL trajectory ($TAG) ---"
.venv/bin/python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for p in sorted(out.glob("gate_kl_ep*.json")):
    ep = int("".join(ch for ch in p.stem if ch.isdigit()))
    data = json.loads(p.read_text())
    mean_kl = sum(v["mean_kl"] for v in data.values()) / len(data)
    mean_tv = sum(v["mean_tv"] for v in data.values()) / len(data)
    rows.append((ep, mean_kl, mean_tv))
for ep, kl, tv in sorted(rows):
    print(f"  ep{ep:>5}  mean_KL={kl:.5f}  mean_TV={tv:.4f}")
EOF
