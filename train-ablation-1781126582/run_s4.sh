#!/bin/bash
# s4ref15: S4 reference-seat smoke (15M) on the FULL adopted production spec
# (S14 knobs). 20% of episodes seat one player on a fixed reference deck from
# the TRAIN split (even pool indices); odd indices are the HOLDOUT used only
# for eval. New metric under validation: ref_anchor_winrate (drafter winrate
# vs fixed decks — the external promotion yardstick per report §26.4).
# Readout: draftref full-pool 192 + holdout 192 (+ holdout controls on
# s14prod45 ep2000/final for comparability), windowed holdout draftref,
# KL/critic finals, H2H ladder 977/500/100, ref-filtered deck report,
# league + ref-anchor trajectory summary.
# Detach with: setsid nohup bash train-ablation-1781126582/run_s4.sh \
#   > /tmp/s4.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
TAG=s4ref15
TRAIN_INDICES="0,2,4,6,8,10,12,14,16"
HOLDOUT_INDICES="1,3,5,7,9,11,13,15,17"
mkdir -p "$RESULTS" experiments/runlogs "experiments/abl_snapshots/$TAG" "experiments/league/$TAG"

echo "[s4] $(date +%F_%T) launching $TAG (15M, prod spec + ref seats 0.20 train-split)"
AZK_REWARD_SHAPING_ANNEAL=1 AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 AZK_PORTAL_GP_BONUS=0.3 \
AZK_XGATE_MASK=1 AZK_PFSP=1 \
AZK_EARLY_TEMPO_BONUS=0.1 AZK_EARLY_TEMPO_CAP=4 \
AZK_DMG_MITIGATION_BONUS=0.15 AZK_DMG_MITIGATION_CAP=10 \
AZK_DRAFT_REF_SEAT_PROB=0.20 \
AZK_DRAFT_REF_DECK_INDICES="$TRAIN_INDICES" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=build/python/src:python/src \
.venv/bin/python python/src/train.py \
  --config python/config/azuki_deckbuild_native_3090.ini \
  --jsonl-log experiments/runlogs \
  --env.deck_snapshot_every 25 \
  --league.keep_recent 6 --league.keep_mid 4 --league.keep_old 3 \
  --train.checkpoint_interval 100 \
  --train.seed 42 \
  --tag "$TAG" \
  --train.total_timesteps 15000000 \
  --env.deck_snapshot_dir "experiments/abl_snapshots/$TAG" \
  --league.state_path "experiments/league/$TAG/league_state.json" \
  --league.opponent_dir "experiments/league/$TAG/opponents" \
  --policy.gate_id_embedding_enabled false \
  --policy.deck_pick_smoothing_eps 0.02 \
  --env.draft_same_element_matchup_prob 0.35 \
  --env.draft_cross_gate_replay_prob 0.15 \
  --league.frozen_ratio 0.4 \
  > "/tmp/train_${TAG}.log" 2>&1
echo "[s4] $(date +%F_%T) $TAG train exited rc=$?"

CKPT=$(ls -t experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[s4] NO CHECKPOINT FOUND — aborting readout"
  touch "$RESULTS/${TAG}.FAILED"
  exit 1
fi
DIR=$(dirname "$CKPT")
echo "[s4] final checkpoint: $CKPT"

draftref() {  # ckpt episodes json extra...
  local C=$1 EPS=$2 J=$3; shift 3
  [ -f "$J" ] && return 0
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/draft_vs_reference_eval.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --checkpoint "$C" --episodes "$EPS" --argmax --json "$J" "$@" \
    > /dev/null 2>&1 || echo "[s4] draftref $J failed"
}

# ---- final evals: full pool (legacy comparability) + holdout (clean) ----
draftref "$CKPT" 192 "$RESULTS/${TAG}_draftref.json"
draftref "$CKPT" 192 "$RESULTS/${TAG}_draftref_holdout.json" --deck-indices "$HOLDOUT_INDICES"
echo "[s4] $(date +%F_%T) final draftrefs done"

# ---- holdout controls: S14 peak + final (same holdout yardstick) ----
S14DIR=$(ls -d experiments/azuki_local_s14prod45_* 2>/dev/null | head -1)
if [ -n "$S14DIR" ]; then
  draftref "$S14DIR/model_azuki_local_002000.pt" 192 "$RESULTS/s14prod45_draftref_ep002000_holdout.json" --deck-indices "$HOLDOUT_INDICES"
  draftref "$S14DIR/model_azuki_local_002930.pt" 192 "$RESULTS/s14prod45_draftref_final_holdout.json" --deck-indices "$HOLDOUT_INDICES"
  echo "[s4] $(date +%F_%T) holdout controls done"
fi

# ---- windowed holdout draftref (anchor-vs-external correlation data) ----
for EP in 000300 000500 000700 000900; do
  C="$DIR/model_azuki_local_${EP}.pt"
  [ -f "$C" ] || continue
  draftref "$C" 96 "$RESULTS/${TAG}_draftref_holdout_ep${EP}.json" --deck-indices "$HOLDOUT_INDICES"
  echo "[s4] $(date +%F_%T) windowed holdout draftref ep$EP done"
done

# ---- KL + critic finals ----
mkdir -p "$RESULTS/run15_${TAG}"
OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
  --checkpoint "$CKPT" --episodes 8 --device cpu \
  --json "$RESULTS/run15_${TAG}/gate_kl_final.json" > /dev/null 2>&1 || true
OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/probe_critic_gate.py \
  --checkpoint "$CKPT" --episodes 12 --device cpu \
  --json "$RESULTS/run15_${TAG}/critic_final.json" > /dev/null 2>&1 || true
echo "[s4] $(date +%F_%T) probes done"

# ---- H2H ladder (15M: 977/500/100) ----
run_h2h() {
  local A=$1 B=$2 LA=$3 LB=$4 OUT=$5
  [ -f "$DIR/model_azuki_local_${A}.pt" ] || { echo "[s4] h2h $OUT: missing $A"; return 0; }
  [ -f "$DIR/model_azuki_local_${B}.pt" ] || { echo "[s4] h2h $OUT: missing $B"; return 0; }
  PYTHONPATH=build/python/src:python/src .venv/bin/python python/src/head_to_head_eval.py \
    --config python/config/azuki_deckbuild_3090.ini \
    --checkpoint-a "$DIR/model_azuki_local_${A}.pt" \
    --checkpoint-b "$DIR/model_azuki_local_${B}.pt" \
    --label-a "$LA" --label-b "$LB" --episodes 192 --num-envs 12 \
    --json "$RESULTS/h2h_${TAG}_${OUT}.json" > "/tmp/h2h_${TAG}_${OUT}.log" 2>&1 || echo "[s4] h2h $OUT failed"
  echo "[s4] $(date +%F_%T) h2h $OUT done"
}
run_h2h 000977 000500 final mid final_vs_mid
run_h2h 000977 000100 final early final_vs_early
run_h2h 000500 000100 mid early mid_vs_early

# ---- deck report (ref episodes excluded from per-gate stats) ----
PYTHONPATH=python/src .venv/bin/python train-ablation-1781126582/analyze_decks.py \
  "experiments/abl_snapshots/$TAG" --buckets 4 --top 12 \
  --csv "$RESULTS/${TAG}_deck_report.csv" \
  > "$RESULTS/${TAG}_deck_report.txt" 2>&1 || true
echo "[s4] $(date +%F_%T) deck report done"

# ---- league + ref-anchor summary ----
.venv/bin/python - "$TAG" > "$RESULTS/${TAG}_league_summary.txt" 2>&1 <<'PY' || true
import json, sys, glob
from pathlib import Path
tag = sys.argv[1]
state_path = Path(f"experiments/league/{tag}/league_state.json")
print(f"== league state ==")
if state_path.exists():
    st = json.loads(state_path.read_text())
    print(f"champion_policy_id: {st.get('champion_policy_id')}")
    print(f"pool entries: {len(st.get('entries', []))}")
else:
    print("MISSING")
print()
print("== ref anchor + league trajectories ==")
for lp in sorted(glob.glob(f"experiments/runlogs/*{tag}*.jsonl")):
    rows = []
    with open(lp) as h:
        for line in h:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        continue
    print(f"{lp}: {len(rows)} rows")
    keys = ("environment/ref_anchor_winrate", "environment/ref_seat_rate",
            "league/pfsp_picked_winrate", "environment/league/candidate_winrate_vs_champion",
            "environment/league/promotion_accepted")
    for k in keys:
        vals = [(i, r[k]) for i, r in enumerate(rows) if k in r and isinstance(r[k], (int, float))]
        if not vals:
            print(f"  {k}: ABSENT")
            continue
        n = len(vals)
        fifths = [vals[int(n*f/5):int(n*(f+1)/5)] for f in range(5)]
        means = [sum(v for _, v in chunk)/len(chunk) if chunk else float('nan') for chunk in fifths]
        print(f"  {k}: n={n} fifths=" + " ".join(f"{m:.4f}" for m in means))
PY
echo "[s4] $(date +%F_%T) league summary done"
touch "$RESULTS/${TAG}.HEADLINE_DONE"
echo "S4_HEADLINE_DONE"
