#!/bin/bash
# s14prod45: FULL PRODUCTION SPEC at 45M — the first run combining every adopted
# lever: anneal 12/40 + floor 0.15, portal-GP 0.3, pick-eps 0.02, oversample 0.35,
# xgate replay 0.15 + AZK_XGATE_MASK, TEXT-ONLY gates (gate_id false, S8),
# league frozen_ratio 0.4 + persistent PFSP (S9/S10), S12 early-tempo cap4
# (0.1/4) and S13 damage-mitigation (0.15/10). Control = s11final45 (same seed
# 42, same spec minus S12+S13; both user-designed rewards were validated at 15M
# with gate_id TRUE, so this run also confirms they compose with text-only gates
# at 45M).
# Readout: final draftref 384 (per-gate winrates) + windowed draftref (ep1000..2900)
# + KL/critic final probes + H2H ladder (2930/1500/500) + per-gate strategy
# reports (training-meta deck report for S14 AND s11final45 control; eval-grade
# gate-vs-gate matchup matrix via profile_gate_matchups.py) + league telemetry
# (promotions + PFSP) + per-checkpoint KL/critic trajectories.
# Detach with: setsid nohup bash train-ablation-1781126582/run_s14.sh \
#   > /tmp/s14.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
TAG=s14prod45
mkdir -p "$RESULTS" experiments/runlogs "experiments/abl_snapshots/$TAG" "experiments/league/$TAG"

echo "[s14] $(date +%F_%T) launching $TAG (45M, production spec + S12 cap4 + S13 dmg)"
AZK_REWARD_SHAPING_ANNEAL=1 AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 AZK_PORTAL_GP_BONUS=0.3 \
AZK_XGATE_MASK=1 AZK_PFSP=1 \
AZK_EARLY_TEMPO_BONUS=0.1 AZK_EARLY_TEMPO_CAP=4 \
AZK_DMG_MITIGATION_BONUS=0.15 AZK_DMG_MITIGATION_CAP=10 \
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
  --train.total_timesteps 45000000 \
  --env.deck_snapshot_dir "experiments/abl_snapshots/$TAG" \
  --league.state_path "experiments/league/$TAG/league_state.json" \
  --league.opponent_dir "experiments/league/$TAG/opponents" \
  --policy.gate_id_embedding_enabled false \
  --policy.deck_pick_smoothing_eps 0.02 \
  --env.draft_same_element_matchup_prob 0.35 \
  --env.draft_cross_gate_replay_prob 0.15 \
  --league.frozen_ratio 0.4 \
  > "/tmp/train_${TAG}.log" 2>&1
echo "[s14] $(date +%F_%T) $TAG train exited rc=$?"

CKPT=$(ls -t experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[s14] NO CHECKPOINT FOUND — aborting readout"
  touch "$RESULTS/${TAG}.FAILED"
  exit 1
fi
DIR=$(dirname "$CKPT")
echo "[s14] final checkpoint: $CKPT"

# ---- final draftref (384 eps -> ~48/gate for per-gate winrates) ----
PYTHONPATH=build/python/src:python/src \
.venv/bin/python python/src/draft_vs_reference_eval.py \
  --config python/config/azuki_deckbuild_native_3090.ini \
  --checkpoint "$CKPT" --episodes 384 --argmax \
  --json "$RESULTS/${TAG}_draftref.json" > "/tmp/draftref_${TAG}.log" 2>&1 || true
echo "[s14] $(date +%F_%T) final draftref done"

# ---- final KL + critic probes ----
mkdir -p "$RESULTS/run45_${TAG}"
OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
  --checkpoint "$CKPT" --episodes 8 --device cpu \
  --json "$RESULTS/run45_${TAG}/gate_kl_final.json" > /dev/null 2>&1 || true
OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/probe_critic_gate.py \
  --checkpoint "$CKPT" --episodes 12 --device cpu \
  --json "$RESULTS/run45_${TAG}/critic_final.json" > /dev/null 2>&1 || true
echo "[s14] $(date +%F_%T) final probes done"

# ---- windowed draftref (meta-cycle protocol: never judge a 45M run by one ckpt) ----
for EP in 001000 001500 002000 002500 002900; do
  C="$DIR/model_azuki_local_${EP}.pt"
  [ -f "$C" ] || continue
  J="$RESULTS/${TAG}_draftref_ep${EP}.json"
  [ -f "$J" ] && continue
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/draft_vs_reference_eval.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --checkpoint "$C" --episodes 96 --argmax \
    --json "$J" > /dev/null 2>&1 || true
  echo "[s14] $(date +%F_%T) windowed draftref ep$EP done"
done

# ---- H2H ladder: final vs mid vs early (192 eps, legacy config, seat-fair) ----
run_h2h() {
  local A=$1 B=$2 LA=$3 LB=$4 OUT=$5
  [ -f "$DIR/model_azuki_local_${A}.pt" ] || { echo "[s14] h2h $OUT: missing ckpt $A"; return 0; }
  [ -f "$DIR/model_azuki_local_${B}.pt" ] || { echo "[s14] h2h $OUT: missing ckpt $B"; return 0; }
  PYTHONPATH=build/python/src:python/src .venv/bin/python python/src/head_to_head_eval.py \
    --config python/config/azuki_deckbuild_3090.ini \
    --checkpoint-a "$DIR/model_azuki_local_${A}.pt" \
    --checkpoint-b "$DIR/model_azuki_local_${B}.pt" \
    --label-a "$LA" --label-b "$LB" --episodes 192 --num-envs 12 \
    --json "$RESULTS/h2h_${TAG}_${OUT}.json" > "/tmp/h2h_${TAG}_${OUT}.log" 2>&1 || echo "[s14] h2h $OUT failed"
  echo "[s14] $(date +%F_%T) h2h $OUT done"
}
run_h2h 002930 001500 final mid final_vs_mid
run_h2h 002930 000500 final early final_vs_early
run_h2h 001500 000500 mid early mid_vs_early

# ---- per-gate strategy reports ----
# (a) training-meta deck report (on-policy, 6 time buckets shows evolution)
PYTHONPATH=python/src .venv/bin/python train-ablation-1781126582/analyze_decks.py \
  "experiments/abl_snapshots/$TAG" --buckets 6 --top 12 \
  --csv "$RESULTS/${TAG}_deck_report.csv" \
  > "$RESULTS/${TAG}_deck_report.txt" 2>&1 || true
# (b) control comparison: same report for s11final45 (spec minus S12+S13)
PYTHONPATH=python/src .venv/bin/python train-ablation-1781126582/analyze_decks.py \
  "experiments/abl_snapshots/s11final45" --buckets 6 --top 12 \
  --csv "$RESULTS/s11final45_deck_report.csv" \
  > "$RESULTS/s11final45_deck_report.txt" 2>&1 || true
echo "[s14] $(date +%F_%T) deck reports done"

# (c) eval-grade gate-vs-gate matchup matrix + per-gate action/comp profiles
OMP_NUM_THREADS=6 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/profile_gate_matchups.py \
  --checkpoint "$CKPT" --episodes 16 --device cpu \
  --out-prefix "$RESULTS/${TAG}_gate" \
  > "/tmp/gate_matchups_${TAG}.log" 2>&1 || echo "[s14] gate matchup profiler failed"
echo "[s14] $(date +%F_%T) gate matchup profiler done"

# ---- league telemetry summary (promotions + PFSP) ----
.venv/bin/python - "$TAG" > "$RESULTS/${TAG}_league_summary.txt" 2>&1 <<'PY' || true
import json, sys, glob
from pathlib import Path
tag = sys.argv[1]
state_path = Path(f"experiments/league/{tag}/league_state.json")
print(f"== league state ({state_path}) ==")
if state_path.exists():
    st = json.loads(state_path.read_text())
    print(f"champion_policy_id: {st.get('champion_policy_id')}")
    entries = st.get("entries", [])
    print(f"pool entries: {len(entries)}")
    for e in entries:
        keys = {k: e.get(k) for k in ("policy_id", "kind", "promoted_at", "games", "win_rate", "created_step") if k in e}
        print(f"  {keys}")
else:
    print("MISSING")
print()
print("== jsonl runlog league stats (last-20% means) ==")
logs = sorted(glob.glob(f"experiments/runlogs/*{tag}*.jsonl"))
if not logs:
    print("no runlog found")
keys = ("league/pfsp_picked_winrate", "league/pfsp_pool_min_winrate",
        "league/frozen_matchup_fraction", "league/promotions", "league/pool_size")
for lp in logs:
    rows = []
    with open(lp) as h:
        for line in h:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        continue
    tail = rows[int(len(rows) * 0.8):]
    print(f"{lp}: {len(rows)} rows")
    for k in keys:
        vals = [r[k] for r in tail if k in r and isinstance(r[k], (int, float))]
        if vals:
            print(f"  {k}: mean={sum(vals)/len(vals):.4f} last={vals[-1]:.4f} n={len(vals)}")
    # any promotion-ish keys we did not anticipate
    seen = sorted({k for r in rows for k in r if "promot" in k or "champion" in k})
    for k in seen:
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
        if vals:
            print(f"  {k}: first={vals[0]} last={vals[-1]}")
PY
echo "[s14] $(date +%F_%T) league summary done"
echo "S14_HEADLINE_DONE"

# ---- per-checkpoint trajectories (KL every ckpt, critic every 200) ----
for C in $(ls experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | sort); do
  EP=$(basename "$C" | grep -oE '[0-9]+' | tail -1)
  J="$RESULTS/run45_${TAG}/gate_kl_ep${EP}.json"
  [ -f "$J" ] && continue
  OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
    --checkpoint "$C" --episodes 8 --device cpu --json "$J" > /dev/null 2>&1 || true
  echo "[s14] $(date +%F_%T) gate_kl ep$EP"
done
for C in $(ls experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | sort); do
  EP=$(basename "$C" | grep -oE '[0-9]+' | tail -1)
  case $EP in *00) ;; *) continue ;; esac
  case $((10#${EP} % 200)) in 0) ;; *) continue ;; esac
  J="$RESULTS/run45_${TAG}/critic_ep${EP}.json"
  [ -f "$J" ] && continue
  OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/probe_critic_gate.py \
    --checkpoint "$C" --episodes 6 --device cpu --json "$J" > /dev/null 2>&1 || true
  echo "[s14] $(date +%F_%T) critic ep$EP"
done

touch "$RESULTS/${TAG}.ALL_DONE"
echo "[s14] $(date +%F_%T) S14_ALL_DONE"
