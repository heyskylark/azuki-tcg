#!/bin/bash
# s1auxann: 45M confirmation of the auxvd recipe (first nonzero KL at 15M).
#   auxv1  : vboot 0.05
#   auxd1  : sibdiff 2.0 (clipped, cap 0.05)
#   auxvd1 : both
# After each arm: draftref 96 + KL/critic probes on the final checkpoint.
# Detach with: setsid nohup bash train-ablation-1781126582/run_aux_matrix.sh \
#   > /tmp/aux_matrix.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
mkdir -p "$RESULTS" experiments/runlogs

run_arm() {
  local TAG=$1 VBOOT=$2 SIBDIFF=$3
  mkdir -p "experiments/abl_snapshots/$TAG" "experiments/league/$TAG"
  echo "[auxmx] $(date +%F_%T) launching $TAG vboot=$VBOOT sibdiff=$SIBDIFF"
  AZK_REWARD_SHAPING_ANNEAL=1 AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
  AZK_REWARD_SHAPING_ANNEAL_FINAL=0.05 AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
  AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 AZK_PORTAL_GP_BONUS=0.3 \
  AZK_DRAFT_VBOOT_COEF=$VBOOT AZK_DRAFT_SIBDIFF_COEF=$SIBDIFF AZK_DRAFT_SIBDIFF_CAP=0.05 \
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
    --policy.gate_id_embedding_enabled true \
    --policy.deck_pick_smoothing_eps 0.02 \
    --env.draft_same_element_matchup_prob 0.35 \
    --env.draft_cross_gate_replay_prob 0.15 \
    --league.frozen_ratio 0.4 \
    > "/tmp/train_${TAG}.log" 2>&1
  echo "[auxmx] $(date +%F_%T) $TAG exited rc=$?"
  local CKPT
  CKPT=$(ls -t experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && return 0
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/draft_vs_reference_eval.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --checkpoint "$CKPT" --episodes 96 --argmax \
    --json "$RESULTS/${TAG}_draftref.json" > "/tmp/draftref_${TAG}.log" 2>&1 || true
  mkdir -p "$RESULTS/run15_${TAG}"
  OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
    --checkpoint "$CKPT" --episodes 8 --device cpu \
    --json "$RESULTS/run15_${TAG}/gate_kl_final.json" > /dev/null 2>&1 || true
  OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/probe_critic_gate.py \
    --checkpoint "$CKPT" --episodes 12 --device cpu \
    --json "$RESULTS/run15_${TAG}/critic_final.json" > /dev/null 2>&1 || true
  echo "[auxmx] $(date +%F_%T) $TAG probes done"
}


export AZK_XGATE_MASK=1  # keep S3 replay+masking (production spec)
export AZK_PFSP=1
run_arm s9pfsp 0 0


echo "[auxmx] $(date +%F_%T) MATRIX DONE"

# per-checkpoint trajectories (KL growth is the question)
for CKPT in $(ls experiments/azuki_local_s1auxann_*/model_azuki_local_*.pt 2>/dev/null | sort); do
  EP=$(basename "$CKPT" | grep -oE '[0-9]+' | tail -1)
  JSON="$RESULTS/run45_s1auxann/gate_kl_ep${EP}.json"
  mkdir -p "$RESULTS/run45_s1auxann"
  [ -f "$JSON" ] && continue
  OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
    --checkpoint "$CKPT" --episodes 8 --device cpu --json "$JSON" > /dev/null 2>&1 || true
  echo "[s1auxann] $(date +%F_%T) gate_kl ep$EP"
done
for CKPT in $(ls experiments/azuki_local_s1auxann_*/model_azuki_local_*.pt 2>/dev/null | sort); do
  EP=$(basename "$CKPT" | grep -oE '[0-9]+' | tail -1)
  JSON="$RESULTS/run45_s1auxann/critic_ep${EP}.json"
  [ -f "$JSON" ] && continue
  OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/probe_critic_gate.py \
    --checkpoint "$CKPT" --episodes 6 --device cpu --json "$JSON" > /dev/null 2>&1 || true
  echo "[s1auxann] $(date +%F_%T) critic ep$EP"
done
echo "[s1auxann] $(date +%F_%T) ALL DONE"
