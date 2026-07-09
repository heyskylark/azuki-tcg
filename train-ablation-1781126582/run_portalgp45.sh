#!/bin/bash
# portalgp45: 45M scale-up of portalgp1 (best 15M arm: draftref 46.9%, 2.5x
# portal exposure, no external damage). Recipe + AZK_PORTAL_GP_BONUS=0.3.
# Tests whether portal exposure + 3x horizon moves sibling-gate KL off zero;
# per-checkpoint KL/critic trajectories afterwards.
# Detach with: setsid nohup bash train-ablation-1781126582/run_portalgp45.sh \
#   > /tmp/portalgp45_chain.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
mkdir -p "$RESULTS" experiments/runlogs experiments/abl_snapshots/portalgp45 experiments/league/portalgp45

export AZK_REWARD_SHAPING_ANNEAL=1
export AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0
export AZK_REWARD_SHAPING_ANNEAL_FINAL=0.05
export AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12
export AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40
export AZK_PORTAL_GP_BONUS=0.3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=build/python/src:python/src \
.venv/bin/python python/src/train.py \
  --config python/config/azuki_deckbuild_native_3090.ini \
  --jsonl-log experiments/runlogs \
  --env.deck_snapshot_every 25 \
  --league.keep_recent 6 --league.keep_mid 4 --league.keep_old 3 \
  --train.checkpoint_interval 100 \
  --train.seed 42 \
  --tag portalgp45 \
  --train.total_timesteps 45000000 \
  --env.deck_snapshot_dir experiments/abl_snapshots/portalgp45 \
  --league.state_path experiments/league/portalgp45/league_state.json \
  --league.opponent_dir experiments/league/portalgp45/opponents \
  --policy.gate_id_embedding_enabled true \
  --policy.deck_pick_smoothing_eps 0.02 \
  --env.draft_same_element_matchup_prob 0.35 \
  > /tmp/train_portalgp45.log 2>&1
echo "[portalgp45] $(date +%F_%T) train exited rc=$?"

CKPT=$(ls -t experiments/azuki_local_portalgp45_*/model_azuki_local_*.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/draft_vs_reference_eval.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --checkpoint "$CKPT" --episodes 192 --argmax \
    --json "$RESULTS/run45_portalgp45_draftref.json" \
    > /tmp/draftref_portalgp45.log 2>&1 || true
fi

trajectory() {
  local TAG=$1 SCRIPT=$2 PREFIX=$3 EPISODES=$4
  local OUT="$RESULTS/run45_${TAG}"
  mkdir -p "$OUT"
  for CKPT in $(ls experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | sort); do
    local EP JSON
    EP=$(basename "$CKPT" | grep -oE '[0-9]+' | tail -1)
    JSON="$OUT/${PREFIX}_ep${EP}.json"
    [ -f "$JSON" ] && continue
    echo "[portalgp45] $(date +%F_%T) $PREFIX ep$EP"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python "train-ablation-1781126582/$SCRIPT" \
      --checkpoint "$CKPT" --episodes "$EPISODES" --device cpu \
      --json "$JSON" > /dev/null 2>&1 || true
  done
}
trajectory portalgp45 probe_gate_kl.py gate_kl 4
trajectory portalgp45 probe_critic_gate.py critic 6
echo "[portalgp45] $(date +%F_%T) ALL DONE"
