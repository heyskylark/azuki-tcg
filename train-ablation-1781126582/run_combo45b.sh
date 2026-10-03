#!/bin/bash
# Clean combo45 rerun (combo45b): the resumed combo45 leg is INVALID — the
# model-only resume (fresh optimizer + restarted lr schedule) collapsed policy
# entropy to 0.08 and cratered draft-vs-ref to 26.6%. This rerun is fresh from
# scratch with the full recipe; crash-truncation fix is live so the 00:57-class
# engine bug cannot deadlock it.
# Then: draftref 192 + per-checkpoint gate-KL/critic trajectories for
# combo45b and anneal45 (anneal45's own run remains valid).
# Detach with: setsid nohup bash train-ablation-1781126582/run_combo45b.sh \
#   > /tmp/run45m_chain7.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
mkdir -p "$RESULTS" experiments/runlogs

STEPS=45000000

COMMON_ARGS=(
  --config python/config/azuki_deckbuild_native_3090.ini
  --jsonl-log experiments/runlogs
  --env.deck_snapshot_every 25
  --league.keep_recent 6 --league.keep_mid 4 --league.keep_old 3
  --train.checkpoint_interval 100
  --train.seed 42
)

launch() {
  local TAG=$1; shift
  mkdir -p "experiments/abl_snapshots/$TAG" "experiments/league/$TAG"
  echo "[chain] $(date +%F_%T) launching $TAG steps=$STEPS extra=$*"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/train.py \
    "${COMMON_ARGS[@]}" \
    --tag "$TAG" \
    --train.total_timesteps "$STEPS" \
    --env.deck_snapshot_dir "experiments/abl_snapshots/$TAG" \
    --league.state_path "experiments/league/$TAG/league_state.json" \
    --league.opponent_dir "experiments/league/$TAG/opponents" \
    "$@" \
    > "/tmp/train_${TAG}.log" 2>&1
  echo "[chain] $(date +%F_%T) $TAG exited rc=$?"
}

draftref() {
  local TAG=$1
  local CKPT
  CKPT=$(ls -t experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    echo "[chain] $TAG: no checkpoint found for draftref eval"
    return 0
  fi
  echo "[chain] $(date +%F_%T) draftref eval $TAG ckpt=$CKPT"
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/draft_vs_reference_eval.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --checkpoint "$CKPT" --episodes 192 --argmax \
    --json "$RESULTS/run45_${TAG}_draftref.json" \
    > "/tmp/draftref_${TAG}.log" 2>&1 || echo "[chain] draftref $TAG failed (non-fatal)"
}

trajectory() {
  local TAG=$1 SCRIPT=$2 PREFIX=$3 EPISODES=$4
  local OUT="$RESULTS/run45_${TAG}"
  mkdir -p "$OUT"
  for CKPT in $(ls experiments/azuki_local_${TAG}_*/model_azuki_local_*.pt 2>/dev/null | sort); do
    local EP JSON
    EP=$(basename "$CKPT" | grep -oE '[0-9]+' | tail -1)
    JSON="$OUT/${PREFIX}_ep${EP}.json"
    [ -f "$JSON" ] && continue
    echo "[chain] $(date +%F_%T) $PREFIX $TAG ep$EP"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    .venv/bin/python "train-ablation-1781126582/$SCRIPT" \
      --checkpoint "$CKPT" --episodes "$EPISODES" --device cpu \
      --json "$JSON" > /dev/null 2>&1 || echo "[chain] $PREFIX $TAG ep$EP failed (non-fatal)"
  done
}

# --- combo45b: fresh full-recipe run ---
export AZK_REWARD_SHAPING_ANNEAL=1
export AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0
export AZK_REWARD_SHAPING_ANNEAL_FINAL=0.05
export AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12
export AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40
launch combo45b \
  --policy.gate_id_embedding_enabled true \
  --policy.deck_pick_smoothing_eps 0.02 \
  --env.draft_same_element_matchup_prob 0.35
draftref combo45b
unset AZK_REWARD_SHAPING_ANNEAL AZK_REWARD_SHAPING_ANNEAL_INITIAL AZK_REWARD_SHAPING_ANNEAL_FINAL \
      AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES

# --- evidence trajectories (CPU): valid runs only ---
for TAG in combo45b anneal45; do
  trajectory "$TAG" probe_gate_kl.py gate_kl 4
  trajectory "$TAG" probe_critic_gate.py critic 6
done

echo "[chain] $(date +%F_%T) ALL DONE"
