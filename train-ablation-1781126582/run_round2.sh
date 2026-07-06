#!/bin/bash
# Round-2 ablation chain: ctrl2 -> anneal1 -> combo1, serial on the single 3090.
# Each arm: 15M steps, seed 42, mb2048, ckpt interval 100, own league dir,
# snapshots every 25th episode. After each arm: draft-vs-reference eval (96 eps).
# Detach with: setsid nohup bash train-ablation-1781126582/run_round2.sh \
#   > /tmp/round2_chain.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
mkdir -p "$RESULTS" experiments/runlogs

COMMON_ARGS=(
  --config python/config/azuki_deckbuild_3090.ini
  --jsonl-log experiments/runlogs
  --env.deck_snapshot_every 25
  --league.keep_recent 2 --league.keep_mid 1 --league.keep_old 1
  --train.checkpoint_interval 100
  --train.minibatch_size 2048
  --train.seed 42
)

launch() {
  local TAG=$1; shift
  local STEPS=$1; shift
  # remaining args: extra train.py args; env vars come via caller's `env -i`? no —
  # exported vars set by the caller before invoking launch.
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
    --config python/config/azuki_deckbuild_3090.ini \
    --checkpoint "$CKPT" --episodes 96 --argmax \
    --json "$RESULTS/round2_${TAG}_draftref.json" \
    > "/tmp/draftref_${TAG}.log" 2>&1 || echo "[chain] draftref $TAG failed (non-fatal)"
}

STEPS=15000000

# --- arm 1: control on new stack ---
launch ctrl2 "$STEPS"
draftref ctrl2

# --- arm 2: reward shaping anneal (per-env episode counts! ~330 steps/ep-env) ---
export AZK_REWARD_SHAPING_ANNEAL=1
export AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0
export AZK_REWARD_SHAPING_ANNEAL_FINAL=0.05
export AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=8
export AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=25
launch anneal1 "$STEPS"
draftref anneal1

# --- arm 3: gate identity embedding only (anneal OFF) ---
unset AZK_REWARD_SHAPING_ANNEAL AZK_REWARD_SHAPING_ANNEAL_INITIAL AZK_REWARD_SHAPING_ANNEAL_FINAL \
      AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES
launch gateid1 "$STEPS" \
  --policy.gate_id_embedding_enabled true
draftref gateid1

# --- arm 4: anneal + gate id + pick exploration (the ceiling arm) ---
export AZK_REWARD_SHAPING_ANNEAL=1
export AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0
export AZK_REWARD_SHAPING_ANNEAL_FINAL=0.05
export AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=8
export AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=25
launch combo1 "$STEPS" \
  --policy.gate_id_embedding_enabled true \
  --policy.deck_pick_smoothing_eps 0.05
draftref combo1
unset AZK_REWARD_SHAPING_ANNEAL AZK_REWARD_SHAPING_ANNEAL_INITIAL AZK_REWARD_SHAPING_ANNEAL_FINAL \
      AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES

echo "[chain] $(date +%F_%T) ALL ARMS COMPLETE"
