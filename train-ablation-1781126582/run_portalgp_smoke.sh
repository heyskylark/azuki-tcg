#!/bin/bash
# portalgp1 smoke arm (15M): full combo recipe + AZK_PORTAL_GP_BONUS=0.3.
# Validates: early portal usage rises, external quality does not crater,
# gate-conditioning signals (KL/critic) improve vs combo45b's early ckpts.
# Gates the 45M portal-GP run. Runs alongside the CPU probe matrix.
# Detach with: setsid nohup bash train-ablation-1781126582/run_portalgp_smoke.sh \
#   > /tmp/portalgp_smoke.log 2>&1 < /dev/null &
set -u
ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"
RESULTS=train-ablation-1781126582/results
mkdir -p "$RESULTS" experiments/runlogs experiments/abl_snapshots/portalgp1 experiments/league/portalgp1

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
  --tag portalgp1 \
  --train.total_timesteps 15000000 \
  --env.deck_snapshot_dir experiments/abl_snapshots/portalgp1 \
  --league.state_path experiments/league/portalgp1/league_state.json \
  --league.opponent_dir experiments/league/portalgp1/opponents \
  --policy.gate_id_embedding_enabled true \
  --policy.deck_pick_smoothing_eps 0.02 \
  --env.draft_same_element_matchup_prob 0.35 \
  > /tmp/train_portalgp1.log 2>&1
echo "[portalgp] $(date +%F_%T) train exited rc=$?"

CKPT=$(ls -t experiments/azuki_local_portalgp1_*/model_azuki_local_*.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/draft_vs_reference_eval.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --checkpoint "$CKPT" --episodes 96 --argmax \
    --json "$RESULTS/portalgp1_draftref.json" \
    > /tmp/draftref_portalgp1.log 2>&1 || true
fi
echo "[portalgp] $(date +%F_%T) DONE"
