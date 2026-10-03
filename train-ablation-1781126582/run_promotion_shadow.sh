#!/usr/bin/env bash
# Task-3-derived 15M shadow run for promotion-v2. The archive evaluator may
# record decisions and payoff telemetry, but promotion_shadow_mode prevents it
# from admitting/evicting policies or moving the production anchor.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

TAG=promotionv2_shadow15_final
PARENT=experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt
RUN_ROOT="experiments/league/$TAG"
if [[ -e "$RUN_ROOT/league_state.json" ]] ||
   [[ -e "$RUN_ROOT/league_state_promotion.json" ]]; then
  echo "[promotion-shadow] existing state for $TAG; refusing a non-fresh restart" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT" "experiments/runlogs" "experiments/abl_snapshots/$TAG"

env \
  AZK_REWARD_SHAPING_ANNEAL=1 \
  AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
  AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 \
  AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
  AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 \
  AZK_PORTAL_GP_BONUS=0.3 \
  AZK_XGATE_MASK=1 \
  AZK_PFSP=1 \
  AZK_EARLY_TEMPO_BONUS=0.1 \
  AZK_EARLY_TEMPO_CAP=4 \
  AZK_DMG_MITIGATION_BONUS=0.15 \
  AZK_DMG_MITIGATION_CAP=10 \
  AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
  AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
  AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4 \
  AZK_RESUME_ALLOW_BINDING_MISMATCH=1 \
  AZK_RESUME_ALLOW_SOURCE_DRIFT=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/train.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --resume-checkpoint "$PARENT" \
    --jsonl-log experiments/runlogs \
    --tag "$TAG" \
    --train.seed 42 \
    --train.total_timesteps 15000000 \
    --train.checkpoint_interval 100 \
    --env.deck_snapshot_every 25 \
    --env.deck_snapshot_dir "experiments/abl_snapshots/$TAG" \
    --league.state_path "$RUN_ROOT/league_state.json" \
    --league.opponent_dir "$RUN_ROOT/opponents" \
    --league.promotion_shadow_mode true \
    --league.production_anchor_checkpoint "$PARENT" \
    2>&1 | tee "/tmp/train_${TAG}.log"
