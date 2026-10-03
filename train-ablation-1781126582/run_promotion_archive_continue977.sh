#!/usr/bin/env bash
# Continue the validated promotion-v2 campaign from p977 to the cumulative
# 45M-row horizon. Quality admission is enabled, but the production anchor and
# PPO opponent distribution remain fixed by their independent controls.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

TAG=promotionv2_archive45_final_resume977
SOURCE_RUN=experiments/azuki_local_promotionv2_shadow15_final_resume300_contiguous_178432002673
CHECKPOINT="$SOURCE_RUN/model_azuki_local_000977.pt"
TRAINER_STATE="$SOURCE_RUN/trainer_state_000977.pt"
SOURCE_LEAGUE_ROOT=experiments/league/promotionv2_shadow15_final
RUN_ROOT=experiments/league/promotionv2_archive45_final
RUN_LOG="experiments/runlogs/${TAG}.jsonl"
EXPECTED_MODEL_SHA=6baac2e2451269f95fbf1a6b93c032029742b018a209a24f127074aaff842e17
EXPECTED_TRAINER_SHA=cc461689f4057801a5337f1fc3e790f6edc4436313a883f545cfd2f9d2a86e7c

if [[ ! -f "$CHECKPOINT" ]] || [[ ! -f "$TRAINER_STATE" ]]; then
  echo "[promotion-archive] missing matching p977 checkpoint/trainer state" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_LEAGUE_ROOT/league_state.json" ]] ||
   [[ ! -f "$SOURCE_LEAGUE_ROOT/league_state_promotion.json" ]]; then
  echo "[promotion-archive] missing completed shadow league state" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]] || [[ -e "$RUN_LOG" ]] ||
   compgen -G "experiments/azuki_local_${TAG}_*" >/dev/null; then
  echo "[promotion-archive] existing output for $TAG; refusing duplicate resume" >&2
  exit 1
fi

actual_model_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
actual_trainer_sha=$(sha256sum "$TRAINER_STATE" | awk '{print $1}')
if [[ "$actual_model_sha" != "$EXPECTED_MODEL_SHA" ]]; then
  echo "[promotion-archive] p977 model hash mismatch: $actual_model_sha" >&2
  exit 1
fi
if [[ "$actual_trainer_sha" != "$EXPECTED_TRAINER_SHA" ]]; then
  echo "[promotion-archive] p977 trainer-state hash mismatch: $actual_trainer_sha" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT" "experiments/runlogs" "experiments/abl_snapshots/$TAG"
cp "$SOURCE_LEAGUE_ROOT/league_state.json" "$RUN_ROOT/league_state.json"
cp "$SOURCE_LEAGUE_ROOT/league_state_promotion.json" "$RUN_ROOT/league_state_promotion.json"

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
  AZK_RESUME_COMPLETED_EPISODES=83 \
  AZK_RESUME_ALLOW_BINDING_MISMATCH=1 \
  AZK_RESUME_ALLOW_SOURCE_DRIFT=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=build/python/src:python/src \
  .venv/bin/python python/src/train.py \
    --config python/config/azuki_deckbuild_native_3090.ini \
    --resume-checkpoint "$CHECKPOINT" \
    --resume-load-optimizer \
    --resume-restart-lr-schedule \
    --no-resume-auto-reset-critic \
    --jsonl-log experiments/runlogs \
    --tag "$TAG" \
    --train.seed 42 \
    --train.learning_rate 0.0003 \
    --train.ent_coef 0.002 \
    --train.ent_coef_anneal_initial 0.002 \
    --train.ent_coef_anneal_final 0.002 \
    --train.total_timesteps 45000000 \
    --train.checkpoint_interval 100 \
    --env.deck_snapshot_every 25 \
    --env.deck_snapshot_dir "experiments/abl_snapshots/$TAG" \
    --league.state_path "$RUN_ROOT/league_state.json" \
    --league.opponent_dir "$RUN_ROOT/opponents" \
    --league.promotion_shadow_mode false \
    --league.promotion_archive_affects_training_pool false \
    --league.promotion_panel_refresh_epochs 3000 \
    --league.production_anchor_checkpoint \
      experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt \
    2>&1 | tee "/tmp/train_${TAG}.log"
