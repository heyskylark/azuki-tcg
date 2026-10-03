#!/usr/bin/env bash
# Four 15M competitive-reward arms on the complete adopted S14 production
# stack. Milestones are epoch 100 (~1.54M), epoch 300 (~4.61M), and the final
# epoch (~15M). No reference-seat training or expensive strategy probes.
set -uo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT" || exit 1

RESULTS=train-ablation-1781126582/results/reward_smokes
S14_DIR=experiments/azuki_local_s14prod45_178404911119
S14_LOG=experiments/runlogs/s14prod45_178404911119.jsonl
mkdir -p "$RESULTS" experiments/runlogs

COMMON_ENV=(
  AZK_REWARD_SHAPING_ANNEAL=1
  AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0
  AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15
  AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12
  AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40
  AZK_PORTAL_GP_BONUS=0.3
  AZK_XGATE_MASK=1
  AZK_PFSP=1
  AZK_EARLY_TEMPO_BONUS=0.1
  AZK_EARLY_TEMPO_CAP=4
  AZK_DMG_MITIGATION_BONUS=0.15
  AZK_DMG_MITIGATION_CAP=10
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  PYTHONPATH=build/python/src:python/src
)

find_run_dir() {
  local tag=$1
  find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

find_final_checkpoint() {
  local dir=$1
  find "$dir" -maxdepth 1 -type f -name 'model_azuki_local_*.pt' | sort | tail -1
}

train_arm() {
  local tag=$1
  shift
  local extra_env=("$@")
  local existing_dir
  existing_dir=$(find_run_dir "$tag")
  if [[ -n "$existing_dir" ]] && [[ -f "$(find_final_checkpoint "$existing_dir")" ]] &&
      [[ -f "$RESULTS/$tag/TRAIN_DONE" ]]; then
    echo "[reward-smoke] $tag already complete; skipping training"
    return 0
  fi

  mkdir -p "$RESULTS/$tag" "experiments/abl_snapshots/$tag" \
    "experiments/league/$tag"
  printf '%s\n' "${COMMON_ENV[@]}" "${extra_env[@]}" \
    > "$RESULTS/$tag/environment.txt"
  echo "[reward-smoke] $(date +%F_%T) launching $tag"
  env "${COMMON_ENV[@]}" "${extra_env[@]}" \
    .venv/bin/python python/src/train.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --jsonl-log experiments/runlogs \
      --env.deck_snapshot_every 25 \
      --league.keep_recent 6 --league.keep_mid 4 --league.keep_old 3 \
      --train.checkpoint_interval 100 \
      --train.seed 42 \
      --tag "$tag" \
      --train.total_timesteps 15000000 \
      --env.deck_snapshot_dir "experiments/abl_snapshots/$tag" \
      --league.state_path "experiments/league/$tag/league_state.json" \
      --league.opponent_dir "experiments/league/$tag/opponents" \
      --policy.gate_id_embedding_enabled false \
      --policy.deck_pick_smoothing_eps 0.02 \
      --env.draft_same_element_matchup_prob 0.35 \
      --env.draft_cross_gate_replay_prob 0.15 \
      --league.frozen_ratio 0.4 \
      > "/tmp/train_${tag}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[reward-smoke] $tag training failed rc=$rc"
    touch "$RESULTS/$tag/TRAIN_FAILED"
    return "$rc"
  fi
  touch "$RESULTS/$tag/TRAIN_DONE"
  echo "[reward-smoke] $(date +%F_%T) $tag training complete"
}

run_draftref() {
  local checkpoint=$1
  local episodes=$2
  local output=$3
  [[ -f "$output" ]] && return 0
  PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/draft_vs_reference_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --episodes "$episodes" --argmax \
      --json "$output" > "${output%.json}.log" 2>&1
}

eval_s14_controls() {
  mkdir -p "$RESULTS/s14_control"
  for spec in 000100:96 000300:96 001000:96; do
    local epoch=${spec%%:*}
    local episodes=${spec##*:}
    local checkpoint="$S14_DIR/model_azuki_local_${epoch}.pt"
    local output="$RESULTS/s14_control/draftref_ep${epoch}.json"
    if [[ -f "$checkpoint" ]]; then
      run_draftref "$checkpoint" "$episodes" "$output" || return $?
      cp -f "train-ablation-1781126582/results/run45_s14prod45/gate_kl_ep${epoch}.json" \
        "$RESULTS/s14_control/gate_kl_ep${epoch}.json" 2>/dev/null || true
    fi
  done
}

eval_arm() {
  local tag=$1
  local dir
  dir=$(find_run_dir "$tag")
  if [[ -z "$dir" ]]; then
    echo "[reward-smoke] no run directory for $tag"
    return 1
  fi
  local final_checkpoint
  final_checkpoint=$(find_final_checkpoint "$dir")
  if [[ -z "$final_checkpoint" ]]; then
    echo "[reward-smoke] no checkpoint for $tag"
    return 1
  fi

  mkdir -p "$RESULTS/$tag"
  local final_epoch
  final_epoch=$(basename "$final_checkpoint" | grep -oE '[0-9]+' | tail -1)
  local specs=("000100:96" "000300:96" "$final_epoch:96")
  for spec in "${specs[@]}"; do
    local epoch=${spec%%:*}
    local episodes=${spec##*:}
    local checkpoint="$dir/model_azuki_local_${epoch}.pt"
    [[ -f "$checkpoint" ]] || continue
    echo "[reward-smoke] $(date +%F_%T) $tag draftref ep$epoch"
    run_draftref "$checkpoint" "$episodes" \
      "$RESULTS/$tag/draftref_ep${epoch}.json" || return $?
  done

  local pids=()
  for spec in "${specs[@]}"; do
    local epoch=${spec%%:*}
    local checkpoint="$dir/model_azuki_local_${epoch}.pt"
    local output="$RESULTS/$tag/gate_kl_ep${epoch}.json"
    [[ -f "$checkpoint" ]] || continue
    [[ -f "$output" ]] && continue
    (
      OMP_NUM_THREADS=4 PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
        .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
          --checkpoint "$checkpoint" --episodes 8 --device cpu \
          --json "$output" > "${output%.json}.log" 2>&1
    ) &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || return $?
  done

  local runlog
  runlog=$(find experiments/runlogs -maxdepth 1 -type f \
    -name "${tag}_*.jsonl" -printf '%T@ %p\n' | sort -n | tail -1 | \
    cut -d' ' -f2-)
  if [[ -n "$runlog" ]] && [[ -f "$S14_LOG" ]]; then
    .venv/bin/python train-ablation-1781126582/compare_runs.py \
      "$S14_LOG" "$runlog" --points 10 \
      --metrics SPS,episode_length,p0_attack_selected_rate,p1_attack_selected_rate,p0_gate_portal_selected_rate,p1_gate_portal_selected_rate,p0_activate_garden_or_leader_ability_selected_rate,p1_activate_garden_or_leader_ability_selected_rate,p0_entity_damage_dealt,p1_entity_damage_dealt,p0_generated_ikz_conversion_rate,p1_generated_ikz_conversion_rate,p0_temporary_charge_realized,p1_temporary_charge_realized,p0_temporary_attack_damage_realized,p1_temporary_attack_damage_realized,p0_contextual_response_reserve_opportunities,p1_contextual_response_reserve_opportunities \
      > "$RESULTS/$tag/action_and_sps_metrics.txt"
  fi
  touch "$RESULTS/$tag/EVAL_DONE"
}

run_one() {
  local tag=$1
  shift
  train_arm "$tag" "$@" || return $?
  eval_arm "$tag"
}

eval_s14_controls || exit $?

requested=${1:-all}
case "$requested" in
  all|rs1entity15|rs2ikzconv15|rs3tempreal15|rs4reserve15) ;;
  *)
    echo "usage: $0 [all|rs1entity15|rs2ikzconv15|rs3tempreal15|rs4reserve15]"
    exit 2
    ;;
esac

if [[ "$requested" == all || "$requested" == rs1entity15 ]]; then
  run_one rs1entity15 \
    AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0.025 \
    AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP=6 || exit $?
fi
if [[ "$requested" == all || "$requested" == rs2ikzconv15 ]]; then
  run_one rs2ikzconv15 \
    AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0 \
    AZK_GENERATED_IKZ_CONVERSION_BONUS=0.05 \
    AZK_GENERATED_IKZ_CONVERSION_STEP_CAP=4 || exit $?
fi
if [[ "$requested" == all || "$requested" == rs3tempreal15 ]]; then
  run_one rs3tempreal15 \
    AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
    AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
    AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4 || exit $?
fi
if [[ "$requested" == all || "$requested" == rs4reserve15 ]]; then
  run_one rs4reserve15 \
    AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0 \
    AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0.08 || exit $?
fi

touch "$RESULTS/SMOKES_DONE"
echo "[reward-smoke] $(date +%F_%T) requested smoke chain complete"
