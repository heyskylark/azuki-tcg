#!/usr/bin/env bash
# Fresh 45M confirmations for competitive reward arms that cleared the 15M
# rule. Run one arm at a time so the qualified set remains an explicit choice.
# Readout matches the existing S14 meta-cycle checkpoints and final seeds.
set -uo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT" || exit 1

RESULTS=train-ablation-1781126582/results/reward_45m
S14_RESULTS=train-ablation-1781126582/results
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
  find "$dir" -maxdepth 1 -type f -name 'model_azuki_local_*.pt' | \
    sort | tail -1
}

prepare_s14_controls() {
  local out="$RESULTS/s14_control"
  mkdir -p "$out"
  local epoch
  for epoch in 001000 001500 002000 002500 002900; do
    cp -f "$S14_RESULTS/s14prod45_draftref_ep${epoch}.json" \
      "$out/draftref_ep${epoch}.json" || return $?
    cp -f "$S14_RESULTS/run45_s14prod45/gate_kl_ep${epoch}.json" \
      "$out/gate_kl_ep${epoch}.json" || return $?
  done
  cp -f "$S14_RESULTS/s14prod45_draftref.json" \
    "$out/draftref_final.json" || return $?
  cp -f "$S14_RESULTS/run45_s14prod45/gate_kl_ep002930.json" \
    "$out/gate_kl_ep002930.json" || return $?
}

train_arm() {
  local tag=$1
  shift
  local extra_env=("$@")
  local existing_dir
  existing_dir=$(find_run_dir "$tag")
  local existing_final=""
  if [[ -n "$existing_dir" ]]; then
    existing_final=$(find_final_checkpoint "$existing_dir")
  fi

  if [[ -n "$existing_final" ]] && [[ -f "$RESULTS/$tag/TRAIN_DONE" ]]; then
    echo "[reward-45m] $tag already trained; skipping training"
    return 0
  fi
  if [[ -n "$existing_dir" ]] || \
      [[ -f "experiments/league/$tag/league_state.json" ]]; then
    echo "[reward-45m] $tag has incomplete state; refusing a non-fresh restart"
    touch "$RESULTS/$tag/PARTIAL_BLOCKED"
    return 1
  fi

  mkdir -p "$RESULTS/$tag" "experiments/abl_snapshots/$tag" \
    "experiments/league/$tag"
  printf '%s\n' "${COMMON_ENV[@]}" "${extra_env[@]}" \
    > "$RESULTS/$tag/environment.txt"
  echo "[reward-45m] $(date +%F_%T) launching $tag"
  env "${COMMON_ENV[@]}" "${extra_env[@]}" \
    .venv/bin/python python/src/train.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --jsonl-log experiments/runlogs \
      --env.deck_snapshot_every 25 \
      --league.keep_recent 6 --league.keep_mid 4 --league.keep_old 3 \
      --train.checkpoint_interval 100 \
      --train.seed 42 \
      --tag "$tag" \
      --train.total_timesteps 45000000 \
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
    echo "[reward-45m] $tag training failed rc=$rc"
    touch "$RESULTS/$tag/TRAIN_FAILED"
    return "$rc"
  fi
  touch "$RESULTS/$tag/TRAIN_DONE"
  echo "[reward-45m] $(date +%F_%T) $tag training complete"
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

eval_arm() {
  local tag=$1
  local dir
  dir=$(find_run_dir "$tag")
  if [[ -z "$dir" ]]; then
    echo "[reward-45m] no run directory for $tag"
    return 1
  fi
  local final_checkpoint
  final_checkpoint=$(find_final_checkpoint "$dir")
  if [[ -z "$final_checkpoint" ]]; then
    echo "[reward-45m] no checkpoint for $tag"
    return 1
  fi

  local out="$RESULTS/$tag"
  mkdir -p "$out"
  local final_epoch
  final_epoch=$(basename "$final_checkpoint" | grep -oE '[0-9]+' | tail -1)
  local window_epochs=(001000 001500 002000 002500 002900)
  local kl_epochs=("${window_epochs[@]}" "$final_epoch")

  # KL is CPU-only. Six OMP=2 workers fit beside the two fixed-seat draftref
  # workers on this 12-core/24-thread host and hide behind the longer GPU eval.
  local pids=()
  local epoch checkpoint output
  for epoch in "${kl_epochs[@]}"; do
    checkpoint="$dir/model_azuki_local_${epoch}.pt"
    output="$out/gate_kl_ep${epoch}.json"
    [[ -f "$checkpoint" ]] || continue
    [[ -f "$output" ]] && continue
    (
      OMP_NUM_THREADS=2 \
        PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
        .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
          --checkpoint "$checkpoint" --episodes 8 --device cpu \
          --json "$output" > "${output%.json}.log" 2>&1
    ) &
    pids+=("$!")
  done

  local rc=0
  echo "[reward-45m] $(date +%F_%T) $tag draftref final (384)"
  run_draftref "$final_checkpoint" 384 "$out/draftref_final.json" || rc=$?
  for epoch in "${window_epochs[@]}"; do
    checkpoint="$dir/model_azuki_local_${epoch}.pt"
    [[ -f "$checkpoint" ]] || continue
    echo "[reward-45m] $(date +%F_%T) $tag draftref ep$epoch (96)"
    run_draftref "$checkpoint" 96 "$out/draftref_ep${epoch}.json" || rc=$?
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
  done
  [[ $rc -eq 0 ]] || return "$rc"

  local runlog
  runlog=$(find experiments/runlogs -maxdepth 1 -type f \
    -name "${tag}_*.jsonl" -printf '%T@ %p\n' | sort -n | tail -1 | \
    cut -d' ' -f2-)
  if [[ -n "$runlog" ]] && [[ -f "$S14_LOG" ]]; then
    .venv/bin/python train-ablation-1781126582/compare_runs.py \
      "$S14_LOG" "$runlog" --points 20 \
      --metrics SPS,episode_length,p0_attack_selected_rate,p1_attack_selected_rate,p0_noop_selected_rate,p1_noop_selected_rate,p0_gate_portal_selected_rate,p1_gate_portal_selected_rate,p0_play_selected_rate,p1_play_selected_rate,p0_ability_selected_rate,p1_ability_selected_rate,p0_avg_leader_health,p1_avg_leader_health,p0_entity_damage_dealt,p1_entity_damage_dealt,p0_generated_ikz_conversion_rate,p1_generated_ikz_conversion_rate,p0_temporary_charge_realized,p1_temporary_charge_realized,p0_temporary_attack_damage_realized,p1_temporary_attack_damage_realized,p0_contextual_response_reserve_opportunities,p1_contextual_response_reserve_opportunities \
      > "$out/action_and_sps_metrics.txt"
  fi
  touch "$out/EVAL_DONE"
  echo "[reward-45m] $(date +%F_%T) $tag evaluation complete"
}

run_one() {
  local tag=$1
  shift
  train_arm "$tag" "$@" || return $?
  eval_arm "$tag"
}

prepare_s14_controls || exit $?

requested=${1:-}
case "$requested" in
  rs1entity45)
    run_one "$requested" \
      AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0.025 \
      AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP=6
    ;;
  rs2ikzconv45)
    run_one "$requested" \
      AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0 \
      AZK_GENERATED_IKZ_CONVERSION_BONUS=0.05 \
      AZK_GENERATED_IKZ_CONVERSION_STEP_CAP=4
    ;;
  rs3tempreal45)
    run_one "$requested" \
      AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
      AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
      AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4
    ;;
  rs4reserve45)
    run_one "$requested" \
      AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0 \
      AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0.08
    ;;
  *)
    echo "usage: $0 {rs1entity45|rs2ikzconv45|rs3tempreal45|rs4reserve45}"
    exit 2
    ;;
esac
