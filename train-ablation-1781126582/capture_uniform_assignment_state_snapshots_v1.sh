#!/usr/bin/env bash
# Preserve model/trainer/league state at registered Stage 1 checkpoint windows.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${UNIFORM_ASSIGNMENT_CAMPAIGN:-uniform_assignment_confirm45_v1}
RESULT_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage1/$CAMPAIGN"
read -r -a EPOCHS <<< "${UNIFORM_ASSIGNMENT_STATE_EPOCHS:-5200 5800 6800 7800}"
ARMS=(control uniform_assignment)

latest_run_dir() {
  local tag=$1
  find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 {print $2}'
}

link_or_copy() {
  local source=$1 destination=$2
  if ! ln "$source" "$destination" 2>/dev/null; then
    cp -a "$source" "$destination"
  fi
}

capture_epoch() {
  local arm=$1 epoch=$2
  local tag="${CAMPAIGN}_${arm}"
  local league_root="experiments/league/$CAMPAIGN/$arm"
  local destination
  destination=$(printf '%s/state_snapshots/%s/p%06d' "$RESULT_ROOT" "$arm" "$epoch")
  if [[ -f "$destination/SNAPSHOT_DONE" ]]; then
    return
  fi

  while true; do
    local run_dir model trainer metadata league promotion candidate_epoch
    local model_name trainer_name metadata_name
    run_dir=$(latest_run_dir "$tag")
    if [[ -z "$run_dir" ]]; then
      sleep 10
      continue
    fi
    model=$(printf '%s/model_azuki_local_%06d.pt' "$run_dir" "$epoch")
    trainer=$(printf '%s/trainer_state_%06d.pt' "$run_dir" "$epoch")
    metadata="$model.meta.json"
    model_name=$(basename "$model")
    trainer_name=$(basename "$trainer")
    metadata_name=$(basename "$metadata")
    league="$league_root/league_state.json"
    promotion="$league_root/league_state_promotion.json"
    if [[ ! -f "$model" || ! -f "$trainer" || ! -f "$metadata" || \
          ! -f "$league" || ! -f "$promotion" ]]; then
      sleep 10
      continue
    fi
    candidate_epoch=$(jq -r '
      .current_candidate_policy_id as $id |
      (.policies[] | select(.policy_id == $id) | .created_epoch) // -1
    ' "$league")
    if [[ "$candidate_epoch" != "$epoch" ]]; then
      sleep 5
      continue
    fi
    jq -e ".update == $epoch" "$metadata" >/dev/null

    local temporary="${destination}.tmp.$$"
    rm -rf "$temporary"
    mkdir -p "$temporary"
    link_or_copy "$model" "$temporary/$model_name"
    link_or_copy "$trainer" "$temporary/$trainer_name"
    link_or_copy "$metadata" "$temporary/$metadata_name"
    cp -a "$league" "$temporary/league_state.json"
    cp -a "$promotion" "$temporary/league_state_promotion.json"
    (
      cd "$temporary"
      sha256sum \
        "$model_name" \
        "$trainer_name" \
        "$metadata_name" \
        league_state.json \
        league_state_promotion.json
    ) > "$temporary/sha256.txt"
    printf 'campaign=%s\narm=%s\nepoch=%s\nsource_run=%s\n' \
      "$CAMPAIGN" "$arm" "$epoch" "$run_dir" > "$temporary/snapshot.txt"
    touch "$temporary/SNAPSHOT_DONE"
    mkdir -p "$(dirname "$destination")"
    mv "$temporary" "$destination"
    printf '[state-snapshot] %s %s p%s captured\n' "$CAMPAIGN" "$arm" "$epoch"
    return
  done
}

for arm in "${ARMS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    capture_epoch "$arm" "$epoch"
  done
done

touch "$RESULT_ROOT/STATE_SNAPSHOTS_DONE"
printf '[state-snapshot] %s complete\n' "$CAMPAIGN"
