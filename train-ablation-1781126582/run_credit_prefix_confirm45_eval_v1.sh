#!/usr/bin/env bash
# Evaluate both causal contrasts from the three-arm 45M confirmation.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${CREDIT_PREFIX_CONFIRM_CAMPAIGN:-credit_prefix_confirm45_v1}
TRAIN_ROOT="train-ablation-1781126582/results/next_ablation_v1/stage3/$CAMPAIGN"
CREDIT_VIEW="train-ablation-1781126582/results/next_ablation_v1/stage2/${CAMPAIGN}_credit_view"

if [[ ! -f "$TRAIN_ROOT/LADDER_TRAIN_DONE" ]]; then
  echo "[$CAMPAIGN-eval] waiting for three-arm training"
  while [[ ! -f "$TRAIN_ROOT/LADDER_TRAIN_DONE" ]]; do
    for arm in standard_control control_no_prefix random_main_prefix; do
      if [[ -f "$TRAIN_ROOT/$arm/STARTUP_INVARIANTS_FAILED" ]]; then
        echo "[$CAMPAIGN-eval] $arm startup guard failed" >&2
        exit 1
      fi
    done
    sleep 30
  done
fi

for arm in standard_control control_no_prefix random_main_prefix; do
  [[ -f "$TRAIN_ROOT/$arm/TRAIN_DONE" ]] || {
    echo "[$CAMPAIGN-eval] missing completed arm: $arm" >&2
    exit 1
  }
done

mkdir -p "$CREDIT_VIEW"
ln -sfn "$ROOT/$TRAIN_ROOT/standard_control" "$CREDIT_VIEW/control"
ln -sfn "$ROOT/$TRAIN_ROOT/control_no_prefix" "$CREDIT_VIEW/full_episode_credit"
cp -a "$TRAIN_ROOT/parent_manifest.json" "$CREDIT_VIEW/parent_manifest.json"
cp -a "$TRAIN_ROOT/training_gate.json" "$CREDIT_VIEW/training_gate.json"

parent_epoch=$(awk -F= '$1=="parent_epoch" {print $2}' "$TRAIN_ROOT/campaign_config.txt")
target_epoch=$(awk -F= '$1=="target_epoch" {print $2}' "$TRAIN_ROOT/campaign_config.txt")
updates=$(awk -F= '$1=="updates" {print $2}' "$TRAIN_ROOT/campaign_config.txt")
credit_coef=$(awk -F= '$1=="credit_coef" {print $2}' "$TRAIN_ROOT/campaign_config.txt")
batch_drafts=$(awk -F= '$1=="batch_drafts" {print $2}' "$TRAIN_ROOT/campaign_config.txt")
update_interval=$(awk -F= '$1=="update_interval" {print $2}' "$TRAIN_ROOT/campaign_config.txt")
printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\nupdates=%s\ncredit_mode=retained_rows\ncandidate_arm=full_episode_credit\ncredit_coef=%s\nbatch_drafts=%s\nupdate_interval=%s\n' \
  "${CAMPAIGN}_credit_view" "$parent_epoch" "$target_epoch" "$updates" \
  "$credit_coef" "$batch_drafts" "$update_interval" \
  > "$CREDIT_VIEW/campaign_config.txt"
touch "$CREDIT_VIEW/LADDER_TRAIN_DONE"

export FULL_CREDIT_LADDER_CAMPAIGN="${CAMPAIGN}_credit_view"
export FULL_CREDIT_CANDIDATE_ARM=full_episode_credit
./train-ablation-1781126582/run_full_episode_credit_eval_v1.sh

export RANDOM_PREFIX_LADDER_CAMPAIGN="$CAMPAIGN"
./train-ablation-1781126582/run_random_main_prefix_eval_v1.sh

touch "$TRAIN_ROOT/CONFIRMATION_EVALUATION_DONE"
echo "[$CAMPAIGN-eval] both causal contrasts complete"
