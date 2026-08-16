#!/usr/bin/env bash
# Matched p3900->p4870 continuation testing terminal-only Fire leader credit.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${LEADER_CREDIT_CAMPAIGN:-strategy_recovery_leader15_v1}
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
PARENT_DIR=experiments/azuki_local_strategy_recovery_ladder15_v1_tempo_dedup_resume3300_178445644982
PARENT_MODEL="$PARENT_DIR/model_azuki_local_003900.pt"
PARENT_TRAINER="$PARENT_DIR/trainer_state_003900.pt"
PARENT_METADATA="$PARENT_MODEL.meta.json"
PARENT_LEAGUE=experiments/league/strategy_recovery_ladder15_v1/tempo_dedup/league_state.json
PARENT_PROMOTION=experiments/league/strategy_recovery_ladder15_v1/tempo_dedup/league_state_promotion.json
EXPECTED_MODEL_SHA=05f36461e636d6827b4b705a0f32f7c3bacc8972d152a84dbe63a0a628c38c45
EXPECTED_TRAINER_SHA=ad124ead1eb89707b1538ed39826722924178dfafec92fa43a00a1e86af290a8
EXPECTED_LEAGUE_SHA=a0cec57d05549f2a40af37c4d2849c96ad47bec190ac6ad96a13f2aca6f1b36d
EXPECTED_PROMOTION_SHA=9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a
EXPECTED_PARENT_EPISODES=323
PARENT_EPOCH=3900
TARGET_EPOCH=4870
TOTAL_TIMESTEPS=74803200
PEAK_LR=${LEADER_CREDIT_PEAK_LR:-0.00003}
SPS_HARD_FLOOR=1235
SPS_RELATIVE_FLOOR=0.95
TRAIN_INDICES=0,2,4,6,8,10,12,14,16
HOLDOUT_INDICES=1,3,5,7,9,11,13,15,17
OPPORTUNITY_GAMES=64
OPPORTUNITY_SHARDS=4
OPPORTUNITY_SEED0=820001

arms=(control leader_credit)
credit_coefs=(0 0.05)

require_hash() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$CAMPAIGN] $label hash mismatch: $actual" >&2
    exit 1
  fi
}

latest_jsonl() {
  local tag=$1
  find experiments/runlogs -maxdepth 1 -type f -name "${tag}_*.jsonl" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

cleanup_training_processes() {
  local tag=$1
  local pids=()
  mapfile -t pids < <(
    pgrep -f -- "python/src/train.py.*--tag ${tag}([[:space:]]|$)" 2>/dev/null || true
  )
  if (( ${#pids[@]} == 0 )); then
    return
  fi
  echo "[$CAMPAIGN] cleaning residual $tag processes: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 2
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

sps_summary() {
  local jsonl=$1
  .venv/bin/python - "$jsonl" <<'PY'
import json
import statistics
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("SPS"), (int, float)):
            rows.append(row)
steady = rows[20:] if len(rows) > 40 else rows[1:]
if not steady:
    steady = rows
values = [float(row["SPS"]) for row in steady]
print(
    len(rows),
    f"{statistics.median(values):.6f}",
    f"{statistics.median(values[-100:]):.6f}",
)
PY
}

signal_summary() {
  local jsonl=$1 output=$2
  .venv/bin/python - "$jsonl" "$output" <<'PY'
import json
from pathlib import Path
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("SPS"), (int, float)):
            rows.append(row)

def values(key):
    return [
        float(row[key]) for row in rows
        if isinstance(row.get(key), (int, float))
    ]

payload = {
    "metric_rows": len(rows),
    "max_labeled_rows": max(values("losses/win_prob_aux_labeled_rows"), default=0.0),
    "captured_records": sum(values("environment/leader_credit/captured")),
    "trained_examples": sum(values("losses/leader_credit_examples")),
    "max_clipfrac": max(values("losses/leader_credit_clipfrac"), default=0.0),
    "aux_train_seconds": sum(values("losses/leader_credit_train_seconds")),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

run_h2h() {
  local checkpoint_a=$1 checkpoint_b=$2 label_a=$3 label_b=$4 output=$5
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_policy_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint-a "$checkpoint_a" --checkpoint-b "$checkpoint_b" \
      --label-a "$label_a" --label-b "$label_b" --batch-envs 12 \
      --seeds 42001701,52001704,62001707,72001710,82001713,92001716 \
      --max-steps 600 --device cuda --json "$output" \
      > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '.summary.episodes == 192 and (.games | length) == 192 and .summary.timeout_rate == 0' \
    "$output" >/dev/null
}

run_reference_eval() {
  local checkpoint=$1 arm=$2 split=$3 indices=$4 output=$5
  if [[ ! -f "$output" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/native_reference_eval.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --checkpoint "$checkpoint" --opponent-checkpoint "$PARENT_MODEL" \
      --candidate-label "$arm" --opponent-label p3900_parent \
      --split "$split" --deck-indices "$indices" \
      --training-reference-indices "$TRAIN_INDICES" \
      --holdout-reference-indices "$HOLDOUT_INDICES" \
      --seeds 42009919,52009922 --batch-envs 12 --device cuda \
      --max-steps 600 --json "$output" \
      > >(tee "${output%.json}.log") 2>&1
  fi
  jq -e '.summary.episodes == 288 and (.games | length) == 288 and .summary.timeout_rate == 0' \
    "$output" >/dev/null
}

run_opportunity_eval() {
  local checkpoint=$1 arm=$2 arm_results=$3
  local output_dir="$arm_results/opportunity"
  local games_per_shard=$((OPPORTUNITY_GAMES / OPPORTUNITY_SHARDS))
  mkdir -p "$output_dir"
  local pids=()
  local shard
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    local log="$output_dir/shard${shard}.log"
    local shard_seed=$((OPPORTUNITY_SEED0 + 7919 * games_per_shard * shard))
    if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]; then
      continue
    fi
    rm -f "$temporary"
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 \
    nice -n 3 .venv/bin/python train-ablation-1781126582/play_selfplay_games.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --games "$games_per_shard" \
      --seed0 "$shard_seed" --device cpu --log-legal-actions \
      --out "$temporary" > "$log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  (( failed == 0 ))
  for ((shard = 0; shard < OPPORTUNITY_SHARDS; shard++)); do
    local output="$output_dir/shard${shard}.jsonl"
    local temporary="${output}.tmp"
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$output"
    fi
    [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $games_per_shard ]]
  done
  PYTHONPATH=python/src:train-ablation-1781126582 \
  .venv/bin/python train-ablation-1781126582/analyze_opportunity_rates.py \
    "$output_dir"/shard*.jsonl --label "$arm" \
    --early-tempo-bonus 0.1 --early-tempo-cap 4 \
    --portal-gp-bonus 0.3 --shaping-scale 0.15 \
    --early-tempo-dedup-portal-abilities \
    --json "$arm_results/opportunity.json" \
    | tee "$arm_results/opportunity.log"
  jq -e ".n_games == $OPPORTUNITY_GAMES" "$arm_results/opportunity.json" >/dev/null
}

write_control_strength_gate() {
  local input=$1 output=$2
  .venv/bin/python - "$input" "$output" <<'PY'
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
blocks = defaultdict(list)
for game in payload["games"]:
    blocks[str(game["block_id"])].append(float(game["candidate_score"]))
scores = np.asarray([np.mean(v) for _, v in sorted(blocks.items())])
rng = np.random.default_rng(42_905_001)
sampled = scores[rng.integers(0, len(scores), size=(50_000, len(scores)))].mean(axis=1)
score = float(payload["summary"]["score"])
result = {
    "passed": bool(score >= 0.40 and np.quantile(sampled, 0.90) >= 0.45),
    "score": score,
    "lcb80": float(np.quantile(sampled, 0.10)),
    "ucb80": float(np.quantile(sampled, 0.90)),
    "episodes": int(payload["summary"]["episodes"]),
    "timeout_rate": float(payload["summary"]["timeout_rate"]),
}
result["passed"] = result["passed"] and result["timeout_rate"] == 0.0
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["passed"] else 2)
PY
}

unset AZK_RESUME_COMPLETED_EPISODES AZK_RESUME_ALLOW_SCHEDULE_REWIND
unset AZK_DRAFT_REF_DECK_INDICES AZK_DRAFT_REF_OPPONENT_ONLY AZK_DRAFT_REF_SEAT_PROB
unset AZK_LEADER_TERMINAL_CREDIT_COEF AZK_LEADER_TERMINAL_CREDIT_GATE_CODES
unset AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
if [[ ! -f "$RESULT_ROOT/runtime_sha256.txt" ]]; then
  sha256sum \
    "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_LEAGUE" "$PARENT_PROMOTION" \
    python/src/azk_puffer/trainer.py python/src/league_training.py \
    python/src/train.py python/src/policy/v2/tcg_policy.py \
    python/src/policy/v2/tcg_sampler.py python/src/tcg.h \
    python/config/azuki_deckbuild_native_3090.ini build/python/src/binding*.so \
    train-ablation-1781126582/run_leader_credit_ladder15_v1.sh \
    train-ablation-1781126582/leader_credit_ladder_report.py \
    > "$RESULT_ROOT/runtime_sha256.txt"
  git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
  printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\ntotal_timesteps=%s\npeak_lr=%s\ncredit_coef=0.05\ncredit_interval=4\n' \
    "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$TOTAL_TIMESTEPS" "$PEAK_LR" \
    > "$RESULT_ROOT/campaign_config.txt"
fi

for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  credit_coef=${credit_coefs[$index]}
  tag="${CAMPAIGN}_${arm}"
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/$tag"
  train_log="$arm_results/train.live.log"
  mkdir -p "$arm_results"

  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] $arm training already complete"
  else
    if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null; then
      echo "[$CAMPAIGN] existing uncompleted run for $tag; refusing ambiguous resume" >&2
      exit 1
    fi
    if [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
      echo "[$CAMPAIGN] existing uncompleted state for $arm; refusing overwrite" >&2
      exit 1
    fi
    mkdir -p "$arm_league/opponents" "$snapshot_dir"
    cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
    cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"
    touch "$train_log"
    echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm coef=$credit_coef"
    env \
    AZK_REWARD_LEADER_DELTA_WEIGHT=1.25 \
    AZK_REWARD_BOARD_DELTA_WEIGHT=0.35 \
    AZK_REWARD_NOOP_PENALTY=0.02 \
    AZK_TRUNCATION_BOARD_EDGE_WEIGHT=0.45 \
    AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0.15 \
    AZK_REWARD_SHAPING_ANNEAL=1 \
    AZK_REWARD_SHAPING_ANNEAL_INITIAL=1.0 \
    AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15 \
    AZK_REWARD_SHAPING_ANNEAL_WARMUP_EPISODES=12 \
    AZK_REWARD_SHAPING_ANNEAL_RAMP_EPISODES=40 \
    AZK_PORTAL_GP_BONUS=0.3 \
    AZK_PORTAL_OUTCOME_BONUS=0 \
    AZK_XGATE_MASK=1 \
    AZK_PFSP=1 \
    AZK_EARLY_TEMPO_BONUS=0.1 \
    AZK_EARLY_TEMPO_CAP=4 \
    AZK_EARLY_TEMPO_TURNS=2 \
    AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES=1 \
    AZK_DMG_MITIGATION_BONUS=0.15 \
    AZK_DMG_MITIGATION_CAP=10 \
    AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0 \
    AZK_ENTITY_DAMAGE_EXCHANGE_STEP_CAP=6 \
    AZK_GENERATED_IKZ_CONVERSION_BONUS=0 \
    AZK_GENERATED_IKZ_CONVERSION_STEP_CAP=4 \
    AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
    AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
    AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4 \
    AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0 \
    AZK_DRAFT_REF_SEAT_PROB=0 \
    AZK_LEADER_TERMINAL_CREDIT_COEF="$credit_coef" \
    AZK_LEADER_TERMINAL_CREDIT_CLIP=0.2 \
    AZK_LEADER_TERMINAL_CREDIT_GATE_CODES=AZK01-122,STT04-002 \
    AZK_LEADER_TERMINAL_CREDIT_UPDATE_INTERVAL=4 \
    AZK_LEADER_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS=10 \
    AZK_RESUME_ALLOW_BINDING_MISMATCH=1 \
    AZK_RESUME_ALLOW_SOURCE_DRIFT=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python python/src/train.py \
      --config python/config/azuki_deckbuild_native_3090.ini \
      --resume-checkpoint "$PARENT_MODEL" \
      --resume-load-optimizer --resume-restart-lr-schedule \
      --no-resume-auto-reset-critic \
      --jsonl-log experiments/runlogs --tag "$tag" \
      --train.seed 42 --train.learning_rate "$PEAK_LR" \
      --train.ent_coef 0.002 \
      --train.ent_coef_anneal_initial 0.002 \
      --train.ent_coef_anneal_final 0.002 \
      --train.total_timesteps "$TOTAL_TIMESTEPS" \
      --train.checkpoint_interval 100 \
      --env.deck_snapshot_every 25 \
      --env.deck_snapshot_dir "$snapshot_dir" \
      --league.state_path "$arm_league/league_state.json" \
      --league.opponent_dir "$arm_league/opponents" \
      --league.eval_interval 100000 \
      --league.quick_eval_interval 100000 \
      --league.full_eval_interval 100000 \
      --league.promotion_shadow_mode true \
      --league.promotion_archive_affects_training_pool false \
      --league.promotion_panel_refresh_epochs 100000 \
      --league.production_anchor_checkpoint \
        experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt \
      > >(tee "$train_log") 2>&1 &
    train_pid=$!

    startup_checked=0
    startup_failed=0
    guard_failed=0
    label_checked=0
    last_guard_count=0
    while kill -0 "$train_pid" 2>/dev/null; do
      if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$train_log"; then
        if [[ "$arm" == leader_credit ]] &&
           ! grep -q '\[leader-credit\] enabled:' "$train_log"; then
          sleep 1
          continue
        fi
        if ! grep -q "total_epochs=$TARGET_EPOCH" "$train_log" ||
           ! grep -q "AZK_RESUME_COMPLETED_EPISODES=$EXPECTED_PARENT_EPISODES" "$train_log" ||
           ! grep -q 'epoch=3900' "$train_log" ||
           ! grep -q 'remaining_epochs=970' "$train_log"; then
          echo "[$CAMPAIGN] $arm startup invariant failure" \
            | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          startup_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
        if [[ "$arm" == leader_credit ]] &&
           ! grep -q 'update_interval=4.*gates=AZK01-122,STT04-002' "$train_log"; then
          echo "[$CAMPAIGN] candidate leader-credit config was not activated" \
            | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          startup_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
        startup_checked=1
        touch "$arm_results/STARTUP_INVARIANTS_OK"
        echo "[$CAMPAIGN] $arm startup invariants passed"
      fi

      jsonl=$(latest_jsonl "$tag")
      if [[ -n "$jsonl" ]]; then
        guard=$(.venv/bin/python - "$jsonl" <<'PY'
import json
import statistics
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("SPS"), (int, float)):
            rows.append(row)
values = [float(row["SPS"]) for row in rows]
prev = statistics.median(values[-40:-20]) if len(values) >= 40 else 0.0
curr = statistics.median(values[-20:]) if len(values) >= 40 else 0.0
labeled = max((float(row.get("losses/win_prob_aux_labeled_rows", 0)) for row in rows), default=0.0)
examples = sum(float(row.get("losses/leader_credit_examples", 0)) for row in rows)
captured = sum(float(row.get("environment/leader_credit/captured", 0)) for row in rows)
print(len(rows), prev, curr, labeled, examples, captured)
PY
        )
        read -r guard_count previous_median current_median labeled examples captured <<< "$guard"
        if (( ! label_checked && guard_count >= 20 )); then
          if ! awk -v labels="$labeled" 'BEGIN { exit !(labels > 0) }'; then
            echo "[$CAMPAIGN] $arm terminal labels stayed empty for 20 updates" \
              | tee "$arm_results/LABEL_GUARD_FAILED"
            guard_failed=1
            kill -INT "$train_pid" 2>/dev/null || true
            break
          fi
          if [[ "$arm" == leader_credit ]] &&
             ! awk -v examples="$examples" -v captured="$captured" \
               'BEGIN { exit !((examples > 0) && (captured > 0)) }'; then
            echo "[$CAMPAIGN] candidate delayed-credit records stayed empty for 20 updates" \
              | tee "$arm_results/CREDIT_GUARD_FAILED"
            guard_failed=1
            kill -INT "$train_pid" 2>/dev/null || true
            break
          fi
          label_checked=1
          touch "$arm_results/LABEL_SIGNAL_OK"
          echo "[$CAMPAIGN] $arm terminal signal guard passed"
        fi
        if (( guard_count >= 40 && guard_count > last_guard_count )); then
          last_guard_count=$guard_count
          if awk -v a="$previous_median" -v b="$current_median" -v t="$SPS_HARD_FLOOR" \
            'BEGIN { exit !((a < t) && (b < t)) }'; then
            printf 'previous_median=%s\ncurrent_median=%s\nthreshold=%s\n' \
              "$previous_median" "$current_median" "$SPS_HARD_FLOOR" \
              | tee "$arm_results/SPS_GUARD_FAILED"
            guard_failed=1
            kill -INT "$train_pid" 2>/dev/null || true
            break
          fi
        fi
      fi
      sleep 5
    done

    set +e
    wait "$train_pid"
    train_status=$?
    set -e
    cleanup_training_processes "$tag"
    if (( guard_failed )); then
      exit 2
    fi
    if (( startup_failed || ! startup_checked || ! label_checked )); then
      exit 1
    fi
    if (( train_status != 0 )); then
      echo "[$CAMPAIGN] $arm training exited with status $train_status" >&2
      exit "$train_status"
    fi

    run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" -print -quit)
    checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
    trainer_state="$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
    metadata="$checkpoint.meta.json"
    jsonl=$(latest_jsonl "$tag")
    if [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]] || [[ -z "$jsonl" ]]; then
      echo "[$CAMPAIGN] $arm lacks its final atomic checkpoint or metrics" >&2
      exit 1
    fi
    jq -e ".update == $TARGET_EPOCH and .env_completed_episodes >= $EXPECTED_PARENT_EPISODES" \
      "$metadata" >/dev/null
    cp -a "$jsonl" "$arm_results/train.jsonl"
    cp -a "$train_log" "$arm_results/train.log"
    printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
    sha256sum "$checkpoint" "$trainer_state" > "$arm_results/checkpoint_sha256.txt"
    sps_summary "$arm_results/train.jsonl" > "$arm_results/sps_summary.txt"
    signal_summary "$arm_results/train.jsonl" "$arm_results/terminal_credit_summary.json"
    touch "$arm_results/TRAIN_DONE"
  fi

  run_dir=$(<"$arm_results/run_dir.txt")
  checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
  run_h2h "$checkpoint" "$PARENT_MODEL" "$arm" p3900_parent \
    "$arm_results/h2h_vs_parent.json"
  if [[ "$arm" == control ]]; then
    if ! write_control_strength_gate \
      "$arm_results/h2h_vs_parent.json" "$arm_results/control_strength_gate.json"; then
      touch "$RESULT_ROOT/CONTROL_STRENGTH_FAILED"
      echo "[$CAMPAIGN] control failed strength gate; candidate will not train" >&2
      exit 3
    fi
  fi
done

control_run=$(<"$RESULT_ROOT/control/run_dir.txt")
candidate_run=$(<"$RESULT_ROOT/leader_credit/run_dir.txt")
control_checkpoint="$control_run/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
candidate_checkpoint="$candidate_run/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
run_h2h "$candidate_checkpoint" "$control_checkpoint" leader_credit control \
  "$RESULT_ROOT/leader_credit/h2h_vs_control.json"

for arm in "${arms[@]}"; do
  arm_results="$RESULT_ROOT/$arm"
  run_dir=$(<"$arm_results/run_dir.txt")
  checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
  run_reference_eval "$checkpoint" "$arm" train "$TRAIN_INDICES" \
    "$arm_results/draftref_train.json"
  run_reference_eval "$checkpoint" "$arm" holdout "$HOLDOUT_INDICES" \
    "$arm_results/draftref_holdout.json"
  if [[ ! -f "$arm_results/decks.json" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python train-ablation-1781126582/dump_gate_decks.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --episodes 48 --device cuda \
      --json "$arm_results/decks.json" \
      > >(tee "$arm_results/decks.log") 2>&1
  fi
  if [[ ! -f "$arm_results/gate_kl.json" ]]; then
    PYTHONPATH=build/python/src:python/src \
    .venv/bin/python train-ablation-1781126582/probe_gate_kl.py \
      --config python/config/azuki_deckbuild_3090.ini \
      --checkpoint "$checkpoint" --episodes 48 --device cuda \
      --json "$arm_results/gate_kl.json" \
      > >(tee "$arm_results/gate_kl.log") 2>&1
  fi
  run_opportunity_eval "$checkpoint" "$arm" "$arm_results"
  touch "$arm_results/ARM_DONE"
done

read -r _ control_sps _ < "$RESULT_ROOT/control/sps_summary.txt"
read -r _ candidate_sps _ < "$RESULT_ROOT/leader_credit/sps_summary.txt"
ratio=$(awk -v a="$candidate_sps" -v c="$control_sps" 'BEGIN { printf "%.6f", a / c }')
printf 'candidate_median=%s\ncontrol_median=%s\nratio=%s\nthreshold=%s\n' \
  "$candidate_sps" "$control_sps" "$ratio" "$SPS_RELATIVE_FLOOR" \
  > "$RESULT_ROOT/leader_credit/sps_relative_to_control.txt"
if awk -v ratio="$ratio" -v floor="$SPS_RELATIVE_FLOOR" \
  'BEGIN { exit !(ratio < floor) }'; then
  touch "$RESULT_ROOT/leader_credit/SPS_RELATIVE_GUARD_FAILED"
fi

PYTHONPATH=python/src:train-ablation-1781126582 \
.venv/bin/python train-ablation-1781126582/leader_credit_ladder_report.py \
  "$RESULT_ROOT" --json "$RESULT_ROOT/ladder_report.json" \
  --markdown "$RESULT_ROOT/ladder_report.md"
touch "$RESULT_ROOT/LADDER_DONE"
echo "[$CAMPAIGN] LADDER_DONE"
