#!/usr/bin/env bash
# Matched p4870->p5847 continuation for Step 5b whole-draft terminal credit.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=${DRAFT_CREDIT_LADDER_CAMPAIGN:-draft_terminal_credit_ladder15_v1}
SMOKE_ROOT=${DRAFT_CREDIT_SMOKE_ROOT:-train-ablation-1781126582/results/draft_terminal_credit_smoke_v8}
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
LEAGUE_ROOT="experiments/league/$CAMPAIGN"
PARENT_DIR=experiments/azuki_local_strategy_recovery_leader15_v1_control_178447770084
PARENT_MODEL="$PARENT_DIR/model_azuki_local_004870.pt"
PARENT_TRAINER="$PARENT_DIR/trainer_state_004870.pt"
PARENT_METADATA="$PARENT_MODEL.meta.json"
PARENT_LEAGUE=experiments/league/strategy_recovery_leader15_v1/control/league_state.json
PARENT_PROMOTION=experiments/league/strategy_recovery_leader15_v1/control/league_state_promotion.json
EXPECTED_MODEL_SHA=def350888ecbf9014e590fb8540080cb26b6c14852882caa85def3f6db8a4f76
EXPECTED_TRAINER_SHA=ce1aa27a1ab58e4c9060f6de99b86cdfc6e8fe815803d9b3ba04155848aa3486
EXPECTED_LEAGUE_SHA=22b247ba70c6d6a72f92ab4af1e805e874c2aa5337a4bba5ea165145f3e236de
EXPECTED_PROMOTION_SHA=9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a
EXPECTED_PARENT_EPISODES=404
PARENT_EPOCH=4870
TARGET_EPOCH=5847
TOTAL_TIMESTEPS=89809920
PEAK_LR=${DRAFT_CREDIT_LADDER_PEAK_LR:-0.00003}
SPS_HARD_FLOOR=1235
SPS_RELATIVE_FLOOR=0.95

arms=(control draft_terminal_long)

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

latest_run_dir() {
  local tag=$1
  find experiments -maxdepth 1 -type d -name "azuki_local_${tag}_*" \
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
scores = np.asarray([np.mean(values) for _, values in sorted(blocks.items())])
rng = np.random.default_rng(42_906_001)
sampled = scores[rng.integers(0, len(scores), size=(50_000, len(scores)))].mean(axis=1)
score = float(payload["summary"]["score"])
result = {
    "passed": bool(
        score >= 0.40
        and np.quantile(sampled, 0.90) >= 0.45
        and float(payload["summary"]["timeout_rate"]) == 0.0
    ),
    "score": score,
    "lcb80": float(np.quantile(sampled, 0.10)),
    "ucb80": float(np.quantile(sampled, 0.90)),
    "episodes": int(payload["summary"]["episodes"]),
    "timeout_rate": float(payload["summary"]["timeout_rate"]),
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["passed"] else 2)
PY
}

write_training_summary() {
  local arm=$1 jsonl=$2 output=$3
  .venv/bin/python - "$arm" "$jsonl" "$output" <<'PY'
import json
import statistics
import sys
from pathlib import Path

arm = sys.argv[1]
rows = {}
for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    epoch = row.get("epoch")
    if isinstance(epoch, (int, float)) and 4871 <= int(epoch) <= 5847:
        rows[int(epoch)] = row
expected = set(range(4871, 5848))
if set(rows) != expected:
    raise RuntimeError(f"{arm} epoch mismatch: missing={sorted(expected - set(rows))[:20]}")
ordered = [rows[epoch] for epoch in sorted(rows)]

def values(key, source=ordered):
    return [
        float(row[key]) for row in source
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]

steady = ordered[20:]
sps = values("SPS", steady)
position_metrics = {}
for position in ("leader", "early", "middle", "late"):
    count_key = f"losses/draft_credit_{position}_examples"
    counts = values(count_key)
    position_metrics[position] = {"examples": sum(counts)}
    for suffix in (
        "advantage_mean",
        "advantage_abs_mean",
        "advantage_std",
        "sign_outcome_agreement",
        "baseline_bce",
        "baseline_brier",
        "baseline_pred_mean",
    ):
        key = f"losses/draft_credit_{position}_{suffix}"
        weighted = [
            (float(row[key]), float(row[count_key]))
            for row in ordered
            if isinstance(row.get(key), (int, float))
            and isinstance(row.get(count_key), (int, float))
            and float(row[count_key]) > 0.0
        ]
        position_metrics[position][suffix] = (
            sum(value * count for value, count in weighted) / sum(count for _, count in weighted)
            if weighted else None
        )

trained_rows = [
    row for row in ordered if float(row.get("losses/draft_credit_examples", 0.0)) > 0.0
]
summary = {
    "arm": arm,
    "metric_rows": len(ordered),
    "epoch_first": int(ordered[0]["epoch"]),
    "epoch_final": int(ordered[-1]["epoch"]),
    "learning_rate_final": float(ordered[-1]["learning_rate"]),
    "steady_sps_median": statistics.median(sps),
    "steady_sps_p10": sorted(sps)[max(0, int(0.1 * (len(sps) - 1)))],
    "last100_sps_median": statistics.median(sps[-100:]),
    "terminal_labels_max": max(values("losses/win_prob_aux_labeled_rows"), default=0.0),
    "captured_records": sum(values("environment/draft_credit/captured")),
    "labeled_records": sum(values("environment/draft_credit/labeled")),
    "truncated_records": sum(values("environment/draft_credit/truncated")),
    "incomplete_episodes": sum(values("environment/draft_credit/incomplete_episodes")),
    "incomplete_records": sum(values("environment/draft_credit/incomplete_records")),
    "pending_records_final": float(ordered[-1].get("environment/draft_credit/pending_records", 0.0)),
    "ready_records_final": float(ordered[-1].get("environment/draft_credit/ready", 0.0)),
    "trained_examples": sum(values("losses/draft_credit_examples")),
    "training_epochs": len(trained_rows),
    "fixed_batch_rows": sum(values("losses/draft_credit_fixed_batch_rows")),
    "aux_wall_seconds": sum(values("losses/draft_credit_train_seconds")),
    "aux_gpu_seconds": sum(values("losses/draft_credit_gpu_seconds")),
    "importance_mean_median": (
        statistics.median(values("losses/draft_credit_importance_mean", trained_rows))
        if trained_rows else None
    ),
    "clipfrac_max": max(values("losses/draft_credit_clipfrac", trained_rows), default=None),
    "entropy_median": (
        statistics.median(values("losses/draft_credit_entropy", trained_rows))
        if trained_rows else None
    ),
    "gradient_norm_median": (
        statistics.median(values("losses/draft_credit_gradient_norm", trained_rows))
        if trained_rows else None
    ),
    "raw_gradient_norm_median": (
        statistics.median(values("losses/draft_credit_raw_gradient_norm", trained_rows))
        if trained_rows else None
    ),
    "trainer_shaped_reward_multiplier_min": min(
        values("environment/trainer_shaped_reward_multiplier"), default=1.0
    ),
    "trainer_shaped_reward_multiplier_max": max(
        values("environment/trainer_shaped_reward_multiplier"), default=1.0
    ),
    "native_reward_shaping_scale_final": float(
        ordered[-1]["environment/reward_shaping_scale"]
    ),
    "timeout_rate_max": max(values("environment/timeout_truncation_rate"), default=0.0),
    "positions": position_metrics,
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
}

require_hash "$PARENT_MODEL" "$EXPECTED_MODEL_SHA" parent-model
require_hash "$PARENT_TRAINER" "$EXPECTED_TRAINER_SHA" parent-trainer
require_hash "$PARENT_LEAGUE" "$EXPECTED_LEAGUE_SHA" parent-league
require_hash "$PARENT_PROMOTION" "$EXPECTED_PROMOTION_SHA" parent-promotion
jq -e ".update == $PARENT_EPOCH and .env_completed_episodes == $EXPECTED_PARENT_EPISODES" \
  "$PARENT_METADATA" >/dev/null
if [[ ! -f "$SMOKE_ROOT/SMOKE_DONE" ]]; then
  echo "[$CAMPAIGN] passing Step 5b smoke is required: $SMOKE_ROOT" >&2
  exit 1
fi
jq -e '.status == "pass" and .integrity_pass == true' "$SMOKE_ROOT/report.json" >/dev/null
CREDIT_COEF=$(jq -er '.calibrated_coefficient | select(. > 0 and . <= 4)' \
  "$SMOKE_ROOT/report.json")

mkdir -p "$RESULT_ROOT" "$LEAGUE_ROOT" experiments/runlogs
sha256sum \
  "$PARENT_MODEL" "$PARENT_TRAINER" "$PARENT_LEAGUE" "$PARENT_PROMOTION" \
  "$SMOKE_ROOT/calibration.json" "$SMOKE_ROOT/report.json" \
  python/src/azk_puffer/trainer.py python/src/league_training.py python/src/train.py \
  python/src/policy/v2/tcg_policy.py python/src/policy/v2/tcg_sampler.py \
  python/src/tcg.h python/config/azuki_deckbuild_native_3090.ini \
  build/python/src/binding*.so \
  train-ablation-1781126582/run_draft_terminal_credit_ladder15_v1.sh \
  > "$RESULT_ROOT/runtime_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/git_head.txt"
printf 'campaign=%s\nparent_epoch=%s\ntarget_epoch=%s\ntotal_timesteps=%s\npeak_lr=%s\ncredit_coef=%s\ncredit_interval=4\ncredit_batch_size=512\nwindows=4870,5200,5500,5847\n' \
  "$CAMPAIGN" "$PARENT_EPOCH" "$TARGET_EPOCH" "$TOTAL_TIMESTEPS" "$PEAK_LR" \
  "$CREDIT_COEF" > "$RESULT_ROOT/campaign_config.txt"

for arm in "${arms[@]}"; do
  coefficient=0
  if [[ "$arm" == draft_terminal_long ]]; then
    coefficient=$CREDIT_COEF
  fi
  coefficient_fmt=$(printf '%.6f' "$coefficient")
  tag="${CAMPAIGN}_${arm}"
  arm_results="$RESULT_ROOT/$arm"
  arm_league="$LEAGUE_ROOT/$arm"
  snapshot_dir="experiments/abl_snapshots/$tag"
  live_log="$arm_results/train.live.log"
  mkdir -p "$arm_results"

  if [[ -f "$arm_results/TRAIN_DONE" ]]; then
    echo "[$CAMPAIGN] $arm already trained"
  else
    if compgen -G "experiments/azuki_local_${tag}_*" >/dev/null ||
       [[ -e "$arm_league" ]] || [[ -e "$snapshot_dir" ]]; then
      echo "[$CAMPAIGN] existing incomplete state for $arm; refusing overwrite" >&2
      exit 1
    fi
    mkdir -p "$arm_league/opponents" "$snapshot_dir"
    cp -a "$PARENT_LEAGUE" "$arm_league/league_state.json"
    cp -a "$PARENT_PROMOTION" "$arm_league/league_state_promotion.json"

    echo "[$CAMPAIGN] $(date --iso-8601=seconds) start $arm coef=$coefficient"
    env \
      -u AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV \
      -u AZK_RESUME_KEEP_CURRENT_REWARD_ENV \
      -u AZK_RESUME_COMPLETED_EPISODES \
      -u AZK_RESUME_ALLOW_SCHEDULE_REWIND \
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
      AZK_TRAINER_SHAPED_REWARD_ANNEAL=0 \
      AZK_PORTAL_GP_BONUS=0.3 \
      AZK_PORTAL_OUTCOME_BONUS=0 \
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
      AZK_DRAFT_VBOOT_COEF=0 \
      AZK_DRAFT_SIBDIFF_COEF=0 \
      AZK_DRAFT_SIBDIFF_CAP=0.05 \
      AZK_DRAFT_AUX_ANNEAL=1 \
      AZK_XGATE_MASK=1 \
      AZK_PFSP=1 \
      AZK_DRAFT_REF_SEAT_PROB=0 \
      AZK_LEADER_TERMINAL_CREDIT_COEF=0 \
      AZK_DRAFT_TERMINAL_CREDIT_COEF="$coefficient" \
      AZK_DRAFT_TERMINAL_CREDIT_CLIP=0.2 \
      AZK_DRAFT_TERMINAL_CREDIT_UPDATE_INTERVAL=4 \
      AZK_DRAFT_TERMINAL_CREDIT_BATCH_SIZE=512 \
      AZK_DRAFT_TERMINAL_CREDIT_SEED=420051 \
      AZK_DRAFT_TERMINAL_CREDIT_LABEL_WARMUP_EPOCHS=10 \
      AZK_DRAFT_TERMINAL_CREDIT_GRAD_PROBE=0 \
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
        --env.deck_snapshot_every 100 \
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
        > >(tee "$live_log") 2>&1 &
    train_pid=$!

    startup_checked=0
    label_checked=0
    guard_failed=0
    last_guard_count=0
    while kill -0 "$train_pid" 2>/dev/null; do
      if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$live_log"; then
        if [[ "$arm" == draft_terminal_long ]] &&
           ! grep -q '\[draft-terminal-credit\] enabled:' "$live_log"; then
          sleep 1
          continue
        fi
        if ! grep -q "total_epochs=$TARGET_EPOCH" "$live_log" ||
           ! grep -q "AZK_RESUME_COMPLETED_EPISODES=$EXPECTED_PARENT_EPISODES" "$live_log" ||
           ! grep -q 'epoch=4870' "$live_log" ||
           ! grep -q 'remaining_epochs=977' "$live_log"; then
          echo "[$CAMPAIGN] $arm startup invariant failure" \
            | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          guard_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
        if [[ "$arm" == draft_terminal_long ]] &&
           ! grep -q "coef=${coefficient_fmt}.*update_interval=4.*batch_size=512" "$live_log"; then
          echo "[$CAMPAIGN] candidate terminal-credit config was not activated" \
            | tee "$arm_results/STARTUP_INVARIANTS_FAILED"
          guard_failed=1
          kill -INT "$train_pid" 2>/dev/null || true
          break
        fi
        startup_checked=1
        touch "$arm_results/STARTUP_INVARIANTS_OK"
      fi

      jsonl=$(latest_jsonl "$tag")
      if [[ -n "$jsonl" ]]; then
        guard=$(.venv/bin/python - "$jsonl" <<'PY'
import json
import statistics
import sys

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(row.get("SPS"), (int, float)):
        rows.append(row)
values = [float(row["SPS"]) for row in rows]
prev = statistics.median(values[-40:-20]) if len(values) >= 40 else 0.0
curr = statistics.median(values[-20:]) if len(values) >= 40 else 0.0
labels = max((float(row.get("losses/win_prob_aux_labeled_rows", 0)) for row in rows), default=0.0)
examples = sum(float(row.get("losses/draft_credit_examples", 0)) for row in rows)
captured = sum(float(row.get("environment/draft_credit/captured", 0)) for row in rows)
incomplete = sum(float(row.get("environment/draft_credit/incomplete_episodes", 0)) for row in rows)
print(len(rows), prev, curr, labels, examples, captured, incomplete)
PY
        )
        read -r count previous_median current_median labels examples captured incomplete <<< "$guard"
        if (( ! label_checked && count >= 20 )); then
          if ! awk -v labels="$labels" 'BEGIN { exit !(labels > 0) }'; then
            echo "[$CAMPAIGN] $arm terminal labels stayed empty for 20 updates" \
              | tee "$arm_results/LABEL_GUARD_FAILED"
            guard_failed=1
            kill -INT "$train_pid" 2>/dev/null || true
            break
          fi
          if [[ "$arm" == draft_terminal_long ]] &&
             ! awk -v e="$examples" -v c="$captured" -v i="$incomplete" \
               'BEGIN { exit !((e > 0) && (c > 0) && (i == 0)) }'; then
            echo "[$CAMPAIGN] candidate delayed-credit integrity guard failed" \
              | tee "$arm_results/CREDIT_GUARD_FAILED"
            guard_failed=1
            kill -INT "$train_pid" 2>/dev/null || true
            break
          fi
          label_checked=1
          touch "$arm_results/LABEL_SIGNAL_OK"
        fi
        if (( count >= 40 && count > last_guard_count )); then
          last_guard_count=$count
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
    if (( guard_failed || ! startup_checked || ! label_checked )); then
      exit 2
    fi
    if (( train_status != 0 )); then
      echo "[$CAMPAIGN] $arm training failed with status $train_status" >&2
      exit "$train_status"
    fi

    run_dir=$(latest_run_dir "$tag")
    jsonl=$(latest_jsonl "$tag")
    checkpoint="$run_dir/model_azuki_local_005847.pt"
    trainer_state="$run_dir/trainer_state_005847.pt"
    if [[ -z "$run_dir" ]] || [[ -z "$jsonl" ]] ||
       [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]]; then
      echo "[$CAMPAIGN] $arm lacks atomic p5847 output" >&2
      exit 1
    fi
    cp -a "$jsonl" "$arm_results/train.jsonl"
    cp -a "$live_log" "$arm_results/train.log"
    printf '%s\n' "$run_dir" > "$arm_results/run_dir.txt"
    sha256sum "$checkpoint" "$trainer_state" > "$arm_results/checkpoint_sha256.txt"
    write_training_summary "$arm" "$jsonl" "$arm_results/training_summary.json"
    touch "$arm_results/TRAIN_DONE"
  fi

  run_dir=$(<"$arm_results/run_dir.txt")
  checkpoint="$run_dir/model_azuki_local_005847.pt"
  run_h2h "$checkpoint" "$PARENT_MODEL" "$arm" p4870_parent \
    "$arm_results/h2h_vs_parent_final.json"
  if [[ "$arm" == control ]]; then
    if ! write_control_strength_gate \
      "$arm_results/h2h_vs_parent_final.json" "$arm_results/control_strength_gate.json"; then
      touch "$RESULT_ROOT/CONTROL_STRENGTH_FAILED"
      echo "[$CAMPAIGN] control failed its parent-strength gate" >&2
      exit 3
    fi
  fi
done

control_run=$(<"$RESULT_ROOT/control/run_dir.txt")
candidate_run=$(<"$RESULT_ROOT/draft_terminal_long/run_dir.txt")
run_h2h \
  "$candidate_run/model_azuki_local_005847.pt" \
  "$control_run/model_azuki_local_005847.pt" \
  draft_terminal_long control \
  "$RESULT_ROOT/draft_terminal_long/h2h_vs_control_final.json"

control_sps=$(jq -r '.steady_sps_median' "$RESULT_ROOT/control/training_summary.json")
candidate_sps=$(jq -r '.steady_sps_median' "$RESULT_ROOT/draft_terminal_long/training_summary.json")
ratio=$(awk -v a="$candidate_sps" -v c="$control_sps" 'BEGIN { printf "%.9f", a / c }')
printf 'candidate_median=%s\ncontrol_median=%s\nratio=%s\nthreshold=%s\n' \
  "$candidate_sps" "$control_sps" "$ratio" "$SPS_RELATIVE_FLOOR" \
  > "$RESULT_ROOT/draft_terminal_long/sps_relative_to_control.txt"
if awk -v ratio="$ratio" -v floor="$SPS_RELATIVE_FLOOR" \
  'BEGIN { exit !(ratio < floor) }'; then
  touch "$RESULT_ROOT/draft_terminal_long/SPS_RELATIVE_GUARD_FAILED"
  echo "[$CAMPAIGN] candidate failed the relative SPS gate" >&2
  exit 4
fi

touch "$RESULT_ROOT/TRAINING_DONE"
echo "[$CAMPAIGN] TRAINING_DONE"
