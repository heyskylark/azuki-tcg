#!/usr/bin/env bash
# Resume the strategy-recovery control from its atomic p3300 state after the
# original 1300-SPS guard fired on a <1% dip. The scheduler is restored as-is;
# it must not restart. On success, merge the non-overlapping metric ranges and
# return to the main ladder for evaluation and candidate arms.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
cd "$ROOT"

CAMPAIGN=strategy_recovery_ladder15_v1
RESULT_ROOT="train-ablation-1781126582/results/$CAMPAIGN"
ARM_RESULTS="$RESULT_ROOT/control"
LEAGUE_ROOT="experiments/league/$CAMPAIGN/control"
SOURCE_RUN=experiments/azuki_local_strategy_recovery_ladder15_v1_control_178444379051
SOURCE_MODEL="$SOURCE_RUN/model_azuki_local_003300.pt"
SOURCE_TRAINER="$SOURCE_RUN/trainer_state_003300.pt"
SOURCE_METADATA="$SOURCE_MODEL.meta.json"
SOURCE_JSONL=experiments/runlogs/strategy_recovery_ladder15_v1_control_178444379051.jsonl
TAG=strategy_recovery_ladder15_v1_control_resume3300
SNAPSHOT_DIR=experiments/abl_snapshots/strategy_recovery_ladder15_v1_control
TRAIN_LOG="$ARM_RESULTS/train.resume3300.live.log"
TARGET_EPOCH=3900
TOTAL_TIMESTEPS=59904000
SPS_HARD_FLOOR=1235
EXPECTED_MODEL_SHA=e7ca84c12abbfd1bd2a1037193b918a6d767a79ba18d198e6b4c20825e0ccaa7
EXPECTED_TRAINER_SHA=90bc84be0fd539f6c68e892ce852927ff16a805fa266fa7159f414976b184317
EXPECTED_LEAGUE_SHA=76341df7aaafc5982865dcd764a649b5eb2d942efc6a5370e3ab0a1c1fee9938
EXPECTED_PROMOTION_SHA=9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a
EXPECTED_JSONL_SHA=55fdf53f02bda90f663465689fc553c94903b5355007ca303e7ec81ad5103a36

require_hash() {
  local path=$1 expected=$2 label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[$TAG] $label hash mismatch: $actual" >&2
    exit 1
  fi
}

latest_jsonl() {
  find experiments/runlogs -maxdepth 1 -type f -name "${TAG}_*.jsonl" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

require_hash "$SOURCE_MODEL" "$EXPECTED_MODEL_SHA" source-model
require_hash "$SOURCE_TRAINER" "$EXPECTED_TRAINER_SHA" source-trainer
require_hash "$LEAGUE_ROOT/league_state.json" "$EXPECTED_LEAGUE_SHA" source-league
require_hash "$LEAGUE_ROOT/league_state_promotion.json" "$EXPECTED_PROMOTION_SHA" source-promotion
require_hash "$SOURCE_JSONL" "$EXPECTED_JSONL_SHA" source-jsonl
jq -e '.update == 3300 and .global_step == 31056911 and .env_completed_episodes == 273' \
  "$SOURCE_METADATA" >/dev/null
jq -e --arg path "$ROOT/$SOURCE_MODEL" \
  '.history[-1].event == "checkpoint_ingested" and
   .history[-1].epoch == 3300 and .history[-1].checkpoint_path == $path' \
  "$LEAGUE_ROOT/league_state.json" >/dev/null

if [[ -f "$ARM_RESULTS/TRAIN_DONE" ]]; then
  echo "[$TAG] control training is already complete"
  exec bash train-ablation-1781126582/run_strategy_recovery_ladder15_v1.sh
fi
if compgen -G "experiments/azuki_local_${TAG}_*" >/dev/null; then
  echo "[$TAG] existing continuation output; refusing ambiguous resume" >&2
  exit 1
fi
if [[ -f "$ARM_RESULTS/SPS_GUARD_FAILED" ]]; then
  mv "$ARM_RESULTS/SPS_GUARD_FAILED" \
    "$ARM_RESULTS/SPS_GUARD_FALSE_POSITIVE_P3326"
fi

sha256sum \
  "$SOURCE_MODEL" "$SOURCE_TRAINER" \
  "$LEAGUE_ROOT/league_state.json" "$LEAGUE_ROOT/league_state_promotion.json" \
  "$SOURCE_JSONL" train-ablation-1781126582/resume_strategy_recovery_control_p3300.sh \
  > "$ARM_RESULTS/resume_p3300_input_sha256.txt"
printf '%s\n' \
  'The original guard used 1300 SPS as a hard floor and stopped on windows' \
  '1286.8/1298.0 despite a 1431 run-wide median. This resume uses 1235,' \
  'which is 5% below the observed 1300 lower bound.' \
  > "$ARM_RESULTS/SPS_GUARD_FALSE_POSITIVE_NOTE.txt"
touch "$TRAIN_LOG"

echo "[$TAG] $(date --iso-8601=seconds) resume p3300 without LR restart"
env \
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
AZK_EARLY_TEMPO_DEDUP_PORTAL_ABILITIES=0 \
AZK_DMG_MITIGATION_BONUS=0.15 \
AZK_DMG_MITIGATION_CAP=10 \
AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0 \
AZK_GENERATED_IKZ_CONVERSION_BONUS=0 \
AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08 \
AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025 \
AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4 \
AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0 \
AZK_DRAFT_REF_SEAT_PROB=0 \
AZK_RESUME_ALLOW_BINDING_MISMATCH=1 \
AZK_RESUME_ALLOW_SOURCE_DRIFT=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
PYTHONPATH=build/python/src:python/src \
.venv/bin/python python/src/train.py \
  --config python/config/azuki_deckbuild_native_3090.ini \
  --resume-checkpoint "$SOURCE_MODEL" \
  --resume-load-optimizer \
  --no-resume-auto-reset-critic \
  --jsonl-log experiments/runlogs \
  --tag "$TAG" \
  --train.seed 42 \
  --train.learning_rate 0.0001 \
  --train.ent_coef 0.002 \
  --train.ent_coef_anneal_initial 0.002 \
  --train.ent_coef_anneal_final 0.002 \
  --train.total_timesteps "$TOTAL_TIMESTEPS" \
  --train.checkpoint_interval 100 \
  --env.deck_snapshot_every 25 \
  --env.deck_snapshot_dir "$SNAPSHOT_DIR" \
  --league.state_path "$LEAGUE_ROOT/league_state.json" \
  --league.opponent_dir "$LEAGUE_ROOT/opponents" \
  --league.eval_interval 100000 \
  --league.quick_eval_interval 100000 \
  --league.full_eval_interval 100000 \
  --league.promotion_shadow_mode true \
  --league.promotion_archive_affects_training_pool false \
  --league.promotion_panel_refresh_epochs 100000 \
  --league.production_anchor_checkpoint \
    experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt \
  > >(tee "$TRAIN_LOG") 2>&1 &
train_pid=$!

startup_checked=0
startup_failed=0
lr_checked=0
guard_failed=0
last_guard_count=0
while kill -0 "$train_pid" 2>/dev/null; do
  if (( ! startup_checked )) && grep -q '\[train\] epoch plan:' "$TRAIN_LOG"; then
    if ! grep -q "total_epochs=$TARGET_EPOCH" "$TRAIN_LOG" ||
       ! grep -q 'AZK_RESUME_COMPLETED_EPISODES=273' "$TRAIN_LOG" ||
       ! grep -q 'epoch=3300' "$TRAIN_LOG" ||
       grep -q 'restarted LR schedule' "$TRAIN_LOG"; then
      echo "[$TAG] startup invariant failure" \
        | tee "$ARM_RESULTS/RESUME3300_STARTUP_FAILED"
      startup_failed=1
      kill -INT "$train_pid" 2>/dev/null || true
      break
    fi
    startup_checked=1
    touch "$ARM_RESULTS/RESUME3300_STARTUP_OK"
    echo "[$TAG] startup invariants passed"
  fi

  jsonl=$(latest_jsonl)
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
if len(rows) < 40:
    print(len(rows), 0.0, 0.0, rows[0]["learning_rate"] if rows else 0.0)
else:
    print(
        len(rows),
        statistics.median(float(row["SPS"]) for row in rows[-40:-20]),
        statistics.median(float(row["SPS"]) for row in rows[-20:]),
        rows[0]["learning_rate"],
    )
PY
    )
    read -r guard_count previous_median current_median first_lr <<< "$guard"
    if (( ! lr_checked && guard_count >= 1 )); then
      if ! awk -v lr="$first_lr" 'BEGIN { exit !((lr > 0.00005) && (lr < 0.00008)) }'; then
        echo "[$TAG] LR continuity failure: first_lr=$first_lr" \
          | tee "$ARM_RESULTS/RESUME3300_LR_FAILED"
        startup_failed=1
        kill -INT "$train_pid" 2>/dev/null || true
        break
      fi
      lr_checked=1
      printf 'first_lr=%s\n' "$first_lr" > "$ARM_RESULTS/resume3300_lr_check.txt"
      echo "[$TAG] LR continuity passed: first_lr=$first_lr"
    fi
    if (( guard_count >= 40 && guard_count > last_guard_count )); then
      last_guard_count=$guard_count
      if awk -v a="$previous_median" -v b="$current_median" -v t="$SPS_HARD_FLOOR" \
        'BEGIN { exit !((a < t) && (b < t)) }'; then
        printf 'previous_median=%s\ncurrent_median=%s\nthreshold=%s\n' \
          "$previous_median" "$current_median" "$SPS_HARD_FLOOR" \
          | tee "$ARM_RESULTS/RESUME3300_SPS_GUARD_FAILED"
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
if (( guard_failed )); then
  exit 2
fi
if (( startup_failed || ! startup_checked || ! lr_checked )); then
  exit 1
fi
if (( train_status != 0 )); then
  echo "[$TAG] training exited with status $train_status" >&2
  exit "$train_status"
fi

run_dir=$(find experiments -maxdepth 1 -type d -name "azuki_local_${TAG}_*" -print -quit)
checkpoint="$run_dir/model_azuki_local_$(printf '%06d' "$TARGET_EPOCH").pt"
trainer_state="$run_dir/trainer_state_$(printf '%06d' "$TARGET_EPOCH").pt"
jsonl=$(latest_jsonl)
if [[ ! -f "$checkpoint" ]] || [[ ! -f "$trainer_state" ]] || [[ -z "$jsonl" ]]; then
  echo "[$TAG] missing final atomic checkpoint or JSONL" >&2
  exit 1
fi

cp -a "$SOURCE_JSONL" "$ARM_RESULTS/train.part1_p2931_p3300_source.jsonl"
cp -a "$jsonl" "$ARM_RESULTS/train.part2_p3301_p3900_source.jsonl"
.venv/bin/python - \
  "$SOURCE_JSONL" "$jsonl" "$ARM_RESULTS/train.jsonl" <<'PY'
import json
from pathlib import Path
import sys

rows = {}
for path, lower, upper in (
    (Path(sys.argv[1]), 2931, 3300),
    (Path(sys.argv[2]), 3301, 3900),
):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = row.get("epoch")
            if isinstance(epoch, (int, float)) and lower <= int(epoch) <= upper:
                rows[int(epoch)] = row
expected = set(range(2931, 3901))
if rows.keys() != expected:
    raise RuntimeError(
        f"Merged metric epochs differ: missing={sorted(expected - rows.keys())[:5]} "
        f"extra={sorted(rows.keys() - expected)[:5]}"
    )
output = Path(sys.argv[3])
with output.open("w", encoding="utf-8") as handle:
    for epoch in sorted(rows):
        handle.write(json.dumps(rows[epoch], separators=(",", ":")) + "\n")
PY
.venv/bin/python - "$ARM_RESULTS/train.jsonl" > "$ARM_RESULTS/sps_summary.txt" <<'PY'
import json
import statistics
import sys

values = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if isinstance(row.get("SPS"), (int, float)):
            values.append(float(row["SPS"]))
steady = values[20:]
print(len(values), f"{statistics.median(steady):.6f}", f"{statistics.median(steady[-100:]):.6f}")
PY
.venv/bin/python - "$ARM_RESULTS/train.live.log" "$TRAIN_LOG" "$ARM_RESULTS/train.log" <<'PY'
from pathlib import Path
import sys

part1 = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
part2 = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
Path(sys.argv[3]).write_text(part1 + "\n\n===== RESUME P3300 =====\n\n" + part2, encoding="utf-8")
PY
printf '%s\n' "$run_dir" > "$ARM_RESULTS/run_dir.txt"
sha256sum "$checkpoint" "$trainer_state" > "$ARM_RESULTS/checkpoint_sha256.txt"
touch "$ARM_RESULTS/TRAIN_DONE"
echo "[$TAG] control training complete; returning to ladder"
exec bash train-ablation-1781126582/run_strategy_recovery_ladder15_v1.sh
