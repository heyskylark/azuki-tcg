#!/usr/bin/env bash
# Train one registered specialist continuation (build_continuation_inputs.py) to its
# step budget, resuming from the latest checkpoint if interrupted, then evaluate the
# checkpoints nearest every +25M learner steps on that element's games of the s30m
# paired schedule (compare against runs/baseline_u8223_eval_s30m.json).
# Usage: run_continuation.sh <run-root>
set -euo pipefail

RT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RT"
export PYTHONPATH="$RT/build/python/src:$RT/python/src"
export LD_LIBRARY_PATH="$RT/build/_deps/flecs_src-build"
PY="$RT/.venv/bin/python"
root="$1"
plan="$root/experiment.json"
read -r element stage steps parent parent_step config < <("$PY" -c "
import json,sys; p=json.load(open(sys.argv[1]))
print(p['element'], p['continuation']['stage'], p['continuation']['additional_learner_steps'],
      p['parent']['checkpoint'], p['parent']['global_step'], p['config'])" "$plan")

if [[ ! -f "$root/final_result.json" ]]; then
  latest="$("$PY" - "$root" <<'EOF'
import json, sys
from pathlib import Path
best = None
for meta in Path(sys.argv[1]).glob("artifacts/*/model_azuki_local_*.pt.meta.json"):
  data = json.loads(meta.read_text())
  model = Path(str(meta)[: -len(".meta.json")])
  if model.is_file() and (model.parent / f"checkpoint_{int(data['update']):06d}.manifest.json").is_file():
    if best is None or data["update"] > best[0]:
      best = (data["update"], model, int(data["global_step"]))
if best:
  print(best[1], best[2])
EOF
)"
  if [[ -z "$latest" ]]; then
    run_stage="$stage"
    echo "[continuation] $(date -Is) training $element/$run_stage from $parent"
    "$PY" train-specialist/configs/run_specialist_stage.py --run-root "$root" --runtime "$RT" --stage "$run_stage" \
      --config "$config" --resume-checkpoint "$parent" --additional-learner-steps "$steps" --restart-lr-schedule
  else
    read -r checkpoint checkpoint_step <<<"$latest"
    remaining=$(( parent_step + steps - checkpoint_step ))
    (( remaining > 0 )) || { echo "[continuation] checkpoint at target but no final result; review" >&2; exit 1; }
    n=1; while [[ -e "$root/stages/${stage}_r$n" ]]; do n=$((n + 1)); done
    run_stage="${stage}_r$n"
    resume_config="$root/specialist_${element}_${run_stage}.ini"
    sed "s#^jsonl_log = .*#jsonl_log = $RT/$root/logs/specialist_${element}_${run_stage}.jsonl#" "$config" > "$resume_config"
    echo "[continuation] $(date -Is) resuming $element/$run_stage from $checkpoint (+$remaining)"
    "$PY" train-specialist/configs/run_specialist_stage.py --run-root "$root" --runtime "$RT" --stage "$run_stage" \
      --config "$resume_config" --resume-checkpoint "$checkpoint" --additional-learner-steps "$remaining" --allow-binding-change
  fi
  cp "$root/stages/$run_stage/result.json" "$root/final_result.json"
fi

all_decks="$("$PY" -c "import json; print(','.join(str(i) for i in range(len(json.load(open('train-specialist/decks/curated_deck_pool.json'))['decks']))))")"
mkdir -p "$root/trend"
for target in 25 50 75 100; do
  checkpoint="$("$PY" - "$root" "$parent_step" "$target" <<'EOF'
import json, sys
from pathlib import Path
root, base, target = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) * 1e6
rows = []
for meta in root.glob("artifacts/*/model_azuki_local_*.pt.meta.json"):
  rows.append((abs(json.loads(meta.read_text())["global_step"] - base - target), str(meta)[: -len(".meta.json")]))
print(min(rows)[1])
EOF
)"
  out="$root/trend/${element}_$(basename "$checkpoint" .pt).json"
  [[ -f "$out" ]] && continue
  echo "[continuation] $(date -Is) evaluating +${target}M checkpoint $(basename "$checkpoint")"
  "$PY" python/src/specialist_ensemble_eval.py --config "$config" --specialist "$element=$checkpoint" \
    --candidate-elements "$element" --heldout-deck-indices "$all_decks" \
    --seeds 42001701,52001704,62001707,72001710 --json "$out"
done
echo "[continuation] $(date -Is) done"
