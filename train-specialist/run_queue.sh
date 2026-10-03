#!/usr/bin/env bash
# Train the four element specialists one at a time from u8223 to +30M learner
# steps, then evaluate the routed ensemble against the u8223/u9305 generalists.
# Re-running resumes an interrupted element from its latest saved checkpoint.
# Stops on first failure.
set -euo pipefail

RT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RT"
export PYTHONPATH="$RT/build/python/src:$RT/python/src"
export LD_LIBRARY_PATH="$RT/build/_deps/flecs_src-build"
PY="$RT/.venv/bin/python"
PARENT=/home/skylark/git/azuki-tcg/train-ablation-1781126582/results/control_strategy_scale_30m_v1/runs/control/artifacts/azuki_local_control_strategy_scale_30m_s43_179038406361/model_azuki_local_008223.pt
STEPS=30000000
ELEMENTS=(${ELEMENTS:-water earth lightning fire})

# Prints "<checkpoint> <global_step>" for the newest saved checkpoint of a run, or nothing.
latest_checkpoint() {
  "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
best = None
for meta in Path(sys.argv[1]).glob("artifacts/*/model_azuki_local_*.pt.meta.json"):
  data = json.loads(meta.read_text())
  model = Path(str(meta)[: -len(".meta.json")])
  manifest = model.parent / f"checkpoint_{int(data['update']):06d}.manifest.json"
  if model.is_file() and manifest.is_file() and (best is None or data["update"] > best[0]):
    best = (data["update"], model, int(data["global_step"]))
if best:
  print(best[1], best[2])
EOF
}

for element in "${ELEMENTS[@]}"; do
  root="train-specialist/runs/$element"
  if [[ -f "$root/final_result.json" ]]; then
    echo "[queue] $element already complete"
    continue
  fi
  base_config="$root/specialist_$element.ini"
  if [[ ! -f "$root/experiment.json" ]]; then
    echo "[queue] $(date -Is) building inputs for $element"
    "$PY" train-specialist/configs/build_specialist_inputs.py --element "$element" --run-root "$root" \
      --config-out "$base_config" --stage s30m
  fi
  checkpoint=""
  checkpoint_step=""
  read -r checkpoint checkpoint_step <<<"$(latest_checkpoint "$root")" || true
  if [[ -z "${checkpoint:-}" ]]; then
    stage=s30m
    echo "[queue] $(date -Is) training $element from u8223"
    "$PY" train-specialist/configs/run_specialist_stage.py --run-root "$root" --runtime "$RT" --stage "$stage" \
      --config "$base_config" --resume-checkpoint "$PARENT" \
      --additional-learner-steps "$STEPS" --restart-lr-schedule
  else
    parent_step="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['parent']['global_step'])" "$root/experiment.json")"
    remaining=$(( parent_step + STEPS - checkpoint_step ))
    if (( remaining <= 0 )); then
      echo "[queue] $element checkpoint already at target but no final result; review required" >&2
      exit 1
    fi
    n=1
    while [[ -e "$root/stages/s30m_r$n" ]]; do n=$((n + 1)); done
    stage="s30m_r$n"
    config="$root/specialist_${element}_$stage.ini"
    "$PY" - "$base_config" "$config" "$root/logs/specialist_${element}_$stage.jsonl" <<'EOF'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
text, count = re.subn(r"(?m)^jsonl_log = .*$", f"jsonl_log = {Path(sys.argv[3]).resolve()}", text)
assert count == 1, "expected exactly one jsonl_log line"
Path(sys.argv[2]).write_text(text)
EOF
    echo "[queue] $(date -Is) resuming $element from $checkpoint (step $checkpoint_step, +$remaining)"
    "$PY" train-specialist/configs/run_specialist_stage.py --run-root "$root" --runtime "$RT" --stage "$stage" \
      --config "$config" --resume-checkpoint "$checkpoint" \
      --additional-learner-steps "$remaining" --allow-binding-change
  fi
  cp "$root/stages/$stage/result.json" "$root/final_result.json"
done

specialist_args=()
for element in water earth lightning fire; do
  checkpoint="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" \
    "train-specialist/runs/$element/final_result.json")"
  specialist_args+=(--specialist "$element=$checkpoint")
done
all_decks="$("$PY" -c "import json; print(','.join(str(i) for i in range(len(json.load(open('train-specialist/decks/curated_deck_pool.json'))['decks']))))")"
echo "[queue] $(date -Is) evaluating ensemble"
"$PY" python/src/specialist_ensemble_eval.py --config train-specialist/runs/water/specialist_water.ini \
  "${specialist_args[@]}" --heldout-deck-indices "$all_decks" \
  --seeds 42001701,52001704,62001707,72001710 --json train-specialist/runs/ensemble_eval_s30m.json
echo "[queue] $(date -Is) done"
