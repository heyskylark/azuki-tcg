#!/usr/bin/env bash
# Paired strength trend for one element: evaluate its intermediate specialist
# checkpoints on that element's games of the s30m ensemble schedule.
set -euo pipefail
RT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$RT"
export PYTHONPATH="$RT/build/python/src:$RT/python/src" LD_LIBRARY_PATH="$RT/build/_deps/flecs_src-build"
element="$1"; shift
out="train-specialist/analysis/trend"
all_decks="$(.venv/bin/python -c "import json; print(','.join(str(i) for i in range(len(json.load(open('train-specialist/decks/curated_deck_pool.json'))['decks']))))")"
for update in "$@"; do
  checkpoint="$(ls train-specialist/runs/$element/artifacts/*/model_azuki_local_$(printf %06d "$update").pt)"
  json="$out/${element}_u${update}.json"
  [[ -f "$json" ]] && continue
  .venv/bin/python python/src/specialist_ensemble_eval.py --config "train-specialist/runs/$element/specialist_$element.ini" \
    --specialist "$element=$checkpoint" --candidate-elements "$element" --heldout-deck-indices "$all_decks" \
    --seeds 42001701,52001704,62001707,72001710 --json "$json" 2>&1 | grep "ensemble-eval"
done
