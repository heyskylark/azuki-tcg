#!/usr/bin/env bash
# Behavior traces + tactical fixtures for the Earth continuation checkpoints (earth_c75, earth_c100),
# paired against the existing u8223 traces/results, then rebuild descriptors and tables.
set -euo pipefail
RT=/home/skylark/git/azuki-tcg-specialist-runtime
A=$RT/train-specialist/analysis
O=$A/behavior
cd "$RT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$RT/build/python/src:$RT/python/src:$O/tools:$A/tactical LD_LIBRARY_PATH=$RT/build/_deps/flecs_src-build
SHARDS=4
(
  cd "$A/tactical"
  "$RT/.venv/bin/python" tactical.py earth --model earth_c75 --model earth_c100 > "$A/tactical/logs/earth_c1.log" 2>&1
  echo "[review] tactical done"
) &
tactical=$!
jobs=()
for arm in earth_c75 earth_c100; do
  for mode in argmax sample; do
    for ((s = 0; s < SHARDS; s++)); do jobs+=("$arm $mode $s"); done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "${WORKERS:-10}" -L 1 bash -c '
  arm=$0 mode=$1 s=$2
  out='"$O"'/traces/earth__${arm}__${mode}__s${s}.jsonl
  [[ -f $out.done ]] && exit 0
  '"$RT"'/.venv/bin/python '"$O"'/tools/run_traces.py --element earth --arm $arm --mode $mode \
    --shards '"$SHARDS"' --shard-index $s --out $out > '"$O"'/logs/earth__${arm}__${mode}__s${s}.log 2>&1 \
    && touch $out.done'
echo "[review] traces done"
wait "$tactical"
"$RT/.venv/bin/python" "$O/tools/analyze.py" --reps 2000
"$RT/.venv/bin/python" "$O/tools/report_tables.py"
echo "[review] done"
