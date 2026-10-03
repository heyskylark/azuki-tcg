#!/usr/bin/env bash
# Launch every (element, arm, mode, shard) trace job. Idempotent: skips finished shards (.done) and
# shards claimed by another launcher (atomic mkdir .lock). u9305 runs deterministic mode only.
set -euo pipefail
RT=/home/skylark/git/azuki-tcg-specialist-runtime
O=$RT/train-specialist/analysis/behavior
SHARDS=${SHARDS:-4}
cd "$RT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$RT/build/python/src:$RT/python/src:$O/tools LD_LIBRARY_PATH=$RT/build/_deps/flecs_src-build
mkdir -p "$O/traces" "$O/logs"
jobs=()
for element in water earth lightning fire; do
  for arm in "$element" u8223 u9305; do
    for mode in argmax sample; do
      [[ $arm == u9305 && $mode == sample ]] && continue
      for ((s = 0; s < SHARDS; s++)); do
        jobs+=("$element $arm $mode $s")
      done
    done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "${WORKERS:-12}" -L 1 bash -c '
  element=$0 arm=$1 mode=$2 s=$3
  out='"$O"'/traces/${element}__${arm}__${mode}__s${s}.jsonl
  [[ -f $out.done ]] && exit 0
  mkdir $out.lock 2>/dev/null || exit 0
  '"$RT"'/.venv/bin/python '"$O"'/tools/run_traces.py --element $element --arm $arm --mode $mode \
    --shards '"$SHARDS"' --shard-index $s --out $out > '"$O"'/logs/${element}__${arm}__${mode}__s${s}.log 2>&1 \
    && touch $out.done'
