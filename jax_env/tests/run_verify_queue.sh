#!/usr/bin/env bash
# Job-pool driver for verify_vector_fullpool.py runs.
#
# Usage:
#   run_verify_queue.sh <steps> <fail_static:0|1> <tag> <case> [case...]
#
# Keeps up to MAXPAR (default 4) verifier processes running — the RAM-safe cap
# on this 128 GB box (each peaks 20-27 GB during XLA compile). Per-case logs go
# to /tmp/verify_logs/<tag>_<case>.log; one PASS/FAIL/DONE summary line per
# case is appended to /tmp/verify_logs/<tag>_summary.txt as each case ends.
set -u
STEPS="$1"; FS="$2"; TAG="$3"; shift 3
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR=/tmp/verify_logs
SUMMARY="$LOGDIR/${TAG}_summary.txt"
MAXPAR="${MAXPAR:-4}"
mkdir -p "$LOGDIR"
: > "$SUMMARY"

run_case() {
  local case="$1"
  local log="$LOGDIR/${TAG}_${case}.log"
  FAIL_STATIC="$FS" PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO/build/python/src:$REPO/python/src:$REPO/jax_env" \
    JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$REPO/.venv/bin/python" -u "$REPO/jax_env/tests/verify_vector_fullpool.py" \
    "$STEPS" "$case" > "$log" 2>&1
  local rc=$?
  local verdict="FAIL"
  if [ $rc -eq 0 ]; then
    verdict="PASS"
  fi
  local last
  last="$(grep -E "no divergence|all cases ended|Error|error|assert" "$log" | tail -1 | cut -c1-160)"
  echo "$(date +%H:%M:%S) $verdict $case rc=$rc :: $last" >> "$SUMMARY"
}

for case in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do
    wait -n || true
  done
  run_case "$case" &
done
wait
echo "$(date +%H:%M:%S) QUEUE-DONE" >> "$SUMMARY"
echo "queue done"
