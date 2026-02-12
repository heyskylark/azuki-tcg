#!/usr/bin/env bash
set -euo pipefail

attempts="${CUDA_WAIT_ATTEMPTS:-60}"
sleep_seconds="${CUDA_WAIT_SLEEP_SECONDS:-2}"

check_cuinit() {
  python3 - <<'PY'
import ctypes
import sys

try:
    rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
except OSError as exc:
    print(f"cuInit loader error: {exc}", flush=True)
    sys.exit(1)

print(f"cuInit rc {rc}", flush=True)
sys.exit(0 if rc == 0 else 1)
PY
}

for i in $(seq 1 "${attempts}"); do
  if check_cuinit; then
    exit 0
  fi

  echo "CUDA not ready (${i}/${attempts}); retrying in ${sleep_seconds}s..." >&2
  sleep "${sleep_seconds}"
done

echo "CUDA failed to initialize after ${attempts} attempts." >&2
exit 1
