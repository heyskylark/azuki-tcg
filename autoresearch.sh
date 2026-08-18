#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/python/.venv-codex/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Required benchmark interpreter is missing: %s\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$ROOT/build/CMakeCache.txt" ]]; then
  printf 'Configured native build directory is missing: %s\n' "$ROOT/build" >&2
  exit 1
fi

cmake --build "$ROOT/build" \
  --target azuki_puffer_env world_tests \
  --parallel "$(nproc)"

export AZK_BUILD_PYTHON_DIR="$ROOT/build/python/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=42
export PYTHONPATH="$ROOT/python/src:$ROOT/build/python/src"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export NEPTUNE_MODE=offline

ctest --test-dir "$ROOT/build" --output-on-failure
"$PYTHON" -m pytest -q \
  "$ROOT/python/src/test_native_obs_equivalence.py" \
  "$ROOT/python/tests/test_tcg_policy.py" \
  "$ROOT/python/tests/test_tcg_sampler_argmax.py"

exec "$PYTHON" "$ROOT/scripts/autoresearch_fixed_sps.py"
