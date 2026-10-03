#!/bin/bash
# Single-process full-suite parity verification for the RTX 3090 workstation.
# engine_step compiles once; the in-process jit cache + on-disk persistent cache
# serve every test. Pass extra pytest args through ("$@"), e.g. a -k filter.
set -u
cd "$(dirname "$0")/.."
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jaxcache16}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${XLA_FLAGS:-}"
export PYTHONPATH=build/python/src:python/src:jax_env
V=python/.venv-codex/bin/python
echo "START $(date +%H:%M:%S)  cache=$JAX_COMPILATION_CACHE_DIR  xla=$XLA_FLAGS"
$V -m pytest jax_env/tests/ -q --no-header -rN --tb=line -p no:cacheprovider "$@"
echo "EXIT=$? END $(date +%H:%M:%S)"
