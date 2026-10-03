#!/bin/bash
# Single-process full-suite verification: engine_step compiles ONCE, the
# in-memory jit cache serves all tests (the on-disk cache does not persist
# across processes here, so per-file runs would recompile each time).
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/ubuntu/code/azuki-tcg
echo "START $(date +%H:%M:%S)"
.venv/bin/python -m pytest jax_env/tests/ -q --no-header -rN --tb=line -p no:cacheprovider
echo "EXIT=$? END $(date +%H:%M:%S)"
