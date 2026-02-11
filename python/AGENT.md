# Python Training Notes (v0 model)

Use this as a quick memory aid before debugging or tuning `python/src/train.py`.

## Must-have setup

- Native binding must be importable:
  - `PYTHONPATH=build/python/src:python/src:$PYTHONPATH`
  - If you built in a different tree, replace `build/python/src` with that build path.
- After C engine or binding changes (`src/`, `include/`, `python/src/*.h`, `python/src/binding.c`):
  - Rebuild before running training.
  - Example: `cmake -S . -B build-cuda-validation -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-cuda-validation --target azuki_puffer_env -j`

## CUDA sanity checks

- Driver/device: `nvidia-smi`
- Torch CUDA in env:
  - `uv run --active python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"`
- Binding import with training PYTHONPATH:
  - `PYTHONPATH=build-cuda-validation/python/src:python/src:$PYTHONPATH uv run --active python -c "import torch, binding; print(torch.cuda.is_available(), hasattr(binding, 'env_init'))"`

## Known training pitfalls

- `python/config/azuki.ini`:
  - `vec.num_envs` must be divisible by `vec.num_workers` for multiprocessing.
  - Current default is set to `num_envs=1024`, `num_workers=8` (valid).
- PuffeRL invariant:
  - `train.batch_size >= train.minibatch_size` must hold.
  - For short smoke tests, override both (example below uses `128`).
- Common runtime trap:
  - Old `build/` cache may point at a stale absolute source path after moving repos.
  - Symptom: CMake cache source mismatch.
  - Fix: configure a fresh build dir.

## Intentional strict mask abort behavior

- The training stack is expected to abort loudly when mask creation is inconsistent.
- Keep this behavior enabled while validating new action/ability logic.
- If abort logs look truncated, ensure stderr is flushed before abort in C debug paths.

## Quick validation commands (short runs)

Use these for fast pass/fail after policy/env changes.

1. Multiprocessing + CUDA
`PYTHONPATH=build-cuda-validation/python/src:python/src:$PYTHONPATH uv run --active python python/src/train.py --config python/config/azuki.ini --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 2 --vec.batch-size 4 --train.device cuda --train.total-timesteps 512 --train.batch-size 128 --train.minibatch-size 128 --train.bptt-horizon 1 --train.update-epochs 1`

2. Serial + CUDA
`PYTHONPATH=build-cuda-validation/python/src:python/src:$PYTHONPATH uv run --active python python/src/train.py --config python/config/azuki.ini --vec.backend Serial --vec.num-envs 4 --train.device cuda --train.total-timesteps 256 --train.batch-size 128 --train.minibatch-size 128 --train.bptt-horizon 1 --train.update-epochs 1`

## Useful cleanup when interrupted runs leave workers

- Find live training processes:
  - `ps -eo pid,ppid,cmd | rg "python/src/train.py|uv run --active python python/src/train.py"`
- Kill stale train workers:
  - `pkill -f "python/src/train.py" || true`

## Throughput tuning checklist (next passes)

- Measure env vs model bottleneck first (SPS and dashboard utilization).
- If GPU underutilized and CPU has headroom:
  - increase `vec.num_envs` and keep `num_envs % num_workers == 0`
  - tune `vec.batch_size` to keep worker batches balanced
- If GPU memory allows:
  - increase `train.batch_size` and `train.minibatch_size` together
  - keep `minibatch_size` divisible by `bptt_horizon`
- Consider enabling mixed precision (`train.precision=bfloat16`) for Ampere+.
- Try `train.compile=True` only after functional stability (debugging gets harder).

## Baseline throughput snapshot (2026-02-10)

Short probe config:
`--train.total-timesteps 1024 --train.batch-size 256 --train.minibatch-size 256 --train.bptt-horizon 1 --train.update-epochs 1`

- `num_envs=4, num_workers=2` -> `SPS ~150.79`
- `num_envs=8, num_workers=4` -> `SPS ~257.02`
- `num_envs=16, num_workers=8` -> `SPS ~404.66`
- `num_envs=32, num_workers=8` -> `SPS ~491.52`

Observed from dashboard samples in these short runs:
- GPU utilization remained low (roughly single-digit %), suggesting environment-side bottlenecks dominate at this scale.

## Speed sweep snapshot (2026-02-11)

Short sweep config family:
- Multiprocessing backend
- `train.bptt_horizon=1`
- `train.update_epochs=1`
- `train.minibatch_size=train.batch_size`

Representative results:
- `64 envs / 4 workers / vec.batch=64` -> `~1123 SPS`
- `128 envs / 8 workers / vec.batch=128` -> `~1649 SPS`
- `256 envs / 8 workers / vec.batch=256` -> `~2127 SPS`
- `512 envs / 8 workers / vec.batch=512` -> `~2949 SPS`
- `480 envs / 12 workers / vec.batch=480` -> `~3501 SPS` (best observed)
- Pooled cases (`vec.batch < num_envs`) were slower in this environment for the tested points.

## Precision + compile findings (2026-02-11)

Tested on:
- `vec.num_envs=480`
- `vec.num_workers=12`
- `vec.batch_size=480`
- `train.total_timesteps=9600`
- `train.batch_size=1920`
- `train.minibatch_size=1920`
- `train.bptt_horizon=1`
- `train.update_epochs=1`

Results:
- `float32`:
  - `last-epoch SPS ~2796.47`
  - `mean SPS ~2325.26`
- `bfloat16`:
  - `last-epoch SPS ~2943.17`
  - `mean SPS ~2414.18`
- Practical takeaway:
  - `bfloat16` gave a small but consistent speed gain (~5% on this run shape).

`compile=True` status:
- `compile=True` (with `bfloat16`) did not reach epoch logs in the validation window.
- Observed prolonged Triton autotune output and Dynamo recompilation warnings, e.g.:
  - recompile limit at `python/src/policy/tcg_policy.py:98`
  - shape/key guard churn (`weapon_scalar` in warning context)
- Treat compile as non-viable for short/medium runs until graph dynamism is reduced.

## Critical gotchas discovered

- Batch divisibility rule (important):
  - With current PuffeRL buffer logic, `train.batch_size` should be a multiple of `total_agents` (for `bptt_horizon=1`, this is effectively `num_envs * agents_per_env`).
  - If violated, you can hit tensor assignment size mismatches during `trainer.evaluate()`.
  - Example failing pattern seen: `num_envs=384` (768 agents), `train.batch_size=1024`.

- CLI bool parsing trap:
  - In PufferLib config parser flow, passing `--train.compile False` can still evaluate as truthy due argparse bool typing.
  - To keep compile disabled, prefer:
    - omit the flag entirely when config already has `compile = False`, or
    - set config value directly in `.ini` and avoid CLI bool override.

## Machine-tuned config profile

- Added: `python/config/azuki_speed_3090.ini`
- Purpose: host-specific faster defaults for this 3090 + 12-core machine:
  - `vec.num_envs=480`
  - `vec.num_workers=12`
  - `vec.batch_size=480`
  - `train.precision=bfloat16`
  - `train.compile=False`
- Quick smoke command:
  - `PYTHONPATH=build-cuda-validation/python/src:python/src:$PYTHONPATH uv run --active python python/src/train.py --config python/config/azuki_speed_3090.ini --train.total-timesteps 3840 --train.batch-size 1920 --train.minibatch-size 1920 --train.bptt-horizon 1 --train.update-epochs 1`
