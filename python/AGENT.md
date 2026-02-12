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

## Env-side optimization pass (2026-02-11)

Implemented:
- Action-mask reset micro-optimization:
  - Stopped clearing all `AZK_MAX_LEGAL_ACTIONS` slots on every mask build.
  - Keep only required fields reset (`legal_action_count`, `primary_action_mask`).
- Action enumerator micro-optimization:
  - `azk_build_action_mask_for_player(...)` no longer zeroes the full `AzkActionMaskSet`.
  - It now resets only `head0_mask` and `legal_action_count` before enumeration.
- Experimental shared observation refresh path:
  - `create_training_observation_data_pair(...)`
  - `azk_engine_observe_training_all(...)`

A/B switch for debugging:
- Default (unset): shared all-player refresh (`pair`) path.
- `AZK_OBS_REFRESH_MODE=legacy`: forces old per-player refresh loop.

Measured results:
- Training benchmark (default mode, same config, 9600 timesteps):
  - Before env passes: `last3 SPS ~2712.14`
  - After pass 1: `last3 SPS ~2869.79`
  - After pass 2: `last3 SPS ~2920.65`
- Pair vs legacy training A/B (same config, 9600 timesteps):
  - `default(last3) ~2609.41`
  - `pair(last3) ~2719.99`
- Raw binding env-step benchmark:
  - post-pass default long run: `~5732 steps/s`
  - post-pass pair long run: `~5975 steps/s`
  - pair advantage in this sample: `~+4.2%`

Notes:
- Action-mask semantics are preserved.
- Keep `AZK_OBS_REFRESH_MODE=legacy` available for bisects/regression checks.

## Mask-abort debugging notes (2026-02-11, post-reboot)

CUDA/container state:
- `nvidia-smi` reports `NVIDIA GeForce RTX 3090`, driver `590.48.01`, CUDA `13.1`.
- Torch in env sees CUDA:
  - `torch 2.9.0+cu128`
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 1`

Deterministic repro pattern for zero legal-action masks:
- Use random-valid-action rollout over many seeds and stop before stepping if
  `obs["action_mask"]["legal_action_count"] == 0`.
- This catches mask creation bugs without relying on downstream invalid-action abort.

Two concrete zero-mask root causes found and fixed:

1. `STT02-016` self-discard counting bug
- Symptom:
  - Ability context entered `ABILITY_PHASE_COST_SELECTION` with
    `source_card_def_id=34 (CARD_DEF_STT02_016)` and `cost_target_type=FRIENDLY_HAND`
    while hand later had zero legal discard targets.
- Cause:
  - During deferred zone updates, the just-cast spell could still appear in hand
    and be counted as a valid discard cost target.
- Fix:
  - In `src/abilities/cards/stt02_016.c`, `stt02_016_validate_cost_target(...)`
    now rejects `target == card`.
  - Also applied same guard in `src/abilities/cards/st01_007.c` to avoid similar
    self-discard edge cases.

2. Response-phase NOOP schema mismatch
- Symptom:
  - `PHASE_RESPONSE_WINDOW` could expose `legal_action_count=0` with no ability
    subphase active.
- Cause:
  - `ACT_NOOP` was allowed by validation in response phase but not included in
    response phase action schema mask.
- Fix:
  - `src/validation/action_schema.c` now includes
    `AZK_PHASE_MASK(PHASE_RESPONSE_WINDOW)` for `ACT_NOOP`.

Validation after fixes:
- Zero-mask scanner:
  - `NO_ZERO_MASK_FOUND up_to_seed 1200` (random-valid rollout, up to 2500 steps/seed).
- Short training smokes:
  - Serial backend run completed without mask abort.
  - Multiprocessing backend run completed without mask abort/hang.
- C tests:
  - `ctest --test-dir build-cuda-validation --output-on-failure` passed.

## Env profiling + fast-path mask pass (2026-02-11, env-side optimization)

Added opt-in profilers:
- Action mask profiler (C side, `src/validation/action_enumerator.c`):
  - `AZK_MASK_PROFILE=1`
  - `AZK_MASK_PROFILE_EVERY=<N>`
  - `AZK_MASK_PROFILE_VERBOSE=1` for per-action-type counts/hit-rates
- Env step profiler (C binding, `python/src/tcg.h`):
  - `AZK_ENV_PROFILE=1`
  - `AZK_ENV_PROFILE_EVERY=<N>`

Example:
- `AZK_MASK_PROFILE=1 AZK_MASK_PROFILE_EVERY=1000 AZK_ENV_PROFILE=1 AZK_ENV_PROFILE_EVERY=500 ... train.py ...`

Fast-path mask changes:
- Kept strict validation semantics (still uses `azk_validate_*` for final acceptance).
- Added action-specific candidate generation for high-cost types:
  - `ACT_PLAY_ENTITY_TO_GARDEN`
  - `ACT_PLAY_ENTITY_TO_ALLEY`
  - `ACT_ATTACH_WEAPON_FROM_HAND`
  - `ACT_PLAY_SPELL_FROM_HAND`
  - `ACT_GATE_PORTAL`
  - `ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY`
  - `ACT_ACTIVATE_ALLEY_ABILITY`
- Also retained dynamic bounds in generic fallback:
  - hand index bounded by real hand count
  - IKZ bool toggle bounded to `0` when token unavailable
- Pair-observation refresh now skips full mask builds for non-active players:
  - in `create_training_observation_data_pair(...)`, inactive masks are reset to
    empty directly instead of calling `azk_build_action_mask_for_player(...)`.

Profiled impact (Serial probe, same command before/after):
- Mask build at 1000 calls:
  - before: `avg_us=8.66`, `normal_avg_us=22.66`, `validate=69323`
  - after:  `avg_us=6.15`, `normal_avg_us=16.51`, `validate=43679`
- Rough reduction:
  - mask avg time: `~29%` lower
  - normal-path mask time: `~27%` lower
  - validator invocations: `~37%` lower
- Env profile snapshot after pass:
  - `avg_step_us ~58-67`
  - `tick_share ~0.51-0.55`
  - `refresh_share ~0.40-0.44`
  - implication: post-pass, env tick + observation refresh are still dominant.

Training SPS sanity after pass (speed config, short 9600-step probes):
- observed last-epoch SPS samples:
  - `~2848.55`
  - `~2955.63`
- still noisy run-to-run, but consistent with the profile improvement trend.

## Env-side optimization pass 2: observation lookup pruning (2026-02-11)

CUDA/container recheck:
- `nvidia-smi` reports `RTX 3090` with driver `590.48.01`.
- Torch sees CUDA in this container:
  - `torch 2.9.0+cu128`
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 1`

Code changes (`src/utils/training_observation_util.c`):
- Removed redundant `ZoneIndex` fetches in hot observation paths.
- Board observation now avoids a second `ZoneIndex` lookup after slot resolution.
- Hand/discard observation now uses ordered-child index directly for `zone_index`
  instead of fetching `ZoneIndex` per card.
- Ability-context selection observation now uses fallback index directly (no
  extra `ZoneIndex` read + overwrite).

Validation:
- Rebuild: `cmake --build build-cuda-validation --target azuki_puffer_env -j`
- C tests: `ctest --test-dir build-cuda-validation --output-on-failure` passed.
- Multiprocessing smoke (CUDA) passed:
  - `--vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 2`

Profile comparison (Serial, same command shape before/after):
- `EnvProfile`:
  - `avg_step_us`: `84.43 -> 79.94` (`~5.3%` lower)
  - `avg_refresh_us`: `38.09 -> 34.23` (`~10.1%` lower)
  - `tick_share`: `0.508 -> 0.528` (tick now relatively more dominant)
- `ObsProfile`:
  - `avg_pair_us`: `37.67 -> 33.75` (`~10.4%` lower)
  - `avg_my_us`: `9.81 -> 9.16`
  - `avg_mask_us`: `17.05 -> 14.45` (state-dependent but improved in this probe)

Speed sanity (3090 speed config, 9600-step short probe):
- last-epoch SPS sample: `~2796.96`
- still within prior short-run noise band (`~2.8k-3.0k`).

Next env-side target:
- Instrument and break down tick internals (`azk_engine_tick`) to isolate the
  largest bucket (`phase gate`, queue processing, or `ecs_progress`) before
  applying the next optimization.

## Env-side optimization pass 3: phase-gate lookup cache (2026-02-11)

Code changes:
- Added `PhaseGateCache` ECS component (`include/components/components.h`,
  `src/components/components.c`) to hold pre-resolved pipeline/system IDs.
- Updated `src/systems/phase_gate.c`:
  - Resolve pipeline entity IDs and phase-gate system entity once during
    `init_phase_gate_system(...)`.
  - Store cache via `ecs_singleton_set_ptr(world, PhaseGateCache, &cache)`.
  - `PhaseGate(...)` and `run_phase_gate_system(...)` now read cached IDs
    instead of repeated `ecs_lookup(...)` calls each tick.

Why this matters:
- Previous path did name lookups in hot tick paths:
  - `run_phase_gate_system(...)` looked up `"PhaseGate"` each call.
  - `set_pipeline_for_phase(...)` looked up pipeline names each call.

Validation:
- Rebuild: `cmake --build build-cuda-validation --target azuki_puffer_env -j`
- C tests: `ctest --test-dir build-cuda-validation --output-on-failure` passed.
- Multiprocessing CUDA smoke passed:
  - `--vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 2`

Profile comparison (Serial probe vs pass-2 baseline):
- `EnvProfile`:
  - `avg_step_us`: `79.94 -> 71.46` (`~10.6%` lower)
  - `avg_tick_us`: `42.24 -> 37.95` (`~10.2%` lower)
  - `avg_refresh_us`: `34.23 -> 30.43` (`~11.1%` lower)
- `ObsProfile`:
  - `avg_pair_us`: `33.75 -> 30.09`
  - `avg_my_us`: `9.16 -> 7.70`
  - `avg_mask_us`: `14.45 -> 13.74`

Speed sanity (3090 config, short 9600-step probes):
- Run A last-epoch SPS: `~2706.66`
- Run B last-epoch SPS: `~2953.40`
- Mid-epoch samples remained around `~2.85k-3.39k`; short-run SPS remains noisy.

Current interpretation:
- Env-side changes are lowering per-step C runtime in serial profiling.
- Multiprocessing training throughput still has notable variance run-to-run, so
  evaluate improvements with small probe batches (2-3 runs) before final calls.

## Training debug + invalid-action fix (2026-02-11, follow-up)

Context:
- User asked to continue validation/tuning and investigate AEC bottlenecks +
  high-ROI env optimizations on this 3090 host.

Important clarification:
- The previously reported `AEC steps/s ~4406.13` did **not** use
  `python/config/azuki_speed_3090.ini`.
- That number came from a standalone local microbenchmark script that directly
  exercised `AzukiTCG` (no PuffeRL train loop/config parsing).

Build/runtime alignment fixes:
- Existing bindings in `build-cuda-validation/`, `build-validation/`, and
  `build-debug-validation/` were linked against `libpython3.13`, while runtime
  here is Python 3.12.
- Created new build tree and rebuilt binding for local runtime:
  - `cmake -S . -B build-312 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython3_EXECUTABLE=$(python3 -c 'import sys; print(sys.executable)')`
  - `cmake --build build-312 --target azuki_puffer_env -j`

PufferLib API-compat fixes (local project code):
- `python/src/training_utils.py`:
  - Replaced removed `pufferl.make_parser`/`load_config_file` path with local
    config loader that merges:
    1) `pufferlib/config/default.ini`
    2) project `.ini` (e.g. `azuki_speed_3090.ini`)
    3) CLI overrides.
  - Added derived `train.use_rnn` key expected by current PuffeRL.

Invalid-action root cause and fix:
- Symptom:
  - Immediate aborts with sampled invalid action `[4,0,0,0]` while legal mask
    showed only NOOP/end-turn.
- Repro diagnostics:
  - In policy path, decoded `legal_action_count` was `65535` for both agents.
  - Raw wrapped env observations had correct masks when viewed via
    `PettingZooPufferEnv.obs_struct`.
- Root cause:
  - `TCG` policy used `pufferlib.pytorch.nativize_dtype(env.emulated)` metadata
    that produced incorrect byte offsets for this aligned structured dtype,
    causing action-mask fields to be read from wrong bytes.
- Fix:
  - `python/src/policy/tcg_policy.py`
    - Added local structured-dtype metadata builder based on numpy field offsets
      from `env.emulated["emulated_observation_dtype"]`.
    - Switched policy observation parsing to this metadata for
      `pufferlib.pytorch.nativize_tensor(...)`.
- Validation:
  - Single-env serial policy probe now reports:
    - `primary true count [2, 0]`
    - `legal_count [2, 0]`
    - legal actions `[[0,0,0,0], [23,0,0,0]]`
    - sampled action legal and first step succeeds.
  - Short CUDA train smoke now completes (no invalid-action abort):
    - `--train.total-timesteps 3840 --train.batch-size 1920 --train.minibatch-size 1920 --train.bptt-horizon 1 --train.update-epochs 1`
    - epoch logs emitted normally.

AEC/turn-wrapper microbench refresh (current code):
- Direct AEC env loop: `~1037.01 steps/s`
- `turn_based_aec_to_parallel(...)` wrapper loop: `~763.27 steps/s`
- Interpretation:
  - Wrapper/conversion overhead remains material (`~26%` lower throughput in
    this probe), consistent with AEC bridge being a meaningful bottleneck.

Notes on observation ordering experiment:
- Tried recursively sorting dict keys in `observation_to_dict`; it did not
  resolve invalid-action issue. Root cause was offset metadata mismatch above.

Post-fix cleanup:
- Reverted temporary observation key-sorting experiment in `python/src/observation.py`
  after confirming it was unrelated to the root cause.
- Kept the actual fix in `python/src/policy/tcg_policy.py` (native dtype offsets
  built from `emulated_observation_dtype`), which resolves the invalid-action
  abort path.

## Parallel-env pass + SPS sweep (2026-02-11, follow-up 2)

Goal:
- Remove `turn_based_aec_to_parallel` bridge overhead where possible.
- Test higher-throughput optimized settings on this 3090 machine.
- Investigate native PufferEnv + `vec_step` fast-path viability.

### Implemented: direct ParallelEnv path (no AEC bridge)

New file:
- `python/src/tcg_parallel.py` (`AzukiTCGParallel`)
  - Uses the same C binding (`env_init`/`env_step`) but exposes PettingZoo
    `ParallelEnv` directly.
  - Keeps action encoding and observation dict schema identical to current
    training stack.

Wiring:
- `python/src/training_utils.py`
  - Added `env.direct_parallel` switch in `make_azuki_env(...)`.
  - When enabled:
    - `AzukiTCGParallel -> MultiagentEpisodeStats -> PettingZooPufferEnv`
  - Existing default path remains:
    - `AzukiTCG -> turn_based_aec_to_parallel -> MultiagentEpisodeStats -> PettingZooPufferEnv`

### Implemented: C vec-step safety for completed matches

File:
- `python/src/env_binding.h`
  - `vec_step(...)` now checks if env is terminal (`terminals[0] && terminals[1]`)
    and performs `c_reset(env)` before continuing, avoiding invalid post-terminal
    `c_step(...)` calls.

Also updated `vec_init`/`env_init` ndim checks to accept n-d arrays for reward
and done buffers (needed for packed vector experiments).

### Native PufferEnv + vec_init experiment (blocked)

Explored `python/src/tcg_native.py` path:
- Intended to use raw native obs bytes (`OBSERVATION_CTYPE`) and C `vec_step`.
- Blocker: PufferLib nativization assumes contiguous leaf slices. Our aligned
  struct has interleaved array-of-struct fields (e.g., weapon slots), so raw
  byte slicing/viewing cannot recover those fields correctly without custom
  strided extraction support.
- Result:
  - Left `env.native` explicitly disabled in `training_utils.py` with a clear
    runtime error message to prevent accidental broken runs.

### Direct-parallel microbench (env-only)

Parallel loop benchmark (random legal actions):
- AEC bridge path (`turn_based_aec_to_parallel(AzukiTCG)`): `~1592.24 steps/s`
- Direct parallel path (`AzukiTCGParallel`): `~2293.26 steps/s`
- Improvement: `~+44%` in this env-only probe.

### Optimized training sweeps (using 3090 profile shape)

Command family:
- Multiprocessing backend
- `train.bptt_horizon=1`, `train.update_epochs=1`
- matched batch/minibatch divisibility

1) `480 envs / 12 workers / batch 480` baseline path
- last 3 epoch SPS: `~3401`, `~3350`, `~3115`
- mean(last3): `~3289`

2) `480 envs / 12 workers / batch 480` direct-parallel path
- last 3 epoch SPS: `~3372`, `~3309`, `~3392`
- mean(last3): `~3358`
- delta vs baseline mean(last3): `~+2.1%`

3) `720 envs / 12 workers / batch 720` baseline path
- last 3 epoch SPS: `~4321`, `~4109`, `~4091`
- mean(last3): `~4174`

4) `720 envs / 12 workers / batch 720` direct-parallel path
- run A last 3 epoch SPS: `~4472`, `~4605`, `~4546` (mean `~4541`)
- run B last 3 epoch SPS: `~4464`, `~4556`, `~4228` (mean `~4416`)
- direct-parallel lift vs 720 baseline: roughly `~+6% to +9%` depending on run
  variance.

Best observed in these sweeps:
- `~4605 SPS` epoch sample on 720-env direct-parallel run.

### New tuned config

Added:
- `python/config/azuki_speed_3090_parallel.ini`
  - `vec.num_envs=720`
  - `vec.num_workers=12`
  - `vec.batch_size=720`
  - `env.direct_parallel=True`
  - `train.precision=bfloat16`, `compile=False`

Recommended short probe with this config:
- `--train.total-timesteps 14400 --train.batch-size 2880 --train.minibatch-size 2880 --train.bptt-horizon 1 --train.update-epochs 1`

### Updated C-side hotspot signal

With direct-parallel env loop and `AZK_ENV_PROFILE=1`:
- `avg_step_us ~25.4`
- `avg_tick_us ~3.05`
- `avg_refresh_us ~20.94`
- `refresh_share ~0.82`

Interpretation:
- Observation refresh now dominates C step time in this path.
- Highest C-env ROI remains reducing observation refresh cost further.

## Learning-validation pivot (2026-02-11)

User asked whether current speed is "fast enough" and to prioritize validating
that training is actually learning.

### Decision
- Yes: current throughput is sufficient to pivot to learning validation.
- Stable observed train throughput on this host/config is already in the
  `~4.2k-4.5k SPS` range (with direct-parallel speed profile), which is enough
  to iterate quickly on learning signal checks.

### Added evaluator
- New script: `python/src/evaluate_checkpoint.py`
  - Evaluates a checkpoint against a random-legal opponent.
  - Supports seat control (`--policy-seat 0|1`) for first/second player bias.
  - Uses fixed-horizon episodes via `--max-steps` because many episodes do not
    terminate within short windows.
  - Reports:
    - `avg_cumulative_reward` (primary dense learning signal for short horizon)
    - `positive_reward_rate`
    - win/loss/draw + timeout counts.

### Why fixed-horizon reward (current state)
- In short probes (`max_steps=250`), episodes frequently timeout before terminal
  winners are reached, so raw win-rate is low signal at this stage.
- Dense shaped reward provides faster/usable "is policy improving?" feedback.

### Baseline vs trained comparison (same seed/horizon)

Config:
- `python/config/azuki_speed_3090_parallel.ini`
- `episodes=8`, `max_steps=250`, `device=cuda`, `seed=20260211`

Untrained policy (no checkpoint):
- Seat 0:
  - `avg_cumulative_reward = +0.0146947577`
  - `positive_reward_rate = 0.625`
- Seat 1:
  - `avg_cumulative_reward = -0.0874910448`
  - `positive_reward_rate = 0.125`

Trained policy (`experiments/177078829082.pt`, 115,200 timesteps run):
- Seat 0:
  - `avg_cumulative_reward = +0.0118126057`
  - `positive_reward_rate = 0.625`
- Seat 1:
  - `avg_cumulative_reward = -0.0070117706`
  - `positive_reward_rate = 0.25`

Combined seat-average reward:
- Untrained: `(-0.036398...)`
- Trained:   `(+0.002400...)`
- Directionally positive shift after short training.

Interpretation:
- Learning signal is present but still early/noisy.
- Biggest observed gain is second-seat performance (less negative reward).
- This is enough evidence to continue validation-focused passes before spending
  more time on deep env micro-optimization.

### Training run used for checkpoint
- Command:
  - `PYTHONPATH=build-312/python/src:python/src:$PYTHONPATH uv run --active python python/src/train.py --config python/config/azuki_speed_3090_parallel.ini --train.total-timesteps 115200 --train.batch-size 2880 --train.minibatch-size 2880 --train.bptt-horizon 1 --train.update-epochs 1`
- Outcome:
  - Completed 40 epochs
  - SPS after warmup mostly `~4.2k-4.5k`
  - Produced checkpoint: `experiments/177078829082.pt`

## Why PufferLib "User Stats" can appear blank (2026-02-11)

Symptom:
- Dashboard shows empty `User Stats` columns for long stretches, even though
  custom stats exist in env terminal infos.

Root cause:
- In this stack, custom stats (e.g. win/leader_health aggregates) are emitted
  primarily at episode end (terminal/truncation).
- With the fast profile (`720 envs`, `2 agents/env`), a `1,000,000` agent-step
  run gives each env only about:
  - `1,000,000 / (720 * 2) ≈ 694` environment turns.
- If matches do not finish (or truncate) within that per-env turn budget, stats
  may remain empty for the whole run.

Validation:
- Instrumented `PuffeRL.evaluate()` behavior and confirmed `self.stats` keys
  populate only once an episode boundary is reached.
- Example with `AZK_MAX_TICKS_PER_EPISODE=40`:
  - stats keys appear around eval 20 (`0/episode_return`, `0/win`, etc.).

Implemented support:
- `python/src/tcg.h` now supports optional env cap:
  - `AZK_MAX_TICKS_PER_EPISODE=<positive int>`
  - default is disabled (`0`) to preserve current behavior when unset.

Practical recommendation for 1M-step runs:
- Set `AZK_MAX_TICKS_PER_EPISODE` to a value below per-env step budget so stats
  appear predictably during training (e.g. `200`-`300`).

## Long-run stall at ~737k steps (2026-02-11)

Observed live process state during a "frozen" 5M run:
- trainer process alive and CPU-hot
- one worker process CPU-hot
- most other workers sleeping
- GPU memory still allocated but utilization near 0%

Interpretation:
- Likely rare env-step livelock in worker path:
  - worker stuck inside a single `c_step` auto-tick progression loop
  - trainer waits for complete worker batch, so visible global progress halts.

Hardening added:
- `python/src/tcg.h`
  - Added auto-tick safety guard:
    - env var: `AZK_MAX_AUTO_TICKS_PER_STEP`
    - default: `20000` (set `0` to disable)
  - If a single env step exceeds this many auto-ticks without requiring action
    and without game-over, env now:
    - forces truncation (`truncations[0]=truncations[1]=DONE`)
    - records episode stats
    - returns control to trainer instead of hanging indefinitely.

Run recommendation for long jobs:
- Use both safeguards for stability + stats visibility:
  - `AZK_MAX_AUTO_TICKS_PER_STEP=20000`
  - `AZK_MAX_TICKS_PER_EPISODE=250` (or 300-500 depending desired episode span)

## W&B run audit: `9r6dqnny` (`azuki_3090_parallel_7m`) (2026-02-11)

Link audited:
- `https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/9r6dqnny`

### Facts from run metadata
- State: `finished`
- Config:
  - `vec.num_envs=720`, `vec.num_workers=12`, `env.direct_parallel=true`
  - `train.total_timesteps=7000000`
  - `train.batch_size=auto`, `bptt_horizon=16`, `update_epochs=2`
- Final summary:
  - `agent_steps=6,981,120`
  - `epoch=303`
  - `SPS~3687`

### Why it ended below 7,000,000 steps
- With current PuffeRL loop, stopping is epoch-driven.
- Effective steps/epoch in this run were `~23,040`.
- `303 * 23,040 = 6,981,120`.
- So this run ended on epoch boundary below requested timesteps.

### Learning signal findings
- Losses/optimization looked numerically stable:
  - `entropy ~0.74` (still fairly stochastic)
  - `approx_kl ~7e-4`, `clipfrac ~0.003`
  - `explained_variance ~0.995` (value model fits current target signal)
- Episodic game-outcome signal remained weak:
  - `environment/0/win` and `environment/1/win` final both `0`
  - episodic keys (`episode_length`, `episode_return`, `win`) appeared only a
    few times in the run history and were mostly null per epoch.
  - when present, `episode_length` showed `1000` and near-zero symmetric
    returns (`~+/-0.013`) -> timeout-like outcomes and no decisive winner.

Interpretation:
- PPO is training stably against the shaped reward target but not receiving
  enough decisive terminal outcome signal to learn "finish/win" behavior.
- This is consistent with sparse episode boundaries and mostly draw/timeout
  trajectories.

## Follow-up implementation: items 1, 2, and 4 (2026-02-11)

Applied first-pass changes requested before item 3:

1) Terminal diagnostics in user stats
- C log schema extended in `python/src/tcg.h` + `python/src/binding.c`:
  - `p0_episode_return`, `p1_episode_return`
  - `timeout_truncation_rate`
  - `auto_tick_truncation_rate`
  - `gameover_terminal_rate`
  - `winner_terminal_rate`
- Python env info fan-out updated (`python/src/tcg.py`, `python/src/tcg_parallel.py`)
  to publish:
  - `azk_episode_return`, `azk_episode_length`
  - `azk_timeout_truncation`, `azk_auto_tick_truncation`
  - `azk_gameover_terminal`, `azk_winner_terminal`
- `episode_length` and per-seat episode returns are now tracked explicitly in C.

2) Reward rebalance for truncation outcomes
- Added truncation reward path in `python/src/tcg.h`:
  - timeout truncation penalty (`TRUNCATION_TIMEOUT_PENALTY`)
  - stronger auto-tick-guard truncation penalty (`TRUNCATION_AUTO_TICK_PENALTY`)
  - leader-health edge tie-breaker (`TRUNCATION_LEADER_EDGE_WEIGHT`)
- This replaces previous zero reward on truncation, so timeout-heavy runs now
  receive directional learning signal instead of flat draws.
- Per-episode return accumulation now includes both shaped step rewards and
  terminal/truncation rewards.

4) Epoch/timestep quantization visibility + optional alignment
- `python/src/train.py` now prints an explicit epoch plan before training:
  - effective batch size, total epochs, effective total timesteps.
- Added optional config key:
  - `train.align_total_timesteps_up = True|False`
  - if `True`, requested timesteps are rounded up to the next full epoch batch.
  - if `False`, trainer prints how many steps will be dropped at epoch boundary.
- Enabled in optimized config:
  - `python/config/azuki_speed_3090_parallel.ini` now sets
    `align_total_timesteps_up = True`.

Smoke validation after implementation:
- Rebuilt native module:
  - `cmake --build build-312 --target azuki_puffer_env -j`
- Python syntax check:
  - `python -m py_compile python/src/train.py python/src/tcg.py python/src/tcg_parallel.py`
- Training smoke (optimized parallel config) passed:
  - `AZK_MAX_TICKS_PER_EPISODE=20 ... --train.total-timesteps 23040`
- Training smoke with forced fast episode boundaries passed and emitted new stats:
  - `AZK_MAX_TICKS_PER_EPISODE=8 ... --train.total-timesteps 23040`
  - dashboard/user stats showed:
    - `azk_timeout_truncation=1.0`
    - `azk_auto_tick_truncation=0.0`
    - `azk_gameover_terminal=0.0`
    - `azk_winner_terminal=0.0`
    - non-zero per-seat `azk_episode_return`

Bug encountered/fixed during this pass:
- Using raw info keys `episode_length` / `episode_return` in env infos caused a
  PufferLib worker crash (`AttributeError: 'int' object has no attribute 'append'`)
  due key collision with built-in episodic fields.
- Fixed by renaming custom keys with `azk_` prefix.

Deferred (item 3, intentionally not implemented in this pass):
- Adaptive/curriculum episode-cap strategy (e.g., staged
  `AZK_MAX_TICKS_PER_EPISODE` schedule by training progress) is queued for next
  implementation pass after validating this run.

## Runtime compatibility note: WandbLogger `no_model_upload` (2026-02-11)

Observed failure:
- `KeyError: 'no_model_upload'` from `pufferlib.pufferl.WandbLogger.__init__`.

Root cause:
- Current `SkyPufferLib` logger expects top-level keys that are not present in
  its current `default.ini` (`no_model_upload`, and sometimes `tag` /
  `wandb_group` depending CLI usage).
- Our custom config loader previously passed `trainer_args` through without
  filling those defaults.

Fix applied:
- `python/src/training_utils.py` now injects compatibility defaults:
  - `parsed["no_model_upload"] = False` if missing
  - `parsed["tag"] = None` if missing
  - `parsed["wandb_group"] = None` if missing
  - `parsed["wandb_project"] = train.project` fallback if missing

Validation:
- W&B-offline smoke run with full `--wandb` CLI flags now launches, trains, and
  exits cleanly without `KeyError`.

## W&B run check: `sgloyce7` missing user stats (2026-02-11)

Run:
- `https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/sgloyce7`

Findings from W&B API:
- Run state: `running` during query.
- `historyKeys` contains only train/perf/system keys (no `environment/0/*` or
  `environment/1/*` keys at all).
- Config shows:
  - `vec.num_envs=720`, `bptt_horizon=16`, `update_epochs=2`
  - aligned `train.total_timesteps=1013760` (epoch boundary alignment)
- latest `agent_steps=1013760`, `epoch=44`.

Interpretation:
- This is consistent with **no episode boundary reached yet** rather than a
  user-stats logging crash.
- With 2 agents/env, rough per-env action budget so far:
  - `1013760 / (720 * 2) ~= 704` env actions per environment.
- If `AZK_MAX_TICKS_PER_EPISODE=1000`, truncation would not trigger yet, so
  episode-end stats (`win`, `leader_health`, `azk_*`) are never emitted.

Actionable threshold for next runs:
- To guarantee at least one periodic episode boundary in a 1M-step run at this
  scale, set `AZK_MAX_TICKS_PER_EPISODE` below about `700` (practically
  `250-500` is better for denser signal).

## W&B run check: `u1z6768z` (post-fix 1M run) (2026-02-11)

Run:
- `https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/u1z6768z`

Comparison vs prior 1M run (`sgloyce7`):
- Prior run had **no** `environment/*` keys (no episode boundaries reached).
- This run has `environment/*` + `azk_*` keys present (2 logged points), so the
  user-stats pipeline is working and episode boundaries are now occurring.

Key observations:
- Final steps: `agent_steps=1,013,760`, `epoch=44` (aligned epoch stop).
- Throughput: `SPS ~3531.6` (lower than `~4020.6` in boundary-free run, expected
  due to periodic truncation/reset overhead).
- Episode diagnostics:
  - `environment/0/azk_episode_length=300`
  - `environment/1/azk_episode_length=300`
  - `azk_timeout_truncation=1`
  - `azk_gameover_terminal=0`
  - `azk_winner_terminal=0`
  - `azk_auto_tick_truncation=0`
- Interpretation: boundaries are now mostly/entirely timeout truncations; games
  still not reaching natural terminal win/loss within the cap.

Learning-signal read:
- Numerical PPO stability is good (`explained_variance ~0.995`, KL/clipfrac
  controlled), but decisive game-finishing behavior has not emerged yet.
- This supports moving to plan 3 (adaptive cap curriculum) to shape trajectory
  length over training rather than keeping a fixed hard cap.

## Plan 3 implemented: adaptive episode-cap curriculum (2026-02-11)

Implemented in C env (`python/src/tcg.h`):
- New optional curriculum scheduler for timeout cap:
  - `AZK_MAX_TICKS_CURRICULUM=1` enables it.
  - `AZK_MAX_TICKS_CURRICULUM_INITIAL` (default: `min(300, final)`).
  - `AZK_MAX_TICKS_CURRICULUM_FINAL` (default: `AZK_MAX_TICKS_PER_EPISODE` if set, else `1000`).
  - `AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES` (default: `0`).
  - `AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES` (default: `5000`).
- Per-env schedule (deterministic):
  - hold `initial` through warmup episodes
  - linearly ramp to `final` over ramp episodes
  - hold `final` after ramp completes
- Default behavior unchanged when `AZK_MAX_TICKS_CURRICULUM` is unset.

New diagnostic metric:
- Added `curriculum_episode_cap` to env logs (`python/src/binding.c`) and
  surfaced to user stats as `azk_curriculum_episode_cap` in both env wrappers.

Smoke validation:
- Rebuilt binding for current runtime:
  - `cmake -S . -B build-312 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython3_EXECUTABLE=/root/.local/share/uv/python/cpython-3.13.5-linux-x86_64-gnu/bin/python -DPython3_NumPy_INCLUDE_DIR=/puffertank/venv/lib/python3.13/site-packages/numpy/core/include`
  - `cmake --build build-312 --target azuki_puffer_env -j`
- Serial smoke (`4 envs`, CPU) with tiny curriculum:
  - `AZK_MAX_TICKS_CURRICULUM=1`
  - `AZK_MAX_TICKS_CURRICULUM_INITIAL=8`
  - `AZK_MAX_TICKS_CURRICULUM_FINAL=16`
  - `AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES=1`
- Observed expected ramp in dashboard/user stats:
  - early epoch: `azk_curriculum_episode_cap=8.0`, `episode_length=8`
  - subsequent epochs: `azk_curriculum_episode_cap=16.0`, `episode_length=16`

## W&B run check: `sng99rlt` (3M steps, curriculum enabled) (2026-02-11)

Run:
- `https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/sng99rlt`

Summary:
- Completed: `agent_steps=3,018,240`, `epoch=131`, `SPS~3522.9`.
- User stats are now consistently present (not blank), with 4 episode-boundary
  updates over this run.
- Curriculum progressed: final `azk_curriculum_episode_cap=663` and
  `azk_episode_length=663` (indicates cap schedule is active in production run).

Learning/termination signal:
- `azk_timeout_truncation=1`, `azk_gameover_terminal=0`, `azk_winner_terminal=0`,
  `win=0` at final summary.
- PPO remains numerically stable (`explained_variance~0.995`, KL/clipfrac controlled),
  but run still did not show natural terminal wins.

Interpretation:
- Plan 3 implementation worked technically (cap curriculum + observability).
- At 3M steps, policy still appears to optimize shaped/time-limited play rather
  than reliably finishing games.

## Stall fix: auto-tick guard counter bug (2026-02-11)

Observed during analysis of stalled 10M run (`0mczv6qw`):
- Run heartbeat/history stopped advancing while process remained alive.
- Likely waiting on a worker stuck in long `c_step` auto-tick progression.

Root cause found in `python/src/tcg.h`:
- `auto_tick_count` was incremented only inside:
  - `if (g_env_profile.enabled) { ... auto_tick_count++; }`
- Therefore, with profiling disabled (normal runs), `AZK_MAX_AUTO_TICKS_PER_STEP`
  guard could never trigger, effectively disabling the anti-livelock protection.

Fix applied:
- Increment `auto_tick_count` unconditionally once per `azk_engine_tick(...)`.
- Keep timing accumulation conditional on profiling only.

Impact:
- `AZK_MAX_AUTO_TICKS_PER_STEP` now works in normal (non-profiled) training
  runs and should prevent this class of silent worker stalls.

## W&B run check: `ekuilmyq` (10M steps, post-stall-fix) (2026-02-11)

Run:
- `https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/ekuilmyq`

Key facts:
- Completed: `agent_steps=10,022,400`, `epoch=435`, `SPS~3597`.
- User stats present regularly (`52` logged points for environment metrics).
- No obvious runtime stall pattern; run finished.

Learning signal:
- PPO numerics stable:
  - `losses/explained_variance ~0.993`
  - `approx_kl/clipfrac` in normal range
  - entropy reduced vs earlier runs (`~0.54`), indicating policy is becoming
    more deterministic.
- But outcome behavior still timeout-dominant:
  - `azk_timeout_truncation=1`
  - `azk_gameover_terminal=0`
  - `azk_winner_terminal=0`
  - `win=0`

Curriculum behavior:
- Final `azk_curriculum_episode_cap=253` (near initial cap of 250).
- This indicates the current schedule is too slow to ramp under observed
  per-env episode counts. With `RAMP_EPISODES=10000`, cap barely moved in 10M
  steps.

Interpretation:
- Observation/action plumbing appears healthy (stats coherent, stable training,
  no invalid-action abort signatures).
- Primary bottleneck is curriculum schedule design, not obvious obs corruption.

Recommended next change:
- Keep training, but make curriculum materially faster so matches can reach
  deeper phases:
  - e.g. `AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES` in low hundreds (or lower),
    not `10000`.
  - alternatively change curriculum driver from episode-count to step-count.

## W&B run audit: `7rvxoz1o` (2026-02-11)

Run link:
- https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/7rvxoz1o

Observed end-of-run summary (from W&B run data):
- state: `finished`
- steps: `10,022,400`
- epoch: `435`
- SPS: `~3829`
- custom env stats keys were present (user stats logging pipeline is active)

Outcome signals:
- `environment/0/azk_timeout_truncation = 1.0`
- `environment/0/azk_gameover_terminal = 0.0`
- `environment/0/azk_winner_terminal = 0.0`
- `environment/0/win = 0.0`
- same shape for player `1/*` metrics

Curriculum/tick-cap progression:
- `environment/*/azk_curriculum_episode_cap` progressed during training and finished around `326`.
- This is higher than earlier baselines, but still not enough to routinely reach natural terminal outcomes.

Interpretation:
- This does **not** look like a total observation pipeline failure:
  - action masking/user stats are being produced and logged,
  - training loop/losses/throughput are stable,
  - no widespread NaNs or collapse pattern.
- Current failure mode is more consistent with sparse terminal signal + timeout-dominated trajectories.

High-ROI next checks (before deep architecture changes):
1. Add a one-shot observation integrity probe in training:
   - log non-zero counts and simple checksums for key obs tensors (`cards`, `stats`, `history`, masks) every N steps,
   - verify variation across episodes/seeds and between players.
2. Force a "terminal-reach" debug profile:
   - temporarily raise episode cap ceiling and/or reduce game complexity seeds,
   - confirm the policy can observe at least some non-timeout terminals during PPO data collection.
3. Add terminal mix metrics to default dashboard:
   - rolling ratio of timeout vs winner/gameover terminals,
   - median episode length at terminal events.

If these checks pass, treat learning as "slow but valid" and run longer with adjusted curriculum/reward shaping rather than assuming broken observations.

## Observation/mask integrity debug pass (2026-02-11)

Goal:
- Add low-friction diagnostics to detect whether observations/masks are static, malformed, or collapsing to NOOP-heavy behavior.

Implementation:
- Added `python/src/obs_debug.py` with an episode tracker that computes per-agent debug stats.
- Wired tracker into both env wrappers:
  - `python/src/tcg.py`
  - `python/src/tcg_parallel.py`
- Debug is opt-in via env vars:
  - `AZK_OBS_DEBUG=1`
  - `AZK_OBS_DEBUG_SAMPLE_EVERY=<N>`

New per-episode user stats (logged as `environment/<agent>/azk_dbg_*`):
- `azk_dbg_obs_samples`
- `azk_dbg_legal_action_count_mean`
- `azk_dbg_legal_action_count_min`
- `azk_dbg_legal_action_count_max`
- `azk_dbg_legal_zero_rate`
- `azk_dbg_primary_mask_nonzero_mean`
- `azk_dbg_noop_only_rate`
- `azk_dbg_legal_primary_mismatch_rate`
- `azk_dbg_obs_signature_unique_ratio`
- `azk_dbg_player_hand_count_std`
- `azk_dbg_opponent_hand_count_std`

Validation smoke (local in-container):
- Multiprocessing smoke could not run in this sandbox due `/dev/shm` permission (`PermissionError` from Python sharedctypes).
- Serial smoke succeeded and emitted all `azk_dbg_*` keys in epoch logs.
- Example observed values from smoke:
  - `azk_dbg_legal_primary_mismatch_rate = 0.0` (good: no "legal_count>0 with empty primary mask")
  - non-zero `azk_dbg_obs_signature_unique_ratio` and hand-count stds (suggests observations vary over episode)
  - high timeout regime still visible via `azk_timeout_truncation=1.0`

Recommended debug-first run command (real machine):
- Enable debug metrics for a short run first:
`AZK_OBS_DEBUG=1 AZK_OBS_DEBUG_SAMPLE_EVERY=4 AZK_MAX_AUTO_TICKS_PER_STEP=20000 AZK_MAX_TICKS_CURRICULUM=1 AZK_MAX_TICKS_CURRICULUM_INITIAL=250 AZK_MAX_TICKS_CURRICULUM_FINAL=1200 AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES=0 AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES=10000 PYTHONPATH=build-312/python/src:python/src:$PYTHONPATH uv run --active python python/src/train.py --config python/config/azuki_speed_3090_parallel.ini --wandb --wandb-project azuki_tcg --wandb-group rtx3090 --train.total-timesteps 1000000 --train.name azuki_3090_parallel_1m_obsdebug --tag obsdebug`

Readout guidance:
- If `azk_dbg_obs_signature_unique_ratio` is near zero and hand-count stds are ~0 for long windows, suspect observation staleness.
- If `azk_dbg_legal_primary_mismatch_rate > 0`, suspect mask packing/translation issue.
- If `azk_dbg_noop_only_rate` is very high for both players alongside timeout=1.0, focus on reward shaping / curriculum to escape sparse-signal mode.

## W&B run audit: `of029pzg` (obsdebug, 2026-02-11)

Run link:
- https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/of029pzg

Status at check time:
- W&B state: `running`
- heartbeat: `2026-02-11T23:13:23Z`
- steps: `1,013,760` (target was ~1,000,000)
- epochs logged: `44`
- SPS: `~3723`

Debug-metric readout (new `azk_dbg_*`):
- `azk_dbg_legal_primary_mismatch_rate = 0.0` (both players)
- `azk_dbg_obs_signature_unique_ratio ~0.027` (non-zero, not flatlined)
- hand-count stds are non-zero (`~0.08-0.11`)
- legal-action means are non-zero (`~4.4` p0, `~5.5` p1)
- NOOP-only rate non-zero but not dominant for both (`~0.195` p0, `~0.045` p1)

Outcome/learning signals:
- timeout regime still dominant:
  - `azk_timeout_truncation = 1.0`
  - `azk_gameover_terminal = 0.0`
  - `azk_winner_terminal = 0.0`
  - `win = 0.0`
- episode cap progressed to `313`, but still no natural terminals.

Interpretation:
- No clear sign of the previous severe observation-plumbing bug.
- Current bottleneck remains sparse terminal learning signal under timeout-heavy trajectories.
- Since run state is still `running` with no heartbeat movement since `23:13:23Z`, process may have exited without clean finalize or be stalled.

## Reward/curriculum/action-debug implementation pass (2026-02-11)

Confirmed phase/sub-phase model inputs:
- Main phase and ability sub-phase are both embedded and fed to policy global context:
  - `python/src/policy/tcg_policy.py` uses `game_phase_encoder` and `ability_phase_encoder`.
  - `_encode_global_context(...)` reads both `structured_obs['phase']` and `ability_context['phase']`.

Implemented plan items:

1) Action-choice diagnostics (selected action histogram)
- Added per-episode selected-action rates to C log pipeline:
  - `p0_noop_selected_rate`, `p1_noop_selected_rate`
  - `p0_attack_selected_rate`, `p1_attack_selected_rate`
  - `p0_play_selected_rate`, `p1_play_selected_rate`
  - `p0_ability_selected_rate`, `p1_ability_selected_rate`
  - `p0_target_selected_rate`, `p1_target_selected_rate`
- Source changes:
  - `python/src/tcg.h` (action counters + per-episode aggregation)
  - `python/src/binding.c` (expose new log keys)
  - `python/src/tcg.py`, `python/src/tcg_parallel.py` (fan out to `azk_*` user stats)

2) Reward shaping v1 (leader delta + board delta + noop shaping)
- Added tunable dense reward components on top of existing PBRS potential delta:
  - leader edge delta term (`AZK_REWARD_LEADER_DELTA_WEIGHT`)
  - board edge delta term (`AZK_REWARD_BOARD_DELTA_WEIGHT`)
  - NOOP penalty when alternatives existed (`AZK_REWARD_NOOP_PENALTY`)
- Added truncation board-edge shaping:
  - `AZK_TRUNCATION_BOARD_EDGE_WEIGHT`
- Implementation in `python/src/tcg.h` via reward tuning init + `apply_shaped_rewards(...)` and `apply_truncation_rewards(...)`.

3) Curriculum shaping v2 (forced long episodes mix)
- Added deterministic long-episode injection into episode-cap curriculum:
  - `AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_EVERY`
  - `AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_CAP`
- Behavior: every Nth episode uses long cap; others use warmup/ramp cap.
- Implemented in `python/src/tcg.h` within `current_episode_ticks_limit(...)`.

4) Success metrics for timeout-vs-progress
- Existing terminal mix metrics retained and emphasized:
  - `azk_timeout_truncation`, `azk_gameover_terminal`, `azk_winner_terminal`, `win`
- Added direct chosen-action metrics (from item #1) to support diagnosis of NOOP-heavy behavior.

Validation smoke:
- Rebuilt native module:
  - `cmake --build build-312 --target azuki_puffer_env -j`
- Serial smoke run passed (sandbox blocks multiprocessing `/dev/shm` here):
  - New selected-action metrics visible in logs (`azk_noop_selected_rate`, `azk_play_selected_rate`, etc.)
  - Debug integrity metrics still present (`azk_dbg_*`)
  - Curriculum cap showed long-episode injection behavior (`azk_curriculum_episode_cap` observed at injected long cap)

Suggested next W&B run config:
- Keep obs debug enabled while validating reward/curriculum effects.
- Start with 1m-3m step run and monitor:
  - timeout truncation down
  - winner/gameover terminals up
  - NOOP selected rate down (especially when legal alternatives exist)

## W&B run audit: `w47rtigz` (shaping_v1, 2026-02-12)

Run link:
- https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/w47rtigz

Run summary:
- state: `finished`
- steps: `3,018,240`
- epochs: `131`
- SPS: `~3528`

Terminal/outcome signals:
- `azk_timeout_truncation = 1.0` (both players)
- `azk_gameover_terminal = 0.0`
- `azk_winner_terminal = 0.0`
- `win = 0.0`
- `azk_curriculum_episode_cap = 250` at run end

Action-choice diagnostics (selected action rates):
- p0: noop `~0.552`, attack `~0.082`, play `~0.277`, ability `~0.0069`, target `~0.082`
- p1: noop `~0.434`, attack `~0.064`, play `~0.231`, ability `~0.0069`, target `~0.264`
- Interpretation: policy is not stuck in pure NOOP; it is selecting non-NOOP actions regularly.

Observation/mask debug checks:
- `azk_dbg_legal_primary_mismatch_rate = 0.0` (both)
- `azk_dbg_obs_signature_unique_ratio ~0.046` (non-zero)
- `azk_dbg_legal_action_count_mean` non-zero (`~3.75` p0, `~4.83` p1)

Optimization/losses snapshot:
- `losses/policy_loss = 0.0547` (positive this run snapshot)
- `losses/value_loss = 0.00107`
- `losses/approx_kl = 0.00661`, `clipfrac = 0.0904`
- These indicate active optimization, but not yet translating into terminal-game completion.

Conclusion:
- There is some behavioral movement (non-NOOP action mix), but still no evidence of progress to natural game completion.
- The dominant bottleneck remains timeout-only trajectories and sparse terminal outcomes.

## Next-step recommendation after `w47rtigz` (2026-02-12)

Decision:
- Do **not** only run longer with the exact same settings.
- Use a mixed strategy: modest shaping/curriculum adjustments + longer training horizon.

Why:
- `w47rtigz` shows non-NOOP behavior and healthy obs/mask integrity, but still:
  - `timeout_truncation=1.0`
  - `winner_terminal=0.0`
  - `gameover_terminal=0.0`
- This means optimization is active, but terminal learning signal is still too sparse.

Recommended immediate changes before next long run:
1. Increase long-episode exposure:
- `AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_EVERY=3`
- `AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_CAP=1800` (or 2000)
2. Increase terminal pressure slightly:
- `AZK_REWARD_NOOP_PENALTY=0.03` (from 0.02)
- `AZK_TRUNCATION_BOARD_EDGE_WEIGHT=0.4` (from 0.3)
3. Keep leader/board dense terms but avoid over-scaling:
- `AZK_REWARD_LEADER_DELTA_WEIGHT=1.5`
- `AZK_REWARD_BOARD_DELTA_WEIGHT=0.5`

Training horizon guidance:
- Next run should be at least `10M` steps with this setup before deciding it fails.
- If terminal metrics remain exactly zero by ~`5M` under increased long episodes, prioritize stronger curriculum intervention over pure step scaling.

External PPO scale context (for expectation-setting):
- PPO in common benchmarks often uses millions to tens of millions of steps even for moderate tasks.
- SB3 docs reference PPO PyBullet benchmarks at around `2M` steps.
- Atari-style PPO implementations commonly run around `10M` timesteps as a baseline scale.
- Very complex self-play domains can require orders of magnitude more experience (e.g., OpenAI Five scaled PPO over months of training).

Practical takeaway for Azuki:
- Given partial observability, long horizons, and combinatorial action space, expecting first clear terminal-game competence around low single-digit millions is optimistic; mid/high eight-figure step budgets may be realistic unless shaping/curriculum sharply improves signal density.

## Command + checkpoint note (2026-02-12)

Next recommended long run command (10M, shaping/curriculum v1.1):
- Use from repo root (`/workspace`) with your valid `WANDB_API_KEY`.
- See assistant response for full multiline command.

Checkpoint/resume status in current code:
- Current custom trainer entrypoint (`python/src/train.py`) always constructs a fresh policy via `build_policy(...)` and does not load a checkpoint before training.
- Therefore, there is no true in-place "resume optimizer state + epoch" flow right now from this script.
- PufferLib has generic `--load-model-path` handling in `pufferl.py`, but this custom training wrapper is not wired to apply it for training restarts yet.
- Current practical behavior: start a new run (fresh optimizer state). To support warm-start resume, `train.py`/`build_policy` would need explicit checkpoint load wiring.

## Resume CLI support added (2026-02-12)

Implemented in `python/src/train.py`:
- `--resume-checkpoint <path>`
  - accepts either a checkpoint file (`model_*.pt`) or checkpoint directory.
  - if directory is passed, newest `model_*.pt` is auto-selected.
- `--resume-load-optimizer`
  - also restores optimizer + `global_step` + `epoch` from `trainer_state.pt` when present.
- `--resume-strict`
  - strict key matching for model state_dict loading.

Behavior details:
- Model weights are loaded before trainer construction.
- Optimizer/state restoration (optional) occurs after trainer construction.
- If resuming optimizer/global_step, `train.total-timesteps` is interpreted as the absolute target in current run process:
  - example: if checkpoint had `global_step=3,018,240` and you want +10M more, set `train.total-timesteps` to about `13,018,240` (or higher aligned boundary).

Practical usage:
- Warm-start only (new optimizer dynamics): use `--resume-checkpoint` only.
- Full resume-like continuation (model + optimizer + step counters): add `--resume-load-optimizer`.

## Mid-run check: `34oh1onj` (~halfway, 2026-02-12)

Status at check:
- state: `running`
- step: `5,391,360`
- epoch: `234`
- SPS: `~4013`

Terminal progress:
- `azk_timeout_truncation = 1.0`
- `azk_gameover_terminal = 0.0`
- `azk_winner_terminal = 0.0`
- `win = 0.0`

Action-choice mix (not pure NOOP):
- p0 noop/play/attack/target/ability: `~0.582 / 0.270 / 0.066 / 0.074 / 0.0075`
- p1 noop/play/attack/target/ability: `~0.468 / 0.226 / 0.046 / 0.256 / 0.0047`

Integrity checks remain healthy:
- `azk_dbg_legal_primary_mismatch_rate = 0`
- non-zero obs variability metrics and legal-action means.

Interpretation:
- Optimization is active and behavior is not all-NOOP, but no midpoint evidence yet of natural game completion.

## W&B run audit: `34oh1onj` final (shaping_v1_1, 2026-02-12)

Run link:
- https://wandb.ai/heyskylark-self-affiliated/azuki_tcg/runs/34oh1onj

Summary:
- state: `finished`
- steps: `10,022,400`
- epochs: `435`
- SPS: `~3795`

Terminal outcomes:
- `azk_timeout_truncation = 1.0`
- `azk_gameover_terminal = 0.0`
- `azk_winner_terminal = 0.0`
- `win = 0.0`
- `azk_curriculum_episode_cap = 251` at end

Action-choice diagnostics:
- p0 noop/play/attack/target/ability: `~0.585 / 0.257 / 0.076 / 0.078 / 0.005`
- p1 noop/play/attack/target/ability: `~0.452 / 0.251 / 0.047 / 0.245 / 0.005`
- Not all-NOOP, but still high NOOP share for p0.

Integrity checks:
- `azk_dbg_legal_primary_mismatch_rate = 0.0`
- non-zero obs variability and legal-action metrics persist.

Compared to prior 10M shaping run (`w47rtigz`):
- No terminal breakthrough yet (still timeout-only).
- Slight policy-loss improvement but no objective improvement on game completion.
- Throughput remained strong.

Conclusion:
- Current shaping/curriculum still fails to produce natural terminal outcomes.
- Next iteration should increase long-episode exposure materially (higher cap + more frequent long episodes) and add stronger anti-timeout pressure.

## Curriculum + timeout shaping update (2026-02-12)

Implemented in `python/src/tcg.h`:
- Stronger timeout penalties and edge weighting at truncation:
  - `TRUNCATION_TIMEOUT_PENALTY`: `0.20 -> 0.35`
  - `TRUNCATION_AUTO_TICK_PENALTY`: `0.35 -> 0.60`
  - `TRUNCATION_LEADER_EDGE_WEIGHT`: `1.0 -> 1.25`
  - `TRUNCATION_BOARD_EDGE_WEIGHT`: `0.25 -> 0.45`
- Stronger anti-idle shaping:
  - `SHAPED_NOOP_PENALTY`: `0.01 -> 0.02`
- More terminal-exposure pressure when curriculum is enabled (`AZK_MAX_TICKS_CURRICULUM=1`):
  - default `AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES`: `5000 -> 3000`
  - default `AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_EVERY`: `0 -> 8`
  - default long episode cap now auto-expands to `max(final_cap + 400, 1600)` unless overridden by env var.

Action telemetry now tracked per player and exported to puffer user stats:
- `azk_attach_weapon_from_hand_selected_rate`
- `azk_play_spell_from_hand_selected_rate`
- `azk_activate_garden_or_leader_ability_selected_rate`
- `azk_activate_alley_ability_selected_rate`
- `azk_gate_portal_selected_rate`
- `azk_play_entity_to_alley_selected_rate`
- `azk_play_entity_to_garden_selected_rate`

Validation:
- Rebuilt `azuki_puffer_env` in `build-312`.
- Serial smoke with forced short episodes and curriculum completed without regressions.
- Confirmed new per-action user stats and curriculum cap stats appear in epoch logs.

Recommended stronger curriculum run command:
`AZK_MAX_AUTO_TICKS_PER_STEP=20000 AZK_MAX_TICKS_PER_EPISODE=1200 AZK_MAX_TICKS_CURRICULUM=1 AZK_MAX_TICKS_CURRICULUM_INITIAL=220 AZK_MAX_TICKS_CURRICULUM_FINAL=1200 AZK_MAX_TICKS_CURRICULUM_WARMUP_EPISODES=0 AZK_MAX_TICKS_CURRICULUM_RAMP_EPISODES=12000 AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_EVERY=6 AZK_MAX_TICKS_CURRICULUM_LONG_EPISODE_CAP=2000 PYTHONPATH=build-312/python/src:python/src:$PYTHONPATH uv run --active python python/src/train.py --config python/config/azuki_speed_3090_parallel.ini --wandb --wandb-project azuki_tcg --wandb-group rtx3090 --train.total-timesteps 10000000 --train.name azuki_3090_parallel_10m_curriculum_v2 --tag None`

## Run audit + engine rule check (2026-02-12, run `w61h0vnf`)

W&B snapshot (while run still active):
- Run id: `w61h0vnf` (display `morning-river-19`), state `running`.
- Latest summary still shows timeout-only episodes:
  - `environment/0/azk_timeout_truncation=1`
  - `environment/0/azk_gameover_terminal=0`
  - `environment/0/azk_winner_terminal=0`
- Curriculum cap currently logged near `221`, with episode length also `221`.
- Action profile at latest point is asymmetric:
  - p0 `azk_attack_selected_rate ~0.1667`, `noop ~0.5`
  - p1 `azk_attack_selected_rate ~0.0`, `noop ~0.9767`

Interpretation:
- There may be occasional attack upticks, but no terminal conversion yet; behavior remains timeout-dominated.
- The high seat asymmetry suggests policy-collapse-like behavior in one seat at least for current episodes.

Important engine consistency finding (deck-out):
- `draw_cards_with_deckout_check(...)` enforces immediate deck-out loss when deck reaches zero after draw (used by several abilities).
- Start-of-turn draw path in `src/systems/start_phase.c` currently uses `move_cards_to_zone(...)`, which only loses when draw fails from an already-empty deck.
- Therefore current engine behavior is inconsistent:
  - ability draw: immediate deck-out on reaching 0
  - normal draw: deck-out on next draw attempt

Implication for "can it run forever":
- In training, episodes cannot run forever when `AZK_MAX_TICKS_PER_EPISODE` is set (they truncate).
- Engine-side terminal by deck-out is currently not consistently immediate across code paths; this should be unified for rules correctness.

## Resume fix for PyTorch 2.6+ `weights_only` default (2026-02-12)

Issue:
- Resuming with `--resume-load-optimizer` failed on `trainer_state.pt` with:
  - `_pickle.UnpicklingError: Weights only load failed...`
- Cause: PyTorch 2.6+ changed `torch.load(..., weights_only=True)` by default.
  `trainer_state.pt` can contain non-tensor Python objects (e.g. `defaultdict`) from optimizer state.

Fix:
- Updated `python/src/train.py` in `_maybe_restore_trainer_state(...)` to load with:
  - `torch.load(trainer_state_path, map_location="cpu", weights_only=False)`

Scope:
- This affects only optimizer/global_step resume state loading.
- Model weight loading path remains unchanged.

## Run diagnosis: `jilrabie` (2026-02-12)

W&B findings:
- Run finished at ~25.0M agent steps.
- `learning_rate=0` in summary/history.
- Episodes remain timeout-capped:
  - `environment/{0,1}/azk_episode_length=2000`
  - `environment/{0,1}/azk_timeout_truncation=1`
  - `environment/{0,1}/azk_gameover_terminal=0`
  - `environment/{0,1}/azk_winner_terminal=0`

Interpretation:
- With LR at 0, PPO updates effectively stop; this can fully explain "no learning" behavior despite long runtime.

NOOP legality audit:
- `ABILITY_PHASE_COST_SELECTION`: NOOP is not enumerated (correct).
- `ABILITY_PHASE_EFFECT_SELECTION`: NOOP is only enumerated when `effect_min == 0` (correct).
- `ABILITY_PHASE_SELECTION_PICK`: NOOP is currently always enumerated in mask.
  - This appears intentional for "up to" style picks, but currently unconditional in enumerator.
  - Ability resolution accepts NOOP in SELECTION_PICK and routes to `azk_process_skip_selection`.

Action-head / stuck concerns:
- No direct evidence from this run of invalid-action crashes; training completed.
- The stronger blocker visible in run metrics is LR=0 + timeout-dominated trajectories.

Deck-out concern:
- 0 deckouts in this run is plausible with decision-tick cap=2000 and nontrivial turn progression/card recycling.
- Not by itself proof of broken terminal handling.
