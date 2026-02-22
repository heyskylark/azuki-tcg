# Python Training Debug Ledger

Last updated: 2026-02-17

## Goal
Eliminate resume/eval collapse and prove long-run resume stability for Azuki training.

## Environment
- CUDA confirmed in sandbox: `RTX 3090`, driver `590.48.01`.

## Root Symptoms (legacy checkpoints)
- Resume/eval collapse appears immediately after loading old checkpoints:
  - `entropy ~ 1e-5` to `1e-6`
  - `noop_selected_rate ~ 0.988-0.996`
- Reproduced across multiple legacy checkpoints:
  - `000217` (~5m): entropy `0.1401`, noop `0.9733`
  - `000434` (~10m): entropy `0.00636`, noop `0.9957`
  - `001085` (~25m): entropy `4.07e-06`, noop `0.9886`
  - `002170` (~50m): entropy `9.06e-06`, noop `0.9886`
- Changing `AZK_RESUME_COMPLETED_EPISODES` (`0` to `500000`) did not change collapse.

## Fixes Implemented
- `python/src/train.py`
  - Added source hash fingerprinting in `resume_config_fingerprint` for:
    - `python/src/policy/tcg_policy.py`
    - `python/src/policy/tcg_sampler.py`
    - `python/src/tcg.h`
    - `python/src/train.py`
    - `python/src/training_utils.py`
  - Persisted/validated source hashes during resume.
  - Fixed scheduler/optimizer restore interaction:
    - save/load `scheduler_state_dict`
    - do not overwrite restored optimizer LR from scheduler
    - maintain scheduler `_last_lr` from optimizer groups when possible.
- `python/src/evaluate_checkpoint.py`
  - Switched to same load/validation path as training:
    - `_compute_runtime_fingerprint`
    - `_resume_config_fingerprint`
    - `_validate_resume_metadata`
    - `_load_model_weights`

## Validation Experiments (post-fix)
1. Long base run (new checkpoint format)
- Run: `experiments/azuki_local_177129839783`
- Command target: 108 epochs (`2,488,320` steps)
- Produced: `model_azuki_local_000108.pt`
- Metadata includes `resume_config_fingerprint.source_hashes` (count `5`).

2. Resume continuation from epoch 108
- Resume from:
  - `experiments/azuki_local_177129839783/model_azuki_local_000108.pt`
  - `--resume-load-optimizer --no-resume-auto-reset-critic`
- Continued to:
  - run `experiments/azuki_local_177129915579`
  - final checkpoint `model_azuki_local_000135.pt` at `3,110,400` steps
- Resume health signals:
  - reset-start probe: entropy `0.988040`, noop `0.617801` (non-collapsed)
  - rollout health: entropy `0.741495`, noop `0.666580` (non-collapsed)
  - steady epochs 109-135: entropy ~`0.89-0.95`, noop generally ~`0.22-0.25`
  - no `[resume] WARNING: collapse-to-noop` messages.
- Trainer restore confirmed:
  - `scheduler_restored=True`
  - `optimizer_lrs=[0.0]` (expected for resumed schedule state).

3. Repeatability check: resume again from epoch 135
- Resume from:
  - `experiments/azuki_local_177129915579/model_azuki_local_000135.pt`
  - target `3,133,440` steps (epoch 136)
- Output run: `experiments/azuki_local_177129940942`
- Health signals:
  - reset-start probe: entropy `0.771816`, noop `0.605042`
  - rollout health: entropy `0.694745`, noop `0.648481`
  - completed `epoch 136` and wrote `model_azuki_local_000136.pt`
- Confirms resume stability is repeatable on checkpoints produced after fix.

## Current Conclusion
- Long resume path (108 -> 135 epochs) is now stable and does not collapse.
- Legacy checkpoints from pre-fix era still collapse, but new checkpoints with source-hash fingerprints + corrected scheduler restore resume correctly.

## New: Configurable Sampler Anneal (2026-02-17)
- Implemented runtime-configurable sampler params in `python/src/policy/tcg_sampler.py`:
  - `set_sampling_params(...)`
  - `get_sampling_params()`
  - Defaults preserved (`subaction_temperature=1.2`, `smoothing_eps=0.05`).
- Added linear anneal scheduling in `python/src/train.py` based on `global_step`:
  - supports start/final values and start/end points (fraction or absolute step),
  - applies continuously every epoch,
  - survives resume using restored `global_step`,
  - includes checkpoint-step anneal offset when optimizer/global_step restore is skipped,
  - logs metrics into trainer stats:
    - `sampler/subaction_temperature`
    - `sampler/smoothing_eps`
    - `anneal/ent_coef`
- Added matching sampler setting in `python/src/evaluate_checkpoint.py` using checkpoint `global_step`.

### New config keys (`[policy]`)
- `subaction_temperature_initial`
- `subaction_temperature_final`
- `subaction_temperature_anneal_start_frac`
- `subaction_temperature_anneal_end_frac`
- `subaction_temperature_anneal_start_step` (optional override)
- `subaction_temperature_anneal_end_step` (optional override)
- `smoothing_eps_initial`
- `smoothing_eps_final`
- `smoothing_eps_anneal_start_frac`
- `smoothing_eps_anneal_end_frac`
- `smoothing_eps_anneal_start_step` (optional override)
- `smoothing_eps_anneal_end_step` (optional override)

### New config keys (`[train]`)
- `ent_coef_anneal_initial`
- `ent_coef_anneal_final`
- `ent_coef_anneal_start_frac`
- `ent_coef_anneal_end_frac`
- `ent_coef_anneal_start_step` (optional override)
- `ent_coef_anneal_end_step` (optional override)

### Verification
- Smoke run with override:
  - `subaction_temperature: 1.2 -> 1.0` over `46080` steps
  - `smoothing_eps: 0.05 -> 0.0` over `46080` steps
  - observed epoch stats:
    - epoch 1: `temp=1.2`, `smoothing=0.05`
    - epoch 2: `temp=1.1`, `smoothing=0.025`
- Resume smoke from epoch-1 checkpoint restored anneal position:
  - startup applied step `23040`
  - `temp=1.1`, `smoothing=0.025` immediately after resume.
- Entropy-coef anneal smoke:
  - configured `ent_coef: 0.01 -> 0.0` over `46080` steps
  - observed:
    - epoch 1: `anneal/ent_coef = 0.01`
    - epoch 2: `anneal/ent_coef = 0.005`
  - resume from epoch-1 checkpoint restored `ent_coef=0.005` at `global_step=23040`.
- No-optimizer-resume anneal offset probe (CPU tiny run):
  - resume without `--resume-load-optimizer` from checkpoint at `global_step=128`,
  - trainer printed `anneal_offset=128`,
  - epoch 1 used resumed midpoint values (`temp=1.1`, `smoothing=0.025`, `ent_coef=0.005`),
  - epoch 2 reached final values (`temp=1.0`, `smoothing=0.0`, `ent_coef=0.0`).

## Operational Guidance
- Treat pre-hash checkpoints as potentially incompatible with current runtime.
- Use checkpoints generated after these fixes for tiered training / entropy-anneal experiments.

## New: League Done-Row Index Bug Fix (2026-02-17)
- Symptom:
  - League-enabled runs could crash early in `evaluate()` with:
    - `IndexError: The shape of the mask [2] ... does not match ... [1440, 4096]`
  - This could happen even when `league/active=0` and `activate_after_steps` was large (e.g. `50m`), because the league trainer class still executes its RNN done-state reset path.
- Root cause:
  - `LeaguePuffeRL._zero_done_states()` treated `done_rows` as a boolean mask:
    - `torch.as_tensor(done_rows, dtype=torch.bool)`
  - But caller passes row ids (`env_id_np[done_mask]`), not a full boolean mask.
  - Casting row ids to bool produced a short mask and shape mismatch.
- Fix:
  - Updated `python/src/league_training.py`:
    - convert `done_rows` to `int64` row indices,
    - filter to valid bounds `[0, total_agents)`,
    - `np.unique` to deduplicate,
    - index LSTM state tensors with `torch.long` row ids.
- Tests:
  - Added regression test in `python/tests/test_league_training_utils.py`:
    - `test_zero_done_states_uses_row_indices`
    - verifies indexed zeroing works and ignores invalid rows.
  - Ran:
    - `PYTHONPATH=python/src python -m unittest -v python/tests/test_league_training_utils.py` (pass)
- Smoke validation:
  - Ran no-W&B CUDA smoke with league enabled and delayed activation:
    - `--league-enable --league-activate-after-steps 50000000`
    - reached epoch 6 (`138,240` steps) without crash.
    - `environment/league/active` remained `0.0` throughout (expected pre-50m).
