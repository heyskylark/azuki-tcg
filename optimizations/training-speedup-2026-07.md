# E2E training speedup: 574 → ~5,000+ SPS (2026-07-05)

Goal: ≥3-5× end-to-end PPO training throughput vs the recorded baseline
(`azuki_speed_3090.ini`, 480 envs / 12 workers, RTX 3090 + 12-core CPU +
128 GB RAM; steady-state ≈ 574 SPS median recorded 2026-07-03, ≈ 607 SPS
re-measured fresh on 2026-07-05 before these changes).

Result: **5,387 SPS steady-state (median-of-medians over 3 fresh 100k runs)
= 9.4× vs 574 / 8.9× vs 607; 4,899 SPS sustained median over a full
1M-step run (8.5× / 8.1×). The single worst post-warmup epoch observed in
any final run (4,921) is still 8.1× the fresh baseline.**

## Where the time actually went (measured, not assumed)

Fresh-baseline phase split (perf counters, 100k run): learn 60%,
env-wait 21%, rollout-forward 14%, train-forward 3.6%. Three pathologies
accounted for nearly all of it:

1. **`ScalarRunningNorm` synced GPU→CPU on every call** (`.cpu().numpy()`
   into gymnasium's `RunningMeanStd`, then re-registered buffers on CPU,
   forcing H2D on every later use). It is called ~90×/forward — including
   once per board slot via the weapons loop — and the policy stays in train
   mode during rollout, so this fired every rollout step too.
2. **Per-slot Python loops in `encode_observations`** (50-slot zones ×
   ~15 fields each as separate tensor ops + a per-slot weapons sub-encode)
   produced thousands of tiny autograd nodes; backward ran 17× the cost of
   forward.
3. **The Python observation path**: C wrote a packed struct, Python
   converted it to dicts (ctypes getattr per field), PettingZoo AEC→parallel
   conversion, then pufferlib `emulate()` re-flattened the dicts into a
   24,388-byte int32-heavy layout. 12.4× the cost of the raw C step, in
   every worker.

After fixing 1–3, the torch profiler exposed the real killer that had been
hiding under everything: **`aten::index_put_` backward (indexing_backward_kernel)
= 82.8% of an entire epoch** — `_text_feature_table()[idx]` advanced-indexed a
grad-requiring [vocab, 56] table with up to ~5.7M indices per minibatch
(weapons sub-zones), 72 calls × 200 ms.

## The changes

### 1. Native observation path (python/src/azk_native.py)
- `AzukiNativeEnv(PufferEnv)` drives the existing C `binding.vec_init/vec_step/
  vec_log` API. C writes packed `TrainingObservationData` structs directly
  into pufferlib's shared-memory rows — zero copies, no PettingZoo, no dicts,
  no per-step Python beyond one `vec_step` call and an episode-end `vec_log`.
- Layout identity: per-agent rows of 9,416 bytes viewed as
  (num_envs, 2×9,416) are exactly the per-env struct pairs C writes.
- `vec_step` in `env_binding.h` now auto-resets on truncation-ended episodes
  too (it previously only checked terminals).
- Config: `env.native = true`, `env.native_envs_per_instance = N`,
  `vec.num_envs = num_workers` (one native vec-env object per worker).
- Reward components and win labels no longer travel through info dicts.
  The env guarantees rewards carry only the terminal component on
  episode-end steps and only the shaped component otherwise, so the trainer
  derives both, plus win-prob labels (`terminal & reward>0`; truncation → 0),
  from (rewards, terminals, truncations) as pure GPU ops.

### 2. Policy decode/encode vectorization (policy/v2/tcg_policy.py)
- All observation layouts are normalized into one canonical tree: each zone
  is a dict of whole-zone field tensors (B, S) ((B, S, W) for weapons) built
  either from strided `as_strided` views over the packed struct (native) or
  one stack per field (legacy dtype, deck-building path).
- Every `_encode_*` consumes (B, S) tensors; the per-slot loops and the
  per-slot weapons sub-encodes are gone (one weapons call per zone).
- `ScalarRunningNorm` is fully GPU-resident (float64 device buffers, Chan
  parallel combine — same math as RunningMeanStd, no host syncs, same buffer
  names for checkpoint compat).
- **Metadata embedding table**: per-card metadata embeddings are computed
  once per forward over the ~vocab-size card list; per-occurrence lookups use
  `nn.functional.embedding` (optimized scatter backward). Occurrence-weighted
  running-norm stats are preserved via a no-grad update pass. This removed
  the 82.8% indexing_backward wall.
- Fixed latent bug: legacy zone slots were iterated in *string-sorted* order
  ('f0','f1','f10','f11',...,'f2'), scrambling slot order in the reference
  matrices used by the legal-action scorer. Canonicalization preserves true
  dtype order.
- Bit-exact equivalence between native-packed and legacy-emulated paths was
  verified (encodings, all context matrices, values, logits, logprobs,
  entropy: max|diff| = 0.0 with identical weights).

### 3. Trainer hot-loop (azk_puffer/trainer.py)
- evaluate(): no per-step `.cpu()`/`.item()`-style syncs beyond the two
  existing ep-index reads; reward components + win-prob episode bookkeeping
  as device tensor ops (episode-id table + one gather at epoch end).
- train(): losses/metrics accumulate as device tensors (single host sync per
  epoch); win-prob aux is masked BCE without data-dependent branches; AMP
  context properly scoped around forward+loss (previously `__enter__` was
  called twice per minibatch and never exited, leaving bf16 autocast on for
  everything including the optimizer); upstream advantage-kernel import
  hoisted out of the per-minibatch path.

## Benchmark results

Protocol: `train.py --config python/config/azuki_native_3090.ini`
(hyperparameters identical to `azuki_speed_3090.ini`: 480 C envs / 960
agents, batch auto=15,360, bptt 16, update_epochs 2, minibatch 8,192, muon,
bf16), 100k steps, idle box, steady-state = median epoch SPS excluding the
warm-up epoch. Both-seat step accounting, same as the baseline measurement.

| Run | Epoch SPS (post-warmup) | Median |
|---|---|---|
| Baseline (recorded 2026-07-03) | 577 / 593 / 423 / 574 / 572 | **574** |
| Baseline (fresh re-run 2026-07-05, pre-change) | 614 / 607 / 462 / 607 / 616 | **607** |
| Final bench run 1 (100k) | 5,659 / 5,600 / 5,457 / 5,133 / 4,921 | **5,457** |
| Final bench run 2 (100k) | 5,669 / 5,346 / 5,398 / 5,091 / 5,013 | **5,346** |
| Final bench run 3 (100k) | 5,682 / 5,684 / 5,387 / 5,092 / 4,974 | **5,387** |
| **Final (median of 3 medians)** | | **5,387 (9.4× / 8.9×)** |
| Native 1M validation (sustained, 65 epochs) | 4,314–5,132 | **4,899** |

Attribution datapoint: the *legacy* env path (PettingZoo + emulate) run with
the new policy/trainer code reaches 1,744 SPS — i.e. the policy/trainer
fixes alone are worth ≈2.9× even without the native obs path; the native
path multiplies that to ≈8.9×.

Phase totals (profiled-epoch counters, 100k): eval 45.2s → 5.1s
(env-wait 26.6 → 1.4, rollout-forward 17.5 → 3.6), train 81.1 → 9.6s
(learn 76.5 → 7.2).

## Validation

- **Bit-exact equivalence test** (`python/src/test_native_obs_equivalence.py`,
  passes under pytest): identical weights fed the same game states through
  the packed-native and legacy-emulated layouts produce max|diff| = 0.0 on
  encodings, all action-context matrices, values, legal-action logits,
  logprobs and entropy.
- **1M-step A/B at matched steps** (native env vs legacy env, both on the
  new policy/trainer):

  | steps | entropy (native / legacy) | explained_var (native / legacy) |
  |---|---|---|
  | 150k | 0.540 / 0.494 | 0.522 / 0.496 |
  | 300k | 0.425 / 0.372 | 0.663 / 0.616 |
  | 500k | 0.350 / 0.292 | 0.690 / 0.665 |
  | 750k | 0.304 / 0.250 | 0.694 / 0.676 |
  | 1M   | 0.294 / 0.262 | 0.706 / 0.700 |

  Win-prob accuracy 0.91+ on both; self-play winrate ≈ 0.45–0.55; 100%
  winner-terminated episodes; zero truncations; **zero illegal actions**
  (the C engine aborts the process on any illegal submit, so both runs
  completing is a hard mask-correctness check over ~10⁶ steps each).
- C unit tests pass (ctest 2/2) after the `env_binding.h` change.
- Deck-building smoke (`azuki_deckbuild_smoke.ini`, legacy PettingZoo path +
  deck_context through the rewritten encoders): completes clean, losses sane.

## Notes / follow-ups

- The deck-building path still uses the legacy PettingZoo pipeline (it now
  benefits from the vectorized encoders via the legacy canonicalizer, but
  not from the native env).
- torch.compile / CUDA-graphs and rollout/learn overlap were left on the
  table; at ~5k SPS the remaining split is ≈ learn 7s / eval-forward 3.6s /
  train-forward 2.3s per profiled window, so another ~1.5-2× is plausibly
  available if ever needed.
