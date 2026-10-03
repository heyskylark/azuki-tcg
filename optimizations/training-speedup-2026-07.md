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

# Round 2: 5,387 → 7,148 SPS (Muon) / 8,631 (Adam option) — 2026-07-05/06

**Result: 7,148 SPS median-of-medians over 3× 200k runs on the shipped
config (1.31-1.33× this round; 12.5× vs the original 574), with training
semantics unchanged up to bf16/compile numerics: same optimizer, same
sample/opt-step-per-env-step ratios, bit-exact eager encodings, zero illegal
actions across all runs. Per-run steady medians: 7,058 / 7,199 / 7,148;
losses at 200k: entropy 0.53±0.01, explained_var 0.34-0.40, win-prob acc
0.74-0.85 — in line with the round-1 A/B trajectories.**

**The 7.5-8k target is crossed only via the flagged optimizer option: fused
Adam on the same stack measured 8,631 SPS, but needs an lr retune + quality
revalidation (the probe at Muon's lr 0.015 collapses entropy). That decision
is left open deliberately.**

Protocol note: steady-state = median of epochs >3k SPS over a 200k run
(compile+graph-capture warmup consumes the first ~4 epochs; the original
100k protocol leaves too few clean epochs under a compiled config).

Working log below; single-run medians (post-warmup epochs) unless noted.
Baseline re-measured after instrumenting the trainer to split `learn` into
`learn_backward` / `learn_opt` (profile + dashboard now show both).

Profiled-window split at baseline (~14.8s): learn_backward 4.29, learn_opt
3.15 (3 Muon/Newton-Schulz steps/epoch ≈ 210ms each), eval_forward 3.45,
train_forward 2.33, env 1.43.

| Config | Median SPS | Notes |
|---|---|---|
| A: baseline (a06b432, mb 8192) | **5,450** | matches recorded 5,387-5,457 |
| B2: mb 15360 = 2x7680 accum | 4,932 | 2 opt steps but int-truncation quirk means +25% fwd/bwd samples -> net loss |
| C: mb 12288 (2 chunks, no accum) | **5,829** | same 24,576 samples as baseline, 2 opt steps (learn_opt 3.15->2.06); needs expandable_segments |
| D: C + de-synced forward | **5,949** | removed 21 `.any()` gates + `_ensure_valid_mask` branch + trim `.item()` moved early + trainer ep-index mirrors; bit-exact test passes |
| E: D + compile encode/decode (mode=default) | 6,191 | recompile_limit=8 caps the pathological frames to eager fast; lucky window (see E4) |
| E3: E + recompile_limit=64 (200k) | 5,528 | WORSE: letting dynamo recompile `_PackedField.extract` per field spec (~100 specs) burns minutes and regresses eval_forward 3.7->4.5/window |
| F2: compile TCGLSTM.forward + dynamo-disabled canonicalizer | 1,700 | catastrophic: disable() inside the compiled train graph re-processes per call |
| E4: E structure re-confirmed (200k) | 5,643 | high variance (5,012-6,644), mid-run recompile stalls; train side wins (-0.7s bwd) but rollout guard overhead (+0.3s) eats it |
| G: 960 envs (update_epochs 1.6 to hold sample:env ratio) | OOM | bucket-1024 decode transient at batch 1920 collides with train peaks on 24GB |
| G2: 720 envs | OOM | same (CUBLAS alloc) — 480 envs is the memory-optimal point on this card |
| H3: D + manual CUDA-graph rollout | **6,579** | eval_forward 3.74->1.92s/window; one graph per trim bucket, private pools, autocast cache_enabled=False during capture (load-bearing: cached bf16 weight casts get freed + go stale) |
| I2: H3 + train-side compile (encode/decode roots; capture swaps in eager originals) | **6,826** | backward 4.34->3.61, train_forward 2.37->2.02/window; dynamo guards read the CUDA RNG seed, so capture must record the eager fns |
| K: I2 + vec.batch_size 6 (double-buffer) | **7,193** | env wait 1.50->0.38/window (two 480-agent groups alternate); rollout graphs recapture at batch 480 |
| J: K + fused Adam (SPS probe only) | 8,631 | learn_opt 2.05->0.09/window; NOT quality-neutral: lr 0.015 is Muon-tuned, entropy collapses — adopting Adam = retune + revalidation decision |
| L: K + sampling captured in-graph | 7,032 | no measurable win vs K (within noise); kept — fewer launches, params-snapshot guard for anneal configs |
| **FINAL: shipped config, 3x200k** | **7,058 / 7,199 / 7,148 -> 7,148** | Muon, quality-neutral; window split ≈ bwd 3.6 / eval_fwd 2.5 / opt 2.05 / train_fwd 2.0 / env 0.38 |

Post-stack ceiling notes: the remaining window is GPU compute (LSTM GEMMs +
NS + backward), not launch overhead — the eval forward now costs ~2.5s/window
of real math at the same total FLOPs. The next honest levers are the ones the
round-1 report named as design decisions: the optimizer (Adam = 8.6k
measured) or shrinking the 4096-hidden LSTM.

Key findings so far:
- `total_minibatches = int(update_epochs*batch/chunk)` truncates: the recorded
  baseline actually consumes 24,576 of 30,720 sample-passes per epoch
  ("1.6 update epochs"). Any accumulation config that hits the full 30,720
  loses more on backward than it saves on optimizer steps. minibatch 12288
  divides exactly into 2 chunks = baseline sample count with 2 opt steps.
- `azuki_native_3090.ini` had an inert `min_batch_size` key (real knob is
  `train.minibatch_size`).
- Muon NS already runs in bf16 via heavyball stochastic_round; NS on the
  (16384,4096) LSTM hh matrix alone is ~6.2 TFLOP/step -> ~130ms of the
  ~210ms step. thinky_polar_express only applies to square matrices.
- 2 optimizer steps/epoch is the floor: an effective minibatch of 24,576 via
  accumulation is rejected by the trainer (`batch_size 15360 must be >=
  minibatch_size`) — optimizer batches cannot span epochs.
- max-autotune-no-cudagraphs never produced an epoch within ~11 min of
  compile on this 12-core box; not viable here.

## The shipped stack (config: azuki_native_3090.ini)

1. **minibatch_size 12288** — same sample-passes as 8192 (int-truncation),
   2 Muon steps/epoch instead of 3.
2. **De-synced forward** — the 21 `if mask.any()` gates in
   `_gather_legal_action_refs` compute unconditionally; `_ensure_valid_mask`
   branch-free; the trim-bucket `.item()` moved to encode start (short queue);
   trainer `ep_lengths/ep_indices` slice reads mirrored in Python ints.
   Bit-exact vs baseline (test_native_obs_equivalence passes).
3. **Manual CUDA graphs for the rollout forward** (`train.cuda_graphs`):
   `TCGLSTM.enable_rollout_cuda_graphs()` captures encode+LSTM-cell+decode
   *and sampling* per legal-action trim bucket and replays. Requirements
   discovered the hard way:
   - capture under `autocast(..., cache_enabled=False)` — otherwise the
     graph records pointers to autocast's cached bf16 weight casts, which are
     freed when the ambient context exits (illegal memory access) and would
     be stale after optimizer steps anyway;
   - one private memory pool per bucket graph — pool sharing is only safe
     when replay order matches capture order, and buckets replay in
     data-dependent order;
   - capture the *eager* encode/decode (swap the compiled ones out during
     capture) — dynamo guards read the CUDA RNG seed, illegal during capture;
   - in-graph multinomial is fine (default generator advances per replay);
     the sampler returns the graph's presampled static outputs, guarded by a
     sampling-params snapshot so anneal configs fall back to eager sampling.
4. **Train-side torch.compile** (`train.compile`, mode=default): compile
   roots at `encode_observations`/`decode_actions`. The default
   recompile_limit=8 is load-bearing (caps the per-field `_PackedField.extract`
   frames to eager fast). Whole-rollout compile is a measured loss
   (guard/wrapper overhead > fusion win at batch 960) — hence graphs for
   rollout, compile for train.
5. **vec.batch_size 6 double-buffering** — two 480-agent groups alternate;
   env stepping hides behind the other group's forward (env wait
   1.50 -> 0.38s/window).
6. **expandable_segments allocator** (set-if-unset in train.py) — removes the
   fragmentation OOM between the bucket-1024 decode transients and everything
   else.

Rollout/learn overlap (double-buffered training) was evaluated and skipped:
after CUDA graphs the rollout is no longer launch-bound, so the premise
(hiding launch gaps under learn) is gone; on one GPU both phases compete for
the same SMs, and it introduces one-batch off-policy staleness — a training
semantics change, not a pipeline optimization.

Optimizer decision left open: fused Adam on this stack measured **8,631 SPS**
(learn_opt 2.05 -> 0.09 s/window) but is NOT quality-neutral (lr 0.015 is
Muon-tuned; entropy collapsed in the probe). Adopting it = lr retune + A/B
revalidation.
