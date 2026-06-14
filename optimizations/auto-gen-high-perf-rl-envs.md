# Automatic Generation of High-Performance RL Environments

**Authors:** Seth Karten (Princeton), Rahul Dev Appapogu (Independent), Chi Jin (Princeton)
**arXiv:** 2603.12145v2 [cs.LG], 17 May 2026 · https://arxiv.org/pdf/2603.12145
**Local copy:** `optimizations/paper.pdf` (raw text: `optimizations/paper.txt`)

> Distilled markdown for fast reference. The Performance Optimization Guide (Appendix B)
> is the most directly actionable part for the azuki-tcg engine — see bottom.

---

## TL;DR

A closed-loop methodology for translating slow reference RL environments into
high-performance equivalents using coding agents, guided by **hierarchical
verification** + **cross-backend policy transfer**. Environment sim usually eats
50–90% of RL wall-clock; with performance envs it drops below 4% at 200M params.

Five case studies, target chosen by structure (**JAX** for small-state/parallel-on-GPU,
**Rust** for sequential/memory-intensive):

| Env | Source → Target | Speedup | Notes |
|-----|-----------------|---------|-------|
| EmuRust | C/Python (PyBoy) → Rust+PyO3 | 1.5× PPO | Cycle-accurate Game Boy emulator |
| PokeJAX | TypeScript (Showdown, 100K LoC) → JAX | 22,320× PPO | First GPU-parallel Pokémon battle sim |
| HalfCheetah | MuJoCo → JAX | 1.04× vs MJX, 37× vs Gym | Parity w/ hand-optimized engine |
| TCGJax | Web rules → Python → JAX | 6.6× PPO | First Pokémon TCG Pocket env; contamination control |
| Puffer Pong | C (PufferLib) → Rust+JAX | 42× PPO | Beats already-optimized C baseline via `lax.scan` fusion |

---

## Methodology: 4-level hierarchical verification

Given reference env `E_ref` in `L_src`, produce behaviorally-equivalent `E_perf` in
`L_tgt`: same obs/rewards/terminations per step for any seed+actions. Continuous
envs relax to ε-equivalence (per-component L∞ tolerance).

- **L1 — Component generation (property tests):** translate each module, verify in
  isolation. Catches dynamics bugs (e.g. mass-matrix symmetry, bias-force bounds)
  *before* they propagate.
- **L2 — Interaction tests:** verify composed modules / cross-module interactions;
  repair while preserving L1.
- **L3 — Rollout comparison:** full episodes, matched seeds + action sequences,
  per-timestep output diff (100 episodes). Discrepancies → root-cause + new L1/L2 tests.
- **L4 — Cross-backend policy transfer:** train policy in `E_perf`, eval in `E_ref`
  (and vice versa). If reward is statistically indistinguishable (TOST equivalence,
  p<0.05), there is **no sim-to-sim gap**. A detected gap feeds back to L1–L3.

Failures at any level trigger targeted repair + re-verification. Algorithm 1
formalizes this as a 4-phase loop. All reference translations used **Gemini 3 Flash
Preview** (`gemini --yolo`), but the method is agent-agnostic — re-validated with
**Claude Sonnet 4.6** (Pong) and **Claude Opus 4.6** (HalfCheetah), identical prompts.

### Key finding: hierarchical verification is *necessary* (H3)
HalfCheetah with **L3-only** failed to converge after **42 iterations** — the agent
couldn't isolate a Coriolis sign error from end-to-end rollout failures. Full
hierarchy converged in **5 iterations**. Complexity threshold lies between simple
game logic (Pong, where L3-only works) and rigid-body physics with ≥6 DOF.

---

## Why JAX as a target (relevant to a TCG)

- TCGJax: extracted rules from web → 29K-line Python reference → 4K-line JAX.
  Python ref at 23K SPS (16 procs) too slow; JAX at 153K SPS (batch 4K) converges
  to reward 1.0 in ~12 min.
- 1,370 move effects dispatched via `jax.lax.switch` → large XLA HLO graph (45s JIT
  for PokeJAX). **Every step pays for all branches** regardless of which move is
  used — known cost of branchless GPU execution.
- Pong's 42× comes from `jax.lax.scan`-fused rollouts: entire rollout compiles into
  one GPU kernel, zero CPU↔GPU transfer. C environments cannot exploit this fusion.

### Translation cost (all iterations, Gemini 3 Flash)
| Metric | EmuRust | PokeJAX | HalfCheetah | TCG | Pong |
|--------|---------|---------|-------------|-----|------|
| Target LoC | 2k | 55k | 1k | 4.2k | 235/318 |
| Modules | 5 | 30 | 5 | 11 | 1 |
| Tests | 52 | 2k | 69 | 50 | 12 |
| Agent cost | $0.43 | $6 | $3.26 | $4.98 | $0.05 |
| Iterations | 72 | 63 | 20 | 51 | 13 |

Re-translating when a reference updates costs <$1; the test suite is a regression guard.

---

## Appendix B — Performance Optimization Guide (the actionable part)

> Applied *after* an env passes L1–L3. Don't change API, simulator, or reward logic.

### B.1 JAX checklist (ordered by typical impact)
1. **Fixed-size state arrays.** No lists/dicts/variable-length — pad to max capacity
   with a sentinel (`-1` / `NO_CARD_ID`). TCG Pocket: card zones → `(MAX_HAND_SIZE,)`
   arrays, enabling full-engine JIT.
2. **Branchless conditionals with `jnp.where`.** Computes both branches, selects by
   mask → no warp divergence. Multi-way: nested `jnp.where` or `jax.lax.switch`.
   Reserve `lax.cond` for *unbatched* cases (under `vmap`, `lax.cond` runs both
   branches anyway).
3. **`vmap` for batch parallelism.** Write single-instance logic, then `jax.vmap`.
   Mark shared constants (card DBs) with `in_axes=None` to broadcast, not duplicate.
4. **JIT the outer interface.** `jax.jit` the vmapped step/reset; warm up once at init.
5. **`lax.scan` for multi-step fusion.** Fuse the rollout loop into one kernel,
   killing per-step CPU→GPU dispatch (CartPole: 3.2×).
6. **Minimize dtypes.** `int8` for categorical/flags, `float32` only for arithmetic.
7. **Pre-allocate reward/obs buffers.** Update in-place via `.at[].set()`; avoid
   `jnp.concatenate`/`stack` in the hot path.
8. **Normalize obs at the source**, inside the JIT'd step; pre-compute denominators.

### B.2 Rust checklist
1. **Rayon `par_iter_mut`** to step all envs in parallel (near-linear to physical
   cores, 8–16×). EmuRust packs 128 envs in one process via Rayon, beating PyBoy's
   one-process-per-core (zero IPC overhead).
2. **Pre-allocate obs/reward/terminal buffers** once; reuse via slice copies.
3. **`#[inline(always)]`** on step/obs/reward hot functions (profile first).
4. **Const lookup tables** for game mechanics (e.g. element effectiveness matrices).
5. **Frame-skip fast path** (skip render for intermediate frames; saved ~60% in EmuRust).
6. **`Arc<Vec<...>>`** for shared immutable data (ROMs, card DBs) — one copy regardless
   of batch.
7. **Compact struct layout:** separate hot/cold data, `i32` over `i64`, pack bools.
8. **Efficient PyO3:** zero-copy `PyReadonlyArrayN` in, write into pre-allocated NumPy
   out; one Python→Rust call for *all* envs, not per-env.

### B.3 Optimization agent prompt (verbatim skeleton)
> "The [JAX/Rust] environment has passed all verification tests. Now optimize for max
> SPS. Constraints: all L1/L2/L3 tests must keep passing; don't change external API
> (step/reset/obs/reward shapes); don't change simulator or reward logic. Apply [the
> checklist above] in order. After each optimization: run full test suite, measure SPS
> at batch sizes [32,128,512,2048,8192], report per-change speedup. Begin with a
> profiling analysis to find the current bottleneck, then target it first."

---

## Notes / caveats from the paper
- Not all envs get faster — already hand-optimized engines reach *parity* (MJX 1.04×).
  Best applied to unoptimized/new envs.
- Equivalence verification is only as strong as the policy used: a stronger policy
  visits states random/weak rollouts miss. Re-run L4 as training improves.
- Limits: non-deterministic external deps (network, hardware-in-the-loop) or unbounded
  dynamic allocation need extra engineering. Best fit: reproducible transitions, clear
  module boundaries, fixed-size state.
- Training hyperparams used: LR 2.5e-4, clip 0.2, 4 epochs, GAE λ=0.95, γ=0.99,
  per-env matched batch sizes. Benchmarks on 1× RTX 5090.

## Relevance to azuki-tcg
This project is exactly the target class: a C-engine TCG with a Python/PufferLib RL
loop. The JAX checklist (fixed-size state, `lax.switch` for card/move dispatch,
`lax.scan` rollout fusion) and the Rust/PyO3 binding advice map directly onto a
potential high-throughput rewrite of the engine or its env step. TCGJax (Pokémon TCG
Pocket) is the closest published analog.
