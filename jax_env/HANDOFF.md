# JAX Environment Conversion — Handoff & 1:1 Verification Status

**Branch:** `skylark/train-optimizations`
**Goal:** A JAX/XLA GPU re-implementation of the Azuki TCG environment that is **bit-exact 1:1** with the C engine, so policies trained in JAX (fast, GPU-batched) transfer to the C engine for inference/gameplay. The C engine remains the oracle and the gameplay/inference runtime.
**Reference paper:** `optimizations/` (Karten et al. 2026, arXiv 2603.12145) — GPU-resident env design.

> ⚠️ **READ FIRST — uncommitted state.** The *entire* `jax_env/` directory is **untracked in git** (so is `.venv/`, `optimizations/`, and the new config files). None of the JAX conversion or the parity fixes below are committed. **Commit `jax_env/` before/after moving systems** or the work is lost. Tracked working-tree changes are only the C-side debug instrumentation + `python/src/training_utils.py`.

---

## 0. Current status at handoff (one paragraph)

The JAX env is functionally complete: full C-engine port (ECS→array state, all phases, combat, IKZ, gate-portal flow), **137 card abilities ported**, selection-zone runtime, when-takes-damage/redirect, passive-aura layer, leaders+gates. Five whole-episode divergence classes were found and fixed, four **confirmed** by step-by-step eager replay against C. The remaining gap to 1:1 is a **cluster of passive-aura timing quirks** (C's deferred-observer + buff-queue + flush-timing mechanics) plus a few **discrete bugs**, with root causes fully diagnosed (below). A single-process full-suite verification was running at handoff but on *old* code (pre-`c6`/passive work) and is compile-bound (~6h; see §6 — the enlarged selection mask made the XLA compile pathologically slow).

---

## 1. How to run (resume on a new machine)

### 1.1 Host env (no docker/uv — see memory `training-env-host-setup`)
```bash
# from repo root
python3 -m venv .venv --system-site-packages
.venv/bin/pip install <pinned deps>   # jax[cuda], numpy, pufferlib, pytest, etc.
```
JAX picks GPU by default. For **no-compile eager debugging**, force CPU: `JAX_PLATFORMS=cpu`.

### 1.2 Build the C engine (the oracle + the binding the tests compare against)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# tests import the compiled binding from build/python/src
```

### 1.3 Run the parity suite (authoritative 1:1 gate)
```bash
bash jax_env/run_verify_single.sh         # single process: engine_step compiles ONCE, all tests share the jit cache
# = pytest jax_env/tests/ -q  with JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache
```
Key tests:
- `jax_env/tests/test_l3_fullpool.py` — **the definitive gate.** Plays the 18 production decks (16 pool + 2 starter) mirror + cross, 600 steps each, asserting **full state parity + legal-action mask parity at every step**, terminal agreement, and `ab_scratch[3]==0` (no unimplemented ability ever reached = "everything imported"). Cases: `MIRROR_CASES=[(i,i,12345+i)]`, `CROSS_CASES=[(i,(i+1)%N,7000+i)]`.
- `test_l3_abilities_batch4.py`, `test_l3_passives.py`, `test_l1_*`, `test_l2_vanilla.py`, `test_l3_abilities_batch{1,2,3}.py`.

### 1.4 Training (the original optimization goal — SPS uplift)
- Vectorized JAX env: `python/src/azk_puffer/jax_vector.py` (untracked).
- Smoke config: `python/config/azuki_jax_smoke.ini` (untracked).
- `python/src/training_utils.py` has the training-integration changes (tracked-modified).
- Original C training entry: `python/src/train.py` (`PYTHONPATH=build/python/src:python/src`).

---

## 2. Architecture of the JAX env (`jax_env/azuki_jax/`)

State is a flat pytree of arrays (`state.py`, `State` NamedTuple) instead of ECS. `engine_step(state, action)` is one `jax.jit` function; `build_mask(state)` enumerates legal actions. Everything is `lax.while_loop`/`lax.cond`/`lax.switch` so it runs eagerly under `jax.disable_jit()` (no compile) and compiled under jit.

| File | Responsibility |
|---|---|
| `state.py` | `State` pytree: zones, stats, buffs, ability FSM, latches, redirect queue, etc. |
| `env.py` / `setup.py` | init from decks (`init_state_with_decks`, `deck_tables_from_card_lists`) |
| `engine/step.py` | `engine_step`, `stabilize`/`auto_resolve` (micro_tick ladder = C `azk_engine_tick`) |
| `engine/phases.py` | phase machine, **end-of-turn expiry** (eot atk/hp/combat/carapace modifiers) |
| `engine/apply.py` | per-action appliers (`apply_gate_portal`, `apply_attach_weapon`, `_enter_board_slot`, …) |
| `engine/helpers.py` | zone/stat helpers, `stt02_012_garden_event` latch, eot resets |
| `engine/triggers.py` | triggered-effect queue (`queue_effect`, `pop_triggered`, on-play/enter-garden/takes-damage) |
| `engine/validate.py`, `masks.py` | legal-action validation + the (1024,4) mask (the **compile bottleneck**, see §6) |
| `abilities/runtime.py` | ability FSM: `begin_ability`, `_clear_context` (once-per-turn mark), confirm/cost/effect/selection |
| `abilities/cards_impl.py` | shared effect primitives: `deal_effect_damage` (+ Pekiro redirect), `apply_attack_modifier`, discards |
| `abilities/cards_batch{1..4}.py` | per-card ability ports (registered into the dispatch) |
| `abilities/passives.py` | **passive-aura layer** — `recompute_passives` (current-board recompute; see §5.1) |
| `abilities/cards_batch_passives.py` | per-aura contribution fns (`_azk01_010/019/073`, `_stt02_012`, …) |
| `abilities/selection.py` | selection-zone runtime (reveal/pick/bottom-deck/equip) |
| `abilities/effects.py` | `resolve_triggered_effect` (drains the trig queue → `begin_ability`) |
| `observe.py`, `rewards.py`, `rng.py` | obs encoding, reward shaping, RNG |

---

## 3. What's COMPLETED (with verification status)

### 3.1 Core port — DONE, verified by L1/L2/L3 suites
ECS→array state, all phases, combat resolution, IKZ tap/untap, deferred-op semantics, ability FSM (CONFIRMATION/COST_SELECTION/EFFECT_SELECTION/SELECTION_PICK/BOTTOM_DECK), gate-portal flow, 137 card abilities, leaders+gates (16), selection runtime, when-takes-damage + AZK01-062 redirect, passive-aura layer (9 cards). (Tasks #1–#13 complete.)

### 3.2 Divergence fixes found this branch

| Tag | Root cause | Fix location | Status |
|---|---|---|---|
| **W-bug1** | cross-player instance compare: `ti != src` excluded the wrong instance when same index belongs to different players | `abilities/cards_batch4.py` `_stt04_009_target`, `_azk01_062_target`: `(tp != owner) | (ti != src)` | ✅ confirmed (eager) |
| **V4 + X** | selection capacity hard-capped at 8; real selections exceed 8 (`MAX_SELECTION_ZONE_SIZE = MAX_DECK_SIZE = 50`) | `abilities/selection.py` `MAX_MOVED`; `masks.py` `MAX_SELECTION_ROWS` → `MAX_SELECTION_ZONE_SIZE` (mask universe 870→2116) | ✅ confirmed (eager) |
| **V3** | gate-portal validate ran *post*-placement; C inserts the portaled card + discards the displaced occupant via **deferred** ECS ops, so the gate-ability validate sees the **pre**-placement garden | `engine/apply.py` `apply_gate_portal`: set scratch + compute `card_ok=validate_card(...)` **before** `_enter_board_slot`; cost-selection still re-enumerates post-placement (fizzles if the displaced occupant was the only sacrifice) | ✅ confirmed (eager, clean 140 steps) |
| **W-bug2** | AZK01-062 "Pekiro" is `[Once/Turn]`; C's `maybe_queue_pekiro_redirect` can't re-queue the spent ability so it deals direct; JAX deferred regardless → stuck pending redirect, AZK01-062 wrongly survived | `abilities/cards_impl.py` `deal_effect_damage`: add `& ~once_used` (`once_per_turn_used[tp,ti]&1`, same slot `_clear_context` marks) to the redirect-defer predicate | ✅ confirmed (eager, W-7 clean to ep end) |
| **c6** | C expires eot **attack** modifiers only in **garden+leader**, NOT alley (`end_phase.c:88-89`); JAX expired them in alley too (`in_mod` included alley), so a friendly **alley** entity's eot atk buff (e.g. STT04-001) was wrongly stripped | `engine/phases.py`: new `atk_eot_zones = (z==GARDEN)|(z==LEADER)` used for `cur_atk`/`atk_buff_eot` reset (combat/carapace keep `in_mod` which *does* include alley) | ⚠️ applied, **not yet eager-verified** |

"Confirmed (eager)" = step-by-step C-vs-JAX parity replay diverged before the fix and runs clean after, via `jax_env/tests/diag_eager.py` under `JAX_PLATFORMS=cpu` (no compile).

---

## 4. Diagnostic tooling created (use these to reproduce + debug divergences)

All in `jax_env/tests/`. Eager (`JAX_PLATFORMS=cpu`, `jax.disable_jit()`) avoids the ~hours XLA compile but runs ~5–40 s/step (pool decks are slow). The **C engine is ~ms/step**, so C-only replays are the fast path for inspecting C's ground truth.

| Tool | Use |
|---|---|
| `diag_eager.py batch4 <GROUP> <SEED> [steps]` / `diag_eager.py pool <i> <j> <SEED> [steps]` | step-by-step C-vs-JAX parity; prints **FIRST DIVERGENCE** with full context (state keys, mask diff, ability ctx, recent actions). Primary divergence finder. |
| `diag_c_only_pool.py <i> <j> <seed> <maxstep>` | **fast C-only** replay (same rng+driver filter → identical C trajectory). With `DBG_PASSIVE=1` prints the C passive instrumentation. Dumps C garden0/garden1 per step. |
| `diag_evolution.py <i> <j> <seed> <start> <end> [keys...]` | dump chosen semantic keys for both engines across a step window (reusable per-bug). |
| `diag_c16_counts.py` | c16-specific STT02-012 count/latch dump. |
| `diag_v3*.py`, `diag_case.py`, `diag_split.py`, `diag_c_only.py` | earlier V3 / split-jit probes (mostly superseded; keep for reference). |

**C-side debug instrumentation (env-gated, currently in the working tree — REMOVE before final/commit):**
- `DBG_PASSIVE=1` → `src/abilities/cards/azk01_019.c` (`[Cjay]` all-Normal decisions), `src/abilities/cards/stt02_012.c` (`[C012]` garden counts + apply/remove), `src/utils/status_util.c` (`[Cqueue]` buff-queue flush). All inert without the env var, but they add `<stdio.h>`/`<stdlib.h>` and `fprintf` — **strip them for the clean engine** (and recompile). The earlier redirect/gate-portal debug in `damage_util.c`/`azk01_124.c` was already reverted.

---

## 5. What REMAINS for 1:1 (the actual work to finish)

### 5.1 ⬛ THE BIG ONE — passive-aura timing cluster (c10, c15, c16, m16, likely more)

**Symptom:** garden-state-dependent auras show ±1 stat vs C in specific windows.
- `c10` (pool 10v11 s7010 @95): AZK01-019 "Jay" (+2 hp when garden all-Normal) — JAX hp 3, C hp 1.
- `c15` (pool 15v16 s7015 @102): AZK01-010 (+2 atk, same condition).
- `c16` (pool 16v17 s7016 @69) & `m16` (pool 16 s12361 @95): STT02-012 (+1/+1 when own_garden_entities − opp ≥ 2) — diverges **both directions**.

**Root cause (confirmed via `DBG_PASSIVE` C-only replays):** C maintains passive buffs as **sticky `(AttackBuff/HealthBuff, source)` pairs**, updated only by **garden add/remove observers** that **queue** apply/remove decisions, flushed by `azk_process_passive_buff_queue` (`azuki_engine.c:404`, **deferred while an ability FSM is active**). Two distinct quirks:
1. **Deferred-insert entry-lag (c10/c15):** a freshly-*played* aura's ChildOf-garden insert is deferred, so at its own `init`/`update_*_buff` the card reads **NOT in play** → its self-buff is **not** applied until the *next* garden event. Evidence: `[Cjay] ent1048 NOT in play -> remove`, C garden hp 1. JAX `recompute_passives` reads the final board and applies immediately (hp 3).
2. **Multi-event net (c16/m16):** in a single step with two garden events (gate portal = remove-displaced **then** add-portaled), STT02-012's observer fires remove (intermediate counts) then apply (final counts), but only the **removal** lands in the flushed queue (`[Cqueue] flush ent994 is_removal=1`, no matching apply — flush/ability-phase boundary splits them) → C ends **unbuffed**; JAX's end-of-step recompute sees diff≥2 and applies +1/+1.

**JAX today:** `abilities/passives.py::recompute_passives` recomputes all aura contributions from the **current board** at end of `apply_user_action` and every `micro_tick` (respects the ability-phase skip, but not the event/queue/deferred-insert semantics). STT02-012 alone has a partial `stt02_012_latch` (`state.py`, `engine/helpers.py::stt02_012_garden_event`, 6 call sites) that approximates the event model but is imperfect (hence c16/m16).

**Fix required (the multi-hour port):** replace per-step recompute with a faithful **event-driven, queued, sticky** model matching C:
   - Maintain per-(player,instance) latched contributions; update them only at the **same garden-mutation points** C's observers fire (add/remove; STT02-012 observes **both** gardens, AZK01-010/019/073 observe **own** garden — confirmed via each card's `*_init_passive_observers`).
   - Replicate **deferred-insert**: a card entering the garden is NOT counted/contributing at *its own* entry event (it reads not-in-play); other auras DO see it.
   - Replicate the **queue** flush timing + ordering + the ability-phase deferral so multi-event steps net to C's result (remove-then-apply may not both land).
   - Replicate the **removal off-by-one** the existing STT02-012 latch already encodes (`is_removal` subtracts 1 from the event-side garden).
   - Audit **all** auras in `cards_batch_passives.py` for the same class (AZK01-010, AZK01-019, AZK01-073, STT02-012, and the rest of the 9), not just the 4 observed.

**Recommended approach:** instrument C flush boundaries (add an `fprintf` at `azuki_engine.c:404` printing ability-phase + queue count) to nail the exact remove/apply landing per step, then port the queue+observer mechanism. Verify each aura with `diag_c_only_pool.py … DBG_PASSIVE=1` vs `diag_eager.py pool`.

### 5.2 ⬛ Discrete: combat-death discard ordering (c11)
`c11` (pool 11v12 s7011 @77): `discard1` same multiset, **transposed order** — C `[…,14,14,13]` vs JAX `[…,14,13,14]`, after 3 attacks (steps 73-75) killing multiple entities. The order entities enter the discard pile on simultaneous/sequential combat deaths differs. **Not yet pinned** — read C's death/discard order in `src/systems/combat_resolve_phase.c` + `src/utils/damage_util.c` (combat damage → death → `discard_card`) and match JAX's combat-death discard ordering (`engine/` combat + `cards_impl` discard). Likely an attacker-vs-defender or slot-order tiebreak. Add a `discard0/discard1` dump to `diag_evolution.py` window 71-77 to see it.

### 5.3 ⬛ Verify c6 fix
`c6` fix (§3.2) is applied but not eager-verified. Re-run `diag_eager.py pool 6 7 7006 90` (expect clean past step 82).

### 5.4 ⬛ Full enumeration — the 26 cases never pre-checked
Only **10** baseline-failing pool pairs were eager-pre-checked (m1, m15, m16, c0, c6, c7, c10, c11, c15, c16). Results: **c7 PASSED**; c6/c10/c11/c15/c16/m16 diverged (above); **m1/m15/c0 ran clean past ~step 100** (slow, killed before finishing — promising, re-confirm). The other **26** of 36 fullpool cases (mirrors 0,2-14,17; crosses 1-5,8,9,12-14,17) plus all L1/L2/L3/batch tests have **not** been checked against the current code. **The single-process full-suite (`run_verify_single.sh`) is the enumerator** — run it after the §5.1–5.3 fixes; it lists the first divergence per failing case via the assertion message.

### 5.5 ⬛ Final 1:1 gate + training smoke (task #14)
After all divergences close: full `pytest jax_env/tests/` green (esp. `test_l3_fullpool.py`), then a short training smoke (`azuki_jax_smoke.ini`) to confirm end-to-end SPS uplift vs the C env (the original goal).

---

## 6. Verification bottleneck (important for iteration speed)

`engine_step` compiles **once per process** (the on-disk JAX cache does **not** persist it across processes here). The **selection-mask enlargement** in the V4/X fix (`NUM_CANDIDATES` 870→**2116** from `MAX_SELECTION_ROWS` 8→50) made the XLA HLO compile **pathologically slow** (~6h observed, single-thread CPU-bound, GPU idle until done). This makes the full-suite loop impractical.

**Two mitigations (do one before the passive port to make iteration tractable):**
1. **Shrink the mask bound.** 50 (=`MAX_DECK_SIZE`) is the theoretical max selection; the real max is much smaller. Find the largest selection any card actually creates (search reveal/search effects), set `MAX_SELECTION_ROWS` to that + margin (e.g. 16–24), re-verify the V4/X cases still pass. Should cut the compile to ~30 min.
2. **Iterate on eager pre-checks** (`diag_eager.py pool …`, `JAX_PLATFORMS=cpu`, no compile, ~70 min to ~step 110, run failing pairs in parallel) and use the full-suite (jit) only for the final authoritative gate.

---

## 7. Recommended completion sequence

1. **Commit `jax_env/` now** (and `.venv`-independent config). Don't lose the work in the move.
2. Strip `DBG_PASSIVE` debug from the 3 C files (§4), recompile clean.
3. **Shrink the selection-mask bound** (§6.1) so the verify loop is ~30 min, not ~6h.
4. **Verify c6** (§5.3); **fix c11 discard order** (§5.2).
5. **Port the passive event/queue model** (§5.1) — the big one; verify each aura via `DBG_PASSIVE` C-only vs eager.
6. **Run the full single-process suite** → fix whatever the 26 un-checked cases surface; loop until green.
7. **`test_l3_fullpool.py` fully green** = 1:1 achieved (state + mask + terminal parity, 600 steps × 36 cases, `ab_scratch[3]==0`).
8. **Training smoke** + SPS comparison vs C (original goal).

---

## 8. Task tracker mapping (in-session task list)

- #14 *(in_progress)* — Full-pool 1:1 verification + final suite + training smoke ← the umbrella; blocked by below.
- #15 ✅ V3 fix (confirmed) · #16 ✅ V4/X selection-capacity (confirmed) · #17 ✅ W-group bug1+bug2 (confirmed).
- #18 *(pending)* STT02-012 latch — **superseded by #21** (whole passive cluster).
- #19 *(pending)* c11 combat-death discard ordering (§5.2).
- #20 *(pending)* c6 STT04-001 eot atk — **fix applied (§3.2), needs verify (§5.3)**.
- #21 *(pending)* **passive-aura cluster** (§5.1) — the major remaining work; root cause confirmed.

---

## 9. Gotchas

- **`PYTHONPATH` must include `build/python/src`** for the `binding`/`conftest.CRef` C oracle the tests compare against, plus `python/src` and `jax_env`.
- Tests drive both engines with the **same `numpy.default_rng(seed)` + `DRIVER_TYPES_4` filter on C's legal actions** → identical trajectories; a C-only replay reproduces the exact C state at any divergence step.
- Recompiling the C engine while a verify run holds the old `.so` is **mmap-safe** (running process keeps its loaded copy); behavior is unchanged by the env-gated debug.
- Eager (`jax.disable_jit`) is correctness-equivalent to jit but slow; use it to *find* divergences, the C-only replay to *explain* them, the jit full-suite to *gate* them.
- Determinism: both the C engine RNG and the Python `default_rng` must be seeded identically for reproducibility.
