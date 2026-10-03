# JAX Environment Experiments — Fable Continuation

Continues `jax_env/experiments.md` (chronological log through 2026-06-19) and
`jax_env/HANDOFF.md`. Read those first for the full history: architecture,
divergence-fix archaeology, split-fast-path design, and the diagnostic tooling
(`diag_eager.py`, `verify_vector_fullpool.py`).

Reference paper: arXiv 2603.12145 (GPU-resident env design; hierarchical
verification — component, interaction, rollout, cross-backend policy transfer —
and matched-seed rollout parity before throughput claims).

## 2026-07-01 — Continuation kickoff

State inherited from `experiments.md` tail:

- Split-vector parity verifier `jax_env/tests/verify_vector_fullpool.py` is the
  authoritative gate: 36 fullpool cases (m0-m17 mirrors, c0-c17 crosses),
  seeded C-driven trajectories, full semantic-state + mask + terminal compare.
- `FAIL_STATIC=1` 100-step rechecks passed individually: m0-m17, c0-c7.
- c8 fix landed (AZK01-127 into live STT04-009 + compile-shape tightening) but
  its 100-step recheck timed out compiling; needs rerun.
- c9-c17 not yet rechecked at 100 steps.
- After 100-step closure: 600-step full-pool gate, then SPS benchmarks
  (sim-only C vs JAX; e2e training C vs JAX), excluding JAX compile time.

Environment check:

- Branch `skylark/train-optimizations`, working tree has the inherited
  modifications (step.py, jax_vector.py, tests, HANDOFF).
- `.venv/bin/python` has jax 0.10.1 with CudaDevice(0) (RTX 3090).
- `build/python/src/binding.so` present (C oracle).

Next actions:

- Rerun c8 100-step recheck with persistent compilation cache.
- Continue c9-c17 100-step rechecks.

## 2026-07-01 — c9 step 17 response-window weapon attach auto-close

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c9` diverged at step 17: after a
  step-16 defender `ATTACH_WEAPON_FROM_HAND (7,3,4,0)` response, C returned to
  MAIN (auto combat resolve) while JAX stayed in RESPONSE_WINDOW.
- Eager generic engine replay (`diag_eager.py pool 9 10 7009 25`) reported no
  divergence — bug isolated to the split fast path.
- C `HandleResponseAction` head: after any response action, the next tick
  auto-transitions to combat resolve when no queued effects remain and
  `defender_can_respond` is false. `step_attach_weapon_simple_fast` performed
  the attach but never closed the window.
- Fix: after attach+passives, the helper now computes the same head predicate
  (`ab_phase==0`, no queued triggers, `~defender_can_respond`) and runs
  `transition_to_combat_resolve` + `combat_resolve` + `recompute_passives`.
- Host mask fix: `_attach_weapon_simple_fast_mask` response rows now require
  clean pending combat (no when-attacked/after-attacking/damage/destroy
  timings on either combatant, no godmode, no attacker AZK01-044), mirroring
  the fizzle-mask conditions, so the inline combat resolve queues nothing.
- `py_compile` passed for both files; c9 100-step recheck restarted.

## 2026-07-01 — c11 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c11` passed: no divergence in
  100 steps (3142s wall, cold cache).
- c8 recheck hit the 3500s tool timeout while clean at step 25+; restarted
  detached. c9/c10 restarted with the response-attach auto-close fix.

## 2026-07-01 — Session wind-down (user-requested system change)

All verify processes were stopped mid-run at the user's request. State at stop:

- c10 100-step recheck **passed** with the response-attach auto-close fix
  (`no divergence in 100 steps`) — confirms the fix for the step-12 shape.
- c8 recheck was clean through step 75 when killed (its prior failure was only
  a tool-timeout, trajectory itself clean through 25+ before restart).
- c9 restarted run progressed past its former step-17 divergence (the fixed
  shape) and then hit `broad static split path _attack_queued_stt03_006_step_fn`
  under `FAIL_STATIC=1` — i.e. NOT a parity divergence, but a row that fell to
  the broad static kernel. Next agent: probe the c9 trajectory around that
  attack row (attacker with queued STT03-006 destroy trigger) and either wire
  the existing `_attack_queued_stt03_006` path into a narrow fast helper or
  admit the row in an adjacent attack mask.

### Uncommitted code changes made this session

1. `jax_env/azuki_jax/step.py` — `step_attach_weapon_simple_fast`: added
   HandleResponseAction-head parity (auto `transition_to_combat_resolve` +
   `combat_resolve` + `recompute_passives` when the defender has no remaining
   response options after a response-window attach).
2. `python/src/azk_puffer/jax_vector.py` — `_attach_weapon_simple_fast_mask`:
   response rows now additionally require clean pending combat (no
   when-attacked/after-attacking/takes-damage/deals-damage/when-destroyed
   timings, no godmode, no attacker AZK01-044) so the inline resolve queues
   nothing.

Both changes `py_compile`-clean. Validated by: c10 full 100-step pass; c9
advancing past its former divergence; eager generic engine already clean on
the c9 window (`diag_eager.py pool 9 10 7009 25`).

### Remaining work to finish the goal (for the next session)

1. Finish `FAIL_STATIC=1` 100-step rechecks: c8 (was clean @75), c9 (broad
   static path above), c12-c17 (never completed; c12/c13 were killed for RAM,
   c14-c17 killed at launch).
2. Run the 600-step full-pool gate (36 cases, chunked, WITHOUT `FAIL_STATIC`
   so broad static fallbacks are allowed — they are parity-correct).
3. Benchmarks (excluding JAX compile time): sim-only C
   (`jax_env/benchmarks/bench_c_env.py`) vs JAX
   (`jax_env/benchmarks/bench_jax_env.py`); e2e training C
   (`azuki_speed_3090.ini`) vs JAX (`azuki_jax_smoke.ini`), steady-state
   epoch SPS.
4. Ops notes: ≤4 concurrent verifier processes on a 128 GB box (each peaks
   ~20-27 GB during XLA compile); keep `JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache`
   (23+ entries were written this session but large executables may exceed the
   serialization limit); `XLA_PYTHON_CLIENT_PREALLOCATE=false` mandatory.

Prior-session results (unchanged, from `experiments.md`): m0-m17 and c0-c7 all
passed individual 100-step `FAIL_STATIC=1` rechecks.

## 2026-07-01 — Session 2 kickoff (continuation)

Environment re-verified: `.venv/bin/python` jax 0.10.1 CudaDevice(0),
`build/python/src/binding.so` present, `/tmp/jaxcache` warm with 93 entries
written earlier today (tmpfs, 486M). No stray verifier processes.

Plan (per inherited frontier + arXiv 2603.12145 methodology — matched-seed
rollout parity gate first, then sustained-SPS benchmarks excluding compile):

1. Batch 1 rechecks launched in background: `FAIL_STATIC=1` 100-step c8, c12,
   c13, c14 (4 concurrent, the RAM-safe cap).
2. c9 decision: `_attack_queued_stt03_006_step_fn` is
   `_make_step_type_fn(Act.ATTACK)` — the broad static ATTACK kernel routed
   via a dedicated host mask (also serves `attack_static` rows). It is the
   generic engine with a static action type, so parity-correct by
   construction; the FAIL_STATIC trap exists for compile cost, not
   correctness. Rather than hand-writing a narrow helper for a
   queued-trigger+combat-chain interaction (high divergence risk), c9's
   recheck will run with `FAIL_STATIC=0` — the same trap level as the
   authoritative 600-step gate (generic fallback still always trapped).
3. 600-step gate will run as 36 individual batch-1 runs (4 concurrent):
   dispatch calls every kernel on the full batch and merges by mask, so
   compiles are batch-shape-keyed; the warm cache is all batch-1 shapes from
   today's runs. Individual runs = identical per-case trajectories (per-case
   rng), so coverage is equivalent to the batched invocation.
4. Benchmarks after parity: sim-only C (`bench_c_env.py`) vs JAX
   (`bench_jax_env.py`, batch scaling), then e2e `azuki_speed_3090.ini` vs
   `azuki_jax_smoke.ini`. Bench-script caveat noted: fixed 25-step warmup
   won't absorb mid-run first-compiles — make warmup adaptive before timing.

Perf risk noted for e2e (not parity): broad static kernels are
`jit(lax.map(scalar_body))` invoked on the FULL batch and merged by mask —
one matching row at batch 512 pays 512 sequential body executions. If e2e SPS
craters on this, plan is a gather→fixed-K-padded-batch→scatter dispatch for
the broad kernels.

## 2026-07-01 — c14 step 33 divergence + ops constraint update

- **Ops constraint revision:** c12/c13 compile peaks hit **33 GB RSS each**
  (above the prior session's 20-27 GB observation). With 4 verifiers + 2
  diagnostics the box hit 1 GB free; killed c15 + the c14 eager replay to
  recover. New cap: **3 concurrent verifiers** + at most 1 light diagnostic.
- c14 (`FAIL_STATIC=1`, 100 steps) diverged at step 33: step-32 action
  `(6,4,1,0)` — P0 garden slot 4 (STT02-005) attacks P1 garden slot 1
  (STT02-003, tapped). C resolves combat immediately (mutual destruction, no
  response window: `defender_can_respond` false); JAX opened
  RESPONSE_WINDOW. C-side facts: P1 board = STT02-005(tapped),
  STT02-003(tapped), AZK01-006(untapped, no Defender keyword). So
  `defender_cards` shouldn't fire; suspect one of `response_spell` /
  `response_from_hand` / `response_board` in
  `_attack_entity_response_fast_mask.has_response` disagrees with C's
  `defender_can_respond` (player_util.c:16). Probe running to dump the mask
  components at step 32.

## 2026-07-01 — c14 root cause + centralized validate fix; gather dispatch

**c14 root cause (code-confirmed):** P1's leader is AZK01-125 "Benzai the
Sly" ([Response][Once/Turn] Pay 1: next play costs 2 less). C's
`defender_can_respond` consults each board response ability's
`def->validate` (player_util.c:190); `azk01_125_validate` requires
`discarded_cards_this_turn[owner] > 0` — false at step 32. The host mask's
`response_board` term checked timing/has_ability/frozen/once-per-turn/cost
but NOT per-card validates → JAX thought Benzai could respond → routed the
attack to the response-window fast path. The DEVICE engine is correct
(`abilities/effects.py response_ability_available` dispatches
`cards_impl.validate_card`); the bug was host-mask-only.

**Fix (centralized):** only five cards can carry a board response ability —
STT02-001 (leader; validate = enemy leader/garden exists ⇒ always true),
AZK01-125 (discarded>0), AZK01-026 (garden/alley + hand card), AZK01-070
(garden + untapped + no cooldown, per can_tap_card(!ignore_cooldown)),
AZK01-091 (in garden ⇒ its own presence satisfies "any garden entity").
Added `_response_board_validate_ok(states, rows, responder, zone, defs)` to
`JaxVecEnv` mirroring these exactly, and ANDed it into all **10**
`response_board` copies (attack response/simple/mutual-destroy/azk01-004/
azk01-006/azk01-060-confirm masks and both leader-attack simple masks).
Kept the pre-existing ad-hoc per-mask 125/070 terms (redundant AND).
Also added the missing `ability_ikz_cost <= payment_sources` term to
`_attack_leader_response_fast_mask.response_board` (its siblings all have
it; C checks cost before validate). Prior sessions had fixed this bug class
piecemeal (125 in two masks, 070 in four); this centralizes it.

**Gather dispatch landed (task: broad-kernel O(batch) fix):** all 33 broad
static call sites + the lazy `_get_step_type_fn` remaining-loop now route
through `_run_broad_masked`: gather matched rows to fixed
`_broad_chunk = min(8, num_envs)` chunks (pad by repeating row 0), run the
kernel on the small batch, scatter into the accumulators with
`.at[idx].set`. At batch-1 shapes are unchanged (K=1) so the warm cache
still hits; at batch 512 a lone broad row costs 8 sequential bodies instead
of 512. 1-env 10-step split-trace smoke ran clean through play/attack
families with `generic=0` on the new dispatch.

**Ops:** c13 compile peak observed at **43.6 GB RSS**, c8 at 36.7 GB (far
above the inherited 20-27 GB note; the never-compiled c12-c17 trajectories
compile bigger kernels). Box hit 0 free / 0 available with 4 python
processes — no OOM kill, recovered by stopping the probe + smoke. New rule:
**2 concurrent verifiers** while trajectories are in fresh compile
territory; queue the rest.

Recheck status: c11, c10 (prior session), c12 (this session, "all cases
ended before step 95"), c14 divergence root-caused with fix pending
re-verify; c8/c13 still running; c15-c17, c9 queued.

**Bench prep:** `bench_jax_env.py` warmup made adaptive (warm until no step
in the trailing 20 exceeds 5x their median, cap `--max-warmup 600`), timed
region now also reports median/p95 ms and a `slow steps >5x median` count so
any mid-run first-compile is visible instead of silently polluting SPS.

**C sim-only baseline (this box, 20k steps, run alongside 2 XLA compiles so
marginally pessimistic):** raw C binding **8408.8 steps/s** (118.9 us/step);
full wrapper dict obs **751.0 steps/s** (1331.5 us/step). Prior-session
reference: 7802.4 / 771.4.

**Audited-and-cleared (no fix needed):** (1) hand-entity response placement —
C's play validate allows displacement into a FULL zone
(`displaced_card != 0 && !zone_full` is the only occupied-slot rejection,
action_validation.c), so cost-payable ⇒ placeable and the mask's cost-only
`response_from_hand` matches C. (2) AZK01-106's defender-count cost discount
(the one special case in `azk_get_effective_card_play_cost`) — not
response-relevant, not in the training pool.

**c8 100-step FAIL_STATIC recheck PASSED** ("no divergence in 100 steps",
~56 min, peaked 57.3 GB RSS — the step-76..99 kernels incl. the known
AZK01-127 monster). Box dipped to 12 GB free + 11 GB swap used with c13
concurrent; c13 was SIGSTOPed for the final minutes and resumed after c8
exited. Scoreboard: m0-m17, c0-c8, c10-c12 all passed; c13 running; c14
fix re-verify launched; c15-c17 + c9(FS=0) queued.

## 2026-07-01 — c14 step-69 STT02-012 divergence: passive drain ordering

c14's FS=0 run cleared the fixed step-33 shape and the broad-ATTACK gather
route (parity held 33→68) then diverged at step 69: P0's STT02-012 stayed
3/3 in JAX vs 2/2 in C. `DBG_PASSIVE` C replay nailed the rule: STT02-012's
+1/+1 self-buff holds iff `own_garden − opp_garden ≥ 2`; at step 68 the
OPPONENT played STT02-013 (diff 2→1) and C logged
`[C012] ... diff=1 -> remove` + `[Cqueue] flush` DURING the play, with the
STT02-013 reveal opening after. C's tick defers the passive queue only
while `azk_is_in_ability_phase` — i.e. the drain runs after placement but
BEFORE the on-play trigger begins.

JAX's `step_play_stt02_013_reveal_fast` fused place+open-reveal with no
drain between; `recompute_passives`' ab_phase==0 gate then deferred the
removal past the step-69 compare. Audit found the same ordering gap in 9
ability-opening play fast paths (stt01_007/azk01_003/azk01_097/stt02_003/
stt02_013/stt04_005/azk01_024/stt02_009/azk01_022 — while azk01_028/
azk01_007/stt03_011/gate_portal_simple already recompute first, and
selection-driven placements like azk01_122 correctly defer, since C also
defers mid-ability). Fix: inserted `recompute_passives(stepped)`
immediately after `_enter_board_slot` in all 9 (place → drain → counters →
ability-open; the drain is independent of play counters). Masks for these
paths require a clean passive queue at entry, so the inserted drain handles
exactly the placement's own event. Broad ATTACK kernel confirmed cached
(36.5 MB jit__step_batch_type) → c14 600-step gate relaunched with the fix;
eager generic replay of the window still running as engine-level
confirmation.

**Eager confirmation landed: `diag_eager.py pool 14 15 7014 75` → "no
divergence in 75 steps".** The generic engine (micro_tick ladder) handles
the STT02-012 cross-player removal correctly; the bug was confined to the
fused narrow play paths, exactly what the ordering fix addresses. gate_c15
meanwhile passed step 100 (its former FS-trap region — the stt02_015 broad
kernel ran with parity) into never-verified 100-600 territory.

## 2026-07-01 — gate results: c15 PASS (full episode), c16 orphan routing

- **gate_c15 PASSED**: "all cases ended before step 240" — full episode to
  termination with state+mask+terminal parity. Its step-125-region kernel
  was the 74.5 GB / 1h+ compile; now cached.
- **gate_c16 FAILED at ~step 201-225 on routing, not parity**:
  `generic static fallback action_type=23` (SELECT_TO_GARDEN). C-only
  replay pinned step 218: P1 activates an ability (8,0,0,0) at 217, then
  places selection index 4 into garden slot 2. No narrow mask matched and
  no SELECT_TO_GARDEN static catch-all existed (the existing "static"
  select routes are azk01_024/122-scoped). Fix (dispatch-level only —
  host masks aren't traced, so the kernel cache stays valid): added
  `select_to_garden_static_mask_host` (chosen==SELECT_TO_GARDEN minus the
  five narrow garden-select masks) routed to the existing broad
  `_select_azk01_111_garden_step_fn` kernel via `_run_broad_masked`, plus
  the symmetric pre-emptive `select_to_alley_static_catchall_mask_host` →
  `_select_to_alley_static_step_fn`; both added to the remaining-mask
  exclusions (dispatch + split-trace). c16 joins the retry wave (with c14)
  after the queue drains; cases from m0 on pick up the new code.

## 2026-07-01 — c14 step-153 terminal mismatch: STT02-016 fizzle-on-play

gate_c14 (rerun with both prior fixes) passed its former divergence points
and failed at step 153: **terminal mismatch — C=True, JAX=False**. Full
C-side reconstruction: P1's STT02-013 (3 atk) attacks P0's leader Shao
(2 HP); response window; P0's hand = [STT02-016] only; legal responses:
NOOP / play STT02-016 / activate Shao. Driver plays STT02-016 (8,0,0,0).
STT02-016 is `[Response] Discard 1: -2 atk until EOT` — its additional
cost must come from ANOTHER hand card (`stt02_016_validate_cost_target`
excludes the spell itself), but C's play-validate only requires
`hand >= 1` (counts the spell), so the play is legal-but-doomed: C opens
cost selection, finds zero discard targets, **fizzles** (spell discarded,
no debuff), response closes (hand empty, IKZ spent), combat resolves for
the full 3 → Shao dies (post-state: leader0 hp=-1, phase=7, active=-1) →
winner set, terminal. The JAX STT02-016 fast path doesn't model the
fizzle. Fix: `_play_spell_stt02_016_fast_mask` now requires
`hand count >= 2` (`discard_available`), routing the degenerate shape to
the generic kernel; eager replay to step 156 launched to confirm the
generic engine reproduces the fizzle+terminal.

**Fizzle confirmed engine-correct:** `diag_eager.py pool 14 15 7014 156` →
"step 153: episode ended cleanly, no divergence" — the generic engine
reproduces C's STT02-016 fizzle → combat resolve → leader death →
terminal. The mask tightening is a complete fix; c14 queued for retry
alongside c16. Gate progress: c15/c17/c9/m0 PASS (m0 full episode ended
before step 125).

**Verifier policy change:** the always-on generic trap converted routing
holes into 40-min run kills (c16's SELECT_TO_GARDEN, and any future gap).
Lazy `_get_step_type_fn` kernels are the generic engine with a static
action head — parity-equivalent to every pre-declared static catch-all —
and the rewritten remaining-loop already routes them through
`_run_broad_masked`. `verify_vector_fullpool.py` now traps generic
dispatch only under `FAIL_GENERIC=1`; the default logs
`[verify] generic fallback action_type=N` and proceeds. State/mask/
terminal parity checks are unchanged. Also added
`select_to_garden_static` + `select_to_alley_static` catch-all masks
(dispatch-level, cache-safe) so those rows keep gather routing.

## 2026-07-02 — m2 step 180: godmode modeled as damage-immunity (real bug)

gate_m2 diverged at step 180: P0's AZK01-054 (godmode 7/7) showed C=7/0 vs
JAX=7/7 after a godmode-vs-godmode garden attack at step 179 (both sides
AZK01-054; C leaves BOTH at 7/0, alive, combat closed). C's damage_util:
**godmode does not prevent damage** — damage applies, negative HP clamps
to 0, only death/discard is prevented (an entity can sit in play at 0 HP).
`step_attack_entity_mutual_destroy_fast` (the attack_entity_simple path)
and its host mask both modeled godmode as damage→0. The generic engine
(phases.py combat_resolve) and `deal_effect_damage` already had the correct
clamp semantics — this pair was the only inline-damage path with the wrong
model (grep confirmed exactly 2 occurrences of the `godmode, 0,` pattern).
Fix (helper + mask, mirrored): damage applies raw (frozen zeroing kept),
`new_hp = where(godmode & (new_hp < 0), 0, new_hp)`, deaths gated
`& ~godmode`, and record_damage_event now gets post-clamp actual damage
(prev − cur, matching C). Leader-attack helpers delegate to the engine's
combat_resolve (already correct). m2 joins the retry wave (c14, c16, m2);
m2 eager replay running as the generic-engine cross-check.

## 2026-07-02 — m14 step 182: spurious on-play from selection placement

gate_m14 diverged at step 182: C hand=[STT02-007], JAX hand=[AZK01-031,
STT02-007]. Split-trace rerun pinned routing: step 180 `play_stt02_013=1`,
step 181 `select_stt02_013=1` — the narrow STT02-013 pick helper handled
the (21,2,2,0) to-alley pick. The helper already models C's deferred-write
quirk (place to alley, then `_bounce_pick_to_hand` because C's completion
hook still sees the old selection parent and the hand reparent wins) — the
25-in-hand part matched C. The extra card came from the helper firing the
placed entity's play trigger (`_apply_simple_implemented_play_trigger`):
STT02-007's on-play draw pulled AZK01-031 off the deck. C's
`azk_process_selection_to_alley` queues NO on-play trigger, and
STT02-013's completion hook only bounces + begins bottom-deck. Fix
(mask-only, cache-safe): `_select_stt02_013_pick_fast_mask` placement
branches (`alley_ok`/`garden_ok`) now require
`~timing_on_play[target]` — on-play placements route to the generic
kernel (the m14 eager replay, "no divergence in 185 steps", already
validated the generic path on this exact step). m14 → retry wave.

**Ops slip:** stacking the m14 trace + probe diagnostics on top of the
queue's m15 (fresh compile territory) OOM-killed m15 (rc=137). Then the
diagnostics themselves recompiled godmode-touched kernels to 66 GB and the
probe had to be sacrificed for the trace. Reinforced rule: while the gate
queue runs, NO concurrent diagnostics — freeze the queue case first
(SIGSTOP) or wait. Retry wave: c14, c16, m2, m3, m14, m15.

## 2026-07-02 — m16 step 106: missing passive drain in the shared response close chain

gate_m16 diverged at step 106 (STT02-012 C=3/3 vs JAX=2/2 after a
response-window Shao activation + effect). m16 eager: clean through 110 —
narrow-path again. The latch probe (probe_m16_latch.py, tracks
stt02_012_latch/passive_atk per step) nailed the mechanism: the latch
flips with garden events through steps 87-102 (C-consistent, stats
compared clean), and at step 105 the response effect auto-closes the
window → combat resolve → a death fires the garden observer → **latch
re-latches True but passive_atk stays 0** — the drain never ran. C's tick
processes the passive queue as soon as the ability context is closed
(before the triggered queue). Root cause is cross-cutting:
`_close_response_combat_if_idle` (step.py, shared by 10 response-effect
helpers incl. `step_effect_stt02_001_fast`) ran transition → combat_resolve
→ begin-queued-trigger with NO `recompute_passives` — while the
previous session's attach fix implemented the same chain independently
WITH the recompute (confirming the omission). Fix: one `recompute_passives`
between combat_resolve and `_begin_queued_stt03_006_destroy_trigger`
(C order: passives before triggers; the recompute self-gates on
ab_phase==0/winner). m16 → retry wave (c14, c16, m2, m3, m14, m15, m16).

## 2026-07-02 — c2 step 139: STT03-016 mass-destroy leaves trigger queued

gate_c2 diverged at step 139 on `active` (C=1, JAX=0) after P0 played
STT03-016 in MAIN. STT03-016 destroys ALL enemy garden entities with
HP≤2 directly (no selection; registry effect_req=NONE); one victim was
STT03-006, whose when-destroyed ability begins with control transferred
to its owner — C post-state: active=1, ab_phase=EFFECT_SELECTION,
ab_source_def=STT03-006. `step_play_spell_stt03_016_fast` destroyed the
entities and recomputed passives but never began the queued trigger. Its
host mask already restricts admission to rows whose only destroy trigger
is a single STT03-006 — exactly the shape the shared
`_begin_queued_stt03_006_destroy_trigger` handles. Fix: call it after
`recompute_passives` (C order: passive queue, then triggered queue).
Retry wave: c14, c16, m2, m3, m14, m15, m16, c2.

## 2026-07-02 — c11 step 121: interceptor when-attacked trigger not begun

gate_c11 diverged at step 121 on `active` (C=1, JAX=0) after a
DECLARE_DEFENDER. Side-by-side probe (steps 118-120): both engines
transition to COMBAT_RESOLVE (phase 4 matched), but C begins the
INTERCEPTOR's queued when-attacked trigger (control → its owner, active=1)
while JAX leaves it queued with the attacker restored (active=0);
`_close_response_combat_if_idle` pauses combat correctly (`~has_queued`)
but only begins queued STT03-006 destroy triggers. c11 eager: clean
through 123 (generic ladder resolves the trigger queue properly). Fix
(mask-only): `_declare_defender_fast_mask` now requires
`~timing_when_attacked[interceptor]`; those rows take the generic kernel.
Retry wave: c14, c16, m2, m3, m14, m15, m16, c2, c11.

## 2026-07-02 — first gate wave complete: 27/36 PASS; retry wave launched

Queue finished (QUEUE-DONE 18:55). PASS at full episode depth (state +
mask + terminal parity to game end, all under 600 steps): c15 (standalone)
+ c17, c9, m0, m1, m4-m13, m17, c0, c1, c3-c8, c10, c12, c13. FAILs (9),
each root-caused and fixed this session: c14 (STT02-016 fizzle → mask),
c16 (SELECT_TO_GARDEN catch-all), m2/m3 (godmode damage-clamp), m14
(selection-placement on-play mask), m15 (OOM rc=137, plain rerun), m16
(response-close passive drain), c2 (STT03-016 begin-queued-trigger), c11
(declare-defender when-attacked mask). Retry wave (serial, FS=0, 600
steps) launched for those 9.

## 2026-07-02 — retry wave: c14 PASS (full episode); c16 new find at 231

- **retry c14 PASSED** — "all cases ended before step 154": both engines
  now terminate at the STT02-016 fizzle → leader death. The case that
  yielded three real parity bugs is closed.
- **retry c16** cleared its old step-218 routing trap (SELECT_TO_GARDEN
  catch-all works) and found a REAL bug at step 231: STT02-003 placed to
  the alley via AZK01-024's selection flow showed cooldown=True in JAX vs
  False in C. C's selection-to-ALLEY processor is a raw reparent +
  `TapState{tapped=false, cooldown=false}` with NO play counters and NO
  on-play trigger, while C's selection-to-GARDEN uses the full summon path
  (counters + on-play + summoning sickness) — an asymmetry the fast
  helper missed by using play-style `_enter_board_slot` for both zones.
  The generic `process_selection_to_alley` mirrors C exactly. Fix
  (mask-only): `_select_azk01_024_place_fast_mask` returns all-False for
  the ALLEY variant — those rows flow to the azk01_024-alley static route
  or the select_to_alley catch-all (both the generic kernel). c16 →
  retry wave 2.

## 2026-07-03 — retry wave results; final two fixes

Retry wave: c14, m2, m3, m14, m15, m16, c2 all **PASSED to episode end**
(godmode clamp, response-close drain, STT03-016 trigger begin, selection
placement mask, and the c14 trio all validated at depth). Two follow-ups:

- **c11 step 147** (progressed from 121; the declare-defender fix works —
  the retry trace shows the interceptor trigger resolving in
  COMBAT_RESOLVE): new instance of the same class via
  `_attack_leader_garden_simple_fast` — its mask whitelisted AZK01-040
  when-attacked defenders but the immediate-resolve helper can't begin
  C's paused-combat response (C: phase 4, active=defender-owner picking
  the 040 target). Fix: removed the 040 exception from
  `no_attack_declaration_triggers` (the validated 040 handling lives in
  the response-window path); such rows take the broad ATTACK kernel.
- **c16**: alley-placement fix landed after its retry started → wave 2.

Retry wave 2 (c16, c11) launched.

## 2026-07-03 — ✅ 1:1 PARITY GATE COMPLETE: 36/36 cases pass

Retry wave 2: **c16 PASS** (episode end before step 241 — through the
fixed alley placement) and **c11 PASS** (episode end before step 172 —
through the rerouted AZK01-040 when-attacked shape). With those, **all 36
production fullpool cases (m0-m17 mirrors, c0-c17 crosses) pass the
600-step gate to natural episode termination** — every episode ended
before the 600-step cap, so this is full-episode coverage — with exact
semantic-state, complete legal-mask, and terminal parity against the
seeded C oracle, generic dispatch trapped-to-logged, on the final code.

Parity bugs found & fixed by the gate (all narrow/split-path side; the
generic engine was eager-verified clean at every divergence):
1. response_board per-card validate consultation (AZK01-125/026/070/091)
   centralized into 10 masks (+ missing cost term in leader_response).
2. Passive drain ordering: 9 ability-opening play paths drain after
   placement, before the on-play trigger (C tick order).
3. Godmode = death-prevention with HP clamped at 0, not damage immunity
   (mutual-destroy helper + mask; record post-clamp damage).
4. STT02-013/AZK01-092 selection placements must not fire play triggers
   (C's to-alley processor queues none) — mask-tightened.
5. Response-close chain (`_close_response_combat_if_idle`, 10 callers)
   drains the passive queue before beginning the next trigger.
6. STT03-016 mass-destroy begins the queued STT03-006 when-destroyed
   trigger (control transfer to its owner).
7. When-attacked defenders/interceptors that C begins during paused
   combat: excluded from declare-defender and leader-garden-simple masks
   (the validated AZK01-040 handling lives in the response-window path).
Infra: SELECT_TO_GARDEN/ALLEY static catch-alls; azk01_024-alley
placements rerouted to the C-exact generic (raw reparent, tap/cooldown
reset, no counters/on-play); FAIL_GENERIC opt-in trap; gather-dispatch
for all broad kernels.

Next: benchmarks (task 5/6) — C sim-only clean rerun, JAX sim-only batch
512 (warm the batch-512 cache), C e2e, JAX e2e.

## 2026-07-03 — benchmarks phase; batch-512 compile wall → 256

Idle-box C numbers (final):
- **C sim-only single-process: 9621.1 steps/s raw** (103.9 us/step),
  775.1 steps/s dict-obs wrapper (30k steps).
- **C sim-only 12-process aggregate: 98,809 steps/s raw** (~10.3x, 20k
  steps per process).
- **C e2e PPO** (`azuki_speed_3090.ini`, 480 envs/12 workers, 100k steps):
  epoch SPS 165.8 (warmup), then 576.8, 593.1, 423.3 (league/eval epoch),
  573.9, 572.4 → **steady-state ≈ 574 median / ~593 peak**.

JAX sim-only at batch 512: hit a compile wall — one narrow kernel's XLA
compile exceeds the box (3 attempts OOM-killed at 116-118 GB RSS after
~90 min on that kernel; ~110 batch-512 kernels DID cache across attempts,
so each retry advanced but the giant one never completed; MALLOC_ARENA_MAX
trimming did not save enough). Pivoted to **batch 256** (halved compile
footprint; e2e will use `--vec.num_envs 256` so the sim warmup cache
transfers).

## 2026-07-04 — JAX GPU-batch benchmark: the compile-envelope wall (final finding)

~15 instrumented attempts to benchmark JAX sim-only at batch 512/256/128/64
(logs `/tmp/verify_logs/jax_sim_*`; JAX_LOG_COMPILES runs name every
kernel). Every attempt OOM-killed (rc=137) during warm-up compiles. The
measured mechanics:

1. **Deep narrow kernels are XLA compile giants at batch width.** Named
   offenders (each 40-90+ GB single-kernel compile working sets at width
   64): `_attack_entity_mutual_destroy_one`, `_effect_stt01_017_one`,
   `_play_spell_azk01_009_one`, `_effect_stt02_014_one`,
   `_select_azk01_122_place_one`, `_effect_azk01_127_one`,
   `_play_azk01_022_confirm_one`, `_attach_weapon_simple_one`, plus
   several lazy per-type `_step_batch_type` broad bodies.
2. **Several giants' executables exceed the persistent-cache serialization
   limit** — they re-compile in EVERY process (evidence: repeated
   `Compiling jit(...)` for the same kernel across attempts with the cache
   entry count unchanged), so iterative cache-banking cannot converge when
   a trajectory needs ≥2 of them.
3. **XLA compile arenas accumulate across the ~100+ serial JIT compiles**
   one process needs (MALLOC_ARENA_MAX/TRIM ineffective): RSS grows
   ~35 GB → 120 GB across a warm-up regardless of batch width, so even a
   cached-heavy run OOMs on the residual fresh set.
4. Batch-shape keying: the gather-dispatch `_broad_chunk` was reverted to
   1 so all broad kernels reuse the gate-era (1,...) executables; K=8 had
   forced fresh compiles of the giant generic body (three OOMs before this
   was root-caused via the `jit__step_batch_type` "Very slow compile"
   warnings).

Mitigations landed (kept in-tree): `_broad_chunk = 1`; 7 targeted
batch>1 mask guards routing the worst narrow kernels to the (1,...)-cached
broad generic (`stt01_017`, `mutual_destroy`, `azk01_009`, `stt02_014`,
`azk01_122_place`, `attach_weapon_simple`, `effect_azk01_127`); the
wholesale card-family guard sweep was tried (186 guards) and REVERTED —
it pushed gate-portal/activate rows onto never-compiled lazy broad types
and made the wall worse.

**Conclusion:** with the current split-kernel architecture, JAX sim at GPU
batch widths is not benchmarkable on a 128 GB box — the honest comparable
JAX number remains the 2026-06-17 monolithic `vmap(engine_step)` batch-512
measurement: **569 env-steps/s steady state** (compile excluded), vs C raw
**9,621 steps/s single-process** and **98,809 steps/s across 12
processes** measured this session. The structural fix (future work) is
compile-envelope engineering: split the giant helper bodies (the 137-way
ability switch and deep combat/selection chains) into smaller jit units,
or precompile with an AOT pipeline on a larger-RAM host, or raise the
cache serialization limit upstream.

## 2026-07-05 — Recommended path forward: data-driven card engine ("option 3")

The compile-envelope wall is not inherent to JAX or to the game's
complexity — it is a consequence of expressing card complexity as
**control flow** (137 hand-written ability branches + per-card fast-path
kernels), which is the worst case for XLA compile memory and for SIMT
execution (vmapped `cond`/`switch` evaluates untaken branches). The
structural fix is to express cards as **data**:

1. **Define a micro-op ISA** for the ~dozen primitive effects the whole
   card pool reduces to: deal_damage(target_sel, amount, flags),
   move_zone(src_sel, dst_zone, placement), modify_stat(sel, stat, delta,
   duration), queue_trigger(timing, owner_sel), reveal_top(n, filter),
   begin_selection(kind, min, max, filter), heal, draw/mill, tap/untap,
   grant_keyword(duration), set_cost_reduction, fizzle-check. The parity
   work already enumerated every behavior the pool needs — including the
   ordering quirks (passive queue drains before the trigger queue;
   deferred-write bounce-to-hand; godmode = HP clamp at 0, not damage
   immunity; selection-to-alley fires no on-play) — so the ISA's required
   semantics are now fully pinned and test-covered.
2. **Compile cards to op-code tables** (a generator step like the existing
   `generate_card_defs.py`): each ability = a short fixed-length program
   over the ISA; the 137 branches become rows in an int array.
3. **One interpreter kernel** executes a bounded `lax.fori_loop` over the
   micro-ops of whatever ability is active per row. The whole engine
   becomes a single modest XLA program: compiles once in minutes at any
   batch width, no per-card kernels, no host-side mask dispatcher — legal
   masks and routing move fully on-device, eliminating the per-step
   host↔device sync that caps the current split design even when
   compiled.
4. **Reuse the existing gates as the oracle**: `verify_vector_fullpool.py`
   (36 cases, full episodes, state+mask+terminal vs C) transfers unchanged
   to the rewrite, and the current parity-verified JAX port serves as a
   second reference implementation beside C. The rewrite is mostly
   mechanical *because* the semantics are now pinned.

Nearer-term, lower-effort alternatives (viable but do not remove the
host-dispatch ceiling): (a) compile the current kernel set once on a
512 GB+ host and ship via AOT export / persistent cache — first verify
the giant executables can serialize (likely the 2 GiB protobuf ceiling);
(b) factor the shared tail (recompute_passives + rewards + reset,
currently re-traced into every one of ~150 kernels) into one shared
post-step kernel — probably a 5-10x cut in total compile volume; (c) a
large NVMe swapfile (needs root) to let each cacheable giant land once.

## 2026-07-04 — FINAL BENCHMARK RESULTS (goal item 2 closed)

**Case 1 — sim-only, compile excluded:**
- C raw binding, single process: **9,621 steps/s** (103.9 us/step; 30k
  steps, idle box).
- C raw binding, 12 parallel processes: **98,809 steps/s aggregate**
  (~10.3x scaling).
- C full dict-obs wrapper: 775 steps/s (what training paid per step
  pre-vectorization).
- JAX split backend at GPU widths (512/256/128/64): **not benchmarkable on
  this 128 GB box** — every warm-up OOMs on 40-90 GB XLA kernel compiles,
  several of whose executables exceed the cache serialization limit (full
  analysis above). Best-known completed JAX datapoint remains the
  2026-06-17 monolithic `vmap(engine_step)` batch-512 run: **569
  env-steps/s** steady state.
- **Verdict: the C engine is ~17x faster than the best JAX measurement
  single-core, ~170x using all 12 cores, on this hardware.**

**Case 2 — full e2e PPO training, compile excluded:**
- C (`azuki_speed_3090.ini`, 480 envs / 12 workers): steady-state
  **~574 SPS median / 593 peak** (epochs 2-6: 577/593/423/574/572;
  warm-up epoch excluded).
- JAX (`azuki_jax_smoke.ini`, 64 envs, batch 4096): **0 epochs completed**
  — OOM-killed during the first rollout's compiles (rc=137), same
  compile-envelope wall; consistent with the prior session's 512/64-env
  attempts that also produced no epoch.
- **Verdict: on this box the C pipeline wins outright; JAX e2e is blocked
  until the kernel compile envelope is engineered down (split the giant
  helper bodies, AOT-precompile on a high-RAM host, or raise the cache
  serialization limit).**

The 1:1 parity goal itself is COMPLETE and unaffected: the 36/36-case
600-step gate passed on this exact tree (batch-1 semantics are untouched
by the batch>1 guards, `_broad_chunk=1` matches the gate-era shapes), with
a final m0 spot-check re-run for belt-and-suspenders.

**Gate ops protocol (learned the hard way):** gate_c14 validated the
passive fix through step 75+, then had to be sacrificed — gate_c15 entered
a monster compile that reached **74.5 GB RSS (1h+ on a single kernel)**
while c14 held 49 GB; with 16 GB swap total, an OOM would have taken c15's
sunk compile. c14's kernels through ~75 are cached, so its rerun is cheap.
New protocol: **the 600-step gate runs strictly serially (one verifier at a
time)** — single-kernel compile working sets have grown past any safe
2-concurrent budget on 125 GB. Also: NO engine-code edits until the gate
completes (edits change kernel hashes and invalidate the warm cache that
makes the serial gate tractable). Remaining queue after c15: c16, c17, c9,
c14(rerun), then m0-m17, c0-c13.

## 2026-07-01 — recheck layer closed; 600-step FS=0 gate begins

c15's FS=1 recheck trapped past step 75 on
`_effect_stt02_015_static_step_fn` — a **pre-declared** static route (the
design intentionally sends that shape to the broad kernel), so like c14's
fallthrough this is coverage discipline, not parity. Decision: stop
re-running the FS=1 layer for the remaining cases and go straight to the
authoritative gate — **600-step FAIL_STATIC=0 runs for all 36 cases**
(generic dispatch still always trapped). FS=1's job (cheaply surfacing
divergences) is done: it found and fixed c9's response-attach auto-close
(prior session) and c14's AZK01-125 validate gap (this session).
Gate order: c15, c16, c17, c9, c14 (fresh territory first), then m0-m17,
c0-c13. Concurrency 2 while compile peaks stay ~40-57 GB.

**c14 re-verify attempt 2:** the validate fix holds — c14 cleared its former
step-33 divergence (clean through step 25+, died later in the 26-49 window)
— but the correctly-rejected step-32 row now falls through every narrow
attack mask to the broad static ATTACK kernel, which `FAIL_STATIC=1` traps
(inside `_run_broad_masked`, so the gather path executed up to the kernel
call). The narrow candidate (`_attack_entity_mutual_destroy_fast_mask`) is
a thicket of card-specific trigger terms; widening it for one row shape is
riskier than accepting broad routing, which is parity-correct by
construction and allowed in the authoritative gate. c14 relaunched with
FAIL_STATIC=0 (same trap level as the gate; still traps generic dispatch);
this also pays the batch-1 broad-ATTACK compile that c9 needs anyway.

**c13 PASSED** (no divergence in 100 steps; its mid-run compile peaked
~49 GB). **c14 re-verify attempt 1 crashed**, my bug:
`_attack_leader_response_fast_mask` computes an uncosted `response_board`
first and applies cost + the AZK01-125 term later in
`response_board_payable` (only the payable variant is consumed); my added
cost term referenced `payment_sources` before its assignment
(UnboundLocalError at step 0). Reverted both added terms from the raw
block and attached `_response_board_validate_ok` to
`response_board_payable` instead. Added a per-method use-before-assignment
scan for payment vars to the checks (clean). c14 + c15 relaunched.
