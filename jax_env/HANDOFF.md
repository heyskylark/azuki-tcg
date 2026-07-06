# JAX Environment Conversion — Handoff & 1:1 Verification Status

## ✅ FINAL STATUS — 2026-07-04 (supersedes everything below)

**1:1 parity: CONFIRMED.** All 36 production fullpool cases (m0-m17
mirrors, c0-c17 crosses) pass the authoritative 600-step gate
(`jax_env/tests/verify_vector_fullpool.py`, FAIL_GENERIC-relaxed) to
natural episode termination — every episode ended before the 600-step cap
— with exact semantic-state, complete legal-mask, and terminal parity
against the seeded C oracle, on the final tree. Seven real parity bugs
were found and fixed by the gate (response-ability validates, passive
drain ordering ×2 sites, godmode damage-clamp, selection-placement
trigger/cooldown semantics, STT03-016 trigger begin, when-attacked
interceptor routing) — details in `fable_experiments.md`.

**Benchmarks (compile excluded):**
- Sim-only: C raw **9,621 steps/s** single-process / **98,809 steps/s**
  across 12 processes. JAX at GPU batch widths: NOT benchmarkable on this
  128 GB box — warm-up OOMs on 40-90 GB XLA kernel compiles (several
  executables exceed the persistent-cache serialization limit; compile
  arenas accumulate across the ~100+ serial JITs one process needs). Best
  completed JAX datapoint: 569 env-steps/s (monolithic vmap, batch 512,
  2026-06-17).
- E2E PPO: C **~574 SPS** steady-state (480 envs / 12 workers). JAX: 0
  epochs (first-rollout compile OOM), 64- and 512-env attempts alike.
- Net: on this hardware the C engine wins both cases outright. The JAX
  port's value today is the verified 1:1 parity + infrastructure; its
  performance requires compile-envelope engineering (split the giant
  helper bodies — the 137-way ability switch and deep combat/selection
  chains — into smaller jit units, AOT-precompile on a high-RAM host, or
  lift the cache serialization limit).

Tree state: `_broad_chunk = 1` (broad kernels reuse gate-era (1,...)
executables); 7 narrow masks carry `num_environments > 1` guards routing
their rows to the parity-proven broad kernels (batch-1 verifier semantics
unchanged); SELECT_TO_GARDEN/ALLEY static catch-alls added; the verifier's
generic trap is opt-in via FAIL_GENERIC=1 (default logs and proceeds).

## Current Frontier — 2026-07-01 (superseded; see `fable_experiments.md` for the live log)

State as of the 600-step gate campaign:

- **Recheck layer complete:** individual 100-step `FAIL_STATIC=1` runs
  passed for m0-m17, c0-c8, c10-c13. c14 surfaced two real parity bugs
  (both fixed, below); c15/c16/c17/c9 graduated straight to the FS=0 gate
  because their remaining traps were *designed* broad-static routes, not
  divergences.
- **Parity fixes this session (both host/split-path side; the generic
  engine was verified correct by eager replays):**
  1. `_response_board_validate_ok` (jax_vector.py): C's
     `defender_can_respond` consults per-card `def->validate` for board
     response abilities (AZK01-125 discarded-this-turn, AZK01-026 hand
     card, AZK01-070 garden+untapped+no-cooldown, AZK01-091 in-garden);
     the host masks' `response_board` terms didn't. Centralized helper now
     ANDed into all 10 response_board sites (plus the missing ability-cost
     term in `_attack_leader_response_fast_mask.response_board_payable`).
  2. Passive drain ordering: C flushes the passive buff queue after
     placement but BEFORE the on-play trigger begins (tick defers only
     while `azk_is_in_ability_phase`). Nine ability-opening play fast
     paths in step.py fused place+open with no drain between; each now
     calls `recompute_passives` right after `_enter_board_slot`.
- **Dispatch change:** all broad static kernel call sites route through
  `JaxVecEnv._run_broad_masked` — gather matched rows into
  `min(8, num_envs)`-row chunks, run the kernel small, scatter back. At
  batch 1 shapes are unchanged (cache-compatible); at batch 512 a lone
  broad row no longer pays a 512-row sequential `lax.map` sweep.
- **Authoritative gate in progress:** 600-step `FAIL_STATIC=0` (generic
  dispatch still trapped) per-case runs; c15 passed to episode end
  (~step 239). Remaining 35 cases run **serially** via
  `jax_env/tests/run_verify_queue.sh` — single-kernel compiles peak
  45-75 GB RSS on this 128 GB box, so one verifier at a time, and no
  engine-code edits mid-campaign (cache invalidation).
- **Benchmarks pending** (after the gate, on an idle machine): sim-only
  C vs JAX (`bench_c_env.py` / `bench_jax_env.py` — warmup now adaptive,
  reports median/p95 + slow-step count), then e2e
  `azuki_speed_3090.ini` vs `azuki_jax_smoke.ini`. Fresh C sim-only
  baseline: raw binding 8408.8 steps/s, dict-obs wrapper 751.0 steps/s.

## Latest Continuation Frontier — 2026-06-19 Active Run

This section supersedes the older step-407, step-93, and step-60 notes below for
the current branch state. `jax_env/experiments.md` remains the detailed
chronological log and should be updated after every probe/fix.

### Current Stop Point

- Deterministic `JaxVecEnv(4, seed=1)` split probe reached
  `NO_GENERIC through 4000 steps`.
- Latest saved checkpoint in that route:
  - `/tmp/jax_probe_3999.pkl`
  - runner output ended with `NO_GENERIC through 4000 steps`
- Slow first-use compiles were still observed on newly-covered paths; these were
  tracked in `jax_env/experiments.md` and are not parity proof.
- Current syntax validations passed after the STT01-002 gate fix and pytest
  verifier helper changes:

```bash
PYTHONPATH=build/python/src:python/src:jax_env \
  python/.venv-codex/bin/python -m py_compile \
  jax_env/azuki_jax/step.py \
  jax_env/tests/conftest.py \
  jax_env/tests/test_l2_vanilla.py \
  jax_env/tests/test_l3_abilities.py \
  jax_env/tests/test_l3_abilities_batch1.py \
  jax_env/tests/test_l3_abilities_batch2.py \
  jax_env/tests/test_l3_abilities_batch3.py \
  jax_env/tests/test_l3_abilities_batch4.py \
  jax_env/tests/test_l3_passives.py \
  jax_env/tests/test_obs_packing.py \
  jax_env/tests/test_l3_fullpool.py
```

### Fixes Landed In This Continuation Since The Step-533 Notes

- Residualized broad `SELECT_EFFECT_TARGET` fallback so already-covered effect
  rows do not compile the monolithic static selector.
- Added or widened split coverage for late-frontier rows through step 4000,
  including `AZK01-021`, `AZK01-024`, `AZK01-056`, `STT02-013`, `STT04-005`,
  `AZK01-065`, `STT01-007`, `AZK01-069`, `AZK01-068`, and selected
  `STT02-003` from `AZK01-024`.
- Relaxed several masks only where the helper already performs the required
  passive/STT02-012 bookkeeping or the watcher is inert for that action shape.
- `step_select_azk01_024_place_fast` now opens the normal `STT02-003`
  Watercrafting reveal flow when `AZK01-024` places `STT02-003` from selection.
- The pytest parity drivers now use shared cached `engine_step_static_action`
  and `build_mask` helpers (`jax_env/tests/conftest.py`) instead of compiling
  monolithic dynamic `engine_step`.
- Verified the stale c6 warning: `diag_eager.py pool 6 7 7006 90` now reports
  `no divergence in 90 steps`.
- Fixed split fast-path parity for `STT01-002` gate portal: the fast gate helper
  now runs the non-optional discard-weapon selection path immediately instead
  of opening a confirmation. Targeted replay of m0 step 9 now matches C with
  `ab_phase=0` and no semantic/selection diffs.

### Next Recommended Action

The frontier probe is only a split-dispatch smoke test. Full C/JAX parity is
still in progress. Current verification status:

- Broad/full pytest JIT verification is still impractical inside the 3600s
  harness timeout; even static-action broad fallbacks compile for tens of
  minutes.
- Use scalar eager diagnostics for pinpointing true engine divergences and the
  `/tmp/vector_fullpool_parity.py` chunk verifier for split fast-path parity.
- Historical hotspot rechecks passed in scalar eager through their known failure
  windows: c6 (90), c10/c15/c16/m16 (130), c11 (100).

If parity passes, run the requested SPS benchmarks:

- JAX and C engine simulation-only throughput.
- JAX and C end-to-end training pipeline throughput.

Keep every new attempt and result in `jax_env/experiments.md`.

## Latest Stop Point — 2026-06-18 User-Requested Winddown

The user explicitly stopped further code work after the step-60 probe and asked
for this handoff to be updated. Do not continue editing code from this segment
unless the user starts a new coding run.

### Process State

- All probes/traces were wound down.
- Final process check:

```bash
pgrep -af "python/src/azk_puffer/jax_vector.py|\.venv/bin/python -u -"
```

- Result: no matches, exit code `1`.

### New Probe Result

The deterministic `JaxVecEnv(4, seed=1)` route was reproduced to bench step 60
and stopped before `env.send(actions)`.

Step 60 active/chosen summary:

- `active=[1, 1, 1, 1]`
- `chosen=[19, 6, 14, 6]`
- attack rows were env `1` and env `3`

Env `1` was already handled by the existing attack response path:

- action `[6, 3, 5, 0]`
- active player `1`
- attacker `STT02-007` in garden slot `3`, instance `18`
- defender was player `0` leader `STT03-001`
- mask hits:
  - `_attack_leader_response_fast_mask=True`
  - all other inspected attack masks false

Env `3` is the remaining generic row:

- action `[6, 5, 3, 0]`
- active player `1`
- attacker is player `1` leader `AZK01-119`, instance `0`
- defender is player `0` garden `STT02-003`, slot `3`, instance `23`
- attacker has `STT01-012` attached, instance `42`
- opposing response-like cards:
  - player `0` leader `AZK01-125`
  - two `STT02-016` cards in hand
- all inspected attack fast masks were false
- queues/context at the decision point were clean:
  - `phase=MAIN`
  - `ab_phase=NONE`
  - `trig_count=0`
  - `redirect_count=0`
  - `passive_queue_count=0`
  - `combat_attacker=-1`
  - `winner=-1`

Interpretation:

- This is not safe to route through the plain attack-response declaration mask.
- The generic engine queues the attacker's When Attacking chain, including
  attached weapons. Here `STT01-012` is a weapon with When Attacking: mill one.
- The likely correct narrow fast path is:
  - declare leader attack into a garden target
  - require exactly the clean `STT01-012` attached-weapon trigger shape
  - apply `STT01-012`'s one-card mill/deckout behavior
  - then open the defender response window
  - only match rows where the defender really has a response, because the
    helper should not skip directly to combat resolve

### Partial Code Edit Present

Important: one code edit was made immediately before the user stopped code work.
It is intentionally documented here because it is not fully wired or validated.

- `jax_env/azuki_jax/step.py` now contains a new helper:
  - `step_attack_stt01_012_response_fast`
- The helper is intended to model the env `3` shape above:
  - leader attacker
  - attached `STT01-012`
  - direct mill via `mill_with_deckout`
  - response-window entry for the defender
- It has NOT been wired into `python/src/azk_puffer/jax_vector.py`.
- It has NOT been syntax-checked or trace-validated after this latest edit.
- No host mask exists yet.
- No JIT wrapper, dispatch merge, trace counter, or remaining-mask exclusion has
  been added yet.

Recommended next action when code work resumes:

1. Decide whether to keep and finish the partial
   `step_attack_stt01_012_response_fast` helper or replace it with a different
   narrow implementation.
2. If keeping it, add the vector harness plumbing:
   - import `step_attack_stt01_012_response_fast`
   - add `_stt01_012_id`
   - add `_attack_stt01_012_response_one`
   - add `_attack_stt01_012_response_step_fn`
   - add `_attack_stt01_012_response_fast_mask`
   - include it in handled masks, trace output, merge dispatch, and remaining
     mask exclusion
3. Keep the mask conservative:
   - `MAIN`, `ATTACK`, no active ability/context/queued effects
   - attacker target is the leader (`action[1] == GARDEN_SIZE`)
   - defender target exists and is leader or garden as supported by the helper
   - active leader has exactly one attached `STT01-012` and no other
     attack-declaration trigger sources
   - no Kira redirect (`AZK01-034`) or other declaration-time trigger chain
   - defender has a response option, using the same payable response checks as
     the existing response masks
   - no passive watcher or modifier shape outside what the helper models
4. Then run:

```bash
PYTHONPATH=build/python/src:python/src:jax_env \
  .venv/bin/python -m py_compile \
  jax_env/azuki_jax/step.py \
  python/src/azk_puffer/jax_vector.py

git diff --check -- \
  jax_env/azuki_jax/step.py \
  python/src/azk_puffer/jax_vector.py \
  jax_env/HANDOFF.md
```

5. Resume with the short-circuit trace command from the previous section and
   confirm whether bench step 60 advances to the next frontier.

## Latest Split-Action Frontier — 2026-06-18 Pause Handoff

This section is the current handoff for continuing the split-action JAX vector
work. It supersedes the older frontier sections below for the current branch
state. The older sections remain useful for archaeology and for understanding
why specific fast paths exist.

### Current Stop Point

- Active work was intentionally paused for a new coding-agent harness.
- All long-running trace/probe Python processes were stopped. A final
  `pgrep -af "python/src/azk_puffer/jax_vector.py|.venv/bin/python -u -"`
  returned no matches.
- No C engine files were changed in this pause segment; engine rebuild was not
  required.
- Files modified by this segment:
  - `jax_env/azuki_jax/step.py`
  - `python/src/azk_puffer/jax_vector.py`
  - `jax_env/HANDOFF.md`

### Validation Completed

These checks passed after the latest code edits:

```bash
PYTHONPATH=build/python/src:python/src:jax_env \
  .venv/bin/python -m py_compile \
  jax_env/azuki_jax/step.py \
  python/src/azk_puffer/jax_vector.py

git diff --check -- \
  jax_env/azuki_jax/step.py \
  python/src/azk_puffer/jax_vector.py
```

No full `4 105` trace was completed after the final edit because work was
paused at the user's request.

### Latest Trace Frontier

The latest short-circuit trace command was:

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache \
PYTHONUNBUFFERED=1 \
AZK_JAX_SPLIT_TRACE=1 \
AZK_JAX_BENCH_TRACE=1 \
PYTHONPATH=build/python/src:python/src:jax_env \
timeout 900 .venv/bin/python -u python/src/azk_puffer/jax_vector.py 4 105 1 \
| awk '/JaxVecEnv bench|construct:|async_reset|first send|second recv|bench step|generic=[1-9]/{print; fflush()} /generic=[1-9]/{exit}'
```

Current result:

- The deterministic `JaxVecEnv(4, seed=1)` route is split-only through bench
  step 59.
- The first remaining generic fallback is bench step 60.
- Step 60 starts with action types `6,14,19`.
- The split trace line at step 60 showed:
  - `effect_stt01_017=1`
  - `bottom_deck=1`
  - `attack_leader_response=1`
  - `generic=1 generic_types=6`
- `Act.ATTACK == 6`, so the next row to inspect is one unhandled attack row.
- The step 60 generic row has not been probed yet. Start there.

Suggested first probe for the next agent:

- Reproduce the deterministic route to `step_i == 60`, stop before `env.send`.
- Dump rows where the chosen active action type is `Act.ATTACK`.
- Print active player, phase, legal rows, board zones, attacker/defender card
  codes, combat state, trigger/redirect/passive counts, and mask hits for:
  - `_attack_entity_response_fast_mask`
  - `_attack_leader_response_fast_mask`
  - `_attack_entity_mutual_destroy_fast_mask`
  - `_attack_leader_simple_fast_mask`
  - `_attack_azk01_004_leader_fast_mask`
  - `_attack_stt01_006_effect_fast_mask`
  - `_attack_azk01_060_confirm_fast_mask`
- Then decide whether this is a safe mask widening or a new dedicated attack
  fast path.

### Fixes Landed In This Pause Segment

#### AZK01-056 Reveal Flow

- Added `step_play_azk01_056_reveal_fast`.
- Added `step_select_azk01_056_pick_fast`.
- Added `JaxVecEnv` masks, JIT wrappers, dispatch merges, trace counters, and
  source ids for AZK01-056.
- Added AZK01-056 support to selection pick/noop and bottom-deck cleanup
  whitelists.
- This covered the previously observed `play_azk01_056=1` row.

#### STT02-001 Leader Response Ability

- Added `step_activate_stt02_001_fast`.
- Added `step_effect_stt02_001_fast`.
- Added `JaxVecEnv` wrappers, masks, dispatch merges, trace counters, and
  source id.
- Behavior modeled:
  - `[Response][Once/Turn] Pay 1 IKZ`.
  - Target enemy leader or enemy garden entity.
  - Apply `-1` attack until EOT.
  - Clear ability context after the required target is selected.
- This covered the previously observed response-window leader activation row.

#### AZK01-040 / Attack Response Handling

- Widened `_attack_entity_response_fast_mask` to allow clean attacks into
  `AZK01-040` when its `When Attacked` response is the only special handling.
- Widened `_effect_azk01_040_fast_mask` for selected leader target and skip
  shapes and for clean attacker death handling.
- The fast path now resolves clean pending combat after the AZK01-040 effect
  where appropriate.

#### AZK01-127 Response Spell Effect

- Fixed `step_effect_azk01_127_fast` for the case where the response spell
  kills the current combat attacker.
- The fast path now clears the spell context, closes the response window when
  the defender has no further response, transitions to combat resolve, and
  runs combat resolution if no queued work remains.
- Relaxed `_effect_azk01_127_fast_mask` to allow clean lethal damage to the
  combat attacker. The old host mask rejected this because it did not account
  for the subsequent combat fizzle/resolve behavior.
- The earlier failing step 99 row was:
  - active player 0
  - action `[14, 3, 0, 0]`
  - source `AZK01-127` in discard
  - target enemy garden `AZK01-003` at 1 HP that was also the combat attacker
  - now handled by `effect_azk01_127=1`.

#### STT01-002 Optional Selection Skip

- Extended `_selection_pick_noop_fast_mask` to include `STT01-002` when:
  - `ab_costs_applied` is true
  - `ab_scratch[2] == 2`
  - selection pick count/pick max matches the existing shared skip fast path
- This uses the existing `step_selection_pick_noop_fast`, which calls the
  shared selection runtime. For STT01-002, skipping runs the registered
  selection-complete hook that returns remaining selection cards to discard and
  clears the context.
- This fixed the former step 43 generic `NOOP` row:
  - `phase=MAIN`
  - `ab_phase=SELECTION_PICK`
  - source `STT01-002`
  - legal alternatives were `SELECT_TO_EQUIP`.

#### STT03-006 Combat-Death Follow-Up

- Extended `step_response_noop_entity_combat_fast` so destroyed `STT03-006`
  is detected on either combat side, not only when it is the attacker.
- The fast path now:
  - tracks whether attacker-side or defender-side `STT03-006` was destroyed
  - draws for the correct owner
  - opens the mandatory discard-from-hand effect for the correct source and
    owner when that owner still has a hand card
  - sets `active_player` to the STT03-006 owner when that follow-up effect is
    opened
- Extended `_response_noop_entity_combat_fast_mask` so defender-side
  `STT03-006` when-destroyed is allowed under the same clean no-passive-death
  constraints as the attacker-side case.
- This fixed:
  - former step 44 generic response-window `NOOP`, where player 1
    `STT02-007` attacked player 0 `STT03-006` and destroyed it
  - former step 45 generic `STT03-006` effect target row, where the effect
    source was in discard and needed `active_player == ab_owner`.

### Important Current Caveats

- Do not assume the older "frontier through step 99" notes below are the active
  first-generic frontier. They came from broader traces that allowed earlier
  generic fallbacks to compile and continue. With the newer fast-path fixes,
  the current first generic fallback is step 60 attack type 6.
- The split-fast approach is still compile-heavy. Short-circuit traces are the
  fastest way to find the next unhandled row without waiting for generic
  fallback compilation.
- Keep using host masks conservatively. Most fixes in this segment were either
  exact card-specific fast paths or narrow mask widenings tied to already-modeled
  device behavior.
- If touching C engine code later, rebuild engine/native before web service
  work. No such rebuild was needed for this segment.

## Latest Split-Action Frontier — 2026-06-18

Continuation work on the JAX vector split fast paths advanced the deterministic
`JaxVecEnv(4, seed=1)` benchmark frontier through step 45. The latest traced
run now handles the former step-41 `AZK01-004` attack and `STT02-009` play rows
with split fast paths; the next generic fallback is one response-window `NOOP`
row at bench step 46.

- Closed the latest 4-env frontier rows:
  - Step 35 response-window entity-combat `NOOP` where both attacker and
    defender die and the destroyed attacker is `STT03-006`. The fast path now
    discards both entities, draws for `STT03-006`, and opens its mandatory
    discard effect when the owner still has a hand card.
  - Step 36 `STT03-006` effect selection, discarding the chosen friendly hand
    card and clearing ability context.
  - Step 36/37 `AZK01-009` spell setup and effect selection. The spell now pays
    cost, discards from hand, opens a one-target effect, then grants Charge to
    a cost-4-or-less garden entity.
  - Step 39 clean response-window play of response-playable entities from hand
    through the existing simple entity play kernel; this covered `AZK01-035`
    played to alley.
  - Step 39 clean leader attack by `AZK01-062` where the defending leader has
    zero attack, so Pekiro's takes-damage timing is not relevant to the attack
    resolution.
  - Step 41 `AZK01-004` attacking a zero-attack leader with no responses. The
    new fast path applies Alley Thug's +1 attack until end of turn before
    resolving nonlethal leader combat damage.
  - Step 41 `STT02-009` played from hand to alley. Dedicated play,
    confirmation, cost-target, and optional effect-target fast paths now cover
    Aya's on-play bounce flow when return-to-hand trigger/passive side effects
    are clean.
  - Main-NOOP host masks now allow permanent timed grants (`GRANT_PHASE_NONE`)
    while still rejecting start/end-ticking timed grants that the simple cleanup
    helpers do not process.
  - The `AZK01-011` main-NOOP host mask now allows `GRANT_PHASE_END` timed
    grants because that specialized step uses full `end_turn`, which performs
    end-phase timed grant ticking. It still rejects `GRANT_PHASE_START` grants
    because the following start-turn helper is intentionally narrow.
- Latest traced frontier:
  - Command used:
    `PYTHONPATH=build/python/src:python/src:jax_env
    AZK_JAX_SPLIT_TRACE=1 AZK_JAX_BENCH_TRACE=1 timeout 900
    .venv/bin/python -u python/src/azk_puffer/jax_vector.py 4 60`.
  - The run stays split-only through bench step 45. Bench step 41 now reports
    `play_stt02_009=1`, `attack_azk01_004=1`, `generic=0`.
  - Bench step 46 reports
    `gate_simple=1`, `confirm_stt01_002=1`,
    `main_noop_azk01_011=1`, `generic=1 generic_types=0`.
    The traced process was stopped after this frontier row was identified.
  - Remaining step 46 row:
    - Env 0: active player 0, `RESPONSE_WINDOW`, action `[0, 0, 0, 0]`
      (`NOOP`). Combat is player 1 `AZK01-036` attacking player 0
      `AZK01-062`; both entities would die. The response entity-combat fast
      path correctly rejects this because combat/destroy/takes-damage trigger
      handling is not clean (`no_triggers` is false).
- Validation passed after the edits:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m
    py_compile jax_env/azuki_jax/step.py
    python/src/azk_puffer/jax_vector.py`
  - `git diff --check -- jax_env/azuki_jax/step.py
    python/src/azk_puffer/jax_vector.py jax_env/HANDOFF.md`

The older split-frontier sections below are historical debugging notes. They
are still useful for archaeology, but this section supersedes their current
frontier and validation status.

## Latest Split-Action Frontier — 2026-06-17

Continuation work on the JAX vector split fast paths advanced the deterministic
`JaxVecEnv(4, seed=1)` benchmark frontier from step 27 to step 34.

- Closed step 27: added a guarded response-window `NOOP` entity-combat fast
  path for clean combat where the attacker dies and the defender survives.
  The known row after `AZK01-127` now reports `response_noop=1`, `generic=0`.
- Closed step 29:
  - Added `AZK01-045` play/reveal and selection-pick fast paths for the
    Obsidian top-5 reveal flow, plus bottom-deck cleanup coverage for its
    remaining selection cards.
  - Added `AZK01-002` immediate heal spell fast path. It pays/discards the
    spell, increments play counters, clears next-play reduction, and heals the
    owner leader through the same `heal_leader` helper the generic ability
    uses.
  - The traced row now shows `play_azk01_045=1`,
    `spell_azk01_002=1`, `generic=0`.
- Latest traced frontier:
  - Command used:
    `PYTHONPATH=build/python/src:python/src:jax_env
    AZK_JAX_SPLIT_TRACE=1 AZK_JAX_BENCH_TRACE=1 timeout 900
    .venv/bin/python -u python/src/azk_puffer/jax_vector.py 4 40`.
  - The run stays split-only through bench step 33. Bench step 34 has
    `generic=1 generic_types=10`.
  - Step 34 remaining generic row is `GATE_PORTAL`: row 3, active player 0,
    action `[10, 2, 2, 0]`, phase `MAIN`, `ab_phase=0`. It portals
    `AZK01-006` from alley slot 2 to garden slot 2 via `STT04-002`.
    The row otherwise looks simple, but `passive_queue_count=4`, so the
    existing `gate_simple` mask rejects it. Do not just ignore that guard:
    the likely next task is to handle/drain pending passive state before this
    gate path, or prove the queue is stale and safe to clear in this exact
    fast-path state.
- Validation passed after the edits:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m
    py_compile jax_env/azuki_jax/step.py
    python/src/azk_puffer/jax_vector.py`
  - `git diff --check -- jax_env/azuki_jax/step.py
    python/src/azk_puffer/jax_vector.py`

## Current Status — 2026-06-17 Continuation

This section supersedes all older "remaining work" notes below for the current
branch state. The lower sections are historical debugging notes and still useful
for archaeology, but the passive-aura items described there have since been
implemented and re-verified.

- Production-pool parity is green on the main shards that previously failed:
  - `jax_env/tests/test_l3_passives.py`: **10 passed** in
    `3812.18s (1:03:32)`.
  - `test_l3_fullpool.py::test_fullpool_mirror`: **18 passed** in
    `5124.18s (1:25:24)`.
  - `test_l3_fullpool.py::test_fullpool_cross`: all 18 cases reached 100% and
    pytest exited `0`; the final summary line did not flush in the captured
    terminal output.
  - `test_obs_packing.py::test_packed_observation_ability_deck[222]`:
    **1 passed** in `2308.40s (0:38:28)`.
- Remaining verification caveat:
  - A combined `test_l3_env.py` run passed the first seed, then the second seed
    drove RSS to roughly 119 GB and was killed before OOM. A prior isolated
    seed-77 run passed, and the latest passive/obs changes do not affect the
    vanilla env deck path.
- Final parity fixes in this continuation:
  - Ported C-like sticky passive buff queue semantics for self-passive
    `AZK01-010`, `AZK01-019`, `AZK01-073`, and `STT02-012` garden events,
    including the observed Flecs callback ordering and queue cap behavior.
  - Corrected `AZK01-073` to queue only on active-state transitions while
    `AZK01-010/019` queue on watched garden/alley events.
  - Removed stale test normalization that stripped leader activation rows from
    C packed observations; leaders are now ported, and C/JAX compare the real
    public ABI rows.
- Fresh benchmark numbers:
  - C raw binding: **7802.4 steps/s** (`128.2 us/step`) from
    `PYTHONPATH=build/python/src:python/src timeout 180 .venv/bin/python -u
    jax_env/benchmarks/bench_c_env.py 5000`.
  - C full wrapper dict obs: **771.4 steps/s** (`1296.3 us/step`) from the
    same run.
  - C e2e PPO training (`azuki_speed_3090.ini`, 480 envs, 12 workers,
    61,440 agent steps): epoch SPS **274.03** startup, then **582.65**,
    **601.28**, **594.66**; steady-state average of the last three epochs is
    about **592.86 SPS**.
  - JAX lean `vmap(engine_step)` batch 512:
    compile+first **3349.4s**, steady-state **569 env-steps/s**
    (`899.21 ms/batch-step`).
- E2E/vector benchmark caveat:
  - JAX e2e PPO at 512 envs reached first-rollout compilation but produced no
    epoch/SPS before the bounded run was terminated; GPU was idle while CPU RSS
    climbed.
  - A smaller 64-env JAX e2e probe behaved the same way: no first epoch before
    termination, GPU idle, CPU RSS climbing.
  - Current conclusion: the JAX engine is parity-viable, but the monolithic
    first-step XLA compile makes the current trainer integration
    non-benchmarkable for e2e SPS. The sim-only JAX number is also slower than
    the C raw binding and slightly below the C wrapper/training e2e rate.
  - JAX persistent cache is not solving this yet because large GPU executables
    exceed the cache serialization limit observed during these runs.
- Performance continuation notes after the e2e failure:
  - Refactored `apply_user_action` and `micro_tick` away from straight-line
    `jnp.where` masking toward `lax.cond`/`lax.switch` control flow. A
    25-step eager C/JAX smoke (`diag_eager.py batch4 W 42 25`) stayed clean
    after those control-flow edits.
  - Added static-action step helpers and a split-action `JaxVecEnv` path
    (`AZK_JAX_SPLIT_ACTIONS=1`, default) so the vector backend can compile
    kernels by primary action type instead of only the monolithic dynamic step.
  - Important JAX caveat observed: wrapping branchy scalar steps in `vmap`
    reintroduced select-like branch evaluation, so the split-action path now
    uses `lax.map` for the static action kernels.
  - Added a pregame mulligan fast path in `JaxVecEnv`. Tiny vector probe:
    `python/src/azk_puffer/jax_vector.py 1 1` now reports reset/observe
    compile+run about **31s**, first pregame send compile+run about **1.5s**,
    and completes at about **80 agent-steps/s** for that one-step smoke.
  - Added a guarded main-phase `NOOP` fast path for rows with no ability FSM,
    combat, existing trigger queue, pending passive drain, EOT trigger
    candidates, or start-trigger candidates. It calls the existing
    `apply_noop_main -> end_turn -> start_of_turn` helpers and keeps unsafe
    rows on the generic split path. Shared engine eager parity smoke
    (`diag_eager.py batch4 W 42 25`) still reports `no divergence in 25 steps`.
  - Added a narrower simple-start / simple-main-NOOP path. The pregame
    shortcut now resolves the post-mulligan `START_OF_TURN` auto phase instead
    of exposing a zero-legal state to the trainer.
  - Added narrow vector fast paths for the early random trajectory:
    simple entity play, `STT01-003` on-play mill, `STT01-002` gate portal into
    confirmation, optional confirmation decline/clear, simple weapon attach,
    non-lethal no-response garden attack into leader, `AZK01-058` mutual
    destruction into `STT01-003`, the `STT04-003` start-of-each-turn
    self-damage path including the death case, `STT04-014` simple play,
    `STT04-016`/`AZK01-059` spell/cost/effect handling, `AZK01-122` gate
    selection placement, `AZK01-065` Fire Orb including chained `AZK01-059`
    takes-damage resolution, `AZK01-121` leader activation, and response-window
    attack declarations against garden entities. Each path is host-guarded and
    unsafe rows stay on the generic split kernels.
  - Verified vector progress:
    - `jax_vector.py 1 3` now completes; type-2 play and type-10 gate portal
      use fast paths.
    - `jax_vector.py 1 10` now completes without generic kernels. Clean run:
      reset compile+run **30.3s**, first pregame send **0.5s**, then 10 traced
      steps complete at **2 agent-steps/s** (`1127.6 ms/step`) on the tiny
      1-env probe.
    - `jax_vector.py 1 20` now completes with `AZK_JAX_SPLIT_TRACE=1` and no
      generic fallbacks. Latest traced run: reset compile+run **30.8s**, first
      pregame send **0.6s**, all 20 bench steps stayed on split fast paths, and
      the tiny 1-env probe reported **3 agent-steps/s** (`774.4 ms/step`).
      This is still compile-heavy and not an e2e SPS result.
    - A traced `jax_vector.py 1 100` probe now completes without generic
      kernels. Latest traced run used
      `AZK_JAX_SPLIT_ACTIONS=1 AZK_JAX_SPLIT_TRACE=1 AZK_JAX_BENCH_TRACE=1`
      and reported **5 agent-steps/s** (`373.2 ms/step`) on the tiny 1-env
      probe. This is still compile-heavy and not an e2e SPS result, but it
      confirms the seed-1 random trajectory stays split-only through 100 bench
      steps.
    - Latest 4-env split frontier update:
      - Added guarded fast paths for the next seed-1 4-env trajectory segment:
        `STT01-004` confirm/pick handling, `STT04-004` confirm/effect, clean
        response-window `NOOP` into leader combat, `STT02-002` zero-power gate
        portal as a no-op extension of the simple gate path, `STT01-006`
        attack/effect flow, zero-legal truncation rows, and `STT01-005` alley
        activation plus its two discard-target selections.
      - Also widened clean leader-attack response detection to include playable
        response cards in hand, not just board/defender responses.
      - Important compile note: `STT01-006` effect selection originally called
        full `auto_resolve`, which pulled a large generic compile into the fast
        path. It now uses the narrow `phase_gate` transition and the host mask
        only admits rows that open a response window.
      - Current traced command:
        `PYTHONPATH=build/python/src:python/src:jax_env
        AZK_JAX_SPLIT_TRACE=1 AZK_JAX_BENCH_TRACE=1 timeout 900
        .venv/bin/python -u python/src/azk_puffer/jax_vector.py 4 60`.
        The run is now split-only through bench step 16 for seed 1. Step 16
        showed `effect_stt01_005=1`, `main_noop=1`, `pregame=1`,
        `play_simple=1`, and `generic=0`.
      - New frontier is bench step 17 with `generic=1`, action type `1`
        (`PLAY_ENTITY_TO_GARDEN`). Probe identified the row as active player 1
        playing `AZK01-033` (`Elder Hoshin`) from hand index 4 to garden slot
        1. This card is an on-play selection-zone reveal card (top 5, reveal up
        to one Steelborn, bottom the rest), so it should get a dedicated reveal
        fast path rather than being folded into simple play.
    - Late trace milestones now covered: step 91 `gate_simple`, step 92
      `select_azk01_122`, step 93 `spell_azk01_065`, step 94
      `effect_azk01_065`, step 95 `effect_azk01_059`, step 96
      `activate_azk01_121`, step 97 `attack_entity_simple`, step 98
      `main_noop`, and step 99 `attack_entity_response`.
    - Latest lightweight validation: `py_compile` passed for
      `jax_env/azuki_jax/step.py` and `python/src/azk_puffer/jax_vector.py`;
      `git diff --check -- jax_env/azuki_jax/step.py
      python/src/azk_puffer/jax_vector.py` passed after the step-17 frontier
      changes.
  - Remaining vector/e2e caveat: the split-fast approach is making the trainer
    integration executable one action family at a time, but it is not yet a
    full benchmarkable JAX e2e path. Generic fallbacks may still appear beyond
    the current 100-step seed-1 frontier, and when they do they still trigger
    large CPU-side XLA compiles with GPU idle.

**Branch:** `skylark/train-optimizations`
**Goal:** A JAX/XLA GPU re-implementation of the Azuki TCG environment that is **bit-exact 1:1** with the C engine, so policies trained in JAX (fast, GPU-batched) transfer to the C engine for inference/gameplay. The C engine remains the oracle and the gameplay/inference runtime.
**Reference paper:** `optimizations/` (Karten et al. 2026, arXiv 2603.12145) — GPU-resident env design.

> ⚠️ **READ FIRST — uncommitted state.** This branch has many tracked JAX env
> modifications plus several untracked benchmark/debug helpers. Do not assume the
> historical note that "all of `jax_env/` is untracked" is still true. Run
> `git status --short` before committing and stage only the intended parity,
> benchmark, and integration files.

---

## 0. Current status at handoff (one paragraph)

The JAX env is functionally broad enough for the production fullpool route: all core phases, combat, IKZ, gate-portal flow, leaders/gates, selection runtime, passive layer, and the imported card abilities used by the 18-deck training pool are ported. The active gap is no longer a known semantic divergence; it is finishing authoritative rollout coverage and avoiding compile-heavy broad fallback paths. Historical c6/c10/c11/c15/c16/m16 warnings now replay clean through their known failure windows, and the split `JaxVecEnv(4, seed=1)` frontier reached `NO_GENERIC through 4000 steps`. Treat lower divergence notes as archaeology unless a fresh verifier reproduces them.

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

### 1.3 Run parity verification
```bash
PYTHONPATH=build/python/src:python/src:jax_env \
  JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python jax_env/tests/verify_vector_fullpool.py 600 m0 m1 m2 m3
```
`verify_vector_fullpool.py` is the practical split-vector parity gate for the
training backend. It accepts case names (`m0`..`m17`, `c0`..`c17`) so the 36
fullpool cases can be run in compile-safe chunks; omitting case names runs all
cases in one batch and may be too large for routine iteration.

`jax_env/tests/test_l3_fullpool.py` remains the pytest-form fullpool gate, but
the current broad static-action JIT path is compile-heavy. Prefer the split
verifier for iteration, then rerun pytest subsets once compile cost is
controlled.

### 1.4 Training and benchmarks
- Vectorized JAX env: `python/src/azk_puffer/jax_vector.py`.
- Smoke config: `python/config/azuki_jax_smoke.ini`.
- JAX sim-only benchmark: `jax_env/benchmarks/bench_jax_env.py`.
- C sim-only benchmark: `jax_env/benchmarks/bench_c_env.py`.
- Training entry: `python/src/train.py` (`PYTHONPATH=build/python/src:python/src`).

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

### 5.1 ✅ historical passive-aura timing cluster rechecked clean

The old c10/c15/c16/m16 passive-aura cluster is not the current blocker. Fresh
scalar eager replays now cover the previously observed windows:
- `c10` (pool 10v11 s7010) clean through 130 steps.
- `c15` (pool 15v16 s7015) clean through 130 steps.
- `c16` (pool 16v17 s7016) clean through 130 steps.
- `m16` (pool 16v16 s12361) clean through 130 steps.

Keep the historical notes below for regression context only. They document the
C behavior that the JAX passive model must continue to match if a fresh replay
reproduces this class again:
- C maintains passive buffs as sticky `(AttackBuff/HealthBuff, source)` pairs.
- Passive observer decisions are tied to C garden/alley ChildOf events and queue
  flush timing, not just the final board snapshot.
- AZK01-010/019 watch own garden+alley; AZK01-073 watches own garden only;
  STT02-012 watches both gardens.
- Re-entry after leaving play does not necessarily re-fire at entry; the
  persistent observer can re-evaluate on the next watched event.

Do not start another passive queue rewrite from this stale section unless a new
`diag_eager.py pool …` or split-vector verifier run reproduces a concrete
divergence.

#### 5.1.1 historical mechanism notes (2026-06-14)

Step-correlated `DBG_PASSIVE` traces (`jax_env/tests/trace_019.py`,
`trace_jax_019.py`) for AZK01-019 in c10/m16 pinned the C rule as
observer-registration-scoped rather than a simple deferred-insert rule:

- **First-EVER entry into play fires immediately.** When a self-buff passive is
  played for the first time (garden OR alley), C registers its observer and the
  init update evaluates with the card in play.
- **Re-entry after leaving play does NOT re-fire at entry, but the observer
  persists and re-fires on the NEXT garden/alley event.**
  `azk_sync_card_abilities` skips `init_passive_observers` while
  `PassiveObserverContext` exists (`components/abilities.c:344-346`).
- **Zone MOVE of an in-play card keeps its buff** (alley→garden gate-portal):
  observer persists, its EcsOnAdd fires, and it re-evaluates. Not a
  leave/re-entry.

### 5.2 ✅ c11 combat-death discard ordering — current eager replay clean
`diag_eager.py pool 11 12 7011 100` now reports `no divergence in 100
steps`, covering the stale step-77 discard-order warning. Keep the historical
notes below for context only; do not treat c11 as open unless a fresh verifier
run reproduces it.

### 5.3 ✅ c6 STT04-001 EOT ATK fix verified
`diag_eager.py pool 6 7 7006 90` now reports `no divergence in 90 steps`,
covering the stale step-82 warning.

### 5.4 ⬛ Full enumeration — the 26 cases never pre-checked
The previously-known hotspot set now replays clean through its failure windows:
c6 (90), c10/c15/c16/m16 (130), c11 (100). The other 26 of 36 fullpool cases
(mirrors 0,2-14,17; crosses 1-5,8,9,12-14,17) plus all L1/L2/L3/batch tests
still need authoritative full-suite coverage. The single-process full-suite is
still the enumerator, but current JIT compile cost exceeds the harness timeout;
use scalar eager diagnostics and split-vector chunks to isolate any new
divergence before attempting the full suite.

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

1. Continue scalar eager and split-vector chunk verification for the remaining
   fullpool cases; the known hotspot set is currently clean.
2. Keep using cached static-action verifier helpers for pytest subsets; avoid
   the monolithic dynamic `jax.jit(engine_step)` path for parity work.
3. Run the full single-process suite once compile cost is controlled enough to
   finish inside the harness window; fix any fresh divergence with a minimal
   repro first.
4. `test_l3_fullpool.py` or equivalent split-vector fullpool verification green
   = 1:1 achieved (state + mask + terminal parity, 600 steps × 36 cases).
5. Training smoke + SPS comparison vs C (original goal).

---

## 8. Task tracker mapping (in-session task list)

- #14 *(in_progress)* — Full-pool 1:1 verification + final suite + training smoke ← the umbrella.
- #15 ✅ V3 fix (confirmed) · #16 ✅ V4/X selection-capacity (confirmed) · #17 ✅ W-group bug1+bug2 (confirmed).
- #18 ✅ STT02-012 latch — stale warning superseded by current hotspot rechecks.
- #19 ✅ c11 combat-death discard ordering (§5.2) — eager clean to step 100.
- #20 ✅ c6 STT04-001 eot atk — eager clean to step 90.
- #21 ✅ passive-aura cluster (§5.1) — historical hotspot set eager-clean through known failure windows; watch only if fresh replay reproduces.

---

## 9. Gotchas

- **`PYTHONPATH` must include `build/python/src`** for the `binding`/`conftest.CRef` C oracle the tests compare against, plus `python/src` and `jax_env`.
- Tests drive both engines with the **same `numpy.default_rng(seed)` + `DRIVER_TYPES_4` filter on C's legal actions** → identical trajectories; a C-only replay reproduces the exact C state at any divergence step.
- Recompiling the C engine while a verify run holds the old `.so` is **mmap-safe** (running process keeps its loaded copy); behavior is unchanged by the env-gated debug.
- Eager (`jax.disable_jit`) is correctness-equivalent to jit but slow; use it to *find* divergences, the C-only replay to *explain* them, the jit full-suite to *gate* them.
- Determinism: both the C engine RNG and the Python `default_rng` must be seeded identically for reproducibility.

---

## 10. Target hardware (resuming on an RTX 3090, 24 GB VRAM)

- **System RAM matters more than VRAM for the verify loop.** The ~6 h `engine_step` compile is a single-thread **CPU** XLA-HLO pass whose working set hit **~43 GB of system RAM** (not VRAM). Make sure the 3090 box has **≥48–64 GB system RAM**, or you'll thrash/OOM during compile. **Shrinking the selection-mask bound (§6.1) cuts both compile time *and* compile RAM** — do it first; it's the single biggest unblock on a workstation.
- **24 GB VRAM bounds batch size, not correctness.** Per-env state is tiny, but the batched rollout (`vec.num_envs` / PPO `batch_size`) plus the compiled executable's intermediate buffers must fit 24 GB. Start conservative (e.g. a few hundred envs) and scale up; the env itself is GPU-resident so throughput should still dwarf the C env.
- Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` (already set in `run_verify_single.sh`) so JAX doesn't grab all 24 GB up front — important when the C engine/binding also runs in-process for the parity tests.
- Compile *time* is GPU-independent (CPU-bound HLO), so a 3090 won't be slower to compile than the pod — but it also won't be faster. The mask-shrink is the only real lever.
- Verify CUDA/jaxlib match the 3090 (Ampere, sm_86): install the `jax[cuda12]` wheel matching the box's CUDA, and run `scripts/wait_for_cuda.sh` semantics aren't needed off-docker. Eager debugging (`JAX_PLATFORMS=cpu`) needs no GPU at all.
