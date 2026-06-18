# JAX Environment Conversion — Handoff & 1:1 Verification Status

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

#### 5.1.1 REFINED mechanism (session 2, 2026-06-14 — supersedes the §5.1 "deferred-insert entry-lag" framing)

Step-correlated `DBG_PASSIVE` traces (`jax_env/tests/trace_019.py`, `trace_jax_019.py`, both new) for AZK01-019 in c10/m16 pin the real C rule. It is **observer-registration-scoped**, not a simple deferred-insert:

- **First-EVER entry into play fires immediately.** When a self-buff passive is played for the first time (garden OR alley), C registers its observer and the init update evaluates with the card **in play** → the buff applies that step. Evidence (m16 s12361 step 44): `[Cjay] ent966 p0 in_play all_normal=1 -> APPLY +0/+2`, garden hp 3 at step 45 — **no lag**. So the §5.1 "fresh garden play lags" claim is WRONG; first plays do not lag.
- **Re-entry after leaving play does NOT re-fire at entry, but the observer persists and re-fires on the NEXT garden/alley event.** `azk_sync_card_abilities` skips `init_passive_observers` while `PassiveObserverContext` exists (`components/abilities.c:344-346`). Evidence (c10 s7010): ent for AZK01-019 buffed during its first stint (alley step47 → garden move step52, hp 3), hp-buff correctly removed at step 89 (garden no longer all-Normal), card **leaves play** step 90, **re-enters** garden step 95 reading **hp 1 (unbuffed)** with NO `[Cjay]` firing 93–98, then **re-buffs to hp 3 by step 101** (a garden event between 98–101 re-fired the persisting observer). So a re-entry lags exactly until the next watched garden/alley event.
- **Zone MOVE of an in-play card keeps its buff** (alley→garden gate-portal): observer persists, its EcsOnAdd fires → re-evaluates. Not a leave/re-entry.
- Watched zones (re-confirmed from each card's `*_init_passive_observers`): AZK01-010/019 = own garden+alley; **AZK01-073 = own garden ONLY** (single observer, `azk01_073.c:95`); STT02-012 = both players' gardens.

**Two heuristic attempts this session, both incomplete (reverted):** (a) a `passive_armed` latch armed at every garden/alley mutation hook — over/under-applied because the JAX mutation-hook set does not exactly equal C's ChildOf-observer firing set; (b) a `passive_left_play` "dead after first leave" gate — fixed the c10 step-98 over-apply (advanced to 101) but then under-applied at 101 because **C re-buffs after re-entry** (the observer is NOT permanently dead). Conclusion: the correct port is the full **event-driven sticky queue** (re-entry lags one event, then re-arms), and getting it bit-exact requires matching C's observer firing set precisely — the genuinely hard "multi-hour" part. STT02-012's c16/m16 divergence is the **separate** count-latch multi-event-netting bug (012 buffed at step 62, C removes at 69 via the last-event-not-landing flush split; JAX latch keeps the last decision).

**Non-passive engine is otherwise verified 1:1**: a single-process jit suite run reached 47 tests (l1/l2/l3-abilities + batch1-4, which DO include AZK01-010/019/073/012 crafted decks) with **0 failures** before being OOM-killed by a concurrently-launched benchmark (run the suite ALONE — its compile peaks ~40 GB and the bench compile adds another ~40 GB > 125 GB). The passive divergences are specific to certain production-pool pairings (fullpool cases ~c10/c15/c16/m16), not the batch decks.

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

---

## 10. Target hardware (resuming on an RTX 3090, 24 GB VRAM)

- **System RAM matters more than VRAM for the verify loop.** The ~6 h `engine_step` compile is a single-thread **CPU** XLA-HLO pass whose working set hit **~43 GB of system RAM** (not VRAM). Make sure the 3090 box has **≥48–64 GB system RAM**, or you'll thrash/OOM during compile. **Shrinking the selection-mask bound (§6.1) cuts both compile time *and* compile RAM** — do it first; it's the single biggest unblock on a workstation.
- **24 GB VRAM bounds batch size, not correctness.** Per-env state is tiny, but the batched rollout (`vec.num_envs` / PPO `batch_size`) plus the compiled executable's intermediate buffers must fit 24 GB. Start conservative (e.g. a few hundred envs) and scale up; the env itself is GPU-resident so throughput should still dwarf the C env.
- Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` (already set in `run_verify_single.sh`) so JAX doesn't grab all 24 GB up front — important when the C engine/binding also runs in-process for the parity tests.
- Compile *time* is GPU-independent (CPU-bound HLO), so a 3090 won't be slower to compile than the pod — but it also won't be faster. The mask-shrink is the only real lever.
- Verify CUDA/jaxlib match the 3090 (Ampere, sm_86): install the `jax[cuda12]` wheel matching the box's CUDA, and run `scripts/wait_for_cuda.sh` semantics aren't needed off-docker. Eager debugging (`JAX_PLATFORMS=cpu`) needs no GPU at all.
