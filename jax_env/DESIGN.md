# Azuki TCG — JAX Environment Design

Goal: GPU-resident, vmappable JAX translation of the battle environment that is
*behaviorally equivalent* to the C engine (per Karten et al. 2026, 4-level
hierarchical verification), then integrated into training for higher SPS.
The C engine remains the source of truth for gameplay/inference.

## Baselines (A10, this host, 2026-06-11)

| Metric | Value |
|---|---|
| Raw C engine step (1 core, random legal actions) | 8,204 steps/s (122 µs) |
| Full Python wrapper step (`observation_to_dict`) | 669 steps/s (1,496 µs) |
| End-to-end training SPS (deckbuild smoke, 32 envs / 8 workers, LSTM-4096) | 107.6 |

## Results so far (2026-06-12)

Verification: **13/13 tests pass** — L1 setup bit-exactness; L2 full-episode
semantic + mask-order parity (vanilla decks, hundreds of steps incl. combat,
interception, weapons, IKZ token, EOT cleanup); L3 env-level parity incl.
shaped/terminal rewards (±2e-5) over complete episodes.

JAX env throughput (full step: masks + auto-resolve + rewards + auto-reset):

| batch | env-steps/s |
|---|---|
| 32 | 4,377 |
| 256 | 23,028 |
| 1,024 | 50,040 |
| 4,096 | 66,884 |
| 8,192–16,384 | ~68,500 (ceiling) |

vs the C path training actually pays (669/s/core ⇒ ~5.4K/s at 8 workers):
**~12.5× env-throughput uplift**, with observations born on-GPU.

Coverage: vanilla rules complete; ability runtime (FSM, targeting, masks)
verified; **59 of 118 pool-reachable ability cards ported and episode-verified
vs C** (tests: 40/40 green). Production-pool fidelity: 0.43 unimplemented-
ability hits/episode under random play (was ~18 pre-abilities). Remaining 59
cards by blocker: 23 selection-zone, 16 leaders/gates, 9 passive observers,
5 when-takes-damage, 2 when-equipped, 4 other. C engine remains ground truth
+ the gameplay/inference engine.

Trainer integration: `vec.backend=Jax` (`azk_puffer/jax_vector.py`, byte-exact
emulation obs packing in `azuki_jax/observe.py`); end-to-end training at
~530 SPS vs 308–435 matched C config (107.6 production baseline); env share
of wall time ~20× smaller. Config: `python/config/azuki_jax_smoke.ini`.

Ops note: long multi-test JAX processes accumulate CUDA graphs on the A10 and
OOM — run heavy pytest files in separate processes or set
`XLA_FLAGS="--xla_gpu_enable_command_buffer="`.

92% of per-step env cost is the Python dict conversion layer; the trainer
additionally pays pufferlib emulation packing + multiprocessing IPC. The JAX
env eliminates all of these (obs produced packed, on GPU).

## Reference contract (what must match, byte-for-byte)

The production env is `python/src/binding.c` + `python/src/tcg.h` (v1) wrapping
`azk_engine_*`. One env step =

1. If both terminals set → `c_reset` (next episode; consumes starter+deck RNG).
2. If active player's `legal_action_count == 0` → zero-legal truncation.
3. Submit active player's action `[type, sub1, sub2, sub3]` (must be legal — C aborts otherwise).
4. Auto-tick until `requires_action` or game over (queues → phase gate → ecs_progress ladder).
5. Refresh both players' `TrainingObservationData` (incl. per-player legal action mask, ≤1024 tuples).
6. Rewards: terminal ±5 / truncation (timeout 0.35, auto-tick 0.60, zero-legal) with leader/board
   edge terms / shaped PBRS (tanh phi delta × time_weight(0.95^t) + 1.25·leader_edge_delta +
   0.35·board_edge_delta − 0.02·noop_with_alternatives), acting player +r, opponent −r.

Episode-level RNG: `starter_rng_state` / `deck_rng_state` advance once per reset via
xorshift32 (`advance_episode_seed`); engine RNG (`GameState.rng_state`) seeds from
the env seed and is consumed *only* by deck shuffles (xorshift32 Fisher–Yates,
deck top = array end; draws pop from the end).

Reachable card universe in training: 137 codes (16 pool decks + 2 starter decks);
full def table has 193. Decks are 62 cards: 50 main + leader + gate + 10 IKZ.

## Package layout (`jax_env/azuki_jax/`)

- `constants.py` — sizes, enums (phases, action types, zones, target types) mirroring C.
- `cards.py` — card def tables (193) as jnp constant arrays, generated from
  `src/generated/card_defs.c` by `tools/gen_card_tables.py`.
- `abilities/registry.py` — per-card ability flag/cost/timing/target tables
  (mirrors `kAbilityRegistry` + additional registry).
- `abilities/effects.py` — per-ability JAX implementations (validate / apply_costs /
  apply_effects / on_cost_paid / on_selection_complete) dispatched via `lax.switch`.
- `state.py` — `State` pytree: fixed-size arrays only (below).
- `rng.py` — xorshift32, Fisher–Yates deck shuffle, seed mixers (bit-exact vs C).
- `setup.py` — deck expansion → instance arrays, game init, mulligan deal, engine stabilize.
- `masks.py` — legal-action enumeration in exactly the C enumerator's order.
- `engine/` — action application (26 types), combat, phase systems, ability FSM,
  auto-resolve loop (`lax.while_loop` over the C tick ladder).
- `observe.py` — TrainingObservationData as a dict of arrays + byte-packing to the
  exact ctypes struct layout (for L3 byte-compare and trainer integration).
- `rewards.py` — PBRS phi / shaped / terminal / truncation (float32, same formulas).
- `env.py` — `reset(seed)` / `step(state, actions)` (single env), `vmap`+`jit` batch API,
  auto-reset-on-done semantics identical to training.

## State representation

Per player, **63 card instances** (50 deck + 1 leader + 1 gate + 10 IKZ + 1 token slot),
struct-of-arrays, indexed `(player, instance)`:

- `def_id:int16`, `zone:int8` (DECK/HAND/GARDEN/ALLEY/LEADER/GATE/IKZ_PILE/IKZ_AREA/
  DISCARD/SELECTION/ATTACHED/ABSENT), `zpos:int8` (order within zone; compacts on removal
  like flecs ordered children; deck top = highest zpos)
- `tapped:bool`, `cooldown:uint8`, `cur_atk:int8`, `cur_hp:int8`
- buff aggregates: `atk_buff_perm/eot`, `hp_buff_perm/eot`, `carapace_buff_perm/eot`,
  `cmb_in_perm/eot`, `cmb_out_perm/eot` (one-shot grants; passive/static buffs are
  *recomputed* from board state each step — replaces the C observer/passive-queue machinery)
- status: `frozen_dur`, `shocked_dur`, `effect_immune_dur` (int8), keyword grant flags +
  timed tag grants (small fixed table), `sacrifice_at_eot:bool`
- `attached_to:int8` (weapon host instance or −1), `weapon_slot_order`
- damage tracker subset needed by abilities; once-per-turn used flags per ability slot

Game-level: phase, active/starting player, turn_number, mulligan_progress, winner,
`rng_state:uint32`, per-turn counters[2], cost reduction[2], combat state
(attacker/defender instance refs, intercepted), ability FSM block (phase, source ref,
owner, optionality, cost/effect target state ≤8, selection zone ≤50 + picks,
scratch ints, saved_active_player), triggered-effect queue (16), recent actions
(2×4×6), tick, time_weight, last_phi[2], last_snapshot, episode_returns[2],
action counters (for log parity), `starter_rng_state`, `deck_rng_state`,
`current_deck_indices[2]`, completed_episodes.

Deck pool: constant `(num_decks, 62)` int16 def-id table (expanded, in C iteration
order) broadcast via `in_axes=None`.

## Verification plan (paper §Methodology)

- **L1**: per-module pytest vs the C binding (RNG/shuffle bit-exactness; setup deal;
  mask enumeration on crafted states; combat math property tests; obs packing).
- **L2**: composed scenarios (play→trigger→combat→death→EOT cleanup) driven through
  both engines from identical seeds with scripted action sequences.
- **L3**: ≥100 full episodes, matched seeds + uniform-random legal actions chosen by
  shared index (mask order parity makes index choice equivalent), byte-diff of obs
  structs + rewards + terminals/truncations each step. Plus pool-deck coverage sweep
  (every deck pairing) and per-card activation coverage tracking.
- **L4-lite**: train briefly in JAX env, eval vs C env (and the reverse) to confirm
  no sim-to-sim gap; full L4 equivalence testing after integration.

## Performance plan (paper Appendix B.1, in order)

Fixed-size arrays (done by construction) → branchless `jnp.where`/`lax.switch` →
`vmap` over envs (card tables `in_axes=None`) → `jit` outer step → measure at
batch 32/128/512/2048/8192 → minimal dtypes (int8/int16) → obs packed in-kernel.
`lax.scan` rollout fusion is N/A for torch-policy integration but used in the
pure-JAX SPS benchmark.

## Integration

Replace the Multiprocessing vec with a `JaxVecEnv` exposing the azk_puffer vector
interface: one jitted call steps all B envs; obs emitted directly in the packed
layout the policy consumes (bypassing PettingZoo emulation + observation_to_dict);
dlpack GPU→torch where possible. Deck building remains a thin wrapper (ported to
vectorized form) ahead of battle, as today. Training config gains a `[vec] backend=Jax`
path with `num_envs` ≫ current (e.g. 1024–8192) and adjusted batch sizes.
