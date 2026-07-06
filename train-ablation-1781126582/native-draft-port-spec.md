# Native deck-building port — implementation checklist (working doc)

Status legend: [ ] todo, [x] done. Keep updated; this is the compaction-proof plan.

## Python side (DONE unless noted)
- [x] observation.py: `_TrainingDeckContextObservationData` (mode i32, gate i16, leader i16,
  main_ids i16[50], main_count u8, candidate_count i32, cand_ids i16[1024], cand_copies u8[1024])
  + `_TrainingObservationDataDeckBuild` (base fields + deck_context appended) +
  DECKBUILD_OBSERVATION_CTYPE/STRUCT_SIZE (=12,604; battle stays 9,416).
- [x] tcg_policy.py: native-layout `deck_context_enabled` from dtype names (was hardcoded False);
  `_canonical_from_packed` deck_context branch (8 fields); return via `cobs` variable.
- [x] trainer.py + league_training.py: game-granular row grouping via
  `driver_env.agents_per_match` (fallback num_agents). Fixes league-on-native worker-granularity
  bug AND base-trainer reward-decomposition/win-prob seat bug for native.
- [x] azk_native.py: `agents_per_match = MAX_PLAYERS_PER_MATCH` class attr.
- [ ] azk_native.py: deck_building mode — instance `emulated` dict with DECKBUILD dtype,
  obs space size, vec_init kwargs (deck_building=1 + catalog arrays + world-seed variance),
  drain + metrics + snapshots in step() when terminals fire.
- [ ] deckbuild_metrics.py (new): per-episode record → full legacy key surface:
  deckbuild/* + azk_step_deckbuild/* (mirror), deckbuild_gatecard/<code>/*,
  deckbuild_result/* (incl gatecard behavior pass-through + ability_rate sum). Mean-aggregate
  over drained records → merge into vec_log dict. Snapshot JSONL record: {ts, episode, seed,
  players:[{gate, leader, main:{code:qty asc def id}, avg_cost, win, attack_rate, spell_rate,
  weapon_rate, portal_rate, play_entity_rate, noop_rate, episode_length, leader_health}]}.
- [ ] training_utils.make_azuki_env: allow native+deck_building; build catalog via
  build_deck_build_catalog(deck_pool); pass flattened arrays (below) + snapshot cfg into
  AzukiNativeEnv.
- [ ] league_eval.py / evaluate_checkpoint / draft_vs_reference_eval: force env.native=false in
  their private vecenv args (they are legacy-wrapper-dependent; policy weights are path-agnostic).
- [ ] Parity tests: test_native_deckbuild_equivalence.py
  (a) semantic: forced gates (AZK_DEBUG_FORCE_GATE_DEF_IDS="a,b") + identical pick sequence on
      DeckBuildingParallelEnv vs native → compare deck_context fields + masks each draft step +
      first battle obs deck_context; (b) encode parity via the existing methodology (drive
      native only, observation_to_dict + emulate → legacy bytes; needs observation_to_dict
      deck_context extension OR direct dict-build helper); (c) metric key-surface equality vs
      legacy _deckbuild_metrics_for_player on a synthetic record.

## Catalog arrays passed Python→C (vec_init kwargs; ordering = wrapper's catalog builder)
- draft_gate_def_ids: unique gate def ids [G] (order: first appearance? use sorted for
  determinism — C only needs lookup; candidate lists are per-gate)
- draft_gate_population: int list [P] of gate def ids (multiset, pool order; P=18 today)
- draft_leader_flat / draft_leader_offsets[G+1]: per-gate leader candidate def ids (asc)
- draft_main_flat / draft_main_offsets[G+1]: per-gate main candidate def ids (asc, elem+NORMAL)
- draft_ikz_def_id: int (IKZ-001 def id = 0)
- deck_building: 1

## C side (python/src/tcg.h + env_binding.h + binding.c)
- [ ] tcg.h: AzkTrainingDeckContextData + TrainingObservationDataDeckBuild structs mirroring
  ctypes EXACTLY (same order/types; default alignment; no packing). Static asserts on sizeof.
- [ ] CAzukiTCG: deck_building flag; draft state per player {gate_def_id, leader_def_id,
  main_ids[50], main_count, copies[MAX_CAND]}; draft_active flag; draft_active_player;
  draft_rng_state; catalog pointers (global per process, set at vec_init); export record +
  valid flag {seed, per player: gate, leader, main[50], win, behavior rates(8), }.
- [ ] c_reset draft branch: destroy engine→NULL; advance draft_rng 2× → gates via population;
  init states; draft_active=1; active=0; fill_draft_observations. World seed for battle:
  advance(draft_rng) captured at reset as episode_world_seed.
- [ ] c_step draft branch (before tcg_active_player_index): validate action
  (type==3, sub2/3==0, sub1<filtered count → abort with message otherwise); resolve filtered
  index→def id by scanning full per-gate list skipping copies>=4; apply (leader if unset else
  main append + copies++); alternate active player (other-if-incomplete else self); if both
  complete → assemble CardInfo decks (leader,gate,mains asc def id,IKZ×10) →
  azk_engine_create_with_decks_and_starting_player(episode_world_seed, next_starting_player,
  ...) → draft_active=0 → refresh_observations + postprocess; else fill_draft_observations.
  NO tick++ during draft (tick budget = battle only). Rewards zeroed; terminals untouched.
- [ ] fill_draft_observations: memset rows; set empty-obs sentinels matching
  empty_training_observation() (card_def_id=-1 zones, ability active_player_index=-1,
  source_card_def_id=-1, critic_privileged -1, combat -1 fields...) — verify against
  observation.py:353-418 field by field; deck_context per player (mode: own state
  1=leader,2=main,0=complete; candidates only for active); action_mask (active only:
  primary[3]=1, count=n, legal_primary[:n]=3, legal_sub1[:n]=i, sub2/3=0).
- [ ] battle-step postprocess (deck-building only): after refresh_observations →
  write deck_context (mode=0, pick-order main log, no candidates) + sanitize
  critic_privileged self_deck/opponent_deck to card_def_id=-1 (check zone_index value from
  _copy_with_sanitized_privileged_decks) for BOTH players. Assert native mask never exposes
  primary 3 during battle.
- [ ] obs strides: deck-building envs write TrainingObservationDataDeckBuild rows;
  refresh into local TrainingObservationData tmp[2] then copy into .base of each row.
- [ ] episode end: record_episode_stats as-is + fill export record (win from log/p{i}_winrate
  equivalents — use same sources as record_episode_stats; behavior rates from
  episode_action_* counters; episode_length=tick; leader_health from snapshot).
- [ ] env_binding.h vec_init: parse new kwargs (int lists) into process-global catalog;
  deck_building flag per env; obs stride awareness (numpy row size = deckbuild struct).
- [ ] binding.c: vec_drain_deck_records(handle) → list of per-episode dicts; clears flags.
  Export sizeof constants: binding.OBS_STRUCT_SIZE / DECKBUILD_OBS_STRUCT_SIZE for runtime
  layout assert in azk_native.
- [ ] AZK_DEBUG_FORCE_GATE_DEF_IDS="a,b" env hook in draft gate sampling (parity tests).

## Gotchas (from mapper reports — do not violate)
1. Draft strictly ALTERNATES P0/P1 each pick (offset-1-prefers-other); completed-early player
   shows deck_context.mode=0 while episode still building (step 101).
2. sub1 indexes the FILTERED candidate list (maxed-out cards removed, list reindexes).
3. main_card_def_ids obs field = pick-order log (-1 padded); deck spec for engine = unique
   (code,qty) asc def id. Two representations.
4. Gate sampling = uniform over multiset population (pool-frequency weighted: 3/18 for
   STT01-002 & STT02-002, 2/18 others; pool=18 decks incl 2 starters).
5. Legacy per-episode np PCG64 chaining is NOT ported (distribution-level parity only);
   parity tests force gates + picks.
6. vec_log invariant: Log struct all-floats; my_log key names unprefixed (p0_*/p1_*) on native.
7. Native uint8 legal_sub arrays cap candidate_count at 255 (today ~80; assert).
8. league granularity fixed to agents_per_match=2 (game-level); inline league evaluator +
   draftref evals stay on legacy serial path (force env.native=false).
9. cuda_graphs stays OFF for deck-building v1 (league row-splitting vs static capture).

## Bench/launch plan after parity
- rebuild, run test_deck_building.py + test_tcg_policy.py + new parity test + ctest
- SPS ladder on azuki_deckbuild_native_3090.ini: baseline → +compile → mb 4096→8192
- soak 2M steps w/ league (promotion at epoch ~100 must survive)
- re-run round-2 chain (run_round2.sh switched to native config): ctrl2/anneal1/gateid1/combo1

## League flat-SPS patch (task 7 — APPLY ONLY AFTER combo1 EXITS; fresh arm
## processes import league_training.py from disk)
Diagnosis (user-confirmed): per-distinct-frozen-policy forward splitting in
LeaguePuffeRL._infer_actions fragments the launch-bound 4096-LSTM forward; with
pool 4 every step pays ~4 tiny extra forwards → 3.7k→1.6k SPS sag.
Fix v1 (OSFP-style windowed sampling — cost independent of pool size):
1. league_training.py LeagueConfig: add `frozen_window_epochs: int = 0` (0 = legacy
   per-game draws) and `max_distinct_frozen: int = 1`.
2. LeaguePuffeRL.__init__: `self._window_policy_ids = None; self._window_index = -1`.
3. New `_refresh_frozen_window(self)`: window_index = epoch // frozen_window_epochs;
   on change (or pool change) draw K=min(max_distinct_frozen, pool) distinct ids via
   self._rng.choice(replace=False). Call at top of evaluate(). Validate ids < pool len.
4. _resample_matchups: final draw uses `self._window_policy_ids` when not None
   (indices into that array → policy ids), else legacy uniform-over-pool.
   NOTE: assignments only change for FINISHED envs (existing behavior) so window
   transitions are gradual over ~1 episode; per-policy LSTM stores unchanged.
5. set_opponent_policies: reset `_window_index=-1; _window_policy_ids=None` BEFORE
   its _resample_matchups call (pool indices shift on refresh).
6. train.py:1619 LeagueConfig(...): plumb frozen_window_epochs=int(league_cfg.get(
   "frozen_window_epochs", 0) or 0), max_distinct_frozen=int(league_cfg.get(
   "max_distinct_frozen", 1) or 1).
7. azuki_deckbuild_native_3090.ini [league]: frozen_window_epochs = 8,
   max_distinct_frozen = 1.
8. Unit test in python/tests/test_league_training_utils.py: with window set,
   _resample_matchups only assigns ids from the window; window redraws on epoch
   boundary and on set_opponent_policies.
9. Benchmark: short 3M arm with pool pre-seeded (resume ctrl2 league dir or ckpt
   interval 40) — expect ~3.2-3.5k flat at pool 4 vs 1.6-2k before.
Secondary (later): single resident opponent-slot module with weight swapping
(saves 0.8GB × pool); CUDA-graph recapture with static learner/frozen row split;
lighter promotion evals (fewer episodes / larger intervals).
