# JAX Environment Experiments

## 2026-06-18 — Continuation kickoff

Sources read:
- `jax_env/HANDOFF.md` current frontier: deterministic `JaxVecEnv(4, seed=1)` split path first generic fallback is bench step 60, one attack row involving active player 1 leader `AZK01-119` attacking player 0 garden `STT02-003`, with attached `STT01-012`; recommended next work is a narrow `STT01-012` attached-weapon attack response fast path and vector harness plumbing.
- arXiv 2603.12145: use hierarchical verification (component, interaction, rollout, cross-backend policy transfer) and require matched-seed rollout parity before throughput claims.

Current repo observation:
- `git status --short` was clean before this file was created.
- `jax_env/experiments.md` did not exist and was created for persistent progress tracking.

Next actions:
- Inspect `jax_env/azuki_jax/step.py` and `python/src/azk_puffer/jax_vector.py` for the documented partial `step_attack_stt01_012_response_fast` work.
- Syntax-check the documented partial before deciding whether to keep it.
- Reproduce or advance the step-60 split-action frontier with conservative host masks only.

## 2026-06-18 — STT01-012 partial inspection

- Found `step_attack_stt01_012_response_fast` in `jax_env/azuki_jax/step.py`.
- Confirmed no `STT01-012` vector harness plumbing exists yet in `python/src/azk_puffer/jax_vector.py`.
- Ran syntax check:
  `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`
- Result: passed.
- Decision: keep the helper and add a conservative vector mask/wrapper/dispatch for only the observed attached-weapon leader-attack-to-garden shape.

## 2026-06-18 — Completion audit frame

Concrete deliverables for the active goal:
- JAX environment matches the C engine under matched seeds/actions: full state, legal masks, rewards, terminals, and no unimplemented ability sentinel.
- Authoritative parity evidence is `jax_env/tests/` with `test_l3_fullpool.py` as the final gate: 36 production pool cases, 600 steps each, state + mask parity.
- Split/vector trainer route must avoid generic monolithic compiles on benchmarked trajectories; current immediate frontier is deterministic `JaxVecEnv(4, seed=1)` step 60.
- Experiments must be recorded in this file as source-of-truth progress.
- Final performance evidence must include sim-only SPS for C and JAX and end-to-end training SPS for C and JAX after compilation/warm-up.

Current state check:
- `git status --short` reported `M python/src/azk_puffer/jax_vector.py` and untracked `jax_env/experiments.md`.
- `step_attack_stt01_012_response_fast` and vector plumbing for `_attack_stt01_012_response_fast_mask` are now present; next step is syntax/trace validation and mask correctness review.

## 2026-06-18 — STT01-012 mask review fix

- Ran syntax check again after current vector wiring:
  `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`
- Result: passed.
- Static review found one unsafe host-mask admission: `_attack_stt01_012_response_fast_mask` counted any board response ability as sufficient, but board response availability also depends on validation/cost/once/frozen state.
- Fix applied in `python/src/azk_puffer/jax_vector.py`: the STT01-012 response mask now only treats payable hand responses or untapped defender cards as proven response-window reasons; board-only response abilities fall back to generic/specialized paths until explicitly modeled.

## 2026-06-18 — Step 60 STT01-012 no-response correction

- Reproduced deterministic `JaxVecEnv(4, seed=1)` pre-step-60 state.
- Attack rows:
  - Env 1: `[6, 3, 5, 0]`, attacker `STT02-001`, handled by existing `attack_leader_response`.
  - Env 3: `[6, 5, 3, 0]`, attacker leader `AZK01-119`, defender garden `STT02-003`, attached `STT01-012`.
- Env 3 had `STT02-016` response spells in hand but `payment_sources=0`; generic should mill STT01-012 then skip response and resolve clean combat.
- Updated `step_attack_stt01_012_response_fast` to run `phase_gate` after the mill and then `combat_resolve` when no queued response/trigger work remains. This covers both response-window and no-response combat outcomes under the host mask.
- Updated `_attack_stt01_012_response_fast_mask` to require clean combat trigger/death conditions instead of requiring a host-proven response.
- Validation:
  - `py_compile` passed for `jax_env/azuki_jax/step.py` and `python/src/azk_puffer/jax_vector.py`.
  - `git diff --check` passed for `jax_env/azuki_jax/step.py`, `python/src/azk_puffer/jax_vector.py`, and `jax_env/experiments.md`.
  - Pre-step-60 mask probe now reports `mask [False, False, False, True]`.
  - Step-60 trace now reports `effect_stt01_017=1 attack_leader_response=1 attack_stt01_012=1 generic=0`.

## 2026-06-18 — Step 62 response fizzle fast path

- After closing step 60, the next deterministic split frontier was step 62: `generic=1 generic_types=0`.
- Probe showed two NOOP rows:
  - Env 0 main-phase NOOP was already `main_noop_simple=True`.
  - Env 2 response-window NOOP had `combat_attacker=2`, `combat_defender=0`, `combat_defender_player=1`; the attacker was no longer in a valid battle zone, so generic combat resolution fizzles.
- Added `step_response_noop_combat_fizzle_fast` and vector plumbing/mask/trace counter for response-window NOOP where a pending combatant is invalid and no when-attacked trigger is queued.
- Probe now reports `fizzle [False, False, True, False]` at step 62, alongside existing `main_noop_simple` and `gate_simple` coverage.

## 2026-06-18 — Step 67 STT02-009 no-confirm alley play

- After closing step 62, the next deterministic split frontier was step 67: `generic=1 generic_types=2`.
- Probe showed env 3 action `[2, 7, 0, 0]`; action hand index `7` resolved by hand `zpos` to `STT02-009`, played to alley slot 0.
- Existing `play_stt02_009_confirm` correctly rejected the row because there was no friendly garden entity with IKZ cost >= 2, so Aya's on-play validate is false and no confirmation should open.
- Updated `_play_entity_simple_fast_mask` to treat `STT02-009` as simple only when its post-play on-play cost target is unavailable. Garden plays still trigger the dedicated confirm path because the played STT02-009 itself is a valid garden cost target.
- Probe now reports `play_alley_simple [False, False, False, True]` for step 67.

## 2026-06-18 — Step 69/70 AZK01-014 and Stonehaven gate

- After closing step 67, the next deterministic split frontier was step 69: `generic=1 generic_types=6`.
- Probe showed env 3 action `[6, 2, 5, 0]`: player 1 `AZK01-014` in garden slot 2 attacking player 0 leader `AZK01-125`, with another friendly garden entity available for Trade Guild Cavalry's When Attacking buff.
- Added `step_attack_azk01_014_effect_fast`, `step_effect_azk01_014_fast`, host masks, JIT wrappers, dispatch, and trace counters. The attack mask probe reports `attack_azk01_014 [False, False, False, True]`.
- After that, the next frontier was step 70: `generic=1 generic_types=10`.
- Probe showed env 1 action `[10, 0, 4, 0]`: `STT03-002` gate portals `AZK01-105` to garden slot 4. The portaled card itself is not a Stonehaven target (`base_hp=3`, gate power 2), but existing friendly `STT02-004` is a valid base-HP-2 target.
- Updated `step_gate_portal_simple_fast` and `_gate_portal_simple_fast_mask` so `STT03-002` opens its effect when any post-portal friendly garden entity is a valid Stonehaven target, not only when the portaled entity is the target.
- Probe now reports `gate_simple [False, True, False, False]` at step 70.

## 2026-06-18 — Step 73 STT04-016 spent AZK01-059 cost

- After closing step 70, the next deterministic split frontier was step 73: `generic=1 generic_types=13`.
- Probe showed env 0 in `STT04-016` cost selection: action `[13, 1, 0, 0]` targeting friendly garden slot 1 `AZK01-059`.
- The target had `hp=1` and `once_per_turn_used=1`; the existing STT04-016 cost mask only admitted nonlethal fresh `AZK01-059` trigger shapes or `STT04-003`.
- Updated `step_select_cost_stt04_016_fast` to handle the spent/lethal `AZK01-059` cost case by applying the discard directly instead of routing through `deal_effect_damage`, which would queue the already-spent takes-damage trigger.
- Updated `_select_cost_stt04_016_fast_mask` to admit that narrow spent/lethal shape when it has no destroy/protection complications.
- Probe now reports `cost_stt04_016 [True, False, False, False]` at step 73.

## 2026-06-18 — Step 74 optional STT04-016 skip and post-trigger response pass

- After closing step 73, the next deterministic split frontier was step 74: `generic=2 generic_types=0`.
- Probe showed:
  - Env 0 main-phase `NOOP` while `STT04-016` was in effect selection (`ab_phase=3`, `ab_eff_min=0`, `ab_eff_max=1`) after the spent/lethal `AZK01-059` cost. This is the optional effect skip.
  - Env 3 response-window `NOOP` for `AZK01-014` attacking a leader after its When Attacking buff had already resolved.
- Extended `step_effect_stt04_016_fast` and `_effect_stt04_016_fast_mask` so `NOOP` skips the optional effect and clears the context.
- Relaxed the response-window leader-combat NOOP mask to not reject already-resolved `When Attacking` sources; it still rejects future `After Attacking`, damage triggers, and `AZK01-044` weapon shock shapes.
- Probe now reports `effect_stt04_016 [True, False, False, False]` and `response_leader [False, False, False, True]` at step 74.

## 2026-06-18 — Current split frontier after step 74 fixes

- Re-ran the concise frontier trace after the step 74 fixes.
- Current deterministic `JaxVecEnv(4, seed=1)` split frontier:
  - `frontier step=79 generic=1 generic_types=8`
  - Mask summary at frontier: `play_simple=1 main_noop=1`; the generic row is a spell play.
- Probe showed env 1 action `[8, 1, 0, 0]`; hand `zpos=1` is `STT02-017` (`Shao's Perseverance`).
- Board state includes active player 1 `STT02-010` in garden. Because `return_to_hand` can queue STT02-010's when-returned optional draw observer, a narrow STT02-017 fast path must either model that follow-up or conservatively reject this row.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - `git diff --check -- jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py jax_env/experiments.md` passed.

## 2026-06-18 — Step 79 through 91 split frontier advance

- Added `step_play_spell_stt02_017_fast` and vector plumbing for Shao's Perseverance.
  - The helper pays/discards the spell, returns low-cost opponent garden entities, and begins the first queued `STT02-010` when-returned optional trigger.
  - Step-79 mask probe reported `stt02_017 [False, True, False, False]`.
- Fixed `step_confirm_clear_fast` so declining an optional confirmation with queued `STT02-010` triggers immediately begins the next queued trigger instead of exposing a bogus main-phase action while `trig_count > 0`.
- Added `step_play_azk01_028_fast` and vector plumbing for Soryu no Rin's garden play:
  - pay/place, discard remaining hand, return all other garden entities for both players, recompute passives.
  - Step-80 probe then reported `azk01_028 [False, False, False, True]` and the previously stuck main NOOP row became `main_simple [False, False, True, False]`.
- Added `STT01-001` activation/effect fast paths:
  - activation pays 1 IKZ and opens charge target selection;
  - effect grants Charge until end of turn to a friendly garden entity with an equipped weapon and cooldown.
  - Step-84 probe reported `activate_stt01_001 [True, False, False, False]`.
- Relaxed `main_noop_stt04_003` to allow persistent attached weapons during the clean one/two-Seer start-each path, and allowed `STT02-012` alley plays through the simple entity path because its passive is garden-only.
- Current deterministic `JaxVecEnv(4, seed=1)` split frontier:
  - `frontier step=93 generic=1 generic_types=6`
  - Probe row: env 0 action `[6, 5, 0, 0]`, active player 0 leader `AZK01-121` attacks player 1 garden slot 0 `AZK01-097`.
  - Existing `attack_leader_simple` only handles attacks into leaders; existing `attack_entity_mutual_destroy` only handles garden attackers. Next work is a clean leader-attacks-garden fast path, likely by declaring combat then running `phase_gate`/`combat_resolve` under a no-trigger/no-response mask.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - `git diff --check -- jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py jax_env/experiments.md jax_env/HANDOFF.md` passed.

## 2026-06-19 — Step 97 through 105 split frontier advance

- Rebuilt the deterministic frontier probe around `JaxVecEnv(4, seed=1)` with split-action generic dispatch trapped before fallback execution.
- Step 97 was `STT02-009` selecting a returning `STT02-010` as its cost target. The fast mask now allows the one returning `STT02-010` case, and the fast effect clears context only after draining the queued when-returned trigger.
- Step 99 was `AZK01-097` resolving a top-5 reveal with no matching weapon. The reveal mask now allows the no-pick branch; the existing fast helper discards the reveal and clears context.
- Step 101 was `AZK01-017` spell play. Added `step_play_spell_azk01_017_fast`, `step_effect_azk01_017_fast`, and vector dispatch/masks for clean optional one-garden plus one-leader damage selection.
- Validation observed so far:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-017 plumbing.
  - Probe after AZK01-017 dispatch advanced to `GENERIC step=105 action_type=2`: active player 0 plays `AZK01-006` from hand to alley while `STT02-012` sits in alley.
- Current working hypothesis: the generic at step 105 was a too-conservative simple entity mask; `STT02-012` in alley is passive-inactive, so it should not block clean play. Mask updated; next probe must confirm the frontier advances beyond step 105.

## 2026-06-19 — Step 107 split frontier

- Probe after the `STT02-012` alley simple-play relaxation advanced the stuck env at step 105, then exposed step 107:
  - First step-107 row was `STT04-005` (`Ruby`) play-to-alley, action type `2`; added STT04-005 top-5 Pyreskin reveal play, pick, noop, and bottom-deck split plumbing.
  - Re-probe then exposed another step-107 row: clean garden-entity attack into a 1-HP leader, action `[6, 1, 5, 0]`.
- `step_attack_leader_simple_fast` previously only accepted non-lethal leader damage and also did not set `winner`; it now admits clean lethal leader attacks and sets `winner=acting` when damage drops leader HP to 0 or below.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT04-005 and lethal-leader attack changes.

## 2026-06-19 — Step 116 AZK01-122 placement

- Probe after the lethal-leader attack change advanced to `GENERIC step=116 action_type=21`: `AZK01-122` selection placement to alley, action `[21, 0, 1, 0]`.
- The legal mask enumerates `ab_sel_cards` even when selected cards are still in hand. The AZK01-122 split mask and helper required `Zone.SELECTION`, so they rejected a legal hand-backed selection row.
- Updated the AZK01-122 placement fast path to accept selected cards in either `SELECTION` or `HAND`; `_detach_from_location` already handles the chosen hand card, and `return_remaining_to_hand` clears only remaining selection entries.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-122 fix.

## 2026-06-19 — Step 119 full-alley STT02-003 play

- Probe after the AZK01-122/STT04-005 trigger handling advanced to `GENERIC step=119 action_type=2`: active player 1 plays `STT02-003` from hand into a full alley slot.
- Existing `step_play_stt02_003_reveal_fast` assumed an empty/non-full destination and wrote the new zpos directly. Reworked it to use `_enter_board_slot`, matching the simple-play replacement behavior.
- Relaxed the STT02-003 reveal mask from `slot_empty & not_full` to the existing replacement-safe predicate: occupied slots are allowed only when the zone is full and the displaced card has no attachments, godmode, or when-destroyed timing.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT02-003 replacement change.

## 2026-06-19 — Step 122 STT03-004 activation

- Probe after the STT02-003 replacement change advanced to `GENERIC step=122 action_type=11`: active player 0 activates `STT03-004` from garden slot 2.
- Added `step_activate_stt03_004_fast` and split-vector wrapper/mask/dispatch for Sloth Scarecrow's no-target main ability: sacrifice itself and heal its leader by 1.
- First mask probe still rejected the row because the card had `cooldown=1`; C legal actions still allow this activated ability while the source is untapped. Removed the cooldown gate from both the helper and the host mask.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the cooldown relaxation.

## 2026-06-19 — Step 125 passive-queue NOOP

- Probe after the STT03-004 activation fix advanced to `GENERIC step=125 action_type=0`: active player 0 chose main-phase `NOOP` while `passive_queue_count=4`.
- Existing main NOOP split masks intentionally required an empty passive queue. Generic static action still accepts the pass action, recomputes queued passive buffs, then ends the turn.
- Added a dedicated `_main_noop_passive_fast_mask` that routes only main-phase `NOOP` rows with pending passive/STT02-012 work, no ability/combat/trigger work, and no end/start timing through the existing full `step_main_noop_fast` helper instead of the simple no-cleanup helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the passive NOOP mask.

## 2026-06-19 — Step 126 STT02-014 spell

- Probe after the passive-queue NOOP mask advanced to `GENERIC step=126 action_type=8`: active player 1 plays `STT02-014` (`Chilling Water`) from hand.
- Added STT02-014 split paths:
  - play helper pays/discards the spell and opens mandatory effect selection when the opponent has a garden entity with IKZ cost <= 2;
  - effect helper freezes the selected enemy garden entity for 2 turns and clears ability context.
- Added vector import/wrapper/JIT/masks/dispatch/trace entries for `play_spell_stt02_014` and `effect_stt02_014`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after STT02-014 plumbing.

## 2026-06-19 — Step 131 AZK01-007 effect with pending passives

- Probe after STT02-014 advanced to `GENERIC step=131 action_type=14`: `AZK01-007` effect selection while `passive_queue_count=4`.
- Existing `step_effect_azk01_007_fast` already applies the selected +1 ATK EOT buff, clears context, and calls `recompute_passives`.
- Relaxed `_effect_azk01_007_fast_mask` to allow pending passive/STT02-012 queue work for this path; the helper drains it after applying the effect, matching generic static action ordering.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask relaxation.

## 2026-06-19 — Step 132 STT03-011 garden play

- Probe after the AZK01-007 effect relaxation advanced to `GENERIC step=132 action_type=1`: active player 0 plays `STT03-011` (`Koyama Farm Plowman`) from hand to garden.
- Added STT03-011 split paths:
  - garden play helper pays/places and opens optional effect selection only when the opponent has a garden entity with base HP <= 2;
  - effect helper supports both optional skip and clean target destroy, then clears context and recomputes passives.
- Host masks require replacement-safe placement and reject destroy targets with attachments, godmode, or when-destroyed timing so queued trigger handling stays on generic paths until modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after STT03-011 plumbing.

## 2026-06-19 — Step 136 STT02-015 response spell

- Probe after STT03-011 advanced to `GENERIC step=136 action_type=8`: active player 1 casts `STT02-015` (`Commune with Water`) from hand during response window.
- Added STT02-015 split paths:
  - response spell play helper pays/discards and opens mandatory effect selection when any garden contains an entity with IKZ cost <= 3;
  - effect helper encodes `ANY_GARDEN_ENTITY` targets as `0..4` friendly garden, `5..9` enemy garden, returns one clean target to hand, clears context, and recomputes passives.
- Host mask blocks targets with attachments, godmode, `when returned` timing, or an `STT02-010` garden observer so return-trigger work stays on generic paths until explicitly modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after STT02-015 plumbing.

## 2026-06-19 — Step 137 AZK01-122 optional skip

- Probe after STT02-015 advanced to `GENERIC step=137 action_type=0`: active player 0 skipped `AZK01-122` optional `SELECTION_PICK` with two Rushfire targets in selection.
- Existing generic `step_selection_pick_noop_fast` already uses `process_skip_selection`; the vector host mask omitted `AZK01-122` from its supported optional-skip sources.
- Added `AZK01-122` to `_selection_pick_noop_fast_mask` and syntax-checked `jax_env/azuki_jax/step.py` plus `python/src/azk_puffer/jax_vector.py`.
- Re-probe advanced the same step to a new row: active player 0 plays `AZK01-007` to garden while `passive_queue_count=4`; next fix is to admit this on-play effect path with pending passive queue work.

## 2026-06-19 — Steps 137/140/144 attack follow-ups

- Step 137 follow-up row: `AZK01-007` garden play while `passive_queue_count=4`. Relaxed `_play_azk01_007_effect_fast_mask` to require only no pending `STT02-012` event bits; its helper already leaves passive draining deferred while effect selection is active, and the existing AZK01-007 effect fast path drains after clearing the ability context.
- Step 137 then exposed `AZK01-047` attacking a leader. Extended the leader-attack fast helper/mask to handle Shiko's `When Attacking` heal-1 trigger and mark `once_per_turn_used`; the helper now recomputes queued passive buffs before damage so pending passive queues match the static generic ordering.
- Step 140 advanced; step 144 exposed `AZK01-047` attacking a garden entity. Extended the entity-combat fast helper/mask to apply the same Shiko heal/once trigger before combat damage/discards.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after these changes.

## 2026-06-19 — Step 149 AZK01-103 activation

- Probe after the AZK01-047 entity-combat path advanced to `GENERIC step=149 action_type=11`: active player 1 activates `AZK01-103` (`Dropline Station`) from garden slot 1.
- Added AZK01-103 split paths:
  - activation enters `COST_SELECTION` for one other untapped friendly Earth garden entity;
  - cost selection taps the source, stores sacrificed HP capped at 5 plus the draw-if-HP>=3 flag, sacrifices the selected clean Earth entity, then enters leader effect selection;
  - effect selection targets either leader (`ANY_LEADER` encoding 0=friendly, 1=enemy), deals stored effect damage, performs the conditional draw/deckout, clears context, and drains deferred passive work.
- Host masks block dirty sacrifice targets with attachments/godmode and route only clean leader effect damage.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after AZK01-103 plumbing.

## 2026-06-19 — Step 155 AZK01-019 passive play

- Probe after AZK01-103 advanced to `GENERIC step=155 action_type=1`: active player 1 plays `AZK01-019` (`Jay`) from hand to garden.
- `AZK01-019` is a self passive (+2 HP when the owner garden contains only Normal entities); the existing simple entity helper did not drain passive observer work, so the host mask rejected the played passive source.
- Updated `step_play_entity_simple_fast` to call `recompute_passives` after placement/triggers, matching the generic post-action passive drain, and allowed `AZK01-019` as a simple played passive source in `_play_entity_simple_fast_mask`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the passive-play change.

## 2026-06-19 — Step 156 AZK01-007 over passive observer

- Probe after AZK01-019 advanced to `GENERIC step=156 action_type=2`: active player 1 plays `AZK01-007` to alley while `AZK01-019` is watching garden/alley passive events.
- The AZK01-007 play helper opens effect selection and leaves passive recompute deferred; the AZK01-007 effect helper already clears context and drains passives. Removed the `no_passive_watch` gate from `_play_azk01_007_effect_fast_mask`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask relaxation.

## 2026-06-19 — Step 157 AZK01-007 effect over passive observer

- Probe after the AZK01-007 play relaxation advanced to `GENERIC step=157 action_type=14`: the same `AZK01-007` effect selection while `AZK01-019` had queued passive work.
- Removed the remaining non-inert passive-watch gate from `_effect_azk01_007_fast_mask`; `step_effect_azk01_007_fast` clears the ability and recomputes passives.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the effect-mask relaxation.

## 2026-06-19 — Step 163 forced response pass

- Probe after the AZK01-007 effect relaxation advanced to `GENERIC step=163 action_type=0`: response-window `NOOP` with no legal alternatives and clean garden-into-leader combat pending.
- Expanded `_response_noop_combat_fizzle_fast_mask` to also admit clean nonlethal leader combat rows; it already dispatches through `transition_to_combat_resolve` + `combat_resolve`, so combat modifiers/carapace are handled by the engine phase code rather than by the older manual leader helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response mask expansion.

## 2026-06-19 — Step 163 zero-damage entity response

- Follow-up probe showed the step-163 leader response row was handled, but the same action type still fell back on a second response-window `NOOP`: `STT03-009` garden attacker with 0 ATK into `AZK01-001` garden defender with 0 ATK.
- `_response_noop_entity_combat_fast_mask` only admitted combat where the attacker dealt positive damage. Added a zero-damage stalemate case for live, clean entity-vs-entity combat; the existing helper already records no damage for amount 0 and clears combat back to main.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the entity-response mask change.

## 2026-06-19 — Step 163 AZK01-047 attack into response

- Probe after the zero-damage response fix still stopped at step 163, now on action type `ATTACK`: `AZK01-047` garden slot 0 attacking the opposing leader while response actions were available.
- `_attack_leader_response_fast_mask` still rejected all `When Attacking` sources, while the no-response leader attack path had already modeled `AZK01-047`'s heal-on-attack. Added the same Shiko handling to `step_attack_leader_response_fast`: recompute passives, declare combat, heal the acting leader if the once flag is unused, set the once flag, then open the response window.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response-attack change.

## 2026-06-19 — Step 163 gate portal over AZK01-019 passive

- Probe after the AZK01-047 response attack fix still stopped at step 163, now on `GATE_PORTAL`: active player 1 portals `AZK01-007` from alley to garden while `AZK01-019` is already watching passive stats.
- `step_gate_portal_simple_fast` already recomputes passives after placement. Relaxed `_gate_portal_simple_fast_mask` to treat `AZK01-019` board watchers and alley-only `STT02-012` as safe for this recompute-backed path, instead of forcing generic fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the gate mask relaxation.

## 2026-06-19 — Step 163 STT04-002 gate target prompt

- Gate debugger showed the step-163 portal row uses `STT04-002` (`Ragefire Gate`) and was rejected because a damaged friendly garden entity existed, so the old simple gate mask's `stt04_002_no_targets` case did not apply.
- Added `STT04-002` gate support:
  - gate portal now opens optional effect selection when a friendly garden entity took damage this turn;
  - new effect helper handles both skip and target selection, applies the gate-power attack modifier only when gate power is positive, then clears context;
  - vector wrapper/JIT/mask/dispatch/trace entries added for `effect_stt04_002`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT04-002 gate path.

## 2026-06-19 — Step 164 STT04-002 skip with deferred passives

- Probe after adding `STT04-002` gate selection advanced to step 164: effect-selection `NOOP` skip while the gate portal had queued passive recompute work (`passive_queue_count=8`) under the still-open ability context.
- Relaxed `_effect_stt04_002_fast_mask` to allow pending passive/STT02-012 work and updated `step_effect_stt04_002_fast` to clear context then call `recompute_passives`, matching the existing deferred-passive pattern used by other effect helpers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the skip/passive relaxation.

## 2026-06-19 — Step 165 STT03-016 with unrelated attachment

- Probe after the STT04-002 skip fix advanced to step 165: `STT03-016` (`Quicksand`) spell play. The opponent had an attached weapon in play, but it was not attached to any garden entity marked for destruction.
- Relaxed `_play_spell_stt03_016_fast_mask` from rejecting any opposing attachment to rejecting only attachments hosted by marked destroy targets. This preserves the helper's conservative behavior for target-hosted attachments while allowing unrelated stale/other attachments.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT03-016 mask change.

## 2026-06-19 — Step 168 simple alley play over AZK01-019

- Probe after the STT03-016 attachment relaxation advanced to step 168: active player 1 plays `AZK01-111` to alley while `AZK01-019` is already on board.
- `step_play_entity_simple_fast` already recomputes passives after placement; relaxed `_play_entity_simple_fast_mask` so existing `AZK01-019` passive watchers do not force generic fallback for otherwise simple plays.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the simple-play watcher change.

## 2026-06-19 — Step 170 AZK01-111 alley activation

- Probe after the AZK01-019 simple-play relaxation advanced to step 170: `AZK01-111` alley activation (`ACTIVATE_ALLEY_ABILITY`).
- Added a conservative `AZK01-111` split path for the observed no-hand-follow-up shape:
  - activation sacrifices the alley source and opens optional enemy-garden effect selection;
  - effect skip/selection clears context and recomputes passives;
  - masks require no eligible cost-<=2 hand entity, so the unimplemented selection follow-up still falls back instead of being skipped incorrectly.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after AZK01-111 activation/effect wiring.

## 2026-06-19 — Step 172 STT04-004 over AZK01-019 passive

- Probe after the AZK01-111 path advanced to step 172: `STT04-004` effect selection targeting the opposing garden while `AZK01-019` is on the source player's board.
- `step_effect_stt04_004_fast` sacrifices the source and can change passive state; updated it to clear context then recompute passives. Relaxed the mask for `AZK01-019` watchers and alley-inactive `STT02-012`, matching the recompute-backed simple-play/effect pattern.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT04-004 passive relaxation.

## 2026-06-19 — Step 172 STT04-004 garden source

- Follow-up probe still stopped on the same `STT04-004` effect row. The source was in garden after on-play, but `_effect_stt04_004_fast_mask` only accepted alley sources.
- Relaxed the source-zone check to allow garden or alley before the helper sacrifices the source.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the source-zone fix.

## 2026-06-19 — Step 174 AZK01-127 lethal response over AZK01-019

- Probe after the `STT04-004` source-zone fix advanced to step 174: `AZK01-127` response effect targeting a 1-HP enemy garden entity while `AZK01-019` was watching the target side.
- Updated `step_effect_azk01_127_fast` to clear context then recompute passives before closing the response/combat gate. Relaxed the mask for `AZK01-019` and alley-inactive `STT02-012` passive watchers on lethal targets.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-127 passive recompute change.

## 2026-06-19 — Step 183 STT03-006 combat death over passive

- Probe after the AZK01-127 recompute fix advanced to step 183: active player 0 attacked with `STT03-006` into a larger garden entity while `AZK01-019` was on board.
- Extended the main-phase entity-combat fast path to handle a single `STT03-006` combat death: discard, recompute passive auras, draw 1, then open the mandatory discard-from-hand effect if a hand card exists.
- Relaxed `_attack_entity_mutual_destroy_fast_mask` only for this `STT03-006` when-destroyed shape, with recompute-safe passive watchers (`AZK01-019`, inert `STT01-008`, inactive-alley `STT02-012`) and no double-`STT03-006` death.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT03-006 combat-death path.

## 2026-06-19 — Step 183 AZK01-105 activation with unrelated attachment

- Re-probe after the `STT03-006` combat-death path advanced to another step-183 row: active player 1 activates garden `AZK01-105` (`Prickly Tumbleweed`) while the opponent has an unrelated attached `AZK01-018`.
- The existing `AZK01-105` activation/effect helpers do not touch unrelated attachments. The host masks were over-conservative by rejecting any attachment anywhere.
- Relaxed `_activate_azk01_105_fast_mask` to reject only attachments on the sacrificing `AZK01-105` source, and relaxed `_effect_azk01_105_fast_mask` to keep only the existing target-attached rejection.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-105` attachment-mask relaxation.

## 2026-06-19 — Step 185 combat and AZK01-105 passive cleanup

- Re-probe after the `AZK01-105` attachment-mask relaxation advanced to step 185 with two attack rows.
- Env 0 was a clean 0-ATK `STT03-009` garden attack into `AZK01-012` while `AZK01-019` was watching the defender's board. The entity-combat helper now recomputes passives after combat, so the mask now treats recompute-safe watchers as valid for all simple-trigger entity combat, not just `STT03-006` deaths.
- Env 3 still had `passive_queue_count=4` after `AZK01-105` effect damage destroyed `AZK01-012`. Updated `step_effect_azk01_105_fast` to clear context and recompute passives before exposing the next action.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the step-185 fixes.

## 2026-06-19 — Step 189 AZK01-020 spell

- Re-probe after the step-185 fixes advanced to step 189: active player 1 plays `AZK01-020` (`Power of Friendship`) from hand in main phase with at least two friendly garden entities.
- Added `AZK01-020` split play/effect paths:
  - play pays/discards the spell and opens mandatory selection for two friendly garden entities;
  - effect records two distinct targets, then applies +1 ATK until EOT in main phase or +1 HP until EOT in response phase before clearing context.
- Added vector import/wrapper/JIT/masks/dispatch/trace/handled wiring for `play_spell_azk01_020` and `effect_azk01_020`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after `AZK01-020` plumbing.

## 2026-06-19 — Step 190 STT04-015 spell

- Re-probe after `AZK01-020` plumbing advanced to step 190: active player 0 plays `STT04-015` (`Detonation Pact`) from hand.
- Added a direct `STT04-015` split path for the clean nonlethal shape: pay/discard the spell, deal 1 effect damage to the owner's leader, then deal 2 effect damage to the enemy leader.
- Added vector import/wrapper/JIT/mask/dispatch/trace/handled wiring for `play_spell_stt04_015`. The mask requires clean leaders (no damage triggers, godmode, carapace, immunity) and nonlethal HP margins, keeping lethal/triggered variants on generic paths until modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after `STT04-015` plumbing.

## 2026-06-19 — Step 202 STT03-011 alley no-effect play

- Re-probe after `STT04-015` plumbing advanced to step 202: active player 0 plays `STT03-011` (`Koyama Farm Plowman`) from hand to alley.
- C `stt03_011_validate` only returns true when the source is in the garden; alley play has no optional destroy prompt. The split simple-play mask was still rejecting implemented `STT03-011` on-play timing everywhere.
- Allowed `STT03-011` alley plays through `_play_entity_simple_fast_mask` as an implemented timing with no effect; garden plays with valid low-base-HP opposing targets still use the dedicated `STT03-011` effect path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the alley no-effect mask change.

## 2026-06-19 — Step 204 AZK01-116 self-damage play

- Re-probe after the `STT03-011` alley fix advanced to step 204: active player 1 plays `AZK01-116` (`Tenmoku Daiki`) from hand to garden.
- `AZK01-116` is a mandatory on-play effect that deals 3 effect damage to its owner's leader. The generic simple-play path rejected it because the on-play ability is implemented but was not in the simple implemented trigger set.
- Added `AZK01-116` to `_apply_simple_implemented_play_trigger`: after placement, it deals 3 effect damage to the owning leader with the played entity as source. The host mask only admits clean nonlethal owner-leader damage rows (no takes-damage timing, godmode, carapace, effect immunity, or grant-godmode).
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-116` simple-trigger path.

## 2026-06-19 — Step 204 AZK01-105 activation over enemy AZK01-019

- Re-probe after the `AZK01-116` path still stopped at step 204, now on active player 0 activating garden `AZK01-105` while the opponent had `AZK01-019` in garden.
- `AZK01-105` activation sacrifices the source and opens a mandatory damage target prompt. The opponent's `AZK01-019` passive is not affected by the source-side sacrifice; the old activation mask rejected it anyway as a generic passive watcher.
- Relaxed `_activate_azk01_105_fast_mask` to allow enemy-side `AZK01-019` and inactive-alley `STT02-012` watchers for the activation prompt. Also relaxed `_effect_azk01_105_fast_mask` for recompute-safe `AZK01-019`/inactive `STT02-012` watchers because the effect helper clears context and recomputes passives after damage.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-105` passive-watch mask changes.

## 2026-06-19 — Step 206 STT03-013 alley no-effect play

- Re-probe after the `AZK01-105` passive-watch changes advanced to step 206: active player 1 plays `STT03-013` (`Stone Masked Ancient`) from hand to alley.
- C `stt03_013_validate` only succeeds while the source is in the garden; alley play has no optional self-tap prompt. The simple-play mask still treated `STT03-013` as an active watcher/played passive source in alley.
- Marked `STT03-013` alley play and already-alley `STT03-013` as inactive for `_play_entity_simple_fast_mask`, while keeping garden entries on the generic/dedicated path until the optional self-tap prompt is modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `STT03-013` alley mask change.

## 2026-06-19 — Step 206 STT03-013 id wiring fix

- Immediate re-probe after the `STT03-013` alley mask change failed during mask evaluation: `_play_entity_simple_fast_mask` referenced `_stt03_013_id` before `JaxVecEnv.__init__` initialized it.
- Added `_stt03_013_id = CODE_TO_ID["STT03-013"]` next to the other STT03 ids.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the id wiring fix.

## 2026-06-19 — Step 206 AZK01-059 combat damage trigger

- Re-probe after the `STT03-013` id fix advanced to another step-206 row: active player 1 attacks with garden `AZK01-059` into an opposing garden entity. `AZK01-059` survives combat damage and should open its once/turn takes-damage buff target prompt.
- Extended `step_attack_entity_mutual_destroy_fast` to pop a clean queued `AZK01-059` takes-damage trigger after combat/passive recompute and enter effect selection. Extended `_attack_entity_mutual_destroy_fast_mask` for the nonlethal attacker-`AZK01-059` shape when it has another friendly garden entity to target and no other combat trigger/response hazards.
- Also made `step_effect_azk01_059_fast` mark the source's once/turn bit when the buff target is selected, so repeated trigger paths cannot reuse it in the same turn.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-059` combat trigger path.

## 2026-06-19 — Step 208 AZK01-103 godmode cost target

- Re-probe after the `AZK01-059` combat trigger path advanced to step 208: active player 1 activates `AZK01-103` (`Dropline Station`) in garden.
- Debugging showed `_activate_azk01_103_fast_mask` saw the source and board correctly but found no Earth cost target because the only valid Earth entity had godmode/grant-godmode. C `azk01_103_validate_cost_target` checks only Earth garden entity, not source, and untapped; `sacrifice_card` itself handles godmode prevention later.
- Removed the activation and cost-selection mask's godmode/grant-godmode target rejection for `AZK01-103`, matching both the C validator and the existing JAX helper behavior.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-103` cost-target mask change.

## 2026-06-19 — Step 209 AZK01-122 into STT04-004

- Re-probe after the `AZK01-103` mask change advanced to step 209: `AZK01-122` (`Rushfire Gate`) selection places `STT04-004` from selection into garden.
- The `AZK01-122` placement fast path only special-cased `STT04-005`; it rejected `STT04-004` because the target has on-play timing. Added `STT04-004` as a supported placement target: the helper suppresses the generic queued on-play trigger and opens the same optional confirmation context used by normal simple play.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-122`/`STT04-004` placement path.

## 2026-06-19 — Step 210 STT02-009 replacement play

- Re-probe after the `AZK01-122`/`STT04-004` path advanced to step 210: active player 1 plays `STT02-009` (`Aya`) to a full garden slot, replacing an existing garden entity.
- `_play_stt02_009_confirm_fast_mask` still required an empty slot and non-full placement zone, while the helper uses `_enter_board_slot` and already supports full-zone replacement.
- Replaced the empty-slot gate with the standard slot/full/displaced-simple check, allowing replacement when the displaced card has no attached cards.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `STT02-009` replacement-play mask change.

## 2026-06-19 — Step 210 STT03-001 over enemy AZK01-019

- Re-probe after the `STT02-009` replacement mask still stopped at step 210, now on active player 0 activating the `STT03-001` leader (`Bobu`) while the opponent had `AZK01-019` in garden.
- Bobu activation pays IKZ and arms the leader latch; it does not change the opponent board or passive stats. The activation mask rejected the enemy `AZK01-019` watcher globally.
- Relaxed `_activate_stt03_001_fast_mask` to allow enemy-side `AZK01-019` plus inactive-alley `STT02-012`/`STT03-013` watchers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the Bobu passive-watch mask change.

## 2026-06-19 — Step 211 STT03-013 gate-portal enter trigger

- Re-probe after the Bobu mask change advanced to step 211: active player 1 gate-portals an alley `STT03-013` (`Stone Masked Ancient`) into garden through `STT03-002`.
- The gate fast path rejected the row because the portaled card has `AWhenEntersGarden`, and the gate/effect masks also treated alley `STT03-013` as a passive watcher. C queues `STT03-013`'s optional self-tap trigger, starts any gate ability first, then processes the queued enter trigger after the gate ability clears.
- Added a narrow queued-trigger bridge for `STT03-013`: gate portal now allows that enter trigger, `STT03-002`/`STT04-002` effect selection can resolve with the queued trigger pending, and clearing the gate effect begins the `STT03-013` confirmation. Added a dedicated fast confirmation path for confirming or declining `STT03-013`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the gate-portal queued-trigger change.

## 2026-06-19 — STT03-013 trigger bridge compile guard

- A re-probe after the first `STT03-013` trigger bridge timed out during early split-step compilation. The culprit was using the generic `resolve_triggered_effect`/`runtime.process_confirm` dispatch inside narrow fast paths, which pulls the full implemented-card switch into those kernels.
- Replaced that with a manual `STT03-013` confirmation begin and manual confirm/decline handling: pop only the queued `AWhenEntersGarden` trigger, set the optional confirmation context directly, tap on confirm, and clear context without card-dispatch switches.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after removing the generic dispatch from the `STT03-013` bridge.

## 2026-06-19 — Step 213 STT04-009 combat-damage fizzle

- Re-probe after the manual `STT03-013` bridge advanced to step 213: active player 1 attacks with `STT04-003` into enemy garden `STT04-009`.
- `STT04-009` has `AWhenTakesDamage`, but C validation requires the last damage to be from a card effect. Combat damage still queues the takes-damage trigger, then auto-processing pops it and validation fails with no ability prompt.
- Allowed the simple entity-combat mask to accept defender `STT04-009` as a combat-damage fizzle and added a narrow pop/drop in `step_attack_entity_mutual_destroy_fast` when the queued `STT04-009` trigger's `last_dmg_from_effect` flag is false.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `STT04-009` combat fizzle path.

## 2026-06-19 — Step 213 AZK01-066 Firestorm

- Re-probe after the `STT04-009` combat fizzle still stopped at step 213, now on active player 1 playing `AZK01-066` (`Firestorm`) from hand.
- Added a direct fast path for `AZK01-066`: pay IKZ, discard the spell, deal 2 effect damage in C order to player 0 leader/garden then player 1 leader/garden, record the spell as the damage source, and recompute passives after deaths.
- The mask is intentionally narrow: it accepts clean targets with no damage/death trigger, redirect, immunity, godmode, or carapace hazards, while allowing inert `AZK01-019` passive watchers that recompute can handle.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-066` fast path.

## 2026-06-19 — Step 214 STT03-013 inert attack watcher

- Re-probe after the `AZK01-066` fast path advanced to step 214: active player 1 attacks with garden `STT02-004` into enemy garden `AZK01-111` while `STT03-013` is already in garden.
- `STT03-013` only has `AWhenEntersGarden`; once its enter trigger has resolved, an in-play copy should not block clean combat fast paths as a generic passive watcher.
- Relaxed the entity-combat and attack-response passive-watch masks to treat in-play `STT03-013` as inert/recompute-safe, matching the already modeled manual enter-trigger bridge.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the attack watcher mask change.

## 2026-06-19 — Step 215 godmode attacker into leader

- Re-probe after the `STT03-013` attack watcher fix advanced to step 215: active player 1 attacks the opposing leader with garden `AZK01-054` (`Teb Fea`), a 7/7 inherent-godmode entity, for lethal damage.
- The leader-attack fast helper only damages the defending leader and taps the attacker; attacker godmode does not affect this outcome. The host mask was over-conservative by rejecting inherent/granted godmode on the attacker.
- Relaxed leader-attack simple/response masks to keep defender godmode protection gates while allowing attacker godmode.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the leader-attack godmode mask change.

## 2026-06-19 — Step 217 AZK01-128 response spell

- Re-probe after the godmode-attacker mask change advanced to step 217: active player 0 casts `AZK01-128` (`Wrong Step`) during a response window while the pending attacking entity is `STT02-008` with current HP 1.
- Added `AZK01-128` split paths:
  - play helper pays/discards the response spell and opens a mandatory effect target prompt;
  - effect helper only accepts selecting the current attacking garden entity with HP <= 2, destroys it, clears context, recomputes passives, then closes the response/combat gate when no further responses or queued triggers remain.
- Host masks reject attached, godmode, and when-destroyed target shapes so dirty destroy-trigger work stays on generic paths until modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-128` response spell wiring.

## 2026-06-19 — Step 221 AZK01-060 confirm into AZK01-062

- Re-probe after `AZK01-128` advanced to step 221: `AZK01-060` (`Scarlett`) optional attack confirmation while attacking enemy garden `AZK01-062`.
- `AZK01-062` queues `AWhenTakesDamage` on combat damage, but its validation requires a pending effect-damage redirect. Combat damage has no redirect, so C auto-pops the invalid trigger and continues.
- Updated `step_confirm_azk01_060_fast` to pop/drop a queued `AZK01-062` combat-damage trigger before discards. Relaxed the confirm mask for that fizzle and made the response-hand check cost-aware so unpayable response cards do not force generic fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-060`/`AZK01-062` confirm fix.

## 2026-06-19 — Step 221 AZK01-060 confirm with payable response

- Follow-up analysis of the same step-221 confirmation showed the defender still had a payable response after `AZK01-060`'s optional buff, so the direct-combat confirm mask correctly stayed off.
- Extended the existing `AZK01-060` response-window confirmation helper to support confirmed buffs: confirm applies Infiltrate, Sacrifice-at-end, and +1 ATK, then clears the context and runs `phase_gate` instead of resolving combat. Decline behavior is unchanged.
- Updated the response-confirm mask to use cost-aware response-hand checks and to route confirmed rows only when a non-defender response remains available.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response-confirm extension.

## 2026-06-19 — Step 227 AZK01-022 on-play bounce

- Re-probe after the `AZK01-060` response-confirm path advanced to step 227: active player 0 plays `AZK01-022` (`Mirage Frog`) from hand zpos 5 to garden slot 4.
- C semantics: optional on-play ability; on confirm, discard one other friendly hand card, then return any garden entity with IKZ cost <= 2 to its owner's hand.
- Added `AZK01-022` split paths:
  - play helper pays/places and opens optional confirmation only when a post-play hand discard and clean low-cost garden bounce target exist;
  - confirm helper enters cost selection;
  - cost helper discards the selected hand card and opens mandatory effect selection;
  - effect helper encodes `ANY_GARDEN_ENTITY` as friendly slots `0..4` and enemy slots `5..9`, returns one clean low-cost garden entity, clears context, and recomputes passives.
- Host masks reject attached, godmode, when-returned, `STT02-010` observer, and non-inert passive-watch shapes so return-trigger work stays on generic paths until modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-022` plumbing.

## 2026-06-19 — Step 241 gate replacement over godmode

- Probe after the `AZK01-022` plumbing advanced to step 241: action type `GATE_PORTAL`.
- Env 1 portaled alley `AZK01-047` into a full garden slot occupied by inherent-godmode `AZK01-054`; C/JAX replacement uses `discard_card_for_replacement` / `_enter_board_slot(... ignore_godmode=True)`, so godmode and when-destroyed triggers should not block the fast gate path. Attached-card displacement remains rejected.
- Relaxed `_gate_portal_simple_fast_mask` displacement safety to reject only missing/attached displaced occupants, not godmode or when-destroyed timing, for full-slot gate replacement.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the gate replacement mask change.

## 2026-06-19 — Step 249 STT02-013 selection-to-alley

- Probe after the gate replacement relaxation advanced to step 249: `STT02-013` selection prompt chose action `[21, 0, 1, 0]`, playing the selected top-deck `STT02-005` entity directly to an alley slot instead of taking it to hand.
- Extended `step_select_stt02_013_pick_fast` to handle both `SELECT_FROM_SELECTION` and `SELECT_TO_ALLEY`: alley placement moves the selected card from selection to the chosen alley slot, applies replacement only when full, increments play counters, queues modeled on-play hooks, and then bottom-decks the remaining reveal cards.
- Updated `_select_stt02_013_pick_fast_mask` to admit clean water-cost<=2 entity selection-to-alley rows while still rejecting on-play-trigger targets, attached replacements, passive-watch hazards, and queued work.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `STT02-013` alley-selection path.

## 2026-06-19 — Step 249 STT02-005 on-play in STT02-013 alley path

- Re-probe still stopped at the same `STT02-013` selection-to-alley row because the selected entity was `STT02-005`, which has a simple implemented on-play draw-if-third-entity trigger.
- Replaced the generic `queue_on_play` call in the STT02-013 alley-selection helper with the same simple implemented/unimplemented trigger shortcut used by `step_play_entity_simple_fast`, then recompute passives.
- Relaxed the STT02-013 alley-selection mask to allow selected targets with modeled simple on-play triggers such as `STT02-005`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `STT02-005` trigger correction.

## 2026-06-19 — Step 250 garden attacker with STT01-012 weapon

- Probe after the STT02-013 alley fix advanced to step 250: player 1 attacked player 0's leader with garden `STT01-004` carrying attached `STT01-012`.
- The existing STT01-012 fast path only admitted leader attackers, while `step_attack_leader_simple_fast` rejected the attached `AWhenAttacking` weapon trigger.
- Extended the leader-attack simple helper/mask to handle exactly one attached `STT01-012`: mill one card before combat damage when the attacker has deck cards, then continue through the existing no-response leader-damage path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `STT01-012` garden-attacker path.

## 2026-06-19 — Step 250 STT01-012 response-opening attack

- Re-probe still stopped at the same garden-attacker `STT01-012` row, which means the defender had a payable response shape; the direct leader-damage simple path correctly stayed off.
- Extended `step_attack_leader_response_fast` and `_attack_leader_response_fast_mask` with the same exactly-one-attached-`STT01-012` mill-before-response handling, so clean leader attacks that open a response window do not fall back to generic.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response-opening `STT01-012` path.

## 2026-06-19 — Step 251 lethal leader attack into response

- Probe after the STT01-012 response-opening path advanced to step 251: `AZK01-054` attacked a 1 HP `STT02-001` leader, but the defender leader has a response ability, so the attack should open response before lethal combat damage resolves.
- Relaxed `_attack_leader_response_fast_mask` from nonlethal-only to any clean leader-damage attack when a response/defender shape exists; the helper only declares combat and opens response.
- Fixed `step_response_noop_leader_combat_fast` and its mask to allow lethal leader combat after the response pass, set `winner`, and keep attacker godmode irrelevant to leader damage.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the lethal response leader-combat changes.

## 2026-06-19 — Step 251 STT04-016 generic friendly cost target

- Probe after the lethal response change advanced the same step to `STT04-016` cost selection: action `[13, 2, 0, 0]` targeting a clean friendly garden entity (`AZK01-056`/`AZK01-004` alternatives), not one of the previously whitelisted `STT04-003` or `AZK01-059` cases.
- C allows any friendly garden entity as the cost target. Relaxed `step_select_cost_stt04_016_fast` to accept any entity; the host mask now admits clean one-damage targets with no protection, takes/deals-damage triggers, lethal when-destroyed trigger, or passive-watch hazards.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the generic STT04-016 cost-target relaxation.

## 2026-06-19 — Step 252 STT04-016 leader effect target

- Probe after the generic STT04-016 cost relaxation advanced to step 252: `STT04-016` effect selection used action `[14, 5, 0, 0]`, which encodes the enemy leader as `GARDEN_SIZE`.
- Extended `step_effect_stt04_016_fast` and `_effect_stt04_016_fast_mask` to decode `ENEMY_LEADER_OR_GARDEN_ENTITY` as garden slots `0..4` plus leader slot `5`, and added an explicit `SELECT_EFFECT_TARGET` guard so optional `NOOP` skip cannot accidentally select a stale garden target.
- Also generalized the queued `AZK01-059` takes-damage mask to accept the trigger owner recorded by the damage event instead of assuming it is the spell owner.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT04-016 effect-target change.

## 2026-06-19 — Step 88 STT02-012 garden play

- Re-probe from seed 1 then exposed an earlier deterministic row at step 88: active player 0 played `STT02-012` (`Young Shao`) from hand to an empty garden slot.
- `step_play_entity_simple_fast` already calls `_enter_board_slot` and `recompute_passives`, which covers the STT02-012 garden latch/update. The host mask only allowed `STT02-012` as a passive-inactive alley play.
- Allowed `STT02-012` garden plays through the simple entity path when no other passive watcher hazards are present.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT02-012 garden-play mask change.

## 2026-06-19 — Step 95 STT02-016 response with no discard target

- Probe after the STT02-012 garden relaxation advanced to step 95: active player 0 cast `STT02-016` from their only hand card during a response window.
- C `stt02_016_validate` checks for at least one hand card before the spell is discarded, while `stt02_016_validate_cost_target` later rejects the source card. This means the play can be legal even if the following cost-selection state has no legal discard target.
- Removed the split play helper/mask's pre-play `other_hand_card` requirement so the fast path mirrors that legal-but-no-cost-target C state instead of falling back to generic.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT02-016 play relaxation.

## 2026-06-19 — Step 99 AZK01-011 end-turn with weapon

- Probe after the STT02-016 play relaxation advanced to step 99: main-phase `NOOP` with a single active `AZK01-011` end-turn sacrifice source carrying attached `STT01-014`.
- `_apply_single_azk01_011_eot` already discards equipped weapons before destroying the `AZK01-011` source; the host mask still rejected all attachments.
- Relaxed `_main_noop_azk01_011_fast_mask` to allow attached weapon cards while still rejecting non-weapon attachments and other cleanup hazards.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-011 attachment relaxation.

## 2026-06-19 — Step 100 AZK01-060 lethal leader confirmation

- Probe after the AZK01-011 attachment relaxation advanced to step 100: `AZK01-060` optional attack confirmation while attacking a 2 HP leader.
- The direct confirmation path only accepted nonlethal leader damage and did not set `winner`; that was safe only for the earlier nonlethal mask.
- Updated `step_confirm_azk01_060_fast` to set the attacker as winner when confirmed/declined combat kills a leader, and relaxed `_confirm_azk01_060_fast_mask` to allow clean leader combat for both confirm and decline when no remaining response/defender path must open.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-060 lethal-leader confirmation fix.

## 2026-06-19 — Step 105 AZK01-086 discard-weapon selection

- Probe after the AZK01-060 lethal confirmation fix advanced to step 105: active player 1 played `AZK01-086` (`Forging Tricks`) with weapon cards in discard.
- Added an `AZK01-086` split play path: pay/discard the spell, move all owner discard weapons into `SELECTION` in discard order, set pick max to `min(count, 5)`, and keep the optional selection context.
- Added split selection-pick support for `AZK01-086` via the shared selection runtime, and relaxed the optional selection `NOOP` mask for multi-pick `AZK01-086` skips after zero or more picks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-086 play/selection wiring.

## 2026-06-19 — Step 129 STT03-006 discard effect with passive queue

- Probe after `AZK01-086` advanced to step 129: `STT03-006` death-trigger effect selection while `passive_queue_count=4`.
- The effect only discards a hand card, clears the ability context, and should then drain the passive queue before exposing the next action.
- Updated `step_effect_stt03_006_fast` to call `recompute_passives` after `_clear_context`, and relaxed `_effect_stt03_006_fast_mask` to allow pending passive/STT02-012 queue work for this recompute-backed path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT03-006 passive-drain fix.

## 2026-06-19 — Step 139 simple alley play with passive queue

- Probe after the STT03-006 passive-drain change advanced to step 139: active player 0 played `AZK01-047` from hand to alley while passive queue work was pending.
- `step_play_entity_simple_fast` already recomputes passives after placement and simple trigger handling. The host mask still required an empty passive queue before entry.
- Removed the passive-clean gate from `_play_entity_simple_fast_mask`; existing no-passive-watch and trigger gates still keep unmodeled observer shapes out.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the simple-play passive-queue relaxation.

## 2026-06-19 — Step 154 AZK01-015 fire-leader on-play

- Probe after the simple-play passive relaxation advanced to step 154: active player 1 played `AZK01-015` to alley under a Fire leader.
- Reused the simple entity play helper by adding `AZK01-015` Fire-leader handling to `_apply_simple_implemented_play_trigger`: after placement it opens mandatory leader effect selection.
- Added `step_effect_azk01_015_fast` and vector mask/dispatch for the clean `ANY_LEADER` effect, dealing 2 effect damage to the selected leader, clearing context, and recomputing passives.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-015 play/effect wiring.

## 2026-06-19 — Step 159 attack rows and STT04-016 skip trigger drain

- Probe after the AZK01-015 wiring advanced to step 159, with two `ATTACK` rows:
  - `AZK01-006` attacking a tapped `AZK01-070`; the no-response entity-combat mask incorrectly treated tapped `AZK01-070` as an available board response.
  - `AZK01-048` attacking a leader; the leader-attack masks rejected harmless attacker-side Carapace even though leader damage does not hit the attacker.
- Refined board-response checks for `AZK01-070` in attack masks to require an untapped, unfrozen garden source, and allowed attacker-side Carapace on clean attacks into leaders.
- Re-probe then exposed an earlier step-67 `NOOP` after skipping optional `STT04-016` effect while a cost-damage `AZK01-059` takes-damage trigger remained queued.
- Updated `step_effect_stt04_016_fast` so both effect selection and optional skip clear the `STT04-016` context and drain a queued `AZK01-059` trigger into its effect-selection context.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT04-016 skip-trigger drain fix.

## 2026-06-19 — Step 65 through 94 response/effect cleanup

- Step 65 exposed `AZK01-105` selecting an effect-immune target. The C validator allows the target; `deal_effect_damage` handles immunity as a no-op. Relaxed the split effect mask so effect immunity does not make the target illegal.
- Step 68/69 exposed attacks into alley targets and response passes after alley combat. Generalized the clean attack and response-opening helpers/masks to decode garden, leader, and alley targets, then allowed response-window entity combat to resolve clean alley defenders.
- Step 81 exposed `AZK01-005` alley play. Added `AZK01-005` as a simple implemented on-play source and added its optional enemy-garden 1-damage effect fast path.
- Step 84 exposed `AZK01-127` response damage followed by combat that queued an invalid `AZK01-062` combat-damage trigger. The response effect helper now pops that combat fizzle after auto-resolving combat.
- Step 94 exposed `AZK01-065` play while its owner leader had 2 HP. Relaxed the `AZK01-065` play/effect masks to admit self-lethal clean leader-cost damage; the effect helper already applies cost damage before target damage.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after these changes.

## 2026-06-19 — Steps 120 through 161 gate/turn cleanup

- Step 120 exposed queued `STT03-006` death-trigger `NOOP` bridging. Added a narrow main-phase bridge that pops one queued `STT03-006` destroyed trigger, draws 1, and opens the mandatory discard effect when hand cards remain.
- Step 122/145 exposed clean response passes with attacker-side godmode, attacker attachments, and zero-attack defender-only damage. Relaxed the response entity-combat mask for those shapes while keeping defender-side attachments and triggered damage/death hazards on generic paths.
- Step 149 exposed a clean `STT04-003` start-each turn pass with a spent/expiring IKZ token. The helper already uses full `end_turn`, so token cleanup is now allowed in the `STT04-003` NOOP mask instead of forcing generic fallback.
- Step 161 exposed `AZK01-122` placing `AZK01-056` from selection into alley. Added `AZK01-056` as a supported Rushfire Gate placement target: the helper suppresses the generic queued on-play trigger and opens Hokuto's top-5 Scorchweaver reveal context directly.
- Also completed earlier local fixes that the previous notes had not recorded: `AZK01-028` can be played to alley, `AZK01-125` leader activation applies the discard-turn cost reduction, and `STT03-006` effect resolution drains deferred passive work.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed before the step-161 probe.

## 2026-06-19 — Steps 184 through 193 passive-safe garden/gate cleanup

- Step 184 exposed `STT03-013` hand play to garden. Added the garden-enter optional confirmation to the simple entity play trigger shortcut and allowed the played `STT03-013` through the simple play mask; the existing `STT03-013` confirm fast path handles confirm/decline and recomputes passives.
- Step 186 exposed a gate portal while a resolved `STT03-013` was already on board. Treated in-play `STT03-013` as an inert watcher for recompute-backed simple play and gate portal paths; queued/unresolved enter triggers still stay out via `ab_phase`/`trig_count` gates.
- Step 192 exposed an `AZK01-018` weapon attach while `AZK01-019` was in alley. The attach helper already recomputes passives, so the attach mask now allows `AZK01-019` and inactive-alley `STT02-012` board watchers.
- Step 193 exposed gate-portal of `AZK01-019` from alley to garden. Allowed `AZK01-019` as a portaled passive source in the gate mask because the helper recomputes passives after placement.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after these changes.

## 2026-06-19 — Steps 196 through 206 AZK01-111/STT03-013 follow-ups

- Step 196 exposed `STT03-016` while the enemy had an `AZK01-019` passive source. Updated the spell helper to recompute passives after destroying low-HP enemy garden entities and allowed non-destroyed `AZK01-019` passive registrations through the mask.
- Step 199 exposed `AZK01-111` alley activation when small hand entities existed for its post-damage selection follow-up. Activation/effect masks no longer reject that state; the effect helper now moves eligible hand entities to `SELECTION_PICK` after optional damage instead of clearing immediately.
- The same activation row had a resolved `STT03-013` on board; treated in-play `STT03-013` as inert for the `AZK01-111` activation watcher mask.
- Step 206 exposed `AZK01-045` alley play with resolved `STT03-013` on board. Allowed that inert watcher in the `AZK01-045` reveal mask.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after these changes.

## 2026-06-19 — Steps 209 through 213 combat/lethal-cost cleanup

- Step 209 exposed clean entity combat into an inherent-godmode defender. Updated the entity-combat helper and mask to model inherent/granted godmode as zero incoming combat damage instead of rejecting the row or damaging the protected card.
- Step 213 exposed `STT04-001` activation at 1 leader HP. C validation does not require the leader to survive its self-damage cost, so the activation helper/mask now admit clean self-lethal cost rows.
- The same step then exposed `AZK01-065` effect selection while resolved `STT03-013` was on board. Treated `STT03-013` as an inert watcher for that mask.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after these changes.

## 2026-06-19 — Steps 216 through 222 combat/attachment cleanup

- Step 216 exposed clean attack resolution into `AZK01-062` and into a resolved `STT03-013`. The leader/garden attack helper now pops invalid `AZK01-062` combat-damage triggers, recomputes passives after auto-combat, and the mask treats resolved `STT03-013` as inert.
- Step 216/218 exposed `STT01-015` attach and subsequent combat with attached attackers. The attach helper now applies Tenraku's discard-threshold when-equipped bonus inline, and the attach/attack masks allow clean non-`AZK01-044` attacker attachments.
- Step 218/219 exposed response-window entity combat followed by stale passive queues. The response entity-combat helper now recomputes passives after combat discards, and its mask treats resolved `STT03-013` as inert.
- Step 221 exposed `AZK01-062` attacking `AZK01-061`; both combat-damage triggers are invalid under the observed damage state. The no-response attack mask now admits invalid `AZK01-062`/`AZK01-061` takes-damage fizzles, and the helper pops those queued fizzles after combat.
- Step 222 exposed `STT04-001` leader activation while tapped after attacking. C validation does not require untapped for this main ability, so the host mask now matches the helper and allows tapped leaders.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after these changes.

## 2026-06-19 — Steps 228 through 248 gate/weapon/passive cleanup

- Step 228 exposed `AZK01-043` weapon attach to a leader/garden target while other passive work existed. The attach mask now treats alley-targeting weapons (`AZK01-043`/`AZK01-095`) as inert passive watchers and allows recompute-backed attach rows with pending passive queues.
- Step 240 exposed `AZK01-124` gate portal when the post-portal devotion-cost selection had no valid targets. The gate mask now admits the no-target fizzle path so the simple helper can clear it without opening ability context.
- Step 244 exposed a leader attack with an attached `STT01-016` on non-Raizan `AZK01-119`. Both leader attack masks now ignore invalid/fizzling `STT01-016` when-attacking triggers while still rejecting valid Zanbato trigger rows for the generic path.
- Step 248 exposed `AZK01-010` garden play as a self-passive source. The simple entity play mask now admits played `AZK01-010` and treats existing `AZK01-010` watchers like the existing recompute-backed `AZK01-019` case.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after these changes.

## 2026-06-19 — Step 249 AZK01-010 gate watcher

- Probe after the AZK01-010 play fix advanced to `GENERIC step=249 action_type=10`: `STT02-002` gate portal while an `AZK01-010` passive source was already in the garden.
- `step_gate_portal_simple_fast` already recomputes passives after moving the alley card to garden. The gate mask now treats existing and portaled `AZK01-010` passive watchers like the existing recompute-backed `AZK01-019` path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this gate mask change.

## 2026-06-19 — Step 251 AZK01-009 over passive watcher

- Probe after the gate watcher fix advanced to `GENERIC step=251 action_type=14`: `AZK01-009` effect selection granting Charge while `AZK01-010` was in garden.
- `AZK01-009` only grants a timed Charge tag and does not change zones, damage, or passive stat inputs. Removed the passive-watcher gate from `_effect_azk01_009_fast_mask` while keeping pending passive queues blocked.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this effect mask change.

## 2026-06-19 — Step 255 AZK01-042 spell setup

- Probe after the `AZK01-009` effect fix advanced to `GENERIC step=255 action_type=8`: `AZK01-042` (`Thunderclap`) play from hand with at least three enemy garden entities.
- Added split play/effect paths for `AZK01-042`: pay/discard into mandatory three-target enemy-garden selection, reject duplicate picks, then apply 3/2/1 effect damage in pick order on the third selection.
- The effect mask keeps dirty damage targets with takes-damage/destroyed triggers or lethal attached hosts on the generic path; the helper recomputes passives after clearing context so killing `AZK01-010`-style passive sources drains correctly.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this spell path.

## 2026-06-19 — Step 258 STT02-012 gate portal

- Probe after the `AZK01-042` path advanced to `GENERIC step=258 action_type=10`: gate portal moving `STT02-012` from alley to garden.
- `step_gate_portal_simple_fast` uses `_enter_board_slot` plus `recompute_passives`, so portaled `STT02-012` self/passive observer work is handled. The gate mask now admits `STT02-012` as a portaled passive source instead of forcing generic fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this gate mask change.

## 2026-06-19 — Step 259 AZK01-126 selection over passive queue

- Probe after the `STT02-012` portal fix advanced to `GENERIC step=259 action_type=18`: `AZK01-126` selection pick returning an echoed discard spell while the portal left passive work pending.
- `step_select_azk01_126_pick_fast` now recomputes passives after `process_selection_pick`; the mask no longer rejects pending passive/STT02 work for this finishing selection.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this selection/passive drain change.

## 2026-06-19 — Step 260 play over active STT02-012

- Probe after the `AZK01-126` selection fix advanced to `GENERIC step=260 action_type=1`: simple garden play while an active `STT02-012` passive source was already in garden.
- `step_play_entity_simple_fast` already runs `_enter_board_slot` and `recompute_passives`; the play mask now treats existing `STT02-012` watchers as safe for this recompute-backed path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this play mask change.

## 2026-06-19 — Step 267 AZK01-043 attach trigger marker

- Probe after the active-`STT02-012` play fix advanced to `GENERIC step=267 action_type=7`: attaching `AZK01-043` to a garden entity.
- `AZK01-043`/`AZK01-095` are alley-targeting passive weapons; their equipped observer is derived by `attr_attack_alley`, not an effect selection. The attach mask now treats their when-equipped timing marker like an inline/inert attach trigger.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this attach mask change.

## 2026-06-19 — Step 267 AZK01-039 host when-equipped

- Follow-up probe still stopped on the same attach row: the weapon marker was allowed, but the host `AZK01-039` has a `When Equipped` grant-Charge effect.
- `step_attach_weapon_simple_fast` now applies `AZK01-039`'s permanent Charge grant inline via `apply_charge_grant`; the attach mask treats that host timing as modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this host when-equipped change.

## 2026-06-19 — Step 270 response combat passive death

- Probe after the attach fixes advanced to `GENERIC step=270 action_type=0`: response-window `NOOP` resolving entity combat where `AZK01-037` kills an `AZK01-010` defender.
- `step_response_noop_entity_combat_fast` already recomputes passives after combat. Its mask now allows recompute-backed passive sources (`AZK01-010`, `AZK01-019`, `STT02-012`) in the passive-death watch set instead of forcing generic fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this response combat mask change.

## 2026-06-19 — Step 272 AZK01-123 leader ability

- Probe after the response-combat passive-death fix advanced to `GENERIC step=272 action_type=11`: active player 1 activates leader `AZK01-123` (`Goro Graveloth`) with action `[11, 5, 0, 0]`.
- Added split activation/effect paths for `AZK01-123`: activation pays 1 IKZ, opens one friendly garden-entity target selection, and the effect applies +1 health until end of turn then marks the once-per-turn flag.
- Added vector import/wrapper/JIT/masks/dispatch/trace entries for `activate_azk01_123` and `effect_azk01_123`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this leader ability path.

## 2026-06-19 — Step 273 attach over AZK01-010 watcher

- Follow-up probe showed `AZK01-123` activation passed the host mask but still fell through the final split fallback because the footer `remaining` mask omitted the new activate/effect masks. Added both masks to the footer set.
- Probe then advanced to `GENERIC step=273 action_type=7`: attaching `STT01-014` (`Tenshin`) to `STT04-011` while an `AZK01-010` passive source was already in garden.
- `step_attach_weapon_simple_fast` already recomputes passives after attach and handles `STT01-014` by opening its optional leader-damage effect selection. The attach mask now treats existing/target `AZK01-010` passive watchers as safe for this recompute-backed path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this attach mask change.

## 2026-06-19 — Step 275 STT02-012 attacking as passive source

- Probe after the attach watcher change advanced to `GENERIC step=275 action_type=6`: garden `STT02-012` attacking an opposing garden entity while it was an active passive source.
- `step_attack_entity_mutual_destroy_fast` already calls `recompute_passives` after combat damage/discards. The entity-combat mask now treats active `STT02-012` passive watchers as recompute-safe, not only inactive alley `STT02-012`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this attack mask change.

## 2026-06-19 — Step 277 STT01-009 alley play

- Probe after the `STT02-012` attack mask change advanced to `GENERIC step=277 action_type=2`: playing `STT01-009` from hand to alley while existing `AZK01-010` passive work was recompute-safe.
- `STT01-009` is a passive source only while in garden; simple play already recomputes passives after placement. The simple play mask now admits played `STT01-009` and treats existing `STT01-009` board watchers as recompute-safe for this path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this play mask change.

## 2026-06-19 — STT01-009 split id crash

- Follow-up probe after the Step 277 mask change crashed before parity comparison because `JaxVecEnv` referenced `_stt01_009_id` without initializing it.
- Added the missing `STT01-009` id next to the other `STT01-*` cached IDs.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after adding the id.

## 2026-06-19 — Step 282 AZK01-017 lethal target

- Probe after the `STT01-009` id fix advanced to `GENERIC step=282 action_type=14`: `AZK01-017` effect selection targeting opponent garden slot 3 (`STT02-006`) at 1 HP.
- The AZK01-017 effect mask was needlessly nonlethal-only and also rejected inherent effect-immune targets even though the C legal mask permits selecting them and damage resolves to a no-op. The helper now recomputes passives after finishing/skipping the effect, and the mask admits clean 1-HP or effect-immune leader/entity targets while still rejecting real damage/destroy triggers, attachments on damaged entity targets, godmode, and carapace.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this effect damage change.

## 2026-06-19 — Step 283 STT02-001 response over inactive STT01-009

- Probe after the AZK01-017 mask change advanced to `GENERIC step=283 action_type=11`: active player 0 activates leader `STT02-001` during a response window while `STT01-009` is in the opponent alley.
- `STT01-009` only has a passive while in garden, and STT02-001 activation only opens its effect-selection context. The STT02-001 activation mask now treats alley `STT01-009` as an inactive passive watcher instead of forcing generic fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this activation mask change.

## 2026-06-19 — Step 286 AZK01-024 play

- Probe after the STT02-001 activation mask change advanced to `GENERIC step=286 action_type=1`: active player 0 plays `AZK01-024` (`Fumiko`) from hand to garden slot 2.
- Added split paths for AZK01-024 play/confirmation/cost/selection placement. The play path opens optional confirmation when a friendly garden entity exists after placement; cost selection returns one clean friendly garden entity to hand, excludes that returned card from the follow-up hand-to-selection scan, and clears context if no cost<=2 entity remains.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-024 plumbing.
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python - <<'PY' ... import JaxVecEnv ... PY` passed after the vector wiring.

## 2026-06-19 — Step 286 STT01-009 gate portal follow-up

- Follow-up probe passed the AZK01-024 play row and then stopped on another step-286 row: gate portal moving `STT01-009` from alley to garden.
- `step_gate_portal_simple_fast` already uses `_enter_board_slot` and `recompute_passives`. The gate mask now treats alley `STT01-009` as inactive before the move and `STT01-009` as a recompute-safe portaled passive source.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this gate mask change.

## 2026-06-19 — Step 292 AZK01-125 response activation

- Probe after the STT01-009 gate fix advanced to `GENERIC step=292 action_type=11`: active player 0 activates leader `AZK01-125` during a response window after discarding a card this turn.
- AZK01-125 is registered with both Main and Response timing. The fast helper/mask now allow the same no-target cost-reduction activation in response windows while preserving the main-phase clean-combat gate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after this response activation change.

## 2026-06-19 — Step 297 AZK01-015 water on-play

- Probe after the AZK01-125 response activation fix advanced to `GENERIC step=297 action_type=2`: active player 0 plays `AZK01-015` to alley under water leader `AZK01-125`.
- The previous AZK01-015 split support only covered the Fire leader branch. The simple play helper now applies the Water/Earth/Lightning immediate branches directly after payment, while Fire still opens effect selection; the simple-play mask admits only branch-valid AZK01-015 rows.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-015 immediate-branch change.

## 2026-06-19 — Step 299 STT02-001 response activation

- Probe after the AZK01-015 immediate-branch change advanced to `GENERIC step=299 action_type=11`: active player 0 activates `STT02-001` in response while passive watchers remain on board.
- The activation itself only pays 1 IKZ, marks the once-per-turn source, and opens effect selection. The host mask now allows passive watchers when payment is proven to use token/IKZ-area sources rather than tapping a garden IKZ source, and the fast helper now sets the once-per-turn bit on activation.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT02-001 activation change.

## 2026-06-19 — Step 300 AZK01-058 attack declaration

- Probe after the STT02-001 activation change advanced to `GENERIC step=300 action_type=6`: active player 0 attacks with garden `AZK01-058` into garden `AZK01-047` while the defender has response options.
- `AZK01-058`'s After Attacking trigger is not a declaration-time trigger. The entity-attack response mask now permits that source for response-window declaration and still leaves combat/after-attacking resolution to the later combat path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-058 attack-declaration mask change.

## 2026-06-19 — Step 301 response NOOP entity combat

- Probe after the AZK01-058 declaration mask change advanced to `GENERIC step=301 action_type=0`: response-window NOOP rows for entity-vs-entity combat.
- Two conservative gates were too narrow: `STT01-009` passive self-buff is recompute-safe when combat deaths occur, and `AZK01-058` After Attacking is a no-op when the source dies in the same combat before its trigger validates. The response NOOP entity-combat mask now admits those clean shapes.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response NOOP entity-combat mask change.

## 2026-06-19 — Step 310 AZK01-119 activation

- Probe after the response NOOP entity-combat mask change advanced to `GENERIC step=310 action_type=11`: active player 1 activates leader `AZK01-119` with a friendly garden entity carrying attached weapons and multiple weapons in discard.
- `AZK01-119` activation was only available through the generic path. Added leader activation and effect-selection split helpers: activation pays 3 IKZ, sets the once-per-turn bit, and opens a one-target effect; effect selection validates a friendly equipped garden entity, counts owner discard weapons, and applies capped `+ATK` until end of turn.
- Fixed the shared `AZK01-119` target validator to require the target's owner, not the active player, so out-of-turn/replayed contexts validate the same target shape as C.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py jax_env/azuki_jax/abilities/cards_batch4.py` passed after the AZK01-119 split path and validator change.

## 2026-06-19 — Step 313 AZK01-010 mutual combat

- Probe after the AZK01-119 split path advanced to `GENERIC step=313 action_type=6`: active player 0 attacks with garden `AZK01-010` into opponent garden `STT01-010`; both combatants die, and no response action is available.
- The mutual-destroy mask already recomputes passives after combat but did not classify `AZK01-010` as recompute-safe, so the presence of JD's self-passive forced a generic fallback. Added `AZK01-010` to that mask's recompute-safe passive watcher set.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask change.

## 2026-06-19 — Step 316 STT04-016 cost into AZK01-062

- Probe after the AZK01-010 mutual-combat mask change advanced to `GENERIC step=316 action_type=13`: `STT04-016` cost selection targeting friendly garden `AZK01-062` at slot 1.
- `Collateral Burst` cost damage can legally target `AZK01-062`; C queues Pekiro's pending damage redirect before immunity/carapace/godmode checks, so the split cost mask must admit that redirect shape instead of treating all takes-damage targets as generic-only.
- Updated the STT04-016 cost mask to allow unused `AZK01-062` redirect targets and the STT04-016 effect mask to continue while that single pending redirect trigger is queued.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask change.

## 2026-06-19 — STT04-016 queued AZK01-062 trigger follow-up

- Follow-up probe after admitting `AZK01-062` as an STT04-016 cost target advanced past the cost row but left a pending Pekiro redirect/trigger after STT04-016 effect resolution; the next fallback was a normal main action while `trig_count=1` and `redirect_count=1`.
- Generic `engine_step` runs `auto_resolve` after clearing an ability and would pop the queued triggered effect. The STT04-016 fast effect path only began `AZK01-059`; it now also begins queued `AZK01-062` redirects through `resolve_triggered_effect`.
- Added a split path for resolving `AZK01-062` effect selection/decline: consume the matching redirect entry, re-deal from the original damage source, suppress same-target re-redirect, clear context, recompute passives, fizzle self requeues with no pending redirect, and begin a chained `AZK01-062` if redirecting to another Pekiro.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the trigger follow-up and split path wiring.

## 2026-06-19 — STT04-016 AZK01-062 trigger merge correction

- Follow-up probe still stopped at the same step-318 normal play with `trig_count=1`/`redirect_count=1`.
- Static review found the new `AZK01-062` begin state was computed but only merged under the old `AZK01-059` condition. The STT04-016 effect path now merges begun triggered contexts for either `AZK01-059` or `AZK01-062`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the merge-condition correction.

## 2026-06-19 — Step 319 STT01-011 alley play

- Probe after the AZK01-062 trigger merge advanced to `GENERIC step=319 action_type=2`: active player 1 plays `STT01-011` from hand to an alley slot after a `STT01-008` garden play.
- `STT01-011` is a recompute-mode passive aura in the JAX engine (`STT01-016` weapon attack aura) and `step_play_entity_simple_fast` already calls `recompute_passives` after placement. The simple-play mask now treats an existing or newly played `STT01-011` as recompute-safe instead of forcing generic.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the simple-play mask change.

## 2026-06-19 — STT01-011 id wiring

- Follow-up probe crashed before parity comparison because `_play_entity_simple_fast_mask` referenced `_stt01_011_id` after adding STT01-011 to the recompute-safe set, but the id field had not been initialized.
- Added `_stt01_011_id` beside the other STT01 ids.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the id wiring.

## 2026-06-19 — Step 320 STT01-011 gate portal

- Probe after wiring `_stt01_011_id` advanced to `GENERIC step=320 action_type=10`: active player 1 gate-portals the alley `STT01-011` into an open garden slot.
- Gate portal already calls `recompute_passives` after placement. The gate mask now treats existing or portaled `STT01-011` as the same recompute-safe passive aura used by simple hand plays.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the gate mask change.

## 2026-06-19 — Step 323 AZK01-004 leader attack response

- Probe after the STT01-011 gate-portal mask change advanced to `GENERIC step=323 action_type=6`: `AZK01-004` attacks the opposing leader while the defender can respond.
- The dedicated AZK01-004 leader-attack helper only covered the no-response direct-damage case. It now applies the +1 attack trigger first, then either opens the response window when `defender_can_respond` is true or resolves immediate leader damage when no response exists.
- The host mask no longer computes or requires `no_response`; the helper selects the response/direct branch from the post-declaration state.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response-branch change.

## 2026-06-19 — Step 326 STT01-011 passive during mutual combat

- Probe after the AZK01-004 response branch advanced to `GENERIC step=326 action_type=6`: active player 0 attacks a garden `AZK01-004` with garden `AZK01-006`; both are clean 1/1 combatants while `STT01-011` is present in the opponent garden.
- The mutual-combat helper already recomputes passives after combat. Its host mask now treats `STT01-011` as recompute-safe, matching the simple play and gate-portal treatment of that passive aura.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mutual-combat mask change.

## 2026-06-19 — Step 327 STT02-003 over STT01-011 aura

- Probe after the mutual-combat mask change advanced to `GENERIC step=327 action_type=2`: active player 0 plays `STT02-003` to alley while the opponent has `STT01-011` in garden.
- `STT01-011` is a recompute-safe aura when an unrelated reveal card enters an empty alley slot; the STT02-003 reveal-play mask was still treating any existing `STT01-011` as a generic-only passive watcher.
- Updated `_play_stt02_003_reveal_fast_mask` to ignore existing `STT01-011` watchers for this play shape.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask change.

## 2026-06-19 — Step 327 STT04-016 cost into STT04-009

- Re-probe passed the `STT02-003` reveal-play row and stopped on another step-327 row: `STT04-016` cost selection targeting friendly garden `STT04-009`.
- `STT04-009` legally triggers only after taking effect damage. The STT04-016 cost mask now admits the nonlethal `STT04-009` cost target, and the STT04-016 effect path begins queued `STT04-009` triggers after clearing the STT04-016 context.
- Added split support for confirming and resolving `STT04-009`: confirmation enters effect selection; effect selection deals capped reflected effect damage to a clean non-source leader/garden target and recomputes passives.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT04-009 trigger path.

## 2026-06-19 — Step 328 STT01-006 with STT01-012 attached

- Probe after the STT04-009 path advanced to `GENERIC step=328 action_type=6`: active player 1 attacks the opposing leader with garden `STT01-006` carrying attached `STT01-012`.
- The STT01-006 attack fast path already opens Haruhi's mandatory when-attacking effect, but its mask rejected any attached trigger. It now admits the single attached `STT01-012` shape.
- The STT01-006 attack helper queues the attached `STT01-012` trigger behind the STT01-006 effect, and the STT01-006 effect helper resolves that queued mill trigger after clearing Haruhi's effect before opening response/combat.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT01-006/STT01-012 queue handling.

## 2026-06-19 — Probe timeout after STT01-006/STT01-012 change

- Re-ran the deterministic split frontier probe after the STT01-006/STT01-012 queue fix.
- Result: no generic fallback was observed before the harness timeout; the run reached `ok step=310` and then hit the 1800s command timeout, likely during first-time compilation of later split branches.
- Decision: rerun the same probe with a longer timeout so the post-step-328 frontier can be observed instead of treating this as a parity result.

## 2026-06-19 — Step 328 STT04-016 effect into STT03-006

- Longer re-probe advanced past the step-328 STT01-006 attack row and stopped on `STT04-016` effect selection with a queued `STT04-009` trigger from the cost target.
- The random action selected enemy garden slot 0, `STT03-006`, which dies to the 2 effect damage and queues its when-destroyed draw/discard trigger behind the existing `STT04-009` trigger.
- Updated the STT04-016 effect mask to admit the clean `STT03-006` destruction target. The STT04-016 helper already begins the queued `STT04-009`; confirm/decline/effect masks for `STT04-009` now allow a queued `STT03-006` behind it, and the clear/effect helpers begin that queued `STT03-006` trigger after the `STT04-009` context clears.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the queued `STT03-006` handling.

## 2026-06-19 — Step 330 STT04-009 into second STT03-006

- Re-probe advanced to `GENERIC step=330 action_type=14`: `STT04-009` effect selection with one `STT03-006` death trigger already queued, targeting the remaining enemy `STT03-006`.
- This is a legal chain: `STT04-009` kills a second `STT03-006`, then the existing queued `STT03-006` trigger begins first and the newly queued one remains behind it.
- Updated the STT04-009 effect mask to admit the clean `STT03-006` destruction target, and updated the STT03-006 effect helper/mask to allow and begin a queued follow-up `STT03-006` trigger after the current STT03-006 effect clears.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the chained `STT03-006` handling.

## 2026-06-19 — Handoff refresh after step 330

- Updated `jax_env/HANDOFF.md` top section to supersede the stale step-93 frontier.
- Current recorded frontier remains the post-fix long probe after step 330: `STT04-009` destroying a second `STT03-006` while one `STT03-006` trigger is already queued.
- No new parity result yet; the deterministic frontier probe is still running.

## 2026-06-19 — Step 331 STT02-013 over STT01-011 aura

- Long deterministic probe after the step-330 chain fix advanced to `GENERIC step=331 action_type=2`.
- Row: active player 0 plays hand `STT02-013` to empty alley slot 2 while opponent `STT01-011` is in garden.
- Existing `STT02-013` reveal-play mask treated any `STT01-011` on board as a generic-only passive watcher; this is the same recompute-safe aura shape already admitted for `STT02-003`.
- Updated `_play_stt02_013_reveal_fast_mask` to ignore `STT01-011` in the passive-watch gate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask change.

## 2026-06-19 — Step 331 STT03-006 restore-active trigger

- Re-probe after the `STT02-013` mask fix passed the play row but stopped at the same step on `GENERIC step=331 action_type=14`.
- Dumped rows included two `STT02-001` response effects and one active `STT03-006` discard effect. The suspicious row is the chained `STT03-006`: owner/active player 1, source in discard, `trig_count=1`, and original active player likely saved for restoration after the triggered effect resolves.
- Existing `_effect_stt03_006_fast_mask` rejected any `ab_restores_active`/`ab_saved_active` context even though `step_effect_stt03_006_fast` clears context through `_clear_context`, which already restores the saved active player.
- Removed the restore-active rejection from the `STT03-006` effect mask; the rest of the mask still requires the exact discard source, hand target, clean queues, and optional queued follow-up `STT03-006`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask change.

## 2026-06-19 — Step 332 STT02-013 selection over STT01-011 aura

- Re-probe after allowing restore-active `STT03-006` effects advanced to `GENERIC step=332 action_type=18`.
- Row: active player 0 resolves `STT02-013` selection pick, choosing selected index 2 (`STT02-005`, water IKZ cost 1) to hand while opponent `STT01-011` remains in garden.
- `_select_stt02_013_pick_fast_mask` still treated `STT01-011` as a generic-only passive watcher; selection-to-hand does not change board passives, and selection-to-alley already recomputes passives on placement.
- Updated the `STT02-013` selection mask to ignore `STT01-011` in the same passive-watch gate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask change.

## 2026-06-19 — Step 333 STT01-005 activation with empty hand

- Re-probe after the `STT02-013` selection fix advanced to `GENERIC step=333 action_type=12`.
- Row: active player 1 activates alley `STT01-005` from slot 4 with an empty hand.
- C validation only requires `STT01-005` in alley and deck count at least one. Its deferred-cost behavior uses the pre-cost hand count: with zero hand cards before activation, it sacrifices/draws three and resolves immediately instead of entering a discard selection.
- Existing split mask/helper required `hand_count >= 2` and always entered effect selection. Removed the hand-count mask gate and updated `step_activate_stt01_005_fast` to clear context immediately when the pre-cost discard count is zero.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the activation fix.

## 2026-06-19 — Handoff refresh after step 333

- Updated `jax_env/HANDOFF.md` top section again so it no longer points at the step-330 frontier.
- Current recorded frontier is the post-fix long probe after step 333: `STT01-005` alley activation with an empty pre-cost hand.
- No new parity result yet; the deterministic frontier probe is still running.

## 2026-06-19 — Step 337 response NOOP leader into garden

- Re-probe after the `STT01-005` activation fix advanced to `GENERIC step=337 action_type=0`.
- Dumped NOOP rows included a response-window pass with pending combat: player 1 leader `STT01-001` attacking player 0 garden `AZK01-022`.
- Existing response NOOP masks handled fizzle, garden/entity attacker combat, and attacker-into-leader damage, but not a clean leader attacker into a garden defender after response priority passed.
- Extended `_response_noop_combat_fizzle_fast_mask` to admit clean non-leader defender combat; it still rejects when-attacked, takes/deals-damage, when-destroyed, frozen, and godmode shapes, and dispatches through the existing `transition_to_combat_resolve` + `combat_resolve` helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response NOOP mask change.

## 2026-06-19 — Handoff refresh after step 337

- Updated `jax_env/HANDOFF.md` top section to reflect the response-window NOOP combat frontier/fix.
- Current recorded frontier is the post-fix long probe after step 337.

## 2026-06-19 — Step 337 AZK01-031 Tidal Insight split path

- Re-probe after the response NOOP fix advanced to `GENERIC step=337 action_type=8`.
- Row: active player 0 plays hand `AZK01-031` (`Tidal Insight`) from `zpos=0`; board contains no known dirty target interaction for this reveal spell.
- Added AZK01-031 split paths:
  - spell play pays/discards, reveals top 3, enters optional Water-card selection when a Water card is present, or bottom/top-deck ordering when no pick is available;
  - selection pick moves one Water card to hand and sends remaining revealed cards to the bottom/top-deck ordering phase;
  - top-deck selection action routes `Act.TOP_DECK_CARD` through `selection.process_top_deck`.
- Added vector import/wrapper/JIT/masks/dispatch/trace entries for `play_spell_azk01_031`, `select_azk01_031`, and `top_deck`.
- Also admitted `AZK01-031` into optional selection-skip and bottom-deck masks so the "up to 1" pick and remaining-card cleanup stay split instead of falling back to generic.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-031 plumbing.

## 2026-06-19 — Handoff refresh after AZK01-031

- Updated `jax_env/HANDOFF.md` top section so it no longer points at the response-NOOP frontier as the latest code change.
- Current recorded frontier is the post-fix long probe after step 337 `AZK01-031` spell play.

## 2026-06-19 — AZK01-031 remaining-mask correction

- First AZK01-031 re-probe still stopped at the same step-337 spell row.
- Root cause: the new AZK01-031 split masks were included in the trace-only `remaining_mask` but omitted from the actual fallback `remaining` mask near the generic dispatch, so even matched rows still called `_get_step_type_fn(8)`.
- Added `play_spell_azk01_031_mask_host`, `select_azk01_031_pick_mask_host`, and `top_deck_card_mask_host` to the actual fallback exclusion list.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the remaining-mask fix.

## 2026-06-19 — Handoff validation note after AZK01-031 correction

- Updated the `jax_env/HANDOFF.md` validation sentence to mention the AZK01-031 fallback-exclusion fix rather than only the initial split-path plumbing.

## 2026-06-19 — Step 351 AZK01-124 gate portal

- Probe after the AZK01-031 fallback-exclusion fix advanced to `GENERIC step=351 action_type=10`.
- Row: active player 1 gate-portals `STT03-013` from alley slot 4 to garden slot 3 while using `AZK01-124` (`Gate of Devotion`).
- Existing gate portal split mask only admitted AZK01-124 when no sacrifice target existed; this row has valid untapped friendly garden entities with IKZ cost within the portaled card's gate power.
- Added AZK01-124 gate-portal confirmation/cost/effect split paths:
  - portal opens optional confirmation when clean sacrifice targets exist;
  - confirm enters cost selection;
  - cost selection sacrifices one clean non-portaled friendly garden entity, stores its health as damage, and either clears immediately when no enemy garden effect target exists or enters optional effect selection;
  - effect selection either skips or deals the stored damage to a clean enemy garden entity.
- Added vector import/wrapper/JIT/masks/dispatch/trace entries for `confirm_azk01_124`, `select_cost_azk01_124`, and `effect_azk01_124`, and admitted clean AZK01-124 target-bearing gate portals in `_gate_portal_simple_fast_mask`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-124 plumbing.

## 2026-06-19 — Handoff refresh after AZK01-124

- Updated `jax_env/HANDOFF.md` top section to record the step-351 AZK01-124 gate-portal frontier and validation note.

## 2026-06-19 — Step 352 AZK01-124 queued STT03-013 trigger

- Re-probe after the AZK01-124 portal split advanced to `GENERIC step=352 action_type=16`.
- Row: active player 1 confirms `AZK01-124` while the portaled `STT03-013` enter-garden trigger is queued behind the active Gate of Devotion confirmation (`trig_count=1`, `trig_timing=15`).
- Relaxed the AZK01-124 confirm/cost/effect masks to allow exactly that queued `STT03-013` follow-up while preserving redirect/combat/passive cleanliness.
- Updated AZK01-124 cost/effect clearing and generic confirm-clear handling to begin a queued `STT03-013` optional confirmation after the active context clears, matching static trigger-drain order.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the queued `STT03-013` handling.

## 2026-06-19 — Step 358 STT04-004 over inert STT03-013

- Re-probe after queued `STT03-013` handling advanced to `GENERIC step=358 action_type=14`.
- Dumped rows included `STT04-004` effect selection from alley targeting enemy garden slot 0 while `STT03-013` was already in garden.
- `_effect_stt04_004_fast_mask` rejected any `STT03-013` board presence as a passive watcher, but `STT03-013` is inert after its enter-garden optional trigger has resolved.
- Relaxed that passive-watch gate to ignore `STT03-013`, matching existing simple-play and gate-portal mask treatment.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the STT04-004 mask change.

## 2026-06-19 — Step 361 AZK01-028 over STT01-011 aura

- Re-probe after the STT04-004 mask relaxation advanced to `GENERIC step=361 action_type=1`.
- Row: active player 0 plays `AZK01-028` to empty garden slot 2 while opponent `STT01-011` is in garden.
- The AZK01-028 helper pays/places the card, discards the remaining hand, returns garden entities to hand, and recomputes passives; `STT01-011` has no return trigger and is already treated as recompute-safe in other split masks.
- Relaxed `_play_azk01_028_fast_mask` to ignore `STT01-011` in its passive-watch gate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the AZK01-028 mask change.

## 2026-06-19 — Step 368 gate portal over STT01-009 passive

- Re-probe after the AZK01-028 mask relaxation advanced to `GENERIC step=368 action_type=10`.
- Row: active player 1 gate-portals alley `STT01-008` to garden while `STT01-009` is already in garden.
- The gate-portal helper already runs `recompute_passives` after placement; `STT01-009` is a deterministic self passive keyed only by garden zone plus weapon discard count and is treated as recompute-safe elsewhere.
- Relaxed `_gate_portal_simple_fast_mask` to ignore `STT01-009` in the passive-watch gate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the gate mask change.

## 2026-06-19 — Step 371 STT02-003 over STT01-009 passive

- Re-probe after the gate mask relaxation advanced to `GENERIC step=371 action_type=1`.
- Row: active player 0 plays `STT02-003` from hand to empty garden slot 0 while opponent `STT01-009` remains in garden.
- The STT02-003 reveal-play mask ignored `STT01-008` and `STT01-011` passive watchers but not `STT01-009`; this self passive is unaffected by the opponent's hand play and is recompute-safe for this split path.
- Relaxed `_play_stt02_003_reveal_fast_mask` to ignore `STT01-009` in its passive-watch gate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the STT02-003 mask change.

## 2026-06-19 — Step 378 AZK01-127 over inert STT03-013

- Re-probe after the STT02-003 passive-watch relaxation advanced to `GENERIC step=378 action_type=14`.
- Row: active player 1 resolves `AZK01-127` response effect targeting enemy garden slot 3 while two `STT03-013` cards are already in garden.
- The AZK01-127 effect mask rejected lethal 1-damage targets when any non-inert passive watcher was on board; `STT03-013` is inert after its enter-garden optional trigger has resolved and is already ignored by other split masks.
- Relaxed `_effect_azk01_127_fast_mask` to treat `STT03-013` as inert for this passive-watch gate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the AZK01-127 mask change.

## 2026-06-19 — Step 380 STT02-015 over STT01-009 passive

- Re-probe after the AZK01-127 inert-watch relaxation advanced to `GENERIC step=380 action_type=14`.
- Row: active player 0 resolves `STT02-015` response effect, returning an enemy low-cost garden entity while `STT01-009` is in the opponent garden.
- The STT02-015 helper returns the target, clears context, and recomputes passives; `STT01-009`/`STT01-011`/resolved `STT03-013` watchers are safe for this path when no `STT02-010` return observer exists.
- Relaxed `_effect_stt02_015_fast_mask` inert passive-watch handling accordingly.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the STT02-015 mask change.

## 2026-06-19 — Step 402 AZK01-032 over STT01-011 aura

- Re-probe after the STT02-015 mask relaxation advanced to `GENERIC step=402 action_type=13`.
- Row: active player 0 resolves `AZK01-032` cost selection, returning friendly garden `AZK01-028`, while opponent `STT01-011` is in alley.
- The AZK01-032 cost/effect helpers clear context and recompute passives after returns, and still reject `STT02-010` return observers and targets with return triggers.
- Relaxed AZK01-032 cost/effect passive-watch gates for recompute-safe static/resolved watchers (`STT01-008`, `STT01-009`, `STT01-011`, `AZK01-019`, `STT03-013`).
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the AZK01-032 mask changes.

## 2026-06-19 — Step 407 main NOOP with AZK01-011 cleanup

- Re-probe after the AZK01-032 mask changes advanced to `GENERIC step=407 action_type=0`.
- Dumped rows were main-phase `NOOP`; rows with active `AZK01-011` in garden need its end-turn sacrifice hook before turn cleanup.
- Reworked `step_main_noop_azk01_011_fast` to use the full recompute/end-turn/start-turn sequence after applying the single AZK01-011 EOT hook instead of the simple no-cleanup start-turn shortcut.
- Relaxed the AZK01-011 main-NOOP mask for recompute-safe passive watchers and cleanup that the full phase helpers now handle, while still requiring no extra start triggers, no sacrifice/token cleanup, and only one AZK01-011 EOT source.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the main-NOOP changes.

## 2026-06-19 — Step 425 AZK01-006 when-attacked bridge

- Re-probe after the AZK01-011 main-NOOP fix advanced to `GENERIC step=425 action_type=6`.
- Dumped attack rows included env 2 attacking opponent garden `AZK01-006`, whose optional When Attacked return-to-hand trigger is implemented by the generic engine but not by any split attack path.
- Added a conservative `_attack_azk01_006_when_attacked_fast_mask` plus a static-action `Act.ATTACK` JIT bridge for unmodified garden-vs-garden attacks into unattached `AZK01-006`. The bridge uses `step_with_legal_count_static_action`, so it keeps generic JAX semantics for this rare trigger shape without invoking the dynamic step-type fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the AZK01-006 split bridge.

## 2026-06-19 — Step 426 AZK01-006 confirmation bridge

- Re-probe after the AZK01-006 attack bridge advanced to `GENERIC step=426 action_type=16`.
- Row: combat-resolve confirmation for optional `AZK01-006` When Attacked trigger, source owner/source `(1, 7)`, legal actions `CONFIRM_ABILITY` or `NOOP`.
- Added conservative static-action bridges for both confirm and decline while `AZK01-006` is the optional source in combat resolve. This keeps the generic trigger semantics and prevents the split router from falling back to dynamic step-type selection.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the confirmation bridge.

## 2026-06-19 — Step 434 STT02-009 over enemy AZK01-019

- Re-probe after the AZK01-006 confirmation bridge advanced to `GENERIC step=434 action_type=2`.
- Row: active player 0 plays `STT02-009` from hand to alley slot 2 while opponent has `AZK01-019` in garden. The existing STT02-009 confirm path rejected the row because its passive-watch gate scanned both boards.
- `_enter_board_slot` only queues self-passive zone events for the acting player. Relaxed `_play_stt02_009_confirm_fast_mask` to ignore enemy `AZK01-019` while still rejecting active-player self-passive watchers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the STT02-009 mask relaxation.

## 2026-06-19 — Step 436 STT02-009 cost over enemy AZK01-019

- Re-probe after the STT02-009 play mask relaxation advanced to `GENERIC step=436 action_type=13`.
- Row: `STT02-009` cost selection returning active-player garden slot 3 (`AZK01-011`) while the only passive watcher on the other board is enemy `AZK01-019`.
- Relaxed `_select_cost_stt02_009_fast_mask` the same way as the play mask: enemy `AZK01-019` cannot observe an acting-player cost return, while active-player passive watchers remain rejected.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the STT02-009 cost mask relaxation.

## 2026-06-19 — Step 437 STT02-009 effect over enemy AZK01-019

- First re-probe after the STT02-009 cost mask relaxation timed out after printing through step 420; retry completed the new compile and advanced to `GENERIC step=437 action_type=14`.
- Row: `STT02-009` effect selection targeting opponent garden `AZK01-006` while that opponent also has `AZK01-019` in garden.
- Updated `step_effect_stt02_009_fast` to run `recompute_passives` after clearing the ability context, matching generic `apply_user_action_static`'s passive drain after an ability action.
- Relaxed `_effect_stt02_009_fast_mask` to admit target-owner `AZK01-019`; other passive watchers remain rejected.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT02-009 effect update.

## 2026-06-19 — Step 440 leader attack with incoming-only staff

- Re-probe after the STT02-009 effect update advanced to `GENERIC step=440 action_type=6`.
- Row: active leader `STT04-001` attacks opposing leader while equipped with `AZK01-018` (`cmb_in_perm` only). The attack declaration only opens/stores combat or deals outgoing leader damage; the attacker's incoming modifier is irrelevant for this action.
- Relaxed leader-attack response/simple masks to ignore attacker-side incoming combat modifiers while still rejecting outgoing attacker modifiers and defender modifiers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the leader-attack mask relaxation.

## 2026-06-19 — Step 444 response-window weapon attach

- Re-probe after the leader-attack mask relaxation advanced to `GENERIC step=444 action_type=7`.
- Row: defender in response window attaches a weapon from hand to its own board while combat is pending. The existing simple attach helper preserves phase/combat/active player and is valid for this response action; only the host mask was main-phase-only.
- Relaxed `_attach_weapon_simple_fast_mask` to admit clean response-window attach actions where `combat_defender_player` is the active player.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the attach mask relaxation.

## 2026-06-19 — Step 452 static play-entity bridge

- Re-probe after response-window attach advanced to `GENERIC step=452 action_type=2`.
- Rows included unsupported/special play-to-alley shapes (`AZK01-024`, `AZK01-069`) that require full generic on-play/selection semantics rather than another narrow mask-only relaxation.
- Added a static-action play-entity bridge for remaining play-to-garden/play-to-alley rows not covered by dedicated split masks. It calls `step_with_legal_count_static_action` with static `PLAY_ENTITY_TO_GARDEN`/`PLAY_ENTITY_TO_ALLEY`, preserving generic JAX semantics without invoking dynamic step-type fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the static play bridge.

## 2026-06-19 — Step 426 AZK01-006 manual confirmation helper

- Verbose re-probe after the static play bridge completed step 425, then timed out before returning from step 426 with action types `[0, 10, 16, 14]`; the row was `AZK01-006` optional When Attacked confirmation.
- Replaced the static `CONFIRM_ABILITY`/`NOOP` bridges for this shape with `step_confirm_azk01_006_when_attacked_fast`, which applies the optional return on confirm, clears the transferred ability context, recomputes passives, and runs combat resolution directly.
- Removed the separate no-op static bridge from `python/src/azk_puffer/jax_vector.py` so this shape no longer asks XLA to compile the generic static confirmation action.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the manual helper.

## 2026-06-19 — Manual confirmation probe through step 450

- Re-ran the verbose deterministic `JaxVecEnv(4, seed=1)` probe with `AZK_JAX` generic fallback trapped.
- The manual `AZK01-006` helper fixed the previous step-426 compile stall:
  - step 425 static play bridge still compiled slowly (`841.3s`);
  - step 426 returned in `4.4s`;
  - step 437 `STT02-009` effect helper compiled in `206.8s`;
  - result was `NO_GENERIC through 450 steps`.
- The remaining slowdown is compile cost from broad/static split branches, not a generic fallback at step 426.

## 2026-06-19 — Step 453 AZK01-024 decline and AZK01-069 selection skip

- Continued from a step-450 checkpoint to avoid replaying earlier long compiles.
- Step 452 still used the static play bridge and compiled slowly (`871.7s`), then the next generic fallback was step 453 with action type `NOOP`.
- Generic rows:
  - env 0: optional `AZK01-024` confirmation decline (`CONFIRM_ABILITY`/`NOOP` legal); extended `step_confirm_azk01_024_fast` and its mask to clear context on decline.
  - env 2: optional `AZK01-069` `SELECTION_PICK` skip; added `AZK01-069` to the shared selection-pick NOOP mask whitelist.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the step-453 changes.

## 2026-06-19 — Step 454 AZK01-069 bottom-deck cleanup

- Re-probe from the step-450 checkpoint advanced through the step-453 NOOP rows, then stopped at step 454.
- Row: active player 1 resolving `AZK01-069` `BOTTOM_DECK` cleanup after skipping the optional selection pick; legal actions were bottom one selected card or bottom all.
- Added `AZK01-069` to the shared bottom-deck card/all mask whitelist. The existing `step_bottom_deck_card_fast` / `step_bottom_deck_all_fast` helpers use the generic selection runtime, so no new helper was needed.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the bottom-deck whitelist update.

## 2026-06-19 — Step 464 AZK01-084 discard-selection spell

- Re-probe from the step-450 checkpoint advanced to step 464.
- Row: active player 1 played `AZK01-084` (`Link` support spell) from hand, with matching Normal low-cost entities in discard.
- Generalized the existing discard-to-selection spell path used by `AZK01-086`:
  - `step_play_spell_azk01_086_fast` now also handles `AZK01-084`, moving Normal entities with IKZ cost <= 6 from discard to selection with pick max 1.
  - `step_select_azk01_086_pick_fast` and its vector mask now also admit `AZK01-084`, relying on the shared selection runtime to move the picked card to hand and return the rest to discard.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-084 path.

## 2026-06-19 — Step 471 AZK01-070 response over passive observer

- Re-probe from the step-450 checkpoint advanced to step 471.
- Row: response-window `AZK01-070` garden activation from slot 4 while friendly `AZK01-019` was on board.
- The activation taps and deals 1 self-damage to `AZK01-070`, then enters effect selection; it does not change zones. Relaxed `_activate_azk01_070_fast_mask` to ignore inert `AZK01-019` passive watching, matching the earlier STT02-009/AZK01-019 relaxations.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-070 mask relaxation.

## 2026-06-19 — Step 472 AZK01-070 effect over passive observer

- Re-probe after the activation mask relaxation advanced one step to the paired `AZK01-070` effect selection.
- Row: active player 1 selected enemy garden slot 3 while friendly `AZK01-019` was on board.
- Relaxed `_effect_azk01_070_fast_mask` to ignore inert `AZK01-019` passive watching; the effect only applies an attack modifier and the helper recomputes passives after clearing context.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the effect-mask relaxation.

## 2026-06-19 — Step 489 attack with pending AZK01-019 passives

- Re-probe from the step-450 checkpoint advanced to step 489.
- Row: active player 1 attacked the opposing leader while `passive_queue_count=4`; board state showed friendly `AZK01-019` as the passive source.
- `step_attack_leader_response_fast` already recomputes passives before declaring the attack. Relaxed `_attack_leader_response_fast_mask` to allow pending passive queues when the only non-inert passive watcher is `AZK01-019`, while still rejecting pending STT02-012 events and other passive sources.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the attack mask relaxation.

## 2026-06-19 — Step 489 attack over inactive STT02-012 alley

- The first step-489 relaxation still rejected the row because `STT02-012` was in alley and counted as a non-inert passive watcher by the host mask.
- `STT02-012` only watches garden events; this row had no pending STT02-012 event bits, and `step_attack_leader_response_fast` recomputes passives before declaration.
- Updated the same attack-leader response mask to treat alley `STT02-012` as inert for this passive-queue admission.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the inactive-STT02-012 relaxation.

## 2026-06-19 — Step 492 AZK01-127 into AZK01-062 redirect

- Re-probe advanced past step 489 and stopped at step 492 on `AZK01-127` effect selection targeting enemy `AZK01-062`.
- The existing `AZK01-127` mask deliberately rejected `AZK01-062` because effect damage defers into the redirect queue and must begin `AZK01-062`'s redirect effect instead of exposing a queued trigger.
- Updated `step_effect_azk01_127_fast` to pop and resolve the queued `AZK01-062` takes-damage trigger when redirect damage is deferred.
- Relaxed `_effect_azk01_127_fast_mask` for the clean unused-redirect `AZK01-062` target shape, while keeping the direct-damage path conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the redirect-trigger update.

## 2026-06-19 — Step 493 AZK01-062 redirect selection during combat

- Re-probe from the step-450 checkpoint advanced through the step-492 `AZK01-127` redirect fix and stopped at step 493.
- Row: active player 1 was resolving `AZK01-062` redirect effect selection during a response-window combat (`combat=(29,0,dp=0)`) with one pending redirect entry. Action `[14, 7, 0, 0]` selected an enemy garden entity as the redirect target.
- Existing `step_effect_azk01_062_fast` and its host mask only admitted non-combat redirect resolution. Updated the helper to close the response window and auto-resolve combat when no queued work or defender response remains, matching the existing response-damage helpers.
- Relaxed `_effect_azk01_062_fast_mask` to admit response-window combat contexts while still requiring an active redirect entry, no trigger/passive/STT02-012 queue work, and clean redirect targets.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the combat redirect update.

## 2026-06-19 — Step 496 AZK01-032 cost over inactive STT02-012 alley

- Saved-checkpoint re-probe confirmed the step-493 `AZK01-062` redirect fix, advanced to step 496, and stopped on `AZK01-032` cost selection.
- Row: active player 0 selected a friendly garden entity as `AZK01-032`'s cost while `STT02-012` was in that player's alley. The cost-return helper/mask already ignored inert `AZK01-019` and other passive observers, but still treated alley `STT02-012` as an active passive watcher.
- Relaxed `_select_cost_azk01_032_fast_mask` and paired `_effect_azk01_032_fast_mask` to treat alley `STT02-012` as inert. The helper remains restricted to clean return targets with no attachments or when-returned timing.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-032 mask relaxation.

## 2026-06-19 — Step 498 static play bridge with pending passives

- Re-probe from `/tmp/jax_probe_496.pkl` advanced through the step-496 `AZK01-032` cost fix and stopped at step 498.
- Row: active player 0 played `STT02-009` from hand to alley while `passive_queue_count=4`; no trigger/redirect/ability context was active and `stt02_012_event_pending` was clear.
- Rather than clone generic play/passive ordering into another card-specific helper, widened `_play_entity_static_bridge_fast_mask` to allow pending passive queues for static play-entity actions. The bridge calls `step_with_legal_count_static_action`, so it preserves generic static play semantics without falling back through the dynamic generic dispatch.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static bridge relaxation.

## 2026-06-19 — Step 498 gate portal with active STT02-012 aura

- Re-probe from `/tmp/jax_probe_498.pkl` confirmed the static play bridge rows, then stopped on another step-498 row with `action_type=10`.
- Row: active player 0 gate-portaled alley slot 4 to garden while `STT02-012` was in that player's garden. The gate helper already calls `_enter_board_slot`, `recompute_passives`, and the queued enter-garden trigger bridge; the host mask was still treating any garden `STT02-012` as an unsupported passive watcher.
- Relaxed `_gate_portal_simple_fast_mask` to treat `STT02-012` as supported for gate portal rows, not only when it is in alley.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the gate-mask relaxation.

## 2026-06-19 — Step 500 AZK01-021 selection pick

- Re-probe from `/tmp/jax_probe_498.pkl` advanced through the static play and gate fixes to step 500, then stopped on `AZK01-021` `SELECTION_PICK`.
- Row: `AZK01-021` had revealed five cards, pick max 1, and action `[18, 2, 0, 0]` selected a Driftward card to hand. The existing reveal-pick fast helper only admitted `AZK01-031` Water picks.
- Generalized the `AZK01-031` reveal-pick helper and host mask to also handle `AZK01-021` Driftward picks with a five-card selection, added `AZK01-021` to optional selection-skip and bottom-deck cleanup masks, and added the cached Driftward subtype table.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-021 reveal-pick update.
  - `PYTHONPATH=jax_env .venv/bin/python - <<'PY' ... cards.subtype_index('Driftward') ... PY` returned subtype index `23`.

## 2026-06-19 — Step 513 AZK01-022 cost over passive auras

- Re-probe from `/tmp/jax_probe_500.pkl` advanced to step 513 and stopped on `SELECT_COST_TARGET` rows.
- Primary uncovered row: `AZK01-022` cost selection in main phase while `STT02-012`/other inert passive auras were on board. The cost action only discards a hand card and opens the already-modeled bounce effect; the paired effect helper recomputes passives after returning the target.
- Relaxed `_select_cost_azk01_022_fast_mask` and `_effect_azk01_022_fast_mask` to ignore inert/simple passive watchers already handled by the helpers (`STT02-012`, `AZK01-019`, `STT03-013`, plus existing `STT01-008`).
- The same generic dump also showed an `STT02-016` response cost row; its existing mask appears to match that shape, so no STT02-016 change was made unless the next probe proves otherwise.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-022 mask relaxation.

## 2026-06-19 — Step 514 leader-garden attack over AZK01-019

- Re-probe from `/tmp/jax_probe_513.pkl` advanced through the step-513 cost fix and stopped at step 514 on a main-phase attack.
- Row: active player 1 leader attacked enemy garden slot 4 while enemy `AZK01-019` was on board. The existing leader/garden combat helper already models the clean no-response combat shape; only the host passive-watch mask treated `AZK01-019` as non-inert.
- Relaxed `_attack_leader_garden_simple_fast_mask` passive watcher admission to ignore inert `AZK01-019` and `STT02-012`, matching prior attack/ability mask relaxations.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the attack-mask relaxation.

## 2026-06-19 — Step 515 response NOOP after when-attacked trigger

- Re-probe from `/tmp/jax_probe_514.pkl` advanced through the step-514 attack fix and stopped at step 515 on `NOOP` rows.
- The response-window row was clean entity combat after an `AZK01-006` defender's When Attacked window had already been processed (`ab_phase=NONE`, `trig_count=0`). The response NOOP entity-combat mask still rejected any defender with When Attacked timing even though no trigger remained queued.
- Relaxed `_response_noop_entity_combat_fast_mask` to ignore declaration-time `when attacked` timing once the response window has no queued trigger/ability work. This lets the existing helper resolve the pending combat.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the response-NOOP mask relaxation.

## 2026-06-19 — Step 516 main NOOP with STT04-003 and timed grants

- Re-probe from `/tmp/jax_probe_515.pkl` advanced through the step-515 response NOOP fix and stopped at step 516 on a main-phase `NOOP`.
- Row: active player 1 had exactly one legal action (`NOOP`) while a clean `STT04-003` start-of-each-turn trigger was present, an alley `STT02-004` retained an EOT attack buff that C intentionally carries through end turn, and player 0's `STT03-006` had a one-tick start-phase Defender grant.
- The existing `STT04-003` main-NOOP fast mask rejected alley EOT attack buffs and any start-phase timed grant. The helper already uses full `end_turn`, so alley attack buffs are safe; `_simple_start_turn_no_triggers` now ticks start-phase timed grants before draw/IKZ grant, and the `STT04-003` mask admits this shape.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the step-516 main-NOOP update.
  - A checkpoint diagnostic from `/tmp/jax_probe_516.pkl` now routes env 1 through `_main_noop_stt04_003_fast_mask`.

## 2026-06-19 — Step 522 AZK01-092 reveal spell

- Re-probe from `/tmp/jax_probe_516.pkl` advanced through step 521 and stopped at step 522 on `PLAY_SPELL_FROM_HAND`.
- The uncovered row was `AZK01-092` (`Lotus of Reflection`): pay 2, reveal top 5, and enter a pick-or-bottom-deck flow for Water cards with cost <= 2. The `AZK01-002` row in the same dump already matched its fast mask; it only appeared because the generic trap reports all rows with the same action type.
- Generalized the existing `AZK01-031` reveal-spell helper and mask to also handle `AZK01-092` with a dynamic reveal count (`3` vs `5`) and the stricter Water/cost <= 2 predicate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-092 play-spell update.
  - A checkpoint diagnostic from `/tmp/jax_probe_522.pkl` now routes env 0 through the generalized `AZK01-031/AZK01-092` play-spell mask and env 3 through the existing `AZK01-002` mask.

## 2026-06-19 — Step 523 AZK01-092 selection-to-garden

- Re-probe from `/tmp/jax_probe_522.pkl` advanced through the `AZK01-092` spell play and stopped at step 523 on `SELECT_TO_GARDEN`.
- Row: `AZK01-092` had five Water/cost <= 2 cards in selection and selected `AZK01-021` to garden slot 1 while the garden was full. The existing `STT02-013` reveal-pick helper only admitted hand/alley placement and only for `STT02-013`.
- Generalized the `STT02-013` reveal-pick helper/mask for `AZK01-092`: source-specific selection count bounds (`3` vs `5`), `SELECT_TO_GARDEN` for `AZK01-092`, garden replacement checks, and admission for queued `AZK01-021` on-play work after the current reveal-selection bottom-deck flow.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-092 selection-placement update.
  - A checkpoint diagnostic from `/tmp/jax_probe_523.pkl` now routes env 0 through `_select_stt02_013_pick_fast_mask`.

## 2026-06-19 — Step 524 AZK01-092 bottom-deck with queued placement work

- Re-probe from `/tmp/jax_probe_523.pkl` advanced through the `AZK01-092` selection-to-garden helper and stopped at step 524 on `BOTTOM_DECK_CARD`.
- Row: `AZK01-092` was in `BOTTOM_DECK` with four remaining selection cards after placing `AZK01-021`; the placement left one queued trigger and two passive recompute events. The bottom-deck mask previously required all queues/passive work to be empty.
- Added `AZK01-092` to the reveal bottom-deck source allowlist and allowed its bottom-deck actions to proceed while the queued on-play/passive work waits behind the active bottom-deck phase.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the bottom-deck mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_524.pkl` now routes env 0 through `_bottom_deck_azk01_003_fast_mask(all_cards=False)`.

## 2026-06-19 — Bottom-deck completion now drains queued work

- The first step-524 re-probe showed that routing the `AZK01-092` bottom-deck actions was not enough: after `BOTTOM_DECK_ALL`, the fast bottom-deck wrapper left the selected `AZK01-021` on-play trigger and passive queue pending while exposing normal attack actions at step 526.
- Updated `step_bottom_deck_card_fast` and `step_bottom_deck_all_fast` to, when a bottom-deck action clears the active selection context, recompute passives and begin the next queued triggered effect via `pop_effect`/`resolve_triggered_effect`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the bottom-deck completion update.

## 2026-06-19 — Step 533 AZK01-069 Beanz selection pick

- Re-probe from `/tmp/jax_probe_524.pkl` advanced through the bottom-deck completion fix to step 533 and stopped on `SELECT_FROM_SELECTION`.
- Row: active `AZK01-069` reveal selection picked `AZK01-067`, a Beanz card, from a five-card selection. The existing subtype reveal-pick helper only admitted `AZK01-033` Steelborn picks.
- Generalized the `AZK01-033` selection-pick helper/mask to also support `AZK01-069` with a cached Beanz subtype table.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the Beanz pick update.
  - A checkpoint diagnostic from `/tmp/jax_probe_533.pkl` now routes env 2 through `_select_azk01_033_pick_fast_mask`.

## 2026-06-19 — Steps 539-568 response/gate continuation

- Re-probe from `/tmp/jax_probe_533.pkl` advanced to step 539 on `AZK01-029` (`Aquatic Veil`) response spell.
- Added `AZK01-029` split play/cost/effect paths: response play pays/discards and opens two-card hand cost selection; cost selection discards two distinct other hand cards; effect selection targets any leader/garden entity and applies `-3 ATK` EOT before clearing context.
- Step 541 exposed `STT02-015` effect selection while `AZK01-019` was on board; relaxed the effect mask for the already-recomputed passive watcher.
- Steps 555-559 exposed `AZK01-024` selection over self-passive watchers and its selection placing `AZK01-022` to alley. Relaxed `AZK01-019`/`AZK01-073` watcher gates, then bridged the placed `AZK01-022` into its optional bounce confirmation context so the next cost/effect rows stay specialized.
- Step 568 exposed `AZK01-120` gate portal of `STT01-004` from alley to empty garden slot while `AZK01-073` was watching the garden. `step_gate_portal_simple_fast` already runs `_enter_board_slot` plus passive recompute; relaxed `_gate_portal_simple_fast_mask` to treat `AZK01-073` as an inert self-passive watcher for this recompute-backed path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the gate mask relaxation.
  - A checkpoint diagnostic from `/tmp/jax_probe_568.pkl` now reports `_gate_portal_simple_fast_mask == [0, 0, 0, 1]` for the forced env-3 gate action.

## 2026-06-19 — Steps 593 and 618 passive watcher relaxations

- Re-probe from `/tmp/jax_probe_568.pkl` advanced to step 593 after several first-time JIT compiles. The uncovered action type was leader/garden activation:
  - env 0 `STT02-001` response activation already matched `_activate_stt02_001_fast_mask`;
  - env 1 `STT03-001` main activation was rejected only because an existing garden `STT03-013` was treated as a non-inert watcher.
- Relaxed `_activate_stt03_001_fast_mask` to treat `STT03-013` as inert regardless of garden/alley zone; the helper only pays IKZ and arms Bobu's latch, so the existing enter-garden optional trigger source cannot fire from this action.
- Re-probe then advanced to step 618 on `ATTACH_WEAPON_FROM_HAND`: active player 0 attached `AZK01-094` to garden `AZK01-019` while `AZK01-073` was on board.
- `step_attach_weapon_simple_fast` and `step_attach_stt01_013_confirm_fast` both recompute passives after attaching. Relaxed their host masks to treat `AZK01-073` the same way as `AZK01-019`/`AZK01-010` for watched board and target-passive checks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after both fixes.
  - Checkpoint diagnostics report `_activate_stt03_001_fast_mask == [0, 1, 0, 0]` at `/tmp/jax_probe_593.pkl` and `_attach_weapon_simple_fast_mask == [0, 0, 0, 1]` at `/tmp/jax_probe_618.pkl`.

## 2026-06-19 — Step 621 AZK01-117 Ignition Pact

- Re-probe from `/tmp/jax_probe_618.pkl` advanced to step 621 on `PLAY_SPELL_FROM_HAND`: active player 1 played `AZK01-117` (`Ignition Pact`) from hand.
- C source check: `azk01_117.c` validates any garden entity with IKZ cost <= 5, applies a cost of 2 effect damage to the owner's leader, then grants Charge EOT to the selected garden entity.
- Added split play/effect paths:
  - play helper pays/discards the spell, applies the self-damage cost with the spell as damage source, marks costs applied, and opens mandatory effect selection;
  - effect helper accepts friendly/enemy garden target encoding `0..4` / `5..9`, requires an entity with IKZ cost <= 5, grants Charge through EOT, and clears context.
- Host masks conservatively block the play path when the self-damage leader has takes-damage timing, prevention, godmode, carapace, effect immunity, or <=2 HP; dirty damage routes remain generic.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after AZK01-117 wiring.
  - A checkpoint diagnostic from `/tmp/jax_probe_621.pkl` reports `_play_spell_azk01_117_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 644 AZK01-072 Beanz Mentor attack

- Re-probe from `/tmp/jax_probe_621.pkl` advanced to step 644 on `ATTACK`: active player 0 attacked the opposing leader with garden `AZK01-072` (`Beanz Mentor`).
- C source check: `azk01_072.c` is a mandatory `When Attacking` effect that targets another friendly garden Beanz card and gives it `+1 ATK` EOT.
- Reused and generalized the existing `AZK01-014` attack/effect split path for `AZK01-072`:
  - attack declaration now opens the same effect-selection context for either `AZK01-014` or `AZK01-072`;
  - effect selection now branches by source, requiring Beanz subtype for `AZK01-072` targets and applying `+1` instead of `AZK01-014`'s `+2`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after AZK01-072 generalization.
  - A checkpoint diagnostic from `/tmp/jax_probe_644.pkl` reports `_attack_azk01_014_effect_fast_mask == [0, 0, 1, 0]` for the observed `AZK01-072` leader attack.

## 2026-06-19 — Step 662 STT04-007 takes-damage trigger

- Re-probe from `/tmp/jax_probe_644.pkl` advanced to step 662 on entity combat: active `AZK01-067` attacked enemy garden `STT04-007`.
- `STT04-007` has a mandatory once/turn takes-damage trigger that gives itself `+1 ATK` EOT. The generic helper already records combat damage and queues takes-damage triggers, but the split helper only inlined AZK01-059 and STT04-009 follow-ups.
- Extended `step_attack_entity_mutual_destroy_fast` to pop a queued `STT04-007` combat-damage trigger, apply the `+1 ATK` EOT buff, and mark once-per-turn used for clean nonlethal combat.
- Relaxed `_attack_entity_mutual_destroy_fast_mask` for the narrow nonlethal `STT04-007` defender trigger shape.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after STT04-007 handling.
  - A checkpoint diagnostic from `/tmp/jax_probe_662.pkl` reports `_attack_entity_mutual_destroy_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 663 STT04-007 lethal fizzle and AZK01-047 response attack

- Re-probe from `/tmp/jax_probe_662.pkl` advanced to step 663 on two `ATTACK` rows:
  - env 2: `AZK01-007` attacked damaged `STT04-007` for lethal entity combat. The split helper must pop the queued `STT04-007` takes-damage trigger even when the source leaves play, then do nothing further.
  - env 3: `AZK01-047` attacked an enemy garden entity while response spells were legal. The entity-response attack mask still rejected `AZK01-047`'s modeled `When Attacking` heal path.
- Refined `step_attack_entity_mutual_destroy_fast` so an `STT04-007` queued trigger is always popped, applies `+1 ATK`/marks once only if the source remains in garden/alley, and then recomputes the next effect head for the existing `STT04-009` fizzle handling.
- Relaxed `_attack_entity_mutual_destroy_fast_mask` to admit any clean damaged `STT04-007` defender trigger, including lethal combat, and relaxed `_attack_entity_response_fast_mask` for `AZK01-047` because `step_attack_leader_response_fast` already performs the heal/once trigger before opening the response window.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after both step-663 fixes.
  - A checkpoint diagnostic from `/tmp/jax_probe_663.pkl` reports `_attack_entity_mutual_destroy_fast_mask == [0, 0, 1, 0]` for the lethal `STT04-007` row and `_attack_entity_response_fast_mask == [0, 0, 0, 1]` for the `AZK01-047` response row.

## 2026-06-19 — Step 668 AZK01-072 entity attack

- Re-probe from `/tmp/jax_probe_663.pkl` advanced through step 667 and stopped at step 668 on `ATTACK`.
- The actual uncovered generic row was env 2: `AZK01-072` attacked tapped enemy garden `AZK01-116`; env 1 in the dump (`AZK01-054` leader attack into response) already routes through `_attack_leader_response_fast_mask`.
- The existing `AZK01-014/AZK01-072` attack-declaration helper only allowed leader targets. Extended it to declare the same mandatory `When Attacking` effect before either leader or garden-entity combat, and recompute passives after auto-resolved combat.
- Extended `_attack_azk01_014_effect_fast_mask` to admit clean tapped garden targets for `AZK01-072` when future combat has no damage/death triggers, while preserving the existing friendly buff target availability checks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the entity-target update.
  - A checkpoint diagnostic from `/tmp/jax_probe_668.pkl` reports `_attack_azk01_014_effect_fast_mask == [0, 0, 1, 0]` and `_attack_leader_response_fast_mask == [0, 1, 0, 0]` for the two action-type-6 rows.

## 2026-06-19 — Step 684 AZK01-068 discard effect

- Re-probe from `/tmp/jax_probe_668.pkl` advanced to step 684 on `SELECT_EFFECT_TARGET`: `AZK01-068` had already drawn its cost card from an alley on-play ability and needed to discard one hand card.
- C source check: `azk01_068.c` validates alley source plus non-empty deck, applies cost by drawing 1, then discards one selected friendly hand card.
- Generalized the existing `AZK01-016` hand-discard effect helper/mask for `AZK01-068`'s one-card selection shape (`ab_eff_min/max == 1`, costs already applied). The helper already stores selected hand targets, discards on finish, and clears context.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-068` generalization.
  - A checkpoint diagnostic from `/tmp/jax_probe_684.pkl` reports `_effect_azk01_016_fast_mask == [1, 0, 0, 0]` for the observed `AZK01-068` discard row.

## 2026-06-19 — Step 685 NOOP rows after AZK01-060/STT02-003 prompts

- Re-probe from `/tmp/jax_probe_684.pkl` advanced one step and stopped at step 685 on `NOOP`.
- The action-type dump contained two real NOOP rows:
  - env 2 declined `AZK01-060`'s optional when-attacking confirmation while combat against `AZK01-019` was pending;
  - env 3 skipped an `STT02-003` reveal selection. Its existing `_selection_pick_noop_fast_mask` already matched once checked directly.
- `step_confirm_azk01_060_fast` already models confirm/decline combat, but did not recompute self-passives after combat discards. Added recompute after attacker/defender discard and relaxed `_confirm_azk01_060_fast_mask` to allow the recompute-backed simple passive watchers (`AZK01-019`, `AZK01-073`, `STT02-012`, `STT03-013`) instead of forcing generic fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the confirmation/passive update.
  - A checkpoint diagnostic from `/tmp/jax_probe_685.pkl` reports `_confirm_azk01_060_fast_mask == [0, 0, 1, 0]` and `_selection_pick_noop_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 698 AZK01-058 confirmation over passive watcher

- Re-probe from `/tmp/jax_probe_685.pkl` advanced to step 698 on `CONFIRM_ABILITY`: active player 1 confirmed `AZK01-058`'s optional after-attacking effect while opposing `STT01-009` was a registered passive watcher.
- `AZK01-058` confirmation sacrifices its source and moves into one-target effect selection. The existing helper already modeled that flow; it now recomputes passives after the sacrifice.
- Relaxed the `AZK01-058` confirm/effect host masks to treat the recompute-backed/simple passive watchers (`STT01-009`, `STT01-011`, `AZK01-010`, `AZK01-019`, `AZK01-073`, `STT02-012`, `STT03-013`, plus the existing `STT01-008`) as non-blocking for this flow.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-058` passive-watch update.
  - A checkpoint diagnostic from `/tmp/jax_probe_698.pkl` reports `_confirm_azk01_058_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 700 attacking STT04-007 takes-damage trigger

- Re-probe from `/tmp/jax_probe_698.pkl` advanced to step 700 on `ATTACK`: active `STT04-007` attacked tapped enemy garden `STT01-004`, survived combat damage, and needed its own takes-damage trigger to resolve.
- `step_attack_entity_mutual_destroy_fast` already pops/applies any head `STT04-007` takes-damage trigger after combat; the host mask only admitted the defender-side `STT04-007` case.
- Relaxed `_attack_entity_mutual_destroy_fast_mask` to admit attacker-side `STT04-007` damage triggers and added `STT01-009`/`AZK01-073` to the recompute-safe passive watcher set used by this combat path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the attacker-side trigger mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_700.pkl` reports `_attack_entity_mutual_destroy_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 703 AZK01-121 activation over STT01-009

- Re-probe from `/tmp/jax_probe_700.pkl` advanced to step 703 on `ACTIVATE_ABILITY`: active player 1 used `AZK01-121` leader activation (`[11, 5, 0, 0]`) while opposing `STT01-009` was on board.
- `AZK01-121` only pays IKZ and applies an EOT attack modifier to the leader, capped by entities played this turn; the passive watcher does not observe this mutation.
- Relaxed `_activate_azk01_121_fast_mask` for the same inert/simple passive watcher set used by adjacent activation paths.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the passive-watch relaxation.
  - A checkpoint diagnostic from `/tmp/jax_probe_703.pkl` reports `_activate_azk01_121_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 708 AZK01-087 spell and continuation to 730

- Re-probe from `/tmp/jax_probe_703.pkl` advanced to step 708 on `PLAY_SPELL_FROM_HAND`: active player 1 played `AZK01-087` (`Mizuryuu's Torrent`) from hand.
- C source check: `azk01_087.c` plays when the opponent has garden cards, then selects up to two enemy garden entities with combined IKZ cost <= 5 and bottom-decks them.
- Generalized the existing `AZK01-017` play-spell helper/mask to also open `AZK01-087`'s two-target effect context, then added `step_effect_azk01_087_fast` plus host mask/dispatch for enemy-garden selection, duplicate/combined-cost validation, optional skip, bottom-decking selected targets, and passive recompute after the zone moves.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the `AZK01-087` wiring.
  - A checkpoint diagnostic from `/tmp/jax_probe_708.pkl` reports `_play_spell_azk01_017_fast_mask == [0, 0, 0, 1]` for the observed `AZK01-087` play row.
  - The frontier probe from `/tmp/jax_probe_708.pkl` ran with no generic fallback through step 730.

## 2026-06-19 — Step 731 AZK01-124 portal cost target

- Re-probe from `/tmp/jax_probe_708.pkl` advanced to step 731 on `GATE_PORTAL`: active player 0 used `AZK01-124` to portal `AZK01-105` from alley slot 3 to garden slot 3, with `STT03-006` in garden slot 4 as a valid devotion cost target.
- The gate portal helper and later `AZK01-124` cost-selection helper already allow sacrifice targets with when-destroyed timing here; the vector host mask was stricter and excluded those cost targets, incorrectly forcing generic fallback.
- Relaxed `_gate_portal_simple_fast_mask` so `AZK01-124` cost-target existence matches the helper: any untapped friendly garden entity with IKZ cost at or below gate power and no attachment on that target can open the confirmation, including `STT03-006`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_731.pkl` reports `_gate_portal_simple_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 733 AZK01-124 cost sacrifice

- Re-probe from `/tmp/jax_probe_731.pkl` advanced through the gate-portal confirmation and stopped at step 733 on `SELECT_COST_TARGET`: active player 0 selected garden slot 4 `STT03-006` as `AZK01-124`'s devotion cost after the portal.
- The `AZK01-124` cost-selection helper uses `sacrifice_card`, so it intentionally does not queue `WHEN_DESTROYED` triggers. The host mask still rejected when-destroyed targets, duplicating the earlier portal-target mismatch one phase later.
- Removed the when-destroyed exclusion from `_select_cost_azk01_124_fast_mask` while keeping the existing attachment and godmode gates conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the cost-mask update.
  - Re-probe from `/tmp/jax_probe_731.pkl` advanced past step 733 and saved `/tmp/jax_probe_736.pkl` before the long compile/probe run timed out at step 736.

## 2026-06-19 — Steps 769/772 Bobu and AZK01-103 over pending passives

- Re-probe from `/tmp/jax_probe_736.pkl` advanced to step 769 on `ACTIVATE_ABILITY`: active player 0 used `STT03-001` while opposing garden `STT02-012` was present. The Bobu activation only pays IKZ and arms the latch, so `STT02-012` is inert for this action; `_activate_stt03_001_fast_mask` now treats all `STT02-012` watchers as inert here.
- After that, re-probe advanced to step 772 on `ACTIVATE_ABILITY`: active player 0 activated garden-slot-0 `AZK01-103` while a pending `STT02-012` passive recompute was queued.
- Updated `step_activate_azk01_103_fast` to recompute passives after opening the cost-selection context, and relaxed `_activate_azk01_103_fast_mask` to admit pending passive/STT02-012 queue work for this activation.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after both activation fixes.
  - A checkpoint diagnostic from `/tmp/jax_probe_772.pkl` reports `_activate_azk01_103_fast_mask == [0, 1, 0, 0]` for the observed row.

## 2026-06-19 — Step 779 AZK01-045 reveal over pending passives

- Re-probe from `/tmp/jax_probe_772.pkl` advanced to step 779 on `PLAY_ENTITY_TO_ALLEY`: active player 0 played hand `AZK01-045` to alley while a pending `STT02-012` passive recompute remained queued.
- `AZK01-045`'s reveal helper opens a selection/bottom-deck context after placement. It now recomputes passives after placement and before revealing, matching the generic post-action passive drain before the next legal mask is exposed.
- Relaxed `_play_azk01_045_reveal_fast_mask` to admit pending passive/STT02-012 queue work and to treat `STT02-012` watchers as recompute-backed for this reveal path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the reveal/passive update.
  - A checkpoint diagnostic from `/tmp/jax_probe_779.pkl` reports `_play_azk01_045_reveal_fast_mask` for alley play as `[0, 1, 0, 0]`.

## 2026-06-19 — Step 784 STT03-002 effect over pending passives

- Re-probe from `/tmp/jax_probe_779.pkl` advanced to step 784 on `SELECT_EFFECT_TARGET`: `STT03-002`'s Stonehaven Gate effect selected friendly garden slot 3 while a pending `STT02-012` passive recompute was still queued.
- The `STT03-002` effect helper already clears the ability context and recomputes passives after granting Defender, so the host mask does not need to force generic solely for pending passive/STT02-012 queue work.
- Relaxed `_effect_stt03_002_fast_mask` by removing the passive queue and `stt02_012_event_pending` gates from its clean predicate.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask relaxation.
  - A checkpoint diagnostic from `/tmp/jax_probe_784.pkl` reports `_effect_stt03_002_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 797 AZK01-024 cost over inactive STT02-012

- Re-probe from `/tmp/jax_probe_784.pkl` advanced to step 797 on `SELECT_COST_TARGET`: active player 1 selected a friendly garden entity for `AZK01-024`'s return-to-hand cost while player 0 had `STT02-012` in alley.
- `STT02-012` is a garden-only passive, so an alley copy is inert for `AZK01-024`'s cost bounce. The host mask treated it as a non-inert passive watcher and forced generic fallback.
- Relaxed `_select_cost_azk01_024_fast_mask` to allow inactive-alley `STT02-012` watchers while keeping the existing gates for non-inert watchers and return-trigger targets.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_797.pkl` reports `_select_cost_azk01_024_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 801 response effect selections

- Re-probe from `/tmp/jax_probe_797.pkl` advanced to step 801 on `SELECT_EFFECT_TARGET` with two response-window rows:
  - env 2 `STT02-015` selected enemy garden slot 4; existing `_effect_stt02_015_fast_mask` already matched it once checked directly.
  - env 3 `AZK01-127` selected enemy garden slot 4 `AZK01-054`, a 7-HP inherent-godmode entity.
- `deal_effect_damage` still applies nonlethal damage to godmode targets and only prevents death, so rejecting all godmode targets was too strict for `AZK01-127`'s 1-damage response effect.
- Relaxed `_effect_azk01_127_fast_mask` to allow inherent/granted godmode when the 1 damage is nonlethal, preserving the lethal godmode guard.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_801.pkl` reports `_effect_stt02_015_fast_mask == [0, 0, 1, 0]` and `_effect_azk01_127_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 824 AZK01-032 cost over STT02-012

- Re-probe from `/tmp/jax_probe_801.pkl` advanced to step 824 on `SELECT_COST_TARGET`: active player 1 selected garden slot 0 `AZK01-006` as `AZK01-032`'s cost while friendly garden `STT02-012` was present.
- Returning a friendly garden entity changes the garden-count predicate that `STT02-012` watches. The cost helper now recomputes passives immediately after `return_to_hand`, before deciding whether to enter effect selection.
- Relaxed `_select_cost_azk01_032_fast_mask` to treat `STT02-012` as recompute-backed instead of forcing generic fallback whenever it is on board.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_824.pkl` reports `_select_cost_azk01_032_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 825 AZK01-032 effect over STT02-012

- Re-probe from `/tmp/jax_probe_824.pkl` advanced one step to `AZK01-032`'s optional enemy return effect while `STT02-012` passive work was still queued from the cost return.
- The effect helper returns an opponent garden entity or skips, clears the context, and now recomputes passives before exposing the next legal mask.
- Relaxed `_effect_azk01_032_fast_mask` to admit pending passive/STT02-012 queue work and treat `STT02-012` as recompute-backed, while preserving the `STT02-010` return observer guard.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the effect helper/mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_825.pkl` reports `_effect_azk01_032_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 832 STT02-015 over inactive STT02-012

- Re-probe from `/tmp/jax_probe_825.pkl` advanced to step 832 on `SELECT_EFFECT_TARGET`: active player 0 resolved `STT02-015` in the response window and targeted friendly garden slot 1 while an own `STT02-012` was in alley.
- `STT02-012` is inert in alley, and the `STT02-015` effect helper recomputes passives after returning the selected garden entity to hand.
- Relaxed `_effect_stt02_015_fast_mask` to treat `STT02-012` as recompute-backed/inert for this return effect instead of forcing generic fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_832.pkl` reports `_effect_stt02_015_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 848 STT02-009 cost over STT02-012

- Re-probe from `/tmp/jax_probe_832.pkl` advanced to step 848 on `SELECT_COST_TARGET`: active player 1 selected garden slot 4 `STT02-012` as `STT02-009`'s return-to-hand cost.
- Returning the cost target can change `STT02-012`'s garden-count passive state. The STT02-009 cost helper now recomputes passives immediately after `return_to_hand`, before choosing effect-selection vs exhausted flow.
- Relaxed `_select_cost_stt02_009_fast_mask` to allow `STT02-012` on board for this recompute-backed cost flow.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_848.pkl` reports `_select_cost_stt02_009_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 849 STT02-009 optional skip

- Re-probe from `/tmp/jax_probe_848.pkl` advanced one step to `STT02-009`'s optional enemy return effect with a `NOOP` skip while passive work from the cost return was pending.
- The effect helper already clears context and recomputes passives on both select and skip. The host mask still rejected pending passive/STT02-012 work before the helper could drain it.
- Relaxed `_effect_stt02_009_fast_mask` to admit pending passive/STT02-012 queue work and to treat `STT02-012` as recompute-backed for this effect.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_849.pkl` reports `_effect_stt02_009_fast_mask == [0, 1, 0, 0]`; the second NOOP row already matches `_main_noop_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 891 leader activations over passive watchers

- Re-probe from `/tmp/jax_probe_849.pkl` advanced to step 891 on `ACTIVATE_ABILITY` with two leader rows: env 0 `AZK01-125` and env 2 `STT03-001`.
- `AZK01-125` already matched its activation mask after direct diagnostic. `STT03-001` Bobu activation was still rejecting inert/simple passive watchers (`STT01-009`, `STT01-011`, `AZK01-010`, `AZK01-073`) even though the helper only pays IKZ and arms the Bobu latch.
- Relaxed `_activate_stt03_001_fast_mask` for the same simple passive watcher set used by adjacent leader activations.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the Bobu mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_891.pkl` reports `_activate_azk01_125_fast_mask == [1, 0, 0, 0]` and `_activate_stt03_001_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 895 leader attack response over STT02-012

- Re-probe from `/tmp/jax_probe_891.pkl` advanced to step 895 on `ATTACK`: env 1 attacked the opponent leader with garden `AZK01-105` while `STT02-012` passive work was pending and the opponent leader had a response ability.
- The response attack helper recomputes passives before declaring the attack and opening the response window, so queued `STT02-012` work should not force generic fallback for non-`STT02-012` attackers.
- Relaxed `_attack_leader_response_fast_mask` to treat `STT02-012` as a recompute-backed passive watcher and to allow pending `stt02_012_event_pending` unless the attacker itself is `STT02-012`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - A checkpoint diagnostic from `/tmp/jax_probe_895.pkl` reports `_attack_leader_response_fast_mask == [0, 1, 0, 0]` and `_attack_leader_simple_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 895 AZK01-128 response destroys attached attacker

- The same checkpoint also had env 3 resolving `AZK01-128` in `EFFECT_SELECTION` during a response window. C legal selected target index 0: the current attacker had HP 1 and an attached weapon.
- `destroy_card` intentionally leaves weapons attached, matching C's discard behavior, so the fast mask should not reject an `AZK01-128` target solely because it has attachments.
- Removed the attached-card guard from `_effect_azk01_128_fast_mask`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_azk128_noop.py /tmp/jax_probe_895.pkl` reports `_effect_azk01_128_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 905 AZK01-004 attack buff before response

- Re-probe from `/tmp/jax_probe_903.pkl` advanced to step 905 on `ATTACK`: env 2 attacked a tapped garden entity with `AZK01-004` while the defender had response options.
- The existing `AZK01-004` fast path only targeted leaders. The helper already models the required inline +1 attack buff before checking for a response window, so it can also declare a garden target when response is available and defer combat resolution.
- Generalized `step_attack_azk01_004_leader_fast` target selection to use the action's leader/garden/alley target and relaxed `_attack_azk01_004_leader_fast_mask` to cover clean garden/leader response declarations.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step905.py /tmp/jax_probe_905.pkl` reports `_attack_azk01_004_leader_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 923 weapon attach over STT01-009 watcher

- Re-probe from `/tmp/jax_probe_910.pkl` advanced to step 923 on `ATTACH_WEAPON_FROM_HAND`: env 2 attached `STT01-015` to the leader while `STT01-009` was in alley.
- The attach helper recomputes passives after moving the weapon and paying IKZ. `STT01-009`/`STT01-011` are already treated as simple passive watchers in neighboring play/gate/activation masks, so they should not force generic fallback here.
- Relaxed `_attach_weapon_simple_fast_mask` to treat `STT01-009` and `STT01-011` as supported simple attach watchers for board and target checks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step923.py /tmp/jax_probe_923.pkl` reports `_attach_weapon_simple_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 935 STT01-004 confirmation with passive queue

- Re-probe from `/tmp/jax_probe_923.pkl` advanced to step 935 on `CONFIRM_ABILITY`: env 2 confirmed optional `STT01-004` while generic legal still exposed confirmation despite `passive_queue_count == 5`.
- Confirming `STT01-004` only changes the ability FSM from confirmation to cost selection; it does not consume or mutate passive work. The queued passive work remains pending for the same later drain point as in C.
- Relaxed `_confirm_stt01_004_fast_mask` to allow confirmation with pending passive/STT02 work instead of requiring a clean passive queue.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step935.py /tmp/jax_probe_935.pkl` reports `_confirm_stt01_004_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 936 STT01-004 cost with passive queue

- Re-probe from `/tmp/jax_probe_935.pkl` advanced one step to `STT01-004` cost selection with the same pending passive queue.
- Selecting the weapon cost sacrifices a hand card and reveals the top deck into selection; it does not drain or depend on the queued passive work.
- Relaxed `_select_cost_stt01_004_fast_mask` to allow this cost selection while passive/STT02 work is pending.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step936.py /tmp/jax_probe_936.pkl` reports `_select_cost_stt01_004_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 937 STT01-004 bottom-deck with passive queue

- Re-probe from `/tmp/jax_probe_936.pkl` advanced to `STT01-004` bottom-deck-all after the weapon-cost reveal found no selectable weapon.
- Bottom-decking the revealed selection is independent of the queued passive work left from the original play.
- Relaxed `_bottom_deck_azk01_003_fast_mask` to allow `STT01-004` bottom-deck actions with pending passive/STT02 work, matching the existing exception style for `AZK01-092`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step937.py /tmp/jax_probe_937.pkl` reports `bottom_all == [0, 0, 1, 0]`.

## 2026-06-19 — Step 976 AZK01-127 lethal damage over AZK01-073 watcher

- Re-probe from `/tmp/jax_probe_972.pkl` advanced to step 976 on `AZK01-127` response effect: env 2 targeted enemy garden slot 0 `AZK01-073` at 1 HP.
- `step_effect_azk01_127_fast` applies damage, clears context, and recomputes passives, so simple passive watchers such as `AZK01-073` can be handled without generic fallback.
- Relaxed `_effect_azk01_127_fast_mask`'s passive watcher allow-list to include `STT01-009`, `STT01-011`, `AZK01-010`, and `AZK01-073`, matching adjacent recompute-backed masks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step976.py /tmp/jax_probe_976.pkl` reports `_effect_azk01_127_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 986 AZK01-084 play with passive queue

- Re-probe from `/tmp/jax_probe_976.pkl` advanced to step 986 on `PLAY_SPELL_FROM_HAND`: env 0 played `AZK01-084` while `passive_queue_count == 4`.
- `AZK01-084`/`AZK01-086` play pays IKZ, discards the spell, and moves discard cards into selection; board passive state is not consumed by this transition, and C legal allows it with the queued passive work intact.
- Relaxed `_play_spell_azk01_086_fast_mask` to allow pending passive queue work while still rejecting pending `STT02-012` event work.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step986.py /tmp/jax_probe_986.pkl` reports `_play_spell_azk01_086_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 987 AZK01-084 selection pick with passive queue

- Re-probe from `/tmp/jax_probe_986.pkl` advanced one step to `AZK01-084` `SELECTION_PICK` while `passive_queue_count == 4`.
- The selection pick moves the selected discard entity to hand and clears/returns the selection; the queued passive work remains independent.
- Relaxed `_select_azk01_086_pick_fast_mask` to allow `AZK01-084`/`AZK01-086` selection picks with pending passive queue work while still rejecting pending `STT02-012` event work.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step987.py /tmp/jax_probe_987.pkl` reports `_select_azk01_086_pick_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1008 AZK01-058 after-attack fizzle on death

- Re-probe from `/tmp/jax_probe_987.pkl` advanced to step 1008 on `ATTACK`: env 0 attacked with `AZK01-058` into a tapped garden entity and both combatants died.
- The mutual-combat helper already recomputes passives and does not open `AZK01-058`'s optional after-attacking ability when the attacker leaves play, matching the C legal flow for this row.
- Relaxed `_attack_entity_mutual_destroy_fast_mask` to treat `AZK01-058`'s after-attacking timing as a fizzle when the attacker dies during the combat.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1008.py /tmp/jax_probe_1008.pkl` reports `_attack_entity_mutual_destroy_fast_mask == [1, 0, 0, 0]` and `_attack_entity_response_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1009 response NOOP into frozen defender death

- Re-probe from `/tmp/jax_probe_1008.pkl` advanced to step 1009 on response `NOOP`: env 1 passed response and resolved entity combat where the defender was frozen and died.
- Frozen does not prevent a tapped defender from taking combat damage, and `step_response_noop_entity_combat_fast` ignores frozen status while resolving lethal defender damage and recomputing passives.
- Relaxed `_response_noop_entity_combat_fast_mask` to allow `STT02-010` passive watchers during combat death recompute and to allow a frozen defender when that defender dies in the combat.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1009.py /tmp/jax_probe_1009.pkl` reports `_response_noop_entity_combat_fast_mask == [0, 1, 0, 0]` and `_main_noop_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1031 AZK01-040 combat over AZK01-059 trigger

- Re-probe from `/tmp/jax_probe_1009.pkl` advanced to step 1031 on `AZK01-040` effect selection during combat resolution: env 0 skipped/selected through `AZK01-040` while the opposing attacker was `AZK01-059`.
- `AZK01-059` has a supported takes-damage trigger when it survives damage and another friendly garden entity exists. The `AZK01-040` effect helper clears context and then calls combat resolution, so this supported attacker trigger should not force generic fallback.
- Relaxed `_effect_azk01_040_fast_mask` to allow the specific surviving-attacker `AZK01-059` takes-damage trigger while preserving other damage-trigger guards.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1031.py /tmp/jax_probe_1031.pkl` reports `_effect_azk01_040_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1032 gate portal with queued AZK01-059 trigger

- Re-probe from `/tmp/jax_probe_1031.pkl` advanced to step 1032 on `GATE_PORTAL`: active player 1 used `AZK01-122` to portal an alley `AZK01-059` while a surviving `AZK01-059` takes-damage trigger was still queued from the previous combat.
- Generic legal exposes main actions with this pending trigger; the gate helper does not consume or mutate the trigger queue, so the host mask should not require `trig_count == 0` for this supported pending-trigger shape.
- Relaxed `_gate_portal_simple_fast_mask` to admit exactly one queued `AZK01-059` `WHEN_TAKES_DAMAGE` trigger while preserving the clean-trigger requirement for all other trigger queues.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1032.py /tmp/jax_probe_1032.pkl` reports `_gate_portal_simple_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1033 AZK01-122 selection over queued trigger and opponent watcher

- Re-probe from `/tmp/jax_probe_1032.pkl` reached step 1033 after the `AZK01-122` gate portal. Env 0 selected the revealed `AZK01-056` placement while the prior `AZK01-059` takes-damage trigger was still queued and the opponent had a `STT01-009` passive watcher.
- The placement helper leaves the queued `AZK01-059` trigger intact. Opponent `STT01-009` only watches its owner's discard/zone state, so it is unaffected by placing a selected card on the active player's board.
- Relaxed `_select_azk01_122_place_fast_mask` to admit exactly one queued `AZK01-059` takes-damage trigger and to restrict passive-watch rejection to the selected card owner's board instead of unrelated opponent watchers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1033_masks.py /tmp/jax_probe_1033.pkl` reports all four rows covered, including `select_122_a == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1034 AZK01-056 bottom-deck over queued trigger

- Re-probe from `/tmp/jax_probe_1033.pkl` advanced to step 1034 on `BOTTOM_DECK_CARD`: env 0 was bottom-decking an `AZK01-056` reveal while the queued `AZK01-059` takes-damage trigger from combat still existed.
- `step_bottom_deck_card_fast` already preserves the trigger while selection remains active and pops/resolves it only after the bottom-deck sequence clears the selection, matching generic ordering.
- Relaxed `_bottom_deck_azk01_003_fast_mask` so `AZK01-056` bottom-deck actions admit exactly one queued `AZK01-059` takes-damage trigger with no passive/STT02 pending work.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1034.py /tmp/jax_probe_1034.pkl` reports `bottom_card == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1038 AZK01-059 effect over opponent STT01-009

- Re-probe from `/tmp/jax_probe_1034.pkl` advanced through the queued `AZK01-059` trigger and stopped at its `SELECT_EFFECT_TARGET`: env 0 targeted another friendly garden entity while opponent `STT01-009` was still on board.
- `AZK01-059` only applies an end-of-turn attack modifier to the active player's target and clears context. Opponent `STT01-009` watches only its owner's discard/zone state and is unaffected.
- Relaxed `_effect_azk01_059_fast_mask` to restrict passive-watch rejection to the effect owner's board instead of unrelated opponent watchers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1038.py /tmp/jax_probe_1038.pkl` reports `_effect_azk01_059_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1039 AZK01-062 attacks AZK01-040

- Re-probe from `/tmp/jax_probe_1038.pkl` advanced to step 1039 on `ATTACK`: env 0 attacked tapped `AZK01-040` with `AZK01-062` while opponent `STT01-009` was on board.
- Attack declaration only opens the response window; `AZK01-040`'s when-attacked effect and the later `AZK01-062` combat damage redirect/fizzle are handled by the response/effect/combat steps, not by declaration.
- Relaxed `_attack_entity_response_fast_mask` so `AZK01-040` when-attacked can open the response path on clean damage even when later combat has supported takes-damage work, and allowed the simple passive watcher set at declaration.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1033_masks.py /tmp/jax_probe_1039.pkl` reports `attack_entity_resp == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1041 AZK01-040 skip into AZK01-062 combat fizzle

- Re-probe from `/tmp/jax_probe_1039.pkl` advanced to step 1041 on `NOOP`: env 0 skipped optional `AZK01-040` effect selection, then pending combat would kill both `AZK01-040` and combat-damaged `AZK01-062`.
- `step_effect_azk01_040_fast` now recomputes passives after its auto combat resolution and pops the combat-only `AZK01-062` takes-damage fizzle, matching the existing mutual-combat fast-path cleanup.
- Relaxed `_effect_azk01_040_fast_mask` for combat `AZK01-062` takes-damage fizzles and recompute-safe passive watchers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1041.py /tmp/jax_probe_1041.pkl` reports `_effect_azk01_040_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1043 AZK01-092 places AZK01-022

- Re-probe from `/tmp/jax_probe_1041.pkl` advanced to step 1043 on `SELECT_TO_ALLEY`: env 3 used `AZK01-092`'s reveal selection to place `AZK01-022` from selection.
- The selection helper already queues on-play triggers and bottom-decks the remaining reveal cards before trigger resolution; `AZK01-022` is therefore a supported selection placement target even though its optional bounce is not applied immediately on the placement step.
- Relaxed `_select_stt02_013_pick_fast_mask` to allow selected `AZK01-022` placements through the existing queued-trigger flow.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1043.py /tmp/jax_probe_1043.pkl` reports `_select_stt02_013_pick_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1046 defender AZK01-059 combat trigger

- Re-probe from `/tmp/jax_probe_1043.pkl` advanced to step 1046 on `ATTACK`: env 0 attacked an opposing `AZK01-059`; the attacker died, while the damaged defender survived with another friendly garden entity available.
- The mutual-combat helper already recomputes passives and begins a queued `AZK01-059` takes-damage trigger from the trigger head after combat.
- Relaxed `_attack_entity_mutual_destroy_fast_mask` to recognize the defender-side surviving `AZK01-059` takes-damage trigger, symmetric to the existing attacker-side case.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1033_masks.py /tmp/jax_probe_1046.pkl` reports `attack_entity_simple == [1, 0, 0, 0]` and `attack_leader_resp == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1048 dying AZK01-059 combat trigger fizzle

- Re-probe from `/tmp/jax_probe_1046.pkl` advanced to step 1048 on `ATTACK`: env 0 attacked a tapped `AZK01-059` with `STT01-009`; both entities died in combat.
- Generic queues the `AZK01-059` takes-damage observer during damage resolution, but the source is no longer in `GARDEN`/`ALLEY` after lethal combat, so that trigger fizzles instead of opening effect selection.
- Updated `step_attack_entity_mutual_destroy_fast` to pop a queued `AZK01-059` takes-damage trigger when its source left play, and guarded the surviving-trigger conversion on source zone.
- Relaxed `_attack_entity_mutual_destroy_fast_mask` to admit attacker/defender `AZK01-059` lethal-combat fizzle shapes under otherwise clean takes-damage timing.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1033_masks.py /tmp/jax_probe_1048.pkl` reports `attack_entity_simple == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1048 AZK01-128 over STT01-009 watcher

- After the dying `AZK01-059` fizzle fix, the same checkpoint still routed `AZK01-128` effect selection through generic: env 2 destroyed the current attacking `AZK01-072` while `STT01-009` was in the target owner's garden.
- `step_effect_azk01_128_fast` destroys the low-HP attacker, clears context, and recomputes passives. `STT01-009` only depends on its own garden zone and weapon discard count; destroying a non-weapon attacker does not require generic observer ordering.
- Relaxed `_effect_azk01_128_fast_mask` to treat the same recompute-safe passive watchers as `_effect_azk01_127_fast_mask` (`STT01-009`, `STT01-011`, `AZK01-010`, `AZK01-073`, and existing inert watchers).
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1048_effects.py /tmp/jax_probe_1048.pkl` reports `_effect_stt02_001_fast_mask == [0, 1, 0, 0]` and `_effect_azk01_128_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1066 AZK01-092 places AZK01-021 over AZK01-073 watcher

- Re-probe from `/tmp/jax_probe_1048.pkl` advanced through steps 1048-1065 after several cached/uncached JIT compiles, then stopped at step 1066 on `SELECT_TO_ALLEY`.
- Row: env 3 `AZK01-092` selected revealed `AZK01-021` to alley slot 3 while the opponent had `AZK01-073` in garden.
- `step_select_stt02_013_pick_fast` already recomputes passives after board placement and leaves the selection bottom-deck flow active; opponent `AZK01-073` is recompute-safe and unrelated to the active player's placement.
- Relaxed `_select_stt02_013_pick_fast_mask` to treat `AZK01-073` as an allowed passive watcher for this recompute-backed placement path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1066_select.py /tmp/jax_probe_1066.pkl` reports `_select_stt02_013_pick_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1071 AZK01-065 effect over AZK01-019 watchers

- Re-probe from `/tmp/jax_probe_1066.pkl` advanced to step 1071 on `SELECT_EFFECT_TARGET`: env 0 resolved `AZK01-065` by self-damaging its leader and targeting friendly garden `AZK01-059`, while the opponent had two `AZK01-019` passive watchers.
- `step_effect_azk01_065_fast` recomputes passives after cost/effect damage, so `AZK01-019` watchers are deterministic on this path.
- The same row can queue a lethal `AZK01-059` takes-damage trigger. Updated the helper to pop that trigger if the source left `GARDEN`/`ALLEY` before converting surviving `AZK01-059` triggers to effect selection, matching the combat fizzle handling.
- Relaxed `_effect_azk01_065_fast_mask` to allow `AZK01-019` passive watchers for this recompute-backed path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1071_azk065.py /tmp/jax_probe_1071.pkl` reports `_effect_azk01_065_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1080 STT04-016 cost over AZK01-019 watchers

- Re-probe from `/tmp/jax_probe_1071.pkl` advanced to step 1080 on `SELECT_COST_TARGET`: env 0 selected friendly `STT04-003` as the `STT04-016` damage cost while the opponent had `AZK01-019` passive watchers.
- The cost helper applies one nonlethal effect damage to the selected friendly entity and then opens the optional effect-selection phase; the opponent `AZK01-019` watchers are unaffected by that cost transition.
- Relaxed `_select_cost_stt04_016_fast_mask` to allow `AZK01-019` passive watchers alongside the existing `STT01-008` exception.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1080_stt04016.py /tmp/jax_probe_1080.pkl` reports `_select_cost_stt04_016_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1081 STT04-016 effect into leader over AZK01-019 watchers

- Re-probe from `/tmp/jax_probe_1080.pkl` advanced one step to `STT04-016` optional effect selection: env 0 targeted the opponent leader while the opponent still had `AZK01-019` passive watchers.
- Leader-only damage does not change board composition, so passive watchers are irrelevant when the target is the leader or the action is the optional skip.
- Relaxed `_effect_stt04_016_fast_mask` so passive watchers do not block leader-target selections or skip actions; entity-target damage remains guarded by the existing passive-watch rejection.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1081_stt04016_effect.py /tmp/jax_probe_1081.pkl` reports `_effect_stt04_016_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1082 STT04-003 start-each with opponent AZK01-019

- Re-probe from `/tmp/jax_probe_1081.pkl` advanced one step to main-phase `NOOP`: active player 1 ended turn with one clean `STT04-003` start-each trigger, while only player 0 had `AZK01-019` passive watchers.
- `step_main_noop_stt04_003_fast` handles the ordered one/two-Seer start-each damage path. Opponent `AZK01-019` is unaffected when no `STT04-003` start-each source exists on that player.
- Relaxed `_main_noop_stt04_003_fast_mask` to ignore `AZK01-019` watchers only on players that do not own any admitted `STT04-003` start-each source; same-owner cases still fall back.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1082_noop.py /tmp/jax_probe_1082.pkl` reports `_main_noop_stt04_003_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1083 STT03-016 over stale AZK01-073 observer

- Re-probe from `/tmp/jax_probe_1082.pkl` advanced to step 1083 on `PLAY_SPELL_FROM_HAND`: env 2 played `STT03-016`, destroying the opponent's only HP<=2 garden entity.
- The split helper destroys marked entities in garden order and then recomputes passives. The row had a registered `AZK01-073` observer on a discarded card, not on a marked in-play target.
- Relaxed `_play_spell_stt03_016_fast_mask` so unmarked `AZK01-073` passive observers are treated as recompute-safe, matching the existing `AZK01-019` exception.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1083_stt03016.py /tmp/jax_probe_1083.pkl` reports `_play_spell_stt03_016_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1087 STT04-016 entity effect with recompute-safe watchers

- Re-probe from `/tmp/jax_probe_1083.pkl` advanced to step 1087 on `SELECT_EFFECT_TARGET`: env 0 resolved `STT04-016` into opponent garden `STT01-004` while `AZK01-019` watchers were on that opponent board; env 1 simultaneously had a covered `STT02-014` effect row.
- Updated `step_effect_stt04_016_fast` to recompute passives after applying/skipping the optional damage and clearing context, before beginning any supported queued trigger.
- Relaxed `_effect_stt04_016_fast_mask` to allow the same recompute-safe passive watcher set used by neighboring damage masks while still rejecting garden `STT02-012` watcher work.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1087_effects.py /tmp/jax_probe_1087.pkl` reports `_effect_stt04_016_fast_mask == [1, 0, 0, 0]` and `_effect_stt02_014_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1101 AZK01-120 gate portal into reequip selection

- Re-probe from `/tmp/jax_probe_1087.pkl` advanced to step 1101 on `GATE_PORTAL`: env 0 portaled `AZK01-069` through `AZK01-120` while a cost-1 attached weapon was eligible for Stormchain reequip; env 1's `STT02-002` portal was already covered.
- Added `AZK01-120` gate-portal support to `step_gate_portal_simple_fast`: after placement, it initializes the gate scratch/context, detaches eligible attached weapons into selection via the existing `AZK01-120` on-cost-paid helper, and leaves `SELECTION_PICK` active.
- Relaxed `_gate_portal_simple_fast_mask` to admit `AZK01-120` rows with reequip candidates instead of only no-candidate fizzles.
- Extended the shared select-to-equip and optional selection-skip masks to accept `AZK01-120` reequip selections using the portaled card's gate points and previous-host restriction.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1101_gate.py /tmp/jax_probe_1101.pkl` reports `_gate_portal_simple_fast_mask == [1, 1, 0, 0]`.

## 2026-06-19 — Step 1102 AZK01-120 reequip selection over pending passives

- Re-probe from `/tmp/jax_probe_1101.pkl` advanced one step to `SELECT_TO_EQUIP`: env 0 re-equipped the single `AZK01-094` selected by `AZK01-120`, with passive recompute work still queued from the gate portal board change.
- The shared `process_selection_to_equip` runtime already implements reequip-origin host exclusion, skips on-play counters for reequips, consumes `reequip_prev_host`, and clears the selection when the pick finishes.
- Extended `_select_stt01_002_equip_fast_mask` for `AZK01-120`: it uses the portaled card's gate points as the max weapon cost, enforces the previous-host restriction, and permits the pending passive queue left by the portal.
- Extended `_selection_pick_noop_fast_mask` to allow optional `AZK01-120` skip with the same pending passive queue shape.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the equip/noop mask update.
  - `/tmp/diagnose_step1102_equip.py /tmp/jax_probe_1102.pkl` reports `_select_stt01_002_equip_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1105 STT02-017 no-target fizzle

- Re-probe from `/tmp/jax_probe_1102.pkl` advanced to step 1105 on `PLAY_SPELL_FROM_HAND`: env 1 played `STT02-017` with a Shao leader, but the opponent had no garden entity to return.
- The helper already pays/discards the spell and loops over an empty marked set, matching the no-target fizzle behavior.
- Relaxed `_play_spell_stt02_017_fast_mask` to allow the zero-bounce case and to bypass passive-watch checks only when no return targets exist; nonzero return cases still require clean return targets and supported `when returned` observers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1105_stt02017.py /tmp/jax_probe_1105.pkl` reports `_play_spell_stt02_017_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1141 STT02-011 effect over deferred passive/STT02 work

- Re-probe from `/tmp/jax_probe_1105.pkl` advanced through step 1140, then stopped at step 1141 on `STT02-011` effect selection with `passive_queue_count=2` and pending `STT02-012` event bits from earlier board changes.
- The effect helper only grants `EffectImmune 2` to a friendly garden entity and clears the current ability context; it does not consume trigger, passive, or `STT02-012` event queues.
- Relaxed `_effect_stt02_011_fast_mask` to allow deferred passive/STT02 work for this context-local effect.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1141_stt02011.py /tmp/jax_probe_1141.pkl` reports `_effect_stt02_011_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1173 AZK01-024 cost over opponent STT01-009 watcher

- Re-probe from `/tmp/jax_probe_1141.pkl` advanced through step 1172, then stopped at step 1173 on `SELECT_COST_TARGET`: env 3 used `AZK01-024` with action `[13, 4, 0, 0]` while opponent `STT01-009` had a passive watcher registered.
- `step_select_cost_azk01_024_fast` returns one clean owner garden entity to hand, moves eligible hand entities to selection, and does not recompute passive queues in this cost phase. Opponent `STT01-009` depends on its own garden and discarded weapons, so this unrelated watcher should not block the cost fast path.
- Relaxed `_select_cost_azk01_024_fast_mask` to treat `STT01-009` like the existing inert watcher exceptions for this narrow cost path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1173_azk024.py /tmp/jax_probe_1173.pkl` reports `_select_cost_azk01_024_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1173 STT01-017 effect over inactive alley watcher and immune target

- After the `AZK01-024` cost fix, the same pre-step state exposed env 0: `STT01-017` effect selection in response window, action `[14, 2, 0, 0]`, targeting opponent garden `STT02-006`.
- The only passive watcher blocking the host mask was `STT02-012` in alley; its passive is garden-only and inactive in this state.
- The selected `STT02-006` had `effect_immune_dur = -1`. `deal_effect_damage` treats effect-immune targets as a no-op (`apply = false`) with no damage event, destroy, or trigger queue, so selecting it is safe for this fast path.
- Relaxed `_effect_stt01_017_fast_mask` to ignore alley `STT02-012` watchers and to allow effect-immune targets when the rest of the target remains clean.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1173_stt01017_reasons.py /tmp/jax_probe_1173.pkl` reports `_effect_stt01_017_fast_mask == [1, 1, 0, 0]`.

## 2026-06-19 — Step 1179 AZK01-024 selected AZK01-022 placement over opponent STT01-009

- Re-probe from `/tmp/jax_probe_1173.pkl` advanced to step 1179 on `SELECT_TO_GARDEN`: env 3 placed the single `AZK01-024` selected card (`AZK01-022`) into garden slot 0.
- `step_select_azk01_024_place_fast` uses `_enter_board_slot`, returns remaining selection cards, initializes `AZK01-022`'s optional follow-up confirmation when applicable, and then calls `recompute_passives`.
- The only blocking watcher was opponent `STT01-009`; because the helper recomputes passives after placement and the watcher belongs to the unchanged opponent board, it is safe to treat as recompute-supported for this mask.
- Relaxed `_select_azk01_024_place_fast_mask` to ignore `STT01-009` in the same narrow watcher set that already allowed `STT01-008`, `AZK01-019`, and `AZK01-073`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1179_azk024_place.py /tmp/jax_probe_1179.pkl` reports garden mask `[0, 0, 0, 1]`.

## 2026-06-19 — Step 1203 combat with passive alley-targeting weapon attached

- Re-probe from `/tmp/jax_probe_1179.pkl` advanced to step 1203 on `ATTACK`: env 1 attacked with garden `AZK01-038` slot 2 into opponent garden `STT01-008` slot 3.
- The attacker had attached `AZK01-043`. This weapon has no attack-declaration, damage, or destroy timing; its +ATK was already reflected in `cur_atk`, and the combat helper uses current stats.
- Existing `_attack_entity_mutual_destroy_fast_mask` rejected all attachments. Relaxed it only for attached `AZK01-043`/`AZK01-095`, leaving `STT01-012`, `AZK01-044`, combat-modifier weapons, and all unknown attached weapons on the generic path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1203_attack.py /tmp/jax_probe_1203.pkl` reports `_attack_entity_mutual_destroy_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1228 STT01-017 first target selection with passive watchers

- Re-probe from `/tmp/jax_probe_1203.pkl` advanced to step 1228 on `STT01-017` effect selection: env 1 chose the first target while `STT01-009`/`STT01-011` style passive watchers were present on board.
- For `STT01-017`, selecting the first target only records ability context; damage is delayed until the second target is selected or the optional selection is skipped. Passive watchers should not block a context-only first pick.
- Refined `_effect_stt01_017_fast_mask`: passive watcher cleanliness is still required for finishing actions that can change the board, but first-target selection and finishing selections whose recorded targets take no damage/leave no play may proceed.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1173_stt01017_reasons.py /tmp/jax_probe_1228.pkl` reports `_effect_stt01_017_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1229 STT01-002 confirm with deferred passive/STT02 work

- Re-probe from `/tmp/jax_probe_1228.pkl` advanced one step to `CONFIRM_ABILITY`: env 0 confirmed `STT01-002` after a gate portal while `passive_queue_count=1` and `STT02-012` pending bits were still deferred.
- `step_confirm_stt01_002_fast` only moves eligible discard-pile weapons into selection and updates the ability scratch/context. It does not inspect or consume passive queues.
- Relaxed `_confirm_stt01_002_fast_mask` to permit deferred passive/STT02 work, matching the neighboring selection masks that preserve pending passive work through context-only steps.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1229_stt01002.py /tmp/jax_probe_1229.pkl` reports `_confirm_stt01_002_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1230 STT01-002 equip with deferred passive/STT02 work

- Re-probe from `/tmp/jax_probe_1229.pkl` advanced to `SELECT_TO_EQUIP`: env 0 equipped an `STT01-002`-selected discard weapon while the same deferred passive/STT02 work from the gate portal remained pending.
- `step_select_stt01_002_equip_fast` delegates to `process_selection_to_equip` and preserves existing passive queues; this step is not the consumer of queued passive/STT02 work.
- Relaxed `_select_stt01_002_equip_fast_mask` and the shared optional selection skip mask so `STT01-002` behaves like the already-supported `AZK01-120` reequip path when passive/STT02 work is deferred.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1230_stt01002_equip.py /tmp/jax_probe_1230.pkl` reports `_select_stt01_002_equip_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1237 STT01-012 leader attack with deferred passive death queue

- Re-probe from `/tmp/jax_probe_1230.pkl` advanced to step 1237 with two attack rows; env 3 was already covered by `_attack_entity_response_fast_mask`, while env 1 was leader `STT01-001` with attached `STT01-012` attacking opponent garden `AZK01-037`.
- The STT01-012 attack helper mills, phase-gates, and auto-resolves combat when no response/trigger work remains. `combat_resolve` queues passive zone-change work when the defender dies; that deferred queue is valid and later consumed by the passive main-NOOP path.
- Relaxed `_attack_stt01_012_response_fast_mask` to allow clean defender death without attachments even when passive observers are present, and to use the same supported passive watcher allowlist used by neighboring combat paths.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1237_attacks.py /tmp/jax_probe_1237.pkl` reports `_attack_stt01_012_response_fast_mask == [0, 1, 0, 0]` and `_attack_entity_response_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1238 STT01-005 alley activation over supported passive watchers

- Re-probe from `/tmp/jax_probe_1237.pkl` advanced to step 1238 on `ACTIVATE_ALLEY_ABILITY`: env 1 sacrificed alley `STT01-005`, drew 3, and entered its discard-2 effect prompt.
- Existing `_activate_stt01_005_fast_mask` only tolerated `STT01-008` passive watchers. This activation can queue/defer passive zone-change work from the alley sacrifice, so the same supported watcher allowlist used by combat paths is appropriate.
- Relaxed `_activate_stt01_005_fast_mask` to allow supported passive watchers (`STT01-009`, `STT01-011`, `STT02-010`, `AZK01-010`, `AZK01-019`, `AZK01-073`, `STT02-012`, `STT03-013`) while leaving unknown watchers on the generic path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1238_stt01005.py /tmp/jax_probe_1238.pkl` reports `_activate_stt01_005_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1239 STT01-005 variable discard count

- Re-probe from `/tmp/jax_probe_1238.pkl` advanced to `STT01-005` effect selection: env 1 had `ab_eff_min=ab_eff_max=1`, because the C-compatible activation path uses the pre-cost hand count for the discard prompt.
- `step_activate_stt01_005_fast` already computed the variable discard count, but `step_effect_stt01_005_fast` and `_effect_stt01_005_fast_mask` still required exactly 2 selections.
- Relaxed the STT01-005 effect helper and host mask to accept `ab_eff_min == ab_eff_max` with a value of 1 or 2; the existing finish logic already discards only the selected target(s).
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1239_stt01005_effect.py /tmp/jax_probe_1239.pkl` reports `_effect_stt01_005_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1243 STT01-016 when-attacking leader hit

- Re-probe from `/tmp/jax_probe_1239.pkl` advanced to step 1243: env 1 attacked the opposing leader with garden `STT01-011` carrying attached `STT01-016`.
- `STT01-016` is a valid When Attacking trigger when attached to a Raizan card. It deals 1 effect damage to all opponent garden entities before the leader combat damage.
- Extended `step_attack_leader_simple_fast` to inline this clean STT01-016 trigger before leader damage, using `deal_effect_damage(..., src_player=attacker_owner, src_inst=weapon)` in garden order.
- Relaxed `_attack_leader_simple_fast_mask` only for STT01-016 rows where the attacker is Raizan, the weapon has at least one valid garden target, and every affected garden target has no modeled damage/destroy complication.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1243_leader_reasons.py /tmp/jax_probe_1243.pkl` reports `_attack_leader_simple_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1246 STT01-006 attack declaration over supported watcher

- Re-probe from `/tmp/jax_probe_1243.pkl` advanced to step 1246: env 0 attacked the opposing leader with garden `STT01-006` while opponent `STT02-012` sat in garden.
- The STT01-006 declaration helper only taps/records combat and opens the mandatory damage target prompt. No effect damage is applied until the next `SELECT_EFFECT_TARGET` action, so supported passive watchers should not block declaration itself.
- Relaxed `_attack_stt01_006_effect_fast_mask` to use the same supported passive watcher allowlist as neighboring combat/action paths (`STT01-009`, `STT01-011`, `STT02-010`, `AZK01-010`, `AZK01-019`, `AZK01-073`, `STT02-012`, `STT03-013`).
- Latest validation:
  - `/tmp/diagnose_step1246_stt01006.py /tmp/jax_probe_1246.pkl` reports `_attack_stt01_006_effect_fast_mask == [1, 0, 0, 0]` and `_attack_leader_response_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1253 main-phase attach with deferred STT02-012 work

- Re-probe from `/tmp/jax_probe_1247.pkl` advanced to step 1253: env 0 attached hand `AZK01-094` to leader `STT01-001` while `passive_queue_count=1` and `STT02-012` event bits were deferred from the preceding `STT01-006` leader hit.
- `step_attach_weapon_simple_fast` pays/attaches and then calls `recompute_passives`, which drains pending passive/STT02 work when no ability context is active. Main-phase attach can therefore carry pending STT02 work into the helper.
- Relaxed `_attach_weapon_simple_fast_mask` to allow pending `STT02-012` event bits for main-phase attach rows only; response-window attaches remain conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1253_attach.py /tmp/jax_probe_1253.pkl` reports `_attach_weapon_simple_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1261 STT02-003 reveal play over deferred passive work

- Re-probe from `/tmp/jax_probe_1253.pkl` advanced to step 1261: env 0 played `STT02-003` to garden while `passive_queue_count=1` and `STT02-012` event bits were still deferred.
- The STT02-003 play helper always enters reveal selection or bottom-deck handling when admitted (`deck_count > 0`), so `apply_user_action_static` would also leave passive recompute gated by the new ability context. Preserving the pending passive/STT02 bookkeeping is the parity behavior.
- Relaxed `_play_stt02_003_reveal_fast_mask` to preserve deferred passive/STT02 work and to tolerate the same supported passive watchers used by the neighboring play/combat masks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1261_play.py /tmp/jax_probe_1261.pkl` reports `_play_stt02_003_reveal_fast_mask == [1, 0, 0, 0]`; env 1/3 are covered by existing static/simple play masks.

## 2026-06-19 — Step 1262 STT02-003 selection pick over deferred passives

- Re-probe from `/tmp/jax_probe_1261.pkl` advanced one step to `SELECT_FROM_SELECTION`: env 0 picked `STT02-009` from the STT02-003 reveal while deferred passive/STT02 work remained.
- `step_select_stt02_003_pick_fast` moves the picked Watercrafting card to hand and, when remaining reveal cards exist, stays in `BOTTOM_DECK`; generic static action also leaves recompute gated by that ability context.
- Relaxed `_select_stt02_003_pick_fast_mask` to allow deferred passive/STT02 work only when the pick leaves remaining selection cards (`sel_count > 1`). Single-card picks with deferred passive work stay generic until a recompute-safe finish path is modeled.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1262_select.py /tmp/jax_probe_1262.pkl` reports `_select_stt02_003_pick_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1263 STT02-003 bottom-deck over deferred passives

- Re-probe from `/tmp/jax_probe_1262.pkl` advanced to `BOTTOM_DECK_CARD`: env 0 bottom-decked one remaining STT02-003 reveal card while passive/STT02 work was still deferred.
- Bottom-decking one card from a multi-card reveal keeps the ability context alive, so generic static action also leaves recompute gated. Clearing the final card with deferred passive work remains generic.
- Relaxed the shared bottom-deck mask only for `STT02-003` single-card bottom-deck actions where more than one live selection card remains and no trigger/redirect/combat work is present.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1262_select.py /tmp/jax_probe_1263.pkl` reports bottom-deck-card mask `[1, 0, 0, 0]`.

## 2026-06-19 — Step 1265 STT02-003 bottom-deck-all drains deferred passives

- Re-probe from `/tmp/jax_probe_1263.pkl` advanced to `BOTTOM_DECK_ALL`: env 0 bottom-decked all remaining STT02-003 reveal cards while deferred passive/STT02 work was still present.
- `step_bottom_deck_all_fast` clears the selection context, then runs `recompute_passives` and begins any resulting trigger. That is the correct consumer for this deferred queue.
- Relaxed the shared bottom-deck mask for `STT02-003` bottom-deck actions with no trigger already queued; single-card and all-card bottom-deck paths can now either preserve or drain the deferred passive work according to the helper's existing `done_selection` branch.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1262_select.py /tmp/jax_probe_1265.pkl` reports bottom-deck-all mask `[1, 0, 0, 0]`.

## 2026-06-19 — Step 1271 STT01-006 attack declaration with deferred passives

- Re-probe from `/tmp/jax_probe_1265.pkl` advanced to step 1271: env 0 attacked the opposing leader with `STT01-006` while deferred passive/STT02 work remained after the STT02-003 reveal cleanup.
- The STT01-006 declaration helper only records combat and opens mandatory effect selection. Generic static action would also leave passive recompute gated by the new ability context, so the pending queue must be preserved rather than rejected.
- Relaxed `_attack_stt01_006_effect_fast_mask` to allow existing passive/STT02 bookkeeping at declaration time; unsupported board watchers are still filtered by the supported watcher allowlist.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1246_stt01006.py /tmp/jax_probe_1271.pkl` reports `_attack_stt01_006_effect_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1272 STT01-006 effect consumes deferred passives

- Re-probe from `/tmp/jax_probe_1271.pkl` advanced to step 1272: env 0 selected the opposing leader for `STT01-006` while the same deferred passive/STT02 work was present.
- Generic static action clears the ability context after the effect and then runs passive recompute before exposing the next decision. The fast helper was missing that recompute and the host mask rejected the pending queue.
- Added `recompute_passives` to `step_effect_stt01_006_fast` after the effect/phase-gate/combat branch and relaxed `_effect_stt01_006_fast_mask` to allow existing deferred passive/STT02 work for this finishing action.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1246_stt01006.py /tmp/jax_probe_1272.pkl` reports `_effect_stt01_006_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1276 STT01-006 attached stat weapon attack

- Re-probe from `/tmp/jax_probe_1272.pkl` advanced to step 1276: env 0 attacked a leader with `STT01-006` carrying attached `AZK01-094`.
- `AZK01-094` has no attack/after-attack timing; its stats were already folded into the host. The declaration helper only needs to reject attached cards with attack timing or `AZK01-044`-style shock handling.
- Relaxed `_attack_stt01_006_effect_fast_mask` to allow safe non-triggering attachments while continuing to reject unsafe attached attack triggers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1246_stt01006.py /tmp/jax_probe_1276.pkl` reports `_attack_stt01_006_effect_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1282 STT02-013 selection over STT01-009 watcher

- Re-probe from `/tmp/jax_probe_1276.pkl` advanced to step 1282: env 3 picked `STT02-008` from an `STT02-013` reveal while opponent `STT01-009` was on board.
- This `SELECT_FROM_SELECTION` branch only moves the revealed card to hand and then enters bottom-deck cleanup; it does not change any board zone. `STT01-009` should not block the selection fast path.
- Relaxed `_select_stt02_013_pick_fast_mask` to include the same supported passive watcher exceptions used by neighboring play masks, adding `STT01-009`, `STT02-010`, and `AZK01-010`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1282_stt02013.py /tmp/jax_probe_1282.pkl` reports `_select_stt02_013_pick_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1322 STT01-001 activation over supported watchers

- Re-probe from `/tmp/jax_probe_1282.pkl` advanced to step 1322: env 0 activated leader `STT01-001` while opponent `STT02-012` alley watchers were present.
- The activation only pays IKZ and opens the charge target selection; it does not resolve passive watcher work at activation time.
- Relaxed `_activate_stt01_001_fast_mask` to use the supported passive watcher allowlist already used by adjacent activation/combat paths.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1322_activate.py /tmp/jax_probe_1322.pkl` reports `_activate_stt01_001_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1354 AZK01-072 attack trigger fizzle into response

- Re-probe from `/tmp/jax_probe_1322.pkl` advanced to step 1354: env 3 attacked a leader with `AZK01-072` while no other friendly Beanz entity existed for its When Attacking target.
- The effect is not available, so the attack should proceed as a normal leader attack while preserving the normal response window.
- Relaxed `_attack_leader_response_fast_mask` to treat `AZK01-014`/`AZK01-072` as fizzling when their friendly effect target does not exist. The existing effect-opening path still handles rows with a valid target.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1354_attack.py /tmp/jax_probe_1354.pkl` reports `_attack_leader_response_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1381 zero-damage AZK01-062 combat response pass

- Re-probe from `/tmp/jax_probe_1354.pkl` advanced to response-window `NOOP`: env 0 had `AZK01-062` and `STT02-006` both at 0 ATK in entity combat.
- The existing zero-damage stalemate path still rejected printed takes/deals-damage timing even though no damage event occurs when both damage amounts are zero.
- Refined `_response_noop_entity_combat_fast_mask` so takes/deals-damage timing only blocks when the corresponding combat damage amount is positive.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1381_response.py /tmp/jax_probe_1381.pkl` reports `_response_noop_entity_combat_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1385 AZK01-105 activation over STT03-013 watcher

- Re-probe from `/tmp/jax_probe_1381.pkl` advanced to step 1385: env 2 activated garden `AZK01-105` while own `STT03-013` sat in alley.
- The activation sacrifices the source and opens damage target selection; generic static action leaves passive recompute gated by that ability context. `STT03-013` is already treated as inert/supported in neighboring masks.
- Relaxed `_activate_azk01_105_fast_mask` to use the supported passive watcher allowlist for this deferred context.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1385_activate.py /tmp/jax_probe_1385.pkl` reports `_activate_azk01_105_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1386 AZK01-105 effect over STT03-013 watcher

- Re-probe from `/tmp/jax_probe_1385.pkl` advanced to step 1386: env 2 selected the opposing leader for `AZK01-105`'s stored damage while own `STT03-013` remained in alley.
- `step_effect_azk01_105_fast` applies the damage, clears the ability context, then runs `recompute_passives`; this is the correct consumer for supported passive watcher work after target selection.
- Relaxed `_effect_azk01_105_fast_mask` to use the same supported passive watcher allowlist as `_activate_azk01_105_fast_mask`, including `STT03-013`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1386_azk01105.py /tmp/jax_probe_1386.pkl` reports `_effect_azk01_105_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1394 STT04-016 effect into effect-immune target

- Re-probe from `/tmp/jax_probe_1386.pkl` advanced to step 1394: env 0 selected opponent garden slot 0 `STT02-006` for `STT04-016` while an `AZK01-059` takes-damage trigger from the paid cost was already queued.
- `STT02-006` has permanent effect immunity. `deal_effect_damage` treats the selected damage as a no-op, so there is no new damage/destroy trigger to model; the existing helper already clears the STT04-016 context, recomputes passives, then begins the queued `AZK01-059` trigger.
- Relaxed `_effect_stt04_016_fast_mask` so effect-immune targets are allowed when the rest of the target remains clean; non-immune damage/destroy/timing cases remain conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1394_stt0416.py /tmp/jax_probe_1394.pkl` reports `_effect_stt04_016_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1400 AZK01-058 attack into response window

- Re-probe from `/tmp/jax_probe_1394.pkl` advanced to step 1400: env 0 declared `AZK01-058` attacking the opposing leader while `AZK01-125` supplied a board response action; env 2's simultaneous entity attack was already covered by `_attack_entity_mutual_destroy_fast_mask`.
- `_attack_leader_response_fast_mask` rejected all after-attacking sources, while the no-response leader helper already modeled `AZK01-058`'s post-damage optional confirmation.
- Relaxed the response declaration mask for `AZK01-058` after-attacking and extended `step_response_noop_leader_combat_fast` plus its host mask so a response-pass combat resolution opens the same `AZK01-058` confirmation after damage.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1354_attack.py /tmp/jax_probe_1400.pkl` reports `_attack_leader_response_fast_mask == [1, 0, 0, 0]` and `_attack_entity_mutual_destroy_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1406 STT02-013 places STT02-003 to alley

- Re-probe from `/tmp/jax_probe_1400.pkl` advanced to step 1406: env 0 used `STT02-013` selection action `[21, 2, 2, 0]` to place revealed `STT02-003` into alley slot 2.
- The selection helper already modeled STT02-013 hand/board picks, but its host mask blocked `STT02-003` because that card has an on-play reveal trigger. Generic selection-to-alley queues the placed card's on-play trigger and resolves it after the remaining STT02-013 selection cards are bottom-decked.
- Extended `step_select_stt02_013_pick_fast` to queue `STT02-003`'s on-play trigger for the alley placement path instead of treating it as a simple implemented play trigger; relaxed the host mask for that exact `STT02-003` to-alley case.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1406_stt02013_components.py /tmp/jax_probe_1406.pkl` reports `_select_stt02_013_pick_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1407 bottom-deck after queued STT02-003 on-play

- Re-probe from `/tmp/jax_probe_1406.pkl` advanced one step to `BOTTOM_DECK_ALL`: env 0 was finishing the remaining `STT02-013` reveal cards while the just-placed `STT02-003` on-play trigger was queued.
- `step_bottom_deck_all_fast` already processes bottom-deck cleanup, recomputes passives, then pops and begins a queued trigger when selection cleanup finishes.
- Relaxed `_bottom_deck_azk01_003_fast_mask` for the exact `STT02-013` bottom-deck cleanup shape with one queued `STT02-003` `TIMING_ON_PLAY` trigger and no passive/STT02 deferred work.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1282_stt02013.py /tmp/jax_probe_1407.pkl` reports bottom-deck-all mask `[1, 1, 0, 0]`.

## 2026-06-19 — Step 1419 AZK01-060 optional decline before response

- Re-probe from `/tmp/jax_probe_1407.pkl` advanced to step 1419: env 0 declined `AZK01-060`'s optional when-attacking confirmation while the attack still needed to enter the opposing `AZK01-125` response window; env 3's simultaneous response-window NOOP was already covered.
- `_confirm_azk01_060_response_fast_mask` only admitted decline when a defender intercept was available, and only admitted non-defender response windows on confirm. Declining the optional buff still clears the ability context and lets `phase_gate` open the same response window.
- Relaxed `_confirm_azk01_060_response_fast_mask` so decline is valid when either defender declaration or a non-defender response is available.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1419_azk01060.py /tmp/jax_probe_1419.pkl` reports `_confirm_azk01_060_response_fast_mask == [1, 0, 0, 0]` and response NOOP masks cover env 3.

## 2026-06-19 — Step 1452 AZK01-127 into AZK01-059 trigger

- Re-probe from `/tmp/jax_probe_1419.pkl` advanced to step 1452: env 0 resolved response spell `AZK01-127` into opponent garden slot 0 `AZK01-059` at 2 HP.
- The target takes nonlethal damage and queues `AZK01-059`'s once-per-turn takes-damage trigger. `step_effect_azk01_127_fast` already handles response-window cleanup after direct damage, but did not begin this queued trigger.
- Extended `step_effect_azk01_127_fast` to pop/begin a live `AZK01-059` takes-damage trigger after the response damage, and relaxed `_effect_azk01_127_fast_mask` for that nonlethal, unspent trigger shape with another friendly garden entity available.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1452_azk01127.py /tmp/jax_probe_1452.pkl` reports `_effect_azk01_127_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1453 AZK01-059 trigger during response combat

- Re-probe from `/tmp/jax_probe_1452.pkl` advanced to step 1453: env 0 selected friendly garden slot 4 for the `AZK01-059` trigger that was caused by `AZK01-127` response damage; env 2's simultaneous `STT03-002` effect row was already covered.
- `step_effect_azk01_059_fast` applies the attack buff, marks the once-per-turn bit, clears context, and restores the saved active player. In a response-trigger context, it should preserve the pending combat and response phase for the next response/combat action.
- Relaxed `_effect_azk01_059_fast_mask` to allow clean response-window combat to remain pending while the trigger resolves.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1453_effects.py /tmp/jax_probe_1453.pkl` reports `_effect_azk01_059_fast_mask == [1, 0, 0, 0]` and `_effect_stt03_002_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1466 AZK01-024 places simple on-play entity

- Re-probe from `/tmp/jax_probe_1453.pkl` advanced to step 1466: env 0 used `AZK01-024` selection action `[21, 0, 2, 0]` to place selected `STT02-007` into alley.
- The AZK01-024 placement helper moved the selected entity and cleared the selection context, but did not run simple implemented on-play effects for selected entities. `STT02-007`'s draw is already supported by `_apply_simple_implemented_play_trigger`.
- Extended `step_select_azk01_024_place_fast` to apply simple implemented on-play triggers after placement, and relaxed `_select_azk01_024_place_fast_mask` for `self._simple_play_implemented_ids`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1466_azk01024.py /tmp/jax_probe_1466.pkl` reports `_select_azk01_024_place_fast_mask` alley mask `[1, 0, 0, 0]`.

## 2026-06-19 — Step 1487 STT04-016 cost damages AZK01-059 with no trigger target

- Re-probe from `/tmp/jax_probe_1466.pkl` advanced to step 1487: env 0 had only one friendly garden entity, `AZK01-059`, and selected it for `STT04-016`'s 1-damage cost.
- `AZK01-059` takes damage but has no other friendly garden entity for its trigger target. The cost path should fizzle that queued takes-damage trigger immediately and continue into `STT04-016` effect selection.
- Extended `step_select_cost_stt04_016_fast` to pop the just-queued `AZK01-059` trigger when no valid other garden target exists, and relaxed `_select_cost_stt04_016_fast_mask` for that no-target fizzle shape.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1487_stt0416_cost.py /tmp/jax_probe_1487.pkl` reports `_select_cost_stt04_016_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1506 STT04-004 into AZK01-062 redirect

- Re-probe from `/tmp/jax_probe_1487.pkl` advanced to step 1506: env 2 selected friendly garden slot 4 `AZK01-062` for `STT04-004`'s sacrifice-and-ping effect.
- `deal_effect_damage` defers damage into `AZK01-062`'s redirect queue. The STT04-004 helper previously cleared context without beginning that queued redirect trigger.
- Extended `step_effect_stt04_004_fast` to pop/begin the queued `AZK01-062` takes-damage trigger after context clear, and relaxed `_effect_stt04_004_fast_mask` for a clean unspent `AZK01-062` redirect target.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1506_stt04004.py /tmp/jax_probe_1506.pkl` reports `_effect_stt04_004_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1534 AZK01-044 clean leader attack

- Re-probe from `/tmp/jax_probe_1506.pkl` advanced to step 1534: env 0 attacked the opposing leader with garden `STT01-008` carrying attached `AZK01-044`.
- `AZK01-044` has no attack-declaration trigger; its Lightning Kanabo shock is hardcoded inside `combat_resolve`. `_attack_leader_garden_simple_fast_mask` already routes through `phase_gate` and `combat_resolve`, so its host mask should not reject attached `AZK01-044`.
- Removed the special `AZK01-044` rejection from `_attack_leader_garden_simple_fast_mask` while keeping attached when-attacking trigger rejection intact.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1354_attack.py /tmp/jax_probe_1534.pkl` still shows the older named masks false, and a direct `_attack_leader_garden_simple_fast_mask` probe reports `[1, 0, 0, 0]`.

## 2026-06-19 — Step 1535 STT04-003 start-each with shock cleanup

- After admitting the `AZK01-044` attack, the next frontier was step 1535: env 0 chose MAIN `NOOP` to end player 1's turn while player 0 leader carried `shocked_dur=1` from Lightning Kanabo and player 0 had two `STT04-003` start-of-each-turn watchers.
- `_main_noop_stt04_003_fast_mask` rejected any shocked card even though `step_main_noop_stt04_003_fast` can safely cover this shape once `_simple_start_turn_no_triggers` mirrors start-turn status/untap semantics.
- Updated `_simple_start_turn_no_triggers` to use `_tick_start_statuses` for both players and `_untap_all_for` for the new active player, then relaxed the STT04-003 host mask to keep rejecting frozen/effect-immune status but allow shock cleanup.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - Direct probe on `/tmp/jax_probe_1535.pkl` reports `_main_noop_stt04_003_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1545 AZK01-127 lethal damage to equipped entity

- Re-probe from `/tmp/jax_probe_1535.pkl` advanced to step 1545: env 1 resolved response spell `AZK01-127` against opponent garden slot 4 `STT01-004` at 1 HP with attached `AZK01-094`.
- C `deal_effect_damage_from_source_internal` kills the entity via `discard_card` and does not call `discard_equipped_weapon_cards`; the JAX `deal_effect_damage` helper mirrors that, so an attached weapon is not an unsafe blocker for this direct-damage fast path.
- Relaxed `_effect_azk01_127_fast_mask` so direct damage may target an entity with attached cards when all cards attached to that target are weapons. The existing `AZK01-059` trigger branch remains conservative and still rejects attached targets.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1452_azk01127.py /tmp/jax_probe_1545.pkl` reports `_effect_azk01_127_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1557 AZK01-058 after-attacking after entity combat

- Re-probe from `/tmp/jax_probe_1545.pkl` advanced to step 1557: env 3 passed the response window for `AZK01-058` attacking garden `STT02-006`.
- The existing entity-combat response-pass helper handled clean entity combat but only treated a dead `AZK01-058` after-attacking trigger as a fizzle; it did not begin the optional `AZK01-058` confirmation when the attacker survived.
- Extended `step_response_noop_entity_combat_fast` with the same `AZK01-058` confirmation setup used by the leader-combat response-pass helper, and relaxed `_response_noop_entity_combat_fast_mask` to allow that supported after-attacking branch.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1381_response.py /tmp/jax_probe_1557.pkl` reports `_response_noop_entity_combat_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1559 AZK01-120 re-equipping an on-play weapon

- Re-probe from `/tmp/jax_probe_1557.pkl` advanced to step 1559: env 0 selected `STT01-014` from `AZK01-120`'s re-equip selection and attached it to a different host.
- C `azk_process_selection_to_equip` skips on-play triggers for `selection_to_equip_is_reequip` flows and only queues when-equipped triggers. The existing mask rejected `STT01-014` because it has an on-play ability, and the shared helper only admitted `STT01-002` as the source.
- Updated `step_select_stt01_002_equip_fast` to also admit the `AZK01-120` scratch protocol, and relaxed `_select_stt01_002_equip_fast_mask` so `AZK01-120` re-equips ignore selected-weapon on-play timing while still rejecting unsupported when-equipped triggers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - Direct probe on `/tmp/jax_probe_1559.pkl` reports `_select_stt01_002_equip_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 1566 AZK01-098 confirmation into hand-weapon selection

- Re-probe from `/tmp/jax_probe_1559.pkl` advanced to step 1566: env 1 confirmed `AZK01-098`'s optional alley on-play ability with a cost-<=3 weapon in hand.
- `AZK01-098` shares the confirmation-to-selection shape with `STT01-002`, but pays by tapping itself and moves eligible weapons from hand instead of discard.
- Generalized `step_confirm_stt01_002_fast` and its mask to admit `AZK01-098`: confirmation taps the source, marks costs applied, moves hand weapons with IKZ cost <=3 to selection, and leaves the existing discard-portal scratch behavior unchanged for `STT01-002`.
- Also extended the shared selection-to-equip helper/mask to admit `AZK01-098` after costs are applied; it remains conservative for unsupported selected-weapon on-play timing.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - Direct probe on `/tmp/jax_probe_1566.pkl` reports `_confirm_stt01_002_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1575 STT04-003 start-each with tapped AZK01-011 EOT

- Re-probe from `/tmp/jax_probe_1566.pkl` advanced to step 1575: env 2 ended player 1's turn with tapped garden `AZK01-011` and an opponent `STT04-003` start-of-each-turn watcher.
- A tapped `AZK01-011` has the EOT timing tag but its effect is a no-op. `step_main_noop_stt04_003_fast` can safely skip that no-op EOT ability and still apply the clean STT04-003 start-each damage.
- Relaxed `_main_noop_stt04_003_fast_mask` so tapped garden `AZK01-011` does not block the STT04-003 main-NOOP fast path; untapped `AZK01-011` remains excluded for the dedicated EOT path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - Direct probe on `/tmp/jax_probe_1575.pkl` reports `_main_noop_stt04_003_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1581 AZK01-062 attacker redirect fizzle in entity combat

- Re-probe from `/tmp/jax_probe_1575.pkl` advanced to step 1581: env 3 passed response for `AZK01-062` attacking garden `STT02-013`; the attacker takes lethal combat damage.
- The entity-combat response-pass helper already popped a clean `AZK01-062` takes-damage redirect fizzle when the defender was AZK01-062, but not when the attacker was AZK01-062.
- Extended `step_response_noop_entity_combat_fast` and `_response_noop_entity_combat_fast_mask` to pop the same redirect fizzle for an attacking `AZK01-062` that takes combat damage, while keeping other damage/death triggers conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1381_response.py /tmp/jax_probe_1581.pkl` reports `_response_noop_entity_combat_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1597 AZK01-065 targeting AZK01-062 redirect

- Re-probe from `/tmp/jax_probe_1581.pkl` advanced to step 1597: env 2 resolved `AZK01-065` and selected friendly garden `AZK01-062` as the 5-damage target.
- `deal_effect_damage` defers that damage into `AZK01-062`'s redirect queue before immunity/carapace/death logic. The AZK01-065 helper handled AZK01-059 follow-up triggers but did not begin the queued AZK01-062 redirect.
- Extended `step_effect_azk01_065_fast` to pop/begin a queued `AZK01-062` redirect trigger after context clear, and relaxed `_effect_azk01_065_fast_mask` for a clean unspent AZK01-062 target.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - Direct probe on `/tmp/jax_probe_1597.pkl` reports `_effect_azk01_065_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1599 zero-attack into AZK01-040

- Re-probe from `/tmp/jax_probe_1597.pkl` advanced to step 1599: env 1 attacked tapped garden `AZK01-040` with garden `AZK01-103` at 0 attack.
- C queues `AZK01-040`'s `[When Attacked]` trigger during the combat-resolve transition even if the attacker deals 0 combat damage. The JAX fast masks required positive defender damage before admitting either the clean attack declaration or AZK01-040's effect-selection cleanup.
- Relaxed the clean attack mask only for `AZK01-040`'s supported when-attacked trigger, kept real defender responses separate from that trigger, and allowed `step_effect_azk01_040_fast`'s mask to resolve zero-damage defender combat after skip/select.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask updates.
  - `/tmp/diag_1599_attack_components.py /tmp/jax_probe_1599.pkl` reports `_attack_leader_garden_simple_fast_mask == [0, 1, 0, 0]` and `_attack_entity_response_fast_mask == [0, 0, 0, 0]` for the zero-attack AZK01-040 case.

## 2026-06-19 — Step 1667 lethal AZK01-059 during response combat

- Re-probe from `/tmp/jax_probe_1599.pkl` advanced to step 1667: env 3 passed the response window for attacking `AZK01-059` into tapped garden `STT02-013`; the attacker died to return combat damage while another friendly garden entity remained.
- The generic path records the `AZK01-059` takes-damage trigger before discarding the source, clears combat, then opens `AZK01-059` effect selection from discard. The response-combat helper previously only handled `AZK01-062` fizzle and `AZK01-058` after-attacking follow-up.
- Extended `step_response_noop_entity_combat_fast` to pop/begin a clean queued `AZK01-059` trigger after response combat, and relaxed `_response_noop_entity_combat_fast_mask` for attacker/defender `AZK01-059` trigger shapes with an available other garden target.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1381_response.py /tmp/jax_probe_1667.pkl` reports `_response_noop_entity_combat_fast_mask == [0, 0, 0, 1]`.
  - `/tmp/inspect_after_1667_generic.py /tmp/jax_probe_1667.pkl` matches the generic outcome for env 3: `active=1`, `phase=MAIN`, `ab=EFFECT_SELECTION`, `ab_source=13:AZK01-059`, and `trig_count=0`.

## 2026-06-19 — Step 1776 AZK01-092 optional reveal decline

- Re-probe from `/tmp/jax_probe_1743.pkl` advanced to step 1776: env 2 was in `AZK01-092` reveal `SELECTION_PICK` with five revealed cards and chose `NOOP` to decline the optional pick.
- The shared `step_selection_pick_noop_fast` already uses `process_skip_selection`, which is the correct runtime path for optional reveal declines. Its host mask omitted `AZK01-092` from the supported source list, so the row fell through to generic despite matching the existing helper shape.
- Added `AZK01-092` to `_selection_pick_noop_fast_mask`'s supported sources. The simultaneous env 3 `NOOP` row was already covered by `_main_noop_azk01_011_fast_mask`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1776_noop.py /tmp/jax_probe_1776.pkl` reports `_selection_pick_noop_fast_mask == [0, 0, 1, 0]` and `_main_noop_azk01_011_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1795 STT04-009 combat-damage fizzle during response combat

- Re-probe from `/tmp/jax_probe_1776.pkl` advanced to step 1795: env 2 passed a response window where attacking garden `STT04-009` took non-effect combat damage from tapped `AZK01-022`.
- `STT04-009` only resolves from effect damage. The no-response entity-combat helper already records combat damage and had `AZK01-062` fizzle support, but did not pop `STT04-009`'s combat-damage takes-damage trigger.
- Added the same `STT04-009` combat fizzle used by the direct entity-combat helper to `step_response_noop_entity_combat_fast`, and relaxed `_response_noop_entity_combat_fast_mask` for attacker/defender `STT04-009` combat-damage fizzle shapes. The simultaneous env 3 STT03-002 skip was already covered by `_effect_stt03_002_fast_mask`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1795.py /tmp/jax_probe_1795.pkl` reports `_response_noop_entity_combat_fast_mask == [0, 0, 1, 0]` and `_effect_stt03_002_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1795 STT01-013 attach with STT01-009 watcher

- After the STT04-009 response-combat fix, the same checkpoint exposed env 1 attaching `STT01-013` to the leader while friendly `STT01-009` was in garden.
- `step_attach_stt01_013_confirm_fast` already recomputes passives and opens the dagger confirmation. Its dedicated host mask was more conservative than the generic attach mask and still treated `STT01-009`/`STT01-011` as unsafe attach watchers.
- Relaxed `_attach_stt01_013_confirm_fast_mask` to treat `STT01-009` and `STT01-011` as recompute-safe passive watchers, matching `_attach_weapon_simple_fast_mask`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1795.py /tmp/jax_probe_1795.pkl` reports `_attach_stt01_013_confirm_fast_mask == [0, 1, 0, 0]` alongside the step's response/effect masks.

## 2026-06-19 — Step 1812 AZK01-054 surviving godmode response combat

- Re-probe from `/tmp/jax_probe_1795.pkl` advanced to step 1812: env 3 passed response combat where attacking garden `STT01-008` dealt 2 damage to defending garden `AZK01-054` and died to 7 return damage.
- `AZK01-054` has inherent `Godmode`, but its post-combat HP remained positive (`7 -> 5`). C combat only clamps godmode damage when HP would drop below 0, so this branch is equivalent to normal damage and does not require the generic path.
- Relaxed `_response_noop_entity_combat_fast_mask` so defender `grant_godmode`/inherent `Godmode` is allowed only when `defender_after > 0`, matching the existing attacker-side guard and still rejecting lethal/clamped godmode cases.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diag_1812_components.py /tmp/jax_probe_1812.pkl` reports `_response_noop_entity_combat_fast_mask == [1, 0, 0, 1]`; env 3 is the new godmode-survivor row.

## 2026-06-19 — Step 1817 STT01-006 pinging surviving godmode target

- Re-probe from `/tmp/jax_probe_1812.pkl` advanced to step 1817: env 3 selected opponent garden slot 1 `AZK01-054` as `STT01-006`'s mandatory When Attacking damage target.
- As with the step-1812 combat case, `AZK01-054`'s inherent `Godmode` does not change the effect-damage outcome when the 1 damage leaves positive HP (`7 -> 6`). Lethal/clamped godmode damage remains rejected.
- Relaxed `_effect_stt01_006_fast_mask` so target `grant_godmode`/inherent `Godmode` is allowed only when `target_hp > 1`, matching `deal_effect_damage` semantics for a 1-damage effect.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1817.py /tmp/jax_probe_1817.pkl` reports `_effect_stt01_006_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1860 AZK01-127 into STT04-007 response trigger

- Re-probe from `/tmp/jax_probe_1833.pkl` advanced to step 1860: env 2 resolved response spell `AZK01-127` against opponent garden slot 0 `STT04-007` at 2 HP.
- `STT04-007` queues a takes-damage trigger from effect damage and immediately resolves as a no-target +1 attack EOT buff when once-per-turn is unused and the source remains in garden/alley.
- Extended `step_effect_azk01_127_fast` to pop a queued `STT04-007` takes-damage trigger, apply the +1 EOT attack modifier, and mark the once flag before continuing response-window close/combat resolution. Relaxed `_effect_azk01_127_fast_mask` for the matching nonlethal `STT04-007` target.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1452_azk01127.py /tmp/jax_probe_1860.pkl` reports `_effect_azk01_127_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1867 spent STT04-007 response-combat trigger

- Re-probe from `/tmp/jax_probe_1860.pkl` advanced to step 1867: env 2 passed the response window for `STT04-007` attacking garden `STT02-008`; both entities die from combat damage.
- `STT04-007` had already used its once-per-turn trigger from the previous `AZK01-127` effect damage. Combat still queues a takes-damage trigger; the supported fast path must pop it and apply no effect after the source leaves play / once is spent.
- Extended `step_response_noop_entity_combat_fast` with the same `STT04-007` takes-damage trigger pop/apply-on-survive logic used by direct entity combat, and relaxed `_response_noop_entity_combat_fast_mask` for clean attacker/defender `STT04-007` combat-damage trigger shapes.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diag_1812_components.py /tmp/jax_probe_1867.pkl` reports `_response_noop_entity_combat_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1897 AZK01-117 self-lethal play

- Re-probe from `/tmp/jax_probe_1867.pkl` advanced to step 1897: env 2 played `AZK01-117` from hand while its own `AZK01-121` leader had exactly 2 HP.
- The existing helper already applies the spell cost as 2 effect damage to the owner leader and computes terminal rewards when that sets `winner`. The host mask was unnecessarily rejecting this legal self-lethal branch with `leader_hp > 2`.
- Relaxed `_play_spell_azk01_117_fast_mask` to require only positive clean leader HP; damage triggers/godmode/carapace/effect immunity remain rejected.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1897_azk117.py /tmp/jax_probe_1897.pkl` reports `_play_spell_azk01_117_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1923 STT01-014 attach with STT02-012 garden watcher

- Re-probe from `/tmp/jax_probe_1897.pkl` advanced to step 1923: env 1 attached `STT01-014` from hand while friendly `STT02-012` was in garden.
- Attaching a weapon does not change garden entity counts, and `step_attach_weapon_simple_fast` already recomputes passives after the attach and opens `STT01-014`'s optional leader-damage selection. Treating garden `STT02-012` as an unsafe attach watcher was too conservative.
- Relaxed `_attach_weapon_simple_fast_mask` so `STT02-012` is recompute-safe for attach watcher and attach-target checks, matching the helper's post-attach passive recompute behavior.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1795.py /tmp/jax_probe_1923.pkl` reports `_attach_weapon_simple_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1932 AZK01-022 alley play with deferred STT02-012 passives

- Re-probe from `/tmp/jax_probe_1923.pkl` advanced to step 1932: env 1 played `AZK01-022` to alley while an `STT02-012` passive queue/event was still deferred from earlier garden-count changes.
- Generic static action opens `AZK01-022`'s optional confirmation before passive recompute can drain (`recompute_passives` self-gates while `ab_phase != NONE`). The split helper follows the same shape and leaves the deferred passive work pending under the ability context.
- Relaxed `_play_azk01_022_confirm_fast_mask` for alley placement with deferred `STT02-012` passive work and treated `STT02-012` as an inert watcher for this alley-only placement case. Garden placement remains conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1932_azk022.py /tmp/jax_probe_1932.pkl` reports alley `_play_azk01_022_confirm_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1933 AZK01-022 confirmation with deferred STT02-012 passives

- After the step-1932 alley play, the next frontier was the immediate `AZK01-022` optional confirmation with the same deferred `STT02-012` passive queue/event state.
- Confirming the optional ability only advances the ability FSM into cost selection; it does not mutate board zones, and generic recompute remains gated while the ability context is active.
- Relaxed `_confirm_azk01_022_fast_mask` to allow deferred passive/STT02-012 work for the confirmation step while keeping trigger/redirect/combat/winner and `STT02-010` observer checks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1933_confirm.py /tmp/jax_probe_1933.pkl` reports `_confirm_azk01_022_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1934 AZK01-022 cost selection with deferred passives

- After confirming `AZK01-022`, the next frontier was its cost selection while the same deferred `STT02-012` passive queue/event remained.
- The cost step discards a hand card and moves the ability FSM to effect selection; it does not mutate garden counts, and generic passive recompute is still gated by the active ability context.
- Relaxed `_select_cost_azk01_022_fast_mask` to allow deferred passive/STT02-012 work while retaining trigger/redirect/combat/winner, bounce-target, passive-watch, and `STT02-010` observer checks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1934_cost.py /tmp/jax_probe_1934.pkl` reports `_select_cost_azk01_022_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1935 AZK01-022 effect with deferred passives

- After the `AZK01-022` cost step, the next frontier was its effect selection returning opponent garden slot 1 while deferred `STT02-012` passive work was still pending.
- `step_effect_azk01_022_fast` returns the target, clears the ability context, and calls `recompute_passives`, so this is the point where the deferred `STT02-012` work should drain.
- Relaxed `_effect_azk01_022_fast_mask` to allow pending passive/STT02-012 work for clean bounce targets while keeping `STT02-010` observer and non-inert passive watcher checks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1935_effect.py /tmp/jax_probe_1935.pkl` reports `_effect_azk01_022_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 1938 AZK01-128 response play with stale attacker zone

- Re-probe from `/tmp/jax_probe_1935.pkl` advanced to step 1938: env 3 played response spell `AZK01-128` while `combat_attacker` still pointed at an HP-2 attacker instance that was no longer in garden.
- The current JAX generic `AZK01-128` play validation only checks that `combat_attacker >= 0` and the tracked attacker's HP is <= 2; stricter zone/type checks happen at effect-target selection. The split play mask was stricter than generic and rejected the spell setup.
- Relaxed `_play_spell_azk01_128_fast_mask` to match generic play validation for the response setup: defender player owns the response, `combat_attacker >= 0`, and tracked attacker HP <= 2. Effect selection remains conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1938_azk128.py /tmp/jax_probe_1938.pkl` reports `_play_spell_azk01_128_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1939 AZK01-128 no-target effect skip

- After the stale-attacker `AZK01-128` response play, the next frontier was its effect-selection `NOOP`: no valid garden target remained for the tracked attacker, so generic allows skip because remaining target count is zero.
- Extended `step_effect_azk01_128_fast` to clear the ability context on `NOOP` when no generic target is available, then run the existing response-close/combat continuation. Select-target behavior is unchanged.
- Relaxed `_effect_azk01_128_fast_mask` for the matching no-target `NOOP` branch while still requiring clean response state and falling back when a target exists but is unsupported.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1939_azk128_effect.py /tmp/jax_probe_1939.pkl` reports `_effect_azk01_128_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 1947 STT03-004 alley activation

- Re-probe from `/tmp/jax_probe_1939.pkl` advanced to step 1947: env 2 activated `STT03-004` from alley (`ACTIVATE_ALLEY_ABILITY [12,0,1,0]`).
- `STT03-004`'s no-target ability is legal from alley as well as garden: sacrifice the source, then heal the owner leader by 1. The existing helper only looked up garden sources.
- Generalized `step_activate_stt03_004_fast` and `_activate_stt03_004_fast_mask` to accept both garden/leader and alley activation action shapes, using the correct source slot field and shaped action type.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1947_stt03004.py /tmp/jax_probe_1947.pkl` reports `_activate_stt03_004_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1955 STT03-016 with STT03-006 death trigger

- Re-probe from `/tmp/jax_probe_1947.pkl` advanced to step 1955: env 2 played `STT03-016` while opponent garden contained HP<=2 entities including `STT03-006`.
- `STT03-016` destroys all opposing garden entities with CurStats/Entity type and `cur_hp <= 2`; `STT03-006` queues a When Destroyed draw-then-discard trigger but does not block the immediate spell resolution.
- Tightened `step_play_spell_stt03_016_fast` to mark only opposing garden entities, matching the C card text and existing mask semantics.
- Relaxed `_play_spell_stt03_016_fast_mask` for the narrow supported destroy-trigger shape: at most one marked `STT03-006` When Destroyed trigger and no other marked destroy triggers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1955_stt03016.py /tmp/jax_probe_1955.pkl` reports `_play_spell_stt03_016_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Steps 1956 through 1959 pending STT03-006 trigger interleaving

- After `STT03-016`, the deterministic legal stream continued with the queued `STT03-006` death trigger still pending while normal main/response actions remained legal.
- Relaxed split masks that do not consume or reorder that pending single trigger:
  - `_activate_azk01_123_fast_mask` and `_effect_azk01_123_fast_mask` for the leader +1 HP ability sequence;
  - `_attack_leader_response_fast_mask` / `_attack_leader_simple_fast_mask` for clean leader attacks;
  - `_response_noop_leader_combat_fast_mask` for response-window pass into leader combat resolution.
- Each relaxation requires exactly one queued `STT03-006` When Destroyed trigger with its owner still holding a hand card; other trigger shapes still fall back to generic.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after each mask update.
  - `/tmp/diagnose_step1956_azk01123.py /tmp/jax_probe_1956.pkl` reports `_activate_azk01_123_fast_mask == [0, 0, 1, 0]`.
  - `/tmp/diagnose_step1958_attack.py /tmp/jax_probe_1958.pkl` reports `_attack_leader_response_fast_mask == [0, 0, 1, 0]`.
  - `/tmp/diagnose_step1959_response_noop.py /tmp/jax_probe_1959.pkl` reports `_response_noop_leader_combat_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1977 STT03-004 garden activation with inactive STT03-013 watcher

- Re-probe from `/tmp/jax_probe_1972.pkl` advanced to step 1977: env 2 activated garden `STT03-004` while opponent had `STT03-013` in alley.
- `STT03-013` only triggers on entering garden; an alley copy is inert for `STT03-004` sacrificing itself from garden. The STT03-004 host mask was treating it as a non-inert simple-play watcher.
- Relaxed `_activate_stt03_004_fast_mask` to treat `STT03-013` as inert in the passive-watch check. Garden `STT02-012` and other active watchers remain conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step1977_stt03004.py /tmp/jax_probe_1977.pkl` reports `_activate_stt03_004_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 1991 AZK01-004 garden attack into entity

- Re-probe from `/tmp/jax_probe_1977.pkl` advanced to step 1991: env 0 attacked with garden `AZK01-004` into an enemy garden entity.
- The existing `AZK01-004` fast helper applied the When Attacking +1 ATK buff but only had a manual leader-damage resolve branch. It could not safely resolve entity combat.
- Reworked `step_attack_azk01_004_leader_fast` to declare combat after the buff, then use `transition_to_combat_resolve` + `combat_resolve` for no-response branches. This preserves response-window behavior and delegates entity combat deaths/damage bookkeeping to the shared combat implementation.
- Relaxed `_attack_azk01_004_leader_fast_mask` for clean no-response garden-entity targets with no damage/destroy/attachment complications.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step1991_azk01004.py /tmp/jax_probe_1991.pkl` reports `_attack_azk01_004_leader_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 2016 AZK01-070 response activation with inert watchers

- Re-probe from `/tmp/jax_probe_1991.pkl` advanced to step 2016: env 3 activated response garden `AZK01-070` while `STT01-009`, `STT01-011`, and `STT03-013` were on board.
- `AZK01-070` activation taps and deals 1 nonlethal damage to itself before effect selection; this does not change weapon discard counts, garden/alley membership, or STT03-013's enter-garden state.
- Relaxed `_activate_azk01_070_fast_mask` and `_effect_azk01_070_fast_mask` to treat `STT01-009`, `STT01-011`, `STT02-012`, and `STT03-013` as inert watcher rows for this path while keeping source damage trigger/protection checks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2016_azk01070.py /tmp/jax_probe_2016.pkl` reports `_activate_azk01_070_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2049 garden attacker with attached STT01-012

- Re-probe from `/tmp/jax_probe_2016.pkl` advanced to step 2049: env 3 attacked the opposing leader with a garden `STT01-011` carrying `STT01-012` and an additional inert `STT01-013` weapon.
- Generalized `step_attack_stt01_012_response_fast` from leader-only attackers to either leader or garden attackers, preserving the attached `STT01-012` top-card mill before phase-gate/combat handling.
- Relaxed `_attack_stt01_012_response_fast_mask` to:
  - match either leader or garden attackers;
  - match either garden or leader defenders;
  - require exactly one attached `STT01-012`, but allow other attached weapons when they have no attacking/after-attacking trigger and are not `AZK01-044`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step2049_stt01012.py /tmp/jax_probe_2049.pkl` reports `_attack_stt01_012_response_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2063 one-sided entity combat death

- Re-probe from `/tmp/jax_probe_2049.pkl` advanced to step 2063: response-window `NOOP` resolved entity combat where only the attacker died and the defender took no damage.
- `_response_noop_entity_combat_fast_mask` only admitted mutual/defender damage shapes. Added a clean attacker-death branch requiring no attacker When Destroyed trigger and no passive death watcher.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diag_1812_components.py /tmp/jax_probe_2063.pkl` reports `_response_noop_entity_combat_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2073 STT01-017 finish with passive recompute

- Re-probe after the attacker-death response fix advanced to step 2073: `STT01-017` effect selection finished via `NOOP` after damage had changed passive-derived stats.
- `step_effect_stt01_017_fast` now clears the ability context and recomputes passives on the finish branch; `_effect_stt01_017_fast_mask` admits the final selection/skip step while keeping earlier selection constraints conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step2073_stt01017.py /tmp/jax_probe_2073.pkl` reports `_effect_stt01_017_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 2100 STT01-016 trigger before response

- Re-probe from `/tmp/jax_probe_2073.pkl` advanced to step 2100: active `STT01-001` leader attacked a tapped `AZK01-054` garden entity while carrying `STT01-016`.
- `STT01-016` is valid on a Raizan leader and deals 1 effect damage to all opposing garden entities before the defender response window. The row is clean: all effect targets survive and have no takes-damage/destroy trigger, and `AZK01-054`'s godmode is nonlethal for both effect and later combat damage.
- Extended `step_attack_leader_response_fast` to inline this clean `STT01-016` trigger before opening response, and relaxed `_attack_leader_response_fast_mask` for exactly one clean `STT01-016` attached trigger plus nonlethal defender godmode.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step2100_attack.py /tmp/jax_probe_2100.pkl` reports `_attack_leader_response_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2108 STT01-017 nonlethal godmode target

- Re-probe from `/tmp/jax_probe_2100.pkl` advanced to step 2108: `STT01-017` effect selection targeted `AZK01-054`, whose inherent godmode does not alter a 1-damage nonlethal hit.
- Relaxed `_effect_stt01_017_fast_mask` so inherent/granted godmode only blocks the fast path when the 1-damage effect would drop the target below 0. Clean nonlethal godmode targets now stay on the split path.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2073_stt01017.py /tmp/jax_probe_2108.pkl` reports `_effect_stt01_017_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2138 attack over recompute-safe passive queue

- Re-probe from `/tmp/jax_probe_2108.pkl` advanced through several new compiled shapes and stopped at step 2138 on leader-attack declarations while a stale passive queue was still present.
- `step_attack_leader_response_fast` recomputes passives before declaring combat. Relaxed its passive-queue gate for the same recompute-safe watchers used elsewhere (`STT01-008`, `STT01-009`, `STT01-011`, `STT02-010`, `AZK01-010`, `AZK01-019`, `AZK01-073`, `STT02-012`, inert `STT03-013`).
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2100_attack.py /tmp/jax_probe_2138.pkl` reports `_attack_leader_response_fast_mask == [0, 0, 1, 1]`.

## 2026-06-19 — Step 2165 STT04-001 into AZK01-062 redirect

- Re-probe from `/tmp/jax_probe_2138.pkl` advanced to step 2165: `STT04-001` selected a friendly alley `AZK01-062` as its damage-and-buff target.
- Generic effect damage defers damage into Pekiro's redirect queue, applies the +1 ATK while the target is still in play, clears the STT04-001 context, then begins the queued `AZK01-062` trigger. The split helper now uses the same redirect-capable `deal_effect_damage` path and begins the redirect trigger after clearing/recomputing.
- Relaxed `_effect_stt04_001_fast_mask` for an unused `AZK01-062` redirect target while keeping other damage-trigger/destroy shapes conservative.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step2165_stt04001.py /tmp/jax_probe_2165.pkl` reports `_effect_stt04_001_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 2205 STT01-016 attack trigger with Pekiro redirects

- Re-probe from `/tmp/jax_probe_2165.pkl` advanced to step 2205: `STT01-001` leader with attached `STT01-016` attacked the opposing leader while the opposing garden contained two `AZK01-062` redirect targets.
- Extended `step_attack_leader_simple_fast` for this trigger shape: declare/store pending combat before resolving `STT01-016`, begin the first queued `AZK01-062` redirect instead of resolving leader combat immediately, and preserve combat state under the redirect ability context.
- Relaxed the leader-attack STT01-016 mask for unused `AZK01-062` redirect targets while preserving conservative checks for other takes-damage/deals-damage/destroy/godmode/carapace complications.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step2100_attack.py /tmp/jax_probe_2205.pkl` reports `_attack_leader_simple_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 2206 chained AZK01-062 redirect skip

- The step-2205 attack left an active `AZK01-062` redirect ability, one additional queued Pekiro trigger, and pending combat in main phase.
- `step_effect_azk01_062_fast` now treats a queued Pekiro trigger as valid input while another Pekiro context is active, fizzles only queued Pekiro triggers that have no matching pending redirect, and runs `phase_gate` after the final redirect clears if main-phase combat is still pending.
- Relaxed `_effect_azk01_062_fast_mask` for main-phase pending-combat redirect contexts and queued Pekiro follow-up triggers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the helper/mask update.
  - `/tmp/diagnose_step2206_azk01062.py /tmp/jax_probe_2206.pkl` reports `_effect_azk01_062_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 2255 AZK01-127 over STT02-012 watcher

- Re-probe from `/tmp/jax_probe_2206.pkl` advanced to step 2255: `AZK01-127` response effect selected a 1-HP `STT02-008` while `STT02-012` was in garden.
- The AZK01-127 helper deals damage, clears the ability context, and calls `recompute_passives`, so the `STT02-012` garden-count update is handled after the lethal clean hit.
- Relaxed `_effect_azk01_127_fast_mask` to treat `STT02-012` as recompute-safe for this effect path instead of only when it was in alley.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2255_azk01127.py /tmp/jax_probe_2255.pkl` reports `_effect_azk01_127_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 2259 STT03-016 over STT02-012 watcher

- Re-probe from `/tmp/jax_probe_2255.pkl` advanced to step 2259: `STT03-016` destroyed multiple opposing garden entities at HP <= 2 while `STT02-012` remained in garden.
- `step_play_spell_stt03_016_fast` destroys the marked entities and then calls `recompute_passives`, so the unmarked `STT02-012` observer is safe for this batch-destroy shape.
- Relaxed `_play_spell_stt03_016_fast_mask` to treat unmarked `STT02-012` like other recompute-safe registered passive observers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2259_stt03016.py /tmp/jax_probe_2259.pkl` reports `_play_spell_stt03_016_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 2264 passive NOOP plus stale AZK01-040 combat fizzle

- Re-probe from `/tmp/jax_probe_2259.pkl` advanced to step 2264 with two `NOOP` rows: a main-phase pass with pending `STT02-012` passive work, and a response-window pass where combat referenced a removed attacker into `AZK01-040`.
- The main passive row was already covered by `_main_noop_passive_fast_mask`. The remaining gap was `_response_noop_combat_fizzle_fast_mask` rejecting fizzled combat when the defender had a When Attacked timing tag.
- Relaxed the fizzle branch to allow invalid-combat cleanup regardless of defender When Attacked timing; the helper delegates to `combat_resolve`, which only clears fizzled combat and does not run defender triggers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2264_noop.py /tmp/jax_probe_2264.pkl` reports `_main_noop_passive_fast_mask == [0, 0, 1, 0]` and `_response_noop_combat_fizzle_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2264 AZK01-040 queued before fizzle

- Static review of the step-2264 response `NOOP` showed the prior fizzle relaxation was too broad: `transition_to_combat_resolve` queues the defender's `AZK01-040` When Attacked trigger even when the attacker is stale.
- Corrected the routing so `_response_noop_azk01_040_fast_mask` handles this shape and begins the AZK01-040 effect context; `_response_noop_combat_fizzle_fast_mask` no longer claims `AZK01-040` defenders.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask correction.
  - `/tmp/diagnose_step2264_noop.py /tmp/jax_probe_2264.pkl` now reports `_main_noop_passive_fast_mask == [0, 0, 1, 0]` and `_response_noop_azk01_040_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2265 STT04-002 and stale AZK01-040 effects

- Re-probe from `/tmp/jax_probe_2264.pkl` now routes the response pass through `AZK01-040`; the next frontier at step 2265 had two effect-selection rows with action type 14.
- `STT04-002` was already covered despite pending passive work. The remaining gap was `AZK01-040` effect selection after a stale attacker caused combat to fizzle after the effect.
- Relaxed `_effect_azk01_040_fast_mask` so stale-attacker combat contexts skip combat-damage shape requirements while preserving leader-target damage checks and the `AZK01-040` source/defender constraints.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2265_effects.py /tmp/jax_probe_2265.pkl` reports `_effect_stt04_002_fast_mask == [1, 0, 0, 0]` and `_effect_azk01_040_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2268 AZK01-111 selection-to-garden

- Re-probe from `/tmp/jax_probe_2265.pkl` advanced to step 2268: `AZK01-111` had sacrificed itself, selected an optional damage target, then moved one cost-2 hand entity (`AZK01-006`) into the selection zone with legal `SELECT_TO_GARDEN` placements.
- Added `_select_azk01_111_garden_fast_mask` so AZK01-111 selection-to-garden rows route through the static `SELECT_TO_GARDEN` kernel instead of the generic split fallback. The static kernel preserves generic selection semantics (`process_selection_to_garden`, passive recompute, auto-resolve).
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/continue_jax_checkpoint_save.py 1 4 4000 /tmp/jax_probe_2265.pkl 2265 /tmp/jax_probe 2265` advanced past step 2268 and saved `/tmp/jax_probe_2271.pkl`; it later timed out during a separate compile at step 2271 with no generic fallback reported for AZK01-111.

## 2026-06-19 — Step 2294 AZK01-128 over STT02-012 watcher

- Re-probe from `/tmp/jax_probe_2277.pkl` advanced to step 2294: `AZK01-128` response effect selected the current low-HP attacking entity, which was a garden `STT02-012`.
- `step_effect_azk01_128_fast` destroys the selected attacker, clears the ability context, and calls `recompute_passives`, so `STT02-012` is recompute-safe here like the earlier AZK01-127 and STT03-016 destroy paths.
- Relaxed `_effect_azk01_128_fast_mask` to treat `STT02-012` as an inert passive watcher in this clean destroy shape regardless of garden/alley zone.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the mask update.
  - `/tmp/diagnose_step2294_azk01128.py /tmp/jax_probe_2294.pkl` reports `_effect_azk01_128_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 2389 attack with queued STT03-006 destroy trigger

- Re-probe from `/tmp/jax_probe_2330.pkl` advanced to step 2389: a main-phase attack was legal while one pre-existing `STT03-006` When Destroyed trigger remained queued from the opponent discard.
- Existing narrow attack masks intentionally require an empty trigger queue. Added `_attack_queued_stt03_006_fast_mask` and route this shape through the static `ATTACK` kernel, preserving generic semantics: apply the chosen attack, then let `auto_resolve` process the queued STT03-006 draw/discard ability.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2389_attack.py /tmp/jax_probe_2389.pkl` reports `_attack_queued_stt03_006_fast_mask == [0, 0, 0, 1]` with trigger head `STT03-006` timing 13.

## 2026-06-19 — Step 2390 STT03-006 discard during pending combat

- The step-2389 static attack route advanced to step 2390: resolving the queued `STT03-006` draw/discard trigger left combat context pending and a queued `AZK01-047` When Attacking trigger.
- The narrow `step_effect_stt03_006_fast` path only handles no-combat contexts, so routing this shape through it would skip generic `auto_resolve` combat/trigger continuation. Added `_effect_stt03_006_pending_combat_fast_mask` and route it through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2390_stt03006.py /tmp/jax_probe_2390.pkl` reports `_effect_stt03_006_pending_combat_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2429 response NOOP outside narrow combat helpers

- Re-probe from `/tmp/jax_probe_2390.pkl` advanced to step 2429: one row had a covered main-phase legal-only `NOOP`, while the uncovered row was a response-window `NOOP` with pending combat and available response spells.
- Existing response NOOP helpers only cover narrow combat-resolution/fizzle shapes. Added `noop_static_mask_host` for remaining non-ability `NOOP` rows and route them through the static `NOOP` kernel so generic response-pass semantics are preserved.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2429_noop.py /tmp/jax_probe_2429.pkl` reports `noop_static == [0, 0, 0, 1]`; the main-phase row remains covered by `_main_noop_simple_fast_mask`.

## 2026-06-19 — Step 2445 STT04-017 main spell

- Re-probe from `/tmp/jax_probe_2429.pkl` advanced to step 2445: `STT04-017` was the selected main-phase spell from hand, and no split path existed for its sacrifice-then-damage flow.
- Added `_play_spell_stt04_017_static_mask` and route this card through the static `PLAY_SPELL_FROM_HAND` kernel. This keeps the generated ability runtime responsible for the variable 1..5 sacrifice cost and later leader/garden damage target.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2445_stt04017.py /tmp/jax_probe_2445.pkl` reports `_play_spell_stt04_017_static_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2446 STT04-017 variable sacrifice cost

- After the step-2445 static play, STT04-017 entered cost selection with three friendly garden entities available and action type `SELECT_COST_TARGET`.
- Added static routes for the rest of STT04-017's generated ability flow: cost target selection, optional cost-finish `NOOP` after at least one sacrifice, and the final effect target selection. These preserve the generated runtime for variable 1..5 sacrifices and damage amount stored in `ab_scratch`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static routes.
  - `/tmp/diagnose_step2446_stt04017.py /tmp/jax_probe_2446.pkl` reports `_select_cost_stt04_017_static_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2452 STT04-001 effect fallback

- Re-probe from `/tmp/jax_probe_2446.pkl` advanced to step 2452 with two effect rows. `AZK01-007` was already covered; the uncovered row was `STT04-001` selecting a target that the narrow STT04-001 mask rejects because target damage triggers are not clean.
- Added `_effect_stt04_001_static_mask` and route only the rejected STT04-001 effect-selection shape through the static `SELECT_EFFECT_TARGET` kernel. Existing clean STT04-001 rows still use the narrow helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2452_effects.py /tmp/jax_probe_2452.pkl` reports `_effect_azk01_007_fast_mask == [1, 0, 0, 0]` and `_effect_stt04_001_static_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2475 generic attack fallback

- Re-probe from `/tmp/jax_probe_2452.pkl` advanced to step 2475 with env 1 choosing action `[6, 0, 4, 0]`, a main-phase attack not admitted by any narrow attack helper.
- Added `attack_static_mask_host` for remaining main-phase `ATTACK` rows with no active ability context and route them through the static `ATTACK` kernel. Existing narrow attack helpers still take precedence; this fallback preserves generated attack/combat/trigger semantics for uncovered legal attacks.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2475_attack_static.py /tmp/jax_probe_2475.pkl` reports `attack_static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2478 AZK01-062 redirect effect fallback

- The step-2475 static attack route advanced to step 2478. Env 1 was in `AZK01-062` effect selection with `redirect_count=1`, no queued trigger, and action `[14, 4, 0, 0]`.
- Existing `_effect_azk01_062_fast_mask` intentionally only admitted clean redirect/trigger arrangements and rejected this already-begun redirect effect. Added `_effect_azk01_062_static_mask` and route rejected AZK01-062 select-effect rows through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2478_azk01062_static.py /tmp/jax_probe_2478.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2480 STT04-016 cost fallback

- The step-2478 static AZK01-062 route advanced to step 2480. Env 1 was resolving `STT04-016` cost selection, action `[13, 0, 0, 0]`, targeting the only friendly garden entity `STT04-009` at 1 HP.
- Existing `_select_cost_stt04_016_fast_mask` rejects this target because the narrow helper does not model all STT04-009 damage/death trigger continuation. Added `_select_cost_stt04_016_static_mask` and route rejected STT04-016 cost rows through the static `SELECT_COST_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2480_stt04016_cost.py /tmp/jax_probe_2480.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2506 AZK01-066 Firestorm fallback

- Re-probe from `/tmp/jax_probe_2480.pkl` advanced through step 2505, then hit step 2506 with env 0 playing `AZK01-066` (`Firestorm`) from hand, action `[8, 2, 0, 0]`.
- Existing `_play_spell_azk01_066_fast_mask` rejects boards with take-damage/death/passive hazards. Added `_play_spell_azk01_066_static_mask` and route rejected Firestorm plays through the static `PLAY_SPELL_FROM_HAND` kernel so generated mass-damage trigger ordering is preserved.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2506_azk01066_spell.py /tmp/jax_probe_2506.pkl` reports `fast == [0, 0, 0, 0]` and `static == [1, 0, 0, 0]`.

## 2026-06-19 — Step 2541 AZK01-024 cost fallback

- Re-probe from `/tmp/jax_probe_2506.pkl` advanced to step 2541. Env 1 was resolving `AZK01-024` cost selection, action `[13, 1, 0, 0]`, with multiple friendly garden targets available.
- Existing `_select_cost_azk01_024_fast_mask` rejects some return/passive watcher arrangements. Added `_select_cost_azk01_024_static_mask` and route rejected AZK01-024 cost rows through the static `SELECT_COST_TARGET` kernel so generated return-to-hand and follow-up selection setup are preserved.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2541_azk01024_cost.py /tmp/jax_probe_2541.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2542 AZK01-024 placement fallback

- The step-2541 static AZK01-024 cost route advanced to step 2542. Env 1 selected the returned `STT02-006` from selection into alley slot 3, action `[21, 0, 3, 0]`, while `passive_queue_count=1`.
- Existing `_select_azk01_024_place_fast_mask` requires no pending passive work. Added `_select_azk01_024_place_static_mask` plus static `SELECT_TO_GARDEN`/`SELECT_TO_ALLEY` dispatch for rejected AZK01-024 placement rows.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2542_azk01024_place.py /tmp/jax_probe_2542.pkl` reports `static_alley == [0, 1, 0, 0]`.

## 2026-06-19 — Step 2561 STT01-002 equip fallback

- Re-probe from `/tmp/jax_probe_2542.pkl` advanced to step 2561. Env 2 was resolving `STT01-002` selection-to-equip, action `[22, 0, 5, 0]`, choosing a selected `STT01-014` weapon for the leader.
- Existing `_select_stt01_002_equip_fast_mask` rejects this trigger-bearing weapon/host shape. Added `_select_stt01_002_equip_static_mask` and route rejected STT01-002 equip rows through the static `SELECT_TO_EQUIP` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2561_stt01002_equip.py /tmp/jax_probe_2561.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 0, 1, 0]`.

## 2026-06-19 — Step 2683 STT02-015 response effect fallback

- Re-probe from `/tmp/jax_probe_2561.pkl` advanced to step 2683 after compiling the static `SELECT_TO_EQUIP` route. Env 0 was resolving `STT02-015` response effect selection during pending combat, action `[14, 5, 0, 0]`.
- Existing `_effect_stt02_015_fast_mask` rejects return targets when `STT02-010` observers or other return/passive timing could follow. Added `_effect_stt02_015_static_mask` and route rejected STT02-015 effect rows through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2683_stt02015_effect.py /tmp/jax_probe_2683.pkl` reports `fast == [0, 0, 0, 0]` and `static == [1, 0, 0, 0]`.

## 2026-06-19 — Step 2920 STT03-016 spell fallback

- Re-probe from `/tmp/jax_probe_2683.pkl` advanced to step 2920 after several long static compiles. Env 3 was playing `STT03-016` from hand, action `[8, 1, 0, 0]`, into a board with destroy/passive timing outside the narrow helper.
- Existing `_play_spell_stt03_016_fast_mask` only admits clean immediate destroys. Added `_play_spell_stt03_016_static_mask` and route rejected STT03-016 spell plays through the static `PLAY_SPELL_FROM_HAND` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2920_stt03016_spell.py /tmp/jax_probe_2920.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2924 STT03-004 alley activation fallback

- The step-2920 static spell route advanced to step 2924. Env 3 activated `STT03-004` from alley slot 4, action `[12, 0, 4, 0]`, while passive watcher state made the narrow Sloth Scarecrow mask reject.
- Added `_activate_stt03_004_static_mask` plus static garden/alley activation dispatch. The narrow helper remains preferred; rejected STT03-004 activations now route through the generated activation runtime.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2924_stt03004_activate.py /tmp/jax_probe_2924.pkl` reports `static_alley == [0, 0, 0, 1]`.

## 2026-06-19 — Step 2941 STT01-017 response effect fallback

- The step-2924 static activation route advanced to step 2941. Env 2 was resolving `STT01-017` response effect selection, action `[14, 2, 0, 0]`, against a trigger-bearing enemy garden entity during pending combat.
- Existing `_effect_stt01_017_fast_mask` only admits clean one- or two-target damage selections. Added `_effect_stt01_017_static_mask` and route rejected STT01-017 select-effect rows through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step2941_stt01017_effect.py /tmp/jax_probe_2941.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 0, 1, 0]`.

## 2026-06-19 — Step 3032 AZK01-124 effect fallback

- Re-probe from `/tmp/jax_probe_2941.pkl` advanced to step 3032. Env 0 was resolving `AZK01-124` effect selection after its cost, action `[14, 1, 0, 0]`, with damage/passive timing outside the narrow helper.
- Existing `_effect_azk01_124_fast_mask` only admits clean damage targets. Added `_effect_azk01_124_static_mask` and route rejected AZK01-124 select-effect rows through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the static route.
  - `/tmp/diagnose_step3032_azk01124_effect.py /tmp/jax_probe_3032.pkl` reports `fast == [0, 0, 0, 0]` and `static == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3124 AZK01-031 spell fallback

- Re-probe from `/tmp/jax_probe_3032.pkl` advanced to step 3124. Env 3 was playing `AZK01-031` from hand, action `[8, 2, 0, 0]`, while `passive_queue_count=1`.
- Existing `_play_spell_azk01_031_fast_mask` requires a clean passive queue. Added `_play_spell_azk01_031_static_mask` and route rejected AZK01-031/AZK01-092 reveal spell plays through the static `PLAY_SPELL_FROM_HAND` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after repairing the static-mask insertion.
  - `/tmp/diagnose_step3124_azk01031_spell.py /tmp/jax_probe_3124.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 0, 0, 1]`.

## 2026-06-19 — Step 3125 residual NOOP fallback

- The step-3124 AZK01-031 static route advanced to step 3125, where env 3 was in `SELECTION_PICK` for `AZK01-031` with action `NOOP`, `ab_phase=4`, `passive_queue_count=1`, and selected cards `AZK01-011`, `STT02-006`, `STT02-012`.
- Existing NOOP masks rejected this non-clean selection-pass shape. Added a residual static NOOP mask after all narrow NOOP routes, so legal NOOP rows not proven by fast masks route through the generated `NOOP` runtime instead of falling back to generic monolithic dispatch.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the NOOP residual route.
  - `/tmp/diagnose_step3125_noop_residual.py /tmp/jax_probe_3125.pkl` reports `main == [1, 0, 0, 0]`, `residual == [0, 0, 0, 1]`, and `combined == [0, 0, 0, 1]`.

## 2026-06-19 — Step 3126 AZK01-031 top-deck fallback

- The residual NOOP route advanced to step 3126. Env 3 was resolving `AZK01-031` bottom-deck phase with action `[24, 1, 0, 0]` (`TOP_DECK_CARD`) while `passive_queue_count=1`.
- Existing `_top_deck_azk01_031_fast_mask` requires a clean passive queue. Added static `TOP_DECK_CARD`/`BOTTOM_DECK_CARD`/`BOTTOM_DECK_ALL` routes for legal reveal-selection deck actions rejected by the narrow masks, preserving the specialized fast paths first.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the deck-selection static routes.
  - `/tmp/diagnose_step3126_top_static.py /tmp/jax_probe_3126.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 0, 0, 1]`.

## 2026-06-19 — Step 3143 AZK01-127 effect fallback

- Re-probe from `/tmp/jax_probe_3126.pkl` advanced to step 3143. The generic action type was `SELECT_EFFECT_TARGET`; env 2 was resolving `AZK01-127` during a response-window combat with action `[14, 2, 0, 0]`.
- Existing `_effect_azk01_127_fast_mask` only admits clean damage/redirect/trigger shapes. Added `_effect_azk01_127_static_mask` and route rejected AZK01-127 effect selections through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-127 static route.
  - `/tmp/diagnose_step3143_effect_static.py /tmp/jax_probe_3143.pkl` reports `azk_fast == [0, 0, 0, 0]` and `azk_static == [0, 0, 1, 0]`; env 0's same action type is already covered by `stt_fast == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3184 STT04-016 effect fallback

- Re-probe from `/tmp/jax_probe_3175.pkl` advanced to step 3184. Env 2 was resolving `STT04-016` optional effect selection after cost payment, action `[14, 3, 0, 0]`, while a pending trigger was queued.
- Existing `_effect_stt04_016_fast_mask` rejects this queued-trigger/passive-sensitive target shape. Added `_effect_stt04_016_static_mask` and route rejected STT04-016 effect selections through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT04-016 effect static route.
  - `/tmp/diagnose_step3184_stt04016_effect.py /tmp/jax_probe_3184.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 0, 1, 0]`.

## 2026-06-19 — Step 3191 AZK01-119 activation fallback

- The STT04-016 static route advanced to step 3191. Env 1 activated leader `AZK01-119` with action `[11, 0, 0, 0]`.
- Existing `_activate_azk01_119_fast_mask` expects the leader activation source index as `GARDEN_SIZE` and also requires non-board payment. The legal row used source index `0`; added `_activate_azk01_119_static_mask` and route rejected AZK01-119 leader activations through the static garden/leader activation kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-119 activation static route.
  - `/tmp/diagnose_step3191_azk01119_activate.py /tmp/jax_probe_3191.pkl` reports `fast == [0, 0, 0, 0]` and `static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 3192 selection-pick static fallback

- The AZK01-119 static activation route advanced to step 3192. Env 1 was in `SELECTION_PICK` for source `AZK01-041`, action `[18, 0, 0, 0]`, with one selected `AZK01-094` weapon.
- Existing specialized selection-pick masks do not include AZK01-041. Added a residual static `SELECT_FROM_SELECTION` route for rows that exactly match the pending legal mask and are not covered by a narrow selection-pick helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the selection-pick static route.
  - `/tmp/diagnose_step3192_select_static.py /tmp/jax_probe_3192.pkl` reports `static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 3394 STT02-017 spell fallback

- Re-probe from `/tmp/jax_probe_3252.pkl` advanced to step 3394. Env 0 played `STT02-017` from hand, action `[8, 0, 0, 0]`, into a board with return/passive timing outside the narrow helper.
- Existing `_play_spell_stt02_017_fast_mask` only admits clean Shao bounce flows. Added `_play_spell_stt02_017_static_mask` and route rejected STT02-017 spell plays through the static `PLAY_SPELL_FROM_HAND` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the STT02-017 spell static route.
  - `/tmp/diagnose_step3394_stt02017_spell.py /tmp/jax_probe_3394.pkl` reports `fast == [0, 0, 0, 0]` and `static == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3480 AZK01-105 effect fallback

- Re-probe from `/tmp/jax_probe_3394.pkl` advanced to step 3480. Env 0 was resolving `AZK01-105` effect selection after costs, action `[14, 1, 0, 0]`, with passive-sensitive targets on board.
- Existing `_effect_azk01_105_fast_mask` requires a clean damage/passive shape. Added `_effect_azk01_105_static_mask` and route rejected AZK01-105 effect selections through the static `SELECT_EFFECT_TARGET` kernel.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the AZK01-105 effect static route.
  - `/tmp/diagnose_step3480_azk01105_effect.py /tmp/jax_probe_3480.pkl` reports `fast == [0, 0, 0, 0]` and `static == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3508 confirmation static fallback

- Re-probe from `/tmp/jax_probe_3480.pkl` advanced to step 3508. Generic action type was `CONFIRM_ABILITY`; env 1 confirmed `STT02-009` and env 2 confirmed `STT01-002`.
- Existing narrow confirmation masks rejected these legal confirmation rows. Added a residual static `CONFIRM_ABILITY` route for confirm rows that exactly match the pending legal mask and are not covered by a narrow confirmation helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the confirmation static route.
  - `/tmp/diagnose_step3508_confirm_static.py /tmp/jax_probe_3508.pkl` reports `static == [0, 1, 1, 0]`.

## 2026-06-19 — Step 3509 cost-selection static fallback

- The confirmation static route advanced to step 3509. Env 1 was resolving `STT02-009` cost selection, action `[13, 0, 0, 0]`.
- Existing narrow cost masks rejected the legal STT02-009 cost row. Added a residual static `SELECT_COST_TARGET` route for cost rows that exactly match the pending legal mask and are not covered by a narrow cost helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the cost-selection static route.
  - `/tmp/diagnose_step3509_cost_static.py /tmp/jax_probe_3509.pkl` reports `static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 3530 effect-selection static fallback

- The cost-selection static route advanced to step 3530. Env 1 was resolving `STT04-004` effect selection, action `[14, 6, 0, 0]`.
- Repeated legal `SELECT_EFFECT_TARGET` frontiers were coming from source-specific narrow masks. Added a residual static `SELECT_EFFECT_TARGET` route for legal effect rows, using the pending legal mask as the admission guard.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the effect-selection static route.
  - `/tmp/diagnose_step3530_effect_static.py /tmp/jax_probe_3530.pkl` reports `static == [0, 1, 0, 0]`.

## 2026-06-19 — Step 3588 weapon-attach static fallback

- Re-probe from `/tmp/jax_probe_3549.pkl` advanced to step 3588. Env 2 attached `STT01-013` from hand with action `[7, 0, 1, 0]`.
- Existing attach masks only admit simple weapon attachments or the narrow STT01-013 confirm shape. Added a residual static `ATTACH_WEAPON_FROM_HAND` route for legal weapon-attach rows rejected by those helpers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed after the attach static route.
  - `/tmp/diagnose_step3588_attach_static.py /tmp/jax_probe_3588.pkl` reports `static == [0, 0, 1, 0]`.

## 2026-06-19 — Step 3652 STT03-011 play-static avoidance

- Re-probe from `/tmp/jax_probe_3652.pkl` exposed a non-generic stall before the fallback trap: action rows were `[2, 1, 4, 0]`, `[1, 2, 1, 0]`, `[14, 3, 0, 0]`, and `[8, 1, 0, 0]`.
- Trace before the fix showed `play_static=1`; env 1 was a garden `STT03-011` play rejected by the dedicated fast mask only because an already-on-board `STT03-013` counted as a passive watcher.
- `STT03-013` is inert for another card entering garden; the `STT03-011` helper already recomputes passives. Updated `_play_stt03_011_effect_fast_mask` to treat `STT03-013` like the existing inert `STT01-008` watcher for this route, avoiding the huge generic `PLAY_ENTITY_TO_GARDEN` static compile.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Step-3652 trace now reports `play_static=0` and `play_stt03_011=1`; remaining active rows are covered by `play_simple=1`, `spell_stt01_017=1`, and the effect-selection static route.

## 2026-06-19 — Step 3652 effect-static residualization

- Step 3652 still stalled after the play-static avoidance because the broad residual `SELECT_EFFECT_TARGET` static route overlapped already-covered effect selections.
- Direct mask probe on `/tmp/jax_probe_3652.pkl` showed env 2 `STT01-017` effect selection had `effect_stt01_017_fast == [0, 0, 1, 0]` and `select_effect_raw == [0, 0, 1, 0]`; the catch-all static route would still compile and execute the monolithic effect kernel.
- Changed `select_effect_static_mask_host` to be a true residual: it now subtracts all source-specific effect fast/static masks that are already computed in the split dispatcher.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe for step 3652 reports `select_effect_residual == [0, 0, 0, 0]` when accounting for the `STT01-017` fast mask.

## 2026-06-19 — Step 3727 AZK01-021 reveal play

- Re-probe advanced from step 3652 to step 3727, then stalled on another `play_static=1` route before the fallback trap.
- Env 1 action was `[1, 4, 3, 0]`: active player 1 plays `AZK01-021` from hand to garden. Existing reveal pick/bottom-deck masks already support `AZK01-021`, but the play/reveal setup helper and mask only admitted `AZK01-003`.
- Generalized `step_play_azk01_003_reveal_fast` and `_play_azk01_003_reveal_fast_mask` to also admit `AZK01-021`, using the same top-5/select-1 flow with a `Driftward` subtype match. Existing `AZK01-003` behavior remains BlackJade with self excluded.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3727.pkl` reports `_play_azk01_003_reveal_fast_mask == [0, 1, 0, 0]` and the play-static residual for that row is cleared.

## 2026-06-19 — Frontier through step 3816

- Resume from `/tmp/jax_probe_3727.pkl` advanced without generic fallback through saved checkpoint `/tmp/jax_probe_3816.pkl` before the 3600s runner timeout.
- Slow first-use compiles observed at steps 3727, 3728, 3730, 3731, 3732, 3734, 3752, 3760, 3771, 3779, 3783, and 3810. They were compile cost, not fallback hits; checkpoints advanced after each.
- Step 3816 active rows are `[[14, 3, 0, 0], [1, 0, 4, 0], [6, 3, 5, 0], [14, 5, 0, 0]]`. Direct probe shows the effect rows are already covered by `_effect_stt02_011_fast_mask == [1, 0, 0, 0]` and `_effect_stt01_006_fast_mask == [0, 0, 0, 1]`; the residual `SELECT_EFFECT_TARGET` static route is zero after coverage.
- Next resume point: `/tmp/jax_probe_3816.pkl` at step `3816`.

## 2026-06-19 — Step 3816 AZK01-024 play-static avoidance

- Step 3816 initially showed `play_static=1` for env 1 action `[1, 0, 4, 0]`: `AZK01-024` played to garden while `STT02-012` was only in alley.
- Existing `_play_azk01_024_confirm_fast_mask` treated alley `STT02-012` as an active passive watcher. Relaxed the mask to treat alley `STT02-012` and inert `STT03-013` like other recompute-safe/non-triggering watchers for this confirm setup route.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3816.pkl` reports `_play_azk01_024_confirm_fast_mask == [0, 1, 0, 0]` and the play-static residual for that row is cleared.

## 2026-06-19 — Step 3817 STT02-013 passive-queue play

- Resume from `/tmp/jax_probe_3816.pkl` advanced step 3816, then hit a real generic fallback at step 3817: env 0 plays `STT02-013` to garden with action `[1, 0, 4, 0]` while `passive_queue_count=1` and `STT02-012` is in garden.
- Existing `STT02-013` play/reveal helper manually moved the card and therefore only admitted passive-clean boards. Reworked `step_play_stt02_013_reveal_fast` to use `_enter_board_slot`, matching the generic placement path's passive/STT02-012 zone-event bookkeeping, and switched the context/counter updates to the validated `place` predicate.
- Relaxed `_play_stt02_013_reveal_fast_mask` to allow existing passive queues and modeled `STT02-012`/inert `STT03-013` watchers; other simple-play watchers remain rejected.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3817.pkl` reports `_play_stt02_013_reveal_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3818 STT02-013 select-to-alley

- After the STT02-013 play fix, step 3818 hit generic action type `21` (`SELECT_TO_ALLEY`): env 0 selected revealed `STT02-003` from `STT02-013`'s selection to alley slot 3 while `passive_queue_count=2`.
- Existing `_select_stt02_013_pick_fast_mask` already models `STT02-013`/`AZK01-092` selection to hand/alley/garden and the helper recomputes passives after board placement, but the mask required passive-clean state.
- Relaxed the selection mask to allow deferred passive/STT02-012 queue work only for board placement actions (`SELECT_TO_ALLEY`/`SELECT_TO_GARDEN`), keeping hand-only selection strict.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3818.pkl` reports `_select_stt02_013_pick_fast_mask == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3820 AZK01-024 alley play-static avoidance

- Resume from `/tmp/jax_probe_3818.pkl` advanced to `/tmp/jax_probe_3820.pkl` before another long compile. Trace showed `play_static=1`; env 1 was `AZK01-024` played to alley with action `[2, 0, 0, 0]` while friendly `STT02-012` was in garden.
- `_play_azk01_024_confirm_fast_mask` had only treated alley `STT02-012` as inert; the helper's `_enter_board_slot` path already models garden add/remove passive bookkeeping. Relaxed the mask so `STT02-012` is not a blocker for AZK01-024 play-confirm setup in either zone.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3820.pkl` reports `_play_azk01_024_confirm_fast_mask(... Zone.ALLEY) == [0, 1, 0, 0]`.

## 2026-06-19 — Step 3825 AZK01-056 alley play-static avoidance

- Resume from `/tmp/jax_probe_3820.pkl` advanced to step 3825, then stalled with `play_static=1`.
- Env 1 action was `[2, 2, 4, 0]`: active player 0 plays `AZK01-056` to alley while `STT02-012` is in the opponent's garden. This alley play does not create a garden add/remove event for `STT02-012`, and the reveal helper already handles the no-passive-watch shape otherwise.
- Relaxed `_play_azk01_056_reveal_fast_mask` to treat `STT02-012` and inert `STT03-013` as non-blocking watchers for this reveal play route.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3825.pkl` reports `_play_azk01_056_reveal_fast_mask(... Zone.ALLEY) == [0, 1, 0, 0]`.

## 2026-06-19 — Step 3828 STT04-005 play-static avoidance

- Resume from `/tmp/jax_probe_3825.pkl` advanced to `/tmp/jax_probe_3828.pkl`, then stalled with `play_static=1`.
- Step 3828 rows were `[[1, 1, 1, 0], [20, 0, 0, 0], [10, 4, 0, 0], [0, 0, 0, 0]]`; env 0 active player 1 played `STT04-005` from hand to garden while the opponent had `STT02-012` in garden.
- `step_play_stt04_005_reveal_fast` already uses `_enter_board_slot`, so passive/STT02-012 placement bookkeeping is handled in the helper. Relaxed `_play_stt04_005_reveal_fast_mask` to treat `STT02-012` and inert `STT03-013` as non-blocking watchers for the Ruby reveal play route.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3828.pkl` reports `_play_stt04_005_reveal_fast_mask(... Zone.GARDEN) == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3830 AZK01-065 effect-static avoidance

- Resume from `/tmp/jax_probe_3828.pkl` advanced through the `STT04-005` play to `/tmp/jax_probe_3830.pkl`, then stalled on a broad static compile.
- Step 3830 rows were `[[19, 2, 0, 0], [14, 11, 0, 0], [0, 0, 0, 0], [7, 0, 4, 0]]`. The real uncovered row was env 1 `AZK01-065` effect selection targeting the enemy leader (`target_index=11`) while `STT02-012` was in the opponent garden.
- `_effect_azk01_065_fast_mask` already models self-damage, target damage, redirect/trigger-clean targets, and passive recompute after clearing. Relaxed its passive-watch gate so `STT02-012` does not force the row into `SELECT_EFFECT_TARGET` static fallback.
- Env 3's `ATTACH_WEAPON_FROM_HAND` row was already covered by `_attach_weapon_simple_fast_mask`; the raw `_attach_weapon_static_mask` diagnostic is not residualized, but split dispatch subtracts the simple attach mask before using the static route.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3830.pkl` reports `_effect_azk01_065_fast_mask == [0, 1, 0, 0]` and `_attach_weapon_simple_fast_mask == [0, 0, 0, 1]`.

## 2026-06-19 — Step 3833 STT04-005 bottom-deck-all static avoidance

- Resume from `/tmp/jax_probe_3830.pkl` advanced to `/tmp/jax_probe_3833.pkl`, then stalled on a bottom-deck static fallback.
- Step 3833 rows were `[[20, 0, 0, 0], [25, 0, 0, 0], [19, 0, 0, 0], [0, 0, 0, 0]]`; env 0 was `STT04-005` resolving `BOTTOM_DECK_ALL` with one pending passive queue entry from the earlier play.
- `step_bottom_deck_all_fast` finishes the selection, clears context, recomputes passives, and begins any queued trigger; relaxed `_bottom_deck_azk01_003_fast_mask` so `STT04-005` bottom-deck cleanup can carry deferred passive queue work instead of compiling the broad `BOTTOM_DECK_ALL` static route.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3833.pkl` reports `_bottom_deck_azk01_003_fast_mask(... all_cards=True) == [1, 0, 0, 0]`.

## 2026-06-19 — Step 3941 STT01-007 alley play-static avoidance

- Resume from `/tmp/jax_probe_3833.pkl` advanced to `/tmp/jax_probe_3941.pkl`, then stalled with `play_static=1` and memory growth near the workstation limit, so the runner was cancelled before OOM.
- Step 3941 rows were `[[0, 0, 0, 0], [6, 4, 5, 0], [0, 0, 0, 0], [2, 0, 0, 0]]`. Env 3 was `STT01-007` played from hand to alley while `STT01-009` was already in garden.
- `STT01-009`'s passive depends on weapon count in discard and its own garden presence; playing `STT01-007` to alley does not change that condition. Relaxed `_play_stt01_007_confirm_fast_mask` so `STT01-009` does not force this confirm setup route into play-static fallback.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3941.pkl` reports `_play_stt01_007_confirm_fast_mask(... Zone.ALLEY) == [0, 0, 0, 1]`; the env 1 attack row is already covered by `_attack_leader_simple_fast_mask == [0, 1, 0, 0]`.

## 2026-06-19 — Step 3951 AZK01-069 play-static avoidance

- Resume from `/tmp/jax_probe_3941.pkl` advanced to `/tmp/jax_probe_3951.pkl`, then stalled again on `play_static=1`.
- Step 3951 rows were `[[10, 4, 4, 0], [0, 0, 0, 0], [16, 0, 0, 0], [1, 4, 1, 0]]`. Env 3 was `AZK01-069` played from hand to garden. The previous Beanz work covered `AZK01-069` reveal picks and bottom-deck cleanup, but the play/reveal setup mask and helper still only admitted `AZK01-033` Steelborn reveals.
- Generalized `step_play_azk01_033_reveal_fast` and `_play_azk01_033_reveal_fast_mask` to also admit `AZK01-069`; the helper now chooses Steelborn matching for `AZK01-033` and Beanz matching for `AZK01-069`.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3951.pkl` reports `_play_azk01_033_reveal_fast_mask(... Zone.GARDEN) == [0, 0, 0, 1]`; the other active rows are covered by `gate_simple == [1, 0, 0, 0]` and `_confirm_azk01_022_fast_mask == [0, 0, 1, 0]`.

## 2026-06-19 — Step 3954 AZK01-068 garden play-static avoidance

- Resume from `/tmp/jax_probe_3951.pkl` advanced to `/tmp/jax_probe_3954.pkl`, then stalled with `play_static=1`.
- Step 3954 rows were `[[0, 0, 0, 0], [1, 2, 4, 0], [0, 0, 0, 0], [19, 0, 0, 0]]`. Env 1 was `AZK01-068` played from hand to garden. `AZK01-068` only has an on-play effect in alley, so garden play is a simple entity placement.
- Relaxed `_play_entity_simple_fast_mask` so `AZK01-068` played to garden is treated as an implemented no-effect placement instead of forcing play-static fallback. Alley play remains outside this simple path and must use a real draw/discard setup path when encountered.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3954.pkl` reports `_play_entity_simple_fast_mask(... Zone.GARDEN) == [0, 1, 0, 0]`; env 3's Beanz bottom-deck row is covered by `_bottom_deck_azk01_003_fast_mask(... all_cards=False) == [0, 0, 0, 1]`.

## 2026-06-19 — Step 3965 AZK01-024 places STT02-003

- Resume from `/tmp/jax_probe_3954.pkl` advanced to `/tmp/jax_probe_3965.pkl`, then stalled on an `AZK01-024` placement static route.
- Step 3965 rows were `[[1, 4, 2, 0], [23, 0, 0, 0], [20, 0, 0, 0], [0, 0, 0, 0]]`. Env 1 selected `STT02-003` from `AZK01-024`'s selection to garden (`SELECT_TO_GARDEN`).
- `step_select_azk01_024_place_fast` already handled simple selected entity placement and selected `AZK01-022` confirmation setup. Extended it for selected `STT02-003`: after returning the remaining selection to hand and clearing the `AZK01-024` context, it opens the same top-5 Watercrafting reveal flow used by normal `STT02-003` play.
- Relaxed `_select_azk01_024_place_fast_mask` to admit `STT02-003` placement only when the owner deck is non-empty; the static placement route remains the fallback for other unsupported selected on-play targets.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env .venv/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py` passed.
  - Direct probe on `/tmp/jax_probe_3965.pkl` reports `_select_azk01_024_place_fast_mask(... Zone.GARDEN) == [0, 1, 0, 0]`.

## 2026-06-19 — Frontier reached 4000 steps

- Resume from `/tmp/jax_probe_3965.pkl` reached `NO_GENERIC through 4000 steps`.
- Slow first-use compiles in this segment included steps 3965, 3967, 3970, 3976, and 3992; no generic fallback was reported.
- Next gate is not the frontier probe. Move to true C/JAX parity verification (`jax_env/run_verify.sh -k fullpool`, then the full suite) before benchmarking.

## 2026-06-19 — Fullpool verifier compile mitigation

- The dynamic-dispatch `engine_step` fullpool pytest path remained impractical for the harness wall clock: previous attempts timed out at 3600s before a single fullpool case produced a result.
- Updated `jax_env/tests/test_l3_fullpool.py` to compile and cache one `engine_step_static_action` verifier kernel per primary action type. This matches the split vector backend path under parity test and avoids re-lowering the entire dynamic action dispatcher.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/tests/test_l3_fullpool.py` passed.
  - A single static-action fullpool mirror case and the c6 eager parity diagnostic are currently running.

## 2026-06-19 — c6 eager parity re-check

- `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 6 7 7006 90` completed with `no divergence in 90 steps`.
- This closes the stale HANDOFF warning that the c6 STT04-001 EOT ATK fix still needed eager verification through the historical step-82 failure window.
- Follow-up: running eager checks for the remaining historical fullpool hotspots (`c10`, `c11`, `c15`, `c16`, `m16`) while the static-action fullpool pytest case compiles/runs.

## 2026-06-19 — Static-action verifier shared across pytest parity suite

- The fullpool-only static verifier still left the rest of `jax_env/tests/test_*.py` on `jax.jit(engine_step)`, which would reintroduce the monolithic dispatch compile during the full parity suite.
- Added `cached_static_jit_step` in `jax_env/tests/conftest.py` and switched the L2/L3/passive/obs pytest drivers to use that cached static-action step helper.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/tests/conftest.py jax_env/tests/test_l2_vanilla.py jax_env/tests/test_l3_abilities.py jax_env/tests/test_l3_abilities_batch1.py jax_env/tests/test_l3_abilities_batch2.py jax_env/tests/test_l3_abilities_batch3.py jax_env/tests/test_l3_abilities_batch4.py jax_env/tests/test_l3_passives.py jax_env/tests/test_obs_packing.py jax_env/tests/test_l3_fullpool.py` passed.

## 2026-06-19 — Cached legal-mask verifier

- Extended the pytest verifier helper to cache `build_mask` via `cached_jit_mask()` as well as static step kernels, then removed stale local `jax`/`build_mask` imports from the parity drivers.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/tests/conftest.py jax_env/tests/test_l2_vanilla.py jax_env/tests/test_l3_abilities.py jax_env/tests/test_l3_abilities_batch1.py jax_env/tests/test_l3_abilities_batch2.py jax_env/tests/test_l3_abilities_batch3.py jax_env/tests/test_l3_abilities_batch4.py jax_env/tests/test_l3_passives.py jax_env/tests/test_obs_packing.py jax_env/tests/test_l3_fullpool.py` passed.

## 2026-06-19 — Broad static pytest compile still impractical

- The stale single-case fullpool pytest run launched before the shared verifier helper was cancelled after about 40 minutes with no result and ~45GB RSS. It was compiling the broad `engine_step_static_action` path, not the split fast vector path that matters for throughput.
- Follow-up: created `/tmp/vector_fullpool_parity.py`, a C/JAX fullpool parity diagnostic that initializes all 36 explicit fullpool cases, drives them with the C legal-row RNG choices, compares semantic state + selection state + legal masks each step, and monkey-patches generic split fallback to fail fast.
- Active validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env ... AZK_JAX_SPLIT_TRACE=1 python/.venv-codex/bin/python -u /tmp/vector_fullpool_parity.py 600` is running.

## 2026-06-19 — Historical hotspot eager batch inconclusive

- Parallel CPU eager runs for `c10`, `c11`, `c15`, `c16`, and `m16` printed clean progress through the displayed step-90 window, but the combined background output did not include final `no divergence` or divergence lines.
- Treat this as only a progress signal, not closure. The vector fullpool parity run is the current authoritative gate.

## 2026-06-19 — Vector fullpool verifier chunking

- The all-36-case vector verifier was cancelled after ~34 minutes with no result and ~65GB RSS; compiling the batched shape `(36, ...)` was too large for safe iteration.
- Switched to 4-case chunks using the prior `/tmp/jaxcache` split-probe cache shape. Active chunk:
  - `PYTHONPATH=build/python/src:python/src:jax_env ... JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache AZK_JAX_SPLIT_TRACE=1 python/.venv-codex/bin/python -u /tmp/vector_fullpool_parity.py 600 m0 m1 m2 m3`

## 2026-06-19 — m0 vector split parity divergence at step 10

- The 4-case vector verifier chunk (`m0 m1 m2 m3`) compiled and began running, then failed on `m0` at step 10:
  - mismatch key: `ab_phase`
  - C: `0`
  - JAX split: `1`
  - recent actions ended with step 9 `GATE_PORTAL` row `(10, 4, 1, 0)`.
- Split trace around the failure showed the step-9 row routed through `gate_simple=1`, so the current suspicion is a fast `step_gate_portal_simple_fast`/mask misclassification rather than scalar engine semantics.
- Follow-up validation now running:
  - `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 0 0 12345 15`

## 2026-06-19 — STT01-002 gate portal no-confirm fix

- Root cause of the `m0` step-10 vector divergence: `step_gate_portal_simple_fast` treated `STT01-002` as an optional confirmation ability, leaving JAX in `ab_phase=CONFIRMATION`; C registry marks `STT01-002` as `is_optional=false` and runs `on_cost_paid` immediately, then either enters selection if a valid discard weapon exists or clears to `ABILITY_PHASE_NONE`.
- Changed the fast gate helper to handle `STT01-002` inline:
  - finds discard weapons with IKZ cost <= gate points,
  - immediately moves them to selection with `pick_max=1`,
  - sets `ab_costs_applied=True` and scratch kind `2` only when a selection exists,
  - leaves `ab_phase=NONE` when there are no valid weapons.
- Latest validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py` passed.

## 2026-06-19 — m0 scalar eager cross-check

- `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 0 0 12345 15` completed with `no divergence in 15 steps`.
- This confirms the m0 step-10 mismatch was in the split fast gate path, not the scalar JAX engine.

## 2026-06-19 — m0-m3 vector chunk rerun after STT01-002 fix

- Active validation:
  - `PYTHONPATH=build/python/src:python/src:jax_env ... JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache AZK_JAX_SPLIT_TRACE=1 python/.venv-codex/bin/python -u /tmp/vector_fullpool_parity.py 600 m0 m1 m2 m3`

## 2026-06-19 — m0-m3 vector chunk timeout after fix

- The rerun no longer failed at the previous `m0` step-10 `STT01-002` gate portal point, but it timed out at 3600s before reaching the 25-step progress print. The visible trace showed more split rows after the fixed gate step and a slow compile of `jit__step_batch_type` (~7 minutes) before timeout.
- Follow-up: rerunning the same chunk with `AZK_JAX_SPLIT_TRACE=0` and `max_steps=25` to verify the fixed prefix without trace overhead and with cache warmed by prior compiles.

## 2026-06-19 — STT01-002 fast helper targeted verification

- Targeted CPU no-JIT replay to `m0` step 9 and direct `step_gate_portal_simple_fast` call now matches C after the fixed `(10, 4, 1, 0)` gate portal:
  - C `ab_phase=0`
  - JAX fast `ab_phase=0`
  - semantic/selection diffs: `[]`
- Command:
  - `PYTHONPATH=build/python/src:python/src:jax_env JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u /tmp/check_stt01_gate_fast.py`

## 2026-06-19 — Scalar eager hotspot rechecks relaunched separately

- Relaunched the historical hotspot eager checks as separate background jobs so each case reports its own final status:
  - `pool 10 11 7010 130`
  - `pool 11 12 7011 100`
  - `pool 15 16 7015 130`
  - `pool 16 17 7016 130`
  - `pool 16 16 12361 130`

## 2026-06-19 — m0 attack masks are fast-covered

- Host-side mask replay for `m0` through step 29 showed the attacks at steps 20, 22, and 26 are covered by fast masks, not broad `attack_static`:
  - step 20 `(6, 1, 5, 0)`: `leader_simple`, `leader_garden_simple`
  - step 22 `(6, 0, 5, 0)`: `leader_simple`, `leader_garden_simple`
  - step 26 `(6, 1, 1, 0)`: `entity_mutual`, `leader_garden_simple`
- The vector verifier timeouts after the STT01-002 fix are compile-time cost of fast helpers, not an observed parity failure.

## 2026-06-19 — c11 eager recheck passed

- `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 11 12 7011 100` completed with `no divergence in 100 steps`.
- This covers the stale c11 combat/discard-order warning through the historical step-77 window.

## 2026-06-19 — c10 eager recheck passed

- `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 10 11 7010 130` completed with `no divergence in 130 steps`.
- This covers the stale c10 passive re-entry window noted around steps 95-101 and extends cleanly past step 120.

## 2026-06-19 — c15 eager recheck passed

- `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 15 16 7015 130` completed with `no divergence in 130 steps`.

## 2026-06-19 — c16 eager recheck passed

- `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 16 17 7016 130` completed with `no divergence in 130 steps`.

## 2026-06-19 — m16 eager recheck passed

- `JAX_PLATFORMS=cpu python/.venv-codex/bin/python -u jax_env/tests/diag_eager.py pool 16 16 12361 130` completed with `no divergence in 130 steps`.
- Historical hotspot status after this batch: c6, c10, c11, c15, c16, and m16 all clear through their known failure windows.

## 2026-06-19 — HANDOFF stale passive section patched

- Updated `jax_env/HANDOFF.md` so the passive-aura cluster is historical
  context, not active guidance. Fresh scalar eager replays now supersede the old
  c10/c15/c16/m16 warning through the known failure windows.
- Current parity gate remains rollout comparison, per arXiv 2603.12145's L3
  guidance: matched seeds/actions, full state + selection + legal-mask +
  terminal parity, before any throughput claims.

## 2026-06-19 — m0-m3 vector parity chunk relaunched

- Relaunched the split-vector C/JAX fullpool verifier after the STT01-002 fix:
  `PYTHONPATH=build/python/src:python/src:jax_env JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache XLA_PYTHON_CLIENT_PREALLOCATE=false AZK_JAX_SPLIT_TRACE=0 python/.venv-codex/bin/python -u /tmp/vector_fullpool_parity.py 600 m0 m1 m2 m3`
- This run compares semantic state, selection state, legal masks, and terminal
  status against the C oracle each step, and traps generic split fallback.
- Result: cancelled after about 29 minutes with no first progress print; RSS had
  climbed to roughly 43GB. Treat this as another broad/static compile trap, not
  a parity result.
- Next probe should use `FAIL_STATIC=1` to identify the first broad static route
  instead of letting XLA compile it.

## 2026-06-19 — JAX sim-only benchmark updated

- Replaced the stale monolithic `jax.jit(engine_step)` benchmark in
  `jax_env/benchmarks/bench_jax_env.py` with a split `JaxVecEnv` benchmark that
  matches the training backend, samples random legal rows from the JAX legal
  mask, and excludes warm-up/compile from reported SPS.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/benchmarks/bench_jax_env.py`

## 2026-06-19 — Vector parity verifier tracked

- Added `jax_env/tests/verify_vector_fullpool.py`, the tracked version of the
  split-vector fullpool C/JAX verifier previously living in `/tmp`.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/tests/verify_vector_fullpool.py`

## 2026-06-19 — HANDOFF runbook refreshed

- Updated stale `jax_env/HANDOFF.md` lower runbook text: current status now
  points to rollout coverage/compile fallback as the active gap, not the old
  passive cluster, and documents `verify_vector_fullpool.py` plus the sim-only
  benchmark scripts as the reproducible path forward.

## 2026-06-19 — JAX smoke config header fixed

- Removed the stale `python/config/azuki_jax_smoke.ini` comment claiming ability
  cards are no-op. The config now points readers back to fullpool parity before
  interpreting JAX SPS.

## 2026-06-19 — NOOP residual static narrowed

- `FAIL_STATIC=1 TRACE_ACTIONS=1 verify_vector_fullpool.py 25 m0 m1 m2 m3`
  reached step 11, then failed on broad `_noop_static_step_fn`.
- Trace showed the NOOP row was already covered by `effect_stt03_002=1`; the
  catch-all `noop_residual_static_mask_host` also matched because it did not
  exclude effect-selection contexts.
- Narrowed the residual NOOP static route to `ABILITY_PHASE_SELECTION_PICK`
  only. Effect-selection NOOPs must use source-specific effect masks instead of
  compiling the broad static NOOP kernel.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`

## 2026-06-19 — m0-m3 vector parity active-player divergence

- After the NOOP residual narrowing, `FAIL_STATIC=1 TRACE_ACTIONS=1
  verify_vector_fullpool.py 50 m0 m1 m2 m3` advanced past the step-11
  `STT03-002` skip without broad static fallback.
- New failure: `m2 step 16: active`, after recent step 15 action
  `(14, 5, 0, 0)` in `ph2ab3` effect selection. C active player is 0; JAX
  active player is 1.
- Next debug target is the m2 step-15 effect-selection fast path.

## 2026-06-19 — STT03-006 active restore in combat fast paths

- The m2 step-15 divergence was `STT03-006` resolving its discard-from-hand
  effect after combat destruction while control had transferred to the destroyed
  card's owner. C restored `active_player` to the original attacker after
  `azk_clear_ability_context`; JAX left it on the effect owner.
- Updated both manual `STT03-006` combat-death effect setup blocks in
  `jax_env/azuki_jax/step.py` to save/restore the prior active player when the
  effect owner differs from the current active player, matching
  `runtime.begin_ability` transfer semantics.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`

## 2026-06-19 — queued STT03-006 trigger restore guard

- Follow-up audit found `step_main_noop_stt03_006_trigger_fast`, the exact split
  path for a queued main-phase `STT03-006` death trigger, had the same manual
  transfer gap: it moved `active_player` to the trigger owner without recording
  the previous active player for context clear.
- Added the same save/restore guard there so queued-trigger starts match
  `runtime.begin_ability` when the trigger owner differs from current active.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py`

## 2026-06-19 — stale m2 verifier cancelled

- Cancelled the in-flight `verify_vector_fullpool.py 25 m2` run because it had
  loaded `step.py` before the follow-up queued-`STT03-006` restore patch.
- Re-run parity probes after code changes only; stale verifier results are not
  authoritative for this bug class.

## 2026-06-19 — targeted m2 STT03-006 restore check passed

- First inline eager diagnostic attempt failed because `_TrainingObservationData`
  is attribute-based, not subscriptable; no parity conclusion from that run.
- Re-ran the targeted m2 replay with generic JAX through step 13, then direct
  split helpers for step 14 `ATTACK (6,4,1,0)` and step 15
  `SELECT_EFFECT_TARGET (14,5,0,0)`.
- Result after the attack fast path: C and JAX both transfer active to player 1
  for the `STT03-006` effect; JAX now records `ab_saved_active=0` and
  `ab_restores_active=True`.
- Result after the effect fast path: C and JAX both return to
  `active_player=0`, `phase=MAIN`, `ab_phase=NONE`.

## 2026-06-19 — manual transfer audit

- A read-only reviewer scanned manual `_replace(active_player=...)` ability
  context starts in `jax_env/azuki_jax/step.py` and direct ability helpers.
- Findings: the remaining transfer-sensitive starts either delegate to
  `runtime.begin_ability`/`resolve_triggered_effect` or already carry
  conditional `ab_saved_active`/`ab_restores_active`. The only suspicious manual
  `STT03-006` starts were the three patched paths: queued main-phase trigger and
  the two combat-death setup blocks.

## 2026-06-19 — m2 vector restore recheck passed

- Current-code `FAIL_STATIC=1 TRACE_ACTIONS=1 verify_vector_fullpool.py 25 m2`
  completed with `no divergence in 25 steps across 1 cases`.
- The previous `m2 step 16 active` failure is fixed in the actual
  `JaxVecEnv` split dispatch, not only in the direct eager helper repro.
- Runtime note: this single-case JIT probe took about 2289s wall time and
  peaked around 37GB RSS while compiling/running traced split helpers.

## 2026-06-19 — m0-m3 chunk recheck relaunched

- Relaunched the first mirror chunk after the `STT03-006` restore fix:
  `PYTHONPATH=build/python/src:python/src:jax_env JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache XLA_PYTHON_CLIENT_PREALLOCATE=false AZK_JAX_SPLIT_TRACE=0 FAIL_STATIC=1 TRACE_ACTIONS=0 python/.venv-codex/bin/python -u jax_env/tests/verify_vector_fullpool.py 100 m0 m1 m2 m3`
- This should confirm the former m2 step-16 active-player divergence stays fixed
  while continuing to the next broad-static or semantic parity gap.

## 2026-06-19 — m1 false response phase after attack

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m0 m1 m2 m3` passed the former
  m2 active-player failure, then failed at `m1 step 19: phase` after step 18
  action `(6,3,5,0)`: C was in `MAIN`, JAX remained in `RESPONSE`.
- Host-mask replay before that attack showed `_attack_leader_response_fast_mask`
  matched only because the defending `STT02-001` leader was treated as a payable
  board response. The mask used card play IKZ cost (0 for leaders), and several
  response-window detectors did not require board ability IKZ payment at all.
- Updated duplicated attack/response masks to gate board responses with
  `ABILITY_IKZ_COST <= payment_sources`, while leaving hand response spell/play
  checks on card play cost plus next-play-cost reduction.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`

## 2026-06-19 — targeted m1 response-mask recheck passed

- Replayed m1 to step 18 and evaluated host masks after the board-response cost
  fix.
- `_attack_leader_response_fast_mask` is now `False` and
  `_attack_leader_simple_fast_mask` is `True` for action `(6,3,5,0)`, matching
  C's post-action `MAIN`/no ability-context state.

## 2026-06-19 — m0-m3 chunk recheck after response-cost fix

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after fixing board-response payment checks.

## 2026-06-19 — m1 response spell did not close combat

- Re-run after the board-response payment fix advanced past step 18, then failed
  at `m1 step 24: phase`: after step 22 `STT02-015` response spell and step 23
  effect target selection, C returned to `MAIN` while JAX stayed in
  `RESPONSE_WINDOW`.
- Targeted replay identified the source as `STT02-015` from hand slot 4; its
  effect fast path returned the target and cleared ability context, but did not
  run the post-response auto-advance (`transition_to_combat_resolve` /
  `combat_resolve`) used by generic engine auto-resolve.
- Added `_close_response_combat_if_idle` and applied it to response effect
  helpers for `STT02-015`, `STT02-016`, and `STT02-001`, so response effects
  close/resolve combat when no queued effects or further defender responses
  remain.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`

## 2026-06-19 — targeted STT02-015 response close passed

- Replayed m1 to the response window, then used direct split helpers for
  `STT02-015` play/effect.
- After step 23 effect selection, C and JAX both report `phase=MAIN`,
  `active_player=0`, `ab_phase=NONE`, and JAX has `combat_attacker=-1`.

## 2026-06-19 — m0-m3 chunk recheck after response-close fix

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after adding response-effect auto-close logic.

## 2026-06-19 — response auto-close audit

- A read-only response-window audit found additional split helpers that can
  resolve in `RESPONSE_WINDOW` and clear ability context without auto-closing an
  idle response/combat window: `AZK01-020`, `AZK01-029`, `STT01-017`,
  `AZK01-070`, and immediate response activation `AZK01-125`.
- `AZK01-062`, `AZK01-127`, and `AZK01-128` already duplicate the close/auto
  combat logic inline, so they are not semantic gaps.

## 2026-06-19 — stale m0-m3 recheck cancelled

- Cancelled the in-flight m0-m3 verifier before applying the proactive
  response auto-close fixes from the audit. Any result from that process would
  have used stale `step.py`.

## 2026-06-19 — proactive response auto-close fixes

- Applied `_close_response_combat_if_idle` to the remaining missing
  response-window resolution helpers identified by the audit:
  `step_effect_azk01_020_fast`, `step_effect_azk01_029_fast`,
  `step_effect_stt01_017_fast`, `step_effect_azk01_070_fast`, and
  `step_activate_azk01_125_fast`.
- Kept existing inline close/auto-combat logic in `AZK01-062`, `AZK01-127`, and
  `AZK01-128` unchanged.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`

## 2026-06-19 — m0-m3 chunk recheck after full auto-close sweep

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after the proactive response auto-close sweep.

## 2026-06-19 — m0-m3 advanced to cost-selection static fallback

- Full auto-close sweep recheck advanced past step 25, then failed inside
  `_select_cost_static_step_fn`, i.e. a broad static cost-selection route, not a
  semantic mismatch.
- Next probe should run the same chunk with action tracing around the failure to
  identify the case/card/action that needs a source-specific cost-selection
  split path.

## 2026-06-19 — cost-selection fallback trace launched

- Launched `TRACE_ACTIONS=1 AZK_JAX_SPLIT_TRACE=1 FAIL_STATIC=1
  verify_vector_fullpool.py 40 m0 m1 m2 m3` to identify the first
  `_select_cost_static_step_fn` row.

## 2026-06-19 — cost-selection fallback identified at m1 step 32

- Trace run showed the first `_select_cost_static_step_fn` happens on step 32
  after `m1` chooses `SELECT_COST_TARGET (13,2,0,0)`.
- The preceding m1 actions were `CONFIRM_ABILITY (16,0,0,0)` at step 31 and
  `SELECT_COST_TARGET` at step 32, so the next target is the m1 ability context
  source before step 32.

## 2026-06-19 — STT02-009 cost over STT02-010 observer

- Diagnosed the step-32 cost fallback as `STT02-009` cost selection returning
  friendly `STT02-007` while a single `STT02-010` when-returned observer was in
  garden. C/JAX generic keep the `STT02-009` effect context active and leave the
  `STT02-010` trigger queued (`trig_count=1`) until the `STT02-009` effect
  finishes.
- Relaxed `_select_cost_stt02_009_fast_mask` for the single-observer +
  `STT02-009` effect-available shape. Multiple observers or exhausted
  no-effect cases still stay off this fast path.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`

## 2026-06-19 — targeted STT02-009 cost mask recheck passed

- Replayed m1 to step 32 and re-evaluated the host mask for
  `SELECT_COST_TARGET (13,2,0,0)`.
- `_select_cost_stt02_009_fast_mask` now returns `True` for the single
  `STT02-010` observer + available `STT02-009` effect-target shape.

## 2026-06-19 — m0-m3 chunk recheck after STT02-009 cost fix

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after admitting the single-observer `STT02-009` cost shape.

## 2026-06-19 — m0-m3 advanced to generic NOOP fallback

- Recheck after the `STT02-009` cost-mask relaxation passed the step-32 cost
  fallback and advanced beyond step 25, then failed on generic static fallback
  for action type `NOOP`.
- Need a traced run to identify the exact case/phase/source for the first
  uncovered NOOP row.

## 2026-06-19 — NOOP fallback trace launched

- Launched `TRACE_ACTIONS=1 AZK_JAX_SPLIT_TRACE=1 FAIL_STATIC=1
  verify_vector_fullpool.py 70 m0 m1 m2 m3` to identify the generic NOOP
  fallback row.

## 2026-06-19 — NOOP fallback trace relaunched after resume

- The previous background trace was not available after context resume, so I
  relaunched the same diagnostic:
  `TRACE_ACTIONS=1 AZK_JAX_SPLIT_TRACE=1 FAIL_STATIC=1 verify_vector_fullpool.py 70 m0 m1 m2 m3`.
- Expected output is the first generic NOOP fallback row after the
  `STT02-009` cost-mask fix.

## 2026-06-19 — STT02-009 optional effect skip with queued observer

- The traced m0-m3 run failed after step 33 on generic static `NOOP`.
- The action trace at step 33 had `m1` choosing `NOOP` immediately after
  step 32 `STT02-009` cost selection over a single queued `STT02-010`
  when-returned observer.
- `step_effect_stt02_009_fast` already clears context and begins the queued
  `STT02-010` trigger after an effect skip, but `_effect_stt02_009_fast_mask`
  still rejected every board state with an `STT02-010` observer.
- Relaxed that mask only for the skip row when exactly one queued `STT02-010`
  observer is present; effect-target selection with an observer still stays off
  this fast path.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — m0-m3 chunk recheck after STT02-009 skip fix

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after admitting the single queued-`STT02-010`
  `STT02-009` optional effect skip.

## 2026-06-19 — stale m0-m3 recheck cancelled after residual NOOP exclusion

- The m0-m3 verifier launched before the AZK01-097 residual-static overlap
  cleanup, so I cancelled it rather than using stale code.
- Also excluded `_select_azk01_097_fast_mask` from
  `noop_residual_static_mask_host`; AZK01-097 optional selection declines now
  stay on their source-specific helper instead of also routing to broad static
  `NOOP`.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — m0-m3 chunk recheck after NOOP overlap cleanup

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after the `STT02-009` skip fix and AZK01-097 residual-static
  exclusion.

## 2026-06-19 — m0-m3 advanced to confirmation static fallback

- Recheck after the `STT02-009` skip and AZK01-097 residual-static fixes passed
  step 25, then failed in `_confirm_static_step_fn`.
- This is a broad confirmation decline/accept route, not a semantic mismatch.
- Next run needs action tracing to identify the source card and confirmation row.

## 2026-06-19 — confirmation fallback trace launched

- Launched `TRACE_ACTIONS=1 AZK_JAX_SPLIT_TRACE=1 FAIL_STATIC=1
  verify_vector_fullpool.py 70 m0 m1 m2 m3` to identify the first
  `_confirm_static_step_fn` row.

## 2026-06-19 — STT02-010 confirmation fast path

- Cancelled the confirmation trace after diagnosing the likely row from the
  previous action trace: `STT02-009` skip begins a queued `STT02-010`
  optional draw trigger, and confirming that trigger was routed through broad
  `_confirm_static_step_fn`.
- Added `step_confirm_stt02_010_fast` and vector mask/dispatch for the clean
  main-phase confirmation shape: source `STT02-010` in garden, untapped, owner
  deck non-empty, no cost/effect targets, no combat/redirect/passive work, and
  only supported queued follow-up heads.
- The helper uses `runtime.process_confirm` to tap/draw/clear, then begins the
  next queued `STT02-010`, `STT03-006`, or `STT03-013` trigger when present.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — m0-m3 chunk recheck after STT02-010 confirm fix

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after adding the `STT02-010` confirmation fast path.

## 2026-06-19 — m0-m3 advanced to SELECT_TO_EQUIP static fallback

- Recheck after the `STT02-010` confirmation fast path passed the previous
  confirmation static fallback, then failed in `_select_to_equip_static_step_fn`.
- The failing route is the broad `SELECT_TO_EQUIP` fallback behind
  `_select_stt01_002_equip_static_mask_host`; next trace needs the exact
  selected weapon/host shape so the existing source-specific equip helper can
  be widened safely.

## 2026-06-19 — SELECT_TO_EQUIP fallback trace launched

- Launched `TRACE_ACTIONS=1 AZK_JAX_SPLIT_TRACE=1 FAIL_STATIC=1
  verify_vector_fullpool.py 80 m0 m1 m2 m3` to identify the first
  `_select_to_equip_static_step_fn` row.

## 2026-06-19 — STT01-002 equip of STT01-014

- Trace identified the first `SELECT_TO_EQUIP` static row at m0 step 39:
  action `(22,0,0,0)` with source `STT01-002` and one selected weapon
  `STT01-014`; legal hosts were friendly garden slots `0..3` or leader `5`.
- Existing `step_select_stt01_002_equip_fast` handled simple equips and the
  `STT01-013` on-play confirmation, but the host mask rejected `STT01-014`
  because it has an on-play effect.
- Extended that helper to pop and `resolve_triggered_effect` for a queued
  `STT01-014` on-play trigger after equip, and widened the mask to admit
  `STT01-014` while leaving other on-play weapons on the generic/static path.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — m0-m3 chunk recheck after STT01-014 equip fix

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after adding the `STT01-014` equip/on-play fast handling.

## 2026-06-19 — m3 defender declaration phase mismatch

- Recheck after the `STT01-014` equip/on-play fast handling passed the
  `SELECT_TO_EQUIP` static fallback, then failed semantically at `m3 step 42`.
- Recent action before the mismatch was defender declaration
  `DECLARE_DEFENDER (9,3,0,0)` in response window after an attack.
- C returned to `MAIN`; JAX remained in `RESPONSE_WINDOW`, so the next target is
  the defender-declaration/combat-resolution split path for the m3 state.

## 2026-06-19 — defender declaration auto-close fix

- `step_declare_defender_fast` only tapped/marked the defender and returned,
  so clean declarations with no remaining defender response left JAX stuck in
  `RESPONSE_WINDOW`.
- Added `_close_response_combat_if_idle` after `apply_declare_defender`, matching
  the generic auto-resolve path that transitions through combat resolve when no
  queued effects or response choices remain.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — m0-m3 chunk recheck after defender auto-close

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after adding defender-declaration response auto-close.

## 2026-06-19 — m3 innate Carapace combat mismatch

- Recheck after defender auto-close advanced to a semantic mismatch at
  `m3 step 46`: C kept active player 1 garden slot 3 `AZK01-048` at 2 HP after
  attacking opponent garden slot 4 `STT02-004`; JAX reduced it to 1 HP.
- Initial tapped-target hypothesis was wrong. A longer targeted run exposed an
  earlier regression at m3 step 36 after I zeroed all tapped-target
  counter-damage: C still lets tapped `AZK01-047` deal combat damage.
- Root cause is `AZK01-048`'s innate Carapace 1. The direct entity-combat fast
  path and host mask only checked dynamic `carapace_perm/eot` fields and ignored
  the innate table that generic combat uses via `total_carapace`.
- Updated `step_attack_entity_mutual_destroy_fast` and
  `_attack_entity_mutual_destroy_fast_mask` to subtract innate+dynamic Carapace
  for both combatants; removed the tapped-target counter-damage change.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — targeted m3 Carapace recheck passed

- `TRACE_ACTIONS=1 FAIL_STATIC=1 verify_vector_fullpool.py 47 m3` passed with
  no divergence after the innate-Carapace combat fix.
- This covers both relevant m3 attacks:
  - step 35 tapped `AZK01-047` still deals counter-damage;
  - step 45 attacker `AZK01-048` ignores one point of counter-damage via innate
    Carapace.

## 2026-06-19 — m0-m3 chunk recheck after Carapace fix

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with
  `FAIL_STATIC=1` after the targeted m3 Carapace fix passed.

## 2026-06-19 — stale m0-m3 recheck cancelled for wider Carapace fix

- Cancelled the m0-m3 recheck launched after the first Carapace fix before using
  its result.
- Reason: source audit found other manual split combat helpers still computing
  damage from raw ATK without `total_carapace`, so the in-flight run would be
  stale after the broader correction.

## 2026-06-19 — wider manual combat Carapace fix

- Extended the innate+dynamic Carapace subtraction beyond direct main-phase
  entity combat to the other manual split combat resolvers:
  `step_response_noop_entity_combat_fast` and `step_confirm_azk01_060_fast`.
- Updated the corresponding host masks so death/trigger gates use post-Carapace
  combat damage for `AZK01-048`/`AZK01-109` as well as dynamic Carapace fields.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — targeted m3 recheck after wider Carapace fix

- Relaunched `verify_vector_fullpool.py 47 m3` with `FAIL_STATIC=1` after
  widening the manual combat Carapace fix.

## 2026-06-19 — targeted m3 recheck after wider Carapace passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 47 m3` passed with no divergence after
  the wider manual combat Carapace fix.
- Runtime was dominated by first-use JIT compilation; future verifier runs should
  use the persistent-cache env vars from `jax_env/run_verify.sh`.

## 2026-06-19 — m0-m3 chunk recheck after wider Carapace

- Relaunched `verify_vector_fullpool.py 100 m0 m1 m2 m3` with `FAIL_STATIC=1`
  after the wider Carapace fix and targeted m3 pass.
- This run uses the persistent-cache env vars from `jax_env/run_verify.sh`.

## 2026-06-19 — m0-m3 chunk recheck timed out

- The `verify_vector_fullpool.py 100 m0 m1 m2 m3` recheck after the wider
  Carapace fix timed out at 3600s after printing step 25 ok for all four cases.
- No divergence or fallback was observed before timeout, but the run is not a
  parity pass.
- Next strategy: verify the same frontier as individual cases first, then
  return to batched chunks after more persistent-cache entries exist.

## 2026-06-19 — individual m0 recheck launched

- Launched `verify_vector_fullpool.py 100 m0` with `FAIL_STATIC=1` using the
  persistent-cache env vars.

## 2026-06-19 — m0 advanced to STT03-006 queued-attack fallback

- `verify_vector_fullpool.py 100 m0` with `FAIL_STATIC=1` advanced through
  step 75, then failed on broad `_attack_queued_stt03_006_step_fn`.
- This is a queued `STT03-006` attack-trigger route, not a semantic mismatch.
- Next run needs `TRACE_ACTIONS=1` for m0 to identify the exact step/action and
  current attacker/defender/card shape.

## 2026-06-19 — m0 step 85 inert-attached entity attack

- Trace identified the first broad attack-static row at m0 step 85:
  `ATTACK (6,2,2,0)`.
- C-only replay before the action:
  active player 1 attacks with garden slot 2 `STT01-008` carrying attached
  `STT01-016` and `STT01-013`; defender is player 0 garden slot 2
  `STT01-005`.
- `STT01-016` is only attack-active on a Raizan host; `STT01-008` is not Raizan.
  `STT01-013` has no attack trigger. The direct entity-combat mask was rejecting
  the row solely because these inert weapons were attached to the surviving
  attacker.
- Widened `_attack_entity_mutual_destroy_fast_mask` so attacker attachments
  `STT01-013` and non-Raizan-host `STT01-016` are treated as inert for this
  direct combat fast path. Defender attachments remain rejected.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — individual m0 recheck after inert-attachment fix

- Relaunched `verify_vector_fullpool.py 100 m0` with `FAIL_STATIC=1` after the
  inert attached-weapon attack mask relaxation.

## 2026-06-19 — individual m0 recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m0` passed with no divergence
  after the inert-attached attack mask relaxation.

## 2026-06-19 — individual m1 recheck launched

- Launched `verify_vector_fullpool.py 100 m1` with `FAIL_STATIC=1` using the
  persistent-cache env vars.

## 2026-06-19 — m1 advanced to cost-selection fallback

- `verify_vector_fullpool.py 100 m1` with `FAIL_STATIC=1` passed step 50, then
  failed on broad `_select_cost_static_step_fn`.
- Next run needs `TRACE_ACTIONS=1` for m1 around the post-step-50 cost selection
  row.

## 2026-06-19 — m1 STT02-009 cost with remaining STT02-010 observer

- Trace identified m1 step 60 as `STT02-009` cost selection
  `SELECT_COST_TARGET (13,2,0,0)`.
- C-only replay: `STT02-009` was in player 1 alley, the cost returned player 1
  garden `STT02-010`, and player 0 still had a garden `STT02-010` observer.
  Because an opponent effect target remained, C kept the `STT02-009` effect
  context active, then after effect resolution began the remaining queued
  `STT02-010` confirmation.
- Widened `_select_cost_stt02_009_fast_mask` for exactly this shape:
  returning a `STT02-010` cost target while exactly one other `STT02-010`
  observer remains and the `STT02-009` effect has a target. Also treated
  `STT02-010` as an allowed passive watcher for this cost path.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — individual m1 recheck after STT02-009 cost fix

- Relaunched `verify_vector_fullpool.py 100 m1` with `FAIL_STATIC=1` after the
  `STT02-009` cost-mask relaxation for one remaining `STT02-010` observer.

## 2026-06-19 — m1 STT02-009 effect with queued STT02-010

- Recheck after the m1 cost-mask relaxation advanced to broad
  `_select_effect_static_step_fn`.
- The traced next row is m1 step 61 `SELECT_EFFECT_TARGET (14,0,0,0)` for the
  same `STT02-009`, while exactly one queued/garden `STT02-010` observer remains.
- `step_effect_stt02_009_fast` already pops and begins that queued
  `STT02-010` trigger after effect resolution. The host mask only allowed
  queued `STT02-010` on effect skips, not effect target selection.
- Widened `_effect_stt02_009_fast_mask` to allow select or skip when exactly one
  queued `STT02-010` can be begun, and treated `STT02-010` as an allowed passive
  watcher for this source-specific path.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — individual m1 recheck after STT02-009 effect fix

- Relaunched `verify_vector_fullpool.py 100 m1` with `FAIL_STATIC=1` after the
  queued-`STT02-010` effect-target mask relaxation.

## 2026-06-19 — STT02-009 effect skips invalid returned-card trigger

- Recheck after the m1 effect-mask relaxation reached a semantic mismatch at
  m1 step 62: C transferred active player to player 0 for the remaining
  `STT02-010` confirmation, while JAX stayed on player 1.
- Root cause: `return_to_hand` queues the returned card's own
  `AWhenReturnedToHand` before observer triggers. When the returned cost target
  is `STT02-010`, that first queued head is now in hand and fails validation;
  C skips it and begins the next valid garden `STT02-010` observer. The fast
  helper tried to resolve only the first head.
- Updated `step_effect_stt02_009_fast` to drop an invalid returned-card
  `STT02-010` head and then begin the next queued `STT02-010` observer when
  present.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — individual m1 recheck after invalid-trigger skip fix

- Relaunched `verify_vector_fullpool.py 100 m1` with `FAIL_STATIC=1` after the
  `STT02-009` invalid returned-card trigger skip fix.

## 2026-06-19 — m1 recheck timed out after invalid-trigger skip fix

- The `verify_vector_fullpool.py 100 m1` run after the invalid-trigger skip fix
  timed out at 3600s after step 25 ok, dominated by recompilation of the changed
  STT02-009 effect helper.
- No divergence was observed before timeout, but this is not a pass. Use a
  focused eager replay for the m1 step-61 helper behavior before relaunching the
  longer split verifier.

## 2026-06-19 — focused STT02-009 effect eager replay passed

- Ran a focused eager replay to m1 step 61 and called
  `step_effect_stt02_009_fast` directly on the pre-action JAX state.
- Result matched C after `SELECT_EFFECT_TARGET (14,0,0,0)`:
  active player transferred to player 0, phase stayed MAIN, and the ability
  context began `STT02-010` confirmation.
- This verifies the helper behavior without waiting on GPU JIT compilation.

## 2026-06-19 — individual m1 CPU split recheck launched

- Launched `JAX_PLATFORMS=cpu verify_vector_fullpool.py 100 m1` with
  `FAIL_STATIC=1` to avoid the slow GPU compile path while still verifying the
  split dispatch/masks semantically.

## 2026-06-19 — simplified STT02-009 trigger handoff

- CPU split recheck also timed out during compilation, so compile size is the
  bottleneck, not GPU-only behavior.
- Simplified `step_effect_stt02_009_fast` to call `resolve_triggered_effect`
  only once: it chooses either the first valid `STT02-010` head or, after
  dropping the invalid returned-card head, the second queued observer.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile jax_env/azuki_jax/step.py python/src/azk_puffer/jax_vector.py`.

## 2026-06-19 — simplified STT02-009 focused replay passed

- Re-ran the focused eager replay for m1 step 61 after simplifying the trigger
  handoff.
- Result: active player and ability context again matched C (`active=0`,
  `ab_phase=CONFIRMATION`, source `STT02-010`).

## 2026-06-19 — short m1 split recheck launched

- Launched GPU split `verify_vector_fullpool.py 65 m1` with `FAIL_STATIC=1` to
  verify through the STT02-009/queued-STT02-010 sequence before attempting the
  100-step run again.

## 2026-06-19 — short m1 split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 65 m1` passed with no divergence.
- This covers the STT02-009 cost/effect sequence and queued `STT02-010`
  confirmation handoff that previously failed.

## 2026-06-19 — individual m1 100-step recheck relaunched

- Relaunched `verify_vector_fullpool.py 100 m1` with `FAIL_STATIC=1` after the
  65-step m1 split pass.

## 2026-06-19 — m1 step 91 attack phase mismatch

- `verify_vector_fullpool.py 100 m1` passed through step 75 and then failed
  semantically at m1 step 91: C phase MAIN, JAX phase RESPONSE_WINDOW.
- Recent row was m1 step 90 `ATTACK (6,2,5,0)` from MAIN. Need inspect the
  C state around steps 87-91 and add/relax the correct attack/response
  auto-close path.

## 2026-06-19 — m1 step 91 response false positive diagnosed

- C-only replay around steps 87-91 showed step 90 action `ATTACK (6,2,5,0)`
  should resolve immediately to MAIN: player 1 had no legal response rows after
  the attack.
- Targeted split-mask probe at the pre-step-90 JAX state reported
  `_attack_leader_response_fast_mask == True` and `_attack_leader_simple_fast_mask == False`.
- The false positive was player 1 leader `STT02-001`: host response-board
  detection counted its `[Response][Once/Turn]` ability even though its
  `once_per_turn_used` bit was already set. C `defender_can_respond` checks
  frozen/once/cost/validate before opening the response window.
- Updated attack response/no-response host masks to include frozen and
  once-per-turn gates for board response abilities, and added the
  `ONCE_PER_TURN` table to `JaxVecEnv`.
- Syntax check passed:
  `PYTHONPATH=build/python/src:python/src:jax_env python/.venv-codex/bin/python -m py_compile python/src/azk_puffer/jax_vector.py jax_env/azuki_jax/step.py`.

## 2026-06-19 — m1 step 91 focused mask fix verified

- Re-ran the targeted split replay to the m1 step-90 pre-action state after the
  frozen/once response-board mask change.
- Probe result: `_attack_leader_response_fast_mask == [False]` and
  `_attack_leader_simple_fast_mask == [True]` for `ATTACK (6,2,5,0)`.
- After stepping both engines, C and JAX both reported phase MAIN, active player
  0, and `ab_phase=NONE`. This closes the step-91 phase mismatch.

## 2026-06-19 — m1 step 97 frozen defender entity combat

- `verify_vector_fullpool.py 100 m1` advanced past step 91, then failed the
  `FAIL_STATIC=1` trap at m1 step 97 on `ATTACK (6,3,4,0)`.
- Trace showed active player 1 `STT02-009` attacking player 0 garden slot 4
  `STT02-004`; both were 1/2 entities and combat leaves both at 1 HP.
- Host mask probe showed no response, no triggers, no attachments, clean passive
  state, but `_attack_entity_mutual_destroy_fast_mask` rejected the row because
  the defender had `frozen_dur > 0` from `STT02-014`.
- Decision: frozen status on the defender does not alter incoming combat damage
  or surviving status; keep the attacker-frozen legality guard, but allow frozen
  defenders in the clean entity-combat fast mask.
- Syntax check passed after removing the defender-frozen `no_modifiers` gate.

## 2026-06-19 — m1 step 97 frozen defender damage semantics

- Re-ran `verify_vector_fullpool.py 100 m1`; the row now took the clean
  entity-combat fast path, but step 98 diverged because JAX damaged the frozen
  `STT02-004` defender from 2 HP to 1 HP while C left it at 2 HP.
- Checked `src/utils/combat_util.c`: if the defender has `Frozen`, C deals no
  combat damage by either party. Attack validation only prevents frozen
  attackers.
- Updated `step_attack_entity_mutual_destroy_fast` and its host mask damage
  model to zero both combat damage directions when the defender is frozen.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual m1 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m1` passed with no divergence.
- This covers the prior STT02-009 trigger handoff, the step-91 response
  false-positive fix, and the step-97 frozen-defender combat semantics.

## 2026-06-19 — m2 step 67 AZK01-124 kills STT03-006

- `verify_vector_fullpool.py 100 m2` failed the broad-static trap at step 67:
  `AZK01-124` effect selected enemy garden slot 2, an `STT03-006`.
- C result after the selection: `STT03-006` was destroyed, its when-destroyed
  trigger immediately began, active player transferred to player 0, and
  `ab_phase=EFFECT_SELECTION` for `STT03-006` discard.
- Extended `step_effect_azk01_124_fast` to clear the AZK01-124 context and begin
  a queued `STT03-006` when-destroyed trigger via `resolve_triggered_effect`.
- Relaxed `_effect_azk01_124_fast_mask` to allow lethal `STT03-006` targets
  while still rejecting other when-destroyed targets.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual m2 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m2` passed with no divergence.
- This verifies the AZK01-124 to STT03-006 trigger handoff and the surrounding
  Stonehaven mirror rollout segment.

## 2026-06-19 — m3 step 62 AZK01-128 kills STT03-006

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m3` failed the broad-static trap
  at step 62 on `SELECT_EFFECT_TARGET (14,0,0,0)` for `AZK01-128`.
- C-only replay showed `AZK01-128` destroyed the current attacker, player 0's
  `STT03-006`, then immediately began the destroyed card's when-destroyed
  discard effect with active player 0 and `ab_phase=EFFECT_SELECTION`.
- Extended `step_effect_azk01_128_fast` to clear the response context and begin a
  queued `STT03-006` trigger via `resolve_triggered_effect` before closing the
  response window or auto-resolving combat.
- Relaxed `_effect_azk01_128_fast_mask` to allow the same narrow STT03-006
  when-destroyed target while still rejecting other destroyed-trigger targets.

## 2026-06-19 — individual m3 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m3` passed with no divergence
  after the AZK01-128 to STT03-006 trigger handoff.
- The first post-change compile took roughly 34 minutes; steady-state evidence
  should be gathered with persistent cache reuse before comparing SPS.

## 2026-06-19 — individual m0 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m0` passed with no divergence
  after the m1/m2/m3 fixes, covering the earlier STT01-002 gate handoff and
  inert attachment-mask segments against the current branch state.

## 2026-06-19 — batch m0-m3 100-step probe timed out

- Attempted `FAIL_STATIC=1 verify_vector_fullpool.py 100 m0 m1 m2 m3` after all
  four cases had passed individually.
- Result: timed out at 3600s after reporting `..step 25 ok live=4`; peak observed
  RSS was about 71 GB with heavy CPU-side XLA compilation.
- Decision: continue parity coverage with smaller per-case or fixed small-batch
  shards instead of treating the batch timeout as a semantic failure.

## 2026-06-19 — m4 step 40 AZK01-070 cost timing

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m4` failed at step 40 after
  `ACTIVATE_GARDEN_OR_LEADER_ABILITY (11,4,0,0)` for `AZK01-070` during a
  response window.
- C activation only opened `EFFECT_SELECTION`; the source stayed untapped at 2
  HP until the later target selection. JAX applied the tap and self-damage in
  the activation helper, so it was one ability sub-step ahead.
- Moved `AZK01-070` tap/self-damage cost application from
  `step_activate_azk01_070_fast` into `step_effect_azk01_070_fast`, and changed
  the effect host mask to expect `ab_costs_applied=false` before selection.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m4 step 72 STT03-011 optional confirmation

- Re-ran `FAIL_STATIC=1 verify_vector_fullpool.py 100 m4`; it advanced past
  the AZK01-070 cost-timing fix, then diverged at step 72 after player 0 played
  `STT03-011` to garden slot 4.
- C opens optional `CONFIRMATION` first (`CONFIRM_ABILITY` or `NOOP`) for
  STT03-011's optional on-play destroy. The JAX split play helper skipped
  confirmation and entered `EFFECT_SELECTION` directly.
- Updated `step_play_stt03_011_effect_fast` to enter `AbilityPhase.CONFIRMATION`
  when a valid optional target exists. The observed m4 route declines with
  `NOOP`, which is covered by `step_confirm_clear_fast`.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual m4 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m4` passed with no divergence
  after the AZK01-070 cost-timing and STT03-011 optional-confirmation fixes.

## 2026-06-19 — m5 step 67 STT03-011 confirm accept

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m5` failed the broad
  `_confirm_static_step_fn` trap after step 50.
- C-only replay showed the first trapped confirm was step 67:
  `CONFIRM_ABILITY (16,0,0,0)` accepting player 0 `STT03-011`'s optional
  on-play destroy after the earlier play helper was corrected to open
  `CONFIRMATION`.
- Added `step_confirm_stt03_011_fast`, vector mask/wrapper/JIT/dispatch, and
  excluded it from the broad confirm-static mask. The helper transitions the
  optional confirmation to `EFFECT_SELECTION` without applying effects.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m5 STT03-011 confirm dispatch union fix

- First recheck after adding `step_confirm_stt03_011_fast` still fell through to
  generic `action_type=16` because the new mask was dispatched but omitted from
  the split handled-mask union.
- Added `confirm_stt03_011_mask_host` to the handled union and trace output.
- Syntax check passed again for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m5 STT03-011 actual handled union fix

- Second recheck still fell to generic `action_type=16`: the new mask was in the
  trace handled union but not the actual `handled_mask_host` union near the end
  of `_split_step_by_action`.
- Added `confirm_stt03_011_mask_host` to the actual handled union.
- Syntax check passed again for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m5 STT03-011 post-confirm mandatory effect

- Recheck advanced through the confirm dispatch, then hit a mask mismatch at
  m5 step 68: C legal rows were only `SELECT_EFFECT_TARGET` for STT03-011,
  while JAX also exposed `NOOP`.
- C semantics: the optional choice is consumed by `CONFIRMATION`; after accept,
  the effect target selection is mandatory.
- Updated `step_confirm_stt03_011_fast` to clear `ab_is_optional` and set
  `ab_eff_min=1`, then updated the STT03-011 effect helper/mask to require a
  selected target and reject skip.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual m5 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m5` passed with no divergence
  after the STT03-011 confirm/mandatory-effect fixes.

## 2026-06-19 — individual m6 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m6` passed; the case ended
  before step 69 with no divergence.

## 2026-06-19 — m7 step 14 STT04-004 cost timing

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m7` failed at step 14 after
  player 0 confirmed `STT04-004`'s optional on-play ability.
- C applies STT04-004's sacrifice cost immediately on confirm, leaving the
  source in player 0 discard while the effect target selection is pending.
  JAX left the source in garden until target selection.
- Moved STT04-004 sacrifice cost application from
  `step_effect_stt04_004_fast` to `step_confirm_stt04_004_fast`, set
  `ab_costs_applied=true` after confirmation, and updated the effect helper/mask
  to require the source in discard with costs already applied.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual m7 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m7` passed with no divergence
  after the STT04-004 cost-timing fix.

## 2026-06-19 — m8 step 65 double AZK01-059 combat trigger

- `FAIL_STATIC=1 TRACE_ACTIONS=1 verify_vector_fullpool.py 100 m8` advanced to
  step 65, then hit broad attack-static on `ATTACK (6,4,4,0)`.
- Focused replay showed player 0 `AZK01-059` attacked player 1 tapped
  `AZK01-059`; both took 1 combat damage, survived at 1 HP, and both had valid
  friendly garden buff targets.
- C queues the attacker's `AZK01-059` takes-damage trigger first, opens its
  effect selection, then after that effect is selected transfers active player
  to player 1 for the defender's queued `AZK01-059` trigger.
- Relaxed the clean entity-combat host mask for the narrow double-`AZK01-059`
  surviving trigger shape, and taught `step_effect_azk01_059_fast` to pop/begin
  a queued `AZK01-059` trigger after clearing the current one.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m8 short split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 70 m8` passed with no divergence after
  the double-`AZK01-059` combat-trigger fix.
- This covers the step-65 attack, the attacker's `AZK01-059` effect selection,
  and the handoff into the defender's queued `AZK01-059` trigger.

## 2026-06-19 — m8 step 81 dead AZK01-059 queued NOOP

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m8` then advanced through step
  75 and hit broad `_noop_static_step_fn` at step 81 on `NOOP (0,0,0,0)`.
- Focused replay showed a queued `AZK01-059` takes-damage trigger whose source
  had left garden/alley after step 80 combat. C pops that dead trigger when the
  active player ends the turn, then the `STT04-003` start-each damage still
  applies after EOT attack buffs are cleared.
- Updated the STT04-003 main-NOOP helper/mask to admit exactly one queued dead
  `AZK01-059` trigger, pop it, then run the existing STT04-003 start-each damage
  path.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m8 85-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 85 m8` passed with no divergence after
  the queued-dead-`AZK01-059` main-NOOP fix.
- This covers the step-81 end-turn/start-turn transition, including EOT buff
  cleanup and the `STT04-003` start-each self-damage.

## 2026-06-19 — individual m8 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m8` passed with no divergence; the
  mirror ended before step 96.
- This verifies the double-`AZK01-059` trigger chain and queued-dead-trigger
  NOOP path in the current branch state.

## 2026-06-19 — m9 step 26 AZK01-117 spell cost timing

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m9` reached step 26 with a
  semantic mismatch before the effect-target action: C leader1 stayed at HP 19
  after playing `AZK01-117`, while JAX had already applied the self-damage cost
  and put leader1 at HP 17.
- C replay showed step 25 `PLAY_SPELL_FROM_HAND (8,0,0,0)` discards
  `AZK01-117` and opens `EFFECT_SELECTION`; the 2 self-damage cost is applied
  only when step 26 selects the charge target.
- Updated the `AZK01-117` spell fast play path to defer costs, and the effect
  fast path/mask to apply the self-damage cost immediately before granting
  Charge.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m9 short split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 35 m9` passed with no divergence after
  the `AZK01-117` deferred-cost fix.

## 2026-06-19 — individual m9 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m9` passed with no divergence after
  the `AZK01-117` deferred-cost fix.

## 2026-06-19 — m10 step 49 STT01-002 equip passive timing

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m10` reached step 49 with
  `garden1[2]` attack mismatch after step 48 `SELECT_TO_EQUIP (22,0,2,0)`.
- C replay showed `STT01-002` portal selection equipped `STT01-013` onto
  `STT01-008`; C applied the weapon attack and the `STT01-008` attached-weapon
  passive before opening `STT01-013` optional confirmation, so the host was 4
  attack pre-confirm.
- JAX only had the weapon attack at that boundary. Updated the
  `STT01-002`/selection-to-equip fast path to recompute passives after the
  equip placement and before popping the weapon on-play trigger.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m10 short split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 55 m10` passed with no divergence
  after the selection-equip passive timing fix.

## 2026-06-19 — m10 step 74 AZK01-097 split ownership

- After the m10 equip fix, `FAIL_STATIC=1 verify_vector_fullpool.py 100 m10`
  advanced past step 50 and hit broad `_select_from_selection_static_step_fn`.
- C/JAX inspection at the next boundary showed the state was the supported
  `AZK01-097` weapon pick shape (`SELECT_FROM_SELECTION (18,2,0,0)`), and
  `_select_azk01_097_fast_mask` was true.
- The residual `SELECT_FROM_SELECTION` static mask did not subtract the
  `AZK01-097` fast mask, so both masks owned the action. Added the missing
  exclusion.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual m10 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m10` passed with no divergence
  after the `STT01-002` selection-equip passive timing fix and the
  `AZK01-097` residual-static exclusion.

## 2026-06-19 — m11 step 53 AZK01-070 response with inert AZK01-073

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m11` reached step 53 and hit
  generic action-type fallback for `ACTIVATE_GARDEN_OR_LEADER_ABILITY`.
- C replay showed response-window `AZK01-070` activation from garden slot 1,
  followed by effect selection at step 54. The board contained an enemy
  `AZK01-073` in alley; that passive does not change when `AZK01-070` taps,
  self-damages, or gives an enemy garden entity -1 attack.
- Relaxed the `AZK01-070` response activation/effect masks to treat
  board-resident `AZK01-073` as inert, matching existing masks for other
  localized effects.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m11 60-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 60 m11` passed with no divergence
  after the `AZK01-070` inert-`AZK01-073` mask fix.

## 2026-06-19 — individual m11 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m11` passed with no divergence
  after the `AZK01-070` activation/effect inert-passive mask fix.

## 2026-06-19 — m12 step 95 STT01-017 attached-host target

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m12` reached the late
  response-window `STT01-017` effect and hit broad
  `_effect_stt01_017_static_step_fn` on the second effect target.
- C replay showed the first selected target was an unattached `AZK01-077`; the
  second selected target was a 1-HP `STT01-003` with an attached `STT01-013`.
  C `discard_card` for effect damage resets/discards the host without
  separately discarding its equipped weapon.
- The existing JAX fast effect body already follows that damage/discard shape,
  but the host mask rejected attached targets. Relaxed the `STT01-017` fast
  mask to allow attached garden-entity targets when the target itself is
  otherwise clean.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — m12 97-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 97 m12` passed with no divergence
  after the `STT01-017` attached-target mask fix.

## 2026-06-19 — individual m12 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m12` passed with no divergence
  after the `STT01-017` attached-target mask fix.

## 2026-06-19 — m13 step 50 AZK01-044 leader shock

- `FAIL_STATIC=1 verify_vector_fullpool.py 55 m13` first hit broad
  `_attack_queued_stt03_006_step_fn` at the step-50 attack after
  `STT01-009` equipped `AZK01-044` and attacked the opposing leader.
- C replay showed the attack dealt 4 leader damage and the attached
  `AZK01-044` shock trigger is handled inline by combat damage. The simple
  leader-attack helper was manually applying leader damage but did not call the
  `AZK01-044` shock helper, so the host mask intentionally rejected it.
- Updated `step_attack_leader_simple_fast` to call `_lightning_kanabo` after
  leader combat damage, then removed the explicit `AZK01-044` exclusion from
  `_attack_leader_simple_fast_mask`.

## 2026-06-19 — m13 step 51 alley attack over STT01-009 passive

- After the `AZK01-044` leader-shock fix, m13 advanced through step 50 and hit
  the broad attack static path at step 51: `AZK01-038` attacked an opposing
  alley `STT01-010`.
- The `step_attack_leader_garden_simple_fast` helper already supports alley
  targets and recomputes passives after combat, but its passive-watch mask was
  older than the other attack masks and still treated board `STT01-009` as
  unsafe.
- Relaxed `_attack_leader_garden_simple_fast_mask` to treat the same recompute-
  safe passive watchers as the newer attack masks, including `STT01-009`.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.
- `FAIL_STATIC=1 verify_vector_fullpool.py 55 m13` passed with no divergence.

## 2026-06-19 — m13 step 68 STT01-006 with AZK01-044

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m13` next advanced to step 68
  and hit the broad attack static path on `STT01-006` attacking a garden
  `STT01-009` while equipped with `AZK01-044`.
- The dedicated `STT01-006` attack/effect fast path already opens and resolves
  the when-attacking effect before combat; the remaining blocker was the host
  mask treating `AZK01-044` as unsafe even though combat resolution owns the
  shock trigger.
- Removed the `AZK01-044` exclusion from `_attack_stt01_006_effect_fast_mask`.

## 2026-06-19 — m13 step 70 response combat with AZK01-044

- After the step-68 attack fix, m13 advanced to response-window `NOOP` after
  `STT01-006` dealt its when-attacking effect damage.
- The response entity-combat helper manually resolved combat and did not yet
  apply `AZK01-044` shock. Added `_lightning_kanabo` calls for both combat
  damage directions before damage-event recording, matching `combat_resolve`.
- Relaxed `_response_noop_entity_combat_fast_mask` for already-resolved
  `STT01-006` when-attacking state and for `AZK01-044` attached weapons.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.
- `FAIL_STATIC=1 verify_vector_fullpool.py 75 m13` passed with no divergence.

## 2026-06-19 — individual m13 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m13` passed with no divergence
  after the `AZK01-044` attack/response combat fixes.

## 2026-06-19 — individual m14 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m14` passed with no divergence.

## 2026-06-19 — m15 step 81 AZK01-006 attack compile/parity

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m15` repeatedly timed out after
  step 80 because `_attack_azk01_006_when_attacked_step_fn` still used the
  broad `env_step_static(Act.ATTACK)` JIT.
- Added `step_attack_azk01_006_when_attacked_fast` and rewired the vector
  wrapper to avoid the broad attack compile. For the observed no-response
  shape it declares combat directly into `AZK01-006` optional confirmation,
  saving the attacker as `ab_saved_active` so decline restores active player
  before combat resolution.
- Tightened the host response detector so `AZK01-125` only counts as a response
  ability when `discarded_cards_turn > 0`; otherwise C does not open response.

## 2026-06-19 — m15 step 95 AZK01-011 EOT with inert STT02-012

- After the `AZK01-006` attack fix, m15 advanced to step 95 and hit broad
  `_noop_static_step_fn` on a main-phase pass with a single garden
  `AZK01-011` end-of-turn trigger and an unrelated alley `STT02-012`.
- The dedicated `AZK01-011` EOT noop mask treated `STT02-012` as a blocking
  passive watcher. Added `STT02-012` to that mask's inert passive-watch list.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.
- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m15` passed with no divergence.

## 2026-06-19 — m16 passive-drain attack/gate paths

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m16` first diverged after a
  response-window defender declaration: C had drained pending `STT02-012`
  passive work and showed the garden `STT02-012` as 3/3, while JAX still had
  2/2. Added a `recompute_passives` drain to `step_declare_defender_fast`.
- The next m16 frontier was an `AZK01-004` attack while
  `passive_queue_count=16` and `stt02_012_event_pending` was set. The fast
  helper now drains passives before applying the attack buff, and the mask no
  longer rejects pending passive work for that path.
- The final m16 frontier was an `STT02-002` gate portal while pending
  `STT02-012` passive work existed. `step_gate_portal_simple_fast` now drains
  passives before portaling, and its mask no longer rejects pending
  `STT02-012` events.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.
- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m16` passed with no divergence.

## 2026-06-19 — m17 AZK01-092 selection-to-alley bounce

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 m17` previously diverged after
  step 86: C selected `STT02-008` from `AZK01-092`'s reveal with
  `SELECT_TO_ALLEY`, then showed the card appended to hand and left the alley
  slot empty while entering bottom-deck ordering.
- C-only replay confirmed the source was `AZK01-092` (`ab_source_def=123`), not
  `AZK01-024`; the action row was legal and C's completion hook uses the
  `*_if_still_in_selection` deferred-parent quirk.
- The generic JAX selection processor already models that quirk with
  `_bounce_pick_to_hand`, but the dedicated `step_select_stt02_013_pick_fast`
  split had inlined garden/alley placement without the bounce. Added the bounce
  after placement side effects and refreshed passives from the final hand/board
  layout.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.
- Targeted eager helper check: `AZK01-092` `SELECT_TO_ALLEY` now leaves the
  selected entity in hand, the alley slot empty, `ab_phase=BOTTOM_DECK`, and the
  C-like play counters incremented.

## 2026-06-19 — c0 step 70 STT02-017 over recompute-safe passives

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c0` advanced through step 50 and
  stopped at the broad `STT02-017` spell static path.
- Trace isolated step 70: active player 1 played hand `STT02-017` while the
  opponent board contained return targets `STT01-007` and `STT01-008`, with
  `STT01-011`/`STT02-012` passive watchers elsewhere and a supported
  `STT02-010` return observer in the active garden.
- The fast `STT02-017` helper already returns every low-cost opposing garden
  entity and starts supported `STT02-010` return triggers, but it did not drain
  passives after the mass return and its mask rejected recompute-safe passive
  watchers.
- Added a `recompute_passives` drain after the mass return and relaxed the
  `STT02-017` passive-watch gate for the same deterministic watchers used by
  adjacent split paths (`STT01-008`, `STT01-009`, `STT01-011`, `STT02-012`).
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — c0 step 91 response combat passive drain

- Re-running c0 after the `STT02-017` split fix advanced past step 70 and
  diverged at step 91: C had `STT02-012` back at base 2/2 after combat killed
  a garden entity, while JAX still had the prior +1/+1 passive buff with a
  pending `STT02-012` event.
- Trace showed step 90 used the response-window NOOP fast path that delegates to
  generic `combat_resolve`; that path discarded the dead combatant but did not
  drain passives before returning to main phase.
- Added `recompute_passives` after `combat_resolve` in
  `step_response_noop_combat_fizzle_fast`, matching the dedicated entity-combat
  helper and C's post-combat passive drain.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual c0 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c0` passed with no divergence
  after the `STT02-017` mass-return and response-combat passive-drain fixes.

## 2026-06-19 — c1 step 34 AZK01-124 effect-immune target

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c1` stopped at the broad
  `AZK01-124` effect static path.
- Trace isolated step 34: active player 1 resolved Gate of Devotion's optional
  effect into opposing garden slot 2, an inherent effect-immune `STT02-006`.
- C target validation for `AZK01-124` only requires an enemy garden entity; the
  later `deal_effect_damage` call no-ops on effect immunity. The fast mask was
  stricter than C and rejected any effect-immune target before the helper could
  no-op the damage.
- Relaxed `_effect_azk01_124_fast_mask` so effect-immune targets are accepted
  without the damage-side cleanliness restrictions; non-immune targets still
  keep the existing trigger/godmode/carapace/attachment gates.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual c1 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c1` passed with no divergence
  after the `AZK01-124` effect-immune target mask fix.

## 2026-06-19 — c2 step 38 AZK01-127 combat-destroyed STT03-006

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c2` diverged before step 39:
  C had resolved the second `AZK01-127` response, auto-resolved combat, then
  immediately began the queued destroyed `STT03-006` effect for player 1; JAX
  had resolved combat but left the `STT03-006` trigger queued with
  `active_player=0`.
- Diagnostic replay showed the trigger was not from `AZK01-127` effect damage.
  The response closed after the effect, `combat_resolve` killed both the
  attacker `AZK01-045` and defender `STT03-006`, and C began the defender's
  `STT03-006` draw/discard trigger in the same step.
- Added `_begin_queued_stt03_006_destroy_trigger` and call sites after fast
  response/combat auto-resolution, including the `AZK01-127` response effect
  path after combat fizzle handling. The helper pops only a queue-head
  `TIMING_WHEN_DESTROYED` `STT03-006` while no ability is active, then delegates
  to `resolve_triggered_effect` so the existing triggered-control transfer and
  draw-before-discard semantics stay centralized.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual c2 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c2` passed with no divergence
  after the response/combat `STT03-006` trigger begin fix.

## 2026-06-19 — individual c3 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c3` passed with no divergence.

## 2026-06-19 — individual c4 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c4` passed with no divergence.

## 2026-06-19 — individual c5 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c5` passed with no divergence.

## 2026-06-19 — c6 step 48 AZK01-065 lethal STT04-009 target

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c6` stopped at the broad
  select-effect static path on step 48. The state was `AZK01-065` resolving
  into friendly garden slot 1, a 2-HP `STT04-009`.
- C applies the `AZK01-065` self-damage cost, deals 5 effect damage to
  `STT04-009`, discards the destroyed target, and clears the dead
  `STT04-009` takes-effect-damage trigger without starting its optional effect
  because the source is no longer in garden.
- Added a dead takes-damage fizzle for queue-head `STT04-009` in the
  `AZK01-065` fast effect path, and relaxed the host mask only for lethal,
  clean `STT04-009` targets that will be discarded by the 5 damage.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — individual c6 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c6` passed with no divergence
  after the `AZK01-065`/dead-`STT04-009` trigger-fizzle fix.

## 2026-06-19 — individual c7 100-step split recheck passed

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c7` reached terminal state
  before step 94 with no divergence.

## 2026-06-19 — c8 step 99 AZK01-127 into live STT04-009

- `FAIL_STATIC=1 verify_vector_fullpool.py 100 c8` stopped at the broad
  `AZK01-127` effect static path on step 99.
- The response spell targeted the opposing garden `STT04-009` for 1 effect
  damage. C leaves `STT04-009` alive at 1 HP and immediately begins its
  optional takes-effect-damage trigger, transferring active player to the
  `STT04-009` owner for confirmation.
- Added `STT04-009` queue-head handling to `step_effect_azk01_127_fast` via
  `resolve_triggered_effect`, and relaxed the host mask only for clean,
  nonlethal `STT04-009` targets that take real effect damage and have not used
  their once-per-turn trigger.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.

## 2026-06-19 — c8 AZK01-127 compile-shape tightening

- Two `c8` rechecks reached step 99 and then timed out while compiling the
  modified `AZK01-127` effect helper.
- The first implementation had added a second `resolve_triggered_effect` call
  after `STT04-007` handling. Reworked it to begin queue-head `STT04-009` via
  the existing early `resolve_triggered_effect` call shared with `AZK01-062`,
  reducing duplicated trigger-begin graph size.
- Syntax check passed for `python/src/azk_puffer/jax_vector.py` and
  `jax_env/azuki_jax/step.py`.
