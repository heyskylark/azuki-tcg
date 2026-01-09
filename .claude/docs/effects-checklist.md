# Effects System Implementation Checklist

## Phase 1: Foundation ✅

### 1.1 Update Components
- [x] Update `AbilityPhase` enum in `include/components/components.h`:
  - Add `ABILITY_PHASE_CONFIRMATION = 1`
  - Add `ABILITY_PHASE_COST_SELECTION = 2`
  - Add `ABILITY_PHASE_EFFECT_SELECTION = 3`
- [x] Update `AbilityContext` struct in `include/components/components.h`:
  - Add `bool is_optional`
  - Add `ecs_entity_t owner`
- [x] Add new action types to `ActionType` enum:
  - `ACT_CONFIRM_ABILITY = 16`
  - Move `ACT_MULLIGAN_SHUFFLE` to 17 (must always be highest for `AZK_ACTION_TYPE_COUNT`)
  - Note: `ACT_NOOP` is used to decline optional abilities

### 1.2 Create Ability Registry
- [x] Create `include/abilities/ability_registry.h`:
  - Define `AbilityDef` struct
  - Declare `azk_get_ability_def(CardDefId id)`
  - Declare `azk_has_ability(CardDefId id)`
- [x] Create `src/abilities/ability_registry.c`:
  - Initialize static `kAbilityRegistry[CARD_DEF_COUNT]` array
  - Implement lookup functions

### 1.3 Create ST01-007 Ability Functions
- [x] Create `src/abilities/cards/st01_007.c`:
  - Implement `st01_007_validate()`
  - Implement `st01_007_validate_cost_target()`
  - Implement `st01_007_apply_costs()`
  - Implement `st01_007_apply_effects()`
- [x] Create `include/abilities/cards/st01_007.h`:
  - Declare function prototypes
- [x] Register ST01-007 in `kAbilityRegistry`

---

## Phase 2: Ability System ✅

### 2.1 Create Ability System Core
- [x] Create `include/abilities/ability_system.h`:
  - Declare `azk_trigger_on_play_ability()`
  - Declare `azk_process_ability_confirmation()`
  - Declare `azk_process_ability_decline()`
  - Declare `azk_process_cost_selection()`
  - Declare `azk_process_effect_selection()`
  - Declare `azk_clear_ability_context()`
- [x] Create `src/abilities/ability_system.c`:
  - Implement ability trigger logic
  - Implement phase transition logic
  - Implement cost/effect application

### 2.2 Create Ability Validation
- [x] Validation done inline in `ability_system.c` and via `AbilityDef` function pointers
  - Note: Separate validation file not needed; validation is per-ability via registry

---

## Phase 3: Integration ✅

### 3.1 Integrate with Main Phase
- [x] Modify `src/systems/main_phase.c`:
  - After `summon_card_into_zone_index()`, check for abilities
  - Call `azk_trigger_on_play_ability()` if ability exists
  - Add case handler for `ACT_CONFIRM_ABILITY`
  - Update `ACT_NOOP` handler to use `azk_process_ability_decline()` when in confirmation phase
  - Add case handlers for `ACT_SELECT_COST_TARGET`, `ACT_SELECT_EFFECT_TARGET`

### 3.2 Update Action Enumerator
- [x] Modify `src/validation/action_enumerator.c`:
  - Add `enumerate_ability_actions()` function
  - Generate `ACT_CONFIRM_ABILITY` and `ACT_NOOP` when in confirmation phase
  - Generate `ACT_SELECT_COST_TARGET` options when in cost selection phase
  - Generate `ACT_SELECT_EFFECT_TARGET` options when in effect selection phase
  - Modified `azk_build_action_mask_for_player()` to check ability phase first

### 3.3 Attach Timing Tags
- [x] Timing tags set in `ability_registry.c` during `azk_init_ability_registry()`
  - Registry stores `ecs_id_t timing_tag` per ability

---

## Phase 4: CMake & Build ✅

### 4.1 Update Build
- [x] Update `CMakeLists.txt`:
  - Add `src/abilities/ability_registry.c`
  - Add `src/abilities/ability_system.c`
  - Add `src/abilities/cards/st01_007.c`

---

## Phase 5: Testing ✅

### 5.1 Unit Tests (in `tests/test_world.c`)
- [x] Test `test_ability_registry_lookup()` - verifies registry lookup
- [x] Test `test_st01_007_validate_needs_hand_and_deck()` - tests validation with various board states

### 5.2 Integration Tests (in `tests/test_world.c`)
- [x] Test `test_st01_007_validate_cost_target()` - tests cost target validation
- [x] Test `test_st01_007_ability_flow_confirm_and_execute()` - full flow: trigger → confirm → select cost → verify discard/draw
- [x] Test `test_st01_007_ability_flow_decline()` - decline flow: trigger → ACT_NOOP → verify no changes

---

## Phase 6: Polish & Documentation

### 6.1 Code Cleanup
- [ ] Add inline documentation to new functions
- [ ] Ensure consistent error handling
- [ ] Remove any debug logging

### 6.2 Documentation
- [ ] Update `.codex/docs/effects-and-spells-plan.md` with final implementation notes
- [ ] Document how to add new card abilities

---

## Phase 7: Response Spells (STT02-015)

### 7.1 Core Infrastructure
- [x] Add `ABILITY_TARGET_ANY_GARDEN_ENTITY` to AbilityTargetType enum
- [x] Add `ACT_PLAY_SPELL_FROM_HAND` action spec for PHASE_RESPONSE_WINDOW
- [x] Create `PlaySpellIntent` struct in action_intents.h
- [x] Implement `azk_validate_play_spell_action()` in action_validation.c

### 7.2 Response Window System
- [x] Implement `defender_can_respond()` helper in main_phase.c
- [x] Enable response window transition in `handle_attack()`
- [x] Create `response_phase.c` with `HandleResponseAction()`
- [x] Add combat fizzle check in `combat_resolve_phase.c`
- [x] Add `ABILITY_TARGET_ANY_GARDEN_ENTITY` to action_enumerator.c

### 7.3 Bounce Effect
- [x] Implement `return_card_to_hand()` in card_utils.c

### 7.4 STT02-015 Ability
- [x] Create `stt02_015.c` with validate, validate_effect_target, apply_effects
- [x] Register STT02-015 in ability_registry.c
- [x] Add `azk_trigger_spell_ability()` to ability_system.c

### 7.5 Build Integration
- [x] Wire up response_phase_system in systems/main.c

### 7.6 Bug Fix: Timing Tag Attachment
- [x] Implement `attach_ability_components()` in abilities.c to attach timing tags from registry
- [x] Call `attach_ability_components()` in world.c register_card() after card instantiation

---

## Dependencies

```
Phase 1 (Foundation) ✅
    ↓
Phase 2 (Ability System) ✅
    ↓
Phase 3 (Integration) ✅
    ↓
Phase 4 (Build) ✅
    ↓
Phase 5 (Testing) ✅
    ↓
Phase 6 (Polish) - REMAINING
```

---

## Notes

- Start with the simplest possible implementation
- ST01-007 is a good test case because:
  - It has both cost and effect
  - It's optional (tests confirmation flow)
  - Cost requires target selection (tests selection flow)
  - Effect is automatic (simpler effect handling)
- Once ST01-007 works, the same patterns extend to other cards

---

## Files Created

| File | Purpose |
|------|---------|
| `include/abilities/ability_registry.h` | AbilityDef struct, lookup API |
| `src/abilities/ability_registry.c` | Static registry table |
| `include/abilities/ability_system.h` | Ability processing API |
| `src/abilities/ability_system.c` | Trigger/phase/execution logic |
| `include/abilities/cards/st01_007.h` | ST01-007 function declarations |
| `src/abilities/cards/st01_007.c` | ST01-007 implementations |
| `include/abilities/cards/stt02_015.h` | STT02-015 function declarations |
| `src/abilities/cards/stt02_015.c` | STT02-015 implementations |
| `include/systems/response_phase.h` | Response phase system declarations |
| `src/systems/response_phase.c` | Response window action handler |

## Files Modified

| File | Changes |
|------|---------|
| `include/components/components.h` | Updated `AbilityPhase`, `AbilityContext`, `ActionType` |
| `include/components/abilities.h` | Added `ABILITY_TARGET_ANY_GARDEN_ENTITY` |
| `src/systems/main_phase.c` | Added ability trigger, defender_can_respond, response window transition |
| `src/systems/main.c` | Added init_response_phase_system() call |
| `src/systems/combat_resolve_phase.c` | Added combat fizzle check |
| `src/validation/action_enumerator.c` | Generate ability and spell legal actions, ANY_GARDEN targets |
| `src/validation/action_schema.c` | Added ACT_PLAY_SPELL_FROM_HAND action spec |
| `src/validation/action_validation.c` | Added azk_validate_play_spell_action() |
| `include/validation/action_validation.h` | Declared spell validation function |
| `include/validation/action_intents.h` | Added PlaySpellIntent struct |
| `src/abilities/ability_system.c` | Added azk_trigger_spell_ability(), ANY_GARDEN handling |
| `include/abilities/ability_system.h` | Declared azk_trigger_spell_ability() |
| `src/abilities/ability_registry.c` | Registered STT02-015 ability |
| `src/utils/card_utils.c` | Added return_card_to_hand() |
| `include/utils/card_utils.h` | Declared return_card_to_hand() |
| `src/components/components.c` | Added `azk_init_ability_registry()` call |
| `src/components/abilities.c` | Implemented `attach_ability_components()` to attach timing tags |
| `src/world.c` | Call `attach_ability_components()` when registering cards |
| `tests/test_world.c` | Added 5 new tests for ability system |
| `CMakeLists.txt` | Added new source files |
