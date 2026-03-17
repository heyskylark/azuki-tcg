# C Engine Ability Refactor Checklist

Status: Proposed

Last updated: 2026-03-17

## Goal

Reduce duplication and shrink the high-risk C engine hotspots around:

- ability runtime flow
- targeting and target encoding
- `AbilityContext` sprawl
- repeated selection and bottom-deck flows
- passive observer boilerplate
- ability registration growth as more card sets are added

This plan is intentionally organized to fit the current engine instead of rewriting the whole card system.

## Proposed Target Layout

The current `src/abilities/ability_system.c` and `src/abilities/ability_registry.c` are carrying too many responsibilities. The target shape below keeps the current high-level `abilities/` namespace, but splits runtime concerns into focused modules.

```text
include/
  abilities/
    core/
      ability_runtime.h
      ability_context.h
      ability_flow.h
      ability_triggers.h
      ability_queue.h
    targeting/
      ability_targeting.h
      ability_target_encoding.h
    selection/
      ability_selection.h
      ability_selection_helpers.h
    passive/
      passive_runtime.h
      passive_watchers.h
    registry/
      ability_registry.h
      ability_registry_loader.h
      sets/
        stt01_registry.h
        stt02_registry.h
    cards/
      common/
        reveal_selection.h
        discard_selection.h
      stt01/
        stt01_001.h
        stt01_002.h
        ...
      stt02/
        stt02_001.h
        stt02_002.h
        ...

src/
  abilities/
    core/
      ability_runtime.c
      ability_context.c
      ability_flow.c
      ability_triggers.c
      ability_queue.c
    targeting/
      ability_targeting.c
      ability_target_encoding.c
    selection/
      ability_selection.c
      ability_selection_helpers.c
    passive/
      passive_runtime.c
      passive_watchers.c
    registry/
      ability_registry.c
      ability_registry_loader.c
      sets/
        stt01_registry.c
        stt02_registry.c
    cards/
      common/
        reveal_selection.c
        discard_selection.c
      stt01/
        stt01_001.c
        stt01_002.c
        ...
      stt02/
        stt02_001.c
        stt02_002.c
        ...

  validation/
    action_validation_common.c
    action_validation_ability.c
    action_enumerator_ability.c
```

## Why This Layout

### `abilities/core`

This should own the runtime state machine and phase transitions only.

Current files/functions that map here:

- `src/abilities/ability_system.c`
- `azk_process_ability_confirmation`
- `azk_process_cost_selection`
- `azk_process_effect_selection`
- `azk_process_selection_pick`
- `azk_process_selection_to_alley`
- `azk_process_selection_to_equip`
- `azk_process_bottom_deck`
- `azk_trigger_main_ability`
- `azk_trigger_spell_ability`
- `azk_trigger_leader_response_ability`
- `azk_trigger_gate_portal_ability`
- `azk_process_triggered_effect_queue`

### `abilities/targeting`

This should become the single source of truth for:

- counting legal targets
- enumerating legal targets
- resolving action indices into entities
- maintaining shared index encodings like:
  - enemy garden `0-4`
  - leader at `5`
  - any-garden `0-4 self`, `5-9 enemy`

Current duplicated logic lives in:

- `src/abilities/ability_system.c`
- `src/validation/action_enumerator.c`

### `abilities/selection`

This should own generic selection-zone flows:

- reveal cards into selection
- validate selection candidates
- move picks to hand / alley / equip
- bottom-deck remaining cards
- skip selection

Current duplicated flows live in:

- `src/abilities/ability_system.c`
- `src/abilities/cards/stt01_004.c`
- `src/abilities/cards/stt02_003.c`
- `src/abilities/cards/stt02_013.c`
- `src/abilities/cards/stt01_002.c`

### `abilities/passive`

This should keep passive card logic modular while removing repeated observer lifecycle code.

Current passive scaffolding lives in:

- `src/abilities/cards/stt01_008.c`
- `src/abilities/cards/stt01_009.c`
- `src/abilities/cards/stt01_011.c`
- `src/abilities/cards/stt02_012.c`

### `abilities/registry`

This should separate:

- the registry storage itself
- set-level registration
- active-match loading decisions

Current growth hotspot:

- `src/abilities/ability_registry.c`

## Design Answers

### 1. De-duplicating ability context initialization

Yes. This should be done first.

The repeated setup in:

- `src/abilities/ability_system.c`
  - `azk_trigger_main_ability`
  - `azk_trigger_spell_ability`
  - `azk_trigger_leader_response_ability`
  - `azk_process_triggered_effect_queue`
  - `azk_trigger_gate_portal_ability`

should collapse into a small set of helpers, for example:

```c
bool azk_begin_ability(
  ecs_world_t *world,
  ecs_entity_t source_card,
  ecs_entity_t owner,
  const AbilityDef *def,
  AbilityStartKind start_kind,
  const AbilityStartOptions *opts
);

void azk_reset_ability_context(AbilityContext *ctx);
void azk_init_target_state(AbilityTargetState *state, const AbilityTargetSpec *spec);
bool azk_enter_initial_phase(ecs_world_t *world, AbilityContext *ctx, const AbilityDef *def);
```

The important idea is:

- one helper sets up the context
- one helper computes dynamic target counts
- one helper decides the initial phase
- entry points only supply what is different

Expected end result:

- starting an ability becomes a 5-10 line wrapper instead of 40-90 lines
- triggered, spell, leader, and gate flows stop drifting apart

### 2. Centralizing targeting code

The current duplication between runtime execution and action enumeration should be removed entirely.

The same target model should drive:

1. counting legal targets
2. enumerating action choices
3. resolving selected indices into entities
4. validating that the selected entity is still legal

Recommended shape:

```c
typedef struct {
  int action_index;
  ecs_entity_t entity;
} AbilityTargetChoice;

typedef bool (*AbilityTargetChoiceFn)(
  ecs_world_t *world,
  const AbilityContext *ctx,
  const AbilityDef *def,
  const AbilityTargetChoice *choice,
  void *user_ctx
);

int azk_collect_target_choices(
  ecs_world_t *world,
  const AbilityContext *ctx,
  const AbilityDef *def,
  AbilityTargetKind kind,
  AbilityTargetChoice *out,
  int out_cap
);

bool azk_resolve_target_choice(
  ecs_world_t *world,
  const AbilityContext *ctx,
  const AbilityDef *def,
  AbilityTargetKind kind,
  int action_index,
  ecs_entity_t *out_entity
);
```

Why this is worth it:

- `ability_system.c` and `action_enumerator.c` stop reimplementing the same switch statements
- index encoding bugs only need to be fixed in one place
- adding a new target family becomes one module change, not three

Expected end result:

- `src/validation/action_enumerator.c` becomes thinner
- `src/abilities/ability_system.c` stops containing giant target-resolution switches

### 3. Simplifying `AbilityContext`

The current `AbilityContext` is overloaded. It is mixing:

- invocation metadata
- target buffers
- selection session state
- temporary scratch storage for card-specific behavior

That is why fields are being reused for unrelated meanings:

- `effect_targets[0]` holding the portaled card
- `effect_min` holding a max-cost filter
- selection picks being stored in `effect_targets`

The better model is:

```c
typedef struct {
  uint8_t min_required;
  uint8_t max_allowed;
  uint8_t selected_count;
  ecs_entity_t entities[MAX_ABILITY_SELECTION];
} AbilityTargetState;

typedef enum {
  ABILITY_SCRATCH_NONE = 0,
  ABILITY_SCRATCH_GATE_PORTAL,
  ABILITY_SCRATCH_REVEAL_SELECTION,
  ABILITY_SCRATCH_RETURN_FROM_DISCARD
} AbilityScratchKind;

typedef struct {
  uint8_t count;
  uint8_t picked;
  uint8_t pick_max;
  ecs_entity_t cards[MAX_SELECTION_ZONE_SIZE];
} AbilitySelectionState;

typedef struct {
  AbilityScratchKind kind;
  union {
    struct {
      ecs_entity_t portaled_card;
    } gate_portal;
    struct {
      uint8_t cost_limit;
      uint8_t reveal_count;
    } reveal;
    struct {
      uint8_t max_cost;
    } discard_pick;
  } data;
} AbilityScratchState;

typedef struct {
  AbilityPhase phase;
  ecs_entity_t source_card;
  ecs_entity_t owner;
  bool is_optional;
  bool restores_active_player;
  int8_t saved_active_player_index;
  AbilityTargetState cost;
  AbilityTargetState effect;
  AbilitySelectionState selection;
  AbilityScratchState scratch;
} AbilityContext;
```

The key rule should be:

- stable runtime fields always mean the same thing
- temporary card-specific data must go through a tagged scratch union
- card files should stop overloading unrelated fields

Optimization note:

This is more about correctness and maintainability than CPU time. The current singleton is already small enough. The optimization win comes from:

- fewer branches based on overloaded fields
- fewer special-case card hacks
- less duplicated post-selection cleanup logic

### 4. De-dupping card selection and bottom-deck flows

Yes. The three reveal-and-pick cards should move to shared helpers.

Current family:

- `src/abilities/cards/stt01_004.c`
- `src/abilities/cards/stt02_003.c`
- `src/abilities/cards/stt02_013.c`

These all do some combination of:

- `look_at_top_n_cards`
- copy cards into `ctx->selection_cards`
- count legal candidates
- enter `ABILITY_PHASE_SELECTION_PICK` or `ABILITY_PHASE_BOTTOM_DECK`
- move selected cards to hand if still in selection
- bottom-deck the rest

Recommended approach:

- introduce a generic reveal-selection helper
- keep the card-specific predicate small
- keep the card-specific destination policy small

Example shape:

```c
typedef bool (*AzkSelectionPredicate)(
  ecs_world_t *world,
  ecs_entity_t source_card,
  ecs_entity_t owner,
  ecs_entity_t target,
  const AbilityScratchState *scratch
);

typedef enum {
  AZK_SELECTION_DEST_HAND = 0,
  AZK_SELECTION_DEST_ALLEY_OR_HAND,
  AZK_SELECTION_DEST_EQUIP_ONLY
} AzkSelectionDestinationMode;

bool azk_begin_reveal_selection(
  ecs_world_t *world,
  AbilityContext *ctx,
  uint8_t reveal_count,
  uint8_t pick_max,
  AzkSelectionPredicate predicate
);

void azk_finish_selection_with_bottom_deck(
  ecs_world_t *world,
  AbilityContext *ctx,
  AzkSelectionDestinationMode mode
);
```

Expected end result:

- each of those card files becomes mostly predicate logic plus special destination behavior
- bottom-deck completion logic lives in one place

### 5. Simplifying passive observers while keeping them modular

Yes. The right split is:

- common observer lifecycle in shared runtime code
- card-specific recompute logic kept local and modular

That preserves support for very different card mechanics and future decks, but removes repetitive setup/cleanup code.

Recommended passive model:

1. shared runtime helper owns:
   - context allocation and storage
   - observer registration wrappers
   - observer deletion
   - `PassiveObserverContext` storage
   - optional default cleanup for buffs
2. each card file owns:
   - what events it watches
   - how to recompute desired state
   - what buffs or effects it applies/removes

Concretely, passive card files should shrink toward:

```c
static void stt02_012_recompute(ecs_world_t *world, ecs_entity_t card,
                                const Stt02012PassiveCtx *ctx);

void stt02_012_install_passive(ecs_world_t *world, ecs_entity_t card) {
  azk_passive_init(world, card, sizeof(Stt02012PassiveCtx), ...);
  azk_passive_watch_zone(world, card, player_garden, EcsOnAdd | EcsOnRemove, ...);
  azk_passive_watch_zone(world, card, opponent_garden, EcsOnAdd | EcsOnRemove, ...);
}
```

This is modular enough for more decks because:

- the card logic still lives per card
- common helpers only remove boilerplate
- you can still add fully custom passive cards when needed

Where this likely lands:

- `src/abilities/passive/passive_runtime.c`
- `src/abilities/passive/passive_watchers.c`
- card files keep custom recompute functions

### 6. Registry layout and only loading what matters

Splitting the registry by set absolutely makes sense.

Recommended direction:

- keep one registry storage table keyed by `CardDefId`
- split registration into per-set modules:
  - `src/abilities/registry/sets/stt01_registry.c`
  - `src/abilities/registry/sets/stt02_registry.c`
- add a loader that can register sets or individual cards on demand

Possible API:

```c
void azk_register_ability_set_stt01(AbilityRegistry *registry);
void azk_register_ability_set_stt02(AbilityRegistry *registry);
void azk_ensure_ability_registered(CardDefId id);
void azk_register_abilities_for_deck(const CardInfo *cards, size_t count);
```

Important caveat:

The current heavyweight part is not just ability registration. `azk_register_card_def_resources()` in generated code currently creates every prefab up front. That means:

- ability registry splitting is good for readability immediately
- lazy ability loading alone gives only modest performance benefit
- true "only load what active decks use" needs the generated prefab registration path to support per-set or per-card lazy registration too

That means this effort should be aligned with:

- `scripts/generate_card_defs.py`
- `include/generated/card_defs.h`
- `src/generated/card_defs.c`

Do not hand-edit generated outputs. If prefab loading becomes lazy, it should be done through the generator.

Pragmatic recommendation:

1. split registry by set now for readability and growth control
2. keep runtime ability attachment per instantiated card
3. only pursue lazy prefab creation if startup time or memory becomes a measured issue

Also note:

- passive observers are already instantiated only for live card instances in `src/world.c`
- so inactive cards are not paying observer runtime cost today

## Priority Checklist

## P0: Extract Ability Runtime Bootstrap And Phase Flow

Objective:

- remove repeated ability startup logic
- make every ability entry point use the same context initialization and initial-phase rules

Files and functions to touch:

- `src/abilities/ability_system.c`
  - `azk_trigger_main_ability`
  - `azk_trigger_spell_ability`
  - `azk_trigger_leader_response_ability`
  - `azk_process_triggered_effect_queue`
  - `azk_trigger_gate_portal_ability`
  - `azk_clear_ability_context`
- `include/abilities/ability_system.h`
- `include/components/components.h`
- new:
  - `src/abilities/core/ability_context.c`
  - `src/abilities/core/ability_flow.c`
  - `src/abilities/core/ability_runtime.c`

Detailed work:

- introduce shared helpers for clearing and initializing `AbilityContext`
- introduce one place that decides initial phase:
  - confirmation
  - cost selection
  - effect selection
  - selection flow
  - immediate resolve
- remove duplicated zeroing of `cost_targets` and `effect_targets`
- move active-player control handoff into trigger/runtime helpers

Expected end result:

- `ability_system.c` shrinks substantially
- all ability entry points share one consistent start path
- future ability types stop copying start logic

## P0: Build One Targeting Engine Used By Runtime And Enumeration

Objective:

- eliminate duplicated target counting, encoding, and resolution logic

Files and functions to touch:

- `src/abilities/ability_system.c`
  - `count_available_cost_targets`
  - `count_available_effect_targets`
  - cost target switch in `azk_process_cost_selection`
  - effect target switch in `azk_process_effect_selection`
- `src/validation/action_enumerator.c`
  - `enumerate_ability_actions`
- `include/components/abilities.h`
- new:
  - `src/abilities/targeting/ability_targeting.c`
  - `src/abilities/targeting/ability_target_encoding.c`
  - `include/abilities/targeting/ability_targeting.h`

Detailed work:

- define a single encoded-choice model for every `AbilityTargetType`
- implement shared APIs to:
  - count legal choices
  - enumerate legal choices
  - resolve encoded choice to entity
- have action enumeration call the shared enumerator
- have runtime selection call the shared resolver

Expected end result:

- one switch per target family instead of several duplicated ones
- target encoding bugs fixed once
- new target types become much easier to add

## P1: Redesign `AbilityContext` Into Stable Runtime State Plus Typed Scratch

Objective:

- stop using unrelated fields as temporary card-specific storage

Files and functions to touch:

- `include/components/components.h`
  - `AbilityContext`
- `src/abilities/ability_system.c`
- `src/abilities/cards/stt01_002.c`
- any card file currently using generic fields for special meanings
- observation readers:
  - `src/utils/observation_util.c`
  - `src/utils/training_observation_util.c`
  - `src/utils/v2/training_observation_util.c`
- CLI/debug readers if kept:
  - `src/utils/cli_rendering_util.c`

Detailed work:

- replace flat `cost_min`, `cost_expected`, `cost_filled` style fields with grouped target state structs
- separate selection state from effect targets
- add a tagged `scratch` union for generic temporary data
- audit cards that currently overload:
  - `effect_targets`
  - `effect_min`
  - selection arrays

Expected end result:

- card callbacks stop relying on field reuse tricks
- observation code becomes clearer because each context field has one meaning
- ability bugs become easier to reason about

## P1: Extract Generic Selection And Bottom-Deck Helpers

Objective:

- remove repeated reveal-selection and cleanup logic from card files

Files and functions to touch:

- `src/abilities/cards/stt01_004.c`
- `src/abilities/cards/stt02_003.c`
- `src/abilities/cards/stt02_013.c`
- `src/abilities/cards/stt01_002.c`
- `src/abilities/ability_system.c`
  - `azk_process_selection_pick`
  - `azk_process_selection_to_alley`
  - `azk_process_selection_to_equip`
  - `azk_process_skip_selection`
  - `azk_process_bottom_deck`
  - `azk_process_bottom_deck_all`
- new:
  - `src/abilities/selection/ability_selection.c`
  - `src/abilities/selection/ability_selection_helpers.c`
  - `src/abilities/cards/common/reveal_selection.c`

Detailed work:

- extract helpers for:
  - reveal top N
  - store selection candidates
  - count valid picks
  - move chosen cards to final destination
  - bottom-deck remainder
- centralize "remaining cards" counting logic
- keep only card-specific predicates and destination modes in card files

Expected end result:

- selection-heavy cards shrink noticeably
- bottom-deck behavior is consistent across abilities
- adding future tutor / reveal / pick effects is faster

## P2: Create Shared Passive Observer Runtime Helpers

Objective:

- keep passive cards modular without repeating observer lifecycle boilerplate

Files and functions to touch:

- `src/abilities/cards/stt01_008.c`
- `src/abilities/cards/stt01_009.c`
- `src/abilities/cards/stt01_011.c`
- `src/abilities/cards/stt02_012.c`
- `include/components/abilities.h`
  - `PassiveObserverContext`
- possibly `src/components/abilities.c`
- new:
  - `src/abilities/passive/passive_runtime.c`
  - `src/abilities/passive/passive_watchers.c`

Detailed work:

- extract helpers for:
  - allocating per-card passive ctx
  - registering observers
  - storing observer ids
  - deleting observers
  - optional default cleanup of attack/health buffs
- leave recompute logic in each card file
- optionally add reusable watcher helpers for common event families:
  - zone occupancy / threshold
  - discard threshold
  - weapon attachment

Expected end result:

- passive card files focus on their rule logic
- adding new passive cards does not require rewriting the same install/cleanup code
- future deck-specific passive cards remain easy to add

## P2: Split Registry By Set And Add Loader Layer

Objective:

- stop growing one monolithic ability registry file
- make future sets and partial loading manageable

Files and functions to touch:

- `src/abilities/ability_registry.c`
- `include/abilities/ability_registry.h`
- `src/components/components.c`
  - `azk_init_ability_registry`
- `src/world.c`
  - deck initialization path if loader becomes deck-aware
- generator-related files if prefab loading is later aligned:
  - `scripts/generate_card_defs.py`
  - generated headers and sources

Detailed work:

- move set-specific registration into per-set files
- keep a small central loader that calls the needed set registrars
- optionally register only sets present in the match decks
- if worthwhile later, align prefab registration with the same set boundaries

Expected end result:

- cleaner registry diff when new sets are added
- much smaller per-set files
- eventual path toward loading only the card families used by a match

## P3: Clean Up Dead Ability Abstractions And Cross-Module Duplication

Objective:

- remove unused ability concepts and reduce misleading surface area

Files and functions to touch:

- `include/components/abilities.h`
  - `AbilityFunctions`
  - unused tags if confirmed dead
- `src/components/abilities.c`
- `src/validation/action_enumerator.c`
- `src/validation/action_validation.c`
- `src/utils/status_util.c`

Detailed work:

- remove unused or abandoned abstractions after runtime refactor settles
- review tags that are declared but barely used
- split validation/enumerator ability-specific logic into smaller files
- consider deduping attack/health buff lifecycle in `status_util.c`

Expected end result:

- fewer fake extension points
- smaller headers
- less confusion about which abstraction is actually authoritative

## Deferred / Not Part Of This Plan

- `src/utils/v2/training_observation_util.c` replacing v1 is a separate migration track
- `src/utils/cli_rendering_util.c` cleanup or removal is not part of the ability refactor, unless observation structs are changed and require minimal compatibility updates

## Recommended Execution Order

1. P0 runtime bootstrap and flow extraction
2. P0 target model centralization
3. P1 `AbilityContext` redesign
4. P1 shared selection helpers
5. P2 passive runtime helpers
6. P2 registry split by set
7. P3 dead code and validation cleanup

## Discrete Task Breakdown

### Finish P0 Runtime Bootstrap And Phase Flow

1. Extract `azk_enter_initial_phase(...)` so phase selection happens in one
   place.
2. Move any remaining target-state reset logic out of entry points and into
   shared runtime/context helpers.
3. Move active-player handoff and restore behavior into the same shared runtime
   path.
4. Convert each entry point to the shared path one by one:
   - `azk_trigger_main_ability`
   - `azk_trigger_spell_ability`
   - `azk_trigger_leader_response_ability`
   - `azk_process_triggered_effect_queue`
   - `azk_trigger_gate_portal_ability`
5. Add regression coverage for each start path so they enter the same phase as
   before the refactor.

### P0 Target Model Centralization

1. Define one encoded target-choice model in a new targeting module.
2. Implement shared "count legal targets" APIs for each `AbilityTargetType`.
3. Implement shared "enumerate legal choices" APIs.
4. Implement shared "resolve action index to entity" APIs.
5. Replace runtime cost-target resolution in `src/abilities/ability_system.c`.
6. Replace runtime effect-target resolution in `src/abilities/ability_system.c`.
7. Replace ability action enumeration in `src/validation/action_enumerator.c`.
8. Add cross-check tests so runtime and enumeration produce the same legal
   choices.

### Suggested PR Boundaries

1. Shared initial-phase helper plus one migrated trigger path
2. Migrate all remaining trigger/start paths
3. Target-choice model plus counting/enumeration helpers
4. Runtime target resolver migration
5. Action enumerator migration plus regression tests

## Short Recommendation

If only one thing gets done first, do this:

- centralize ability startup
- centralize targeting

Those two changes will remove the most duplication and make every later cleanup safer.
