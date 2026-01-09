# Effects System Implementation Plan

## Overview

This document describes the implementation plan for card effects/abilities in the Azuki TCG engine. We're starting with a single card (ST01-007) to establish patterns before expanding.

## Reference Card: ST01-007 "Alley Guy"

**Card Text**: "On Play; You may discard 1:Draw 1"

**Breakdown**:
- **Trigger**: `AOnPlay` - activates when entity enters garden or alley
- **Optional**: "You may..." means player can decline the ability
- **Cost**: Discard 1 card from hand (`ABILITY_TARGET_FRIENDLY_HAND`, min=1, max=1)
- **Effect**: Draw 1 card from deck (no target selection, auto-resolves)

---

## Architecture Decisions

### 1. Ability Lifecycle

Abilities follow a state machine with these phases:

```
TRIGGER_DETECTED
    ↓
VALIDATION (can ability be activated?)
    ↓
CONFIRMATION (for optional abilities)
    ↓
COST_SELECTION (player selects cost targets)
    ↓
COST_APPLICATION (mutate game state for cost)
    ↓
EFFECT_SELECTION (player selects effect targets, if needed)
    ↓
EFFECT_APPLICATION (mutate game state for effect)
    ↓
CLEANUP (clear AbilityContext, return to previous phase)
```

### 2. AbilityContext Singleton

The existing `AbilityContext` in `components.h` tracks ability state:

```c
typedef struct {
  AbilityPhase phase;              // Current sub-phase
  ecs_entity_t source_card;        // Card whose ability is being processed
  uint8_t cost_min, effect_min;    // Minimum targets required
  uint8_t cost_expected, effect_expected;  // Expected targets
  uint8_t cost_filled, effect_filled;      // Targets selected so far
  ecs_entity_t cost_targets[MAX_ABILITY_SELECTION];
  ecs_entity_t effect_targets[MAX_ABILITY_SELECTION];
} AbilityContext;
```

**Additions needed**:
- `bool is_optional` - whether ability can be declined
- `ecs_entity_t owner` - player who owns the ability source

### 3. Ability Definition Storage

Using a **static lookup table** indexed by `CardDefId`:

```c
// src/abilities/ability_registry.c
typedef struct {
    bool has_ability;
    bool is_optional;
    AbilityCostRequirements cost_req;
    AbilityEffectRequirements effect_req;
    ecs_id_t timing_tag;  // AOnPlay, AStartOfTurn, etc.

    // Function pointers
    bool (*validate)(ecs_world_t*, ecs_entity_t card, ecs_entity_t owner);
    bool (*validate_cost_target)(ecs_world_t*, ecs_entity_t target);
    bool (*validate_effect_target)(ecs_world_t*, ecs_entity_t target);
    void (*apply_costs)(ecs_world_t*, const AbilityContext*);
    void (*apply_effects)(ecs_world_t*, const AbilityContext*);
} AbilityDef;

static const AbilityDef kAbilityRegistry[CARD_DEF_COUNT] = {
    [CARD_DEF_STT01_007] = {
        .has_ability = true,
        .is_optional = true,
        .cost_req = { .type = ABILITY_TARGET_FRIENDLY_HAND, .min = 1, .max = 1 },
        .effect_req = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
        .timing_tag = AOnPlay,
        .validate = st01_007_validate,
        .validate_cost_target = st01_007_validate_cost_target,
        .validate_effect_target = NULL,
        .apply_costs = st01_007_apply_costs,
        .apply_effects = st01_007_apply_effects,
    },
    // Other cards default to { .has_ability = false }
};
```

**Rationale**: This approach is simple, explicit, and doesn't require runtime component attachment. Each card's ability logic is self-contained in its own functions.

### 4. Timing Tag Application

When cards are instantiated from prefabs, we attach timing tags based on the registry:

```c
void attach_ability_tags(ecs_world_t* world, ecs_entity_t card) {
    const CardId* card_id = ecs_get(world, card, CardId);
    const AbilityDef* def = &kAbilityRegistry[card_id->id];

    if (!def->has_ability) return;

    // Add timing tag (AOnPlay, AStartOfTurn, etc.)
    if (def->timing_tag) {
        ecs_add_id(world, card, def->timing_tag);
    }
}
```

### 5. Integration Points

**After entity placement** (`summon_card_into_zone_index`):
```c
// Check for AOnPlay abilities
if (ecs_has(world, card, AOnPlay)) {
    trigger_on_play_ability(world, card, owner);
}
```

**At start of turn** (in `StartPhase`):
```c
// Query all cards in garden/alley with AStartOfTurn
// Trigger each ability in order
```

---

## New Action Types

Add to `ActionType` enum in `components.h`:

```c
ACT_CONFIRM_ABILITY = 16,      // Accept optional ability
ACT_SELECT_COST_TARGET = 13,   // Already exists
ACT_SELECT_EFFECT_TARGET = 14, // Already exists
ACT_MULLIGAN_SHUFFLE = 17,     // Move to end (must always be highest for AZK_ACTION_TYPE_COUNT)
```

**Note**: `ACT_NOOP` is used to decline optional abilities. No separate decline action needed.

---

## New AbilityPhase Values

Update `AbilityPhase` enum:

```c
typedef enum {
  ABILITY_PHASE_NONE = 0,
  ABILITY_PHASE_CONFIRMATION = 1,      // Waiting for confirm/decline
  ABILITY_PHASE_COST_SELECTION = 2,    // Selecting cost targets
  ABILITY_PHASE_EFFECT_SELECTION = 3,  // Selecting effect targets
} AbilityPhase;
```

---

## ST01-007 Implementation Details

### File: `src/abilities/st01_007.c`

```c
// "On Play; You may discard 1:Draw 1"

bool st01_007_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);

    // Need at least 1 other card in hand to discard
    ecs_entity_t hand = gs->zones[player_num].hand;
    ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
    if (hand_cards.count < 1) return false;  // No cards to discard

    // Need at least 1 card in deck to draw
    ecs_entity_t deck = gs->zones[player_num].deck;
    ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
    if (deck_cards.count < 1) return false;  // Nothing to draw

    return true;
}

bool st01_007_validate_cost_target(ecs_world_t* world, ecs_entity_t target) {
    // Target must be a card in the owner's hand
    // (Full validation happens in the selection handler)
    return target != 0;
}

void st01_007_apply_costs(ecs_world_t* world, const AbilityContext* ctx) {
    // Discard the selected card
    ecs_entity_t to_discard = ctx->cost_targets[0];
    discard_card(world, to_discard);
}

void st01_007_apply_effects(ecs_world_t* world, const AbilityContext* ctx) {
    // Draw 1 card
    ecs_entity_t owner = ctx->owner;
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);

    ecs_entity_t deck = gs->zones[player_num].deck;
    ecs_entity_t hand = gs->zones[player_num].hand;

    move_cards_to_zone(world, deck, hand, 1, NULL);
}
```

---

## Execution Flow for ST01-007

### Step-by-Step

1. **Player action**: `ACT_PLAY_ENTITY_TO_GARDEN` with ST01-007
2. **Main phase system** validates and executes `summon_card_into_zone_index()`
3. **Post-play hook** checks: `ecs_has(world, card, AOnPlay)` → true
4. **Lookup ability**: `kAbilityRegistry[CARD_DEF_STT01_007]`
5. **Validate ability**: `st01_007_validate()` → returns true if hand has cards and deck has cards
6. **Check optional**: `def->is_optional == true`
7. **Enter confirmation phase**:
   - Set `AbilityContext.phase = ABILITY_PHASE_CONFIRMATION`
   - Set `AbilityContext.source_card = card`
   - Set `AbilityContext.owner = player`
   - Game waits for next action

8. **Player action**: `ACT_CONFIRM_ABILITY` (or `ACT_NOOP` to decline)
9. **If confirmed, enter cost selection**:
   - Set `AbilityContext.phase = ABILITY_PHASE_COST_SELECTION`
   - Set `AbilityContext.cost_expected = 1`
   - Game waits for target selection

10. **Player action**: `ACT_SELECT_COST_TARGET` with `subaction_1 = hand_index`
11. **Validate and store target**:
    - Validate: card exists, in player's hand, not the ability source
    - Store in `AbilityContext.cost_targets[0]`
    - Increment `cost_filled`

12. **Cost selection complete** (cost_filled == cost_expected):
    - Call `st01_007_apply_costs()` → discards selected card

13. **Effect selection** (skipped, effect_req.type == NONE)

14. **Apply effects**:
    - Call `st01_007_apply_effects()` → draws 1 card

15. **Cleanup**:
    - Clear `AbilityContext`
    - Return to `PHASE_MAIN`

---

## File Structure

```
src/
├── abilities/
│   ├── ability_registry.h    # AbilityDef struct, lookup functions
│   ├── ability_registry.c    # Static registry table
│   ├── ability_system.c      # Phase handlers, trigger logic
│   └── cards/
│       └── st01_007.c        # ST01-007 specific functions
├── systems/
│   └── main_phase.c          # Modified to check for triggered abilities
└── validation/
    └── ability_validation.c  # Validate cost/effect target selections
```

---

## Open Questions (To Resolve During Implementation)

1. **Timing of cost payment**: Should cost be paid before or after the card enters play?
   - Current assumption: After (card is on board, then ability triggers)

2. **Multiple abilities on one card**: How to handle cards with multiple abilities?
   - Future consideration: Array of AbilityDef per card

3. **Ability stacking**: What if multiple AOnPlay abilities trigger simultaneously?
   - Future consideration: Queue system with player choice for ordering

4. **Fizzling**: If deck becomes empty between cost payment and effect, does ability fizzle?
   - Current assumption: Effect tries to draw, gets 0 cards, no error

---

## Testing Strategy

1. **Unit test**: `st01_007_validate()` returns correct results for various board states
2. **Integration test**: Full flow from play action to draw completion
3. **Edge cases**:
   - Empty hand (can't pay cost)
   - Empty deck (can't draw)
   - Only 1 card in hand (that card is the one being played? No - it's already on board)
   - Player declines optional ability

---

## Implementation Checklist

See `effects-checklist.md` for the step-by-step implementation tasks.
