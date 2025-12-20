# Effects and Spells Plan

## Keyword Abilities

Charge:
- can attack on the turn it enters the garden (won't be put into cooldown).
- Does this mean that tapped cards can also be untapped?

Defender:
- can be tapped to redirect an attack to the defender during the response window.

Immune to Effects
- cannot be damaged by card effects

Applied as a tag CHARGE or DEFENDER.
If it is short lived, we'll apply a keyword ability countdown component:
Tag removed at the start of the turn when the duration is 0. Any countdown this is -1 is permanent.
```c
typedef struct {
  charge_duration: int8_t;
  defender_duration: int8_t;
} KeywordAbilityCountdown;
```

## Card Conditions

Frozen:
Card cannot attack or be damaged

Shocked:
Card cannot be untapped while this effect is active.

Applied as a tag FROZEN or SHOCKED.
If it is short lived, we'll apply a card condition countdown component:
Tag removed at the start of the turn when the duration is 0. Any countdown this is -1 is permanent.
```c
typedef struct {
  frozen_duration: int8_t;
  shocked_duration: int8_t;
} CardConditionCountdown;
```

## Ability Structues

### Ability Context

Card effects and spells will utilize an ability context singleton to track cost selections and effect selections.

### Ability Defs

Ability definitions should be applied as a component to the respective card defs.
The ability defs should be searchable by a map of the card id during runtime build.

```c
typedef void (*init_observer)(ecs_world_t,ecs_entity_t,ecs_entity_t);

// For example, observing when a card is added to the garden or a card is added to the owners discard pile.
// Need to figure out how to have observered effects know if they already applied the effect or not each time it's triggered.
void init_observer(ecs_world_t world, ecs_entity_t card, ecs_entity_t owner) {
  ecs_observer(world, {
    .filter = {
        .terms = {
            // exact entity match: only `card`
            { .id = card },
            // exact parent match: only garden_zone
            { .id = ecs_pair(EcsChildOf, garden_zone) },
        }
    },
    .events = { EcsOnAdd },
    .callback = 
  });

  const ecs_entity_t discard_zone = get_owner_zone(world, owner, ZDiscard);
  ecs_observer(world, {
    .filter = {
        .terms = {
            { .id = ecs_pair(EcsChildOf, discard_zone) },
        }
    },
    .events = { EcsOnAdd },
    .callback =
  });
}

// When no observer is needed, we can use this noop function.
void noop(ecs_world_t world, ecs_entity_t card, ecs_entity_t owner) { (void)world; (void)card; (void)owner; }

typedef enum {
  MAIN = 0,
  RESPONSE = 1,
  ON_PLAY = 2,
  WHEN_ATTACKING = 3,
  OBSERVED = 4, // Triggered by observed events
} AbilityKind;

typedef enum {
  NONE = 0,
  SELF = 1,
  FRIENDLY_HAND = 2,
  FRIENDLY_IKZ = 3,
  FRIENDLY_GARDEN_ENTITY = 4,
  FRIENDLY_ALLEY_ENTITY = 5,
  FRIENDLY_ENTITY_WITH_WEAPON = 6,
  FRIENDLY_LEADER = 7,
  ENEMY_GARDEN_ENTITY = 8,
  ENEMY_LEADER = 9,
  ENEMY_LEADER_OR_GARDEN_ENTITY = 10,
  ANY_LEADER_OR_GARDEN_ENTITY = 11,
} AbilityTargetType;

typedef struct {
  AbilityTargetType: type;
  uint8_t min;
  uint8_t max;
} AbilityTargetRequirements;

typedef struct {
  CardDefId: card_id;
  const char *name;
  AbilityKind: kind;
  is_garden_only: bool;
  is_alley_only: bool;
  is_optional: bool;
  is_once_per_turn: bool;
  can_bypass_cooldown: bool;
  was_applied: bool;
  AbilityTargetRequirements: cost_requirements;
  AbilityTargetRequirements: effect_requirements;
  void (*init_observer)(ecs_world_t,ecs_entity_t,ecs_entity_t);
  void (*validate_cost)(uint8_t); // validate then pass target to AbilityContext
  void (*apply_all_costs)(); // Takke all targets from ability context, run through validate to get intents, apply mutations
  void (*validate_effect)(uint8_t); // Same as cost routine
  void (*apply_all_effects)();
} AbilityDef;

typedef struct {
  void (*init_observer)(ecs_world_t,ecs_entity_t,ecs_entity_t);
  bool (*validate_all)(ecs_world_t); // informs if the ability can be run
  void (*validate_cost)(uint8_t); // validate then pass target to AbilityContext
  void (*apply_all_costs)(); // Take all targets from ability context, run through validate to get intents, apply mutations
  void (*validate_effect)(uint8_t); // Same as cost routine
  void (*apply_all_effects)();
} AbilityDef2;

typedef struct {
  uint8_t cost_min, effect_min;
  uint8_t cost_expected, effect_expected;
  uint8_t cost_filled, effect_filled;
  ecs_entity_t cost_targets[MAX_ABILITY_SELECTION];
  ecs_entity_t effect_targets[MAX_ABILITY_SELECTION];
} AbilityContext
```
Upon attaching the AbilityDef component to a card, we should also apply specific tags depending on the ability kind to the card entity.
For example TON_PLAY, TSTART_OF_TURN, TWHEN_EQUIPPING, TWHEN_EQUIPPED, etc. That way its easy to lookup entities by tags.

- I need some type of ability context in the singleton to track ablity states while it is happening. This might mean the above ablityDef gets a little simpler.
  - Ability context should track cost and effect selection
  - Ability context should prob also be able to store the entity pointer for which the ability is being used from

Need a new phase:
DRAW_SELECT // to select cards from a draw

## Spells

### Main Spells

Can only be played during the main phase.

### Response Spells

Can only be played during the response window.

## Effects


