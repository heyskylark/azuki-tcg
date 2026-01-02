#ifndef AZUKI_ECS_ABILITIES_H
#define AZUKI_ECS_ABILITIES_H

#include <flecs.h>
#include <stdint.h>

typedef enum {
  ABILITY_TARGET_NONE = 0,
  ABILITY_TARGET_SELF = 1,
  ABILITY_TARGET_FRIENDLY_HAND = 2,
  ABILITY_TARGET_FRIENDLY_IKZ = 3,
  ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY = 4,
  ABILITY_TARGET_FRIENDLY_ALLEY_ENTITY = 5,
  ABILITY_TARGET_FRIENDLY_ENTITY_WITH_WEAPON = 6,
  ABILITY_TARGET_FRIENDLY_LEADER = 7,
  ABILITY_TARGET_ENEMY_GARDEN_ENTITY = 8,
  ABILITY_TARGET_ENEMY_LEADER = 9,
  ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY = 10,
  ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY = 11,
  ABILITY_TARGET_ANY_GARDEN_ENTITY =
      12, // 0-4 = self garden, 5-9 = opponent garden
  ABILITY_TARGET_ANY_LEADER = 16, // 0 = friendly leader, 1 = enemy leader
  ABILITY_TARGET_FRIENDLY_SELECTION = 13, // Any card in selection zone
  ABILITY_TARGET_FRIENDLY_SELECTION_WEAPON =
      14,                                   // Weapon card in selection zone
  ABILITY_TARGET_FRIENDLY_HAND_WEAPON = 15, // Weapon card in hand
} AbilityTargetType;

typedef struct {
  AbilityTargetType type;
  uint8_t min;
  uint8_t max;
} AbilityCostRequirements;

typedef struct {
  AbilityTargetType type;
  uint8_t min;
  uint8_t max;
} AbilityEffectRequirements;

typedef struct {
  bool is_once_per_turn;
  bool was_applied;
} AbilityRepeatContext;

typedef struct {
  void (*init_observer)(ecs_world_t, ecs_entity_t, ecs_entity_t);
  bool (*validate_all)(ecs_world_t); // informs if the ability can be run
  void (*validate_cost)(uint8_t); // validate then pass target to AbilityContext
  void (*apply_all_costs)();      // Take all targets from ability context, run
                             // through validate to get intents, apply mutations
  void (*validate_effect)(uint8_t); // Same as cost routine
  void (*apply_all_effects)();
} AbilityFunctions;

extern ECS_COMPONENT_DECLARE(AbilityRepeatContext);
extern ECS_COMPONENT_DECLARE(AbilityCostRequirements);
extern ECS_COMPONENT_DECLARE(AbilityEffectRequirements);
extern ECS_COMPONENT_DECLARE(AbilityFunctions);

/* Ability Timing Tags */
extern ECS_TAG_DECLARE(AOnPlay);
extern ECS_TAG_DECLARE(AStartOfTurn);
extern ECS_TAG_DECLARE(AEndOfTurn);
extern ECS_TAG_DECLARE(AWhenEquipping);
extern ECS_TAG_DECLARE(AWhenEquipped);
extern ECS_TAG_DECLARE(AMain);
extern ECS_TAG_DECLARE(AWhenAttacking);
extern ECS_TAG_DECLARE(AWhenAttacked);
extern ECS_TAG_DECLARE(AResponse);
extern ECS_TAG_DECLARE(AAlleyOnly);
extern ECS_TAG_DECLARE(AGardenOnly);
extern ECS_TAG_DECLARE(AOnceTurn);
extern ECS_TAG_DECLARE(AWhenReturnedToHand);
extern ECS_TAG_DECLARE(AIgnoresCooldown);
extern ECS_TAG_DECLARE(AOnGatePortal);

/* Keyword Ability Tags */
extern ECS_TAG_DECLARE(Charge);
extern ECS_TAG_DECLARE(Defender);
extern ECS_TAG_DECLARE(Infiltrate);
extern ECS_TAG_DECLARE(Godmode);

/* Negative Condition Tags */
extern ECS_TAG_DECLARE(Frozen);
extern ECS_TAG_DECLARE(Shocked);

/* Positive Condition Tags */
extern ECS_TAG_DECLARE(EffectImmune);

/* Card Condition Countdowns - tracks duration of status effects */
/* -1 = permanent, 0 = expired (remove tag), >0 = turns remaining */
typedef struct {
  int8_t frozen_duration;
  int8_t shocked_duration;
  int8_t effect_immune_duration;
} CardConditionCountdown;

extern ECS_COMPONENT_DECLARE(CardConditionCountdown);

/* Attack Buff - used as relationship pair (AttackBuff, source_entity) */
/* Multiple buffs from different sources can coexist on the same entity */
typedef struct {
  int8_t modifier;   // Positive for buff, negative for debuff
  bool expires_eot;  // If true, removed at end of turn
} AttackBuff;

extern ECS_COMPONENT_DECLARE(AttackBuff);

/* Passive Observer Context - stores observer IDs for cleanup */
#define MAX_PASSIVE_OBSERVERS 4

typedef struct {
  ecs_entity_t observers[MAX_PASSIVE_OBSERVERS];
  uint8_t observer_count;
  void *ctx; // Optional allocated context (freed in cleanup)
} PassiveObserverContext;

extern ECS_COMPONENT_DECLARE(PassiveObserverContext);

void azk_register_ability_components(ecs_world_t *world);
void attach_ability_components(ecs_world_t *world, ecs_entity_t card);

#endif
