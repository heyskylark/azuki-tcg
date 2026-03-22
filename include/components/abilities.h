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
  ABILITY_TARGET_FRIENDLY_GARDEN_OR_ALLEY_ENTITY =
      17, // 0-4 = friendly garden, 5-9 = friendly alley
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
extern ECS_TAG_DECLARE(AStartOfEachTurn);
extern ECS_TAG_DECLARE(AEndOfTurn);
extern ECS_TAG_DECLARE(AWhenEquipping);
extern ECS_TAG_DECLARE(AWhenEquipped);
extern ECS_TAG_DECLARE(AMain);
extern ECS_TAG_DECLARE(AWhenAttacking);
extern ECS_TAG_DECLARE(AAfterAttacking);
extern ECS_TAG_DECLARE(AWhenAttacked);
extern ECS_TAG_DECLARE(AWhenTakesDamage);
extern ECS_TAG_DECLARE(AWhenDealsDamage);
extern ECS_TAG_DECLARE(AWhenEntersGarden);
extern ECS_TAG_DECLARE(AResponse);
extern ECS_TAG_DECLARE(AAlleyOnly);
extern ECS_TAG_DECLARE(AGardenOnly);
extern ECS_TAG_DECLARE(AOnceTurn);
extern ECS_TAG_DECLARE(AWhenReturnedToHand);
extern ECS_TAG_DECLARE(AWhenDestroyed);
extern ECS_TAG_DECLARE(AWhenSacrificed);
extern ECS_TAG_DECLARE(AIgnoresCooldown);
extern ECS_TAG_DECLARE(AOnGatePortal);

/* Keyword Ability Tags */
extern ECS_TAG_DECLARE(Charge);
extern ECS_TAG_DECLARE(Defender);
extern ECS_TAG_DECLARE(Infiltrate);
extern ECS_TAG_DECLARE(Godmode);
extern ECS_TAG_DECLARE(SacrificeAtEndOfTurn);
extern ECS_TAG_DECLARE(Taunt);
extern ECS_TAG_DECLARE(Rooted);

/* Negative Condition Tags */
extern ECS_TAG_DECLARE(Frozen);
extern ECS_TAG_DECLARE(Shocked);

/* Positive Condition Tags */
extern ECS_TAG_DECLARE(EffectImmune);

typedef struct {
  int8_t amount;
} CarapaceValue;

extern ECS_COMPONENT_DECLARE(CarapaceValue);

typedef struct {
  int8_t amount;
  bool expires_eot;
} CarapaceBuff;

extern ECS_COMPONENT_DECLARE(CarapaceBuff);

typedef enum {
  TAG_GRANT_TICK_NONE = 0,
  TAG_GRANT_TICK_START_OF_TURN = 1,
  TAG_GRANT_TICK_END_OF_TURN = 2,
} TagGrantTickPhase;

#define MAX_TIMED_TAG_GRANTS 8

typedef struct {
  ecs_id_t tag;
  int8_t remaining_ticks; // -1 = permanent grant, >0 decrements on matching tick
  uint8_t tick_phase;     // TagGrantTickPhase
} TimedTagGrant;

/* Card Condition Countdowns - tracks duration of status effects and timed tag grants */
/* -1 = permanent, 0 = expired (remove tag), >0 = ticks remaining */
typedef struct {
  int8_t frozen_duration;
  int8_t shocked_duration;
  int8_t effect_immune_duration;
  TimedTagGrant timed_tag_grants[MAX_TIMED_TAG_GRANTS];
  uint8_t timed_tag_grant_count;
} CardConditionCountdown;

extern ECS_COMPONENT_DECLARE(CardConditionCountdown);

/* Attack Buff - used as relationship pair (AttackBuff, source_entity) */
/* Multiple buffs from different sources can coexist on the same entity */
typedef struct {
  int8_t modifier;   // Positive for buff, negative for debuff
  bool expires_eot;  // If true, removed at end of turn
} AttackBuff;

extern ECS_COMPONENT_DECLARE(AttackBuff);

/* Health Buff - used as relationship pair (HealthBuff, source_entity) */
/* Multiple buffs from different sources can coexist on the same entity */
typedef struct {
  int8_t modifier;   // Positive for buff, negative for debuff
  bool expires_eot;  // If true, removed at end of turn
} HealthBuff;

extern ECS_COMPONENT_DECLARE(HealthBuff);

/* Combat Damage Modifier - used as relationship pair (CombatDamageModifier, source_entity) */
/* Incoming modifiers adjust damage taken; outgoing modifiers adjust damage dealt. */
typedef struct {
  int8_t incoming_modifier;
  int8_t outgoing_modifier;
  bool expires_eot;
} CombatDamageModifier;

extern ECS_COMPONENT_DECLARE(CombatDamageModifier);

/* Equipped Combat Modifier - static spec granted by a weapon while attached */
typedef struct {
  int8_t incoming_modifier;
  int8_t outgoing_modifier;
  bool requires_leader;
} EquippedCombatModifier;

extern ECS_COMPONENT_DECLARE(EquippedCombatModifier);

/* Passive Observer Context - stores observer IDs for cleanup */
#define MAX_PASSIVE_OBSERVERS 4

typedef struct {
  ecs_entity_t observers[MAX_PASSIVE_OBSERVERS];
  uint8_t observer_count;
  void *ctx; // Optional allocated context (freed in cleanup)
} PassiveObserverContext;

extern ECS_COMPONENT_DECLARE(PassiveObserverContext);

#define AZK_MAX_TRACKED_DAMAGE_SOURCES 8

typedef struct {
  uint16_t turn_marker;
  bool took_damage_this_turn;
  bool dealt_damage_this_turn;
  bool last_taken_from_effect;
  bool last_dealt_from_effect;
  int8_t last_damage_taken;
  int8_t last_damage_dealt;
  ecs_entity_t last_damage_source;
  ecs_entity_t last_damage_recipient;
  ecs_entity_t tracked_sources[AZK_MAX_TRACKED_DAMAGE_SOURCES];
  uint8_t tracked_source_count;
} DamageTracker;

extern ECS_COMPONENT_DECLARE(DamageTracker);

void azk_register_ability_components(ecs_world_t *world);
void attach_ability_components(ecs_world_t *world, ecs_entity_t card);

#endif
