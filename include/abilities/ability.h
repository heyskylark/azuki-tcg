#ifndef AZUKI_ABILITIES_ABILITY_H
#define AZUKI_ABILITIES_ABILITY_H

#include <stdbool.h>
#include <stdint.h>
#include <flecs.h>

#include "generated/card_defs.h"
#include "components.h"

#define AZK_MAX_ABILITIES_PER_CARD 4
#define AZK_MAX_TARGETS_PER_ABILITY 2

typedef enum {
  ABILITY_KIND_ACTIVATED = 0,
  ABILITY_KIND_TRIGGERED = 1,
  ABILITY_KIND_PASSIVE = 2
} AbilityKind;

typedef enum {
  ABILITY_TIMING_MAIN = 0,
  ABILITY_TIMING_RESPONSE = 1,
  ABILITY_TIMING_ON_PLAY = 2,
  ABILITY_TIMING_ON_EQUIP = 3,
  ABILITY_TIMING_ON_ATTACK = 4,
  ABILITY_TIMING_START_OF_TURN = 5,
  ABILITY_TIMING_END_OF_TURN = 6
} AbilityTiming;

typedef enum {
  ABILITY_SOURCE_GARDEN = 0,
  ABILITY_SOURCE_ALLEY = 1,
  ABILITY_SOURCE_LEADER = 2,
  ABILITY_SOURCE_GATE = 3,
  ABILITY_SOURCE_HAND = 4
} AbilitySourceZone;

typedef enum {
  ABILITY_TARGET_NONE = 0,
  ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY = 1,
  ABILITY_TARGET_FRIENDLY_ALLEY_ENTITY = 2,
  ABILITY_TARGET_FRIENDLY_ENTITY_WITH_WEAPON = 3,
  ABILITY_TARGET_FRIENDLY_LEADER = 4,
  ABILITY_TARGET_ENEMY_GARDEN_ENTITY = 5,
  ABILITY_TARGET_ENEMY_LEADER = 6,
  ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY = 7,
  ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY = 8
} AbilityTargetType;

typedef struct {
  AbilityTargetType type;
  uint8_t min;
  uint8_t max;
} AbilityTargetRequirement;

typedef struct {
  uint8_t ikz_cost;
  bool tap_self;
  bool sacrifice_self;
} AbilityCostSpec;

typedef struct {
  CardDefId card_def_id;
  uint8_t ability_index;
  const char *name;
  AbilityKind kind;
  AbilityTiming timing;
  AbilitySourceZone source_zone;
  AbilityCostSpec cost;
  AbilityTargetRequirement cost_targets;
  AbilityTargetRequirement effect_targets;
  bool once_per_turn;
  bool optional;
} AbilityDef;

const AbilityDef *azk_get_ability_defs(size_t *out_count);
const AbilityDef *azk_find_ability_def(CardDefId id, uint8_t ability_index);
bool azk_card_has_activated_ability(CardDefId id);
size_t azk_collect_triggered_abilities(
  CardDefId id,
  AbilityTiming timing,
  const AbilityDef *out_defs[AZK_MAX_ABILITIES_PER_CARD]
);
bool azk_ability_has_target_requirement(const AbilityTargetRequirement *req);
uint8_t azk_ability_target_count(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ecs_entity_t source_card,
  AbilityTargetRequirement req,
  AbilityTargetType required_type_override
);
bool azk_ability_targets_available(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ecs_entity_t source_card,
  AbilityTargetRequirement req,
  AbilityTargetType required_type_override
);

#endif
