#ifndef AZUKI_VALIDATION_ACTION_INTENTS_H
#define AZUKI_VALIDATION_ACTION_INTENTS_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "azuki/zone_types.h"
#include "components/components.h"
#include "constants/game.h"

#define AZK_MAX_IKZ_PAYMENT (IKZ_AREA_SIZE + 1)

typedef struct {
  ecs_entity_t player;
  ecs_entity_t card;
  ZonePlacementType placement_type;
  ecs_entity_t target_zone;
  int zone_index;
  bool use_ikz_token;
  ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT];
  uint8_t ikz_card_count;
  ecs_entity_t displaced_card;
} PlayEntityIntent;

typedef struct {
  ecs_entity_t player;
  ecs_entity_t weapon_card;
  ecs_entity_t target_card;
  bool target_is_leader;
  bool use_ikz_token;
  ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT];
  uint8_t ikz_card_count;
} AttachWeaponIntent;

typedef struct {
  ecs_entity_t player;
  ecs_entity_t alley_card;
  ecs_entity_t target_zone;
  int garden_index;
  ecs_entity_t displaced_card;
  ecs_entity_t gate_card;
} GatePortalIntent;

typedef struct {
  ecs_entity_t attacking_player;
  ecs_entity_t defending_player;
  ecs_entity_t attacking_card;
  ecs_entity_t defending_card;
  uint8_t attacker_index;
  uint8_t defender_index;
  bool attacker_is_leader;
} AttackIntent;

typedef struct {
  ecs_entity_t player;
  ecs_entity_t spell_card;
  bool use_ikz_token;
  ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT];
  uint8_t ikz_card_count;
} PlaySpellIntent;

#endif
