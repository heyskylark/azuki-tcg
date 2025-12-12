#ifndef AZUKI_ABILITIES_RUNTIME_H
#define AZUKI_ABILITIES_RUNTIME_H

#include <flecs.h>

#include "abilities/ability.h"
#include "components.h"

uint16_t azk_make_ability_uid(const AbilityDef *def);
bool azk_is_once_per_turn_used(ecs_world_t *world, ecs_entity_t card, uint8_t ability_index);
void azk_mark_once_per_turn_used(ecs_world_t *world, ecs_entity_t card, uint8_t ability_index);
void azk_reset_once_per_turn_for_player(ecs_world_t *world, ecs_entity_t player);

bool azk_begin_or_resolve_ability(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  ecs_entity_t player,
  ecs_entity_t source_card,
  const AbilityDef *def,
  const ecs_entity_t *ikz_cards,
  uint8_t ikz_card_count,
  bool from_trigger
);

bool azk_append_ability_target(
  AbilityContext *actx,
  AbilitySelectionPhase phase,
  ecs_entity_t target
);

void azk_try_finish_ability(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  bool manual_finish,
  bool user_declined
);

bool azk_validate_target_against_req(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ecs_entity_t source_card,
  ecs_entity_t target,
  AbilityTargetRequirement req,
  AbilityTargetType required_type_override
);

int azk_resolve_ability(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  const AbilityDef *def
);

void azk_clear_ability_context(AbilityContext *actx);
bool azk_trigger_abilities_for_card(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  ecs_entity_t source_card,
  AbilityTiming timing
);

#endif
