#ifndef AZUKI_UTILS_ABILITY_UTIL_H
#define AZUKI_UTILS_ABILITY_UTIL_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "abilities/ability_registry.h"
#include "components/components.h"

#define AZK_MAX_CARD_ABILITIES 8

bool azk_ability_def_has_timing(const AbilityDef *def, ecs_id_t timing_tag);

ecs_entity_t azk_get_ability_source_card(ecs_world_t *world,
                                         ecs_entity_t ability_entity);

const AbilityDef *azk_get_ability_def_for_entity(ecs_world_t *world,
                                                 ecs_entity_t ability_entity);

uint8_t azk_collect_card_abilities(ecs_world_t *world, ecs_entity_t card,
                                   ecs_entity_t *out_abilities,
                                   uint8_t out_cap);

uint8_t azk_collect_card_timed_abilities(ecs_world_t *world, ecs_entity_t card,
                                         ecs_id_t timing_tag,
                                         ecs_entity_t *out_abilities,
                                         uint8_t out_cap);

uint8_t azk_collect_card_action_abilities(ecs_world_t *world, ecs_entity_t card,
                                          ecs_entity_t *out_abilities,
                                          uint8_t out_cap);

ecs_entity_t azk_find_card_action_ability(ecs_world_t *world, ecs_entity_t card,
                                          int8_t action_index);

ecs_entity_t azk_find_card_ability_by_registry_order(ecs_world_t *world,
                                                     ecs_entity_t card,
                                                     uint8_t registry_order);

bool azk_ability_has_timing(ecs_world_t *world, ecs_entity_t ability_entity,
                            ecs_id_t timing_tag);

int8_t azk_determine_action_index_for_card_ability(ecs_world_t *world,
                                                   ecs_entity_t card,
                                                   uint8_t registry_order,
                                                   const AbilityDef *def);

AbilityInvocationMode azk_determine_invocation_mode_for_card_ability(
    ecs_world_t *world, ecs_entity_t card, const AbilityDef *def);

#endif
