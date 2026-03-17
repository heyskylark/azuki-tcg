#ifndef AZUKI_ABILITY_CONTEXT_H
#define AZUKI_ABILITY_CONTEXT_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "abilities/ability_registry.h"

typedef struct {
  bool is_optional;
  bool clamp_effect_expected_to_available;
  uint8_t available_effect_targets;
  ecs_entity_t initial_effect_target;
  uint8_t initial_effect_filled;
} AbilityContextInitOptions;

void azk_reset_ability_context_state(AbilityContext *ctx);

void azk_init_ability_context(AbilityContext *ctx, ecs_entity_t source_card,
                              ecs_entity_t owner, const AbilityDef *def,
                              uint8_t available_cost_targets,
                              const AbilityContextInitOptions *options);

void azk_clear_ability_context(ecs_world_t *world);

#endif // AZUKI_ABILITY_CONTEXT_H
