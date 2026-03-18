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
  AbilityScratchState initial_scratch;
} AbilityContextInitOptions;

void azk_reset_ability_context_state(AbilityContext *ctx);

uint8_t azk_count_remaining_selection_cards(const AbilityContext *ctx);

void azk_init_ability_context(AbilityContext *ctx, ecs_entity_t source_card,
                              ecs_entity_t owner, const AbilityDef *def,
                              uint8_t available_cost_targets,
                              const AbilityContextInitOptions *options);

void azk_clear_ability_context(ecs_world_t *world);

#endif // AZUKI_ABILITY_CONTEXT_H
