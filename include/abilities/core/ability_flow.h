#ifndef AZUKI_ABILITY_FLOW_H
#define AZUKI_ABILITY_FLOW_H

#include <flecs.h>
#include <stdbool.h>

#include "abilities/ability_registry.h"

typedef struct {
  bool select_effects_when_max_positive;
  bool apply_costs_before_effect_selection;
} AbilityInitialPhaseOptions;

bool azk_prepare_effect_selection_after_costs(ecs_world_t *world,
                                              AbilityContext *ctx,
                                              const AbilityDef *def);

bool azk_enter_initial_phase(ecs_world_t *world, AbilityContext *ctx,
                             const AbilityDef *def,
                             const AbilityInitialPhaseOptions *options);

#endif // AZUKI_ABILITY_FLOW_H
