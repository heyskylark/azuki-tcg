#ifndef AZUKI_ABILITY_RUNTIME_H
#define AZUKI_ABILITY_RUNTIME_H

#include <flecs.h>
#include <stdbool.h>
#include <stdint.h>

#include "abilities/ability_registry.h"

typedef struct {
  bool is_optional;
  bool enter_confirmation_when_optional;
  bool transfer_control_on_user_input;
  bool clamp_effect_expected_to_available;
  bool select_effects_when_max_positive;
  bool apply_costs_before_effect_selection;
  bool clear_context_on_immediate_resolve;
  uint8_t available_cost_targets;
  uint8_t available_effect_targets;
  AbilityScratchState initial_scratch;
  const char *confirmation_log;
  const char *applied_log;
  const char *cost_selection_log;
  const char *effect_selection_log;
  const char *selection_log;
} AbilityBeginOptions;

bool azk_begin_ability(ecs_world_t *world, ecs_entity_t source_card,
                       ecs_entity_t owner, const AbilityDef *def,
                       const AbilityBeginOptions *options);

void azk_log_ability_initial_phase_entry(AbilityPhase phase,
                                         const char *cost_log,
                                         const char *effect_log,
                                         const char *selection_log);

void azk_maybe_transfer_triggered_ability_control(ecs_world_t *world,
                                                  AbilityContext *ctx);

void azk_restore_triggered_ability_control(ecs_world_t *world,
                                           const AbilityContext *ctx);

#endif // AZUKI_ABILITY_RUNTIME_H
