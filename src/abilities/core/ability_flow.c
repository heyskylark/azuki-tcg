#include "abilities/core/ability_flow.h"

static bool should_enter_effect_selection(
    const AbilityDef *def, const AbilityInitialPhaseOptions *options) {
  if (options->select_effects_when_max_positive) {
    return def->effect_req.max > 0;
  }

  return def->effect_req.min > 0;
}

bool azk_enter_initial_phase(ecs_world_t *world, AbilityContext *ctx,
                             const AbilityDef *def,
                             const AbilityInitialPhaseOptions *options) {
  if (!ctx || !def) {
    return false;
  }

  const AbilityInitialPhaseOptions default_options = {0};
  const AbilityInitialPhaseOptions *phase_options =
      options ? options : &default_options;

  ctx->phase = ABILITY_PHASE_NONE;

  if (def->cost_req.min > 0) {
    ctx->phase = ABILITY_PHASE_COST_SELECTION;
    return true;
  }

  if (def->on_cost_paid) {
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
    }
    def->on_cost_paid(world, ctx);
    return ctx->phase != ABILITY_PHASE_NONE;
  }

  if (should_enter_effect_selection(def, phase_options)) {
    if (phase_options->apply_costs_before_effect_selection && def->apply_costs) {
      def->apply_costs(world, ctx);
    }
    ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
    return true;
  }

  if (def->apply_costs) {
    def->apply_costs(world, ctx);
  }
  if (def->apply_effects) {
    def->apply_effects(world, ctx);
  }

  return false;
}
