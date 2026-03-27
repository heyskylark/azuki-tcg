#include "abilities/core/ability_flow.h"

static bool should_enter_effect_selection(
    const AbilityContext *ctx, const AbilityDef *def,
    const AbilityInitialPhaseOptions *options) {
  if (!ctx || !def) {
    return false;
  }

  if (ctx->effect.max_allowed == 0) {
    return false;
  }

  if (options->select_effects_when_max_positive) {
    return true;
  }

  return ctx->effect.min_required > 0;
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

  ctx->runtime.phase = ABILITY_PHASE_NONE;
  ctx->runtime.apply_costs_before_effect_selection =
      phase_options->apply_costs_before_effect_selection;

  if (def->cost_req.min > 0) {
    ctx->runtime.phase = ABILITY_PHASE_COST_SELECTION;
    return true;
  }

  if (def->on_cost_paid) {
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
      ctx->runtime.costs_applied = true;
    }
    def->on_cost_paid(world, ctx);
    return ctx->runtime.phase != ABILITY_PHASE_NONE;
  }

  if (should_enter_effect_selection(ctx, def, phase_options)) {
    if (phase_options->apply_costs_before_effect_selection && def->apply_costs) {
      def->apply_costs(world, ctx);
      ctx->runtime.costs_applied = true;
    }
    ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
    return true;
  }

  if (def->apply_costs) {
    def->apply_costs(world, ctx);
    ctx->runtime.costs_applied = true;
  }
  if (def->apply_effects) {
    def->apply_effects(world, ctx);
  }

  return false;
}
