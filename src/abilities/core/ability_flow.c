#include "abilities/core/ability_flow.h"

#include "abilities/targeting/ability_targeting.h"
#include "utils/cli_rendering_util.h"

static uint8_t clamp_effect_target_count(uint8_t available,
                                         uint8_t requested_max) {
  return available < requested_max ? available : requested_max;
}

static void apply_ability_costs_now(ecs_world_t *world, AbilityContext *ctx,
                                    const AbilityDef *def) {
  if (!world || !ctx || !def || !def->apply_costs) {
    return;
  }

  const bool was_deferred =
      ecs_is_deferred(world) && !ecs_stage_is_readonly(world);
  if (was_deferred) {
    ecs_defer_suspend(world);
  }

  def->apply_costs(world, ctx);
  ctx->runtime.costs_applied = true;

  if (was_deferred) {
    ecs_defer_resume(world);
  }
}

bool azk_prepare_effect_selection_after_costs(ecs_world_t *world,
                                              AbilityContext *ctx,
                                              const AbilityDef *def) {
  if (!world || !ctx || !def) {
    return false;
  }

  const uint8_t available_effect_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, ctx->runtime.source_card,
      ctx->runtime.owner);
  ctx->effect.max_allowed = clamp_effect_target_count(
      available_effect_targets, def->effect_req.max);
  if (ctx->effect.min_required > ctx->effect.max_allowed) {
    ctx->effect.min_required = ctx->effect.max_allowed;
  }

  if (ctx->effect.max_allowed == 0) {
    cli_render_logf("[Ability] No valid effect targets remain after costs");
    return false;
  }

  return true;
}

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
      apply_ability_costs_now(world, ctx, def);
    }
    def->on_cost_paid(world, ctx);
    return ctx->runtime.phase != ABILITY_PHASE_NONE;
  }

  if (should_enter_effect_selection(ctx, def, phase_options)) {
    if (phase_options->apply_costs_before_effect_selection && def->apply_costs) {
      apply_ability_costs_now(world, ctx, def);
      if (!azk_prepare_effect_selection_after_costs(world, ctx, def)) {
        if (def->apply_effects) {
          def->apply_effects(world, ctx);
        }
        return false;
      }
    }
    ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
    return true;
  }

  if (def->apply_costs) {
    apply_ability_costs_now(world, ctx, def);
  }
  if (def->apply_effects) {
    def->apply_effects(world, ctx);
  }

  return false;
}
