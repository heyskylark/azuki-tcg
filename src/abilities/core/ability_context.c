#include "abilities/core/ability_context.h"

#include "abilities/core/ability_runtime.h"
#include "components/abilities.h"

static uint8_t clamp_expected_target_count(uint8_t available,
                                           uint8_t requested_max) {
  return available < requested_max ? available : requested_max;
}

void azk_reset_ability_context_state(AbilityContext *ctx) {
  if (!ctx) {
    return;
  }

  *ctx = (AbilityContext){
      .saved_active_player_index = -1,
  };
}

void azk_init_ability_context(AbilityContext *ctx, ecs_entity_t source_card,
                              ecs_entity_t owner, const AbilityDef *def,
                              uint8_t available_cost_targets,
                              const AbilityContextInitOptions *options) {
  if (!ctx || !def) {
    return;
  }

  const AbilityContextInitOptions default_options = {0};
  const AbilityContextInitOptions *init_options =
      options ? options : &default_options;

  azk_reset_ability_context_state(ctx);

  ctx->source_card = source_card;
  ctx->owner = owner;
  ctx->is_optional = init_options->is_optional;
  ctx->cost_min = def->cost_req.min;
  ctx->cost_expected =
      clamp_expected_target_count(available_cost_targets, def->cost_req.max);
  ctx->effect_min = def->effect_req.min;

  if (init_options->clamp_effect_expected_to_available) {
    ctx->effect_expected = clamp_expected_target_count(
        init_options->available_effect_targets, def->effect_req.max);
    if (ctx->effect_expected < ctx->effect_min) {
      ctx->effect_expected = ctx->effect_min;
    }
  } else {
    ctx->effect_expected = def->effect_req.max;
  }

  if (init_options->initial_effect_filled > 0) {
    ctx->effect_targets[0] = init_options->initial_effect_target;
    ctx->effect_filled = init_options->initial_effect_filled;
    if (ctx->effect_expected < init_options->initial_effect_filled) {
      ctx->effect_expected = init_options->initial_effect_filled;
    }
  }
}

void azk_clear_ability_context(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);
  if (!ctx) {
    return;
  }

  if (ctx->source_card != 0 && ecs_has(world, ctx->source_card, AOnceTurn)) {
    ecs_set(world, ctx->source_card, AbilityRepeatContext,
            {.is_once_per_turn = true, .was_applied = true});
  }

  azk_restore_triggered_ability_control(world, ctx);
  azk_reset_ability_context_state(ctx);
  ecs_singleton_modified(world, AbilityContext);
}
