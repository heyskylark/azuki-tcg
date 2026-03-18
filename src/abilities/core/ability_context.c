#include "abilities/core/ability_context.h"

#include "abilities/core/ability_runtime.h"
#include "components/abilities.h"

static uint8_t clamp_expected_target_count(uint8_t available,
                                           uint8_t requested_max) {
  return available < requested_max ? available : requested_max;
}

static void azk_init_ability_target_state(AbilityTargetState *state,
                                          uint8_t min_required,
                                          uint8_t max_allowed) {
  if (!state) {
    return;
  }

  *state = (AbilityTargetState){
      .min_required = min_required,
      .max_allowed = max_allowed,
  };
}

void azk_reset_ability_context_state(AbilityContext *ctx) {
  if (!ctx) {
    return;
  }

  *ctx = (AbilityContext){
      .runtime =
          {
              .saved_active_player_index = -1,
          },
  };
}

uint8_t azk_count_remaining_selection_cards(const AbilityContext *ctx) {
  if (!ctx) {
    return 0;
  }

  uint8_t remaining = 0;
  for (uint8_t i = 0;
       i < ctx->selection.count && i < MAX_SELECTION_ZONE_SIZE; ++i) {
    if (ctx->selection.cards[i] != 0) {
      remaining++;
    }
  }

  return remaining;
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

  ctx->runtime.source_card = source_card;
  ctx->runtime.owner = owner;
  ctx->runtime.is_optional = init_options->is_optional;

  azk_init_ability_target_state(
      &ctx->cost, def->cost_req.min,
      clamp_expected_target_count(available_cost_targets, def->cost_req.max));

  azk_init_ability_target_state(&ctx->effect, def->effect_req.min,
                                def->effect_req.max);

  if (init_options->clamp_effect_expected_to_available) {
    ctx->effect.max_allowed = clamp_expected_target_count(
        init_options->available_effect_targets, def->effect_req.max);
    if (ctx->effect.max_allowed < ctx->effect.min_required) {
      ctx->effect.max_allowed = ctx->effect.min_required;
    }
  }

  ctx->scratch = init_options->initial_scratch;
}

void azk_clear_ability_context(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);
  if (!ctx) {
    return;
  }

  if (ctx->runtime.source_card != 0 &&
      ecs_has(world, ctx->runtime.source_card, AOnceTurn)) {
    ecs_set(world, ctx->runtime.source_card, AbilityRepeatContext,
            {.is_once_per_turn = true, .was_applied = true});
  }

  azk_restore_triggered_ability_control(world, ctx);
  azk_reset_ability_context_state(ctx);
  ecs_singleton_modified(world, AbilityContext);
}
