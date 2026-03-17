#include "abilities/core/ability_runtime.h"

#include "abilities/core/ability_context.h"
#include "abilities/core/ability_flow.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

void azk_log_ability_initial_phase_entry(AbilityPhase phase,
                                         const char *cost_log,
                                         const char *effect_log,
                                         const char *selection_log) {
  switch (phase) {
  case ABILITY_PHASE_COST_SELECTION:
    if (cost_log) {
      cli_render_logf("%s", cost_log);
    }
    break;
  case ABILITY_PHASE_EFFECT_SELECTION:
    if (effect_log) {
      cli_render_logf("%s", effect_log);
    }
    break;
  case ABILITY_PHASE_SELECTION_PICK:
  case ABILITY_PHASE_BOTTOM_DECK:
    if (selection_log) {
      cli_render_logf("%s", selection_log);
    }
    break;
  default:
    break;
  }
}

bool azk_begin_ability(ecs_world_t *world, ecs_entity_t source_card,
                       ecs_entity_t owner, const AbilityDef *def,
                       const AbilityBeginOptions *options) {
  if (!def) {
    return false;
  }

  const AbilityBeginOptions default_options = {0};
  const AbilityBeginOptions *begin_options =
      options ? options : &default_options;

  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);
  if (!ctx) {
    return false;
  }

  azk_init_ability_context(ctx, source_card, owner, def,
                           begin_options->available_cost_targets,
                           &(AbilityContextInitOptions){
                               .is_optional = begin_options->is_optional,
                               .clamp_effect_expected_to_available =
                                   begin_options
                                       ->clamp_effect_expected_to_available,
                               .available_effect_targets =
                                   begin_options->available_effect_targets,
                               .initial_effect_target =
                                   begin_options->initial_effect_target,
                               .initial_effect_filled =
                                   begin_options->initial_effect_filled,
                           });

  if (begin_options->enter_confirmation_when_optional && ctx->is_optional) {
    ctx->phase = ABILITY_PHASE_CONFIRMATION;

    if (begin_options->transfer_control_on_user_input) {
      azk_maybe_transfer_triggered_ability_control(world, ctx);
    }

    if (begin_options->confirmation_log) {
      cli_render_logf("%s", begin_options->confirmation_log);
    }

    ecs_singleton_modified(world, AbilityContext);
    return true;
  }

  bool is_active = azk_enter_initial_phase(
      world, ctx, def,
      &(AbilityInitialPhaseOptions){
          .select_effects_when_max_positive =
              begin_options->select_effects_when_max_positive,
          .apply_costs_before_effect_selection =
              begin_options->apply_costs_before_effect_selection,
      });

  if (!is_active) {
    if (begin_options->clear_context_on_immediate_resolve) {
      azk_clear_ability_context(world);
    } else {
      ecs_singleton_modified(world, AbilityContext);
    }

    if (begin_options->applied_log) {
      cli_render_logf("%s", begin_options->applied_log);
    }

    return false;
  }

  if (begin_options->transfer_control_on_user_input) {
    azk_maybe_transfer_triggered_ability_control(world, ctx);
  }

  azk_log_ability_initial_phase_entry(
      ctx->phase, begin_options->cost_selection_log,
      begin_options->effect_selection_log, begin_options->selection_log);
  ecs_singleton_modified(world, AbilityContext);
  return true;
}

void azk_maybe_transfer_triggered_ability_control(ecs_world_t *world,
                                                  AbilityContext *ctx) {
  if (!ctx) {
    return;
  }

  ctx->restores_active_player = false;
  ctx->saved_active_player_index = -1;

  if (ctx->phase == ABILITY_PHASE_NONE) {
    return;
  }

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  if (!gs) {
    return;
  }

  uint8_t owner_player_num = get_player_number(world, ctx->owner);
  if (gs->active_player_index == owner_player_num) {
    return;
  }

  ctx->restores_active_player = true;
  ctx->saved_active_player_index = gs->active_player_index;

  cli_render_logf("[Ability] Switching control to player %d for triggered ability",
                  owner_player_num);
  gs->active_player_index = (int8_t)owner_player_num;
  ecs_singleton_modified(world, GameState);
}

void azk_restore_triggered_ability_control(ecs_world_t *world,
                                           const AbilityContext *ctx) {
  if (!ctx || !ctx->restores_active_player) {
    return;
  }

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  if (!gs) {
    return;
  }

  if (ctx->saved_active_player_index < 0 ||
      ctx->saved_active_player_index >= MAX_PLAYERS_PER_MATCH ||
      gs->active_player_index == ctx->saved_active_player_index) {
    return;
  }

  cli_render_logf("[Ability] Restoring control to player %d",
                  ctx->saved_active_player_index);
  gs->active_player_index = ctx->saved_active_player_index;
  ecs_singleton_modified(world, GameState);
}
