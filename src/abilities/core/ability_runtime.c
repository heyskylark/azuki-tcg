#include "abilities/core/ability_runtime.h"

#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

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
