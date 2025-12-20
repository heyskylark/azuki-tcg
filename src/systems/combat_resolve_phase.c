#include "systems/combat_resolve_phase.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/combat_util.h"
#include "utils/observation_util.h"

void HandleCombatResolution(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  resolve_combat(world);

  if (is_game_over(world)) {
    gs->phase = PHASE_END_MATCH;
  } else {
    gs->phase = PHASE_MAIN;
  }

  cli_render_logf("[CombatResolution] Combat resolution");
}

void init_combat_resolve_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "CombatResolvePhaseSystem",
      .add = ecs_ids(TCombatResolve)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) }
    },
    .callback = HandleCombatResolution
  });
}