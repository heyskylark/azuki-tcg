#include "components/game_log.h"

ECS_COMPONENT_DECLARE(GameStateLogContext);

void azk_register_game_log_components(ecs_world_t *world) {
  ECS_COMPONENT_DEFINE(world, GameStateLogContext);

  // Initialize GameStateLogContext singleton
  ecs_singleton_set(world, GameStateLogContext, {.count = 0, .turn_number = 0});
}
