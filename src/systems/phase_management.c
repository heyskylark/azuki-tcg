#include "systems/phase_management.h"
#include "components.h"
#include <stdio.h>
#include <stdlib.h>

static ecs_entity_t s_phase_management_system = 0;

static bool is_invalid_action(ActionContext *ac) {
  if (ac->invalid_action) {
    fprintf(stderr, "[PhaseManagement] Invalid mulligan action: %d\n", ac->user_action.type);
    ac->invalid_action = false;
    return true;
  }

  return false;
}

static void mulligan_phase_handler(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (gs->active_player_index == 0) {
    gs->active_player_index = 1;
  } else {
    gs->active_player_index = 0;
    gs->phase = PHASE_START_OF_TURN;
  }
}

static void start_of_turn_phase_handler(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  gs->phase = PHASE_MAIN;
}

void PhaseManagement(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);
  Phase phase = gs->phase;

  if (is_invalid_action(ac)) {
    return;
  }

  switch (phase) {
    case PHASE_PREGAME_MULLIGAN:
      mulligan_phase_handler(world, gs, ac);
      break;
    case PHASE_START_OF_TURN:
      start_of_turn_phase_handler(world, gs, ac);
      break;
    default:
      fprintf(stderr, "[PhaseManagement] Phase not implemented: %d\n", phase);
      exit(EXIT_FAILURE);
  }
}

void init_phase_management_system(ecs_world_t *world) {
  s_phase_management_system = ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "PhaseManagement",
      .add = ecs_ids( ecs_dependson(EcsOnValidate) )
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) },
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) }
    },
    .callback = PhaseManagement
  });
}

void run_phase_management_system(ecs_world_t *world) {
  ecs_assert(s_phase_management_system != 0, ECS_INVALID_PARAMETER, "Phase management system not initialized");
  ecs_run(world, s_phase_management_system, 0, NULL);
}
