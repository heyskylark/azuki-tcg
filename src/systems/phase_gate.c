#include <stdio.h>
#include "systems/phase_gate.h"

void PhaseGate(ecs_iter_t *it) {
  GameState *gs = ecs_field(it, GameState, 0);
  Phase phase = gs->phase;
  printf("Phase: %d\n", phase);
}

void init_phase_gate_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "PhaseGate",
      .add = ecs_ids( ecs_dependson(EcsOnUpdate) )
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) }
    },
    .callback = PhaseGate
  });
}
