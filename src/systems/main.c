#include "systems/main.h"
#include "systems/phase_gate.h"

void init_all_system(ecs_world_t *world) {
  init_phase_gate_system(world);
}
