#include "systems/main.h"
#include "systems/phase_gate.h"
#include "systems/mulligan_phase.h"
#include "systems/main_phase.h"
#include "systems/response_phase.h"
#include "systems/start_phase.h"
#include "systems/end_phase.h"
#include "systems/combat_resolve_phase.h"

void init_all_system(ecs_world_t *world) {
  init_phase_gate_system(world);
  init_start_phase_system(world);
  init_mulligan_phase_system(world);
  init_main_phase_system(world);
  init_response_phase_system(world);
  init_combat_resolve_phase_system(world);
  init_end_phase_system(world);
}
