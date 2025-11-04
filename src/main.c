#include "world.h"
#include "systems/phase_gate.h"
#include "systems/phase_management.h"
#include "utils/phase_utils.h"
#include "utils/actions_util.h"
#include "utils/observation_util.h"
#include "utils/cli_rendering_util.h"

int main(void) {
  ecs_world_t *world = azk_world_init(42);
  cli_render_init();

  bool game_over = false;
  while (!game_over) {
    const GameState *gs = ecs_singleton_get(world, GameState);
    ObservationData observation_data = create_observation_data(world);
    cli_render_draw(&observation_data, gs);
    bool requires_user_action = phase_requires_user_action(gs->phase);
    if (requires_user_action) {
      azk_block_for_user_action(world);
    }

    run_phase_gate_system(world);

    ecs_progress(world, 0);

    run_phase_management_system(world);

    gs = ecs_singleton_get(world, GameState);
    game_over = gs->winner != -1;
  }

  cli_render_shutdown();
  azk_world_fini(world);

  return 0;
}
