#include "world.h"
#include "utils/phase_utils.h"
#include "utils/actions_util.h"

int main(void) {
  ecs_world_t *world = azk_world_init(42);

  bool game_over = false;
  while (!game_over) {
    const GameState *gs = ecs_singleton_get(world, GameState);
    bool requires_user_action = phase_requires_user_action(gs->phase);
    if (requires_user_action) {
      azk_block_for_user_action(world);
    }

    ecs_progress(world, 0);

    game_over = gs->winner != -1;
  }

  azk_world_fini(world);

  return 0;
}
