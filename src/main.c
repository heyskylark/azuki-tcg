#include "world.h"

int main(void) {
  ecs_world_t *world = azk_world_init(42);

  ecs_progress(world, 0);

  azk_world_fini(world);

  return 0;
}
