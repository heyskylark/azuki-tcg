#include "utils/combat_util.h"

int handle_attack(
  ecs_world_t *world,
  ecs_entity_t attacking_player,
  uint8_t garden_attacker_index,
  uint8_t defender_index
) {
  // assert: is attacking player index valid?
  // assert: is attacking player not tapped or on cooldown
  // assert: is defender entity tapped / if leader ignore

  return 0;
}
