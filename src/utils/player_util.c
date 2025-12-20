#include "utils/player_util.h"
#include "components/components.h"

uint8_t get_player_number(ecs_world_t *world, ecs_entity_t player) {
  const PlayerNumber *player_number = ecs_get(world, player, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", player);
  return player_number->player_number;
}
