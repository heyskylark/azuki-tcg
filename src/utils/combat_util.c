#include "utils/combat_util.h"
#include "components.h"
#include "utils/cli_rendering_util.h"
#include "utils/zone_util.h"
#include "utils/player_util.h"
#include "utils/card_utils.h"

int attack(
  ecs_world_t *world,
  ecs_entity_t attacking_player,
  uint8_t garden_attacker_index,
  uint8_t defender_index
) {
  if (garden_attacker_index < 0 || garden_attacker_index > 5) {
    cli_render_logf("Garden attacker index %d is out of bounds", garden_attacker_index);
    return -1;
  } else if (defender_index < 0 || defender_index > 5) {
    cli_render_logf("Defender index %d is out of bounds", defender_index);
    return -1;
  }

  const GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  uint8_t attacking_player_number = get_player_number(world, attacking_player);
  uint8_t defender_player_number = (attacking_player_number + 1) % 2;

  ecs_entity_t attacking_card;
  if (garden_attacker_index == 5) {
    attacking_card = find_leader_card_in_zone(world, gs->zones[attacking_player_number].leader);
    ecs_assert(attacking_card != 0, ECS_INVALID_PARAMETER, "Attacking leader card not found");
  } else {
    attacking_card = find_card_in_zone_index(world, gs->zones[attacking_player_number].garden, garden_attacker_index);
   if (attacking_card == 0) {
      cli_render_logf("Attacking player garden card %d not found", garden_attacker_index);
      return -1;
    } else if (is_card_tapped(world, attacking_card) || is_card_cooldown(world, attacking_card)) {
      cli_render_logf("Attacking player garden card %d is tapped or on cooldown", garden_attacker_index);
      return -1;
    } 
  }

  ecs_entity_t defender_card;
  if (defender_index == 5) {
    defender_card = find_leader_card_in_zone(world, gs->zones[defender_player_number].leader);
    ecs_assert(defender_card != 0, ECS_INVALID_PARAMETER, "Defender leader card not found");
  } else {
    defender_card = find_card_in_zone_index(world, gs->zones[defender_player_number].garden, defender_index);

    if (defender_card == 0) {
      cli_render_logf("Defender card %d not found in garden", defender_index);
      return -1;
    } else if (!is_card_tapped(world, defender_card)) {
      cli_render_logf("Defender card %d is not tapped, cannot attack", defender_index);
      return -1;
    }
  }

  CombatState combat_state = {
    .attacking_card = attacking_card,
    .defender_card = defender_card,
  };
  gs->combat_state = combat_state;

  return 0;
}
