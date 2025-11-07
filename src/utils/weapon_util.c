#include "utils/weapon_util.h"
#include "utils/zone_util.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/cli_rendering_util.h"
#include "generated/card_defs.h"

int attach_weapon_from_hand(
  ecs_world_t *world,
  ecs_entity_t player,
  int hand_index,
  int entity_index,
  bool use_ikz_token
) {
  if (entity_index < 0 || entity_index > 5) {
    cli_render_logf("Entity index %d is out of bounds", entity_index);
    return -1;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  uint8_t player_number = get_player_number(world, player);
  
    
  ecs_entity_t hand_zone = gs->zones[player_number].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand_zone);
  if (hand_index < 0 || hand_index >= hand_cards.count) {
    cli_render_logf("Hand index %d is out of bounds", hand_index);
    return -1;
  }

  ecs_entity_t hand_card = hand_cards.ids[hand_index];
  ecs_assert(hand_card != 0, ECS_INVALID_PARAMETER, "Hand card %d not found", hand_index);

  if (!is_card_type(world, hand_card, CARD_TYPE_WEAPON)) {
    cli_render_logf("Hand card %d is not a weapon", hand_card);
    return -1;
  }

  const IKZCost *weapon_ikz_cost = ecs_get(world, hand_card, IKZCost);
  ecs_assert(weapon_ikz_cost != NULL, ECS_INVALID_PARAMETER, "IKZCost component not found for card %d", hand_card);

  ecs_entity_t ikz_area_zone = gs->zones[player_number].ikz_area;
  uint8_t tappable_ikz_cards_count = 0;
  ecs_entity_t tappable_ikz_cards[weapon_ikz_cost->ikz_cost];

  int ikz_fetch_result = get_tappable_ikz_cards(
    world,
    ikz_area_zone,
    weapon_ikz_cost->ikz_cost,
    &tappable_ikz_cards_count,
    tappable_ikz_cards,
    use_ikz_token
  );
  if (ikz_fetch_result < 0) {
    return ikz_fetch_result;
  }

  if (tappable_ikz_cards_count < weapon_ikz_cost->ikz_cost) {
    cli_render_logf("Not enough untapped IKZ cards to attach weapon %d", hand_card);
    return -1;
  }

  ecs_entity_t entity_card;
  if (entity_index == 5) {
    entity_card = gs->zones[player_number].leader;
    entity_card = find_leader_card_in_zone(world, entity_card);
    if (entity_card == 0) {
      cli_render_logf("Leader card not found in zone %d", entity_card);
      return -1;
    }
  } else {
    entity_card = find_card_in_zone_index(world, gs->zones[player_number].garden, entity_index);
    if (entity_card == 0) {
      cli_render_logf("Entity card %d not found in garden at index %d", entity_card, entity_index);
      return -1;
    }
  }

  const BaseStats *weapon_base_stats = ecs_get(world, hand_card, BaseStats);
  ecs_assert(weapon_base_stats != NULL, ECS_INVALID_PARAMETER, "BaseStats component not found for card %d", hand_card);

  const CurStats *entity_cur_stats = ecs_get(world, entity_card, CurStats);
  ecs_assert(entity_cur_stats != NULL, ECS_INVALID_PARAMETER, "CurStats component not found for card %d", entity_card);

  ecs_set(world, entity_card, CurStats, {
    .cur_atk = entity_cur_stats->cur_atk + weapon_base_stats->attack,
    .cur_hp = entity_cur_stats->cur_hp,
  });

  ecs_add_pair(world, hand_card, EcsChildOf, entity_card);

  for (int32_t i = 0; i < weapon_ikz_cost->ikz_cost; i++) {
    set_card_to_tapped(world, tappable_ikz_cards[i]);
  }

  return 0;
}
