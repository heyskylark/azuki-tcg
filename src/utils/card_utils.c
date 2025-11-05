#include "utils/card_utils.h"
#include "generated/card_defs.h"
#include "components.h"
#include "utils/cli_rendering_util.h"
#include <stdio.h>

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type) {
  const Type *card_type = ecs_get(world, card, Type);
  return card_type != NULL && card_type->value == type;
}

void discard_card(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  const PlayerNumber *player_number = ecs_get_id(world, owner, ecs_id(PlayerNumber));
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t discard_zone = gs->zones[player_number->player_number].discard;

  ecs_remove_id(world, card, ecs_id(ZoneIndex));
  ecs_add_pair(world, card, EcsChildOf, discard_zone);
}

int insert_card_into_zone_index(ecs_world_t *world, ecs_entity_t card, ecs_entity_t zone, int index) {
  bool is_garden_zone = ecs_has_id(world, zone, ecs_id(ZGarden));
  bool is_alley_zone = ecs_has_id(world, zone, ecs_id(ZAlley));
  if (!is_garden_zone && !is_alley_zone) {
    cli_render_logf("Zone %d is not a garden or alley", zone);
    return -1;
  }
  // Garden and alley are the same size
  if (index < 0 || index >= GARDEN_SIZE) {
    cli_render_logf("Index %d is out of bounds for garden or alley", index);
    return -1;
  }

  ecs_entity_t card_owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(card_owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  ecs_entity_t zone_owner = ecs_get_target(world, zone, Rel_OwnedBy, 0);
  ecs_assert(zone_owner != 0, ECS_INVALID_PARAMETER, "Zone %d has no owner", zone);
  ecs_assert(zone_owner == card_owner, ECS_INVALID_PARAMETER, "Card %d and zone %d have different owners", card, zone);

  // Find if card exists in given zone index.
  ecs_entities_t zone_cards = ecs_get_ordered_children(world, zone);
  ecs_entity_t card_with_zone_index;
  for (int32_t i = 0; i < zone_cards.count; i++) {
    ecs_entity_t card = zone_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get_id(world, card, ecs_id(ZoneIndex));
    ecs_assert(zone_index != NULL, ECS_INVALID_PARAMETER, "Card %d in zone %d has no ZoneIndex component", card, zone);
    if (zone_index->index == index) {
      card_with_zone_index = card;
      break;
    }
  }

  bool is_zone_full = zone_cards.count >= GARDEN_SIZE; // Alley and garden are the same size
  if (card_with_zone_index != 0) {
    if (!is_zone_full) {
      cli_render_logf("Card %d is already in zone %d at index %d | Zone is not full", card, zone, index);
      return -1;
    }

    discard_card(world, card_with_zone_index);
    card_with_zone_index = 0;
  }

  ecs_add_pair(world, card, EcsChildOf, zone);
  ecs_set(world, card, ZoneIndex, { .index = index });
  
  return 0;
}

void untap_all_cards_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    ecs_set(world, card, TapState, { .tapped = false, .cooldown = false });
  }
}