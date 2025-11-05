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

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t discard_zone = gs->zones[player_number->player_number].discard;

  ecs_remove_id(world, card, ecs_id(ZoneIndex)); 
  ecs_set(world, card, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, card, EcsChildOf, discard_zone);
}

static void get_tappable_ikz_cards(
  ecs_world_t *world,
  ecs_entity_t ikz_area_zone,
  uint8_t ikz_cost,
  uint8_t *out_ikz_count,
  ecs_entity_t *out_ikz_cards
) {
  ecs_entities_t ikz_area_cards = ecs_get_ordered_children(world, ikz_area_zone);
  for (int32_t i = 0; i < ikz_area_cards.count && *out_ikz_count < ikz_cost; i++) {
    ecs_entity_t ikz_card = ikz_area_cards.ids[i];
    const TapState *tap_state = ecs_get(world, ikz_card, TapState);
    ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "TapState component not found for card %d", ikz_card);
    if (!tap_state->tapped) {
      out_ikz_cards[*out_ikz_count] = ikz_card;
      (*out_ikz_count)++;
    }
  }
}

static ecs_entity_t find_card_in_zone_index(
  ecs_world_t *world,
  ecs_entity_t zone,
  int index
) {
  ecs_entities_t zone_cards = ecs_get_ordered_children(world, zone);
  ecs_entity_t card_with_zone_index = 0;
  for (int32_t i = 0; i < zone_cards.count; i++) {
    ecs_entity_t card = zone_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
    ecs_assert(zone_index != NULL, ECS_INVALID_PARAMETER, "Card %d in zone %d has no ZoneIndex component", card, zone);
    if (zone_index->index == index) {
      card_with_zone_index = card;
      break;
    }
  }

  return card_with_zone_index;
}

int insert_card_into_zone_index (
  ecs_world_t *world,
  ecs_entity_t card,
  ecs_entity_t player,
  ZonePlacementType placement_type,
  int index
) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  const IKZCost *ikz_cost = ecs_get(world, card, IKZCost);
  ecs_assert(ikz_cost != NULL, ECS_INVALID_PARAMETER, "IKZCost component not found for card %d", card);

  const PlayerNumber *player_number = ecs_get(world, player, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", player);

  ecs_entity_t zone = 0;
  switch (placement_type) {
    case ZONE_GARDEN:
      zone = gs->zones[player_number->player_number].garden;
      break;
    case ZONE_ALLEY:
      zone = gs->zones[player_number->player_number].alley;
      break;
    default:
      cli_render_logf("Invalid placement type: %d", placement_type);
      exit(EXIT_FAILURE);
  }

  ecs_entity_t ikz_area_zone = gs->zones[player_number->player_number].ikz_area;
  uint8_t tappable_ikz_cards_count = 0;
  ecs_entity_t tappable_ikz_cards[ikz_cost->ikz_cost];

  get_tappable_ikz_cards(world, ikz_area_zone, ikz_cost->ikz_cost, &tappable_ikz_cards_count, tappable_ikz_cards);
  
  if (tappable_ikz_cards_count < ikz_cost->ikz_cost) {
    cli_render_logf("Not enough untapped IKZ cards to place card %d", card);
    return -1;
  }

  // Garden and alley are the same size
  if (index < 0 || index >= GARDEN_SIZE) {
    cli_render_logf("Index %d is out of bounds for garden or alley", index);
    return -1;
  }

  ecs_entity_t card_owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(card_owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);
  ecs_assert(card_owner == player, ECS_INVALID_PARAMETER, "Card %d and player %d have different owners", card, player);

  // Find if card exists in given zone index.
  ecs_entities_t zone_cards = ecs_get_ordered_children(world, zone);
  ecs_entity_t card_with_zone_index = find_card_in_zone_index(world, zone, index);

  bool is_zone_full = zone_cards.count >= GARDEN_SIZE; // Alley and garden are the same size
  if (card_with_zone_index != 0) {
    if (!is_zone_full) {
      cli_render_logf("Card %d is already in zone %d at index %d | Zone is not full", card_with_zone_index, zone, index);
      return -1;
    }

    discard_card(world, card_with_zone_index);
    card_with_zone_index = 0;
  }

  ecs_add_pair(world, card, EcsChildOf, zone);
  ecs_set(world, card, ZoneIndex, { .index = index });

  if (placement_type == ZONE_GARDEN) {
    ecs_set(world, card, TapState, { .tapped = false, .cooldown = true });
  }

  for (int32_t i = 0; i < ikz_cost->ikz_cost; i++) {
    ecs_set(world, tappable_ikz_cards[i], TapState, { .tapped = true, .cooldown = false });
  }

  return 0;
}

void untap_all_cards_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    ecs_set(world, card, TapState, { .tapped = false, .cooldown = false });
  }
}