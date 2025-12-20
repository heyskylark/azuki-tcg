#include "utils/zone_util.h"
#include "generated/card_defs.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include <stdio.h>

ecs_entity_t find_card_in_zone_index(
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

int get_tappable_ikz_cards(
  ecs_world_t *world,
  ecs_entity_t ikz_area_zone,
  uint8_t ikz_cost,
  uint8_t *out_ikz_count,
  ecs_entity_t *out_ikz_cards,
  bool use_ikz_token
) {
  if (ikz_cost == 0) {
    return 0;
  }

  if (use_ikz_token) {
    ecs_entity_t ikz_token_owner = ecs_get_target(world, ikz_area_zone, Rel_OwnedBy, 0);
    ecs_assert(ikz_token_owner != 0, ECS_INVALID_PARAMETER, "IKZ area zone %d has no owner", ikz_area_zone);

    const IKZToken *ikz_token = ecs_get(world, ikz_token_owner, IKZToken);
    if (ikz_token == NULL) {
      cli_render_logf("No IKZ token found for player %d", ikz_token_owner);
      return -1;
    } else if (is_card_tapped(world, ikz_token->ikz_token)) {
      cli_render_logf("IKZ token %d is tapped", ikz_token->ikz_token);
      return -1;
    }

    ecs_assert(
      is_card_type(world, ikz_token->ikz_token, CARD_TYPE_IKZ) || is_card_type(world, ikz_token->ikz_token, CARD_TYPE_EXTRA_IKZ),
      ECS_INVALID_PARAMETER,
      "IKZ token %d is not an IKZ card or extra IKZ card", ikz_token->ikz_token
    );

    out_ikz_cards[*out_ikz_count] = ikz_token->ikz_token;
    (*out_ikz_count)++;

    if (*out_ikz_count == ikz_cost) {
      return 0;
    }
  }

  ecs_entities_t ikz_area_cards = ecs_get_ordered_children(world, ikz_area_zone);
  for (int32_t i = 0; i < ikz_area_cards.count && *out_ikz_count < ikz_cost; i++) {
    ecs_entity_t ikz_card = ikz_area_cards.ids[i];

    ecs_assert(
      is_card_type(world, ikz_card, CARD_TYPE_IKZ) || is_card_type(world, ikz_card, CARD_TYPE_EXTRA_IKZ),
      ECS_INVALID_PARAMETER,
      "Card %d is not an IKZ card", ikz_card
    );
    
    const TapState *tap_state = ecs_get(world, ikz_card, TapState);
    ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "TapState component not found for card %d", ikz_card);
    if (!tap_state->tapped) {
      out_ikz_cards[*out_ikz_count] = ikz_card;
      (*out_ikz_count)++;
    }
  }

  return 0;
}

static int insert_card_into_zone_index(
  ecs_world_t *world,
  ecs_entity_t card,
  ecs_entity_t player,
  ecs_entity_t zone,
  int index,
  ecs_entity_t displaced_card,
  ZonePlacementType placement_type
) {
  if (index < 0 || index >= GARDEN_SIZE) {
    cli_render_logf("Index %d is out of bounds for garden or alley", index);
    return -1;
  }

  ecs_entity_t card_owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(card_owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);
  ecs_assert(card_owner == player, ECS_INVALID_PARAMETER, "Card %d and player %d have different owners", card, player);

  if (displaced_card != 0) {
    discard_card(world, displaced_card);
  }

  ecs_add_pair(world, card, EcsChildOf, zone);
  ecs_set(world, card, ZoneIndex, { .index = index });

  if (placement_type == ZONE_GARDEN) {
    set_card_to_cooldown(world, card);
  }

  return 0;
}

int summon_card_into_zone_index(
  ecs_world_t *world,
  const PlayEntityIntent *intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER, "PlayEntityIntent is null");

  int results = insert_card_into_zone_index(
    world,
    intent->card,
    intent->player,
    intent->target_zone,
    intent->zone_index,
    intent->displaced_card,
    intent->placement_type
  );
  if (results < 0) {
    return results;
  }

  for (uint8_t i = 0; i < intent->ikz_card_count; ++i) {
    set_card_to_tapped(world, intent->ikz_cards[i]);
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

ecs_entity_t find_gate_card_in_zone(
  ecs_world_t *world,
  ecs_entity_t zone
) {
  ecs_iter_t child_it = ecs_children(world, zone);
  bool gate_child_has_next = ecs_children_next(&child_it);
  ecs_assert(gate_child_has_next, ECS_INVALID_PARAMETER, "Gate zone must contain at least 1 card child");
  ecs_assert(child_it.count == 1, ECS_INVALID_PARAMETER, "Gate zone must contain exactly 1 card, got %d", child_it.count);
  ecs_entity_t gate_card = child_it.entities[0];
  ecs_assert(is_card_type(world, gate_card, CARD_TYPE_GATE), ECS_INVALID_PARAMETER, "Card %d is not a gate", gate_card);
  return gate_card;
}

ecs_entity_t find_leader_card_in_zone(
  ecs_world_t *world,
  ecs_entity_t zone
) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  ecs_assert(cards.count == 1, ECS_INVALID_PARAMETER, "Leader zone must contain exactly 1 card, got %d", cards.count);
  ecs_entity_t leader_card = cards.ids[0];
  ecs_assert(is_card_type(world, leader_card, CARD_TYPE_LEADER), ECS_INVALID_PARAMETER, "Card %d is not a leader", leader_card);
  return leader_card;
}

int gate_card_into_garden(
  ecs_world_t *world,
  const GatePortalIntent *intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER, "GatePortalIntent is null");

  int results = insert_card_into_zone_index(
    world,
    intent->alley_card,
    intent->player,
    intent->target_zone,
    intent->garden_index,
    intent->displaced_card,
    ZONE_GARDEN
  );

  if (results < 0) {
    return results;
  }

  set_card_to_tapped(world, intent->gate_card);

  return 0;
}
