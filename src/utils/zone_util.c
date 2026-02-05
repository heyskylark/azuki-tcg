#include "utils/zone_util.h"

#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/debug_log.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include <stdio.h>

void azk_debug_validate_zone_indices(ecs_world_t *world, ecs_entity_t zone,
                                     uint8_t max_slots) {
#ifdef NDEBUG
  (void)world;
  (void)zone;
  (void)max_slots;
  return;
#else
  ecs_entities_t zone_cards = ecs_get_ordered_children(world, zone);
  if (zone_cards.count > max_slots) {
    AZK_DEBUG_WARN("[ZoneIndex] Zone %d has %d cards; exceeds max slots %u",
                   (int)zone, zone_cards.count, (unsigned)max_slots);
  }

  bool seen[max_slots];
  for (uint8_t i = 0; i < max_slots; ++i) {
    seen[i] = false;
  }

  for (int32_t i = 0; i < zone_cards.count; i++) {
    ecs_entity_t card = zone_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
    if (!zone_index) {
      AZK_DEBUG_WARN("[ZoneIndex] Card %d in zone %d missing ZoneIndex",
                     (int)card, (int)zone);
      continue;
    }
    if (zone_index->index >= max_slots) {
      AZK_DEBUG_WARN("[ZoneIndex] Card %d in zone %d has out-of-range "
                     "ZoneIndex %u",
                     (int)card, (int)zone, (unsigned)zone_index->index);
      continue;
    }
    if (seen[zone_index->index]) {
      AZK_DEBUG_WARN("[ZoneIndex] Duplicate ZoneIndex %u in zone %d",
                     (unsigned)zone_index->index, (int)zone);
      continue;
    }
    seen[zone_index->index] = true;
  }
#endif
}

ecs_entity_t find_card_in_zone_index(ecs_world_t *world, ecs_entity_t zone,
                                     int index) {
  ecs_entities_t zone_cards = ecs_get_ordered_children(world, zone);
  ecs_entity_t card_with_zone_index = 0;
  for (int32_t i = 0; i < zone_cards.count; i++) {
    ecs_entity_t card = zone_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
    ecs_assert(zone_index != NULL, ECS_INVALID_PARAMETER,
               "Card %d in zone %d has no ZoneIndex component", card, zone);
    if (zone_index->index == index) {
      card_with_zone_index = card;
      break;
    }
  }

  return card_with_zone_index;
}

int get_tappable_ikz_cards(ecs_world_t *world, ecs_entity_t ikz_area_zone,
                           uint8_t ikz_cost, uint8_t *out_ikz_count,
                           ecs_entity_t *out_ikz_cards, bool use_ikz_token) {
  if (ikz_cost == 0) {
    return 0;
  }

  if (use_ikz_token) {
    ecs_entity_t ikz_token_owner =
        ecs_get_target(world, ikz_area_zone, Rel_OwnedBy, 0);
    ecs_assert(ikz_token_owner != 0, ECS_INVALID_PARAMETER,
               "IKZ area zone %d has no owner", ikz_area_zone);

    const IKZToken *ikz_token = ecs_get(world, ikz_token_owner, IKZToken);
    if (ikz_token == NULL) {
      return -1;
    } else if (is_card_tapped(world, ikz_token->ikz_token)) {
      return -1;
    }

    ecs_assert(
        is_card_type(world, ikz_token->ikz_token, CARD_TYPE_IKZ) ||
            is_card_type(world, ikz_token->ikz_token, CARD_TYPE_EXTRA_IKZ),
        ECS_INVALID_PARAMETER,
        "IKZ token %d is not an IKZ card or extra IKZ card",
        ikz_token->ikz_token);

    out_ikz_cards[*out_ikz_count] = ikz_token->ikz_token;
    (*out_ikz_count)++;

    if (*out_ikz_count == ikz_cost) {
      return 0;
    }
  }

  ecs_entities_t ikz_area_cards =
      ecs_get_ordered_children(world, ikz_area_zone);
  for (int32_t i = 0; i < ikz_area_cards.count && *out_ikz_count < ikz_cost;
       i++) {
    ecs_entity_t ikz_card = ikz_area_cards.ids[i];

    ecs_assert(is_card_type(world, ikz_card, CARD_TYPE_IKZ) ||
                   is_card_type(world, ikz_card, CARD_TYPE_EXTRA_IKZ),
               ECS_INVALID_PARAMETER, "Card %d is not an IKZ card", ikz_card);

    const TapState *tap_state = ecs_get(world, ikz_card, TapState);
    ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
               "TapState component not found for card %d", ikz_card);
    if (!tap_state->tapped) {
      out_ikz_cards[*out_ikz_count] = ikz_card;
      (*out_ikz_count)++;
    }
  }

  return 0;
}

static int insert_card_into_zone_index(ecs_world_t *world, ecs_entity_t card,
                                       ecs_entity_t player, ecs_entity_t zone,
                                       int index, ecs_entity_t displaced_card,
                                       ZonePlacementType placement_type) {
  if (index < 0 || index >= GARDEN_SIZE) {
    cli_render_logf("Index %d is out of bounds for garden or alley", index);
    return -1;
  }

  ecs_entity_t card_owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(card_owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner",
             card);
  ecs_assert(card_owner == player, ECS_INVALID_PARAMETER,
             "Card %d and player %d have different owners", card, player);

  if (displaced_card != 0) {
    discard_card(world, displaced_card);
  }

  ecs_add_pair(world, card, EcsChildOf, zone);
  ecs_set(world, card, ZoneIndex, {.index = index});

  if (placement_type == ZONE_GARDEN) {
    if (!ecs_has(world, card, Charge)) {
      // Set cooldown state (logging handled by caller after ZONE_MOVED)
      const TapState *tap_state = ecs_get(world, card, TapState);
      ecs_set(world, card, TapState,
              {.tapped = tap_state ? tap_state->tapped : false, .cooldown = true});
    }
  }

  azk_debug_validate_zone_indices(
      world, zone,
      placement_type == ZONE_GARDEN ? GARDEN_SIZE : ALLEY_SIZE);

  return 0;
}

int summon_card_into_zone_index(ecs_world_t *world,
                                const PlayEntityIntent *intent) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER, "PlayEntityIntent is null");

  // Capture source zone and index BEFORE moving card
  ecs_entity_t from_zone_entity =
      ecs_get_target(world, intent->card, EcsChildOf, 0);
  GameLogZone from_zone = azk_zone_entity_to_log_zone(world, from_zone_entity);
  int8_t from_index =
      azk_get_card_index_in_zone(world, intent->card, from_zone_entity);

  int results = insert_card_into_zone_index(
      world, intent->card, intent->player, intent->target_zone,
      intent->zone_index, intent->displaced_card, intent->placement_type);
  if (results < 0) {
    return results;
  }

  // Log zone movement FIRST
  GameLogZone to_zone = intent->placement_type == ZONE_GARDEN ? GLOG_ZONE_GARDEN
                                                              : GLOG_ZONE_ALLEY;
  azk_log_card_zone_moved(world, intent->card, from_zone, from_index, to_zone,
                          (int8_t)intent->zone_index);

  // Log cooldown state change AFTER zone move (if applicable)
  if (intent->placement_type == ZONE_GARDEN &&
      !ecs_has(world, intent->card, Charge)) {
    azk_log_card_tap_state_changed_ex(world, intent->card, GLOG_TAP_COOLDOWN,
                                      GLOG_ZONE_GARDEN,
                                      (int8_t)intent->zone_index);
  }

  // Tap IKZ cards
  for (uint8_t i = 0; i < intent->ikz_card_count; ++i) {
    tap_card(world, intent->ikz_cards[i]);
  }

  return 0;
}

void untap_all_cards_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    const TapState *ts = ecs_get(world, card, TapState);
    // Only log if state actually changes
    if (ts && (ts->tapped || ts->cooldown)) {
      ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
      azk_log_card_tap_state_changed(world, card, GLOG_TAP_UNTAPPED);
    } else {
      ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
    }
  }
}

uint8_t untap_n_ikz_cards(ecs_world_t *world, ecs_entity_t ikz_area,
                          uint8_t max_untap) {
  uint8_t untapped = 0;
  ecs_entities_t cards = ecs_get_ordered_children(world, ikz_area);
  for (int32_t i = 0; i < cards.count && untapped < max_untap; i++) {
    ecs_entity_t card = cards.ids[i];
    const TapState *ts = ecs_get(world, card, TapState);
    if (ts && ts->tapped) {
      ecs_set(world, card, TapState,
              {.tapped = false, .cooldown = ts->cooldown});
      azk_log_card_tap_state_changed(world, card, GLOG_TAP_UNTAPPED);
      untapped++;
    }
  }
  return untapped;
}

ecs_entity_t find_gate_card_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_iter_t child_it = ecs_children(world, zone);
  bool gate_child_has_next = ecs_children_next(&child_it);
  ecs_assert(gate_child_has_next, ECS_INVALID_PARAMETER,
             "Gate zone must contain at least 1 card child");
  ecs_assert(child_it.count == 1, ECS_INVALID_PARAMETER,
             "Gate zone must contain exactly 1 card, got %d", child_it.count);
  ecs_entity_t gate_card = child_it.entities[0];
  ecs_assert(is_card_type(world, gate_card, CARD_TYPE_GATE),
             ECS_INVALID_PARAMETER, "Card %d is not a gate", gate_card);
  return gate_card;
}

ecs_entity_t find_leader_card_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  ecs_assert(cards.count == 1, ECS_INVALID_PARAMETER,
             "Leader zone must contain exactly 1 card, got %d", cards.count);
  ecs_entity_t leader_card = cards.ids[0];
  ecs_assert(is_card_type(world, leader_card, CARD_TYPE_LEADER),
             ECS_INVALID_PARAMETER, "Card %d is not a leader", leader_card);
  return leader_card;
}

int gate_card_into_garden(ecs_world_t *world, const GatePortalIntent *intent) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER, "GatePortalIntent is null");

  // Capture source zone index before moving (alley cards have zone index)
  const ZoneIndex *from_zi = ecs_get(world, intent->alley_card, ZoneIndex);
  int8_t from_index = from_zi ? (int8_t)from_zi->index : -1;

  int results = insert_card_into_zone_index(
      world, intent->alley_card, intent->player, intent->target_zone,
      intent->garden_index, intent->displaced_card, ZONE_GARDEN);

  if (results < 0) {
    return results;
  }

  // Log alley -> garden movement FIRST
  azk_log_card_zone_moved(world, intent->alley_card, GLOG_ZONE_ALLEY, from_index,
                          GLOG_ZONE_GARDEN, (int8_t)intent->garden_index);

  // Log cooldown state change AFTER zone move (if applicable)
  if (!ecs_has(world, intent->alley_card, Charge)) {
    azk_log_card_tap_state_changed_ex(world, intent->alley_card, GLOG_TAP_COOLDOWN,
                                      GLOG_ZONE_GARDEN,
                                      (int8_t)intent->garden_index);
  }

  tap_card(world, intent->gate_card);

  // Trigger gate card's portal ability (if any)
  azk_trigger_gate_portal_ability(world, intent->gate_card, intent->alley_card,
                                  intent->player);

  return 0;
}
