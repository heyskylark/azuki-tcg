#include "utils/card_utils.h"

#include "abilities/ability_system.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/entity_util.h"
#include "utils/player_util.h"
#include <stdio.h>

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type) {
  const Type *card_type = ecs_get(world, card, Type);
  ecs_assert(card_type != NULL, ECS_INVALID_PARAMETER,
             "Type component not found for card %d", card);
  return card_type->value == type;
}

void discard_card(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER,
             "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t discard_zone = gs->zones[player_number->player_number].discard;

  ecs_remove_id(world, card, ecs_id(ZoneIndex));
  ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
  // Reset current stats to base stats (consistent with return_card_to_hand)
  const BaseStats *base = ecs_get(world, card, BaseStats);
  if (base && ecs_has(world, card, CurStats)) {
    ecs_set(world, card, CurStats,
            {.cur_atk = base->attack, .cur_hp = base->health});
  }
  ecs_add_pair(world, card, EcsChildOf, discard_zone);
}

void return_card_to_hand(ecs_world_t *world, ecs_entity_t card) {
  // Check source zone BEFORE changing ChildOf
  ecs_entity_t source_parent = ecs_get_target(world, card, EcsChildOf, 0);

  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER,
             "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = player_number->player_number;
  ecs_entity_t hand_zone = gs->zones[player_num].hand;

  // Determine if card is returning from play (garden/alley)
  bool from_play = false;
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
    if (source_parent == gs->zones[p].garden ||
        source_parent == gs->zones[p].alley) {
      from_play = true;
      break;
    }
  }

  // Discard any equipped weapons before returning to hand
  discard_equipped_weapon_cards(world, card);
  // Remove zone index if present (entity was in garden/alley)
  ecs_remove_id(world, card, ecs_id(ZoneIndex));
  // Reset tap state
  ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
  // Reset current stats to base stats
  const BaseStats *base = ecs_get(world, card, BaseStats);
  if (base) {
    ecs_set(world, card, CurStats,
            {.cur_atk = base->attack, .cur_hp = base->health});
  }
  // Move to hand
  ecs_add_pair(world, card, EcsChildOf, hand_zone);

  cli_render_logf("[CardUtils] Returned card to hand");

  // Trigger observers only if card came from play (garden/alley)
  if (from_play) {
    GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
    gs_mut->entities_returned_to_hand_this_turn[player_num]++;
    ecs_singleton_modified(world, GameState);
    azk_trigger_return_to_hand_observers(world, card);
  }
}

void set_card_to_tapped(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  ecs_set(world, card, TapState,
          {.tapped = true, .cooldown = tap_state->cooldown});
}

void set_card_to_cooldown(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  ecs_set(world, card, TapState,
          {.tapped = tap_state->tapped, .cooldown = true});
}

bool is_card_tapped(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  return tap_state->tapped;
}

bool is_card_cooldown(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  return tap_state->cooldown;
}

bool is_weapon_card(ecs_world_t *world, ecs_entity_t card) {
  const Type *card_type = ecs_get(world, card, Type);
  if (!card_type) {
    return false;
  }
  return card_type->value == CARD_TYPE_WEAPON;
}

int count_weapons_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t children = ecs_get_ordered_children(world, zone);
  int count = 0;
  for (int32_t i = 0; i < children.count; i++) {
    if (is_weapon_card(world, children.ids[i])) {
      count++;
    }
  }
  return count;
}

bool has_subtype(ecs_world_t *world, ecs_entity_t card, ecs_id_t subtype_tag) {
  if (card == 0 || subtype_tag == 0) {
    return false;
  }
  return ecs_has_id(world, card, subtype_tag);
}

bool is_watercrafting_card(ecs_world_t *world, ecs_entity_t card) {
  return has_subtype(world, card, ecs_id(TSubtype_Watercrafting));
}

bool is_water_element_card(ecs_world_t *world, ecs_entity_t card) {
  const Element *elem = ecs_get(world, card, Element);
  if (!elem) {
    return false;
  }
  return elem->element == (uint8_t)CARD_ELEMENT_WATER;
}

int count_subtype_in_zone(ecs_world_t *world, ecs_entity_t zone,
                          ecs_id_t subtype_tag) {
  ecs_entities_t children = ecs_get_ordered_children(world, zone);
  int count = 0;
  for (int32_t i = 0; i < children.count; i++) {
    if (has_subtype(world, children.ids[i], subtype_tag)) {
      count++;
    }
  }
  return count;
}
