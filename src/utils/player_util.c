#include "utils/player_util.h"
#include "abilities/ability_registry.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/zone_util.h"

uint8_t get_player_number(ecs_world_t *world, ecs_entity_t player) {
  const PlayerNumber *player_number = ecs_get(world, player, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", player);
  return player_number->player_number;
}

bool defender_can_respond(ecs_world_t *world, const GameState *gs,
                          uint8_t defender_index) {
  ecs_entity_t hand = gs->zones[defender_index].hand;
  ecs_entity_t ikz_area = gs->zones[defender_index].ikz_area;

  // Count available untapped IKZ
  ecs_entities_t ikz_cards = ecs_get_ordered_children(world, ikz_area);
  uint8_t available_ikz = 0;
  for (int i = 0; i < ikz_cards.count; i++) {
    if (!is_card_tapped(world, ikz_cards.ids[i])) {
      available_ikz++;
    }
  }

  // Check IKZ token
  ecs_entity_t defender_player = gs->players[defender_index];
  const IKZToken *ikz_token = ecs_get(world, defender_player, IKZToken);
  if (ikz_token && ikz_token->ikz_token != 0 &&
      !is_card_tapped(world, ikz_token->ikz_token)) {
    available_ikz++;
  }

  // Check if any card in hand is a response spell with affordable cost
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
  for (int i = 0; i < hand_cards.count; i++) {
    ecs_entity_t card = hand_cards.ids[i];

    // Check if it's a spell with AResponse timing
    if (!is_card_type(world, card, CARD_TYPE_SPELL))
      continue;
    if (!ecs_has(world, card, AResponse))
      continue;

    // Check if we have the ability registered
    const CardId *card_id = ecs_get(world, card, CardId);
    if (!card_id || !azk_has_ability(card_id->id))
      continue;

    // Check IKZ cost
    const IKZCost *ikz_cost = ecs_get(world, card, IKZCost);
    if (!ikz_cost)
      continue;

    if (ikz_cost->ikz_cost <= available_ikz) {
      // Found at least one playable response spell
      return true;
    }
  }

  // Check for declare defender option
  if (!gs->combat_state.defender_intercepted) {
    // Check if attacker has Infiltrate
    if (!ecs_has(world, gs->combat_state.attacking_card, Infiltrate)) {
      // Check for untapped entities with Defender tag
      ecs_entity_t garden = gs->zones[defender_index].garden;
      ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
      for (int i = 0; i < garden_cards.count; i++) {
        ecs_entity_t card = garden_cards.ids[i];
        if (ecs_has(world, card, Defender) && !is_card_tapped(world, card)) {
          return true;
        }
      }
    }
  }

  return false;
}
