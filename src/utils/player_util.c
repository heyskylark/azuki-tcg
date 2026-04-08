#include "utils/player_util.h"
#include "abilities/ability_registry.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/ability_util.h"
#include "utils/card_utils.h"
#include "utils/zone_util.h"
#include "validation/action_validation.h"

uint8_t get_player_number(ecs_world_t *world, ecs_entity_t player) {
  const PlayerNumber *player_number = ecs_get(world, player, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", player);
  return player_number->player_number;
}

bool defender_can_respond(ecs_world_t *world, const GameState *gs,
                          uint8_t defender_index) {
  ecs_entity_t defender = gs->players[defender_index];
  ecs_entity_t hand = gs->zones[defender_index].hand;
  ecs_entity_t ikz_area = gs->zones[defender_index].ikz_area;
  uint8_t available_ikz =
      azk_count_tappable_ikz_sources(world, ikz_area, true);
  GameState response_preview = *gs;
  response_preview.active_player_index = defender_index;

  // Check if any card in hand is a response spell with affordable cost
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
  for (int i = 0; i < hand_cards.count; i++) {
    ecs_entity_t card = hand_cards.ids[i];

    if (!is_card_type(world, card, CARD_TYPE_SPELL))
      continue;

    ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
    uint8_t ability_count = azk_collect_card_action_abilities(
        world, card, abilities, AZK_MAX_CARD_ABILITIES);
    bool has_response_spell = false;
    for (uint8_t j = 0; j < ability_count; ++j) {
      if (azk_ability_has_timing(world, abilities[j], ecs_id(AResponse))) {
        has_response_spell = true;
        break;
      }
    }
    if (!has_response_spell) {
      continue;
    }

    // Check if we have the ability registered
    const CardId *card_id = ecs_get(world, card, CardId);
    if (!card_id || !azk_has_ability(card_id->id))
      continue;

    if (azk_get_effective_card_play_cost(world, gs->players[defender_index],
                                         card) <= available_ikz) {
      // Found at least one playable response spell
      return true;
    }
  }

  for (int i = 0; i < hand_cards.count; i++) {
    ecs_entity_t card = hand_cards.ids[i];
    if (!is_card_type(world, card, CARD_TYPE_ENTITY)) {
      continue;
    }
    if (!azk_can_play_card_from_hand_during_response_window(world, card)) {
      continue;
    }

    for (int use_token = 0; use_token <= 1; ++use_token) {
      UserAction garden_action = {
          .player = defender,
          .type = ACT_PLAY_ENTITY_TO_GARDEN,
          .subaction_1 = i,
          .subaction_3 = use_token,
      };
      for (int slot = 0; slot < GARDEN_SIZE; ++slot) {
        garden_action.subaction_2 = slot;
        if (azk_validate_play_entity_action(world, &response_preview, defender,
                                            ZONE_GARDEN, &garden_action, false,
                                            NULL)) {
          return true;
        }
      }

      UserAction alley_action = {
          .player = defender,
          .type = ACT_PLAY_ENTITY_TO_ALLEY,
          .subaction_1 = i,
          .subaction_3 = use_token,
      };
      for (int slot = 0; slot < ALLEY_SIZE; ++slot) {
        alley_action.subaction_2 = slot;
        if (azk_validate_play_entity_action(world, &response_preview, defender,
                                            ZONE_ALLEY, &alley_action, false,
                                            NULL)) {
          return true;
        }
      }
    }
  }

  for (int i = 0; i < hand_cards.count; i++) {
    ecs_entity_t card = hand_cards.ids[i];
    if (!is_card_type(world, card, CARD_TYPE_WEAPON)) {
      continue;
    }
    if (!azk_can_play_card_from_hand_during_response_window(world, card)) {
      continue;
    }

    for (int use_token = 0; use_token <= 1; ++use_token) {
      UserAction weapon_action = {
          .player = defender,
          .type = ACT_ATTACH_WEAPON_FROM_HAND,
          .subaction_1 = i,
          .subaction_3 = use_token,
      };
      for (int target = 0; target <= GARDEN_SIZE; ++target) {
        weapon_action.subaction_2 = target;
        if (azk_validate_attach_weapon_action(world, &response_preview,
                                              defender, &weapon_action, false,
                                              NULL)) {
          return true;
        }
      }
    }
  }

  ecs_entity_t response_cards[GARDEN_SIZE + ALLEY_SIZE + 1] = {0};
  int response_card_count = 0;

  ecs_entity_t garden = gs->zones[defender_index].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
  for (int i = 0; i < garden_cards.count && response_card_count < GARDEN_SIZE;
       i++) {
    response_cards[response_card_count++] = garden_cards.ids[i];
  }

  ecs_entity_t alley = gs->zones[defender_index].alley;
  ecs_entities_t alley_cards = ecs_get_ordered_children(world, alley);
  for (int i = 0; i < alley_cards.count &&
                  response_card_count < GARDEN_SIZE + ALLEY_SIZE;
       i++) {
    response_cards[response_card_count++] = alley_cards.ids[i];
  }

  ecs_entity_t leader_zone = gs->zones[defender_index].leader;
  ecs_entity_t leader = find_leader_card_in_zone(world, leader_zone);
  if (leader != 0 &&
      response_card_count < GARDEN_SIZE + ALLEY_SIZE + 1) {
    response_cards[response_card_count++] = leader;
  }

  for (int i = 0; i < response_card_count; i++) {
    ecs_entity_t response_card = response_cards[i];
    if (response_card == 0 || ecs_has(world, response_card, Frozen)) {
      continue;
    }

    ecs_entity_t response_abilities[AZK_MAX_CARD_ABILITIES] = {0};
    uint8_t response_ability_count = azk_collect_card_action_abilities(
        world, response_card, response_abilities, AZK_MAX_CARD_ABILITIES);
    for (uint8_t j = 0; j < response_ability_count; ++j) {
      ecs_entity_t response_ability = response_abilities[j];
      if (!azk_ability_has_timing(world, response_ability, ecs_id(AResponse))) {
        continue;
      }

      bool once_turn_blocked = false;
      if (ecs_has(world, response_ability, AOnceTurn)) {
        const AbilityRepeatContext *repeat_ctx =
            ecs_get(world, response_ability, AbilityRepeatContext);
        if (repeat_ctx && repeat_ctx->was_applied) {
          once_turn_blocked = true;
        }
      }
      if (once_turn_blocked) {
        continue;
      }

      const AbilityDef *def =
          azk_get_ability_def_for_entity(world, response_ability);
      if (!def) {
        continue;
      }

      if (def->ikz_cost > available_ikz) {
        continue;
      }

      if (!def->validate ||
          def->validate(world, response_card, gs->players[defender_index])) {
        return true;
      }
    }
  }

  // Check for declare defender option
  if (!gs->combat_state.defender_intercepted) {
    // Check if attacker has Infiltrate
    if (!ecs_has(world, gs->combat_state.attacking_card, Infiltrate)) {
      // Check for untapped entities with Defender tag
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
