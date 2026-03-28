#include "abilities/cards/azk01_028.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

bool azk01_028_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void azk01_028_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t hand = gs->zones[owner_num].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);

  ecs_entity_t to_discard[MAX_HAND_SIZE] = {0};
  uint8_t discard_count = 0;

  for (int32_t i = 0; i < hand_cards.count; ++i) {
    ecs_entity_t card = hand_cards.ids[i];
    if (card != 0 && discard_count < MAX_HAND_SIZE) {
      to_discard[discard_count++] = card;
    }
  }

  for (uint8_t i = 0; i < discard_count; ++i) {
    discard_card(world, to_discard[i]);
  }

  cli_render_logf("[AZK01-028] Discarded owner's hand");
}

void azk01_028_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t to_return[MAX_PLAYERS_PER_MATCH * GARDEN_SIZE] = {0};
  uint8_t return_count = 0;

  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    ecs_entities_t garden_cards =
        ecs_get_ordered_children(world, gs->zones[p].garden);
    for (int32_t i = 0; i < garden_cards.count; ++i) {
      ecs_entity_t target = garden_cards.ids[i];
      if (target == 0 || target == ctx->runtime.source_card) {
        continue;
      }

      if (!is_card_type(world, target, CARD_TYPE_ENTITY)) {
        continue;
      }

      if (return_count < MAX_PLAYERS_PER_MATCH * GARDEN_SIZE) {
        to_return[return_count++] = target;
      }
    }
  }

  for (uint8_t i = 0; i < return_count; ++i) {
    return_card_to_hand(world, to_return[i]);
  }

  cli_render_logf("[AZK01-028] Returned all other garden entities to hand");
}
