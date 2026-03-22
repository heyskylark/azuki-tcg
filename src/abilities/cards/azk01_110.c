#include "abilities/cards/azk01_110.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_110_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (garden_cards.ids[i] != card) {
      return true;
    }
  }

  return false;
}

void azk01_110_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);

  ecs_entity_t to_sacrifice[GARDEN_SIZE] = {0};
  uint8_t sacrifice_count = 0;
  for (int32_t i = 0; i < garden_cards.count && sacrifice_count < GARDEN_SIZE; ++i) {
    ecs_entity_t card = garden_cards.ids[i];
    if (card != 0 && card != ctx->runtime.source_card) {
      to_sacrifice[sacrifice_count++] = card;
    }
  }

  for (uint8_t i = 0; i < sacrifice_count; ++i) {
    sacrifice_card(world, to_sacrifice[i]);
  }

  if (sacrifice_count > 0) {
    apply_attack_modifier(world, ctx->runtime.source_card, ctx->runtime.source_card,
                          (int8_t)sacrifice_count, true);
  }
}
