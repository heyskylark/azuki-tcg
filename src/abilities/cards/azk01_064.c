#include "abilities/cards/azk01_064.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool azk01_064_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

void azk01_064_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  (void)ctx;

  const GameState *gs = ecs_singleton_get(world, GameState);
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    ecs_entities_t garden_cards =
        ecs_get_ordered_children(world, gs->zones[p].garden);
    for (int32_t i = 0; i < garden_cards.count; ++i) {
      ecs_entity_t target = garden_cards.ids[i];
      if (target != 0 && is_card_type(world, target, CARD_TYPE_ENTITY)) {
        deal_effect_damage(world, target, 2);
      }
    }
  }
}
