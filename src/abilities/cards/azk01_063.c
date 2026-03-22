#include "abilities/cards/azk01_063.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static bool is_leader_or_garden_entity(ecs_world_t *world, ecs_entity_t target) {
  if (target == 0) {
    return false;
  }

  if (ecs_has(world, target, TLeader)) {
    return true;
  }

  if (!is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    if (parent == gs->zones[p].garden) {
      return true;
    }
  }

  return false;
}

bool azk01_063_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

bool azk01_063_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;
  return is_leader_or_garden_entity(world, target);
}

void azk01_063_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    deal_effect_damage(world, target, 3);
  }
}
