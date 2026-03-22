#include "abilities/cards/azk01_055.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_055_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

bool azk01_055_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

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
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t enemy_num = owner_num == 0 ? 1 : 0;
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[enemy_num].garden;
}

void azk01_055_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  deal_effect_damage(world, target, 1);
  apply_attack_modifier(world, target, ctx->runtime.source_card, -1, true);
}
