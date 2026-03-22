#include "abilities/cards/azk01_078.h"

#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool azk01_078_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

bool azk01_078_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  const uint8_t enemy_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  const ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  if (parent == gs->zones[enemy_num].garden) {
    return is_card_type(world, target, CARD_TYPE_ENTITY);
  }

  return parent == gs->zones[enemy_num].leader;
}

void azk01_078_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    deal_effect_damage(world, target, 1);
  }
}
