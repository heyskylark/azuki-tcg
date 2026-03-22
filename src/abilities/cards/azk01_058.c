#include "abilities/cards/azk01_058.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_058_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) == gs->zones[owner_num].garden;
}

bool azk01_058_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  const ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  if (parent == gs->zones[owner_num].garden) {
    return is_card_type(world, target, CARD_TYPE_ENTITY);
  }

  return parent == gs->zones[0].leader || parent == gs->zones[1].leader;
}

void azk01_058_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  sacrifice_card(world, ctx->runtime.source_card);
}

void azk01_058_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    apply_attack_modifier(world, target, ctx->runtime.source_card, 2, true);
  }
}
