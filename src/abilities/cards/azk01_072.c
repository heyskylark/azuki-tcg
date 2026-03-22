#include "abilities/cards/azk01_072.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_072_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);

  for (int32_t i = 0; i < garden_cards.count; ++i) {
    ecs_entity_t target = garden_cards.ids[i];
    if (target != card &&
        has_subtype(world, target, ecs_id(TSubtype_Beanz))) {
      return true;
    }
  }

  return false;
}

bool azk01_072_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card ||
      !has_subtype(world, target, ecs_id(TSubtype_Beanz))) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

void azk01_072_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target != 0) {
    apply_attack_modifier(world, target, ctx->runtime.source_card, 1, true);
  }
}
