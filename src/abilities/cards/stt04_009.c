#include "abilities/cards/stt04_009.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool stt04_009_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  const DamageTracker *tracker = ecs_get(world, card, DamageTracker);
  return tracker != NULL && tracker->took_damage_this_turn &&
         tracker->last_taken_from_effect;
}

bool stt04_009_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)owner;
  if (target == 0 || target == card) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[0].garden || parent == gs->zones[1].garden ||
         parent == gs->zones[0].leader || parent == gs->zones[1].leader;
}

void stt04_009_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  const DamageTracker *tracker =
      ecs_get(world, ctx->runtime.source_card, DamageTracker);
  if (tracker == NULL) {
    return;
  }

  int8_t damage = tracker->last_damage_taken;
  if (damage > 2) {
    damage = 2;
  }
  if (damage > 0) {
    deal_effect_damage(world, ctx->effect.entities[0], damage);
  }
}
