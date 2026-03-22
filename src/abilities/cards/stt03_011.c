#include "abilities/cards/stt03_011.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

bool stt03_011_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  const uint8_t enemy_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t cards = ecs_get_ordered_children(world, gs->zones[enemy_num].garden);
  for (int32_t i = 0; i < cards.count; ++i) {
    const BaseStats *base = ecs_get(world, cards.ids[i], BaseStats);
    if (base != NULL && base->health <= 2) {
      return true;
    }
  }

  return false;
}

bool stt03_011_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t enemy_num =
      (get_player_number(world, owner) + 1) % MAX_PLAYERS_PER_MATCH;
  if (ecs_get_target(world, target, EcsChildOf, 0) != gs->zones[enemy_num].garden) {
    return false;
  }

  const BaseStats *base = ecs_get(world, target, BaseStats);
  return base != NULL && base->health <= 2;
}

void stt03_011_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    discard_card(world, ctx->effect.entities[0]);
  }
}
