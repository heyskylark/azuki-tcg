#include "abilities/cards/azk01_119.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_119_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (has_equipped_weapon(world, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_119_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY) ||
      !has_equipped_weapon(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

void azk01_119_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  const int weapon_count =
      count_weapons_in_zone(world, gs->zones[owner_num].discard);
  int8_t attack_buff = (int8_t)weapon_count;
  if (attack_buff > 3) {
    attack_buff = 3;
  }

  if (attack_buff > 0) {
    apply_attack_modifier(world, ctx->effect.entities[0],
                          ctx->runtime.source_card, attack_buff, true);
  }
}
