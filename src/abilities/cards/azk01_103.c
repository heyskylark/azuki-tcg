#include "abilities/cards/azk01_103.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

static bool is_earth_garden_entity(ecs_world_t *world, ecs_entity_t entity,
                                   ecs_entity_t owner) {
  if (!is_card_type(world, entity, CARD_TYPE_ENTITY) ||
      get_card_element(world, entity) != CARD_ELEMENT_EARTH) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, entity, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

bool azk01_103_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (is_earth_garden_entity(world, garden_cards.ids[i], owner) &&
        !is_card_tapped(world, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_103_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  return target != 0 && is_earth_garden_entity(world, target, owner) &&
         !is_card_tapped(world, target);
}

bool azk01_103_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[0].leader || parent == gs->zones[1].leader;
}

void azk01_103_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count == 0) {
    return;
  }

  ecs_entity_t sacrificed = ctx->cost.entities[0];
  const CurStats *stats = ecs_get(world, sacrificed, CurStats);
  int8_t health = stats != NULL ? stats->cur_hp : 0;
  if (health < 0) {
    health = 0;
  }
  if (health > 5) {
    health = 5;
  }

  AbilityContext *ctx_mut = ecs_singleton_get_mut(world, AbilityContext);
  ctx_mut->scratch = (AbilityScratchState){
    .kind = ABILITY_SCRATCH_SACRIFICE_VALUE,
    .data.sacrifice_value =
        {
            .damage = health,
            .draw_after_effect = stats != NULL && stats->cur_hp >= 3,
        },
  };
  ecs_singleton_modified(world, AbilityContext);

  sacrifice_card(world, sacrificed);
}

void azk01_103_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 ||
      ctx->scratch.kind != ABILITY_SCRATCH_SACRIFICE_VALUE) {
    return;
  }

  ecs_entity_t leader = ctx->effect.entities[0];
  if (leader != 0 && ctx->scratch.data.sacrifice_value.damage > 0) {
    deal_effect_damage(world, leader, ctx->scratch.data.sacrifice_value.damage);
  }

  if (ctx->scratch.data.sacrifice_value.draw_after_effect) {
    draw_cards_with_deckout_check(world, ctx->runtime.owner, 1, NULL);
  }
}
