#include "abilities/cards/azk01_104.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static ecs_entity_t owner_leader(ecs_world_t *world, ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  return find_leader_card_in_zone(world, gs->zones[player_num].leader);
}

bool azk01_104_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  return ecs_get_ordered_children(world, gs->zones[owner_num].hand).count > 0 &&
         owner_leader(world, owner) != 0;
}

bool azk01_104_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) == gs->zones[owner_num].hand;
}

void azk01_104_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count > 0 && ctx->cost.entities[0] != 0) {
    discard_card(world, ctx->cost.entities[0]);
  }
}

void azk01_104_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t leader = owner_leader(world, ctx->runtime.owner);
  const BaseStats *base = leader != 0 ? ecs_get(world, leader, BaseStats) : NULL;
  CurStats *cur = leader != 0 ? ecs_get_mut(world, leader, CurStats) : NULL;
  if (base == NULL || cur == NULL) {
    return;
  }

  int16_t missing_hp = (int16_t)base->health - (int16_t)cur->cur_hp;
  int8_t heal_amount =
      (int8_t)(missing_hp <= 0 ? 0 : (missing_hp < 2 ? missing_hp : 2));
  if (heal_amount <= 0) {
    return;
  }

  cur->cur_hp += heal_amount;
  ecs_modified(world, leader, CurStats);
  azk_log_card_stat_change(world, leader, 0, heal_amount, cur->cur_atk,
                           cur->cur_hp);
}
