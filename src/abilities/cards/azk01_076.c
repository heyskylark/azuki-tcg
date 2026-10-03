#include "abilities/cards/azk01_076.h"

#include "components/components.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

#define AZK01_076_HEAL_AMOUNT 2

void azk01_076_apply_charge(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->runtime.source_card == 0) {
    return;
  }

  (void)apply_charge_grant(world, ctx->runtime.source_card,
                           TAG_GRANT_TICK_END_OF_TURN, 1);
}

void azk01_076_apply_heal(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t controller =
      ecs_get_target(world, ctx->runtime.source_card, Rel_OwnedBy, 0);
  if (controller == 0) {
    return;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, controller);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[player_num].leader);
  const BaseStats *base = ecs_get(world, leader, BaseStats);
  CurStats *current = ecs_get_mut(world, leader, CurStats);
  if (base == NULL || current == NULL || current->cur_hp >= base->health) {
    return;
  }

  const int16_t missing = (int16_t)base->health - (int16_t)current->cur_hp;
  const int8_t heal =
      (int8_t)(missing < AZK01_076_HEAL_AMOUNT ? missing
                                               : AZK01_076_HEAL_AMOUNT);
  current->cur_hp += heal;
  ecs_modified(world, leader, CurStats);
  azk_log_card_stat_change(world, leader, 0, heal, current->cur_atk,
                           current->cur_hp);
}
