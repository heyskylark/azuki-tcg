#include "abilities/cards/azk01_050.h"

#include "components/components.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static void heal_owner_leader(ecs_world_t *world, ecs_entity_t owner,
                              int8_t max_heal) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[player_num].leader);

  const BaseStats *base_stats = ecs_get(world, leader, BaseStats);
  CurStats *cur_stats = ecs_get_mut(world, leader, CurStats);
  if (base_stats == NULL || cur_stats == NULL) {
    return;
  }

  int16_t missing_hp = (int16_t)base_stats->health - (int16_t)cur_stats->cur_hp;
  int8_t heal_amount = (int8_t)(missing_hp <= 0
                                    ? 0
                                    : (missing_hp < max_heal ? missing_hp
                                                             : max_heal));
  if (heal_amount == 0) {
    return;
  }

  cur_stats->cur_hp += heal_amount;
  ecs_modified(world, leader, CurStats);
  azk_log_card_stat_change(world, leader, 0, heal_amount, cur_stats->cur_atk,
                           cur_stats->cur_hp);
}

bool azk01_050_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void azk01_050_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  heal_owner_leader(world, ctx->runtime.owner, 2);
}
