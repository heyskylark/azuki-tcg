#include "abilities/cards/azk01_002.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// AZK01-002 "Healing Flutter": [Main] Heal 2 to your leader.

bool azk01_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[player_num].leader);
  const CurStats *cur_stats = ecs_get(world, leader, CurStats);
  ecs_assert(cur_stats != NULL, ECS_INVALID_PARAMETER,
             "Leader %d missing CurStats", leader);
  return cur_stats->cur_hp > 0;
}

void azk01_002_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[player_num].leader);

  const BaseStats *base_stats = ecs_get(world, leader, BaseStats);
  CurStats *cur_stats = ecs_get_mut(world, leader, CurStats);
  ecs_assert(base_stats != NULL, ECS_INVALID_PARAMETER,
             "Leader %d missing BaseStats", leader);
  ecs_assert(cur_stats != NULL, ECS_INVALID_PARAMETER,
             "Leader %d missing CurStats", leader);
  int16_t missing_hp = (int16_t)base_stats->health - (int16_t)cur_stats->cur_hp;
  int8_t heal_amount = (int8_t)(missing_hp <= 0 ? 0
                                                : (missing_hp < 2 ? missing_hp
                                                                  : 2));
  if (heal_amount == 0) {
    return;
  }
  cur_stats->cur_hp += heal_amount;
  ecs_modified(world, leader, CurStats);

  azk_log_card_stat_change(world, leader, 0, heal_amount, cur_stats->cur_atk,
                           cur_stats->cur_hp);
  cli_render_logf("[AZK01-002] Healed owner's leader for %d (HP: %d)",
                  heal_amount, cur_stats->cur_hp);
}
