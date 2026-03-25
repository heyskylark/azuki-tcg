#include "abilities/cards/stt03_010.h"

#include "components/components.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static void heal_owner_leader(ecs_world_t *world, ecs_entity_t owner,
                              int8_t max_heal) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, owner);
  const ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[player_num].leader);
  const BaseStats *base = leader != 0 ? ecs_get(world, leader, BaseStats) : NULL;
  CurStats *cur = leader != 0 ? ecs_get_mut(world, leader, CurStats) : NULL;
  if (base == NULL || cur == NULL) {
    return;
  }

  const int16_t missing_hp = (int16_t)base->health - (int16_t)cur->cur_hp;
  const int8_t heal_amount =
      (int8_t)(missing_hp <= 0 ? 0 : (missing_hp < max_heal ? missing_hp : max_heal));
  if (heal_amount <= 0) {
    return;
  }

  cur->cur_hp += heal_amount;
  ecs_modified(world, leader, CurStats);
  azk_log_card_stat_change(world, leader, 0, heal_amount, cur->cur_atk,
                           cur->cur_hp);
}

bool stt03_010_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)owner;

  const GameState *gs = ecs_singleton_get(world, GameState);
  return gs != NULL && gs->last_combat.attacker == card &&
         gs->last_combat.defender_destroyed &&
         gs->last_combat.defender_was_garden_entity;
}

void stt03_010_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  heal_owner_leader(world, ctx->runtime.owner, 1);
}
