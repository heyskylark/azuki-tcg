#include "utils/weapon_util.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/status_util.h"

bool apply_weapon_attack_bonus(ecs_world_t *world, ecs_entity_t target_card,
                               int8_t weapon_atk) {
  const CurStats *target_stats = ecs_get(world, target_card, CurStats);
  if (!target_stats) {
    cli_render_logf("[Weapon] Missing CurStats for target %llu",
                    (unsigned long long)target_card);
    return false;
  }

  int16_t new_atk = target_stats->cur_atk + weapon_atk;
  if (new_atk < 0)
    new_atk = 0;
  ecs_set(world, target_card, CurStats, {
    .cur_atk = (int8_t)new_atk,
    .cur_hp = target_stats->cur_hp,
  });

  azk_log_card_stat_change(world, target_card, weapon_atk, 0,
                           (int8_t)new_atk, target_stats->cur_hp);

  return true;
}

int attach_weapon_from_hand(ecs_world_t *world,
                            const AttachWeaponIntent *intent) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER,
             "AttachWeaponIntent is null");

  const CurStats *weapon_cur_stats =
      ecs_get(world, intent->weapon_card, CurStats);
  ecs_assert(weapon_cur_stats != NULL, ECS_INVALID_PARAMETER,
             "CurStats component not found for weapon %llu",
             (unsigned long long)intent->weapon_card);

  const CurStats *entity_cur_stats =
      ecs_get(world, intent->target_card, CurStats);
  ecs_assert(entity_cur_stats != NULL, ECS_INVALID_PARAMETER,
             "CurStats component not found for card %llu",
             (unsigned long long)intent->target_card);

  // Log zone movement from hand to equipped (before reparenting changes parent)
  azk_log_card_zone_moved(world, intent->weapon_card,
                          GLOG_ZONE_HAND, intent->hand_index,
                          GLOG_ZONE_EQUIPPED, -1);

  // Attach weapon as child (observers will fire here, queuing passive buffs)
  // Note: ChildOf relationship is deferred, so weapon won't be visible as child
  // until next frame. We directly add weapon attack below.
  ecs_add_pair(world, intent->weapon_card, EcsChildOf, intent->target_card);

  // Directly add weapon attack to target's CurStats
  // We can't use recalculate_attack_from_buffs because:
  // 1. ChildOf relationship is deferred (weapon not visible as child yet)
  // 2. AttackBuff pairs from observers are also deferred
  apply_weapon_attack_bonus(world, intent->target_card,
                            weapon_cur_stats->cur_atk);

  cli_render_logf("[Weapon] Attached weapon (+%d attack) to entity",
                  weapon_cur_stats->cur_atk);

  for (uint8_t i = 0; i < intent->ikz_card_count; ++i) {
    tap_card(world, intent->ikz_cards[i]);
  }

  return 0;
}
