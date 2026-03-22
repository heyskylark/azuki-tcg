#include "utils/entity_util.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/status_util.h"
#include "utils/weapon_util.h"

void reset_entity_health(ecs_world_t *world, ecs_entity_t entity) {
  const CurStats *before_reset = ecs_get(world, entity, CurStats);
  int8_t prev_atk = 0;
  int8_t prev_hp = 0;
  bool had_stats_before = false;
  if (before_reset) {
    prev_atk = before_reset->cur_atk;
    prev_hp = before_reset->cur_hp;
    had_stats_before = true;
  }

  // Recalculate health from base stats + any active health buffs
  // This heals damage while preserving passive health buffs (e.g., stt02_012)
  recalculate_health_from_buffs(world, entity);

  if (!had_stats_before) {
    return;
  }

  const CurStats *after_reset = ecs_get(world, entity, CurStats);
  if (!after_reset) {
    return;
  }

  int8_t atk_delta = (int8_t)(after_reset->cur_atk - prev_atk);
  int8_t hp_delta = (int8_t)(after_reset->cur_hp - prev_hp);

  if (atk_delta == 0 && hp_delta == 0) {
    return;
  }

  azk_log_card_stat_change(world, entity, atk_delta, hp_delta,
                           after_reset->cur_atk, after_reset->cur_hp);
}

static void discard_weapon_card(ecs_world_t *world, ecs_entity_t entity,
                                ecs_entity_t weapon_card) {
  // Get weapon attack BEFORE discarding (entity will be deleted/moved)
  const CurStats *weapon_stats = ecs_get(world, weapon_card, CurStats);
  int8_t weapon_atk = weapon_stats ? weapon_stats->cur_atk : 0;

  remove_weapon_combat_modifier_if_any(world, weapon_card, entity);

  // Discard the weapon card (removes ChildOf relationship, observers will fire)
  discard_card(world, weapon_card);

  // Directly subtract weapon attack from entity's CurStats
  // We can't use recalculate_attack_from_buffs because:
  // 1. ChildOf removal is deferred (weapon still visible as child)
  // 2. AttackBuff pair changes from observers are also deferred
  const CurStats *entity_stats = ecs_get(world, entity, CurStats);
  if (entity_stats) {
    int16_t new_atk = entity_stats->cur_atk - weapon_atk;
    if (new_atk < 0) new_atk = 0;
    ecs_set(world, entity, CurStats, {
      .cur_atk = (int8_t)new_atk,
      .cur_hp = entity_stats->cur_hp,
    });
    azk_log_card_stat_change(world, entity, (int8_t)(-weapon_atk), 0,
                             (int8_t)new_atk, entity_stats->cur_hp);
    cli_render_logf("[Weapon] Detached weapon (-%d attack) from entity",
                    weapon_atk);
  }
}

void discard_equipped_weapon_cards(ecs_world_t *world, ecs_entity_t entity) {
  // Collect weapon cards first to avoid iterator invalidation
  ecs_entity_t weapons[16];
  int weapon_count = 0;

  ecs_iter_t it = ecs_children(world, entity);
  while (ecs_children_next(&it) && weapon_count < 16) {
    for (int i = 0; i < it.count && weapon_count < 16; i++) {
      ecs_entity_t child = it.entities[i];
      if (ecs_has_id(world, child, TWeapon)) {
        weapons[weapon_count++] = child;
      }
    }
  }

  // Discard collected weapons
  for (int i = 0; i < weapon_count; i++) {
    discard_weapon_card(world, entity, weapons[i]);
  }
}
