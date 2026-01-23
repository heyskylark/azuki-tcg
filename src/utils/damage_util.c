#include "utils/damage_util.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool deal_effect_damage(ecs_world_t *world, ecs_entity_t target,
                        int8_t damage) {
  // Check if target is effect immune
  if (is_effect_immune(world, target)) {
    cli_render_logf("[Damage] Effect damage blocked by EffectImmune");
    return false;
  }

  // Apply damage to target's current HP
  CurStats *cur_stats = ecs_get_mut(world, target, CurStats);
  if (cur_stats == NULL) {
    cli_render_logf("[Damage] Target has no CurStats component");
    return false;
  }

  cur_stats->cur_hp -= damage;
  ecs_modified(world, target, CurStats);

  // Log the HP change for frontend state updates
  azk_log_card_stat_change(world, target, 0, -damage, cur_stats->cur_atk,
                           cur_stats->cur_hp);

  cli_render_logf("[Damage] Dealt %d effect damage (HP: %d)", damage,
                  cur_stats->cur_hp);

  // Check for death (HP <= 0)
  if (cur_stats->cur_hp <= 0) {
    if (ecs_has(world, target, TLeader)) {
      // Leader defeated - determine winner based on target's owner
      GameState *gs = ecs_singleton_get_mut(world, GameState);
      ecs_entity_t target_parent = ecs_get_target(world, target, EcsChildOf, 0);

      // Find which player owns this leader
      for (int i = 0; i < MAX_PLAYERS_PER_MATCH; i++) {
        if (target_parent == gs->zones[i].leader) {
          // Owner of defeated leader loses, opponent wins
          gs->winner = (i + 1) % MAX_PLAYERS_PER_MATCH;
          ecs_singleton_modified(world, GameState);
          cli_render_logf("[Damage] Leader defeated - player %d wins",
                          gs->winner);
          break;
        }
      }
    } else {
      // Non-leader entity - discard it
      discard_card(world, target);
      cli_render_logf("[Damage] Entity defeated by effect damage - discarded");
    }
  }

  return true;
}
