#include "utils/damage_util.h"

#include "components/abilities.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
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

  cli_render_logf("[Damage] Dealt %d effect damage (HP: %d)", damage,
                  cur_stats->cur_hp);
  return true;
}
