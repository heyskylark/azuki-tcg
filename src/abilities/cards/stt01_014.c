#include "abilities/cards/stt01_014.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// STT01-014 "Tenshin": [On Play] Deal up to 1 damage to a leader.

bool stt01_014_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  (void)owner;
  (void)world;

  // Leaders always exist, so the ability can always be activated
  return true;
}

bool stt01_014_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  // Target must be a leader (has TLeader tag)
  if (!ecs_has(world, target, TLeader)) {
    return false;
  }

  return true;
}

void stt01_014_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  // For "up to" effects, player can skip target selection (effect_filled == 0)
  if (ctx->effect_filled == 0) {
    cli_render_logf("[STT01-014] Skipped damage (chose not to damage leader)");
    return;
  }

  ecs_entity_t target = ctx->effect_targets[0];

  if (target == 0) {
    cli_render_logf("[STT01-014] No target for damage");
    return;
  }

  // Deal 1 damage (deal_effect_damage handles EffectImmune check)
  bool damage_dealt = deal_effect_damage(world, target, 1);
  if (damage_dealt) {
    const CurStats *target_stats = ecs_get(world, target, CurStats);
    if (target_stats) {
      cli_render_logf("[STT01-014] Dealt 1 damage to leader (HP: %d)",
                      target_stats->cur_hp);
    }
  } else {
    cli_render_logf("[STT01-014] Damage blocked by EffectImmune");
  }
}
