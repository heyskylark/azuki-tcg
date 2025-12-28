#include "abilities/cards/stt01_013.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// STT01-013 "Black Jade Dagger": On Play; You may deal damage to your leader:
// This card gives an additional +1 attack

bool stt01_013_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[player_num].leader);

  // Check if leader has at least 1 HP to pay the cost
  const CurStats *leader_stats = ecs_get(world, leader, CurStats);
  ecs_assert(leader_stats != NULL, ECS_INVALID_PARAMETER,
             "CurStats not found for leader");

  return leader_stats->cur_hp >= 1;
}

void stt01_013_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[player_num].leader);

  // Deal 1 damage to owner's leader (may be blocked by EffectImmune)
  // The +1 attack effect still applies regardless of whether damage is blocked
  bool damage_dealt = deal_effect_damage(world, leader, 1);
  if (damage_dealt) {
    const CurStats *leader_stats = ecs_get(world, leader, CurStats);
    cli_render_logf("[STT01-013] Dealt 1 damage to leader (HP: %d)",
                    leader_stats->cur_hp);
  } else {
    cli_render_logf("[STT01-013] Damage to leader blocked by EffectImmune");
  }
}

void stt01_013_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  // Find the attached target (weapon is child of target entity)
  ecs_entity_t target = ecs_get_target(world, ctx->source_card, EcsChildOf, 0);
  ecs_assert(target != 0, ECS_INVALID_PARAMETER,
             "Weapon is not attached to any entity");

  // Update weapon's CurStats.cur_atk (+1)
  CurStats *weapon_stats = ecs_get_mut(world, ctx->source_card, CurStats);
  ecs_assert(weapon_stats != NULL, ECS_INVALID_PARAMETER,
             "CurStats not found for weapon");

  weapon_stats->cur_atk += 1;
  ecs_modified(world, ctx->source_card, CurStats);

  // Update target entity's CurStats.cur_atk (+1)
  CurStats *target_stats = ecs_get_mut(world, target, CurStats);
  ecs_assert(target_stats != NULL, ECS_INVALID_PARAMETER,
             "CurStats not found for target entity");

  target_stats->cur_atk += 1;
  ecs_modified(world, target, CurStats);

  cli_render_logf("[STT01-013] Weapon gained +1 attack (total: %d)",
                  weapon_stats->cur_atk);
}
