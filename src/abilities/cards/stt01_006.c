#include "abilities/cards/stt01_006.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

// STT01-006 "Silver Current, Haruhi": [Once/Turn][When Attacking] Deal 1 damage
// to a leader or entity in your opponent's garden.

bool stt01_006_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  // Check opponent garden for at least one non-EffectImmune entity
  ecs_entity_t garden = gs->zones[opponent_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
  for (int32_t i = 0; i < garden_cards.count; i++) {
    if (!is_effect_immune(world, garden_cards.ids[i])) {
      return true;
    }
  }

  // Check opponent leader (if not EffectImmune)
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[opponent_num].leader);
  if (leader != 0 && !is_effect_immune(world, leader)) {
    return true;
  }

  return false;
}

bool stt01_006_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  // Target must not have EffectImmune
  if (is_effect_immune(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  // Check if target is in opponent's garden
  if (parent == gs->zones[opponent_num].garden) {
    return true;
  }

  // Check if target is opponent's leader
  if (parent == gs->zones[opponent_num].leader) {
    return true;
  }

  return false;
}

void stt01_006_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect_targets[0];

  if (target == 0) {
    cli_render_logf("[STT01-006] No target for damage");
    return;
  }

  // Deal 1 damage (deal_effect_damage handles EffectImmune check)
  bool damage_dealt = deal_effect_damage(world, target, 1);
  if (damage_dealt) {
    const CurStats *target_stats = ecs_get(world, target, CurStats);
    if (target_stats) {
      cli_render_logf("[STT01-006] Dealt 1 damage to target (HP: %d)",
                      target_stats->cur_hp);
    }
  } else {
    cli_render_logf("[STT01-006] Damage blocked by EffectImmune");
  }
}
