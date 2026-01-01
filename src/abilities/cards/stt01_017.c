#include "abilities/cards/stt01_017.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// STT01-017 "Lightning Orb": [Response] Deal 1 damage to an entity in your
// opponent's garden and 1 damage to another entity in your opponent's garden.

bool stt01_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  // Check opponent garden for at least one entity
  ecs_entity_t garden = gs->zones[opponent_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  // Spell can be cast if opponent has at least 1 entity in garden
  return garden_cards.count >= 1;
}

bool stt01_017_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  // Target must be in opponent's garden (not leader)
  if (parent != gs->zones[opponent_num].garden) {
    return false;
  }

  // This card requires selecting DIFFERENT entities ("another entity")
  // Check if target is already selected
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  for (int i = 0; i < ctx->effect_filled; i++) {
    if (ctx->effect_targets[i] == target) {
      return false;
    }
  }

  return true;
}

void stt01_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  // Deal 1 damage to each selected target (1 or 2 targets)
  for (int i = 0; i < ctx->effect_filled; i++) {
    ecs_entity_t target = ctx->effect_targets[i];

    if (target == 0) {
      continue;
    }

    // Deal 1 damage (deal_effect_damage handles EffectImmune check)
    bool damage_dealt = deal_effect_damage(world, target, 1);
    if (damage_dealt) {
      const CurStats *target_stats = ecs_get(world, target, CurStats);
      if (target_stats) {
        cli_render_logf("[STT01-017] Dealt 1 damage to target (HP: %d)",
                        target_stats->cur_hp);
      }
    } else {
      cli_render_logf("[STT01-017] Damage blocked by EffectImmune");
    }
  }
}
