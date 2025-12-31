#include "abilities/cards/stt01_001.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// STT01-001: [Main] [Once/Turn] Pay 1 IKZ: Give a friendly garden entity
// equipped with a weapon Charge.

bool stt01_001_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);

  // Check if owner has any garden entity with weapon equipped and on cooldown
  ecs_entity_t garden = gs->zones[owner_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  for (int32_t i = 0; i < garden_cards.count; i++) {
    ecs_entity_t entity = garden_cards.ids[i];

    // Must have weapon equipped
    if (!has_equipped_weapon(world, entity)) {
      continue;
    }

    // Must be on cooldown (summoning sickness)
    if (!is_card_cooldown(world, entity)) {
      continue;
    }

    return true; // Found valid target
  }

  return false;
}

bool stt01_001_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);

  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  // Must be in owner's garden (not leader, not opponent's)
  if (parent != gs->zones[owner_num].garden) {
    return false;
  }

  // Must have weapon equipped
  if (!has_equipped_weapon(world, target)) {
    return false;
  }

  // Must be on cooldown (summoning sickness)
  if (!is_card_cooldown(world, target)) {
    return false;
  }

  return true;
}

void stt01_001_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect_targets[0];

  if (target == 0) {
    cli_render_logf("[STT01-001] No target for Charge effect");
    return;
  }

  // Add Charge tag
  ecs_add(world, target, Charge);

  // Clear cooldown but preserve tapped state
  const TapState *tap = ecs_get(world, target, TapState);
  ecs_set(world, target, TapState, {
    .tapped = tap ? tap->tapped : false,
    .cooldown = false
  });

  cli_render_logf("[STT01-001] Granted Charge to target entity");
}
