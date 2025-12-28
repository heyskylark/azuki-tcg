#include "abilities/cards/stt02_011.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

// STT02-011: "Garden only; Main; You may sacrifice this card: choose an entity
// in your garden; it cannot take damage from card effects until the start of
// your next turn."

bool stt02_011_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);

  // Card must be in garden (garden only ability)
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  if (parent != gs->zones[player_num].garden) {
    return false;
  }

  // Must have at least one other entity in garden to target
  // (since this card will be sacrificed, we need a different target)
  ecs_entity_t garden = gs->zones[player_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  // Count entities (not including this card)
  int entity_count = 0;
  for (int32_t i = 0; i < garden_cards.count; i++) {
    if (garden_cards.ids[i] != card) {
      entity_count++;
    }
  }

  return entity_count >= 1;
}

bool stt02_011_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0) {
    return false;
  }

  // Target cannot be the source card (it's being sacrificed)
  if (target == card) {
    return false;
  }

  // Target must be in owner's garden
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t garden = gs->zones[player_num].garden;

  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == garden;
}

void stt02_011_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  // Sacrifice this card (move to discard)
  discard_card(world, ctx->source_card);
  cli_render_logf("[STT02-011] Sacrificed card");
}

void stt02_011_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect_targets[0];

  if (target == 0) {
    cli_render_logf("[STT02-011] No target to apply EffectImmune");
    return;
  }

  // Apply EffectImmune status with duration 2
  // Duration 2 means: expires at start of owner's next turn
  // (decrements at enemy's turn start, then again at owner's turn start)
  apply_effect_immune(world, target, 2);

  cli_render_logf("[STT02-011] Applied EffectImmune to target for 2 turns");
}
