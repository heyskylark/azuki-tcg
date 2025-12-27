#include "abilities/cards/stt02_014.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

// STT02-014 "Chilling Water": [Main] Freeze an entity with cost <= 2 in
// opponent's garden for 2 turns

// Helper to check if an entity has cost <= 2
static bool is_valid_freeze_target(ecs_world_t *world, ecs_entity_t entity) {
  // Must be an entity card
  if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
    return false;
  }

  // Must have IKZ cost <= 2
  const IKZCost *cost = ecs_get(world, entity, IKZCost);
  ecs_assert(cost != NULL, ECS_INVALID_PARAMETER,
             "IKZ cost component not found for entity %d", entity);

  return cost->ikz_cost <= 2;
}

bool stt02_014_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  // Check opponent's garden for valid targets (entities with cost <= 2)
  ecs_entity_t garden = gs->zones[opponent_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  for (int i = 0; i < garden_cards.count; i++) {
    if (is_valid_freeze_target(world, garden_cards.ids[i])) {
      return true; // Found at least one valid target
    }
  }

  return false; // No valid targets in opponent's garden
}

bool stt02_014_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  // Target must be in opponent's garden zone
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  if (parent != gs->zones[opponent_num].garden) {
    return false;
  }

  // Target must have cost <= 2
  return is_valid_freeze_target(world, target);
}

void stt02_014_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect_targets[0];

  if (target == 0) {
    cli_render_logf("[STT02-014] No target to freeze");
    return;
  }

  // Apply Frozen status with duration 2
  apply_frozen(world, target, 2);
  cli_render_logf("[STT02-014] Froze entity for 2 turns");
}
