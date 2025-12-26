#include "systems/combat_resolve_phase.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/combat_util.h"
#include "utils/observation_util.h"
#include "utils/zone_util.h"

// Check if an entity card is still in any garden zone
static bool is_entity_in_garden(ecs_world_t *world, ecs_entity_t card, const GameState *gs) {
  if (card == 0) return false;

  // Check if card is a leader (leaders are always "in play")
  if (ecs_has(world, card, TLeader)) {
    return true;
  }

  // Check if card is in either player's garden
  for (int i = 0; i < MAX_PLAYERS_PER_MATCH; i++) {
    ecs_entity_t garden = gs->zones[i].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
    for (int j = 0; j < garden_cards.count; j++) {
      if (garden_cards.ids[j] == card) {
        return true;
      }
    }
  }

  return false;
}

void HandleCombatResolution(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  // Check for combat fizzle - if attacker or defender was removed during response window
  bool attacker_valid = is_entity_in_garden(world, gs->combat_state.attacking_card, gs);
  bool defender_valid = is_entity_in_garden(world, gs->combat_state.defender_card, gs);

  if (!attacker_valid || !defender_valid) {
    // Combat fizzles - one or both participants were removed
    cli_render_logf("[CombatResolution] Combat fizzled - %s removed",
      !attacker_valid ? "attacker" : "defender");
    gs->combat_state.attacking_card = 0;
    gs->combat_state.defender_card = 0;
    gs->phase = PHASE_MAIN;
    return;
  }

  resolve_combat(world);

  if (is_game_over(world)) {
    gs->phase = PHASE_END_MATCH;
  } else {
    gs->phase = PHASE_MAIN;
  }

  cli_render_logf("[CombatResolution] Combat resolution");
}

void init_combat_resolve_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "CombatResolvePhaseSystem",
      .add = ecs_ids(TCombatResolve)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) }
    },
    .callback = HandleCombatResolution
  });
}