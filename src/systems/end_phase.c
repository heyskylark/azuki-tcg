#include "systems/end_phase.h"
#include "components.h"
#include "utils/cli_rendering_util.h"
#include "utils/entity_util.h"

void HandleEndPhase(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  ecs_entity_t garden_zone = gs->zones[gs->active_player_index].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden_zone);

  for (int32_t i = 0; i < garden_cards.count; i++) {
    cli_render_logf("[EndPhase] Resetting entity health for garden card %d", i);
    ecs_entity_t garden_card = garden_cards.ids[i];

    reset_entity_health(world, garden_card);
    // Expire end of turn effects on player's entities and leader
    discard_equipped_weapon_cards(world, garden_card);
  }

  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;

  cli_render_log("[EndPhase] End phase");
}

void init_end_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "EndPhaseSystem",
      .add = ecs_ids(TEndTurn)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) },
    },
    .callback = HandleEndPhase
  });
}
