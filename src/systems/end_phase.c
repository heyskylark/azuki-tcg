#include "systems/end_phase.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/entity_util.h"
#include "utils/zone_util.h"

void HandleEndPhase(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  ecs_entity_t active_player_garden_zone = gs->zones[gs->active_player_index].garden;
  ecs_entities_t active_player_garden_cards = ecs_get_ordered_children(world, active_player_garden_zone);

  cli_render_logf("[EndPhase] Resetting entity health for active player's garden cards");
  for (int32_t i = 0; i < active_player_garden_cards.count; i++) {
    ecs_entity_t garden_card = active_player_garden_cards.ids[i];

    reset_entity_health(world, garden_card);
    // TODO: Expire end of turn effects on player's entities
    discard_equipped_weapon_cards(world, garden_card);
  }

  ecs_entity_t defending_player_garden_zone = gs->zones[(gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH].garden;
  ecs_entities_t defending_player_garden_cards = ecs_get_ordered_children(world, defending_player_garden_zone);

  cli_render_logf("[EndPhase] Resetting entity health for defending player's garden cards");
  for (int32_t i = 0; i < defending_player_garden_cards.count; i++) {
    ecs_entity_t garden_card = defending_player_garden_cards.ids[i];

    reset_entity_health(world, garden_card);
  }

  ecs_entity_t leader_card = find_leader_card_in_zone(world, gs->zones[gs->active_player_index].leader);
  // TODO: Expire end of turn effects on player's leader
  discard_equipped_weapon_cards(world, leader_card);

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
