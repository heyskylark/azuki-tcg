#include "systems/card_state.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"

void UntapAllCards(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;

  ecs_entity_t alley_zone = state->zones[active_player_index].alley;
  untap_all_cards_in_zone(world, alley_zone);

  ecs_entity_t ikz_area_zone = state->zones[active_player_index].ikz_area;
  untap_all_cards_in_zone(world, ikz_area_zone);

  ecs_entity_t leader_zone = state->zones[active_player_index].leader;
  untap_all_cards_in_zone(world, leader_zone);

  ecs_entity_t gate_zone = state->zones[active_player_index].gate;
  untap_all_cards_in_zone(world, gate_zone);

  cli_render_log("[UntapAllCards] Untapped all cards in zones");
}

void init_card_state_systems(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "UntapAllCardsSystem",
      .add = ecs_ids(TStartOfTurn)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState), .inout = EcsIn },
    },
    .callback = UntapAllCards
  });
}
