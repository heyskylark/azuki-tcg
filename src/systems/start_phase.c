#include "systems/start_phase.h"
#include "components.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/zone_util.h"

void DrawCard(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t deck_zone = state->zones[active_player_index].deck;
  ecs_entity_t hand_zone = state->zones[active_player_index].hand;

  ecs_entity_t out_cards[1] = {0};
  if (!move_cards_to_zone(world, deck_zone, hand_zone, 1, out_cards)) {
    cli_render_log("[DrawCard] No cards in deck");
    // TODO: Player loses
    return;
  }

  cli_render_logf("[DrawCard] Drew card %s", ecs_get_name(world, out_cards[0]));
}

void GrantIKZ(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t ikz_pile_zone = state->zones[active_player_index].ikz_pile;
  ecs_entity_t ikz_area_zone = state->zones[active_player_index].ikz_area;

  if (!move_cards_to_zone(world, ikz_pile_zone, ikz_area_zone, 1, NULL)) {
    return;
  }

  cli_render_log("[GrantIKZ] IKZ granted");
}

static void UntapAllCards(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;

  ecs_entity_t garden_zone = state->zones[active_player_index].garden;
  untap_all_cards_in_zone(world, garden_zone);

  ecs_entity_t ikz_area_zone = state->zones[active_player_index].ikz_area;
  untap_all_cards_in_zone(world, ikz_area_zone);

  ecs_entity_t leader_zone = state->zones[active_player_index].leader;
  untap_all_cards_in_zone(world, leader_zone);

  ecs_entity_t gate_zone = state->zones[active_player_index].gate;
  untap_all_cards_in_zone(world, gate_zone);

  cli_render_log("[UntapAllCards] Untapped all cards in zones");
}

static void handle_phase_transition(GameState *gs) {
  gs->phase = PHASE_MAIN;
}

void StartPhase(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  UntapAllCards(it);
  DrawCard(it);
  GrantIKZ(it);

  gs->phase = PHASE_MAIN;

  cli_render_log("[StartPhase] Start phase");
}

void init_start_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "StartPhaseSystem",
      .add = ecs_ids(TStartOfTurn)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) },
    },
    .callback = StartPhase
  });
}
