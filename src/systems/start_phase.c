#include "systems/start_phase.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/zone_util.h"

void DrawCard(ecs_world_t *world, GameState *gs) {
  ecs_entity_t deck_zone = gs->zones[gs->active_player_index].deck;
  ecs_entity_t hand_zone = gs->zones[gs->active_player_index].hand;

  ecs_entity_t out_cards[1] = {0};
  if (!move_cards_to_zone(world, deck_zone, hand_zone, 1, out_cards)) {
    cli_render_log("[DrawCard] No cards in deck");

    gs->winner = (gs->active_player_index + 1) % 2;

    return;
  }

  cli_render_logf("[DrawCard] Drew card %s", ecs_get_name(world, out_cards[0]));
}

void GrantIKZ(ecs_world_t *world, GameState *gs) {
  ecs_entity_t ikz_pile_zone = gs->zones[gs->active_player_index].ikz_pile;
  ecs_entity_t ikz_area_zone = gs->zones[gs->active_player_index].ikz_area;

  if (!move_cards_to_zone(world, ikz_pile_zone, ikz_area_zone, 1, NULL)) {
    return;
  }

  cli_render_log("[GrantIKZ] IKZ granted");
}

static void UntapAllCards(ecs_world_t *world, GameState *gs) {
  ecs_entity_t garden_zone = gs->zones[gs->active_player_index].garden;
  untap_all_cards_in_zone(world, garden_zone);

  ecs_entity_t ikz_area_zone = gs->zones[gs->active_player_index].ikz_area;
  untap_all_cards_in_zone(world, ikz_area_zone);

  ecs_entity_t leader_zone = gs->zones[gs->active_player_index].leader;
  untap_all_cards_in_zone(world, leader_zone);

  ecs_entity_t gate_zone = gs->zones[gs->active_player_index].gate;
  untap_all_cards_in_zone(world, gate_zone);

  cli_render_log("[UntapAllCards] Untapped all cards in zones");
}

static void handle_phase_transition(ecs_world_t *world, GameState *gs) {
  if (is_game_over(world)) {
    gs->phase = PHASE_END_MATCH;
  } else {
    gs->phase = PHASE_MAIN;
  }
}

void StartPhase(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  UntapAllCards(world, gs);
  DrawCard(world, gs);
  GrantIKZ(world, gs);

  handle_phase_transition(world, gs);

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
