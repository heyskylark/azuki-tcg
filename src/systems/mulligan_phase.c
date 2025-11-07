#include "systems/mulligan_phase.h"
#include "components.h"
#include "utils/deck_utils.h"
#include "utils/actions_util.h"
#include "constants/game.h"
#include "utils/cli_rendering_util.h"

static void HandleMulliganShuffleAction(ecs_world_t *world, GameState *gs) {
  ecs_entity_t deck_zone = gs->zones[gs->active_player_index].deck;
  ecs_entity_t hand_zone = gs->zones[gs->active_player_index].hand;

  ecs_assert(
    move_cards_to_zone(world, hand_zone, deck_zone, INITIAL_DRAW_COUNT, NULL),
    ECS_INVALID_OPERATION,
    "[Mulligan] Failed to move cards from hand to deck"
  );


  if (!move_cards_to_zone(world, deck_zone, hand_zone, INITIAL_DRAW_COUNT, NULL)) {
    cli_render_log("[Mulligan] No cards in deck");
    // TODO: Player loses
    return;
  }

  shuffle_deck(world, deck_zone);
  cli_render_log("[Mulligan] Shuffled deck");
}

static void handle_phase_transition(GameState *gs) {
  if (gs->active_player_index == 0) {
    gs->active_player_index = 1;
  } else {
    gs->active_player_index = 0;
    gs->phase = PHASE_START_OF_TURN;
  }
}

void HandleMulliganAction(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);
  UserAction action = ac->user_action;

  if (!verify_user_action_player(gs, &action)) {
    cli_render_logf("[Mulligan] Action player mismatch: %d != %d", gs->players[gs->active_player_index], action.player);
    ac->invalid_action = true;
    return;
  }

  switch (action.type) {
    case ACT_MULLIGAN_SHUFFLE:
      HandleMulliganShuffleAction(world, gs);
      break;
    case ACT_NOOP:
    case ACT_MULLIGAN_KEEP:
      cli_render_log("[Mulligan] No mulligan performed, end turn");
      break;
    default:
      cli_render_logf("[Mulligan] Unknown mulligan action type: %d", action.type);
      ac->invalid_action = true;
      break;
  }

  handle_phase_transition(gs);
}

void init_mulligan_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "HandleMulliganAction",
      .add = ecs_ids(TMulligan)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) },
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) }
    },
    .callback = HandleMulliganAction
  });
}
