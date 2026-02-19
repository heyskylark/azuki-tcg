#include "systems/mulligan_phase.h"
#include "components/components.h"
#include "utils/deck_utils.h"
#include "utils/actions_util.h"
#include "constants/game.h"
#include "utils/cli_rendering_util.h"

static bool HandleMulliganShuffleAction(ecs_world_t *world, GameState *gs) {
  ecs_entity_t deck_zone = gs->zones[gs->active_player_index].deck;
  ecs_entity_t hand_zone = gs->zones[gs->active_player_index].hand;

  bool moved_to_deck =
      move_cards_to_zone(world, hand_zone, deck_zone, INITIAL_DRAW_COUNT, NULL);
  ecs_assert(moved_to_deck, ECS_INVALID_OPERATION,
             "[Mulligan] Failed to move cards from hand to deck");
  if (!moved_to_deck) {
    cli_render_log("[Mulligan] Failed to move cards from hand to deck");
    return false;
  }

  if (!move_cards_to_zone(world, deck_zone, hand_zone, INITIAL_DRAW_COUNT, NULL)) {
    cli_render_log("[Mulligan] No cards in deck");

    gs->winner = (gs->active_player_index + 1) % 2;

    return true;
  }

  shuffle_deck(world, deck_zone);
  cli_render_log("[Mulligan] Shuffled deck");
  return true;
}

static void handle_phase_transition(ecs_world_t *world, GameState *gs) {
  if (is_game_over(world)) {
    gs->phase = PHASE_END_MATCH;
  } else {
    gs->mulligan_actions_completed++;
    if (gs->mulligan_actions_completed >= MAX_PLAYERS_PER_MATCH) {
      gs->active_player_index = gs->starting_player_index;
      gs->phase = PHASE_START_OF_TURN;
      gs->mulligan_actions_completed = 0;
      return;
    }

    gs->active_player_index =
        (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;
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
      if (!HandleMulliganShuffleAction(world, gs)) {
        ac->invalid_action = true;
        return;
      }
      break;
    case ACT_NOOP:
      cli_render_log("[Mulligan] No mulligan performed, end turn");
      break;
    default:
      cli_render_logf("[Mulligan] Unknown mulligan action type: %d", action.type);
      ac->invalid_action = true;
      break;
  }

  handle_phase_transition(world, gs);
}

void init_mulligan_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "MulliganPhaseSystem",
      .add = ecs_ids(TMulligan)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) },
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) }
    },
    .callback = HandleMulliganAction
  });
}
