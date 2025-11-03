#include "systems/mulligan.h"
#include "components.h"
#include <stdio.h>
#include "utils/deck_utils.h"
#include "utils/actions_util.h"
#include "constants/game.h"

static void HandleMulliganShuffleAction(ecs_world_t *world, GameState *gs) {
  ecs_entity_t deck_zone = gs->zones[gs->active_player_index].deck;
  ecs_entity_t hand_zone = gs->zones[gs->active_player_index].hand;

  ecs_assert(
    draw_cards(world, hand_zone, deck_zone, INITIAL_DRAW_COUNT, NULL),
    ECS_INVALID_OPERATION,
    "[Mulligan] Failed to move cards from hand to deck"
  );


  if (!draw_cards(world, deck_zone, hand_zone, INITIAL_DRAW_COUNT, NULL)) {
    printf("[Mulligan] No cards in deck\n");
    // TODO: Player loses
    return;
  }

  shuffle_deck(world, deck_zone);
  printf("[Mulligan] Shuffled deck\n");
}

void HandleMulliganAction(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);
  UserAction action = ac->user_action;

  if (!verify_user_action_player(gs, &action)) {
    fprintf(stderr, "Mulligan action player mismatch: %d != %d\n", gs->players[gs->active_player_index], action.player);
    return;
  }

  switch (action.type) {
    case ACT_MULLIGAN_SHUFFLE:
      HandleMulliganShuffleAction(world, gs);
      break;
    case ACT_NOOP:
    case ACT_MULLIGAN_KEEP:
      printf("[Mulligan] No action performed\n");
      break;
    default:
      fprintf(stderr, "[Mulligan] Unknown mulligan action type: %d\n", action.type);
      break;
  }
}

void init_mulligan_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "HandleMulliganAction",
      .add = ecs_ids(TMulligan)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState), .inout = EcsIn },
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) }
    },
    .callback = HandleMulliganAction
  });
}