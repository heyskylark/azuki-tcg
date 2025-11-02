#include "systems/economy.h"
#include "queries/card_zone.h"
#include <stdio.h>

void DrawCard(ecs_iter_t *it) {
  const GameState *state = ecs_field(it, GameState, 0);
  ecs_entity_t player = state->players[state->active_player_index];
  ecs_entity_t deck_zone = state->zones[state->active_player_index].deck;
  ecs_entity_t hand_zone = state->zones[state->active_player_index].hand;
  ecs_iter_t it = get_cards_owned_by_player_in_zone(world, player, deck_zone);

  ecs_query_next(&it);
  if (it.count == 0) {
    // TODO: Player loses
    return;
  }

  ecs_entity_t card = it.entities[0];
  ecs_add_pair(world, card, Rel_InZone, hand_zone);
}

void GrantIKZ(ecs_iter_t *it) {
  (void)it;
  printf("Granting IKZ\n");
}

void init_economy_systems(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "DrawCardSystem",
      .add = ecs_ids(TStartOfTurn)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState), .inout = EcsIn }
    },
    .callback = DrawCard
  });
}
