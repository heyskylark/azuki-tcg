#include "systems/economy.h"
#include "utils/deck_utils.h"
#include <stdio.h>

void DrawCard(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t deck_zone = state->zones[active_player_index].deck;
  ecs_entity_t hand_zone = state->zones[active_player_index].hand;

  ecs_entity_t out_cards[1] = {0};
  if (!draw_cards(world, deck_zone, hand_zone, 1, out_cards)) {
    printf("No cards in deck\n");
    // TODO: Player loses
    return;
  }

  printf("Drew card %s\n", ecs_get_name(world, out_cards[0]));
}

void GrantIKZ(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t ikz_pile_zone = state->zones[active_player_index].ikz_pile;
  ecs_entity_t ikz_area_zone = state->zones[active_player_index].ikz_area;

  if (!draw_cards(world, ikz_pile_zone, ikz_area_zone, 1, NULL)) {
    return;
  }

  printf("IKZ granted\n");
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

  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "GrantIKZSystem",
      .add = ecs_ids(TStartOfTurn)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState), .inout = EcsIn }
    },
    .callback = GrantIKZ
  });
}
