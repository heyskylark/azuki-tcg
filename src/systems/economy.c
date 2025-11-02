#include "systems/economy.h"
#include "queries/card_zone.h"
#include <stdio.h>

void DrawCard(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t deck_zone = state->zones[active_player_index].deck;
  ecs_entity_t hand_zone = state->zones[active_player_index].hand;
  int total_cards = 0;
  ecs_entity_t top_card = 0;

  if (!get_top_card_in_zone(world, deck_zone, &top_card, &total_cards)) {
    printf("No cards in deck\n");
    // TODO: Player loses
    return;
  }

  ecs_remove_pair(world, top_card, Rel_InZone, deck_zone);
  ecs_add_pair(world, top_card, Rel_InZone, hand_zone);

  printf("Drew card %s\n", ecs_get_name(world, top_card));
}

void GrantIKZ(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t ikz_pile_zone = state->zones[active_player_index].ikz_pile;
  ecs_entity_t ikz_area_zone = state->zones[active_player_index].ikz_area;
  ecs_entity_t card = 0;

  if (!get_top_card_in_zone(world, ikz_pile_zone, &card, NULL)) {
    return;
  }

  ecs_remove_pair(world, card, Rel_InZone, ikz_pile_zone);
  ecs_add_pair(world, card, Rel_InZone, ikz_area_zone);

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
