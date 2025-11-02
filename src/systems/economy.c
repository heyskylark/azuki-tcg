#include "systems/economy.h"
#include "queries/card_zone.h"
#include <stdio.h>

void DrawCard(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t player = state->players[active_player_index];
  ecs_entity_t deck_zone = state->zones[active_player_index].deck;
  ecs_entity_t hand_zone = state->zones[active_player_index].hand;
  ecs_iter_t cards_in_deck =
    get_cards_owned_by_player_in_zone(world, deck_zone);

  if (!ecs_query_next(&cards_in_deck) || cards_in_deck.count == 0) {
    printf("No cards in deck\n");
    ecs_iter_fini(&cards_in_deck);
    // TODO: Player loses
    return;
  }

  printf("Cards in deck: %d\n", cards_in_deck.count);

  ecs_entity_t card = cards_in_deck.entities[0];
  ecs_remove_pair(world, card, Rel_InZone, deck_zone);
  ecs_add_pair(world, card, Rel_InZone, hand_zone);

  ecs_iter_fini(&cards_in_deck);

  printf("Drew card %s\n", ecs_get_name(world, card));
}

void GrantIKZ(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  const GameState *state = ecs_field(it, GameState, 0);
  int8_t active_player_index = state->active_player_index;
  ecs_entity_t player = state->players[active_player_index];
  ecs_entity_t ikz_pile_zone = state->zones[active_player_index].ikz_pile;
  ecs_entity_t ikz_area_zone = state->zones[active_player_index].ikz_area;
  ecs_iter_t cards_in_ikz_pile =
    get_cards_owned_by_player_in_zone(world, ikz_pile_zone);

  if (!ecs_query_next(&cards_in_ikz_pile) || cards_in_ikz_pile.count == 0) {
    ecs_iter_fini(&cards_in_ikz_pile);
    return;
  }

  ecs_entity_t card = cards_in_ikz_pile.entities[0];
  ecs_remove_pair(world, card, Rel_InZone, ikz_pile_zone);
  ecs_add_pair(world, card, Rel_InZone, ikz_area_zone);

  ecs_iter_fini(&cards_in_ikz_pile);

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
