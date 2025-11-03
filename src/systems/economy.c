#include "systems/economy.h"
#include <stdio.h>

static bool get_top_card_in_zone(
  ecs_world_t *world,
  ecs_entity_t zone,
  ecs_entity_t *out_card,
  int *out_count
) {
  ecs_entities_t children = ecs_get_ordered_children(world, zone);
  int32_t count = children.count;

  printf("Zone %s has %d cards\n", ecs_get_name(world, zone), count);

  if (out_count) {
    *out_count = (int)count;
  }
  if (!count) {
    if (out_card) {
      *out_card = 0;
    }
    return false;
  }

  if (out_card) {
    *out_card = children.ids[count - 1];
  }
  return true;
}

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

  ecs_add_pair(world, top_card, EcsChildOf, hand_zone);

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

  ecs_add_pair(world, card, EcsChildOf, ikz_area_zone);

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
