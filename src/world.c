#include "world.h"
#include "constants/game.h"

static ecs_entity_t make_player_board_zone(
  ecs_world_t *world,
  ecs_entity_t player,
  const char *name,
  ecs_entity_t zone_tag
) {
  ecs_entity_t z = ecs_new(world);
  ecs_set_name(world, z, name);
  ecs_add_id(world, z, zone_tag);
  ecs_add_pair(world, z, Rel_OwnedBy, player);
  return z;
}

static void init_all_player_zones(
  ecs_world_t *world,
  ecs_entity_t player,
  uint8_t player_number
) {
  char zname[32];
  snprintf(zname, sizeof(zname), "Deck_P%d", p);
  make_player_board_zone(world, player, zname, ZDeck);

  snprintf(zname, sizeof(zname), "Hand_P%d", p);
  make_player_board_zone(world, player, zname, ZHand);

  snprintf(zname, sizeof(zname), "Leader_P%d", p);
  make_player_board_zone(world, player, zname, ZLeader);
  
  snprintf(zname, sizeof(zname), "Gate_P%d", p);
  make_player_board_zone(world, player, zname, ZGate);

  snprintf(zname, sizeof(zname), "Garden_P%d", p);
  make_player_board_zone(world, player, zname, ZGarden);

  snprintf(zname, sizeof(zname), "Alley_P%d", p);
  make_player_board_zone(world, player, zname, ZAlley);

  snprinf(zname, sizeof(zname), "IKZPile_P%d", p);
  make_player_board_zone(world, player, zname, ZIKZPileTag);

  snprinf(zname, sizeof(zname), "IKZArea_P%d", p);
  make_player_board_zone(world, player, zname, ZIKZAreaTag);

  snprinf(zname, sizeof(zname), "Discard_P%d", p);
  make_player_board_zone(world, player, zname, ZDiscard);
}

ecs_world_t* azk_world_init(uint32_t seed) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_add_id(world, ecs_id(GameState), EcsSingleton);
  ecs_singleton_set(
    world,
    GameState,
    {
      .seed = seed,
      .active = 0,
      .phase = PHASE_PREGAME_MULLIGAN,
      .response_window = 0,
      .winner = -1
    }
  )

  for (int p=0; p<MAX_PLAYERS_PER_MATCH; p++) {
    ecs_entity_t player = ecs_new(world);
    char pname[16]; snprintf(pname, sizeof(pname), "Player%d", p);
    ecs_set_name(world, player, pname);
    ecs_set(world, player, PlayerId, { (uint8_t)p });

    init_all_player_zones(world, player, p);

    // TODO: Init all deck, leader, gate, and IKZ pile cards here?
  }

  // TODO: Init all systems and pipelines here?
}

void azk_world_fini(ecs_world_t *world) {
  ecs_fini(world);
}
