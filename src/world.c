#include "world.h"
#include "constants/game.h"
#include "systems/main.h"

#include <stdio.h>
#include <stdlib.h>

static const CardInfo shaoDeckCardInfo[18] = {
  { .card_id = CARD_DEF_STT02_001, .card_count = 1 },
  { .card_id = CARD_DEF_STT02_002, .card_count = 1 },
  { .card_id = CARD_DEF_STT02_003, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_004, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_005, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_006, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_007, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_008, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_009, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_010, .card_count = 2 },
  { .card_id = CARD_DEF_STT02_011, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_012, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_013, .card_count = 2 },
  { .card_id = CARD_DEF_STT02_014, .card_count = 2 },
  { .card_id = CARD_DEF_STT02_015, .card_count = 4 },
  { .card_id = CARD_DEF_STT02_016, .card_count = 2 },
  { .card_id = CARD_DEF_STT02_017, .card_count = 2 },
  { .card_id = CARD_DEF_IKZ_001, .card_count = 10 },
};

static const CardInfo raizenDeckCardInfo[18] = {
  { .card_id = CARD_DEF_STT01_001, .card_count = 1 },
  { .card_id = CARD_DEF_STT01_002, .card_count = 1 },
  { .card_id = CARD_DEF_STT01_003, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_004, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_005, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_006, .card_count = 2 },
  { .card_id = CARD_DEF_STT01_007, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_008, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_009, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_010, .card_count = 2 },
  { .card_id = CARD_DEF_STT01_011, .card_count = 2 },
  { .card_id = CARD_DEF_STT01_012, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_013, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_014, .card_count = 4 },
  { .card_id = CARD_DEF_STT01_015, .card_count = 2 },
  { .card_id = CARD_DEF_STT01_016, .card_count = 2 },
  { .card_id = CARD_DEF_STT01_017, .card_count = 4 },
  { .card_id = CARD_DEF_IKZ_001, .card_count = 10 },
};

static void register_card(
  ecs_world_t *world,
  ecs_entity_t player,
  CardDefId card_id,
  uint8_t count,
  PlayerZones *zones
);

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
  uint8_t player_number,
  WorldRef *ref
) {
  char zname[32];
  snprintf(zname, sizeof(zname), "Deck_P%d", player_number);
  ref->zones[player_number].deck = make_player_board_zone(world, player, zname, ZDeck);

  snprintf(zname, sizeof(zname), "Hand_P%d", player_number);
  ref->zones[player_number].hand = make_player_board_zone(world, player, zname, ZHand);

  snprintf(zname, sizeof(zname), "Leader_P%d", player_number);
  ref->zones[player_number].leader = make_player_board_zone(world, player, zname, ZLeader);
  
  snprintf(zname, sizeof(zname), "Gate_P%d", player_number);
  ref->zones[player_number].gate = make_player_board_zone(world, player, zname, ZGate);

  snprintf(zname, sizeof(zname), "Garden_P%d", player_number);
  ref->zones[player_number].garden = make_player_board_zone(world, player, zname, ZGarden);

  snprintf(zname, sizeof(zname), "Alley_P%d", player_number);
  ref->zones[player_number].alley = make_player_board_zone(world, player, zname, ZAlley);

  snprintf(zname, sizeof(zname), "IKZPile_P%d", player_number);
  ref->zones[player_number].ikz_pile = make_player_board_zone(world, player, zname, ZIKZPileTag);

  snprintf(zname, sizeof(zname), "IKZArea_P%d", player_number);
  ref->zones[player_number].ikz_area = make_player_board_zone(world, player, zname, ZIKZAreaTag);

  snprintf(zname, sizeof(zname), "Discard_P%d", player_number);
  ref->zones[player_number].discard = make_player_board_zone(world, player, zname, ZDiscard);
}

static DeckType random_deck_type(unsigned int *state) {
  return (DeckType)(rand_r(state) % 2);
}

ecs_world_t* azk_world_init(uint32_t seed) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  WorldRef ref = {0};
  unsigned int rng_state = seed;

  ecs_add_id(world, ecs_id(GameState), EcsSingleton);
  ecs_singleton_set(
    world,
    GameState,
    {
      .seed = seed,
      .active_player_number = 0,
      .phase = PHASE_PREGAME_MULLIGAN,
      .response_window = 0,
      .winner = -1
    });

  for (int p=0; p<MAX_PLAYERS_PER_MATCH; p++) {
    ecs_entity_t player = ecs_new(world);
    char pname[16]; snprintf(pname, sizeof(pname), "Player%d", p);
    ecs_set_name(world, player, pname);
    ecs_set(world, player, PlayerNumber, { (uint8_t)p });
    ecs_set(world, player, PlayerId, { (uint8_t)p });
    ref.players[p] = player;

    init_all_player_zones(world, player, p, &ref);

    DeckType deck_type = random_deck_type(&rng_state);
    init_player_deck(world, player, deck_type, &ref.zones[p]);
  }

  ecs_add_id(world, ecs_id(WorldRef), EcsSingleton);
  ecs_singleton_set_ptr(world, WorldRef, &ref);

  init_all_system(world);

  return world;
}

void azk_world_fini(ecs_world_t *world) {
  ecs_fini(world);
}

void init_player_deck(
  ecs_world_t *world,
  ecs_entity_t player,
  DeckType deck_type,
  PlayerZones *zones
) {
  switch (deck_type) {
  case RAIZEN:
    for (size_t index = 0; index < (sizeof(raizenDeckCardInfo) / sizeof(raizenDeckCardInfo[0])); index++) {
      register_card(
        world,
        player,
        raizenDeckCardInfo[index].card_id,
        (uint8_t)raizenDeckCardInfo[index].card_count,
        zones
      );
    }
    break;
  case SHAO:
    for (size_t index = 0; index < (sizeof(shaoDeckCardInfo) / sizeof(shaoDeckCardInfo[0])); index++) {
      register_card(
        world,
        player,
        shaoDeckCardInfo[index].card_id,
        (uint8_t)shaoDeckCardInfo[index].card_count,
        zones
      );
    }
    break;
  }
}

static void apply_card_type_tag(ecs_world_t *world, ecs_entity_t entity, CardType type) {
  switch (type) {
  case CARD_TYPE_LEADER:
    ecs_add(world, entity, TLeader);
    break;
  case CARD_TYPE_GATE:
    ecs_add(world, entity, TGate);
    break;
  case CARD_TYPE_ENTITY:
    ecs_add(world, entity, TEntity);
    break;
  case CARD_TYPE_WEAPON:
    ecs_add(world, entity, TWeapon);
    break;
  case CARD_TYPE_SPELL:
    ecs_add(world, entity, TSpell);
    break;
  case CARD_TYPE_IKZ:
    ecs_add(world, entity, TIKZ);
    break;
  default:
    fprintf(stderr, "Error: Unknown CardType %d\n", type);
    exit(EXIT_FAILURE);
    break;
  }
}

// TODO: I think i need to use a zone entity created above per player, it'll affect querying alter on.
// Need to use the zones created in line 46.
static void apply_card_zone_relationship(
  ecs_world_t *world,
  ecs_entity_t card,
  CardType type,
  PlayerZones *zones
) {
  switch (type) {
  case CARD_TYPE_LEADER:
    ecs_add_pair(world, card, Rel_InZone, zones->leader);
    break;
  case CARD_TYPE_GATE:
    ecs_add_pair(world, card, Rel_InZone, zones->gate);
    break;
  case CARD_TYPE_ENTITY:
    ecs_add_pair(world, card, Rel_InZone, zones->deck);
    break;
  case CARD_TYPE_WEAPON:
    ecs_add_pair(world, card, Rel_InZone, zones->deck);
    break;
  case CARD_TYPE_SPELL:
    ecs_add_pair(world, card, Rel_InZone, zones->deck);
    break;
  case CARD_TYPE_IKZ:
    ecs_add_pair(world, card, Rel_InZone, zones->ikz_pile);
    break;
  default:
    fprintf(stderr, "Error: Unknown CardType %d\n", type);
    exit(EXIT_FAILURE);
    break;
  }
}

static void register_card(
  ecs_world_t *world,
  ecs_entity_t player,
  CardDefId card_id,
  uint8_t count,
  PlayerZones *zones
) {
  const CardDef *def = azk_card_def_from_id(card_id);
  if (!def) {
    fprintf(stderr, "Error: Failed to look up CardDef for id %d\n", card_id);
    exit(EXIT_FAILURE);
  }

  for (size_t index = 0; index < count; index++) {
    // Give each card instance a unique name so Flecs doesn't reuse existing entities
    char entity_name[64];
    snprintf(entity_name, sizeof(entity_name), "%s_%zu", def->card_id, index + 1);
    ecs_entity_t card = ecs_entity_init(world, &(ecs_entity_desc_t){
      .name = entity_name,
      .sep = "",
    });

    apply_card_type_tag(world, card, def->type);

    Element element_component = { .element = (uint8_t)def->element };
    ecs_set_ptr(world, card, Element, &element_component);

    if (def->has_base_stats) {
      ecs_set_ptr(world, card, BaseStats, &def->base_stats);
    }

    if (def->has_gate_points) {
      ecs_set_ptr(world, card, GatePoints, &def->gate_points);
    }

    if (def->has_ikz_cost) {
      ecs_set_ptr(world, card, IKZCost, &def->ikz_cost);
    }

    apply_card_zone_relationship(world, card, def->type, zones);

    ecs_add_pair(world, card, Rel_OwnedBy, player);
  }
}
