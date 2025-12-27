#include "world.h"
#include "components/abilities.h"
#include "constants/game.h"
#include "queries/main.h"
#include "systems/main.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"

#include <stdio.h>
#include <stdlib.h>

static const CardInfo shaoDeckCardInfo[18] = {
    {.card_id = CARD_DEF_STT02_001, .card_count = 1},
    {.card_id = CARD_DEF_STT02_002, .card_count = 1},
    {.card_id = CARD_DEF_STT02_003, .card_count = 4},
    {.card_id = CARD_DEF_STT02_004, .card_count = 4},
    {.card_id = CARD_DEF_STT02_005, .card_count = 4},
    {.card_id = CARD_DEF_STT02_006, .card_count = 4},
    {.card_id = CARD_DEF_STT02_007, .card_count = 4},
    {.card_id = CARD_DEF_STT02_008, .card_count = 4},
    {.card_id = CARD_DEF_STT02_009, .card_count = 4},
    {.card_id = CARD_DEF_STT02_010, .card_count = 2},
    {.card_id = CARD_DEF_STT02_011, .card_count = 4},
    {.card_id = CARD_DEF_STT02_012, .card_count = 4},
    {.card_id = CARD_DEF_STT02_013, .card_count = 2},
    {.card_id = CARD_DEF_STT02_014, .card_count = 2},
    {.card_id = CARD_DEF_STT02_015, .card_count = 4},
    {.card_id = CARD_DEF_STT02_016, .card_count = 2},
    {.card_id = CARD_DEF_STT02_017, .card_count = 2},
    {.card_id = CARD_DEF_IKZ_001, .card_count = 10},
};

static const CardInfo raizenDeckCardInfo[18] = {
    {.card_id = CARD_DEF_STT01_001, .card_count = 1},
    {.card_id = CARD_DEF_STT01_002, .card_count = 1},
    {.card_id = CARD_DEF_STT01_003, .card_count = 4},
    {.card_id = CARD_DEF_STT01_004, .card_count = 4},
    {.card_id = CARD_DEF_STT01_005, .card_count = 4},
    {.card_id = CARD_DEF_STT01_006, .card_count = 2},
    {.card_id = CARD_DEF_STT01_007, .card_count = 4},
    {.card_id = CARD_DEF_STT01_008, .card_count = 4},
    {.card_id = CARD_DEF_STT01_009, .card_count = 4},
    {.card_id = CARD_DEF_STT01_010, .card_count = 2},
    {.card_id = CARD_DEF_STT01_011, .card_count = 2},
    {.card_id = CARD_DEF_STT01_012, .card_count = 4},
    {.card_id = CARD_DEF_STT01_013, .card_count = 4},
    {.card_id = CARD_DEF_STT01_014, .card_count = 4},
    {.card_id = CARD_DEF_STT01_015, .card_count = 2},
    {.card_id = CARD_DEF_STT01_016, .card_count = 2},
    {.card_id = CARD_DEF_STT01_017, .card_count = 4},
    {.card_id = CARD_DEF_IKZ_001, .card_count = 10},
};

typedef struct {
  ecs_entity_t players[MAX_PLAYERS_PER_MATCH];
  PlayerZones zones[MAX_PLAYERS_PER_MATCH];
} WorldRef;

typedef struct {
  ecs_entity_t zone;
  uint16_t *size;
} ZonePlacement;

typedef struct {
  uint16_t deck_size;
  uint16_t leader_size;
  uint16_t gate_size;
  uint16_t ikz_pile_size;
} TotalZoneCounts;

static ZonePlacement zone_placement_for_type(PlayerZones *zones,
                                             CardType type) {
  switch (type) {
  case CARD_TYPE_LEADER:
    return (ZonePlacement){zones->leader, &zones->leader_size};
  case CARD_TYPE_GATE:
    return (ZonePlacement){zones->gate, &zones->gate_size};
  case CARD_TYPE_ENTITY:
    return (ZonePlacement){zones->deck, &zones->deck_size};
  case CARD_TYPE_WEAPON:
    return (ZonePlacement){zones->deck, &zones->deck_size};
  case CARD_TYPE_SPELL:
    return (ZonePlacement){zones->deck, &zones->deck_size};
  case CARD_TYPE_IKZ:
    return (ZonePlacement){zones->ikz_pile, &zones->ikz_pile_size};
  default:
    cli_render_logf("Error: Unknown CardType %d", type);
    exit(EXIT_FAILURE);
  }
}

static ecs_entity_t make_player_board_zone(ecs_world_t *world,
                                           ecs_entity_t player,
                                           const char *name,
                                           ecs_entity_t zone_tag) {
  ecs_entity_t z = ecs_new(world);
  ecs_set_name(world, z, name);
  ecs_add_id(world, z, zone_tag);
  ecs_add_id(world, z, EcsOrderedChildren);
  ecs_add_pair(world, z, Rel_OwnedBy, player);
  return z;
}

static void init_all_player_zones(ecs_world_t *world, ecs_entity_t player,
                                  uint8_t player_number, WorldRef *ref) {
  char zname[32];
  snprintf(zname, sizeof(zname), "Deck_P%d", player_number);
  ref->zones[player_number].deck =
      make_player_board_zone(world, player, zname, ZDeck);

  snprintf(zname, sizeof(zname), "Hand_P%d", player_number);
  ref->zones[player_number].hand =
      make_player_board_zone(world, player, zname, ZHand);

  snprintf(zname, sizeof(zname), "Leader_P%d", player_number);
  ref->zones[player_number].leader =
      make_player_board_zone(world, player, zname, ZLeader);

  snprintf(zname, sizeof(zname), "Gate_P%d", player_number);
  ref->zones[player_number].gate =
      make_player_board_zone(world, player, zname, ZGate);

  snprintf(zname, sizeof(zname), "Garden_P%d", player_number);
  ref->zones[player_number].garden =
      make_player_board_zone(world, player, zname, ZGarden);

  snprintf(zname, sizeof(zname), "Alley_P%d", player_number);
  ref->zones[player_number].alley =
      make_player_board_zone(world, player, zname, ZAlley);

  snprintf(zname, sizeof(zname), "IKZPile_P%d", player_number);
  ref->zones[player_number].ikz_pile =
      make_player_board_zone(world, player, zname, ZIKZPileTag);

  snprintf(zname, sizeof(zname), "IKZArea_P%d", player_number);
  ref->zones[player_number].ikz_area =
      make_player_board_zone(world, player, zname, ZIKZAreaTag);

  snprintf(zname, sizeof(zname), "Discard_P%d", player_number);
  ref->zones[player_number].discard =
      make_player_board_zone(world, player, zname, ZDiscard);

  snprintf(zname, sizeof(zname), "Selection_P%d", player_number);
  ref->zones[player_number].selection =
      make_player_board_zone(world, player, zname, ZSelection);
}

static DeckType random_deck_type(unsigned int *state) {
  unsigned int roll = rand_r(state);
  DeckType type = (DeckType)(roll % 2);

  // Rotate the state so subsequent calls start from a different seed
  // bit-pattern.
  const unsigned int bits = (unsigned int)(sizeof(*state) * 8u);
  const unsigned int shift = 1u;
  *state = (*state << shift) | (*state >> (bits - shift));

  return type;
}

static void register_action_context_singleton(ecs_world_t *world) {
  ecs_add_id(world, ecs_id(ActionContext), EcsSingleton);
  ActionContext ac = {0};
  ecs_singleton_set_ptr(world, ActionContext, &ac);
}

static void grant_player_ikz_token(ecs_world_t *world, ecs_entity_t player) {
  const ecs_entity_t ikz_token_prefab = azk_prefab_from_id(CARD_DEF_IKZ_002);
  ecs_assert(ikz_token_prefab != 0, ECS_INVALID_PARAMETER,
             "Prefab not found for IKZ token card");
  const ecs_entity_t ikz_token =
      ecs_new_w_pair(world, EcsIsA, ikz_token_prefab);
  ecs_set_name(world, ikz_token, "IKZTokenCard");
  ecs_set(world, player, IKZToken, {.ikz_token = ikz_token});
}

ecs_world_t *azk_world_init(uint32_t seed) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  WorldRef ref = {0};
  unsigned int rng_state = seed;

  ecs_add_id(world, ecs_id(GameState), EcsSingleton);
  GameState gs = {
      .seed = seed,
      .rng_state = seed,
      .active_player_index = 0,
      .phase = PHASE_PREGAME_MULLIGAN,
      .response_window = 0,
      .winner = -1,
  };

  ecs_add_id(world, ecs_id(AbilityContext), EcsSingleton);
  AbilityContext ac = {0};
  ecs_singleton_set_ptr(world, AbilityContext, &ac);

  ecs_entity_t players[MAX_PLAYERS_PER_MATCH];
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
    ecs_entity_t player = ecs_new(world);
    char pname[16];
    snprintf(pname, sizeof(pname), "Player%d", p);
    ecs_set_name(world, player, pname);
    ecs_set(world, player, PlayerNumber, {(uint8_t)p});
    ecs_set(world, player, PlayerId, {(uint8_t)p});
    ref.players[p] = player;

    init_all_player_zones(world, player, p, &ref);

    // Player going second always gets the IKZ token
    if (p == 1) {
      grant_player_ikz_token(world, player);
    }

    DeckType deck_type = random_deck_type(&rng_state);
    init_player_deck(world, player, deck_type, &ref.zones[p]);

    gs.players[p] = player;
    gs.zones[p] = ref.zones[p];
  }

  ecs_singleton_set_ptr(world, GameState, &gs);
  register_action_context_singleton(world);

  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
    shuffle_deck(world, ref.zones[p].deck);
    move_cards_to_zone(world, ref.zones[p].deck, ref.zones[p].hand,
                       INITIAL_DRAW_COUNT, NULL);
  }

  init_all_queries(world);
  init_all_system(world);

  return world;
}

void azk_world_fini(ecs_world_t *world) { ecs_fini(world); }

static void register_card(ecs_world_t *world, ecs_entity_t player,
                          CardDefId card_id, uint8_t count, PlayerZones *zones,
                          TotalZoneCounts *total_counts) {
  ecs_entity_t prefab = azk_prefab_from_id(card_id);
  ecs_assert(prefab != 0, ECS_INVALID_PARAMETER, "Prefab not found for card %d",
             card_id);
  const CardId *card_id_component = ecs_get_id(world, prefab, ecs_id(CardId));
  ecs_assert(card_id_component != NULL, ECS_INVALID_PARAMETER,
             "CardId component not found for prefab %d", prefab);

  for (size_t index = 0; index < count; index++) {
    // Give each card instance a unique name so Flecs doesn't reuse existing
    // entities
    char entity_name[64];
    const PlayerNumber *pnum = ecs_get(world, player, PlayerNumber);
    ecs_assert(pnum != NULL, ECS_INVALID_PARAMETER,
               "PlayerNumber component not found for player %d", player);

    uint8_t player_number = pnum->player_number;
    snprintf(entity_name, sizeof(entity_name), "%s_P%u_%zu",
             card_id_component->code, player_number, index + 1);
    ecs_entity_t card = ecs_new_w_pair(world, EcsIsA, prefab);
    ecs_set_name(world, card, entity_name);

    // Attach ability timing tags (AOnPlay, AResponse, etc.) if card has an
    // ability
    attach_ability_components(world, card);

    const Type *type = ecs_get_id(world, card, ecs_id(Type));
    ecs_assert(type != NULL, ECS_INVALID_PARAMETER,
               "Type component not found for card %d", card);

    ZonePlacement placement = zone_placement_for_type(zones, type->value);
    ecs_assert(placement.zone != 0, ECS_INVALID_PARAMETER,
               "Card zone not found for type %d", type->value);
    ecs_add_pair(world, card, EcsChildOf, placement.zone);
    ecs_add_pair(world, card, Rel_OwnedBy, player);

    // Used for validating the final card distribution sizes
    if (ecs_has_id(world, placement.zone, ZLeader)) {
      total_counts->leader_size++;
    } else if (ecs_has_id(world, placement.zone, ZGate)) {
      total_counts->gate_size++;
    } else if (ecs_has_id(world, placement.zone, ZIKZPileTag)) {
      total_counts->ikz_pile_size++;
    } else if (ecs_has_id(world, placement.zone, ZDeck)) {
      total_counts->deck_size++;
    }
  }
}

void init_player_deck(ecs_world_t *world, ecs_entity_t player,
                      DeckType deck_type, PlayerZones *zones) {
  TotalZoneCounts total_counts = {0};

  switch (deck_type) {
  case RAIZEN:
    for (size_t index = 0;
         index < (sizeof(raizenDeckCardInfo) / sizeof(raizenDeckCardInfo[0]));
         index++) {
      register_card(world, player, raizenDeckCardInfo[index].card_id,
                    (uint8_t)raizenDeckCardInfo[index].card_count, zones,
                    &total_counts);
    }
    break;
  case SHAO:
    for (size_t index = 0;
         index < (sizeof(shaoDeckCardInfo) / sizeof(shaoDeckCardInfo[0]));
         index++) {
      register_card(world, player, shaoDeckCardInfo[index].card_id,
                    (uint8_t)shaoDeckCardInfo[index].card_count, zones,
                    &total_counts);
    }
    break;
  }

  // Validate the final card distribution sizes
  if (total_counts.deck_size < REQUIRED_DECK_SIZE) {
    cli_render_logf("Error: Deck size is less than required (%d < %d)",
                    total_counts.deck_size, REQUIRED_DECK_SIZE);
    exit(EXIT_FAILURE);
  }
  if (total_counts.leader_size < REQUIRED_LEADER_SIZE) {
    cli_render_logf("Error: Leader size is less than required (%d < %d)",
                    total_counts.leader_size, REQUIRED_LEADER_SIZE);
    exit(EXIT_FAILURE);
  }
  if (total_counts.gate_size < REQUIRED_GATE_SIZE) {
    cli_render_logf("Error: Gate size is less than required (%d < %d)",
                    total_counts.gate_size, REQUIRED_GATE_SIZE);
    exit(EXIT_FAILURE);
  }
  if (total_counts.ikz_pile_size < REQUIRED_IKZ_PILE_SIZE) {
    cli_render_logf("Error: IKZ pile size is less than required (%d < %d)",
                    total_counts.ikz_pile_size, REQUIRED_IKZ_PILE_SIZE);
    exit(EXIT_FAILURE);
  }
}
