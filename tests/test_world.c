#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <flecs.h>

#include "world.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "abilities/cards/st01_007.h"
#include "abilities/cards/stt01_003.h"
#include "abilities/cards/stt01_005.h"
#include "components/abilities.h"
#include "components/components.h"
#include "components/game_log.h"
#include "systems/phase_gate.h"
#include "utils/card_utils.h"
#include "utils/combat_util.h"
#include "utils/deck_utils.h"
#include "utils/game_log_util.h"
#include "utils/observation_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"
#include "validation/action_enumerator.h"
#include "generated/card_defs.h"

static const CardDef *find_card_def_by_entity_name(const char *entity_name) {
  size_t base_length = strcspn(entity_name, "_");
  size_t count = 0;
  const CardDefLookupEntry *entries = azk_card_def_lookup_table(&count);
  for (size_t i = 0; i < count; i++) {
    if (strncmp(entries[i].card_id, entity_name, base_length) == 0 &&
        entries[i].card_id[base_length] == '\0') {
      return entries[i].def;
    }
  }
  return NULL;
}

static bool name_matches_card_id(const char *entity_name, const char *card_id) {
  size_t len = strlen(card_id);
  return strncmp(entity_name, card_id, len) == 0 &&
    (entity_name[len] == '\0' || entity_name[len] == '_');
}

static ecs_entity_t expected_tag_for_type(CardType type) {
  switch (type) {
  case CARD_TYPE_LEADER:
    return TLeader;
  case CARD_TYPE_GATE:
    return TGate;
  case CARD_TYPE_ENTITY:
    return TEntity;
  case CARD_TYPE_WEAPON:
    return TWeapon;
  case CARD_TYPE_SPELL:
    return TSpell;
  case CARD_TYPE_IKZ:
    return TIKZ;
  case CARD_TYPE_EXTRA_IKZ:
    return TExtraIKZCard;
  default:
    return 0;
  }
}

static ecs_entity_t expected_zone_for_type(CardType type, const PlayerZones *zones) {
  switch (type) {
  case CARD_TYPE_LEADER:
    return zones->leader;
  case CARD_TYPE_GATE:
    return zones->gate;
  case CARD_TYPE_ENTITY:
    return zones->deck;
  case CARD_TYPE_WEAPON:
    return zones->deck;
  case CARD_TYPE_SPELL:
    return zones->deck;
  case CARD_TYPE_IKZ:
    return zones->ikz_pile;
  case CARD_TYPE_EXTRA_IKZ:
    return zones->ikz_pile;
  default:
    return 0;
  }
}

static void assert_card_components(
  ecs_world_t *world,
  ecs_entity_t card,
  const CardDef *def,
  const PlayerZones *zones,
  ecs_entity_t player
) {
  ecs_entity_t type_tag = expected_tag_for_type(def->type);
  assert(type_tag != 0);
  assert(ecs_has_id(world, card, type_tag));

  ecs_entity_t expected_zone = expected_zone_for_type(def->type, zones);
  assert(expected_zone != 0);
  assert(ecs_has_pair(world, card, EcsChildOf, expected_zone));

  assert(ecs_has_pair(world, card, Rel_OwnedBy, player));

  const Element *element = ecs_get(world, card, Element);
  assert(element != NULL);
  assert(element->element == (uint8_t)def->element);

  bool has_base_stats = ecs_has(world, card, BaseStats);
  assert(has_base_stats == def->has_base_stats);
  if (def->has_base_stats) {
    const BaseStats *stats = ecs_get(world, card, BaseStats);
    assert(stats != NULL);
    assert(stats->attack == def->base_stats.attack);
    assert(stats->health == def->base_stats.health);
  }

  bool has_gate_points = ecs_has(world, card, GatePoints);
  assert(has_gate_points == def->has_gate_points);
  if (def->has_gate_points) {
    const GatePoints *gp = ecs_get(world, card, GatePoints);
    assert(gp != NULL);
    assert(gp->gate_points == def->gate_points.gate_points);
  }

  bool has_ikz_cost = ecs_has(world, card, IKZCost);
  assert(has_ikz_cost == def->has_ikz_cost);
  if (def->has_ikz_cost) {
    const IKZCost *cost = ecs_get(world, card, IKZCost);
    assert(cost != NULL);
    assert(cost->ikz_cost == def->ikz_cost.ikz_cost);
  }
}

static ecs_entity_t find_player_by_pid(ecs_world_t *world, uint8_t pid_value) {
  ecs_iter_t it = ecs_each_id(world, ecs_id(PlayerId));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t entity = it.entities[i];
      const PlayerId *pid = ecs_get(world, entity, PlayerId);
      if (pid && pid->pid == pid_value) {
        return entity;
      }
    }
  }

  return 0;
}

static ecs_entity_t find_zone_for_player(
  ecs_world_t *world,
  ecs_entity_t player,
  ecs_entity_t zone_tag
) {
  ecs_iter_t it = ecs_each_id(world, ecs_pair(Rel_OwnedBy, player));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t entity = it.entities[i];
      if (ecs_has_id(world, entity, zone_tag)) {
        return entity;
      }
    }
  }

  return 0;
}

static int collect_cards_of_type_in_zone(ecs_world_t *world, ecs_entity_t zone,
                                         CardType type, ecs_entity_t *out_cards,
                                         int max_cards) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  int found = 0;

  for (int32_t i = 0; i < cards.count && found < max_cards; i++) {
    ecs_entity_t card = cards.ids[i];
    const Type *card_type = ecs_get(world, card, Type);
    if (card_type != NULL && card_type->value == type) {
      out_cards[found++] = card;
    }
  }

  return found;
}

static ecs_entity_t create_zone(ecs_world_t *world, ecs_entity_t player,
                                ecs_entity_t zone_tag, const char *name);

static void test_azk_world_init_sets_game_state(void) {
  const uint32_t seed = 1234;
  ecs_world_t *world = azk_world_init(seed);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  assert(gs->seed == seed);
  assert(gs->phase == PHASE_PREGAME_MULLIGAN);
  assert(gs->starting_player_index >= 0);
  assert(gs->starting_player_index < MAX_PLAYERS_PER_MATCH);
  assert(gs->active_player_index == gs->starting_player_index);
  assert(gs->response_window == 0);
  assert(gs->winner == -1);

  bool seen[MAX_PLAYERS_PER_MATCH] = { false };

  ecs_iter_t it = ecs_each_id(world, ecs_id(PlayerId));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t entity = it.entities[i];
      const PlayerId *pid_comp = ecs_get(world, entity, PlayerId);
      if (!pid_comp) {
        continue;
      }
      uint8_t pid = pid_comp->pid;
      assert(pid < MAX_PLAYERS_PER_MATCH);
      seen[pid] = true;

      const PlayerNumber *pnum = ecs_get(world, entity, PlayerNumber);
      assert(pnum != NULL);
      assert(pnum->player_number == pid);
    }
  }

  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; player_index++) {
    assert(seen[player_index]);
  }

  azk_world_fini(world);
}

static void assert_zone_properties(
  ecs_world_t *world,
  ecs_entity_t zone,
  ecs_entity_t zone_tag,
  const char *expected_name,
  ecs_entity_t player
) {
  assert(zone != 0);
  const char *name = ecs_get_name(world, zone);
  assert(name != NULL);
  assert(strcmp(name, expected_name) == 0);
  assert(ecs_has_id(world, zone, zone_tag));
  assert(ecs_has_id(world, zone, EcsOrderedChildren));
  assert(ecs_has_pair(world, zone, Rel_OwnedBy, player));
}

static void test_world_init_creates_player_zones(void) {
  ecs_world_t *world = azk_world_init(77);

  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; player_index++) {
    ecs_entity_t player = find_player_by_pid(world, (uint8_t)player_index);
    assert(player != 0);

    char expected[32];

    ecs_entity_t deck = find_zone_for_player(world, player, ZDeck);
    snprintf(expected, sizeof(expected), "Deck_P%d", player_index);
    assert_zone_properties(world, deck, ZDeck, expected, player);

    ecs_entity_t hand = find_zone_for_player(world, player, ZHand);
    snprintf(expected, sizeof(expected), "Hand_P%d", player_index);
    assert_zone_properties(world, hand, ZHand, expected, player);

    ecs_entity_t leader = find_zone_for_player(world, player, ZLeader);
    snprintf(expected, sizeof(expected), "Leader_P%d", player_index);
    assert_zone_properties(world, leader, ZLeader, expected, player);

    ecs_entity_t gate = find_zone_for_player(world, player, ZGate);
    snprintf(expected, sizeof(expected), "Gate_P%d", player_index);
    assert_zone_properties(world, gate, ZGate, expected, player);

    ecs_entity_t garden = find_zone_for_player(world, player, ZGarden);
    snprintf(expected, sizeof(expected), "Garden_P%d", player_index);
    assert_zone_properties(world, garden, ZGarden, expected, player);

    ecs_entity_t alley = find_zone_for_player(world, player, ZAlley);
    snprintf(expected, sizeof(expected), "Alley_P%d", player_index);
    assert_zone_properties(world, alley, ZAlley, expected, player);

    ecs_entity_t ikz_pile = find_zone_for_player(world, player, ZIKZPileTag);
    snprintf(expected, sizeof(expected), "IKZPile_P%d", player_index);
    assert_zone_properties(world, ikz_pile, ZIKZPileTag, expected, player);

    ecs_entity_t ikz_area = find_zone_for_player(world, player, ZIKZAreaTag);
    snprintf(expected, sizeof(expected), "IKZArea_P%d", player_index);
    assert_zone_properties(world, ikz_area, ZIKZAreaTag, expected, player);

    ecs_entity_t discard = find_zone_for_player(world, player, ZDiscard);
    snprintf(expected, sizeof(expected), "Discard_P%d", player_index);
    assert_zone_properties(world, discard, ZDiscard, expected, player);
  }

  azk_world_fini(world);
}

static void test_world_init_assigns_damage_trackers_to_cards(void) {
  ecs_world_t *world = azk_world_init(78);

  ecs_iter_t it = ecs_each_id(world, ecs_id(CardId));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t entity = it.entities[i];
      if (ecs_has_id(world, entity, EcsPrefab)) {
        continue;
      }

      assert(ecs_has(world, entity, DamageTracker));
    }
  }

  azk_world_fini(world);
}

static void test_world_init_assigns_condition_countdowns_to_cards(void) {
  ecs_world_t *world = azk_world_init(79);

  ecs_iter_t it = ecs_each_id(world, ecs_id(CardId));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t entity = it.entities[i];
      if (ecs_has_id(world, entity, EcsPrefab)) {
        continue;
      }

      assert(ecs_has(world, entity, CardConditionCountdown));
    }
  }

  azk_world_fini(world);
}

static void test_apply_frozen_initializes_countdown_while_deferred(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t entity = ecs_new(world);
  ecs_add(world, entity, TEntity);

  ecs_defer_begin(world);
  apply_frozen(world, entity, 2);
  ecs_defer_end(world);

  const CardConditionCountdown *countdown =
      ecs_get(world, entity, CardConditionCountdown);
  assert(countdown != NULL);
  assert(countdown->frozen_duration == 2);
  assert(countdown->effect_immune_duration == 0);
  assert(ecs_has(world, entity, Frozen));

  ecs_fini(world);
}

static void test_count_tappable_ikz_sources_ignores_zero_token_entity(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = ecs_new(world);
  ecs_entity_t ikz_area = create_zone(world, player, ZIKZAreaTag, "IKZArea");
  ecs_entity_t ikz_card = ecs_new(world);
  ecs_set(world, ikz_card, Type, {.value = CARD_TYPE_IKZ});
  ecs_set(world, ikz_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, ikz_card, EcsChildOf, ikz_area);

  ecs_set(world, player, IKZToken, {.ikz_token = 0, .expires_eot = false});

  uint8_t count = azk_count_tappable_ikz_sources(world, ikz_area, true);
  assert(count == 1);

  ecs_fini(world);
}

static void test_get_tappable_ikz_cards_rejects_zero_token_entity(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = ecs_new(world);
  ecs_entity_t ikz_area = create_zone(world, player, ZIKZAreaTag, "IKZArea");
  ecs_entity_t ikz_card = ecs_new(world);
  ecs_set(world, ikz_card, Type, {.value = CARD_TYPE_IKZ});
  ecs_set(world, ikz_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, ikz_card, EcsChildOf, ikz_area);

  ecs_set(world, player, IKZToken, {.ikz_token = 0, .expires_eot = false});

  ecs_entity_t selected[AZK_MAX_IKZ_PAYMENT] = {0};
  uint8_t selected_count = 0;
  int result =
      get_tappable_ikz_cards(world, ikz_area, 1, &selected_count, selected, true);
  assert(result < 0);
  assert(selected_count == 0);

  ecs_fini(world);
}

static ecs_entity_t create_zone(
  ecs_world_t *world,
  ecs_entity_t player,
  ecs_entity_t zone_tag,
  const char *name
) {
  ecs_entity_t zone = ecs_new(world);
  ecs_set_name(world, zone, name);
  ecs_add_id(world, zone, zone_tag);
  ecs_add_id(world, zone, EcsOrderedChildren);
  ecs_add_pair(world, zone, Rel_OwnedBy, player);
  return zone;
}

static void setup_triggered_ability_control_fixture(
  ecs_world_t *world,
  ecs_entity_t players[MAX_PLAYERS_PER_MATCH],
  PlayerZones zones[MAX_PLAYERS_PER_MATCH],
  ecs_entity_t selis_cards[MAX_PLAYERS_PER_MATCH]
) {
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; player_index++) {
    players[player_index] = ecs_new(world);
    ecs_set(world, players[player_index], PlayerId, {.pid = (uint8_t)player_index});
    ecs_set(world, players[player_index], PlayerNumber,
            {.player_number = (uint8_t)player_index});

    char zone_name[32];
    snprintf(zone_name, sizeof(zone_name), "Hand_P%d", player_index);
    zones[player_index].hand =
        create_zone(world, players[player_index], ZHand, zone_name);
    snprintf(zone_name, sizeof(zone_name), "Deck_P%d", player_index);
    zones[player_index].deck =
        create_zone(world, players[player_index], ZDeck, zone_name);
    snprintf(zone_name, sizeof(zone_name), "Garden_P%d", player_index);
    zones[player_index].garden =
        create_zone(world, players[player_index], ZGarden, zone_name);
  }

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->winner = -1;
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; player_index++) {
    gs->players[player_index] = players[player_index];
    gs->zones[player_index] = zones[player_index];
  }
  ecs_singleton_modified(world, GameState);

  azk_clear_ability_context(world);

  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; player_index++) {
    for (int deck_card_index = 0; deck_card_index < 2; deck_card_index++) {
      ecs_entity_t deck_card = ecs_new(world);
      char deck_card_name[32];
      snprintf(deck_card_name, sizeof(deck_card_name), "DeckCard_P%d_%d",
               player_index, deck_card_index);
      ecs_set_name(world, deck_card, deck_card_name);
      ecs_add_pair(world, deck_card, Rel_OwnedBy, players[player_index]);
      ecs_add_pair(world, deck_card, EcsChildOf, zones[player_index].deck);
    }

    selis_cards[player_index] = ecs_new(world);
    char selis_name[32];
    snprintf(selis_name, sizeof(selis_name), "STT02-010_P%d", player_index);
    ecs_set_name(world, selis_cards[player_index], selis_name);
    ecs_set(world, selis_cards[player_index], CardId, {.id = CARD_DEF_STT02_010});
    ecs_set(world, selis_cards[player_index], TapState,
            {.tapped = false, .cooldown = false});
    ecs_add_pair(world, selis_cards[player_index], Rel_OwnedBy,
                 players[player_index]);
    ecs_add_pair(world, selis_cards[player_index], EcsChildOf,
                 zones[player_index].garden);
    ecs_set(world, selis_cards[player_index], ZoneIndex, {.index = 0});
  }
}

static void test_init_player_deck_raizen(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, { .pid = 0 });
  ecs_set(world, player, PlayerNumber, { .player_number = 0 });

  PlayerZones zones = {0};
  zones.deck = create_zone(world, player, ZDeck, "Deck");
  zones.hand = create_zone(world, player, ZHand, "Hand");
  zones.leader = create_zone(world, player, ZLeader, "Leader");
  zones.gate = create_zone(world, player, ZGate, "Gate");
  zones.garden = create_zone(world, player, ZGarden, "Garden");
  zones.alley = create_zone(world, player, ZAlley, "Alley");
  zones.ikz_pile = create_zone(world, player, ZIKZPileTag, "IKZPile");
  zones.ikz_area = create_zone(world, player, ZIKZAreaTag, "IKZArea");
  zones.discard = create_zone(world, player, ZDiscard, "Discard");

  // Set up GameState singleton (required for passive observer init)
  ecs_add_id(world, ecs_id(GameState), EcsSingleton);
  GameState gs = {0};
  gs.players[0] = player;
  gs.zones[0] = zones;
  ecs_singleton_set_ptr(world, GameState, &gs);

  static const CardInfo expected_cards[] = {
    { CARD_DEF_STT01_001, 1 },
    { CARD_DEF_STT01_002, 1 },
    { CARD_DEF_STT01_003, 4 },
    { CARD_DEF_STT01_004, 4 },
    { CARD_DEF_STT01_005, 4 },
    { CARD_DEF_STT01_006, 2 },
    { CARD_DEF_STT01_007, 4 },
    { CARD_DEF_STT01_008, 4 },
    { CARD_DEF_STT01_009, 4 },
    { CARD_DEF_STT01_010, 2 },
    { CARD_DEF_STT01_011, 2 },
    { CARD_DEF_STT01_012, 4 },
    { CARD_DEF_STT01_013, 4 },
    { CARD_DEF_STT01_014, 4 },
    { CARD_DEF_STT01_015, 2 },
    { CARD_DEF_STT01_016, 2 },
    { CARD_DEF_STT01_017, 4 },
    { CARD_DEF_IKZ_001, 10 },
  };

  int actual_counts[sizeof(expected_cards) / sizeof(expected_cards[0])] = {0};

  init_player_deck(world, player, RAIZEN, &zones);

  int total_cards = 0;
  ecs_iter_t it = ecs_each_id(world, ecs_pair(Rel_OwnedBy, player));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t card = it.entities[i];

      if (!ecs_has(world, card, Element)) {
        continue;
      }

      total_cards++;

      const char *code = ecs_get_name(world, card);
      assert(code != NULL);

      const CardDef *def = find_card_def_by_entity_name(code);
      assert(def != NULL);

      bool matched = false;
      for (size_t idx = 0; idx < (sizeof(expected_cards) / sizeof(expected_cards[0])); idx++) {
        const CardDef *expected_def = azk_card_def_from_id(expected_cards[idx].card_id);
        assert(expected_def != NULL);
        if (name_matches_card_id(code, expected_def->card_id)) {
          actual_counts[idx]++;
          matched = true;
          assert_card_components(world, card, def, &zones, player);
          break;
        }
      }

      assert(matched);
    }
  }

  int expected_total = 0;
  for (size_t idx = 0; idx < (sizeof(expected_cards) / sizeof(expected_cards[0])); idx++) {
    expected_total += expected_cards[idx].card_count;
    if (actual_counts[idx] != expected_cards[idx].card_count) {
      const CardDef *expected_def = azk_card_def_from_id(expected_cards[idx].card_id);
      fprintf(
        stderr,
        "Count mismatch for %s: expected %d, got %d\n",
        expected_def ? expected_def->card_id : "unknown",
        expected_cards[idx].card_count,
        actual_counts[idx]
      );
    }
    assert(actual_counts[idx] == expected_cards[idx].card_count);
  }
  assert(total_cards == expected_total);

  ecs_fini(world);
}

static void test_azk01_001_card_def_and_instantiation(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  const CardDef *def = azk_card_def_from_id(CARD_DEF_AZK01_001);
  assert(def != NULL);
  assert(strcmp(def->card_id, "AZK01-001") == 0);
  assert(strcmp(def->name, "Penny") == 0);
  assert(def->rarity == CARD_RARITY_C);
  assert(def->element == CARD_ELEMENT_NORMAL);
  assert(def->type == CARD_TYPE_ENTITY);
  assert(def->has_base_stats);
  assert(def->base_stats.attack == 0);
  assert(def->base_stats.health == 1);
  assert(def->has_gate_points);
  assert(def->gate_points.gate_points == 0);
  assert(def->has_ikz_cost);
  assert(def->ikz_cost.ikz_cost == 1);
  assert(!azk_has_ability(CARD_DEF_AZK01_001));

  ecs_entity_t prefab = azk_prefab_from_id(CARD_DEF_AZK01_001);
  assert(prefab != 0);
  assert(ecs_has(world, prefab, TEntity));
  assert(ecs_has(world, prefab, Defender));
  assert(ecs_has(world, prefab, TSubtype_Beanz));

  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  PlayerZones zones = {0};
  zones.deck = create_zone(world, player, ZDeck, "Deck_P0");

  ecs_entity_t card = ecs_new_w_pair(world, EcsIsA, prefab);
  ecs_set_name(world, card, "AZK01-001_P0_1");
  ecs_add_pair(world, card, EcsChildOf, zones.deck);
  ecs_add_pair(world, card, Rel_OwnedBy, player);
  attach_ability_components(world, card);

  assert_card_components(world, card, def, &zones, player);
  assert(ecs_has(world, card, Defender));
  assert(ecs_has(world, card, TSubtype_Beanz));

  ecs_fini(world);
}

static void test_ability_registry_lookup(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // ST01-007 should have an ability
  assert(azk_has_ability(CARD_DEF_STT01_007));
  const AbilityDef *def = azk_get_ability_def(CARD_DEF_STT01_007);
  assert(def != NULL);
  assert(def->has_ability);
  assert(def->is_optional);
  assert(def->timing_tag == ecs_id(AOnPlay));
  assert(def->cost_req.min == 1);
  assert(def->cost_req.max == 1);
  assert(def->cost_req.type == ABILITY_TARGET_FRIENDLY_HAND);
  assert(def->effect_req.min == 0);
  assert(def->effect_req.max == 0);
  assert(def->validate != NULL);
  assert(def->validate_cost_target != NULL);
  assert(def->apply_costs != NULL);
  assert(def->apply_effects != NULL);

  // STT01-003 should have an ability (mill effect)
  const AbilityDef *stt01_003_def = azk_get_ability_def(CARD_DEF_STT01_003);
  assert(stt01_003_def != NULL);
  assert(stt01_003_def->has_ability);
  assert(!stt01_003_def->is_optional);
  assert(stt01_003_def->timing_tag == ecs_id(AOnPlay));
  assert(stt01_003_def->cost_req.type == ABILITY_TARGET_NONE);
  assert(stt01_003_def->effect_req.type == ABILITY_TARGET_NONE);
  assert(stt01_003_def->validate != NULL);
  assert(stt01_003_def->apply_effects != NULL);

  const AbilityDef *stt01_012_def = azk_get_ability_def(CARD_DEF_STT01_012);
  assert(stt01_012_def != NULL);
  assert(stt01_012_def->has_ability);
  assert(!stt01_012_def->is_optional);

  const AbilityDef *stt01_006_def = azk_get_ability_def(CARD_DEF_STT01_006);
  assert(stt01_006_def != NULL);
  assert(stt01_006_def->has_ability);
  assert(!stt01_006_def->is_optional);

  const AbilityDef *stt01_014_def = azk_get_ability_def(CARD_DEF_STT01_014);
  assert(stt01_014_def != NULL);
  assert(stt01_014_def->has_ability);
  assert(!stt01_014_def->is_optional);

  const AbilityDef *stt01_016_def = azk_get_ability_def(CARD_DEF_STT01_016);
  assert(stt01_016_def != NULL);
  assert(stt01_016_def->has_ability);
  assert(!stt01_016_def->is_optional);

  const AbilityDef *stt02_003_def = azk_get_ability_def(CARD_DEF_STT02_003);
  assert(stt02_003_def != NULL);
  assert(stt02_003_def->has_ability);
  assert(!stt02_003_def->is_optional);

  const AbilityDef *stt02_005_def = azk_get_ability_def(CARD_DEF_STT02_005);
  assert(stt02_005_def != NULL);
  assert(stt02_005_def->has_ability);
  assert(!stt02_005_def->is_optional);

  const AbilityDef *stt02_007_def = azk_get_ability_def(CARD_DEF_STT02_007);
  assert(stt02_007_def != NULL);
  assert(stt02_007_def->has_ability);
  assert(!stt02_007_def->is_optional);

  const AbilityDef *stt02_013_def = azk_get_ability_def(CARD_DEF_STT02_013);
  assert(stt02_013_def != NULL);
  assert(stt02_013_def->has_ability);
  assert(!stt02_013_def->is_optional);

  const AbilityDef *azk01_002_def = azk_get_ability_def(CARD_DEF_AZK01_002);
  assert(azk01_002_def != NULL);
  assert(azk01_002_def->has_ability);
  assert(!azk01_002_def->is_optional);
  assert(azk01_002_def->timing_tag == ecs_id(AMain));
  assert(azk01_002_def->cost_req.type == ABILITY_TARGET_NONE);
  assert(azk01_002_def->effect_req.type == ABILITY_TARGET_NONE);
  assert(azk01_002_def->validate != NULL);
  assert(azk01_002_def->apply_effects != NULL);

  const AbilityDef *azk01_003_def = azk_get_ability_def(CARD_DEF_AZK01_003);
  assert(azk01_003_def != NULL);
  assert(azk01_003_def->has_ability);
  assert(!azk01_003_def->is_optional);
  assert(azk01_003_def->timing_tag == ecs_id(AOnPlay));
  assert(azk01_003_def->validate != NULL);
  assert(azk01_003_def->on_cost_paid != NULL);
  assert(azk01_003_def->validate_selection_target != NULL);
  assert(azk01_003_def->on_selection_complete != NULL);

  // A card without ability should return NULL or has_ability=false
  const AbilityDef *no_ability = azk_get_ability_def(CARD_DEF_IKZ_001);
  assert(no_ability == NULL || !no_ability->has_ability);

  ecs_fini(world);
}

static void test_st01_007_validate_needs_hand_and_deck(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, { .pid = 0 });
  ecs_set(world, player, PlayerNumber, { .player_number = 0 });

  // Create zones
  ecs_entity_t hand = create_zone(world, player, ZHand, "Hand");
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck");

  // Set up GameState with zones
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].hand = hand;
  gs->zones[0].deck = deck;
  ecs_singleton_modified(world, GameState);

  // Create the ST01-007 card (source of ability)
  ecs_entity_t source_card = ecs_new(world);
  ecs_set(world, source_card, CardId, { .id = CARD_DEF_STT01_007 });

  // Test: Empty hand + empty deck = validation fails
  assert(!st01_007_validate(world, source_card, player));

  // Add a card to hand
  ecs_entity_t hand_card = ecs_new(world);
  ecs_add_pair(world, hand_card, EcsChildOf, hand);
  ecs_set(world, hand_card, Element, { .element = 1 });

  // Test: Hand has card, but deck is empty = validation fails
  assert(!st01_007_validate(world, source_card, player));

  // Add a card to deck
  ecs_entity_t deck_card = ecs_new(world);
  ecs_add_pair(world, deck_card, EcsChildOf, deck);
  ecs_set(world, deck_card, Element, { .element = 1 });

  // Test: Hand has card and deck has card = validation passes
  assert(st01_007_validate(world, source_card, player));

  ecs_fini(world);
}

static void test_st01_007_validate_cost_target(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create two players
  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, { .pid = 0 });
  ecs_set(world, player0, PlayerNumber, { .player_number = 0 });

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, { .pid = 1 });
  ecs_set(world, player1, PlayerNumber, { .player_number = 1 });

  // Create zones
  ecs_entity_t hand0 = create_zone(world, player0, ZHand, "Hand_P0");
  ecs_entity_t hand1 = create_zone(world, player1, ZHand, "Hand_P1");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].hand = hand0;
  gs->zones[1].hand = hand1;
  ecs_singleton_modified(world, GameState);

  // Create source card
  ecs_entity_t source_card = ecs_new(world);
  ecs_set(world, source_card, CardId, { .id = CARD_DEF_STT01_007 });

  // Create card in player0's hand
  ecs_entity_t card_in_hand = ecs_new(world);
  ecs_add_pair(world, card_in_hand, EcsChildOf, hand0);
  ecs_add_pair(world, card_in_hand, Rel_OwnedBy, player0);
  ecs_set(world, card_in_hand, Element, { .element = 1 });

  // Create card in player1's hand
  ecs_entity_t enemy_card = ecs_new(world);
  ecs_add_pair(world, enemy_card, EcsChildOf, hand1);
  ecs_add_pair(world, enemy_card, Rel_OwnedBy, player1);
  ecs_set(world, enemy_card, Element, { .element = 1 });

  // Test: Can target own hand card
  assert(st01_007_validate_cost_target(world, source_card, player0, card_in_hand));

  // Test: Cannot target enemy's hand card
  assert(!st01_007_validate_cost_target(world, source_card, player0, enemy_card));

  // Test: Cannot target non-existent card
  assert(!st01_007_validate_cost_target(world, source_card, player0, 0));

  ecs_fini(world);
}

static void test_st01_007_ability_flow_confirm_and_execute(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, { .pid = 0 });
  ecs_set(world, player, PlayerNumber, { .player_number = 0 });

  // Create zones
  ecs_entity_t hand = create_zone(world, player, ZHand, "Hand_P0");
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t discard = create_zone(world, player, ZDiscard, "Discard_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].hand = hand;
  gs->zones[0].deck = deck;
  gs->zones[0].discard = discard;
  ecs_singleton_modified(world, GameState);

  // Initialize AbilityContext singleton
  azk_clear_ability_context(world);

  // Create the ST01-007 card (the card that was played)
  ecs_entity_t st01_007_card = ecs_new(world);
  ecs_set_name(world, st01_007_card, "ST01-007_test");
  ecs_set(world, st01_007_card, CardId, { .id = CARD_DEF_STT01_007 });
  ecs_set(world, st01_007_card, Element, { .element = 1 });

  // Create a card in hand (to discard)
  ecs_entity_t hand_card = ecs_new(world);
  ecs_set_name(world, hand_card, "HandCard");
  ecs_add_pair(world, hand_card, EcsChildOf, hand);
  ecs_add_pair(world, hand_card, Rel_OwnedBy, player);
  ecs_set(world, hand_card, Element, { .element = 2 });

  // Create cards in deck (to draw from)
  ecs_entity_t deck_card1 = ecs_new(world);
  ecs_set_name(world, deck_card1, "DeckCard1");
  ecs_add_pair(world, deck_card1, EcsChildOf, deck);
  ecs_add_pair(world, deck_card1, Rel_OwnedBy, player);
  ecs_set(world, deck_card1, Element, { .element = 3 });

  ecs_entity_t deck_card2 = ecs_new(world);
  ecs_set_name(world, deck_card2, "DeckCard2");
  ecs_add_pair(world, deck_card2, EcsChildOf, deck);
  ecs_add_pair(world, deck_card2, Rel_OwnedBy, player);
  ecs_set(world, deck_card2, Element, { .element = 4 });

  // Verify initial state
  ecs_entities_t initial_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t initial_deck = ecs_get_ordered_children(world, deck);
  assert(initial_hand.count == 1);
  assert(initial_deck.count == 2);

  // Step 1: Queue the ability (queues for next loop iteration)
  bool queued = azk_trigger_on_play_ability(world, st01_007_card, player);
  assert(queued);

  // Ability is queued, not yet in ability phase
  assert(azk_has_queued_triggered_effects(world));
  assert(!azk_is_in_ability_phase(world));

  // Process the queue (simulates what happens on next game loop)
  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);

  // Should be in confirmation phase (optional ability)
  assert(azk_is_in_ability_phase(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);

  // Step 2: Confirm the ability
  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);

  // Should now be in cost selection phase
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_COST_SELECTION);

  // Step 3: Select cost target (index 0 = the hand card)
  bool cost_selected = azk_process_cost_selection(world, 0);
  assert(cost_selected);

  // Ability should be complete (effect has no targets to select)
  assert(!azk_is_in_ability_phase(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  // Verify final state: hand_card should be in discard, deck_card2 should be in hand
  // (draw takes from end of deck: cards.ids[count - 1 - index])
  ecs_entities_t final_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_discard = ecs_get_ordered_children(world, discard);

  assert(final_hand.count == 1);  // Drew 1 card
  assert(final_deck.count == 1);  // Lost 1 card from deck
  assert(final_discard.count == 1);  // Discarded 1 card

  // The drawn card should be the last card from the deck (deck_card2)
  assert(final_hand.ids[0] == deck_card2);
  // The remaining deck card should be deck_card1
  assert(final_deck.ids[0] == deck_card1);
  // The discarded card should be the original hand card
  assert(final_discard.ids[0] == hand_card);

  ecs_fini(world);
}

static void test_st01_007_ability_flow_decline(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, { .pid = 0 });
  ecs_set(world, player, PlayerNumber, { .player_number = 0 });

  // Create zones
  ecs_entity_t hand = create_zone(world, player, ZHand, "Hand_P0");
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t discard = create_zone(world, player, ZDiscard, "Discard_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].hand = hand;
  gs->zones[0].deck = deck;
  gs->zones[0].discard = discard;
  ecs_singleton_modified(world, GameState);

  // Initialize AbilityContext singleton
  azk_clear_ability_context(world);

  // Create the ST01-007 card
  ecs_entity_t st01_007_card = ecs_new(world);
  ecs_set(world, st01_007_card, CardId, { .id = CARD_DEF_STT01_007 });
  ecs_set(world, st01_007_card, Element, { .element = 1 });

  // Create a card in hand
  ecs_entity_t hand_card = ecs_new(world);
  ecs_add_pair(world, hand_card, EcsChildOf, hand);
  ecs_add_pair(world, hand_card, Rel_OwnedBy, player);
  ecs_set(world, hand_card, Element, { .element = 2 });

  // Create a card in deck
  ecs_entity_t deck_card = ecs_new(world);
  ecs_add_pair(world, deck_card, EcsChildOf, deck);
  ecs_add_pair(world, deck_card, Rel_OwnedBy, player);
  ecs_set(world, deck_card, Element, { .element = 3 });

  // Step 1: Queue the ability
  bool queued = azk_trigger_on_play_ability(world, st01_007_card, player);
  assert(queued);

  // Ability is queued, not yet in ability phase
  assert(azk_has_queued_triggered_effects(world));
  assert(!azk_is_in_ability_phase(world));

  // Process the queue (simulates what happens on next game loop)
  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);

  // Step 2: Decline the ability (use process_ability_decline)
  bool declined = azk_process_ability_decline(world);
  assert(declined);

  // Ability should be cleared
  assert(!azk_is_in_ability_phase(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  // Verify state unchanged: hand and deck should be same as before
  ecs_entities_t final_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_discard = ecs_get_ordered_children(world, discard);

  assert(final_hand.count == 1);
  assert(final_deck.count == 1);
  assert(final_discard.count == 0);
  assert(final_hand.ids[0] == hand_card);
  assert(final_deck.ids[0] == deck_card);

  ecs_fini(world);
}

// ============================================================================
// STT01-003 Tests: "On Play; Put 3 cards from the top of your deck into your
// discard pile. If you have no weapon cards in your discard pile when you
// activate this ability, put 5 cards instead."
// ============================================================================

static void test_stt01_003_mills_5_without_weapons(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  // Create zones
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t discard = create_zone(world, player, ZDiscard, "Discard_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].discard = discard;
  ecs_singleton_modified(world, GameState);

  // Initialize AbilityContext singleton
  azk_clear_ability_context(world);

  // Create deck cards (7 cards - enough to mill 5)
  ecs_entity_t deck_cards[7];
  for (int i = 0; i < 7; i++) {
    deck_cards[i] = ecs_new(world);
    char name[32];
    snprintf(name, sizeof(name), "DeckCard%d", i);
    ecs_set_name(world, deck_cards[i], name);
    ecs_add_pair(world, deck_cards[i], EcsChildOf, deck);
    ecs_add_pair(world, deck_cards[i], Rel_OwnedBy, player);
    // All entity cards (not weapons)
    ecs_set(world, deck_cards[i], Type, {.value = CARD_TYPE_ENTITY});
  }

  // Create the STT01-003 card
  ecs_entity_t stt01_003_card = ecs_new(world);
  ecs_set_name(world, stt01_003_card, "STT01-003_test");
  ecs_set(world, stt01_003_card, CardId, {.id = CARD_DEF_STT01_003});

  // Verify initial state: 7 cards in deck, 0 in discard (no weapons)
  ecs_entities_t initial_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t initial_discard = ecs_get_ordered_children(world, discard);
  assert(initial_deck.count == 7);
  assert(initial_discard.count == 0);

  // Queue the ability
  bool queued = azk_trigger_on_play_ability(world, stt01_003_card, player);
  assert(queued);

  // Process the queue
  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);

  // Mandatory triggered ability should resolve immediately.
  assert(!azk_is_in_ability_phase(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  // Verify final state: 5 cards milled (no weapons in discard)
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_discard = ecs_get_ordered_children(world, discard);

  assert(final_deck.count == 2);    // 7 - 5 = 2 remaining
  assert(final_discard.count == 5); // 5 cards milled

  // Verify the correct cards were milled (top 5 from deck = last 5 in array)
  // deck_cards[6], deck_cards[5], deck_cards[4], deck_cards[3], deck_cards[2]
  // should now be in discard
  assert(final_deck.ids[0] == deck_cards[0]);
  assert(final_deck.ids[1] == deck_cards[1]);

  ecs_fini(world);
}

static void test_stt01_003_mills_3_with_weapons(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  // Create zones
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t discard = create_zone(world, player, ZDiscard, "Discard_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].discard = discard;
  ecs_singleton_modified(world, GameState);

  // Initialize AbilityContext singleton
  azk_clear_ability_context(world);

  // Create a weapon card in discard
  ecs_entity_t weapon_in_discard = ecs_new(world);
  ecs_set_name(world, weapon_in_discard, "WeaponInDiscard");
  ecs_add_pair(world, weapon_in_discard, EcsChildOf, discard);
  ecs_add_pair(world, weapon_in_discard, Rel_OwnedBy, player);
  ecs_set(world, weapon_in_discard, Type, {.value = CARD_TYPE_WEAPON});

  // Create deck cards (7 cards - enough to mill 3)
  ecs_entity_t deck_cards[7];
  for (int i = 0; i < 7; i++) {
    deck_cards[i] = ecs_new(world);
    char name[32];
    snprintf(name, sizeof(name), "DeckCard%d", i);
    ecs_set_name(world, deck_cards[i], name);
    ecs_add_pair(world, deck_cards[i], EcsChildOf, deck);
    ecs_add_pair(world, deck_cards[i], Rel_OwnedBy, player);
    ecs_set(world, deck_cards[i], Type, {.value = CARD_TYPE_ENTITY});
  }

  // Create the STT01-003 card
  ecs_entity_t stt01_003_card = ecs_new(world);
  ecs_set_name(world, stt01_003_card, "STT01-003_test");
  ecs_set(world, stt01_003_card, CardId, {.id = CARD_DEF_STT01_003});

  // Verify initial state: 7 cards in deck, 1 weapon in discard
  ecs_entities_t initial_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t initial_discard = ecs_get_ordered_children(world, discard);
  assert(initial_deck.count == 7);
  assert(initial_discard.count == 1);

  // Queue the ability
  bool queued = azk_trigger_on_play_ability(world, stt01_003_card, player);
  assert(queued);

  // Process the queue
  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);

  // Mandatory triggered ability should resolve immediately.
  assert(!azk_is_in_ability_phase(world));

  // Verify final state: 3 cards milled (weapon was in discard)
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_discard = ecs_get_ordered_children(world, discard);

  assert(final_deck.count == 4);    // 7 - 3 = 4 remaining
  assert(final_discard.count == 4); // 1 original + 3 milled

  // Verify the remaining deck cards (first 4 should remain)
  assert(final_deck.ids[0] == deck_cards[0]);
  assert(final_deck.ids[1] == deck_cards[1]);
  assert(final_deck.ids[2] == deck_cards[2]);
  assert(final_deck.ids[3] == deck_cards[3]);

  ecs_fini(world);
}

static void test_stt01_003_resolves_without_confirmation(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  // Create zones
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t discard = create_zone(world, player, ZDiscard, "Discard_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].discard = discard;
  ecs_singleton_modified(world, GameState);

  // Initialize AbilityContext singleton
  azk_clear_ability_context(world);

  // Create deck cards (5 cards)
  ecs_entity_t deck_cards[5];
  for (int i = 0; i < 5; i++) {
    deck_cards[i] = ecs_new(world);
    char name[32];
    snprintf(name, sizeof(name), "DeckCard%d", i);
    ecs_set_name(world, deck_cards[i], name);
    ecs_add_pair(world, deck_cards[i], EcsChildOf, deck);
    ecs_add_pair(world, deck_cards[i], Rel_OwnedBy, player);
    ecs_set(world, deck_cards[i], Type, {.value = CARD_TYPE_ENTITY});
  }

  // Create the STT01-003 card
  ecs_entity_t stt01_003_card = ecs_new(world);
  ecs_set_name(world, stt01_003_card, "STT01-003_test");
  ecs_set(world, stt01_003_card, CardId, {.id = CARD_DEF_STT01_003});

  // Verify initial state: 5 cards in deck, 0 in discard
  ecs_entities_t initial_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t initial_discard = ecs_get_ordered_children(world, discard);
  assert(initial_deck.count == 5);
  assert(initial_discard.count == 0);

  // Queue the ability
  bool queued = azk_trigger_on_play_ability(world, stt01_003_card, player);
  assert(queued);

  // Process the queue
  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);

  // Mandatory triggered ability should not enter confirmation.
  assert(!azk_is_in_ability_phase(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  // Verify the effect resolved immediately.
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_discard = ecs_get_ordered_children(world, discard);

  assert(final_deck.count == 0);
  assert(final_discard.count == 5);

  ecs_fini(world);
}

static void test_stt01_003_mills_all_if_deck_smaller(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  // Create zones
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t discard = create_zone(world, player, ZDiscard, "Discard_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].discard = discard;
  ecs_singleton_modified(world, GameState);

  // Initialize AbilityContext singleton
  azk_clear_ability_context(world);

  // Create only 2 cards in deck (less than 5 or 3)
  ecs_entity_t deck_cards[2];
  for (int i = 0; i < 2; i++) {
    deck_cards[i] = ecs_new(world);
    char name[32];
    snprintf(name, sizeof(name), "DeckCard%d", i);
    ecs_set_name(world, deck_cards[i], name);
    ecs_add_pair(world, deck_cards[i], EcsChildOf, deck);
    ecs_add_pair(world, deck_cards[i], Rel_OwnedBy, player);
    ecs_set(world, deck_cards[i], Type, {.value = CARD_TYPE_ENTITY});
  }

  // Create the STT01-003 card
  ecs_entity_t stt01_003_card = ecs_new(world);
  ecs_set_name(world, stt01_003_card, "STT01-003_test");
  ecs_set(world, stt01_003_card, CardId, {.id = CARD_DEF_STT01_003});

  // Verify initial state: 2 cards in deck, 0 in discard (no weapons)
  ecs_entities_t initial_deck = ecs_get_ordered_children(world, deck);
  assert(initial_deck.count == 2);

  // Queue and process the ability
  bool queued = azk_trigger_on_play_ability(world, stt01_003_card, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);

  // Ability should complete immediately.
  assert(!azk_is_in_ability_phase(world));

  // Verify final state: all 2 cards milled (would mill 5 but only 2 available)
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_discard = ecs_get_ordered_children(world, discard);

  assert(final_deck.count == 0);    // All milled
  assert(final_discard.count == 2); // Both cards in discard

  // Milling the last card in deck causes deck-out.
  const GameState *final_gs = ecs_singleton_get(world, GameState);
  assert(final_gs->winner == 1);
  assert(final_gs->phase == PHASE_END_MATCH);

  ecs_fini(world);
}

// ============================================================================
// STT01-005 Tests: "Main; Alley Only; You may sacrifice this card: Draw 3 cards and discard 2"
// ============================================================================

static void test_stt01_005_ability_registry_check(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // STT01-005 should have an ability
  assert(azk_has_ability(CARD_DEF_STT01_005));
  const AbilityDef *def = azk_get_ability_def(CARD_DEF_STT01_005);
  assert(def != NULL);
  assert(def->has_ability);
  // Note: is_optional doesn't matter for main abilities - player explicitly triggers them
  assert(!def->is_optional);
  assert(def->timing_tag == ecs_id(AMain));
  // No cost selection (sacrifice is automatic)
  assert(def->cost_req.type == ABILITY_TARGET_NONE);
  assert(def->cost_req.min == 0);
  assert(def->cost_req.max == 0);
  // Effect requires selecting 2 cards from hand to discard
  assert(def->effect_req.type == ABILITY_TARGET_FRIENDLY_HAND);
  assert(def->effect_req.min == 2);
  assert(def->effect_req.max == 2);
  assert(def->validate != NULL);
  assert(def->validate_effect_target != NULL);
  assert(def->apply_costs != NULL);
  assert(def->apply_effects != NULL);

  ecs_fini(world);
}

static void test_stt01_005_validate_needs_deck_cards(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, { .pid = 0 });
  ecs_set(world, player, PlayerNumber, { .player_number = 0 });

  // Create zones
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t alley = create_zone(world, player, ZAlley, "Alley_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].alley = alley;
  ecs_singleton_modified(world, GameState);

  // Create the STT01-005 card in alley
  ecs_entity_t stt01_005_card = ecs_new(world);
  ecs_set(world, stt01_005_card, CardId, { .id = CARD_DEF_STT01_005 });
  ecs_add_pair(world, stt01_005_card, EcsChildOf, alley);

  // Test: Empty deck = validation fails
  assert(!stt01_005_validate(world, stt01_005_card, player));

  // Add a card to deck
  ecs_entity_t deck_card = ecs_new(world);
  ecs_add_pair(world, deck_card, EcsChildOf, deck);
  ecs_set(world, deck_card, Element, { .element = 1 });

  // Test: Deck has card = validation passes
  assert(stt01_005_validate(world, stt01_005_card, player));

  ecs_fini(world);
}

static void test_stt01_005_validate_only_in_alley(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, { .pid = 0 });
  ecs_set(world, player, PlayerNumber, { .player_number = 0 });

  // Create zones
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t alley = create_zone(world, player, ZAlley, "Alley_P0");
  ecs_entity_t garden = create_zone(world, player, ZGarden, "Garden_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].alley = alley;
  gs->zones[0].garden = garden;
  ecs_singleton_modified(world, GameState);

  // Add cards to deck so that's not the blocker
  for (int i = 0; i < 5; i++) {
    ecs_entity_t deck_card = ecs_new(world);
    ecs_add_pair(world, deck_card, EcsChildOf, deck);
    ecs_set(world, deck_card, Element, { .element = 1 });
  }

  // Create the STT01-005 card in garden (not alley)
  ecs_entity_t stt01_005_card = ecs_new(world);
  ecs_set(world, stt01_005_card, CardId, { .id = CARD_DEF_STT01_005 });
  ecs_add_pair(world, stt01_005_card, EcsChildOf, garden);

  // Test: Card in garden = validation fails (AAlleyOnly)
  assert(!stt01_005_validate(world, stt01_005_card, player));

  // Move card to alley
  ecs_remove_pair(world, stt01_005_card, EcsChildOf, garden);
  ecs_add_pair(world, stt01_005_card, EcsChildOf, alley);

  // Test: Card in alley = validation passes
  assert(stt01_005_validate(world, stt01_005_card, player));

  ecs_fini(world);
}

static void test_stt01_005_validate_effect_target(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create two players
  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, { .pid = 0 });
  ecs_set(world, player0, PlayerNumber, { .player_number = 0 });

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, { .pid = 1 });
  ecs_set(world, player1, PlayerNumber, { .player_number = 1 });

  // Create zones
  ecs_entity_t hand0 = create_zone(world, player0, ZHand, "Hand_P0");
  ecs_entity_t hand1 = create_zone(world, player1, ZHand, "Hand_P1");
  ecs_entity_t deck0 = create_zone(world, player0, ZDeck, "Deck_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].hand = hand0;
  gs->zones[0].deck = deck0;
  gs->zones[1].hand = hand1;
  ecs_singleton_modified(world, GameState);

  // Create source card
  ecs_entity_t source_card = ecs_new(world);
  ecs_set(world, source_card, CardId, { .id = CARD_DEF_STT01_005 });

  // Create card in player0's hand
  ecs_entity_t card_in_hand = ecs_new(world);
  ecs_add_pair(world, card_in_hand, EcsChildOf, hand0);
  ecs_add_pair(world, card_in_hand, Rel_OwnedBy, player0);
  ecs_set(world, card_in_hand, Element, { .element = 1 });

  // Create card in player1's hand
  ecs_entity_t enemy_card = ecs_new(world);
  ecs_add_pair(world, enemy_card, EcsChildOf, hand1);
  ecs_add_pair(world, enemy_card, Rel_OwnedBy, player1);
  ecs_set(world, enemy_card, Element, { .element = 1 });

  // Create card in deck (not in hand)
  ecs_entity_t deck_card = ecs_new(world);
  ecs_add_pair(world, deck_card, EcsChildOf, deck0);
  ecs_add_pair(world, deck_card, Rel_OwnedBy, player0);
  ecs_set(world, deck_card, Element, { .element = 1 });

  // Test: Can target own hand card
  assert(stt01_005_validate_effect_target(world, source_card, player0, card_in_hand));

  // Test: Cannot target enemy's hand card
  assert(!stt01_005_validate_effect_target(world, source_card, player0, enemy_card));

  // Test: Cannot target card in deck
  assert(!stt01_005_validate_effect_target(world, source_card, player0, deck_card));

  // Test: Cannot target non-existent card
  assert(!stt01_005_validate_effect_target(world, source_card, player0, 0));

  ecs_fini(world);
}

static void test_stt01_005_ability_flow_full(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create player
  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, { .pid = 0 });
  ecs_set(world, player, PlayerNumber, { .player_number = 0 });

  // Create zones
  ecs_entity_t hand = create_zone(world, player, ZHand, "Hand_P0");
  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t alley = create_zone(world, player, ZAlley, "Alley_P0");
  ecs_entity_t discard = create_zone(world, player, ZDiscard, "Discard_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].hand = hand;
  gs->zones[0].deck = deck;
  gs->zones[0].alley = alley;
  gs->zones[0].discard = discard;
  ecs_singleton_modified(world, GameState);

  // Initialize AbilityContext singleton
  azk_clear_ability_context(world);

  // Create the STT01-005 card in alley
  ecs_entity_t stt01_005_card = ecs_new(world);
  ecs_set_name(world, stt01_005_card, "STT01-005_test");
  ecs_set(world, stt01_005_card, CardId, { .id = CARD_DEF_STT01_005 });
  ecs_set(world, stt01_005_card, Element, { .element = 1 });
  ecs_add_pair(world, stt01_005_card, EcsChildOf, alley);
  ecs_add_pair(world, stt01_005_card, Rel_OwnedBy, player);

  // Create 2 cards already in hand (will keep one, discard both after draw)
  ecs_entity_t hand_card1 = ecs_new(world);
  ecs_set_name(world, hand_card1, "HandCard1");
  ecs_add_pair(world, hand_card1, EcsChildOf, hand);
  ecs_add_pair(world, hand_card1, Rel_OwnedBy, player);
  ecs_set(world, hand_card1, Element, { .element = 2 });

  ecs_entity_t hand_card2 = ecs_new(world);
  ecs_set_name(world, hand_card2, "HandCard2");
  ecs_add_pair(world, hand_card2, EcsChildOf, hand);
  ecs_add_pair(world, hand_card2, Rel_OwnedBy, player);
  ecs_set(world, hand_card2, Element, { .element = 3 });

  // Create 5 cards in deck (will draw 3)
  ecs_entity_t deck_cards[5];
  for (int i = 0; i < 5; i++) {
    deck_cards[i] = ecs_new(world);
    char name[32];
    snprintf(name, sizeof(name), "DeckCard%d", i);
    ecs_set_name(world, deck_cards[i], name);
    ecs_add_pair(world, deck_cards[i], EcsChildOf, deck);
    ecs_add_pair(world, deck_cards[i], Rel_OwnedBy, player);
    ecs_set(world, deck_cards[i], Element, { .element = (uint8_t)(4 + i) });
  }

  // Verify initial state
  ecs_entities_t initial_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t initial_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t initial_alley = ecs_get_ordered_children(world, alley);
  assert(initial_hand.count == 2);
  assert(initial_deck.count == 5);
  assert(initial_alley.count == 1);

  // Step 1: Trigger the ability
  // Main abilities skip confirmation (player already opted in via action)
  bool triggered = azk_trigger_main_ability(world, stt01_005_card, player);
  assert(triggered);

  // Should go directly to effect selection (costs applied, sacrifice + draw 3)
  assert(azk_is_in_ability_phase(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  // Verify: card was sacrificed (moved to discard), 3 cards drawn
  ecs_entities_t after_cost_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t after_cost_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t after_cost_alley = ecs_get_ordered_children(world, alley);
  ecs_entities_t after_cost_discard = ecs_get_ordered_children(world, discard);
  assert(after_cost_hand.count == 5);   // 2 original + 3 drawn
  assert(after_cost_deck.count == 2);   // 5 - 3 = 2
  assert(after_cost_alley.count == 0);  // sacrificed
  assert(after_cost_discard.count == 1); // the sacrificed card

  // Step 3: Select first card to discard (select index 0 in hand)
  bool effect1_selected = azk_process_effect_selection(world, 0);
  assert(effect1_selected);
  // Still in effect selection (need to select 2 cards)
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  // Step 4: Select second card to discard (select index 0 again - next card after first was removed conceptually)
  // Note: The hand still has all cards until apply_effects, so we select index 1
  bool effect2_selected = azk_process_effect_selection(world, 1);
  assert(effect2_selected);

  // Ability should be complete now
  assert(!azk_is_in_ability_phase(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  // Verify final state
  ecs_entities_t final_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_discard = ecs_get_ordered_children(world, discard);

  assert(final_hand.count == 3);    // 5 - 2 discarded
  assert(final_deck.count == 2);    // 5 - 3 drawn
  assert(final_discard.count == 3); // 1 sacrificed + 2 discarded

  ecs_fini(world);
}

static void test_draw_cards_with_deckout_check(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});

  // Create players
  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, { .pid = 0 });
  ecs_set(world, player0, PlayerNumber, { .player_number = 0 });

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, { .pid = 1 });
  ecs_set(world, player1, PlayerNumber, { .player_number = 1 });

  // Create zones
  ecs_entity_t hand = create_zone(world, player0, ZHand, "Hand_P0");
  ecs_entity_t deck = create_zone(world, player0, ZDeck, "Deck_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].hand = hand;
  gs->zones[0].deck = deck;
  gs->phase = PHASE_MAIN;
  gs->winner = -1;
  ecs_singleton_modified(world, GameState);

  // Create exactly 3 cards in deck
  ecs_entity_t deck_cards[3];
  for (int i = 0; i < 3; i++) {
    deck_cards[i] = ecs_new(world);
    ecs_add_pair(world, deck_cards[i], EcsChildOf, deck);
    ecs_set(world, deck_cards[i], Element, { .element = (uint8_t)(i + 1) });
  }

  // Test: Draw 3 cards when deck has exactly 3 = deck-out (deck empty after draw)
  ecs_entity_t drawn[3];
  bool success = draw_cards_with_deckout_check(world, player0, 3, drawn);

  // Should return false because deck is empty after drawing
  assert(!success);

  // Verify deck-out occurred
  gs = ecs_singleton_get_mut(world, GameState);
  assert(gs->winner == 1);  // Player 1 wins (opponent of player 0)
  assert(gs->phase == PHASE_END_MATCH);

  // Verify cards were drawn before deck-out
  ecs_entities_t final_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  assert(final_hand.count == 3);  // All 3 cards drawn
  assert(final_deck.count == 0);  // Deck is empty

  ecs_fini(world);
}

static void test_draw_cards_with_deckout_check_success(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});

  // Create players
  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, { .pid = 0 });
  ecs_set(world, player0, PlayerNumber, { .player_number = 0 });

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, { .pid = 1 });
  ecs_set(world, player1, PlayerNumber, { .player_number = 1 });

  // Create zones
  ecs_entity_t hand = create_zone(world, player0, ZHand, "Hand_P0");
  ecs_entity_t deck = create_zone(world, player0, ZDeck, "Deck_P0");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].hand = hand;
  gs->zones[0].deck = deck;
  gs->phase = PHASE_MAIN;
  gs->winner = -1;
  ecs_singleton_modified(world, GameState);

  // Create 5 cards in deck
  for (int i = 0; i < 5; i++) {
    ecs_entity_t deck_card = ecs_new(world);
    ecs_add_pair(world, deck_card, EcsChildOf, deck);
    ecs_set(world, deck_card, Element, { .element = (uint8_t)(i + 1) });
  }

  // Test: Draw 3 cards when deck has 5 = success (deck has 2 remaining)
  ecs_entity_t drawn[3];
  bool success = draw_cards_with_deckout_check(world, player0, 3, drawn);

  // Should return true (no deck-out)
  assert(success);

  // Verify no deck-out occurred
  gs = ecs_singleton_get_mut(world, GameState);
  assert(gs->winner == -1);  // No winner yet
  assert(gs->phase == PHASE_MAIN);

  // Verify cards were drawn
  ecs_entities_t final_hand = ecs_get_ordered_children(world, hand);
  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  assert(final_hand.count == 3);  // 3 cards drawn
  assert(final_deck.count == 2);  // 2 cards remaining

  ecs_fini(world);
}

static void test_stt02_014_effect_target_uses_zone_index(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create two players
  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, { .pid = 0 });
  ecs_set(world, player0, PlayerNumber, { .player_number = 0 });

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, { .pid = 1 });
  ecs_set(world, player1, PlayerNumber, { .player_number = 1 });

  // Create gardens
  ecs_entity_t garden0 = create_zone(world, player0, ZGarden, "Garden_P0");
  ecs_entity_t garden1 = create_zone(world, player1, ZGarden, "Garden_P1");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->winner = -1;
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].garden = garden0;
  gs->zones[1].garden = garden1;
  ecs_singleton_modified(world, GameState);

  azk_clear_ability_context(world);

  // Create spell card (STT02-014) owned by player0
  ecs_entity_t spell_card = ecs_new(world);
  ecs_set(world, spell_card, CardId, { .id = CARD_DEF_STT02_014, .code = "STT02-014" });
  ecs_set(world, spell_card, Type, { .value = CARD_TYPE_SPELL });
  ecs_add_pair(world, spell_card, Rel_OwnedBy, player0);

  // Create opponent garden entities with reversed child order:
  // First add zoneIndex 1, then zoneIndex 0.
  ecs_entity_t target_z1 = ecs_new(world);
  ecs_set(world, target_z1, CardId, { .id = CARD_DEF_STT02_003, .code = "TARGET_Z1" });
  ecs_set(world, target_z1, Type, { .value = CARD_TYPE_ENTITY });
  ecs_set(world, target_z1, IKZCost, { .ikz_cost = 1 });
  ecs_set(world, target_z1, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, target_z1, Rel_OwnedBy, player1);
  ecs_add_pair(world, target_z1, EcsChildOf, garden1);
  ecs_set(world, target_z1, ZoneIndex, { .index = 1 });

  ecs_entity_t target_z0 = ecs_new(world);
  ecs_set(world, target_z0, CardId, { .id = CARD_DEF_STT02_006, .code = "TARGET_Z0" });
  ecs_set(world, target_z0, Type, { .value = CARD_TYPE_ENTITY });
  ecs_set(world, target_z0, IKZCost, { .ikz_cost = 2 });
  ecs_set(world, target_z0, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, target_z0, Rel_OwnedBy, player1);
  ecs_add_pair(world, target_z0, EcsChildOf, garden1);
  ecs_set(world, target_z0, ZoneIndex, { .index = 0 });

  // Trigger the spell ability and select target index 1
  bool triggered = azk_trigger_spell_ability(world, spell_card, player0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  bool selected = azk_process_effect_selection(world, 1);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  // Verify that the zoneIndex 1 target was frozen, not zoneIndex 0
  assert(ecs_has(world, target_z1, Frozen));
  assert(!ecs_has(world, target_z0, Frozen));

  ecs_fini(world);
}

static void test_stt02_014_action_mask_uses_zone_index(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, {.pid = 1});
  ecs_set(world, player1, PlayerNumber, {.player_number = 1});

  ecs_entity_t garden0 = create_zone(world, player0, ZGarden, "Garden_P0");
  ecs_entity_t garden1 = create_zone(world, player1, ZGarden, "Garden_P1");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->winner = -1;
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].garden = garden0;
  gs->zones[1].garden = garden1;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t spell_card = ecs_new(world);
  ecs_set(world, spell_card, CardId,
          {.id = CARD_DEF_STT02_014, .code = "STT02-014"});
  ecs_set(world, spell_card, Type, {.value = CARD_TYPE_SPELL});
  ecs_add_pair(world, spell_card, Rel_OwnedBy, player0);

  ecs_entity_t target_z1 = ecs_new(world);
  ecs_set(world, target_z1, CardId,
          {.id = CARD_DEF_STT02_003, .code = "TARGET_Z1"});
  ecs_set(world, target_z1, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, target_z1, IKZCost, {.ikz_cost = 1});
  ecs_set(world, target_z1, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, target_z1, Rel_OwnedBy, player1);
  ecs_add_pair(world, target_z1, EcsChildOf, garden1);
  ecs_set(world, target_z1, ZoneIndex, {.index = 1});

  ecs_entity_t target_z0 = ecs_new(world);
  ecs_set(world, target_z0, CardId,
          {.id = CARD_DEF_STT02_006, .code = "TARGET_Z0"});
  ecs_set(world, target_z0, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, target_z0, IKZCost, {.ikz_cost = 3});
  ecs_set(world, target_z0, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, target_z0, Rel_OwnedBy, player1);
  ecs_add_pair(world, target_z0, EcsChildOf, garden1);
  ecs_set(world, target_z0, ZoneIndex, {.index = 0});

  bool triggered = azk_trigger_spell_ability(world, spell_card, player0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(world, gs, 0, &mask);
  assert(built);

  bool found_target_z1 = false;
  bool found_target_z0 = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }
    if (action->subaction_1 == 1) {
      found_target_z1 = true;
    }
    if (action->subaction_1 == 0) {
      found_target_z0 = true;
    }
  }

  assert(found_target_z1);
  assert(!found_target_z0);

  ecs_fini(world);
}

static void test_azk01_002_validate_rejects_dead_leader(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t leader0 = create_zone(world, player0, ZLeader, "Leader_P0");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->zones[0].leader = leader0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t leader_card = ecs_new(world);
  ecs_set(world, leader_card, CardId, {.id = CARD_DEF_STT01_001, .code = "STT01-001"});
  ecs_set(world, leader_card, Type, {.value = CARD_TYPE_LEADER});
  ecs_set(world, leader_card, BaseStats, {.attack = 0, .health = 20});
  ecs_set(world, leader_card, CurStats, {.cur_atk = 0, .cur_hp = 20});
  ecs_add_pair(world, leader_card, Rel_OwnedBy, player0);
  ecs_add_pair(world, leader_card, EcsChildOf, leader0);

  ecs_entity_t spell_card = ecs_new(world);
  ecs_set(world, spell_card, CardId,
          {.id = CARD_DEF_AZK01_002, .code = "AZK01-002"});
  ecs_set(world, spell_card, Type, {.value = CARD_TYPE_SPELL});
  ecs_add_pair(world, spell_card, Rel_OwnedBy, player0);

  const AbilityDef *def = azk_get_ability_def(CARD_DEF_AZK01_002);
  assert(def != NULL);
  assert(def->validate != NULL);
  assert(def->validate(world, spell_card, player0));

  ecs_set(world, leader_card, CurStats, {.cur_atk = 0, .cur_hp = 18});
  assert(def->validate(world, spell_card, player0));

  ecs_set(world, leader_card, CurStats, {.cur_atk = 0, .cur_hp = 0});
  assert(!def->validate(world, spell_card, player0));

  ecs_fini(world);
}

static void test_azk01_002_spell_heals_owner_leader(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t leader0 = create_zone(world, player0, ZLeader, "Leader_P0");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  gs->players[0] = player0;
  gs->zones[0].leader = leader0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t leader_card = ecs_new(world);
  ecs_set(world, leader_card, CardId, {.id = CARD_DEF_STT01_001, .code = "STT01-001"});
  ecs_set(world, leader_card, Type, {.value = CARD_TYPE_LEADER});
  ecs_set(world, leader_card, BaseStats, {.attack = 0, .health = 20});
  ecs_set(world, leader_card, CurStats, {.cur_atk = 0, .cur_hp = 17});
  ecs_add_pair(world, leader_card, Rel_OwnedBy, player0);
  ecs_add_pair(world, leader_card, EcsChildOf, leader0);

  ecs_entity_t spell_card = ecs_new(world);
  ecs_set(world, spell_card, CardId,
          {.id = CARD_DEF_AZK01_002, .code = "AZK01-002"});
  ecs_set(world, spell_card, Type, {.value = CARD_TYPE_SPELL});
  ecs_add_pair(world, spell_card, Rel_OwnedBy, player0);

  bool entered_selection = azk_trigger_spell_ability(world, spell_card, player0);
  assert(!entered_selection);
  const CurStats *leader_stats = ecs_get(world, leader_card, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 19);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  ecs_set(world, leader_card, CurStats, {.cur_atk = 0, .cur_hp = 19});
  entered_selection = azk_trigger_spell_ability(world, spell_card, player0);
  assert(!entered_selection);
  leader_stats = ecs_get(world, leader_card, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 20);

  entered_selection = azk_trigger_spell_ability(world, spell_card, player0);
  assert(!entered_selection);
  leader_stats = ecs_get(world, leader_card, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 20);

  ecs_fini(world);
}

static void test_triggered_ability_confirmation_restores_active_player(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t players[MAX_PLAYERS_PER_MATCH] = {0};
  PlayerZones zones[MAX_PLAYERS_PER_MATCH] = {0};
  ecs_entity_t selis_cards[MAX_PLAYERS_PER_MATCH] = {0};
  setup_triggered_ability_control_fixture(world, players, zones, selis_cards);

  bool queued = azk_queue_triggered_effect(
      world, selis_cards[0], players[0], TIMING_TAG_WHEN_RETURNED_TO_HAND);
  assert(queued);
  queued = azk_queue_triggered_effect(
      world, selis_cards[1], players[1], TIMING_TAG_WHEN_RETURNED_TO_HAND);
  assert(queued);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs->active_player_index == 0);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);
  assert(gs->active_player_index == 0);

  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(!azk_is_in_ability_phase(world));
  assert(gs->active_player_index == 0);
  assert(ecs_get_ordered_children(world, zones[0].hand).count == 1);

  processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);
  assert(gs->active_player_index == 1);

  confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(!azk_is_in_ability_phase(world));
  assert(gs->active_player_index == 0);
  assert(ecs_get_ordered_children(world, zones[1].hand).count == 1);

  ecs_fini(world);
}

static void test_triggered_ability_decline_restores_active_player(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t players[MAX_PLAYERS_PER_MATCH] = {0};
  PlayerZones zones[MAX_PLAYERS_PER_MATCH] = {0};
  ecs_entity_t selis_cards[MAX_PLAYERS_PER_MATCH] = {0};
  setup_triggered_ability_control_fixture(world, players, zones, selis_cards);

  bool queued = azk_queue_triggered_effect(
      world, selis_cards[0], players[0], TIMING_TAG_WHEN_RETURNED_TO_HAND);
  assert(queued);
  queued = azk_queue_triggered_effect(
      world, selis_cards[1], players[1], TIMING_TAG_WHEN_RETURNED_TO_HAND);
  assert(queued);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs->active_player_index == 0);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(!azk_is_in_ability_phase(world));
  assert(gs->active_player_index == 0);
  assert(ecs_get_ordered_children(world, zones[0].hand).count == 1);

  processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);
  assert(gs->active_player_index == 1);

  bool declined = azk_process_ability_decline(world);
  assert(declined);
  assert(!azk_is_in_ability_phase(world));
  assert(gs->active_player_index == 0);
  assert(ecs_get_ordered_children(world, zones[1].hand).count == 0);

  ecs_fini(world);
}

static void test_triggered_mandatory_target_selection_skips_confirmation(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, {.pid = 1});
  ecs_set(world, player1, PlayerNumber, {.player_number = 1});

  ecs_entity_t garden1 = create_zone(world, player1, ZGarden, "Garden_P1");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[1].garden = garden1;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t target = ecs_new(world);
  ecs_add_pair(world, target, Rel_OwnedBy, player1);
  ecs_add_pair(world, target, EcsChildOf, garden1);
  ecs_set(world, target, ZoneIndex, {.index = 0});
  ecs_set(world, target, CurStats, {.cur_atk = 1, .cur_hp = 2});

  ecs_entity_t stt01_006_card = ecs_new(world);
  ecs_set(world, stt01_006_card, CardId,
          {.id = CARD_DEF_STT01_006, .code = "STT01-006"});

  bool queued = azk_queue_triggered_effect(world, stt01_006_card, player0,
                                           TIMING_TAG_WHEN_ATTACKING);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  bool selected = azk_process_effect_selection(world, 0);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *target_stats = ecs_get(world, target, CurStats);
  assert(target_stats != NULL);
  assert(target_stats->cur_hp == 1);

  ecs_fini(world);
}

static void test_azk01_004_when_attacking_buff_expires_end_of_turn(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t garden0 = create_zone(world, player0, ZGarden, "Garden_P0");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->zones[0].garden = garden0;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t alley_thug = ecs_new(world);
  ecs_set(world, alley_thug, CardId,
          {.id = CARD_DEF_AZK01_004, .code = "AZK01-004"});
  ecs_set(world, alley_thug, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, alley_thug, BaseStats, {.attack = 1, .health = 1});
  ecs_set(world, alley_thug, CurStats, {.cur_atk = 1, .cur_hp = 1});
  ecs_add_pair(world, alley_thug, Rel_OwnedBy, player0);
  ecs_add_pair(world, alley_thug, EcsChildOf, garden0);
  ecs_set(world, alley_thug, ZoneIndex, {.index = 0});

  bool queued = azk_queue_triggered_effect(world, alley_thug, player0,
                                           TIMING_TAG_WHEN_ATTACKING);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  // Immediate no-target triggered abilities resolve synchronously and do not
  // leave the engine in an active ability phase.
  assert(!processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *buffed_stats = ecs_get(world, alley_thug, CurStats);
  assert(buffed_stats != NULL);
  assert(buffed_stats->cur_atk == 2);
  assert(ecs_has_pair(world, alley_thug, ecs_id(AttackBuff), alley_thug));

  expire_eot_attack_modifiers_in_zone(world, garden0);

  const CurStats *reset_stats = ecs_get(world, alley_thug, CurStats);
  assert(reset_stats != NULL);
  assert(reset_stats->cur_atk == 1);
  assert(!ecs_has_pair(world, alley_thug, ecs_id(AttackBuff), alley_thug));

  ecs_fini(world);
}

static void test_triggered_mandatory_up_to_effect_skips_confirmation(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, {.pid = 1});
  ecs_set(world, player1, PlayerNumber, {.player_number = 1});

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t stt01_014_card = ecs_new(world);
  ecs_set(world, stt01_014_card, CardId,
          {.id = CARD_DEF_STT01_014, .code = "STT01-014"});

  bool queued = azk_trigger_on_play_ability(world, stt01_014_card, player0);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  bool skipped = azk_process_effect_skip(world);
  assert(skipped);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  ecs_fini(world);
}

static void test_triggered_selection_pick_skips_confirmation_stt02_003(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t selection = create_zone(world, player, ZSelection, "Selection_P0");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].selection = selection;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  for (int i = 0; i < 5; i++) {
    ecs_entity_t deck_card = ecs_new(world);
    ecs_add_pair(world, deck_card, EcsChildOf, deck);
    ecs_add_pair(world, deck_card, Rel_OwnedBy, player);
    ecs_set(world, deck_card, Type,
            {.value = (i == 4) ? CARD_TYPE_SPELL : CARD_TYPE_ENTITY});
    if (i == 4) {
      ecs_add_id(world, deck_card, ecs_id(TSubtype_Watercrafting));
    }
  }

  ecs_entity_t stt02_003_card = ecs_new(world);
  ecs_set(world, stt02_003_card, CardId,
          {.id = CARD_DEF_STT02_003, .code = "STT02-003"});

  bool queued = azk_trigger_on_play_ability(world, stt02_003_card, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  bool skipped = azk_process_skip_selection(world);
  assert(skipped);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_BOTTOM_DECK);

  bool bottom_decked = azk_process_bottom_deck_all(world);
  assert(bottom_decked);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  ecs_fini(world);
}

static void test_triggered_selection_pick_skips_confirmation_stt02_013(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t selection = create_zone(world, player, ZSelection, "Selection_P0");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].selection = selection;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  for (int i = 0; i < 3; i++) {
    ecs_entity_t deck_card = ecs_new(world);
    ecs_add_pair(world, deck_card, EcsChildOf, deck);
    ecs_add_pair(world, deck_card, Rel_OwnedBy, player);
    ecs_set(world, deck_card, Type, {.value = CARD_TYPE_ENTITY});
    ecs_set(world, deck_card, Element, {.element = CARD_ELEMENT_WATER});
    ecs_set(world, deck_card, IKZCost, {.ikz_cost = (uint8_t)(i == 2 ? 2 : 4)});
  }

  ecs_entity_t stt02_013_card = ecs_new(world);
  ecs_set(world, stt02_013_card, CardId,
          {.id = CARD_DEF_STT02_013, .code = "STT02-013"});

  bool queued = azk_trigger_on_play_ability(world, stt02_013_card, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  bool skipped = azk_process_skip_selection(world);
  assert(skipped);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_BOTTOM_DECK);

  bool bottom_decked = azk_process_bottom_deck_all(world);
  assert(bottom_decked);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  ecs_fini(world);
}

static void test_azk01_003_ability_flow_excludes_self_and_adds_black_jade_card(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  ecs_entity_t deck = create_zone(world, player, ZDeck, "Deck_P0");
  ecs_entity_t hand = create_zone(world, player, ZHand, "Hand_P0");
  ecs_entity_t selection = create_zone(world, player, ZSelection, "Selection_P0");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->zones[0].deck = deck;
  gs->zones[0].hand = hand;
  gs->zones[0].selection = selection;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t filler0 = ecs_new(world);
  ecs_add_pair(world, filler0, EcsChildOf, deck);
  ecs_add_pair(world, filler0, Rel_OwnedBy, player);
  ecs_set(world, filler0, CardId, {.id = CARD_DEF_STT02_004, .code = "FILLER0"});
  ecs_set(world, filler0, Type, {.value = CARD_TYPE_ENTITY});

  ecs_entity_t filler1 = ecs_new(world);
  ecs_add_pair(world, filler1, EcsChildOf, deck);
  ecs_add_pair(world, filler1, Rel_OwnedBy, player);
  ecs_set(world, filler1, CardId, {.id = CARD_DEF_STT02_007, .code = "FILLER1"});
  ecs_set(world, filler1, Type, {.value = CARD_TYPE_ENTITY});

  ecs_entity_t courier_in_deck = ecs_new(world);
  ecs_add_pair(world, courier_in_deck, EcsChildOf, deck);
  ecs_add_pair(world, courier_in_deck, Rel_OwnedBy, player);
  ecs_set(world, courier_in_deck, CardId,
          {.id = CARD_DEF_AZK01_003, .code = "AZK01-003"});
  ecs_set(world, courier_in_deck, Type, {.value = CARD_TYPE_ENTITY});
  ecs_add_id(world, courier_in_deck, ecs_id(TSubtype_BlackJade));
  ecs_add_id(world, courier_in_deck, ecs_id(TSubtype_Strider));

  ecs_entity_t filler2 = ecs_new(world);
  ecs_add_pair(world, filler2, EcsChildOf, deck);
  ecs_add_pair(world, filler2, Rel_OwnedBy, player);
  ecs_set(world, filler2, CardId, {.id = CARD_DEF_STT01_005, .code = "FILLER2"});
  ecs_set(world, filler2, Type, {.value = CARD_TYPE_ENTITY});

  ecs_entity_t black_jade_target = ecs_new(world);
  ecs_add_pair(world, black_jade_target, EcsChildOf, deck);
  ecs_add_pair(world, black_jade_target, Rel_OwnedBy, player);
  ecs_set(world, black_jade_target, CardId,
          {.id = CARD_DEF_STT01_004, .code = "STT01-004"});
  ecs_set(world, black_jade_target, Type, {.value = CARD_TYPE_ENTITY});
  ecs_add_id(world, black_jade_target, ecs_id(TSubtype_BlackJade));
  ecs_add_id(world, black_jade_target, ecs_id(TSubtype_Dawnling));

  ecs_entity_t courier = ecs_new(world);
  ecs_set(world, courier, CardId, {.id = CARD_DEF_AZK01_003, .code = "AZK01-003"});

  bool queued = azk_trigger_on_play_ability(world, courier, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  bool rejected = azk_process_selection_pick(world, 2);
  assert(!rejected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  bool selected = azk_process_selection_pick(world, 0);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_BOTTOM_DECK);

  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
  assert(hand_cards.count == 1);
  assert(hand_cards.ids[0] == black_jade_target);

  bool bottom_decked = azk_process_bottom_deck_all(world);
  assert(bottom_decked);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  ecs_entities_t final_deck = ecs_get_ordered_children(world, deck);
  ecs_entities_t final_selection = ecs_get_ordered_children(world, selection);
  assert(final_deck.count == 4);
  assert(final_selection.count == 0);

  ecs_fini(world);
}

static void test_leader_response_enters_effect_selection(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, {.pid = 1});
  ecs_set(world, player1, PlayerNumber, {.player_number = 1});

  ecs_entity_t leader0 = create_zone(world, player0, ZLeader, "Leader_P0");
  ecs_entity_t garden1 = create_zone(world, player1, ZGarden, "Garden_P1");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].leader = leader0;
  gs->zones[1].garden = garden1;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t shao = ecs_new(world);
  ecs_set(world, shao, CardId, {.id = CARD_DEF_STT02_001, .code = "STT02-001"});
  ecs_set(world, shao, Type, {.value = CARD_TYPE_LEADER});
  ecs_add_pair(world, shao, Rel_OwnedBy, player0);
  ecs_add_pair(world, shao, EcsChildOf, leader0);

  ecs_entity_t target = ecs_new(world);
  ecs_set(world, target, CardId, {.id = CARD_DEF_STT02_003, .code = "TARGET"});
  ecs_set(world, target, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, target, IKZCost, {.ikz_cost = 2});
  ecs_add_pair(world, target, Rel_OwnedBy, player1);
  ecs_add_pair(world, target, EcsChildOf, garden1);
  ecs_set(world, target, ZoneIndex, {.index = 0});

  bool triggered = azk_trigger_leader_response_ability(world, shao, player0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  assert(ctx != NULL);
  assert(ctx->runtime.source_card == shao);
  assert(ctx->runtime.owner == player0);
  assert(ctx->effect.max_allowed == 1);

  ecs_fini(world);
}

static void test_gate_portal_enters_selection_flow_stt01_002(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t discard = create_zone(world, player0, ZDiscard, "Discard_P0");
  ecs_entity_t selection =
      create_zone(world, player0, ZSelection, "Selection_P0");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->zones[0].discard = discard;
  gs->zones[0].selection = selection;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT01_002, .code = "STT01-002"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player0);

  ecs_entity_t portaled_card = ecs_new(world);
  ecs_set(world, portaled_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, portaled_card, GatePoints, {.gate_points = 2});

  ecs_entity_t weapon = ecs_new(world);
  ecs_set(world, weapon, CardId, {.id = CARD_DEF_STT01_016, .code = "WEAPON"});
  ecs_set(world, weapon, Type, {.value = CARD_TYPE_WEAPON});
  ecs_set(world, weapon, IKZCost, {.ikz_cost = 1});
  ecs_add_pair(world, weapon, Rel_OwnedBy, player0);
  ecs_add_pair(world, weapon, EcsChildOf, discard);

  azk_trigger_gate_portal_ability(world, gate_card, portaled_card, player0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  assert(ctx != NULL);
  assert(ctx->runtime.source_card == gate_card);
  assert(ctx->runtime.owner == player0);
  assert(ctx->scratch.kind == ABILITY_SCRATCH_DISCARD_SELECTION);
  assert(ctx->scratch.data.discard_selection.max_cost == 2);
  assert(ctx->selection.count == 1);
  assert(ctx->selection.cards[0] == weapon);
  assert(ecs_get_target(world, weapon, EcsChildOf, 0) == selection);

  ecs_fini(world);
}

static void test_start_phase_skips_opening_draw_for_starting_player(void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);

  int initial_hand_p0 = ecs_get_ordered_children(world, gs->zones[0].hand).count;
  int initial_hand_p1 = ecs_get_ordered_children(world, gs->zones[1].hand).count;

  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = 0;
  gs->turn_number = 0;
  ecs_singleton_modified(world, GameState);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  gs = ecs_singleton_get_mut(world, GameState);
  assert(gs->turn_number == 1);
  assert(gs->phase == PHASE_MAIN);
  assert(ecs_get_ordered_children(world, gs->zones[0].hand).count == initial_hand_p0);
  assert(ecs_get_ordered_children(world, gs->zones[1].hand).count == initial_hand_p1);

  int hand_before_second_player = ecs_get_ordered_children(world, gs->zones[1].hand).count;
  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = 1;
  ecs_singleton_modified(world, GameState);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  gs = ecs_singleton_get_mut(world, GameState);
  assert(gs->turn_number == 2);
  assert(gs->phase == PHASE_MAIN);
  assert(ecs_get_ordered_children(world, gs->zones[1].hand).count ==
         hand_before_second_player + 1);

  int hand_before_player0_second_turn =
      ecs_get_ordered_children(world, gs->zones[0].hand).count;
  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  gs = ecs_singleton_get_mut(world, GameState);
  assert(gs->turn_number == 3);
  assert(gs->phase == PHASE_MAIN);
  assert(ecs_get_ordered_children(world, gs->zones[0].hand).count ==
         hand_before_player0_second_turn + 1);

  azk_world_fini(world);
}

static void test_observation_garden_slots_use_zone_index(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  // Initialize singletons
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  // Create two players
  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, { .pid = 0 });
  ecs_set(world, player0, PlayerNumber, { .player_number = 0 });

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, { .pid = 1 });
  ecs_set(world, player1, PlayerNumber, { .player_number = 1 });

  // Create all zones for both players
  PlayerZones zones0 = {0};
  zones0.deck = create_zone(world, player0, ZDeck, "Deck_P0");
  zones0.hand = create_zone(world, player0, ZHand, "Hand_P0");
  zones0.leader = create_zone(world, player0, ZLeader, "Leader_P0");
  zones0.gate = create_zone(world, player0, ZGate, "Gate_P0");
  zones0.garden = create_zone(world, player0, ZGarden, "Garden_P0");
  zones0.alley = create_zone(world, player0, ZAlley, "Alley_P0");
  zones0.ikz_pile = create_zone(world, player0, ZIKZPileTag, "IKZPile_P0");
  zones0.ikz_area = create_zone(world, player0, ZIKZAreaTag, "IKZArea_P0");
  zones0.discard = create_zone(world, player0, ZDiscard, "Discard_P0");
  zones0.selection = create_zone(world, player0, ZSelection, "Selection_P0");

  PlayerZones zones1 = {0};
  zones1.deck = create_zone(world, player1, ZDeck, "Deck_P1");
  zones1.hand = create_zone(world, player1, ZHand, "Hand_P1");
  zones1.leader = create_zone(world, player1, ZLeader, "Leader_P1");
  zones1.gate = create_zone(world, player1, ZGate, "Gate_P1");
  zones1.garden = create_zone(world, player1, ZGarden, "Garden_P1");
  zones1.alley = create_zone(world, player1, ZAlley, "Alley_P1");
  zones1.ikz_pile = create_zone(world, player1, ZIKZPileTag, "IKZPile_P1");
  zones1.ikz_area = create_zone(world, player1, ZIKZAreaTag, "IKZArea_P1");
  zones1.discard = create_zone(world, player1, ZDiscard, "Discard_P1");
  zones1.selection = create_zone(world, player1, ZSelection, "Selection_P1");

  // Set up GameState
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->winner = -1;
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0] = zones0;
  gs->zones[1] = zones1;
  ecs_singleton_modified(world, GameState);

  // Create leader and gate cards for both players
  ecs_entity_t leader0 = ecs_new(world);
  ecs_set(world, leader0, CardId, { .id = CARD_DEF_STT02_001, .code = "Leader_P0" });
  ecs_set(world, leader0, Type, { .value = CARD_TYPE_LEADER });
  ecs_set(world, leader0, CurStats, { .cur_atk = 0, .cur_hp = 20 });
  ecs_set(world, leader0, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, leader0, Rel_OwnedBy, player0);
  ecs_add_pair(world, leader0, EcsChildOf, zones0.leader);

  ecs_entity_t gate0 = ecs_new(world);
  ecs_set(world, gate0, CardId, { .id = CARD_DEF_STT02_002, .code = "Gate_P0" });
  ecs_set(world, gate0, Type, { .value = CARD_TYPE_GATE });
  ecs_set(world, gate0, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, gate0, Rel_OwnedBy, player0);
  ecs_add_pair(world, gate0, EcsChildOf, zones0.gate);

  ecs_entity_t leader1 = ecs_new(world);
  ecs_set(world, leader1, CardId, { .id = CARD_DEF_STT01_001, .code = "Leader_P1" });
  ecs_set(world, leader1, Type, { .value = CARD_TYPE_LEADER });
  ecs_set(world, leader1, CurStats, { .cur_atk = 0, .cur_hp = 20 });
  ecs_set(world, leader1, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, leader1, Rel_OwnedBy, player1);
  ecs_add_pair(world, leader1, EcsChildOf, zones1.leader);

  ecs_entity_t gate1 = ecs_new(world);
  ecs_set(world, gate1, CardId, { .id = CARD_DEF_STT01_002, .code = "Gate_P1" });
  ecs_set(world, gate1, Type, { .value = CARD_TYPE_GATE });
  ecs_set(world, gate1, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, gate1, Rel_OwnedBy, player1);
  ecs_add_pair(world, gate1, EcsChildOf, zones1.gate);

  // Create two garden cards for player0 in out-of-order insertion
  ecs_entity_t garden_card_a = ecs_new(world);
  ecs_set(world, garden_card_a, CardId, { .id = CARD_DEF_STT02_003, .code = "GARDEN_A" });
  ecs_set(world, garden_card_a, Type, { .value = CARD_TYPE_ENTITY });
  ecs_set(world, garden_card_a, IKZCost, { .ikz_cost = 1 });
  ecs_set(world, garden_card_a, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, garden_card_a, Rel_OwnedBy, player0);
  ecs_add_pair(world, garden_card_a, EcsChildOf, zones0.garden);
  ecs_set(world, garden_card_a, ZoneIndex, { .index = 3 });

  ecs_entity_t garden_card_b = ecs_new(world);
  ecs_set(world, garden_card_b, CardId, { .id = CARD_DEF_STT02_006, .code = "GARDEN_B" });
  ecs_set(world, garden_card_b, Type, { .value = CARD_TYPE_ENTITY });
  ecs_set(world, garden_card_b, IKZCost, { .ikz_cost = 2 });
  ecs_set(world, garden_card_b, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, garden_card_b, Rel_OwnedBy, player0);
  ecs_add_pair(world, garden_card_b, EcsChildOf, zones0.garden);
  ecs_set(world, garden_card_b, ZoneIndex, { .index = 1 });

  ObservationData obs = create_observation_data(world, 0);

  assert(obs.my_observation_data.garden[0].id.code == NULL);
  assert(obs.my_observation_data.garden[2].id.code == NULL);
  assert(obs.my_observation_data.garden[4].id.code == NULL);
  assert(obs.my_observation_data.garden[1].id.code != NULL);
  assert(strcmp(obs.my_observation_data.garden[1].id.code, "GARDEN_B") == 0);
  assert(obs.my_observation_data.garden[3].id.code != NULL);
  assert(strcmp(obs.my_observation_data.garden[3].id.code, "GARDEN_A") == 0);

  ecs_fini(world);
}

static void
test_leader_with_multiple_stt01_013_weapons_supports_attack_mask_and_combat(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player0 = ecs_new(world);
  ecs_set(world, player0, PlayerId, {.pid = 0});
  ecs_set(world, player0, PlayerNumber, {.player_number = 0});

  ecs_entity_t player1 = ecs_new(world);
  ecs_set(world, player1, PlayerId, {.pid = 1});
  ecs_set(world, player1, PlayerNumber, {.player_number = 1});

  PlayerZones zones0 = {0};
  zones0.deck = create_zone(world, player0, ZDeck, "Deck_P0");
  zones0.hand = create_zone(world, player0, ZHand, "Hand_P0");
  zones0.leader = create_zone(world, player0, ZLeader, "Leader_P0");
  zones0.gate = create_zone(world, player0, ZGate, "Gate_P0");
  zones0.garden = create_zone(world, player0, ZGarden, "Garden_P0");
  zones0.alley = create_zone(world, player0, ZAlley, "Alley_P0");
  zones0.ikz_pile = create_zone(world, player0, ZIKZPileTag, "IKZPile_P0");
  zones0.ikz_area = create_zone(world, player0, ZIKZAreaTag, "IKZArea_P0");
  zones0.discard = create_zone(world, player0, ZDiscard, "Discard_P0");
  zones0.selection = create_zone(world, player0, ZSelection, "Selection_P0");

  PlayerZones zones1 = {0};
  zones1.deck = create_zone(world, player1, ZDeck, "Deck_P1");
  zones1.hand = create_zone(world, player1, ZHand, "Hand_P1");
  zones1.leader = create_zone(world, player1, ZLeader, "Leader_P1");
  zones1.gate = create_zone(world, player1, ZGate, "Gate_P1");
  zones1.garden = create_zone(world, player1, ZGarden, "Garden_P1");
  zones1.alley = create_zone(world, player1, ZAlley, "Alley_P1");
  zones1.ikz_pile = create_zone(world, player1, ZIKZPileTag, "IKZPile_P1");
  zones1.ikz_area = create_zone(world, player1, ZIKZAreaTag, "IKZArea_P1");
  zones1.discard = create_zone(world, player1, ZDiscard, "Discard_P1");
  zones1.selection = create_zone(world, player1, ZSelection, "Selection_P1");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->winner = -1;
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0] = zones0;
  gs->zones[1] = zones1;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t leader0 = ecs_new(world);
  ecs_add(world, leader0, TLeader);
  ecs_set(world, leader0, CardId, {.id = CARD_DEF_STT01_001, .code = "STT01-001"});
  ecs_set(world, leader0, Type, {.value = CARD_TYPE_LEADER});
  ecs_set(world, leader0, BaseStats, {.attack = 0, .health = 20});
  ecs_set(world, leader0, CurStats, {.cur_atk = 4, .cur_hp = 17});
  ecs_set(world, leader0, TapState, {.tapped = false, .cooldown = false});
  ecs_set(world, leader0, DamageTracker, {0});
  ecs_add_pair(world, leader0, Rel_OwnedBy, player0);
  ecs_add_pair(world, leader0, EcsChildOf, zones0.leader);

  ecs_entity_t gate0 = ecs_new(world);
  ecs_add(world, gate0, TGate);
  ecs_set(world, gate0, CardId, {.id = CARD_DEF_STT01_002, .code = "STT01-002"});
  ecs_set(world, gate0, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate0, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate0, Rel_OwnedBy, player0);
  ecs_add_pair(world, gate0, EcsChildOf, zones0.gate);

  ecs_entity_t leader1 = ecs_new(world);
  ecs_add(world, leader1, TLeader);
  ecs_set(world, leader1, CardId, {.id = CARD_DEF_STT02_001, .code = "STT02-001"});
  ecs_set(world, leader1, Type, {.value = CARD_TYPE_LEADER});
  ecs_set(world, leader1, BaseStats, {.attack = 0, .health = 20});
  ecs_set(world, leader1, CurStats, {.cur_atk = 0, .cur_hp = 20});
  ecs_set(world, leader1, TapState, {.tapped = false, .cooldown = false});
  ecs_set(world, leader1, DamageTracker, {0});
  ecs_add_pair(world, leader1, Rel_OwnedBy, player1);
  ecs_add_pair(world, leader1, EcsChildOf, zones1.leader);

  ecs_entity_t gate1 = ecs_new(world);
  ecs_add(world, gate1, TGate);
  ecs_set(world, gate1, CardId, {.id = CARD_DEF_STT02_002, .code = "STT02-002"});
  ecs_set(world, gate1, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate1, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate1, Rel_OwnedBy, player1);
  ecs_add_pair(world, gate1, EcsChildOf, zones1.gate);

  for (int i = 0; i < 2; ++i) {
    ecs_entity_t weapon = ecs_new(world);
    ecs_add(world, weapon, TWeapon);
    ecs_set(world, weapon, CardId,
            {.id = CARD_DEF_STT01_013, .code = "STT01-013"});
    ecs_set(world, weapon, Type, {.value = CARD_TYPE_WEAPON});
    ecs_set(world, weapon, BaseStats, {.attack = 1, .health = 0});
    ecs_set(world, weapon, CurStats, {.cur_atk = 2, .cur_hp = 0});
    ecs_set(world, weapon, IKZCost, {.ikz_cost = 1});
    ecs_set(world, weapon, DamageTracker, {0});
    ecs_add_pair(world, weapon, Rel_OwnedBy, player0);
    ecs_add_pair(world, weapon, EcsChildOf, leader0);
  }

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(world, gs, 0, &mask);
  assert(built);

  bool found_leader_attack = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_ATTACK && action->subaction_1 == GARDEN_SIZE &&
        action->subaction_2 == GARDEN_SIZE) {
      found_leader_attack = true;
      break;
    }
  }
  assert(found_leader_attack);

  ObservationData before_combat = create_observation_data(world, 0);
  assert(before_combat.my_observation_data.leader.weapon_count == 2);

  AttackIntent intent = {
      .attacking_player = player0,
      .defending_player = player1,
      .attacking_card = leader0,
      .defending_card = leader1,
      .attacker_index = GARDEN_SIZE,
      .defender_index = GARDEN_SIZE,
      .attacker_is_leader = true,
  };

  int attack_result = attack(world, &intent);
  assert(attack_result == 0);
  resolve_combat(world);

  const CurStats *leader0_stats = ecs_get(world, leader0, CurStats);
  const CurStats *leader1_stats = ecs_get(world, leader1, CurStats);
  assert(leader0_stats != NULL);
  assert(leader1_stats != NULL);
  assert(leader0_stats->cur_hp == 17);
  assert(leader1_stats->cur_hp == 16);

  ObservationData after_combat = create_observation_data(world, 0);
  assert(after_combat.my_observation_data.leader.weapon_count == 2);
  assert(after_combat.opponent_observation_data.leader.cur_stats.cur_hp == 16);

  ecs_fini(world);
}

// ============================================================================
// Game Log Tests
// ============================================================================

static void test_game_log_singleton_initialized(void) {
  ecs_world_t *world = azk_world_init(42);

  // GameStateLogContext singleton should exist and be initialized
  const GameStateLogContext *ctx = ecs_singleton_get(world, GameStateLogContext);
  assert(ctx != NULL);
  printf("  Initial log count: %d\n", ctx->count);

  // Add a test log
  azk_log_turn_started(world, 0, 1);

  // Verify log was added
  uint8_t count = azk_get_game_log_count(world);
  printf("  After turn_started log count: %d\n", count);
  assert(count >= 1);  // At least one log (may have others from init)

  // Get logs and verify
  const GameStateLog *logs = azk_get_game_logs(world, &count);
  assert(logs != NULL);
  printf("  Logs pointer: %p, count: %d\n", (void *)logs, count);

  // Find our turn started log
  bool found_turn_started = false;
  for (uint8_t i = 0; i < count; i++) {
    printf("  Log %d: type=%d\n", i, logs[i].type);
    if (logs[i].type == GLOG_TURN_STARTED) {
      found_turn_started = true;
      assert(logs[i].data.turn_started.player == 0);
      assert(logs[i].data.turn_started.turn_number == 1);
    }
  }
  assert(found_turn_started);

  // Test clear
  azk_clear_game_logs(world);
  count = azk_get_game_log_count(world);
  printf("  After clear log count: %d\n", count);
  assert(count == 0);

  azk_world_fini(world);
}

static void test_return_card_to_hand_uses_pending_hand_logs(void) {
  ecs_world_t *world = azk_world_init(42);
  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);

  ecs_entity_t deck = gs->zones[0].deck;
  ecs_entity_t hand = gs->zones[0].hand;
  ecs_entity_t garden = gs->zones[0].garden;

  ecs_entity_t cards[2] = {0};
  int card_count =
      collect_cards_of_type_in_zone(world, deck, CARD_TYPE_ENTITY, cards, 2);
  assert(card_count == 2);

  ecs_add_pair(world, cards[0], EcsChildOf, garden);
  ecs_set(world, cards[0], ZoneIndex, {.index = 0});
  ecs_add_pair(world, cards[1], EcsChildOf, garden);
  ecs_set(world, cards[1], ZoneIndex, {.index = 1});

  int32_t initial_hand_count = ecs_get_ordered_children(world, hand).count;

  azk_clear_game_logs(world);

  ecs_defer_begin(world);
  return_card_to_hand(world, cards[0]);
  return_card_to_hand(world, cards[1]);
  ecs_defer_end(world);
  azk_finalize_pending_zone_move_logs(world);

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  int matched_logs = 0;
  for (uint8_t i = 0; i < log_count; i++) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED) {
      continue;
    }

    const GameLogZoneMoved *zone_moved = &logs[i].data.zone_moved;
    if (zone_moved->card.player != 0 ||
        zone_moved->from_zone != GLOG_ZONE_GARDEN ||
        zone_moved->to_zone != GLOG_ZONE_HAND) {
      continue;
    }

    assert(matched_logs < 2);
    assert(zone_moved->from_index == matched_logs);
    assert(zone_moved->to_index == initial_hand_count + matched_logs);
    matched_logs++;
  }

  assert(matched_logs == 2);
  assert(ecs_get_ordered_children(world, hand).count == initial_hand_count + 2);

  azk_world_fini(world);
}

static void test_move_selection_to_hand_uses_pending_hand_logs(void) {
  ecs_world_t *world = azk_world_init(99);
  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);

  ecs_entity_t deck = gs->zones[0].deck;
  ecs_entity_t hand = gs->zones[0].hand;
  ecs_entity_t selection = gs->zones[0].selection;

  ecs_entity_t cards[2] = {0};
  int card_count =
      collect_cards_of_type_in_zone(world, deck, CARD_TYPE_ENTITY, cards, 2);
  assert(card_count == 2);

  ecs_add_pair(world, cards[0], EcsChildOf, selection);
  ecs_add_pair(world, cards[1], EcsChildOf, selection);

  int32_t initial_hand_count = ecs_get_ordered_children(world, hand).count;

  azk_clear_game_logs(world);

  ecs_defer_begin(world);
  move_selection_to_hand(world, cards[0]);
  move_selection_to_hand(world, cards[1]);
  ecs_defer_end(world);
  azk_finalize_pending_zone_move_logs(world);

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  int matched_logs = 0;
  for (uint8_t i = 0; i < log_count; i++) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED) {
      continue;
    }

    const GameLogZoneMoved *zone_moved = &logs[i].data.zone_moved;
    if (zone_moved->card.player != 0 ||
        zone_moved->from_zone != GLOG_ZONE_SELECTION ||
        zone_moved->to_zone != GLOG_ZONE_HAND) {
      continue;
    }

    assert(matched_logs < 2);
    assert(zone_moved->from_index == matched_logs);
    assert(zone_moved->to_index == initial_hand_count + matched_logs);
    matched_logs++;
  }

  assert(matched_logs == 2);
  assert(ecs_get_ordered_children(world, hand).count == initial_hand_count + 2);

  azk_world_fini(world);
}

static void test_deck_to_selection_to_hand_finalizes_each_committed_step(void) {
  ecs_world_t *world = azk_world_init(7);
  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);

  ecs_entity_t player0 = gs->players[0];
  ecs_entity_t selection = gs->zones[0].selection;
  ecs_entity_t hand = gs->zones[0].hand;
  ecs_entity_t cards[2] = {0};

  int32_t initial_hand_count = ecs_get_ordered_children(world, hand).count;

  azk_clear_game_logs(world);

  ecs_defer_begin(world);
  int moved = look_at_top_n_cards(world, player0, 2, cards);
  ecs_defer_end(world);
  azk_finalize_pending_zone_move_logs(world);

  assert(moved == 2);

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  int selection_logs = 0;
  for (uint8_t i = 0; i < log_count; i++) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED) {
      continue;
    }

    const GameLogZoneMoved *zone_moved = &logs[i].data.zone_moved;
    if (zone_moved->card.player != 0 ||
        zone_moved->from_zone != GLOG_ZONE_DECK ||
        zone_moved->to_zone != GLOG_ZONE_SELECTION) {
      continue;
    }

    assert(selection_logs < 2);
    assert(zone_moved->to_index == selection_logs);
    assert(zone_moved->from_index >= 0);
    selection_logs++;
  }

  assert(selection_logs == 2);
  assert(ecs_get_ordered_children(world, selection).count == 2);

  azk_clear_game_logs(world);

  ecs_defer_begin(world);
  move_selection_to_hand(world, cards[1]);
  ecs_defer_end(world);
  azk_finalize_pending_zone_move_logs(world);

  logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  int hand_logs = 0;
  for (uint8_t i = 0; i < log_count; i++) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED) {
      continue;
    }

    const GameLogZoneMoved *zone_moved = &logs[i].data.zone_moved;
    if (zone_moved->card.player != 0 ||
        zone_moved->from_zone != GLOG_ZONE_SELECTION ||
        zone_moved->to_zone != GLOG_ZONE_HAND) {
      continue;
    }

    hand_logs++;
    assert(zone_moved->from_index == 1);
    assert(zone_moved->to_index == initial_hand_count);
  }

  assert(hand_logs == 1);
  assert(ecs_get_ordered_children(world, hand).count == initial_hand_count + 1);
  assert(ecs_get_ordered_children(world, selection).count == 1);

  azk_world_fini(world);
}

int main(void) {
  test_azk_world_init_sets_game_state();
  test_world_init_creates_player_zones();
  test_world_init_assigns_damage_trackers_to_cards();
  test_world_init_assigns_condition_countdowns_to_cards();
  test_apply_frozen_initializes_countdown_while_deferred();
  test_count_tappable_ikz_sources_ignores_zero_token_entity();
  test_get_tappable_ikz_cards_rejects_zero_token_entity();
  test_init_player_deck_raizen();
  test_azk01_001_card_def_and_instantiation();
  test_ability_registry_lookup();
  test_st01_007_validate_needs_hand_and_deck();
  test_st01_007_validate_cost_target();
  test_st01_007_ability_flow_confirm_and_execute();
  test_st01_007_ability_flow_decline();

  // STT01-003 tests
  test_stt01_003_mills_5_without_weapons();
  test_stt01_003_mills_3_with_weapons();
  test_stt01_003_resolves_without_confirmation();
  test_stt01_003_mills_all_if_deck_smaller();

  // STT01-005 tests
  test_stt01_005_ability_registry_check();
  test_stt01_005_validate_needs_deck_cards();
  test_stt01_005_validate_only_in_alley();
  test_stt01_005_validate_effect_target();
  test_stt01_005_ability_flow_full();
  test_draw_cards_with_deckout_check();
  test_draw_cards_with_deckout_check_success();
  test_stt02_014_effect_target_uses_zone_index();
  test_stt02_014_action_mask_uses_zone_index();
  test_azk01_002_validate_rejects_dead_leader();
  test_azk01_002_spell_heals_owner_leader();
  test_triggered_ability_confirmation_restores_active_player();
  test_triggered_ability_decline_restores_active_player();
  test_triggered_mandatory_target_selection_skips_confirmation();
  test_azk01_004_when_attacking_buff_expires_end_of_turn();
  test_triggered_mandatory_up_to_effect_skips_confirmation();
  test_triggered_selection_pick_skips_confirmation_stt02_003();
  test_triggered_selection_pick_skips_confirmation_stt02_013();
  test_azk01_003_ability_flow_excludes_self_and_adds_black_jade_card();
  test_leader_response_enters_effect_selection();
  test_gate_portal_enters_selection_flow_stt01_002();
  test_start_phase_skips_opening_draw_for_starting_player();
  test_observation_garden_slots_use_zone_index();
  test_leader_with_multiple_stt01_013_weapons_supports_attack_mask_and_combat();

  // Game log tests
  printf("Running game log tests...\n");
  test_game_log_singleton_initialized();
  test_return_card_to_hand_uses_pending_hand_logs();
  test_move_selection_to_hand_uses_pending_hand_logs();
  test_deck_to_selection_to_hand_finalizes_each_committed_step();
  printf("Game log tests passed!\n");

  return 0;
}
