#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <flecs.h>

#include "world.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "abilities/cards/st01_007.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/deck_utils.h"
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

static void test_azk_world_init_sets_game_state(void) {
  const uint32_t seed = 1234;
  ecs_world_t *world = azk_world_init(seed);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  assert(gs->seed == seed);
  assert(gs->phase == PHASE_PREGAME_MULLIGAN);
  assert(gs->active_player_index == 0);
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

  // A card without ability should return NULL or has_ability=false
  const AbilityDef *no_ability = azk_get_ability_def(CARD_DEF_STT01_003);
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

  // Step 1: Trigger the ability
  bool triggered = azk_trigger_on_play_ability(world, st01_007_card, player);
  assert(triggered);

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

  // Step 1: Trigger the ability
  bool triggered = azk_trigger_on_play_ability(world, st01_007_card, player);
  assert(triggered);
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

int main(void) {
  test_azk_world_init_sets_game_state();
  test_world_init_creates_player_zones();
  test_init_player_deck_raizen();
  test_ability_registry_lookup();
  test_st01_007_validate_needs_hand_and_deck();
  test_st01_007_validate_cost_target();
  test_st01_007_ability_flow_confirm_and_execute();
  test_st01_007_ability_flow_decline();
  return 0;
}
