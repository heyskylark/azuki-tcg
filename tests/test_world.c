#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flecs.h>

#include "azuki/engine.h"
#include "world.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "abilities/cards/azk01_103.h"
#include "abilities/targeting/ability_targeting.h"
#include "abilities/cards/st01_007.h"
#include "abilities/cards/azk01_028.h"
#include "abilities/cards/stt01_003.h"
#include "abilities/cards/stt01_005.h"
#include "components/abilities.h"
#include "components/components.h"
#include "components/game_log.h"
#include "systems/main.h"
#include "systems/main_phase.h"
#include "systems/phase_gate.h"
#include "systems/combat_resolve_phase.h"
#include "systems/response_phase.h"
#include "utils/card_utils.h"
#include "utils/ability_util.h"
#include "utils/combat_util.h"
#include "utils/deck_utils.h"
#include "utils/damage_util.h"
#include "utils/game_log_util.h"
#include "utils/observation_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/weapon_util.h"
#include "utils/zone_util.h"
#include "validation/action_enumerator.h"
#include "validation/action_validation.h"
#include "generated/card_defs.h"

static void azk_test_assert_fail(const char *expr, const char *file, int line) {
  fprintf(stderr, "test assertion failed: %s (%s:%d)\n", expr, file, line);
  abort();
}

#define AZK_TEST_ASSERT(expr)                                                 \
  do {                                                                        \
    if (!(expr)) {                                                            \
      azk_test_assert_fail(#expr, __FILE__, __LINE__);                        \
    }                                                                         \
  } while (0)

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

static void initialize_test_card_runtime_components(ecs_world_t *world,
                                                    ecs_entity_t card);

static ecs_entity_t create_basic_entity_card(ecs_world_t *world,
                                             ecs_entity_t player,
                                             ecs_entity_t zone,
                                             CardDefId card_id,
                                             CardElement element,
                                             const char *name,
                                             uint8_t zone_index);

static ecs_entity_t create_basic_weapon_card(ecs_world_t *world,
                                             ecs_entity_t player,
                                             ecs_entity_t zone,
                                             CardDefId card_id,
                                             CardElement element,
                                             const char *name);

static void grant_ikz_cards_to_player(ecs_world_t *world, uint8_t player_index,
                                      int card_count) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  bool moved = move_cards_to_zone(world, gs->zones[player_index].ikz_pile,
                                  gs->zones[player_index].ikz_area,
                                  card_count, NULL);
  assert(moved);
}

static void submit_engine_action_and_advance(AzkEngine *engine,
                                             const UserAction *action) {
  bool submitted = azk_engine_submit_action(engine, action);
  assert(submitted);

  azk_engine_tick(engine);
  while (!azk_engine_requires_action(engine) &&
         !azk_engine_is_game_over(engine)) {
    azk_engine_tick(engine);
  }
}

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

static void setup_single_player_play_fixture(ecs_world_t *world,
                                             ecs_entity_t *out_player,
                                             PlayerZones *out_zones) {
  ecs_set(world, ecs_id(GameState), GameState, {0});
  ecs_set(world, ecs_id(AbilityContext), AbilityContext, {0});

  ecs_entity_t player = ecs_new(world);
  ecs_set(world, player, PlayerId, {.pid = 0});
  ecs_set(world, player, PlayerNumber, {.player_number = 0});

  PlayerZones zones = {0};
  zones.hand = create_zone(world, player, ZHand, "Hand_P0");
  zones.deck = create_zone(world, player, ZDeck, "Deck_P0");
  zones.leader = create_zone(world, player, ZLeader, "Leader_P0");
  zones.garden = create_zone(world, player, ZGarden, "Garden_P0");
  zones.alley = create_zone(world, player, ZAlley, "Alley_P0");
  zones.ikz_area = create_zone(world, player, ZIKZAreaTag, "IKZArea_P0");
  zones.discard = create_zone(world, player, ZDiscard, "Discard_P0");

  ecs_entity_t opponent = ecs_new(world);
  ecs_set(world, opponent, PlayerId, {.pid = 1});
  ecs_set(world, opponent, PlayerNumber, {.player_number = 1});

  PlayerZones opponent_zones = {0};
  opponent_zones.hand = create_zone(world, opponent, ZHand, "Hand_P1");
  opponent_zones.deck = create_zone(world, opponent, ZDeck, "Deck_P1");
  opponent_zones.leader = create_zone(world, opponent, ZLeader, "Leader_P1");
  opponent_zones.garden = create_zone(world, opponent, ZGarden, "Garden_P1");
  opponent_zones.alley = create_zone(world, opponent, ZAlley, "Alley_P1");
  opponent_zones.discard =
      create_zone(world, opponent, ZDiscard, "Discard_P1");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player;
  gs->players[1] = opponent;
  gs->active_player_index = 0;
  gs->winner = -1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 1;
  gs->zones[0] = zones;
  gs->zones[1] = opponent_zones;
  ecs_singleton_modified(world, GameState);

  azk_clear_ability_context(world);

  if (out_player != NULL) {
    *out_player = player;
  }
  if (out_zones != NULL) {
    *out_zones = zones;
  }
}

static ecs_entity_t create_bobu_leader(ecs_world_t *world, ecs_entity_t player,
                                       ecs_entity_t leader_zone,
                                       int8_t current_hp) {
  ecs_entity_t leader = ecs_new(world);
  ecs_set_name(world, leader, "STT03-001_test");
  ecs_set(world, leader, CardId, {.id = CARD_DEF_STT03_001});
  ecs_set(world, leader, Type, {.value = CARD_TYPE_LEADER});
  ecs_set(world, leader, Element, {.element = CARD_ELEMENT_EARTH});
  ecs_set(world, leader, BaseStats, {.attack = 0, .health = 20});
  ecs_set(world, leader, CurStats, {.cur_atk = 0, .cur_hp = current_hp});
  ecs_set(world, leader, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, leader, EcsChildOf, leader_zone);
  ecs_add_pair(world, leader, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, leader);
  attach_ability_components(world, leader);
  return leader;
}

static ecs_entity_t create_basic_leader(ecs_world_t *world, ecs_entity_t player,
                                        ecs_entity_t leader_zone,
                                        CardDefId card_id,
                                        CardElement element,
                                        const char *name) {
  ecs_entity_t leader = ecs_new(world);
  ecs_set_name(world, leader, name);
  ecs_set(world, leader, CardId, {.id = card_id});
  ecs_set(world, leader, Type, {.value = CARD_TYPE_LEADER});
  ecs_add(world, leader, TLeader);
  ecs_set(world, leader, Element, {.element = element});
  ecs_set(world, leader, BaseStats, {.attack = 0, .health = 20});
  ecs_set(world, leader, CurStats, {.cur_atk = 0, .cur_hp = 20});
  ecs_set(world, leader, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, leader, EcsChildOf, leader_zone);
  ecs_add_pair(world, leader, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, leader);
  attach_ability_components(world, leader);
  return leader;
}

static ecs_entity_t create_ikz_card(ecs_world_t *world, ecs_entity_t player,
                                    ecs_entity_t ikz_area, const char *name) {
  ecs_entity_t ikz = ecs_new(world);
  ecs_set_name(world, ikz, name);
  ecs_set(world, ikz, Type, {.value = CARD_TYPE_IKZ});
  ecs_set(world, ikz, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, ikz, EcsChildOf, ikz_area);
  ecs_add_pair(world, ikz, Rel_OwnedBy, player);
  return ikz;
}

static void setup_pending_attack_response_fixture(ecs_world_t *world,
                                                  ecs_entity_t *out_defender,
                                                  PlayerZones *out_defender_zones) {
  ecs_entity_t defender = 0;
  PlayerZones defender_zones = {0};
  setup_single_player_play_fixture(world, &defender, &defender_zones);
  init_phase_gate_system(world);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t attacker_player = gs_ro->players[1];
  PlayerZones attacker_zones = gs_ro->zones[1];

  ecs_entity_t leader = create_basic_leader(
      world, defender, defender_zones.leader, CARD_DEF_STT01_001,
      CARD_ELEMENT_LIGHTNING, "ResponseLeader_P0_Test");
  ecs_entity_t attacker = create_basic_entity_card(
      world, attacker_player, attacker_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "ResponseAttacker_Test", 0);
  create_ikz_card(world, defender, defender_zones.ikz_area, "ResponseIKZ_Test");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  gs->combat_state.attacking_card = attacker;
  gs->combat_state.defender_card = leader;
  gs->combat_state.defender_intercepted = false;
  ecs_singleton_modified(world, GameState);

  if (out_defender != NULL) {
    *out_defender = defender;
  }
  if (out_defender_zones != NULL) {
    *out_defender_zones = defender_zones;
  }
}

static void initialize_test_card_runtime_components(ecs_world_t *world,
                                                    ecs_entity_t card) {
  ecs_set(world, card, DamageTracker, {0});
  ecs_set(world, card, CardConditionCountdown,
          {.frozen_duration = 0,
           .shocked_duration = 0,
           .effect_immune_duration = 0,
           .timed_tag_grant_count = 0});
}

static ecs_entity_t create_earth_entity_card(ecs_world_t *world,
                                             ecs_entity_t player,
                                             ecs_entity_t zone,
                                             const char *name) {
  ecs_entity_t card = ecs_new(world);
  ecs_set_name(world, card, name);
  ecs_set(world, card, CardId, {.id = CARD_DEF_STT03_003});
  ecs_set(world, card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, card, Element, {.element = CARD_ELEMENT_EARTH});
  ecs_set(world, card, BaseStats, {.attack = 1, .health = 1});
  ecs_set(world, card, CurStats, {.cur_atk = 1, .cur_hp = 1});
  ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, card, EcsChildOf, zone);
  ecs_add_pair(world, card, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, card);
  return card;
}

static ecs_entity_t create_basic_entity_card(ecs_world_t *world,
                                             ecs_entity_t player,
                                             ecs_entity_t zone,
                                             CardDefId card_id,
                                             CardElement element,
                                             const char *name,
                                             uint8_t zone_index) {
  ecs_entity_t card = ecs_new(world);
  ecs_set_name(world, card, name);
  ecs_set(world, card, CardId, {.id = card_id});
  ecs_set(world, card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, card, Element, {.element = element});
  ecs_set(world, card, BaseStats, {.attack = 1, .health = 1});
  ecs_set(world, card, CurStats, {.cur_atk = 1, .cur_hp = 1});
  ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
  ecs_set(world, card, ZoneIndex, {.index = zone_index});
  ecs_add_pair(world, card, EcsChildOf, zone);
  ecs_add_pair(world, card, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, card);
  attach_ability_components(world, card);
  return card;
}

static bool test_validate_always_true(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

static ecs_entity_t get_first_timed_ability(ecs_world_t *world,
                                            ecs_entity_t card,
                                            ecs_id_t timing_tag) {
  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_timed_abilities(
      world, card, timing_tag, abilities, AZK_MAX_CARD_ABILITIES);
  assert(ability_count > 0);
  return abilities[0];
}

static ecs_entity_t create_basic_weapon_card(ecs_world_t *world,
                                             ecs_entity_t player,
                                             ecs_entity_t zone,
                                             CardDefId card_id,
                                             CardElement element,
                                             const char *name) {
  ecs_entity_t card = ecs_new(world);
  ecs_set_name(world, card, name);
  ecs_set(world, card, CardId, {.id = card_id});
  ecs_set(world, card, Type, {.value = CARD_TYPE_WEAPON});
  ecs_add(world, card, TWeapon);
  ecs_set(world, card, Element, {.element = element});
  ecs_set(world, card, BaseStats, {.attack = 1, .health = 0});
  ecs_set(world, card, CurStats, {.cur_atk = 1, .cur_hp = 0});
  ecs_set(world, card, IKZCost, {.ikz_cost = 1});
  ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, card, EcsChildOf, zone);
  ecs_add_pair(world, card, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, card);
  attach_ability_components(world, card);
  return card;
}

static void test_azk01_046_prefab_instances_inherit_garden_force_tapped_tag(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t prefab = azk_prefab_from_id(CARD_DEF_AZK01_046);
  assert(prefab != 0);
  assert(ecs_has(world, prefab, AttrGardenForceTapped));

  ecs_entity_t card = ecs_new_w_pair(world, EcsIsA, prefab);
  assert(card != 0);
  assert(ecs_has(world, card, AttrGardenForceTapped));

  ecs_fini(world);
}

static void test_summon_card_into_garden_taps_garden_force_tapped_cards(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t card = create_basic_entity_card(
      world, player, zones.hand, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "garden_force_tapped_summon_test", 0);
  ecs_add(world, card, AttrGardenForceTapped);

  PlayEntityIntent intent = {
      .player = player,
      .card = card,
      .placement_type = ZONE_GARDEN,
      .target_zone = zones.garden,
      .zone_index = 0,
      .displaced_card = 0,
  };

  int result = summon_card_into_zone_index(world, &intent);
  assert(result == 0);
  assert(ecs_get_target(world, card, EcsChildOf, 0) == zones.garden);

  const TapState *tap = ecs_get(world, card, TapState);
  assert(tap != NULL);
  assert(tap->tapped);
  assert(tap->cooldown);

  ecs_fini(world);
}

static void test_untap_all_cards_in_zone_keeps_garden_force_tapped_cards_tapped(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t forced_card = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "garden_force_tapped_untap_test", 0);
  ecs_add(world, forced_card, AttrGardenForceTapped);
  ecs_set(world, forced_card, TapState, {.tapped = true, .cooldown = true});

  ecs_entity_t normal_card = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "normal_untap_test", 1);
  ecs_set(world, normal_card, TapState, {.tapped = true, .cooldown = true});

  untap_all_cards_in_zone(world, zones.garden);

  const TapState *forced_tap = ecs_get(world, forced_card, TapState);
  const TapState *normal_tap = ecs_get(world, normal_card, TapState);
  assert(forced_tap != NULL);
  assert(normal_tap != NULL);

  assert(forced_tap->tapped);
  assert(!forced_tap->cooldown);
  assert(!normal_tap->tapped);
  assert(!normal_tap->cooldown);

  ecs_fini(world);
}

static void test_summon_card_into_garden_logs_garden_force_tapped_metadata(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t card = create_basic_entity_card(
      world, player, zones.hand, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "garden_force_tapped_log_test", 0);
  ecs_add(world, card, AttrGardenForceTapped);

  azk_clear_game_logs(world);

  int result = summon_card_into_zone_index(
      world,
      &(PlayEntityIntent){
          .player = player,
          .card = card,
          .placement_type = ZONE_GARDEN,
          .target_zone = zones.garden,
          .zone_index = 0,
          .displaced_card = 0,
      });
  assert(result == 0);

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  bool found_move_log = false;
  for (uint8_t i = 0; i < log_count; ++i) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED ||
        logs[i].data.zone_moved.card.card_def_id != CARD_DEF_STT03_003) {
      continue;
    }

    const GameLogZoneMoved *move = &logs[i].data.zone_moved;
    if (move->from_zone != GLOG_ZONE_HAND || move->to_zone != GLOG_ZONE_GARDEN) {
      continue;
    }

    found_move_log = true;
    assert(move->to_index == 0);
    assert(move->metadata.tapped);
    assert(move->metadata.cooldown);
  }
  assert(found_move_log);

  ecs_fini(world);
}

static void test_gate_card_into_garden_logs_garden_force_tapped_metadata(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t gate_zone = create_zone(world, player, ZGate, "Gate_P0_Test");
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->zones[0].gate = gate_zone;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set_name(world, gate_card, "GatePortal_ForceTap_Test");
  ecs_set(world, gate_card, CardId,
          {.id = CARD_DEF_STT01_002, .code = "STT01-002"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate_card, EcsChildOf, gate_zone);
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, gate_card);
  attach_ability_components(world, gate_card);

  ecs_entity_t alley_card = create_basic_entity_card(
      world, player, zones.alley, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "gate_portal_force_tapped_log_test", 0);
  ecs_add(world, alley_card, AttrGardenForceTapped);

  azk_clear_game_logs(world);

  int result = gate_card_into_garden(
      world,
      &(GatePortalIntent){
          .player = player,
          .alley_card = alley_card,
          .target_zone = zones.garden,
          .garden_index = 0,
          .displaced_card = 0,
          .gate_card = gate_card,
      });
  assert(result == 0);

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  bool found_move_log = false;
  for (uint8_t i = 0; i < log_count; ++i) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED ||
        logs[i].data.zone_moved.card.card_def_id != CARD_DEF_STT03_003) {
      continue;
    }

    const GameLogZoneMoved *move = &logs[i].data.zone_moved;
    if (move->from_zone != GLOG_ZONE_ALLEY ||
        move->to_zone != GLOG_ZONE_GARDEN) {
      continue;
    }

    found_move_log = true;
    assert(move->to_index == 0);
    assert(move->metadata.tapped);
    assert(move->metadata.cooldown);
  }
  assert(found_move_log);

  ecs_fini(world);
}

static void test_engine_action_play_to_garden_logs_garden_force_tapped_metadata(void) {
  const CardInfo player0_deck[] = {
      {.card_id = CARD_DEF_STT01_001, .card_count = 1},
      {.card_id = CARD_DEF_STT01_002, .card_count = 1},
      {.card_id = CARD_DEF_AZK01_046, .card_count = 50},
      {.card_id = CARD_DEF_IKZ_001, .card_count = 10},
  };
  const CardInfo player1_deck[] = {
      {.card_id = CARD_DEF_STT01_001, .card_count = 1},
      {.card_id = CARD_DEF_STT01_002, .card_count = 1},
      {.card_id = CARD_DEF_AZK01_046, .card_count = 50},
      {.card_id = CARD_DEF_IKZ_001, .card_count = 10},
  };

  AzkEngine *engine = azk_engine_create_with_decks(
      12345, player0_deck,
      sizeof(player0_deck) / sizeof(player0_deck[0]), player1_deck,
      sizeof(player1_deck) / sizeof(player1_deck[0]));
  assert(engine != NULL);

  const GameState *pregame = ecs_singleton_get(engine, GameState);
  assert(pregame != NULL);
  ecs_entity_t player0 = pregame->players[0];
  ecs_entity_t player1 = pregame->players[1];

  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player0,
                  .type = ACT_NOOP,
                  .subaction_1 = 0,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });
  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player1,
                  .type = ACT_NOOP,
                  .subaction_1 = 0,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });

  const GameState *main_phase = ecs_singleton_get(engine, GameState);
  assert(main_phase != NULL);
  assert(main_phase->phase == PHASE_MAIN);
  assert(main_phase->active_player_index == 0);

  azk_clear_game_logs(engine);

  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player0,
                  .type = ACT_PLAY_ENTITY_TO_GARDEN,
                  .subaction_1 = 0,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(engine, &log_count);
  assert(logs != NULL);

  bool found_move_log = false;
  for (uint8_t i = 0; i < log_count; ++i) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED ||
        logs[i].data.zone_moved.card.card_def_id != CARD_DEF_AZK01_046) {
      continue;
    }

    const GameLogZoneMoved *move = &logs[i].data.zone_moved;
    if (move->from_zone != GLOG_ZONE_HAND ||
        move->to_zone != GLOG_ZONE_GARDEN) {
      continue;
    }

    found_move_log = true;
    assert(move->to_index == 0);
    assert(move->metadata.tapped);
    assert(move->metadata.cooldown);
  }
  assert(found_move_log);

  ObservationData obs = {0};
  bool observed = azk_engine_observe(engine, 0, &obs);
  assert(observed);
  assert(obs.my_observation_data.garden[0].id.id == CARD_DEF_AZK01_046);
  assert(obs.my_observation_data.garden[0].tap_state.tapped);
  assert(obs.my_observation_data.garden[0].tap_state.cooldown);

  azk_engine_destroy(engine);
}

static void test_azk01_103_cannot_sacrifice_itself_for_cost(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t station = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_103, CARD_ELEMENT_EARTH,
      "AZK01-103_station", 0);

  assert(!azk01_103_validate(world, station, player));
  assert(!azk01_103_validate_cost_target(world, station, player, station));

  ecs_entity_t ally = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-103_ally", 1);

  assert(azk01_103_validate(world, station, player));
  assert(!azk01_103_validate_cost_target(world, station, player, station));
  assert(azk01_103_validate_cost_target(world, station, player, ally));

  ecs_set(world, ally, TapState, {.tapped = true, .cooldown = false});
  assert(!azk01_103_validate(world, station, player));
  assert(!azk01_103_validate_cost_target(world, station, player, ally));

  ecs_fini(world);
}

static void test_prefab_instances_inherit_targeting_and_ikz_source_tags(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  typedef struct {
    CardDefId card_id;
    ecs_entity_t tag;
  } PrefabTagExpectation;

  const PrefabTagExpectation cases[] = {
      {CARD_DEF_AZK01_037, AttrCanTargetTappedAndUntappedAlley},
      {CARD_DEF_AZK01_038, AttrCanTargetTappedAndUntappedAlley},
      {CARD_DEF_AZK01_077, AttrCanTargetLeaderOnly},
      {CARD_DEF_STT03_007, AttrCountsAsIkzSource},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    ecs_entity_t prefab = azk_prefab_from_id(cases[i].card_id);
    assert(prefab != 0);
    assert(ecs_has_id(world, prefab, cases[i].tag));

    ecs_entity_t card = ecs_new_w_pair(world, EcsIsA, prefab);
    assert(card != 0);
    assert(ecs_has_id(world, card, cases[i].tag));
  }

  ecs_fini(world);
}

static void test_can_target_leader_only_blocks_non_leader_attacks(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "LeaderOnly_AttackerLeader_Test");
  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "LeaderOnly_DefenderLeader_Test");

  ecs_entity_t attacker = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_NORMAL,
      "LeaderOnly_Attacker_Test", 0);
  ecs_add(world, attacker, AttrCanTargetLeaderOnly);

  ecs_entity_t defender = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "LeaderOnly_Defender_Test", 0);
  ecs_set(world, defender, TapState, {.tapped = true, .cooldown = false});

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 0;
  gs->phase = PHASE_MAIN;
  ecs_singleton_modified(world, GameState);

  AttackIntent garden_intent = {0};
  UserAction garden_attack = {
      .player = player,
      .type = ACT_ATTACK,
      .subaction_1 = 0,
      .subaction_2 = 0,
  };
  bool garden_valid = azk_validate_attack_action(world, gs, player, &garden_attack,
                                                 false, &garden_intent);
  assert(!garden_valid);

  AttackIntent leader_intent = {0};
  UserAction leader_attack = {
      .player = player,
      .type = ACT_ATTACK,
      .subaction_1 = 0,
      .subaction_2 = GARDEN_SIZE,
  };
  bool leader_valid = azk_validate_attack_action(world, gs, player, &leader_attack,
                                                 false, &leader_intent);
  assert(leader_valid);
  assert(leader_intent.attacking_card == attacker);

  ecs_fini(world);
}

static void test_stormglass_weapons_grant_and_remove_leader_alley_targeting(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t leader = create_basic_leader(world, player, zones.leader,
                                            CARD_DEF_STT01_001,
                                            CARD_ELEMENT_LIGHTNING,
                                            "Stormglass_Leader_Test");
  ecs_entity_t daggers = create_basic_weapon_card(
      world, player, zones.hand, CARD_DEF_AZK01_043, CARD_ELEMENT_LIGHTNING,
      "Stormglass_Daggers_Test");
  ecs_entity_t katana = create_basic_weapon_card(
      world, player, zones.hand, CARD_DEF_AZK01_095, CARD_ELEMENT_LIGHTNING,
      "Stormglass_Katana_Test");

  assert(!ecs_has(world, leader, AttrCanTargetTappedAndUntappedAlley));

  AttachWeaponIntent daggers_intent = {
      .player = player,
      .weapon_card = daggers,
      .target_card = leader,
      .hand_index = 0,
      .use_ikz_token = false,
      .ikz_card_count = 0,
  };
  assert(attach_weapon_from_hand(world, &daggers_intent) == 0);
  assert(ecs_has(world, leader, AttrCanTargetTappedAndUntappedAlley));

  AttachWeaponIntent katana_intent = {
      .player = player,
      .weapon_card = katana,
      .target_card = leader,
      .hand_index = 1,
      .use_ikz_token = false,
      .ikz_card_count = 0,
  };
  assert(attach_weapon_from_hand(world, &katana_intent) == 0);
  assert(ecs_has(world, leader, AttrCanTargetTappedAndUntappedAlley));

  discard_card(world, daggers);
  assert(ecs_has(world, leader, AttrCanTargetTappedAndUntappedAlley));

  discard_card(world, katana);
  assert(!ecs_has(world, leader, AttrCanTargetTappedAndUntappedAlley));

  ecs_fini(world);
}

static void
test_azk01_018_reduces_only_combat_damage_for_equipped_leader(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  ecs_entity_t leader = create_basic_leader(
      world, player, zones.leader, CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
      "AZK01-018_Leader_Test");
  ecs_set(world, leader, CurStats, {.cur_atk = 0, .cur_hp = 12});
  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "AZK01-018_Opponent_Leader_Test");

  ecs_entity_t enemy = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-018_Enemy_Test", 0);
  ecs_set(world, enemy, BaseStats, {.attack = 2, .health = 1});
  ecs_set(world, enemy, CurStats, {.cur_atk = 2, .cur_hp = 1});

  ecs_entity_t effect_source = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-018_Effect_Source_Test", 1);

  ecs_entity_t weapon_prefab = azk_prefab_from_id(CARD_DEF_AZK01_018);
  assert(weapon_prefab != 0);
  ecs_entity_t weapon = ecs_new_w_pair(world, EcsIsA, weapon_prefab);
  ecs_set_name(world, weapon, "AZK01-018_Weapon_Test");
  ecs_add_pair(world, weapon, Rel_OwnedBy, player);
  ecs_add_pair(world, weapon, EcsChildOf, zones.hand);
  initialize_test_card_runtime_components(world, weapon);
  attach_ability_components(world, weapon);
  AttachWeaponIntent attach_intent = {
      .player = player,
      .weapon_card = weapon,
      .target_card = leader,
      .hand_index = 0,
      .use_ikz_token = false,
      .ikz_card_count = 0,
  };
  assert(attach_weapon_from_hand(world, &attach_intent) == 0);
  assert(get_total_incoming_combat_damage_modifier(world, leader) == -1);

  AttackIntent attack_intent = {
      .attacking_player = player,
      .defending_player = opponent,
      .attacking_card = leader,
      .defending_card = enemy,
      .attacker_index = GARDEN_SIZE,
      .defender_index = 0,
      .attacker_is_leader = true,
  };
  assert(attack(world, &attack_intent) == 0);
  resolve_combat(world);

  const CurStats *leader_stats = ecs_get(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 11);

  bool damaged = deal_effect_damage_from_source(world, effect_source, leader, 1);
  assert(damaged);

  leader_stats = ecs_get(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 10);

  ecs_fini(world);
}

static void
test_azk01_018_does_not_reduce_combat_damage_for_nonleader_host(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "AZK01-018_Entity_Leader_Test");
  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING,
                      "AZK01-018_Entity_Opponent_Leader_Test");

  ecs_entity_t host = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-018_Host_Test", 0);
  ecs_set(world, host, BaseStats, {.attack = 1, .health = 5});
  ecs_set(world, host, CurStats, {.cur_atk = 1, .cur_hp = 5});

  ecs_entity_t enemy = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-018_Entity_Enemy_Test", 0);
  ecs_set(world, enemy, BaseStats, {.attack = 2, .health = 2});
  ecs_set(world, enemy, CurStats, {.cur_atk = 2, .cur_hp = 2});

  ecs_entity_t weapon_prefab = azk_prefab_from_id(CARD_DEF_AZK01_018);
  assert(weapon_prefab != 0);
  ecs_entity_t weapon = ecs_new_w_pair(world, EcsIsA, weapon_prefab);
  ecs_set_name(world, weapon, "AZK01-018_Entity_Weapon_Test");
  ecs_add_pair(world, weapon, Rel_OwnedBy, player);
  ecs_add_pair(world, weapon, EcsChildOf, zones.hand);
  initialize_test_card_runtime_components(world, weapon);
  attach_ability_components(world, weapon);
  AttachWeaponIntent attach_intent = {
      .player = player,
      .weapon_card = weapon,
      .target_card = host,
      .hand_index = 0,
      .use_ikz_token = false,
      .ikz_card_count = 0,
  };
  assert(attach_weapon_from_hand(world, &attach_intent) == 0);
  assert(get_total_incoming_combat_damage_modifier(world, host) == 0);

  AttackIntent attack_intent = {
      .attacking_player = player,
      .defending_player = opponent,
      .attacking_card = host,
      .defending_card = enemy,
      .attacker_index = 0,
      .defender_index = 0,
      .attacker_is_leader = false,
  };
  assert(attack(world, &attack_intent) == 0);
  resolve_combat(world);

  const CurStats *host_stats = ecs_get(world, host, CurStats);
  assert(host_stats != NULL);
  assert(host_stats->cur_hp == 3);

  ecs_fini(world);
}

static void test_is_ikz_card_counts_as_source_only_in_garden(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t prefab = azk_prefab_from_id(CARD_DEF_STT03_007);
  assert(prefab != 0);
  ecs_entity_t card = ecs_new_w_pair(world, EcsIsA, prefab);
  assert(card != 0);
  ecs_set_name(world, card, "STT03-007_IKZ_Source_Test");
  ecs_add_pair(world, card, EcsChildOf, zones.garden);
  ecs_add_pair(world, card, Rel_OwnedBy, player);
  ecs_set(world, card, ZoneIndex, {.index = 0});

  assert(ecs_has(world, card, AttrCountsAsIkzSource));
  assert(azk_card_counts_as_ikz_source(world, card));
  assert(azk_count_tappable_ikz_sources(world, zones.ikz_area, false) == 1);

  ecs_add_pair(world, card, EcsChildOf, zones.alley);
  ecs_set(world, card, ZoneIndex, {.index = 0});

  assert(!azk_card_counts_as_ikz_source(world, card));
  assert(azk_count_tappable_ikz_sources(world, zones.ikz_area, false) == 0);

  ecs_fini(world);
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
    initialize_test_card_runtime_components(world, selis_cards[player_index]);
    attach_ability_components(world, selis_cards[player_index]);
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

static void test_generated_card_def_stt03_015_has_correct_ikz_cost(void) {
  const CardDef *def = azk_card_def_from_id(CARD_DEF_STT03_015);
  assert(def != NULL);
  assert(def->has_ikz_cost);
  assert(def->ikz_cost.ikz_cost == 4);
}

static void test_generated_card_def_stt04_013_has_correct_gate_power(void) {
  const CardDef *def = azk_card_def_from_id(CARD_DEF_STT04_013);
  assert(def != NULL);
  assert(def->has_gate_points);
  assert(def->gate_points.gate_points == 1);
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

static void test_additional_card_abilities_get_sparse_action_indices(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  AZK_TEST_ASSERT(
      azk_set_additional_card_abilities(CARD_DEF_STT03_001, NULL, 0));

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t leader = create_bobu_leader(world, player, zones.leader, 20);
  create_ikz_card(world, player, zones.ikz_area, "IKZ_multi_1");
  create_ikz_card(world, player, zones.ikz_area, "IKZ_multi_2");

  const AbilityDef extra_defs[2] = {
      {
          .has_ability = true,
          .timing_tag = ecs_id(AMain),
          .validate = test_validate_always_true,
      },
      {
          .has_ability = true,
          .timing_tag = ecs_id(AWhenAttacked),
          .validate = test_validate_always_true,
      },
  };
  AZK_TEST_ASSERT(
      azk_set_additional_card_abilities(CARD_DEF_STT03_001, extra_defs, 2));
  AZK_TEST_ASSERT(azk_get_ability_count(CARD_DEF_STT03_001) == 3);

  attach_ability_components(world, leader);

  ecs_entity_t ability0 = azk_find_card_ability_by_registry_order(world, leader, 0);
  ecs_entity_t ability1 = azk_find_card_ability_by_registry_order(world, leader, 1);
  ecs_entity_t ability2 = azk_find_card_ability_by_registry_order(world, leader, 2);
  AZK_TEST_ASSERT(ability0 != 0);
  AZK_TEST_ASSERT(ability1 != 0);
  AZK_TEST_ASSERT(ability2 != 0);

  const AbilityInstance *instance0 = ecs_get(world, ability0, AbilityInstance);
  const AbilityInstance *instance1 = ecs_get(world, ability1, AbilityInstance);
  const AbilityInstance *instance2 = ecs_get(world, ability2, AbilityInstance);
  AZK_TEST_ASSERT(instance0 != NULL);
  AZK_TEST_ASSERT(instance1 != NULL);
  AZK_TEST_ASSERT(instance2 != NULL);
  AZK_TEST_ASSERT(instance0->registry_order == 0);
  AZK_TEST_ASSERT(instance1->registry_order == 1);
  AZK_TEST_ASSERT(instance2->registry_order == 2);
  AZK_TEST_ASSERT(instance0->action_index == 0);
  AZK_TEST_ASSERT(instance1->action_index == 1);
  AZK_TEST_ASSERT(instance2->action_index == AZK_NO_ACTION_INDEX);

  ecs_entity_t action_abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t action_ability_count = azk_collect_card_action_abilities(
      world, leader, action_abilities, AZK_MAX_CARD_ABILITIES);
  AZK_TEST_ASSERT(action_ability_count == 2);
  AZK_TEST_ASSERT(action_abilities[0] == ability0);
  AZK_TEST_ASSERT(action_abilities[1] == ability1);

  UserAction action = {
      .player = player,
      .type = ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
      .subaction_1 = GARDEN_SIZE,
      .subaction_2 = 1,
      .subaction_3 = 0,
  };
  ActivateAbilityIntent intent = {0};
  bool valid = azk_validate_activate_garden_or_leader_ability_action(
      world, ecs_singleton_get(world, GameState), player, &action, true,
      &intent);
  AZK_TEST_ASSERT(valid);
  AZK_TEST_ASSERT(intent.card == leader);
  AZK_TEST_ASSERT(intent.ability_index == 1);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  AZK_TEST_ASSERT(built);

  bool saw_base_action = false;
  bool saw_extra_action = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *candidate = &mask.legal_actions[i];
    if (candidate->type != ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY ||
        candidate->subaction_1 != GARDEN_SIZE) {
      continue;
    }

    if (candidate->subaction_2 == 0) {
      saw_base_action = true;
    } else if (candidate->subaction_2 == 1) {
      saw_extra_action = true;
    }
  }
  AZK_TEST_ASSERT(saw_base_action);
  AZK_TEST_ASSERT(saw_extra_action);

  AZK_TEST_ASSERT(
      azk_set_additional_card_abilities(CARD_DEF_STT03_001, NULL, 0));
  ecs_fini(world);
}

static void test_passive_observer_context_is_scoped_to_ability_entity(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t card = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT01_008, CARD_ELEMENT_NORMAL,
      "STT01-008_passive_scope", 0);
  ecs_entity_t ability = azk_find_card_ability_by_registry_order(world, card, 0);

  assert(ability != 0);
  assert(ecs_get_target(world, ability, Rel_AbilityOf, 0) == card);
  assert(!ecs_has(world, card, PassiveObserverContext));
  assert(ecs_has(world, ability, PassiveObserverContext));

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
  bool triggered = azk_trigger_main_ability(world, stt01_005_card, player, 0);
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
  bool triggered = azk_trigger_spell_ability(world, spell_card, player0, 0);
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

  bool triggered = azk_trigger_spell_ability(world, spell_card, player0, 0);
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

  bool entered_selection = azk_trigger_spell_ability(world, spell_card, player0, 0);
  assert(!entered_selection);
  const CurStats *leader_stats = ecs_get(world, leader_card, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 19);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  ecs_set(world, leader_card, CurStats, {.cur_atk = 0, .cur_hp = 19});
  entered_selection = azk_trigger_spell_ability(world, spell_card, player0, 0);
  assert(!entered_selection);
  leader_stats = ecs_get(world, leader_card, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 20);

  entered_selection = azk_trigger_spell_ability(world, spell_card, player0, 0);
  assert(!entered_selection);
  leader_stats = ecs_get(world, leader_card, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 20);

  ecs_fini(world);
}

static void test_azk01_065_spell_damages_owner_leader_and_selected_target(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  ecs_entity_t owner_leader = create_basic_leader(
      world, player, zones.leader, CARD_DEF_STT01_001, CARD_ELEMENT_FIRE,
      "AZK01-065_owner_leader");
  ecs_entity_t opponent_leader = create_basic_leader(
      world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
      CARD_ELEMENT_WATER, "AZK01-065_opponent_leader");
  (void)opponent_leader;

  ecs_entity_t target = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-065_target", 0);
  ecs_set(world, target, BaseStats, {.attack = 2, .health = 6});
  ecs_set(world, target, CurStats, {.cur_atk = 2, .cur_hp = 6});

  ecs_entity_t spell_card = ecs_new(world);
  ecs_set(world, spell_card, CardId,
          {.id = CARD_DEF_AZK01_065, .code = "AZK01-065"});
  ecs_set(world, spell_card, Type, {.value = CARD_TYPE_SPELL});
  ecs_set(world, spell_card, Element, {.element = CARD_ELEMENT_FIRE});
  ecs_add_pair(world, spell_card, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, spell_card);
  attach_ability_components(world, spell_card);

  bool triggered = azk_trigger_spell_ability(world, spell_card, player, 0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityDef *def = azk_get_ability_def(CARD_DEF_AZK01_065);
  assert(def != NULL);

  AbilityTargetChoice choices[AZK_MAX_ABILITY_TARGET_CHOICES] = {0};
  int choice_count = azk_collect_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, spell_card, player, choices,
      AZK_MAX_ABILITY_TARGET_CHOICES);
  assert(choice_count > 0);

  int target_action_index = -1;
  for (int i = 0; i < choice_count; ++i) {
    if (choices[i].entity == target) {
      target_action_index = choices[i].action_index;
      break;
    }
  }
  assert(target_action_index >= 0);

  bool selected = azk_process_effect_selection(world, target_action_index);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *owner_leader_stats = ecs_get(world, owner_leader, CurStats);
  const CurStats *target_stats = ecs_get(world, target, CurStats);
  assert(owner_leader_stats != NULL);
  assert(target_stats != NULL);
  assert(owner_leader_stats->cur_hp == 17);
  assert(target_stats->cur_hp == 1);

  ecs_fini(world);
}

static void
test_azk01_087_spell_enters_effect_selection_and_revalidates_second_pick(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  ecs_entity_t spell_card = ecs_new(world);
  ecs_set_name(world, spell_card, "AZK01-087_spell");
  ecs_set(world, spell_card, CardId,
          {.id = CARD_DEF_AZK01_087, .code = "AZK01-087"});
  ecs_set(world, spell_card, Type, {.value = CARD_TYPE_SPELL});
  ecs_set(world, spell_card, Element, {.element = CARD_ELEMENT_WATER});
  ecs_add_pair(world, spell_card, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, spell_card);
  attach_ability_components(world, spell_card);

  ecs_entity_t target_cost_3 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-087_target_cost_3", 0);
  ecs_set(world, target_cost_3, IKZCost, {.ikz_cost = 3});

  ecs_entity_t target_cost_1 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_004,
      CARD_ELEMENT_EARTH, "AZK01-087_target_cost_1", 1);
  ecs_set(world, target_cost_1, IKZCost, {.ikz_cost = 1});

  ecs_entity_t target_cost_5 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT04_004,
      CARD_ELEMENT_FIRE, "AZK01-087_target_cost_5", 2);
  ecs_set(world, target_cost_5, IKZCost, {.ikz_cost = 5});

  ecs_entity_t target_cost_6 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT04_014,
      CARD_ELEMENT_FIRE, "AZK01-087_target_cost_6", 3);
  ecs_set(world, target_cost_6, IKZCost, {.ikz_cost = 6});

  bool triggered = azk_trigger_spell_ability(world, spell_card, player, 0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  AzkActionMaskSet mask = {0};
  bool built =
      azk_build_action_mask_for_player(world, ecs_singleton_get(world, GameState),
                                       0, &mask);
  assert(built);

  bool found_noop = false;
  bool found_cost_3 = false;
  bool found_cost_1 = false;
  bool found_cost_5 = false;
  bool found_cost_6 = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
      continue;
    }
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }

    if (action->subaction_1 == 0) {
      found_cost_3 = true;
    }
    if (action->subaction_1 == 1) {
      found_cost_1 = true;
    }
    if (action->subaction_1 == 2) {
      found_cost_5 = true;
    }
    if (action->subaction_1 == 3) {
      found_cost_6 = true;
    }
  }

  assert(found_noop);
  assert(found_cost_3);
  assert(found_cost_1);
  assert(found_cost_5);
  assert(!found_cost_6);

  bool selected = azk_process_effect_selection(world, 0);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  mask = (AzkActionMaskSet){0};
  built = azk_build_action_mask_for_player(world, ecs_singleton_get(world, GameState),
                                           0, &mask);
  assert(built);

  found_noop = false;
  found_cost_3 = false;
  found_cost_1 = false;
  found_cost_5 = false;
  found_cost_6 = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
      continue;
    }
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }

    if (action->subaction_1 == 0) {
      found_cost_3 = true;
    }
    if (action->subaction_1 == 1) {
      found_cost_1 = true;
    }
    if (action->subaction_1 == 2) {
      found_cost_5 = true;
    }
    if (action->subaction_1 == 3) {
      found_cost_6 = true;
    }
  }

  assert(found_noop);
  assert(!found_cost_3);
  assert(found_cost_1);
  assert(!found_cost_5);
  assert(!found_cost_6);

  bool skipped = azk_process_effect_skip(world);
  assert(skipped);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  assert(ecs_get_target(world, target_cost_3, EcsChildOf, 0) ==
         opponent_zones.deck);
  assert(ecs_get_target(world, target_cost_1, EcsChildOf, 0) ==
         opponent_zones.garden);
  assert(ecs_get_target(world, target_cost_5, EcsChildOf, 0) ==
         opponent_zones.garden);
  assert(ecs_get_target(world, target_cost_6, EcsChildOf, 0) ==
         opponent_zones.garden);

  ecs_fini(world);
}

static void
test_azk01_089_main_ability_enters_effect_selection_and_allows_single_five_cost(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  ecs_entity_t source = create_basic_entity_card(
      world, player, zones.alley, CARD_DEF_AZK01_089, CARD_ELEMENT_WATER,
      "AZK01-089_source", 0);

  ecs_entity_t target_cost_3 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-089_target_cost_3", 0);
  ecs_set(world, target_cost_3, IKZCost, {.ikz_cost = 3});

  ecs_entity_t target_cost_1 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_004,
      CARD_ELEMENT_EARTH, "AZK01-089_target_cost_1", 1);
  ecs_set(world, target_cost_1, IKZCost, {.ikz_cost = 1});

  ecs_entity_t target_cost_5 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT04_004,
      CARD_ELEMENT_FIRE, "AZK01-089_target_cost_5", 2);
  ecs_set(world, target_cost_5, IKZCost, {.ikz_cost = 5});

  ecs_entity_t target_cost_6 = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT04_014,
      CARD_ELEMENT_FIRE, "AZK01-089_target_cost_6", 3);
  ecs_set(world, target_cost_6, IKZCost, {.ikz_cost = 6});

  bool triggered = azk_trigger_main_ability(world, source, player, 0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);
  assert(ecs_get_target(world, source, EcsChildOf, 0) == zones.deck);

  AzkActionMaskSet mask = {0};
  bool built =
      azk_build_action_mask_for_player(world, ecs_singleton_get(world, GameState),
                                       0, &mask);
  assert(built);

  bool found_noop = false;
  bool found_cost_3 = false;
  bool found_cost_1 = false;
  bool found_cost_5 = false;
  bool found_cost_6 = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
      continue;
    }
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }

    if (action->subaction_1 == 0) {
      found_cost_3 = true;
    }
    if (action->subaction_1 == 1) {
      found_cost_1 = true;
    }
    if (action->subaction_1 == 2) {
      found_cost_5 = true;
    }
    if (action->subaction_1 == 3) {
      found_cost_6 = true;
    }
  }

  assert(found_noop);
  assert(found_cost_3);
  assert(found_cost_1);
  assert(found_cost_5);
  assert(!found_cost_6);

  bool selected = azk_process_effect_selection(world, 2);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  mask = (AzkActionMaskSet){0};
  built = azk_build_action_mask_for_player(world, ecs_singleton_get(world, GameState),
                                           0, &mask);
  assert(built);

  found_noop = false;
  int effect_target_count = 0;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
      continue;
    }
    if (action->type == ACT_SELECT_EFFECT_TARGET) {
      effect_target_count++;
    }
  }

  assert(found_noop);
  assert(effect_target_count == 0);

  bool skipped = azk_process_effect_skip(world);
  assert(skipped);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  assert(ecs_get_target(world, target_cost_3, EcsChildOf, 0) ==
         opponent_zones.garden);
  assert(ecs_get_target(world, target_cost_1, EcsChildOf, 0) ==
         opponent_zones.garden);
  assert(ecs_get_target(world, target_cost_5, EcsChildOf, 0) ==
         opponent_zones.deck);
  assert(ecs_get_target(world, target_cost_6, EcsChildOf, 0) ==
         opponent_zones.garden);

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
      world,
      get_first_timed_ability(world, selis_cards[0], ecs_id(AWhenReturnedToHand)),
      players[0], TIMING_TAG_WHEN_RETURNED_TO_HAND);
  assert(queued);
  queued = azk_queue_triggered_effect(
      world,
      get_first_timed_ability(world, selis_cards[1], ecs_id(AWhenReturnedToHand)),
      players[1], TIMING_TAG_WHEN_RETURNED_TO_HAND);
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
      world,
      get_first_timed_ability(world, selis_cards[0], ecs_id(AWhenReturnedToHand)),
      players[0], TIMING_TAG_WHEN_RETURNED_TO_HAND);
  assert(queued);
  queued = azk_queue_triggered_effect(
      world,
      get_first_timed_ability(world, selis_cards[1], ecs_id(AWhenReturnedToHand)),
      players[1], TIMING_TAG_WHEN_RETURNED_TO_HAND);
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
  attach_ability_components(world, stt01_006_card);

  bool queued = azk_queue_triggered_effect(
      world,
      get_first_timed_ability(world, stt01_006_card, ecs_id(AWhenAttacking)),
      player0, TIMING_TAG_WHEN_ATTACKING);
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
  attach_ability_components(world, alley_thug);

  bool queued = azk_queue_triggered_effect(
      world,
      get_first_timed_ability(world, alley_thug, ecs_id(AWhenAttacking)),
      player0, TIMING_TAG_WHEN_ATTACKING);
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
  assert(!processed);
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

  bool triggered = azk_trigger_leader_response_ability(world, shao, player0, 0);
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

  azk_trigger_gate_portal_ability(world, gate_card, portaled_card, 0, player0);

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

static void
test_azk01_122_gate_portal_selects_hand_entity_and_grants_charge(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player0 = 0;
  PlayerZones zones0 = {0};
  setup_single_player_play_fixture(world, &player0, &zones0);

  ecs_entity_t selection =
      create_zone(world, player0, ZSelection, "Selection_P0");
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->zones[0].selection = selection;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_AZK01_122, .code = "AZK01-122"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player0);

  ecs_entity_t portaled_card = ecs_new(world);
  ecs_set(world, portaled_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, portaled_card, GatePoints, {.gate_points = 3});

  ecs_entity_t affordable = create_basic_entity_card(
      world, player0, zones0.hand, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-122_affordable", 0);
  ecs_set(world, affordable, IKZCost, {.ikz_cost = 3});

  ecs_entity_t too_expensive = create_basic_entity_card(
      world, player0, zones0.hand, CARD_DEF_STT03_004, CARD_ELEMENT_EARTH,
      "AZK01-122_too_expensive", 1);
  ecs_set(world, too_expensive, IKZCost, {.ikz_cost = 4});

  azk_trigger_gate_portal_ability(world, gate_card, portaled_card, 0, player0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  assert(ctx != NULL);
  assert(ctx->runtime.source_card == gate_card);
  assert(ctx->runtime.owner == player0);
  assert(ctx->scratch.kind == ABILITY_SCRATCH_GATE_PORTAL);
  assert(ctx->scratch.data.gate_portal.portaled_card == portaled_card);
  assert(ctx->selection.count == 1);
  assert(ctx->selection.cards[0] == affordable);
  assert(ecs_get_target(world, affordable, EcsChildOf, 0) == selection);
  assert(ecs_get_target(world, too_expensive, EcsChildOf, 0) == zones0.hand);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_garden_play = false;
  bool found_alley_play = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->subaction_1 != 0) {
      continue;
    }

    if (action->type == ACT_SELECT_TO_GARDEN && action->subaction_2 == 0) {
      found_garden_play = true;
    }
    if (action->type == ACT_SELECT_TO_ALLEY && action->subaction_2 == 0) {
      found_alley_play = true;
    }
  }
  assert(found_garden_play);
  assert(found_alley_play);

  ecs_defer_begin(world);
  bool selected = azk_process_selection_to_garden(world, 0, 0);
  ecs_defer_end(world);
  azk_finalize_pending_zone_move_logs(world);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  assert(ecs_get_target(world, affordable, EcsChildOf, 0) == zones0.garden);
  const ZoneIndex *garden_index = ecs_get(world, affordable, ZoneIndex);
  assert(garden_index != NULL);
  assert(garden_index->index == 0);
  assert(ecs_has(world, affordable, Charge));
  assert(!ecs_has(world, affordable, SacrificeAtEndOfTurn));

  const TapState *tap = ecs_get(world, affordable, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(!tap->cooldown);

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  bool found_keywords_changed = false;
  bool found_untapped = false;
  for (uint8_t i = 0; i < log_count; ++i) {
    if (logs[i].type == GLOG_CARD_KEYWORDS_CHANGED &&
        logs[i].data.keywords_changed.card.card_def_id ==
            CARD_DEF_STT03_003) {
      found_keywords_changed = true;
      assert(logs[i].data.keywords_changed.card.zone == GLOG_ZONE_GARDEN);
      assert(logs[i].data.keywords_changed.card.zone_index == 0);
      assert(logs[i].data.keywords_changed.has_charge);
    }

    if (logs[i].type == GLOG_CARD_TAP_STATE_CHANGED &&
        logs[i].data.tap_changed.card.card_def_id == CARD_DEF_STT03_003 &&
        logs[i].data.tap_changed.new_state == GLOG_TAP_UNTAPPED) {
      found_untapped = true;
      assert(logs[i].data.tap_changed.card.zone == GLOG_ZONE_GARDEN);
      assert(logs[i].data.tap_changed.card.zone_index == 0);
    }
  }
  assert(found_keywords_changed);
  assert(found_untapped);

  assert(ecs_get_target(world, too_expensive, EcsChildOf, 0) == zones0.hand);

  ecs_fini(world);
}

static void test_azk01_122_gate_portal_can_play_to_alley(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player0 = 0;
  PlayerZones zones0 = {0};
  setup_single_player_play_fixture(world, &player0, &zones0);

  ecs_entity_t selection =
      create_zone(world, player0, ZSelection, "Selection_P0");
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->zones[0].selection = selection;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_AZK01_122, .code = "AZK01-122"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player0);

  ecs_entity_t portaled_card = ecs_new(world);
  ecs_set(world, portaled_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, portaled_card, GatePoints, {.gate_points = 3});

  ecs_entity_t affordable = create_basic_entity_card(
      world, player0, zones0.hand, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-122_alley_target", 0);
  ecs_set(world, affordable, IKZCost, {.ikz_cost = 3});

  azk_trigger_gate_portal_ability(world, gate_card, portaled_card, 0, player0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  bool selected = azk_process_selection_to_alley(world, 0, 0);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  assert(ecs_get_target(world, affordable, EcsChildOf, 0) == zones0.alley);
  const ZoneIndex *alley_index = ecs_get(world, affordable, ZoneIndex);
  assert(alley_index != NULL);
  assert(alley_index->index == 0);
  assert(ecs_has(world, affordable, Charge));
  assert(!ecs_has(world, affordable, SacrificeAtEndOfTurn));

  const TapState *tap = ecs_get(world, affordable, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(!tap->cooldown);

  ecs_fini(world);
}

static void test_gate_portal_effect_selection_stt03_002(void) {
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

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].garden = garden0;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT03_002, .code = "STT03-002"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player0);

  ecs_entity_t eligible_target = ecs_new(world);
  ecs_set(world, eligible_target, CardId,
          {.id = CARD_DEF_STT03_003, .code = "ELIGIBLE"});
  ecs_set(world, eligible_target, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, eligible_target, BaseStats, {.attack = 1, .health = 2});
  ecs_add_pair(world, eligible_target, Rel_OwnedBy, player0);
  ecs_add_pair(world, eligible_target, EcsChildOf, garden0);
  ecs_set(world, eligible_target, ZoneIndex, {.index = 0});

  ecs_entity_t ineligible_target = ecs_new(world);
  ecs_set(world, ineligible_target, CardId,
          {.id = CARD_DEF_STT03_004, .code = "INELIGIBLE"});
  ecs_set(world, ineligible_target, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, ineligible_target, BaseStats, {.attack = 3, .health = 4});
  ecs_add_pair(world, ineligible_target, Rel_OwnedBy, player0);
  ecs_add_pair(world, ineligible_target, EcsChildOf, garden0);
  ecs_set(world, ineligible_target, ZoneIndex, {.index = 1});

  ecs_entity_t portaled_card = ecs_new(world);
  ecs_set(world, portaled_card, CardId,
          {.id = CARD_DEF_STT03_005, .code = "PORTALED"});
  ecs_set(world, portaled_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, portaled_card, GatePoints, {.gate_points = 2});

  azk_trigger_gate_portal_ability(world, gate_card, portaled_card, 0, player0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  assert(ctx != NULL);
  assert(ctx->runtime.source_card == gate_card);
  assert(ctx->runtime.owner == player0);
  assert(ctx->scratch.kind == ABILITY_SCRATCH_GATE_PORTAL);
  assert(ctx->scratch.data.gate_portal.portaled_card == portaled_card);

  bool selected = azk_process_effect_selection(world, 0);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_has(world, eligible_target, Defender));
  assert(!ecs_has(world, ineligible_target, Defender));

  ecs_fini(world);
}

static void test_gate_portal_no_valid_targets_auto_resolves_stt03_002(void) {
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

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->players[0] = player0;
  gs->players[1] = player1;
  gs->zones[0].garden = garden0;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT03_002, .code = "STT03-002"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player0);

  ecs_entity_t existing_defender = ecs_new(world);
  ecs_set(world, existing_defender, CardId,
          {.id = CARD_DEF_STT03_003, .code = "EXISTING_DEFENDER"});
  ecs_set(world, existing_defender, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, existing_defender, BaseStats, {.attack = 1, .health = 2});
  ecs_add(world, existing_defender, Defender);
  ecs_add_pair(world, existing_defender, Rel_OwnedBy, player0);
  ecs_add_pair(world, existing_defender, EcsChildOf, garden0);
  ecs_set(world, existing_defender, ZoneIndex, {.index = 0});

  ecs_entity_t too_big = ecs_new(world);
  ecs_set(world, too_big, CardId, {.id = CARD_DEF_STT03_004, .code = "TOO_BIG"});
  ecs_set(world, too_big, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, too_big, BaseStats, {.attack = 3, .health = 4});
  ecs_add_pair(world, too_big, Rel_OwnedBy, player0);
  ecs_add_pair(world, too_big, EcsChildOf, garden0);
  ecs_set(world, too_big, ZoneIndex, {.index = 1});

  ecs_entity_t portaled_card = ecs_new(world);
  ecs_set(world, portaled_card, CardId,
          {.id = CARD_DEF_STT03_005, .code = "PORTALED"});
  ecs_set(world, portaled_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, portaled_card, GatePoints, {.gate_points = 2});

  azk_trigger_gate_portal_ability(world, gate_card, portaled_card, 0, player0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_has(world, existing_defender, Defender));
  assert(!ecs_has(world, too_big, Defender));

  ecs_fini(world);
}

static void test_azk01_064_gate_portal_triggers_when_enters_garden(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player0 = 0;
  PlayerZones zones0 = {0};
  setup_single_player_play_fixture(world, &player0, &zones0);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t player1 = gs->players[1];

  ecs_entity_t gate_zone = create_zone(world, player0, ZGate, "Gate_P0");
  GameState *game_state = ecs_singleton_get_mut(world, GameState);
  assert(game_state != NULL);
  game_state->zones[0].gate = gate_zone;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set_name(world, gate_card, "STT02-002_gate");
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT02_002, .code = "STT02-002"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate_card, EcsChildOf, gate_zone);
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player0);
  initialize_test_card_runtime_components(world, gate_card);
  attach_ability_components(world, gate_card);

  ecs_entity_t zero = create_basic_entity_card(
      world, player0, zones0.alley, CARD_DEF_AZK01_064, CARD_ELEMENT_FIRE,
      "AZK01-064_zero", 0);
  ecs_set(world, zero, BaseStats, {.attack = 7, .health = 7});
  ecs_set(world, zero, CurStats, {.cur_atk = 7, .cur_hp = 7});
  ecs_set(world, zero, GatePoints, {.gate_points = 2});

  ecs_entity_t friendly_garden_entity = create_basic_entity_card(
      world, player0, zones0.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "friendly_garden_entity", 1);
  ecs_set(world, friendly_garden_entity, BaseStats, {.attack = 3, .health = 3});
  ecs_set(world, friendly_garden_entity, CurStats, {.cur_atk = 3, .cur_hp = 3});

  ecs_entity_t enemy_garden_entity = create_basic_entity_card(
      world, player1, gs->zones[1].garden, CARD_DEF_STT03_004,
      CARD_ELEMENT_EARTH, "enemy_garden_entity", 0);
  ecs_set(world, enemy_garden_entity, BaseStats, {.attack = 3, .health = 3});
  ecs_set(world, enemy_garden_entity, CurStats, {.cur_atk = 3, .cur_hp = 3});

  GatePortalIntent intent = {
      .player = player0,
      .alley_card = zero,
      .target_zone = zones0.garden,
      .garden_index = 0,
      .displaced_card = 0,
      .gate_card = gate_card,
  };

  int result = gate_card_into_garden(world, &intent);
  assert(result == 0);
  assert(ecs_get_target(world, zero, EcsChildOf, 0) == zones0.garden);

  const TriggeredEffectQueue *queue =
      ecs_singleton_get(world, TriggeredEffectQueue);
  assert(queue != NULL);
  assert(queue->count == 1);
  assert(queue->effects[0].source_card == zero);
  assert(queue->effects[0].timing_tag == TIMING_TAG_WHEN_ENTERS_GARDEN);

  azk_process_triggered_effect_queue(world);

  const CurStats *zero_stats = ecs_get(world, zero, CurStats);
  assert(zero_stats != NULL);
  assert(zero_stats->cur_hp == 5);

  const CurStats *friendly_stats = ecs_get(world, friendly_garden_entity, CurStats);
  assert(friendly_stats != NULL);
  assert(friendly_stats->cur_hp == 1);

  const CurStats *enemy_stats = ecs_get(world, enemy_garden_entity, CurStats);
  assert(enemy_stats != NULL);
  assert(enemy_stats->cur_hp == 1);

  ecs_fini(world);
}

static void test_azk01_064_gate_portal_damages_all_enemy_garden_slots(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player0 = 0;
  PlayerZones zones0 = {0};
  setup_single_player_play_fixture(world, &player0, &zones0);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t player1 = gs->players[1];
  ecs_entity_t enemy_garden = gs->zones[1].garden;
  ecs_entity_t enemy_discard = gs->zones[1].discard;

  ecs_entity_t gate_zone = create_zone(world, player0, ZGate, "Gate_P0");
  GameState *game_state = ecs_singleton_get_mut(world, GameState);
  assert(game_state != NULL);
  game_state->zones[0].gate = gate_zone;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set_name(world, gate_card, "STT02-002_gate");
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT02_002, .code = "STT02-002"});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate_card, EcsChildOf, gate_zone);
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player0);
  initialize_test_card_runtime_components(world, gate_card);
  attach_ability_components(world, gate_card);

  ecs_entity_t zero = create_basic_entity_card(
      world, player0, zones0.alley, CARD_DEF_AZK01_064, CARD_ELEMENT_FIRE,
      "AZK01-064_zero", 0);
  ecs_set(world, zero, BaseStats, {.attack = 7, .health = 7});
  ecs_set(world, zero, CurStats, {.cur_atk = 7, .cur_hp = 7});
  ecs_set(world, zero, GatePoints, {.gate_points = 2});

  ecs_entity_t enemy_slot0 = create_basic_entity_card(
      world, player1, enemy_garden, CARD_DEF_STT04_005, CARD_ELEMENT_FIRE,
      "enemy_slot0", 0);
  ecs_set(world, enemy_slot0, BaseStats, {.attack = 1, .health = 1});
  ecs_set(world, enemy_slot0, CurStats, {.cur_atk = 1, .cur_hp = 1});

  ecs_entity_t enemy_slot1 = create_basic_entity_card(
      world, player1, enemy_garden, CARD_DEF_STT04_010, CARD_ELEMENT_FIRE,
      "enemy_slot1_stt04_010", 1);
  ecs_set(world, enemy_slot1, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, enemy_slot1, CurStats, {.cur_atk = 1, .cur_hp = 2});
  ecs_add(world, enemy_slot1, Charge);

  ecs_entity_t enemy_slot2 = create_basic_entity_card(
      world, player1, enemy_garden, CARD_DEF_STT04_004, CARD_ELEMENT_FIRE,
      "enemy_slot2", 2);
  ecs_set(world, enemy_slot2, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, enemy_slot2, CurStats, {.cur_atk = 1, .cur_hp = 2});

  ecs_entity_t enemy_slot3 = create_basic_entity_card(
      world, player1, enemy_garden, CARD_DEF_STT04_008, CARD_ELEMENT_FIRE,
      "enemy_slot3", 3);
  ecs_set(world, enemy_slot3, BaseStats, {.attack = 2, .health = 2});
  ecs_set(world, enemy_slot3, CurStats, {.cur_atk = 2, .cur_hp = 2});

  GatePortalIntent intent = {
      .player = player0,
      .alley_card = zero,
      .target_zone = zones0.garden,
      .garden_index = 0,
      .displaced_card = 0,
      .gate_card = gate_card,
  };

  int result = gate_card_into_garden(world, &intent);
  assert(result == 0);
  azk_process_triggered_effect_queue(world);

  assert(ecs_get_target(world, enemy_slot0, EcsChildOf, 0) == enemy_discard);
  assert(ecs_get_target(world, enemy_slot1, EcsChildOf, 0) == enemy_discard);
  assert(ecs_get_target(world, enemy_slot2, EcsChildOf, 0) == enemy_discard);
  assert(ecs_get_target(world, enemy_slot3, EcsChildOf, 0) == enemy_discard);
  assert(ecs_get_ordered_children(world, enemy_garden).count == 0);

  ecs_fini(world);
}

static void test_main_phase_gate_portal_enters_effect_selection_stt03_002(void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t player0 = gs->players[0];
  ecs_entity_t gate_card = find_gate_card_in_zone(world, gs->zones[0].gate);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT03_002, .code = "STT03-002"});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});

  ecs_entity_t eligible_target = ecs_new(world);
  ecs_set(world, eligible_target, CardId,
          {.id = CARD_DEF_STT03_003, .code = "GARDEN_TARGET"});
  ecs_set(world, eligible_target, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, eligible_target, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, eligible_target, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, eligible_target, Rel_OwnedBy, player0);
  ecs_add_pair(world, eligible_target, EcsChildOf, gs->zones[0].garden);
  ecs_set(world, eligible_target, ZoneIndex, {.index = 1});

  ecs_entity_t existing_defender = ecs_new(world);
  ecs_set(world, existing_defender, CardId,
          {.id = CARD_DEF_STT03_005, .code = "EXISTING_DEFENDER"});
  ecs_set(world, existing_defender, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, existing_defender, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, existing_defender, TapState, {.tapped = false, .cooldown = false});
  ecs_add(world, existing_defender, Defender);
  ecs_add_pair(world, existing_defender, Rel_OwnedBy, player0);
  ecs_add_pair(world, existing_defender, EcsChildOf, gs->zones[0].garden);
  ecs_set(world, existing_defender, ZoneIndex, {.index = 2});

  ecs_entity_t alley_card = ecs_new(world);
  ecs_set(world, alley_card, CardId,
          {.id = CARD_DEF_STT03_004, .code = "ALLEY_CARD"});
  ecs_set(world, alley_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, alley_card, BaseStats, {.attack = 3, .health = 3});
  ecs_set(world, alley_card, GatePoints, {.gate_points = 2});
  ecs_set(world, alley_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, alley_card, Rel_OwnedBy, player0);
  ecs_add_pair(world, alley_card, EcsChildOf, gs->zones[0].alley);
  ecs_set(world, alley_card, ZoneIndex, {.index = 0});

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  ac->user_action = (UserAction){
      .player = player0,
      .type = ACT_GATE_PORTAL,
      .subaction_1 = 0,
      .subaction_2 = 0,
      .subaction_3 = 0,
  };
  ecs_singleton_modified(world, ActionContext);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);
  assert(ecs_get_target(world, alley_card, EcsChildOf, 0) == gs->zones[0].garden);
  const ZoneIndex *alley_card_zone_index = ecs_get(world, alley_card, ZoneIndex);
  assert(alley_card_zone_index != NULL);
  assert(alley_card_zone_index->index == 0);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_target = false;
  bool found_existing_defender = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_SELECT_EFFECT_TARGET && action->subaction_1 == 1) {
      found_target = true;
    }
    if (action->type == ACT_SELECT_EFFECT_TARGET && action->subaction_1 == 2) {
      found_existing_defender = true;
    }
  }
  assert(found_target);
  assert(!found_existing_defender);

  ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  ac->user_action = (UserAction){
      .player = player0,
      .type = ACT_SELECT_EFFECT_TARGET,
      .subaction_1 = 1,
      .subaction_2 = 0,
      .subaction_3 = 0,
  };
  ecs_singleton_modified(world, ActionContext);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_has(world, eligible_target, Defender));
  assert(ecs_has(world, existing_defender, Defender));

  azk_world_fini(world);
}

static void test_main_phase_gate_portal_can_target_portaled_card_stt03_002(void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t player0 = gs->players[0];
  ecs_entity_t gate_card = find_gate_card_in_zone(world, gs->zones[0].gate);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT03_002, .code = "STT03-002"});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});

  ecs_entity_t alley_card = ecs_new(world);
  ecs_set(world, alley_card, CardId,
          {.id = CARD_DEF_STT03_003, .code = "PORTALED_TARGET"});
  ecs_set(world, alley_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, alley_card, BaseStats, {.attack = 1, .health = 1});
  ecs_set(world, alley_card, GatePoints, {.gate_points = 1});
  ecs_set(world, alley_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, alley_card, Rel_OwnedBy, player0);
  ecs_add_pair(world, alley_card, EcsChildOf, gs->zones[0].alley);
  ecs_set(world, alley_card, ZoneIndex, {.index = 0});

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  ac->user_action = (UserAction){
      .player = player0,
      .type = ACT_GATE_PORTAL,
      .subaction_1 = 0,
      .subaction_2 = 0,
      .subaction_3 = 0,
  };
  ecs_singleton_modified(world, ActionContext);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);
  assert(ecs_get_target(world, alley_card, EcsChildOf, 0) == gs->zones[0].garden);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_portaled_card = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_SELECT_EFFECT_TARGET && action->subaction_1 == 0) {
      found_portaled_card = true;
    }
  }
  assert(found_portaled_card);

  ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  ac->user_action = (UserAction){
      .player = player0,
      .type = ACT_SELECT_EFFECT_TARGET,
      .subaction_1 = 0,
      .subaction_2 = 0,
      .subaction_3 = 0,
  };
  ecs_singleton_modified(world, ActionContext);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_has(world, alley_card, Defender));

  azk_world_fini(world);
}

static void test_main_phase_gate_portal_can_target_damaged_portaled_card_stt04_002(
    void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t player0 = gs->players[0];
  ecs_entity_t gate_card = find_gate_card_in_zone(world, gs->zones[0].gate);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT04_002, .code = "STT04-002"});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});

  ecs_entity_t alley_card = ecs_new(world);
  ecs_set(world, alley_card, CardId,
          {.id = CARD_DEF_STT04_010, .code = "STT04-010"});
  ecs_set(world, alley_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, alley_card, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, alley_card, CurStats, {.cur_atk = 1, .cur_hp = 2});
  ecs_set(world, alley_card, GatePoints, {.gate_points = 1});
  ecs_set(world, alley_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, alley_card, Rel_OwnedBy, player0);
  ecs_add_pair(world, alley_card, EcsChildOf, gs->zones[0].alley);
  ecs_set(world, alley_card, ZoneIndex, {.index = 0});

  bool damaged = deal_effect_damage(world, alley_card, 1);
  assert(damaged);

  const DamageTracker *pre_portal_tracker = ecs_get(world, alley_card, DamageTracker);
  assert(pre_portal_tracker != NULL);
  assert(pre_portal_tracker->took_damage_this_turn);

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  ac->user_action = (UserAction){
      .player = player0,
      .type = ACT_GATE_PORTAL,
      .subaction_1 = 0,
      .subaction_2 = 0,
      .subaction_3 = 0,
  };
  ecs_singleton_modified(world, ActionContext);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);
  assert(ecs_get_target(world, alley_card, EcsChildOf, 0) == gs->zones[0].garden);

  const DamageTracker *post_portal_tracker =
      ecs_get(world, alley_card, DamageTracker);
  assert(post_portal_tracker != NULL);
  assert(post_portal_tracker->took_damage_this_turn);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_portaled_card = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_SELECT_EFFECT_TARGET && action->subaction_1 == 0) {
      found_portaled_card = true;
    }
  }
  assert(found_portaled_card);

  ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  ac->user_action = (UserAction){
      .player = player0,
      .type = ACT_SELECT_EFFECT_TARGET,
      .subaction_1 = 0,
      .subaction_2 = 0,
      .subaction_3 = 0,
  };
  ecs_singleton_modified(world, ActionContext);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  const CurStats *portal_stats = ecs_get(world, alley_card, CurStats);
  assert(portal_stats != NULL);
  assert(portal_stats->cur_atk == 2);

  azk_world_fini(world);
}

static void
test_main_phase_gate_portal_cannot_target_entity_damaged_last_turn_stt04_002(
    void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  gs->turn_number = 1;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t player0 = gs->players[0];
  ecs_entity_t gate_card = find_gate_card_in_zone(world, gs->zones[0].gate);
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_STT04_002, .code = "STT04-002"});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});

  ecs_entity_t alley_card = ecs_new(world);
  ecs_set(world, alley_card, CardId,
          {.id = CARD_DEF_STT04_010, .code = "STT04-010"});
  ecs_set(world, alley_card, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, alley_card, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, alley_card, CurStats, {.cur_atk = 1, .cur_hp = 2});
  ecs_set(world, alley_card, GatePoints, {.gate_points = 1});
  ecs_set(world, alley_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, alley_card, Rel_OwnedBy, player0);
  ecs_add_pair(world, alley_card, EcsChildOf, gs->zones[0].alley);
  ecs_set(world, alley_card, ZoneIndex, {.index = 0});

  bool damaged = deal_effect_damage(world, alley_card, 1);
  assert(damaged);

  const DamageTracker *pre_turn_advance_tracker =
      ecs_get(world, alley_card, DamageTracker);
  assert(pre_turn_advance_tracker != NULL);
  assert(pre_turn_advance_tracker->took_damage_this_turn);
  assert(pre_turn_advance_tracker->turn_marker == 1);

  gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  const DamageTracker *stale_tracker = ecs_get(world, alley_card, DamageTracker);
  assert(stale_tracker != NULL);
  assert(stale_tracker->took_damage_this_turn);
  assert(stale_tracker->turn_marker == 1);
  assert(!azk_damage_tracker_is_current_turn(world, stale_tracker));

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  ac->user_action = (UserAction){
      .player = player0,
      .type = ACT_GATE_PORTAL,
      .subaction_1 = 0,
      .subaction_2 = 0,
      .subaction_3 = 0,
  };
  ecs_singleton_modified(world, ActionContext);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_get_target(world, alley_card, EcsChildOf, 0) == gs->zones[0].garden);
  const CurStats *portal_stats = ecs_get(world, alley_card, CurStats);
  assert(portal_stats != NULL);
  assert(portal_stats->cur_atk == 1);

  azk_world_fini(world);
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

static void test_start_phase_untaps_active_player_board_and_resources(void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);

  ecs_entity_t active_garden = create_basic_entity_card(
      world, gs->players[0], gs->zones[0].garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "start_phase_active_garden", 0);
  ecs_entity_t active_alley = create_basic_entity_card(
      world, gs->players[0], gs->zones[0].alley, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "start_phase_active_alley", 0);
  ecs_entity_t opponent_garden = create_basic_entity_card(
      world, gs->players[1], gs->zones[1].garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "start_phase_opponent_garden", 0);
  ecs_entity_t opponent_alley = create_basic_entity_card(
      world, gs->players[1], gs->zones[1].alley, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "start_phase_opponent_alley", 0);

  grant_ikz_cards_to_player(world, 0, 1);
  grant_ikz_cards_to_player(world, 1, 1);

  ecs_entity_t active_ikz =
      ecs_get_ordered_children(world, gs->zones[0].ikz_area).ids[0];
  ecs_entity_t opponent_ikz =
      ecs_get_ordered_children(world, gs->zones[1].ikz_area).ids[0];
  ecs_entity_t active_leader = find_leader_card_in_zone(world, gs->zones[0].leader);
  ecs_entity_t active_gate = find_gate_card_in_zone(world, gs->zones[0].gate);

  ecs_set(world, active_garden, TapState, {.tapped = true, .cooldown = true});
  ecs_set(world, active_alley, TapState, {.tapped = true, .cooldown = true});
  ecs_set(world, active_ikz, TapState, {.tapped = true, .cooldown = true});
  ecs_set(world, active_leader, TapState, {.tapped = true, .cooldown = true});
  ecs_set(world, active_gate, TapState, {.tapped = true, .cooldown = true});

  ecs_set(world, opponent_garden, TapState, {.tapped = true, .cooldown = true});
  ecs_set(world, opponent_alley, TapState, {.tapped = true, .cooldown = true});
  ecs_set(world, opponent_ikz, TapState, {.tapped = true, .cooldown = true});

  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = 0;
  gs->turn_number = 0;
  ecs_singleton_modified(world, GameState);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  const TapState *active_garden_tap = ecs_get(world, active_garden, TapState);
  const TapState *active_alley_tap = ecs_get(world, active_alley, TapState);
  const TapState *active_ikz_tap = ecs_get(world, active_ikz, TapState);
  const TapState *active_leader_tap = ecs_get(world, active_leader, TapState);
  const TapState *active_gate_tap = ecs_get(world, active_gate, TapState);
  const TapState *opponent_garden_tap = ecs_get(world, opponent_garden, TapState);
  const TapState *opponent_alley_tap = ecs_get(world, opponent_alley, TapState);
  const TapState *opponent_ikz_tap = ecs_get(world, opponent_ikz, TapState);

  assert(active_garden_tap != NULL);
  assert(active_alley_tap != NULL);
  assert(active_ikz_tap != NULL);
  assert(active_leader_tap != NULL);
  assert(active_gate_tap != NULL);
  assert(opponent_garden_tap != NULL);
  assert(opponent_alley_tap != NULL);
  assert(opponent_ikz_tap != NULL);

  assert(!active_garden_tap->tapped);
  assert(!active_garden_tap->cooldown);
  assert(!active_alley_tap->tapped);
  assert(!active_alley_tap->cooldown);
  assert(!active_ikz_tap->tapped);
  assert(!active_ikz_tap->cooldown);
  assert(!active_leader_tap->tapped);
  assert(!active_leader_tap->cooldown);
  assert(!active_gate_tap->tapped);
  assert(!active_gate_tap->cooldown);

  assert(opponent_garden_tap->tapped);
  assert(opponent_garden_tap->cooldown);
  assert(opponent_alley_tap->tapped);
  assert(opponent_alley_tap->cooldown);
  assert(opponent_ikz_tap->tapped);
  assert(opponent_ikz_tap->cooldown);

  azk_world_fini(world);
}

static void
test_start_phase_shocked_card_skips_next_owner_untap_after_manual_retap(void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);

  ecs_entity_t shocked_card = create_basic_entity_card(
      world, gs->players[0], gs->zones[0].garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "start_phase_shocked_skip_test", 0);
  ecs_set(world, shocked_card, TapState, {.tapped = true, .cooldown = false});
  apply_shocked(world, shocked_card, 1);

  ecs_set(world, shocked_card, TapState, {.tapped = false, .cooldown = false});
  ecs_set(world, shocked_card, TapState, {.tapped = true, .cooldown = false});

  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = 1;
  gs->turn_number = 1;
  ecs_singleton_modified(world, GameState);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  const TapState *after_opponent_start = ecs_get(world, shocked_card, TapState);
  const CardConditionCountdown *after_opponent_countdown =
      ecs_get(world, shocked_card, CardConditionCountdown);
  assert(after_opponent_start != NULL);
  assert(after_opponent_start->tapped);
  assert(ecs_has(world, shocked_card, Shocked));
  assert(after_opponent_countdown != NULL);
  assert(after_opponent_countdown->shocked_duration == 1);

  gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = 0;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  const TapState *after_owner_start = ecs_get(world, shocked_card, TapState);
  const CardConditionCountdown *after_owner_countdown =
      ecs_get(world, shocked_card, CardConditionCountdown);
  assert(after_owner_start != NULL);
  assert(after_owner_start->tapped);
  assert(!ecs_has(world, shocked_card, Shocked));
  assert(after_owner_countdown != NULL);
  assert(after_owner_countdown->shocked_duration == 0);

  azk_world_fini(world);
}

static void test_end_phase_resets_alley_entity_health(void) {
  ecs_world_t *world = azk_world_init_with_starting_player(42, 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);

  ecs_entity_t seer = create_basic_entity_card(
      world, gs->players[0], gs->zones[0].alley, CARD_DEF_STT04_003,
      CARD_ELEMENT_FIRE, "STT04-003_end_phase_reset", 0);
  ecs_set(world, seer, BaseStats, {.attack = 2, .health = 2});
  ecs_set(world, seer, CurStats, {.cur_atk = 2, .cur_hp = 1});

  gs->phase = PHASE_END_TURN;
  gs->active_player_index = 0;
  ecs_singleton_modified(world, GameState);

  run_phase_gate_system(world);
  ecs_progress(world, 0);

  const CurStats *seer_stats = ecs_get(world, seer, CurStats);
  assert(seer_stats != NULL);
  assert(seer_stats->cur_hp == 2);

  const GameState *final_gs = ecs_singleton_get(world, GameState);
  assert(final_gs != NULL);
  assert(final_gs->phase == PHASE_START_OF_TURN);
  assert(final_gs->active_player_index == 1);

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

static void test_stt03_001_blocks_second_main_activation_same_turn(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t leader = create_bobu_leader(world, player, zones.leader, 20);
  create_ikz_card(world, player, zones.ikz_area, "IKZ_1");
  create_ikz_card(world, player, zones.ikz_area, "IKZ_2");

  UserAction action = {
      .player = player,
      .type = ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
      .subaction_1 = GARDEN_SIZE,
  };

  ActivateAbilityIntent intent = {0};
  bool valid = azk_validate_activate_garden_or_leader_ability_action(
      world, ecs_singleton_get(world, GameState), player, &action, true,
      &intent);
  assert(valid);
  assert(intent.card == leader);
  assert(intent.ikz_card_count == 1);

  tap_card(world, intent.ikz_cards[0]);

  bool entered_phase = azk_trigger_main_ability(world, leader, player, 0);
  assert(!entered_phase);
  assert(!azk_is_in_ability_phase(world));

  ecs_entity_t leader_ability = azk_find_card_action_ability(world, leader, 0);
  assert(leader_ability != 0);
  const AbilityRepeatContext *repeat_ctx =
      ecs_get(world, leader_ability, AbilityRepeatContext);
  assert(repeat_ctx != NULL);
  assert(repeat_ctx->was_applied);

  ActivateAbilityIntent second_intent = {0};
  bool second_valid = azk_validate_activate_garden_or_leader_ability_action(
      world, ecs_singleton_get(world, GameState), player, &action, false,
      &second_intent);
  assert(!second_valid);

  ecs_fini(world);
}

static void test_stt03_001_heals_only_once_for_first_destroy_or_sacrifice(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t leader = create_bobu_leader(world, player, zones.leader, 17);
  ecs_entity_t garden_entity =
      create_earth_entity_card(world, player, zones.garden, "EarthGarden");
  ecs_entity_t alley_entity =
      create_earth_entity_card(world, player, zones.alley, "EarthAlley");

  bool entered_phase = azk_trigger_main_ability(world, leader, player, 0);
  assert(!entered_phase);
  assert(!azk_is_in_ability_phase(world));

  const STT03BobuState *active_state = ecs_get(world, leader, STT03BobuState);
  assert(active_state != NULL);
  assert(active_state->expires_turn == 3);

  discard_card(world, garden_entity);

  const CurStats *leader_stats = ecs_get(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 18);

  const STT03BobuState *spent_state = ecs_get(world, leader, STT03BobuState);
  assert(spent_state != NULL);
  assert(spent_state->expires_turn == 0);

  sacrifice_card(world, alley_entity);

  leader_stats = ecs_get(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 18);

  ecs_fini(world);
}

static void test_stt03_012_heals_when_another_entity_is_destroyed_on_opponents_turn(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");
  ecs_entity_t miharu = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_012, CARD_ELEMENT_EARTH,
      "STT03-012_test", 0);
  ecs_entity_t other_entity = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "OtherEntity_test", 1);
  (void)miharu;

  ecs_entity_t leader = find_leader_card_in_zone(world, zones.leader);
  assert(leader != 0);
  CurStats *leader_stats = ecs_get_mut(world, leader, CurStats);
  assert(leader_stats != NULL);
  leader_stats->cur_hp = 16;
  ecs_modified(world, leader, CurStats);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  discard_card(world, other_entity);

  leader_stats = ecs_get_mut(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 18);

  ecs_fini(world);
}

static void test_stt03_012_heals_when_itself_is_destroyed_on_opponents_main_turn(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");
  ecs_entity_t miharu = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_012, CARD_ELEMENT_EARTH,
      "STT03-012_test", 0);

  ecs_entity_t leader = find_leader_card_in_zone(world, zones.leader);
  assert(leader != 0);
  CurStats *leader_stats = ecs_get_mut(world, leader, CurStats);
  assert(leader_stats != NULL);
  leader_stats->cur_hp = 16;
  ecs_modified(world, leader, CurStats);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  discard_card(world, miharu);

  leader_stats = ecs_get_mut(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 18);

  ecs_fini(world);
}

static void test_stt03_012_heals_when_itself_is_destroyed_during_response_window(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");
  ecs_entity_t miharu = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_012, CARD_ELEMENT_EARTH,
      "STT03-012_test", 0);

  ecs_entity_t leader = find_leader_card_in_zone(world, zones.leader);
  assert(leader != 0);
  CurStats *leader_stats = ecs_get_mut(world, leader, CurStats);
  assert(leader_stats != NULL);
  leader_stats->cur_hp = 16;
  ecs_modified(world, leader, CurStats);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 0;
  gs->phase = PHASE_RESPONSE_WINDOW;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  discard_card(world, miharu);

  leader_stats = ecs_get_mut(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 18);

  ecs_fini(world);
}

static void test_response_window_opens_for_hand_response_entity_azk01_035(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t defender = 0;
  PlayerZones defender_zones = {0};
  setup_pending_attack_response_fixture(world, &defender, &defender_zones);

  ecs_entity_t response_entity = ecs_new(world);
  ecs_set_name(world, response_entity, "AZK01-035_response_entity");
  ecs_set(world, response_entity, CardId,
          {.id = CARD_DEF_AZK01_035, .code = "AZK01-035"});
  ecs_set(world, response_entity, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, response_entity, Element, {.element = CARD_ELEMENT_LIGHTNING});
  ecs_set(world, response_entity, BaseStats, {.attack = 2, .health = 2});
  ecs_set(world, response_entity, CurStats, {.cur_atk = 2, .cur_hp = 2});
  ecs_set(world, response_entity, IKZCost, {.ikz_cost = 1});
  ecs_set(world, response_entity, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, response_entity, Rel_OwnedBy, defender);
  ecs_add_pair(world, response_entity, EcsChildOf, defender_zones.hand);
  initialize_test_card_runtime_components(world, response_entity);
  attach_ability_components(world, response_entity);
  assert(azk_can_play_card_from_hand_during_response_window(world,
                                                            response_entity));

  const GameState *before = ecs_singleton_get(world, GameState);
  assert(before != NULL);
  assert(defender_can_respond(world, before, 0));

  run_phase_gate_system(world);

  const GameState *after = ecs_singleton_get(world, GameState);
  assert(after != NULL);
  assert(after->phase == PHASE_RESPONSE_WINDOW);
  assert(after->active_player_index == 0);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(world, after, 0, &mask);
  assert(built);

  bool found_response_entity_play = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_PLAY_ENTITY_TO_GARDEN ||
        action->type == ACT_PLAY_ENTITY_TO_ALLEY) {
      found_response_entity_play = true;
      break;
    }
  }
  assert(found_response_entity_play);

  ecs_fini(world);
}

static void test_response_window_opens_for_response_weapon_azk01_094(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t defender = 0;
  PlayerZones defender_zones = {0};
  setup_pending_attack_response_fixture(world, &defender, &defender_zones);

  ecs_entity_t response_weapon = ecs_new(world);
  ecs_set_name(world, response_weapon, "AZK01-094_response_weapon");
  ecs_set(world, response_weapon, CardId,
          {.id = CARD_DEF_AZK01_094, .code = "AZK01-094"});
  ecs_set(world, response_weapon, Type, {.value = CARD_TYPE_WEAPON});
  ecs_set(world, response_weapon, Element, {.element = CARD_ELEMENT_LIGHTNING});
  ecs_set(world, response_weapon, BaseStats, {.attack = 1, .health = 0});
  ecs_set(world, response_weapon, CurStats, {.cur_atk = 1, .cur_hp = 0});
  ecs_set(world, response_weapon, IKZCost, {.ikz_cost = 1});
  ecs_set(world, response_weapon, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, response_weapon, Rel_OwnedBy, defender);
  ecs_add_pair(world, response_weapon, EcsChildOf, defender_zones.hand);
  initialize_test_card_runtime_components(world, response_weapon);
  attach_ability_components(world, response_weapon);
  assert(azk_can_play_card_from_hand_during_response_window(world,
                                                            response_weapon));

  const GameState *before = ecs_singleton_get(world, GameState);
  assert(before != NULL);
  assert(defender_can_respond(world, before, 0));

  run_phase_gate_system(world);

  const GameState *after = ecs_singleton_get(world, GameState);
  assert(after != NULL);
  assert(after->phase == PHASE_RESPONSE_WINDOW);
  assert(after->active_player_index == 0);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(world, after, 0, &mask);
  assert(built);

  bool found_response_weapon_attach = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_ATTACH_WEAPON_FROM_HAND) {
      found_response_weapon_attach = true;
      break;
    }
  }
  assert(found_response_weapon_attach);

  ecs_fini(world);
}

static void
test_response_window_does_not_allow_playing_azk01_070_from_hand(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t defender = 0;
  PlayerZones defender_zones = {0};
  setup_single_player_play_fixture(world, &defender, &defender_zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t attacker = gs_ro->players[1];
  PlayerZones attacker_zones = gs_ro->zones[1];

  create_basic_leader(world, defender, defender_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "AZK01-070_Leader_Test");
  create_basic_leader(world, attacker, attacker_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "AZK01-070_Opponent_Leader_Test");

  create_basic_entity_card(world, defender, defender_zones.hand,
                           CARD_DEF_AZK01_070, CARD_ELEMENT_NORMAL,
                           "AZK01-070_Hand_Test", 0);
  create_basic_entity_card(world, defender, defender_zones.garden,
                           CARD_DEF_AZK01_070, CARD_ELEMENT_NORMAL,
                           "AZK01-070_Garden_Test", 0);

  GameState response_state = *gs_ro;
  response_state.phase = PHASE_RESPONSE_WINDOW;
  response_state.active_player_index = 0;

  UserAction play_action = {
      .player = defender,
      .type = ACT_PLAY_ENTITY_TO_GARDEN,
      .subaction_1 = 0,
      .subaction_2 = 1,
      .subaction_3 = 0,
  };
  bool can_play_from_hand = azk_validate_play_entity_action(
      world, &response_state, defender, ZONE_GARDEN, &play_action, false, NULL);
  assert(!can_play_from_hand);

  UserAction ability_action = {
      .player = defender,
      .type = ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
      .subaction_1 = 0,
  };
  bool can_activate_response = azk_validate_activate_garden_or_leader_ability_action(
      world, &response_state, defender, &ability_action, false, NULL);
  assert(can_activate_response);

  ecs_fini(world);
}

static void test_alley_attack_opens_response_window_and_resolves_combat(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});
  init_main_phase_system(world);
  init_response_phase_system(world);
  init_combat_resolve_phase_system(world);

  ecs_entity_t defender_player = 0;
  PlayerZones defender_zones = {0};
  setup_single_player_play_fixture(world, &defender_player, &defender_zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t attacker_player = gs_ro->players[1];
  PlayerZones attacker_zones = gs_ro->zones[1];

  create_basic_leader(world, defender_player, defender_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "AlleyAttack_DefenderLeader_Test");
  create_basic_leader(world, attacker_player, attacker_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "AlleyAttack_AttackerLeader_Test");

  ecs_entity_t attacker = create_basic_entity_card(
      world, attacker_player, attacker_zones.garden, CARD_DEF_AZK01_037,
      CARD_ELEMENT_LIGHTNING, "AlleyAttack_Attacker_Test", 0);
  ecs_add(world, attacker, AttrCanTargetTappedAndUntappedAlley);
  ecs_set(world, attacker, BaseStats, {.attack = 3, .health = 2});
  ecs_set(world, attacker, CurStats, {.cur_atk = 3, .cur_hp = 2});
  ecs_set(world, attacker, TapState, {.tapped = false, .cooldown = false});

  ecs_entity_t alley_target = create_basic_entity_card(
      world, defender_player, defender_zones.alley, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AlleyAttack_Target_Test", 0);
  ecs_set(world, alley_target, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, alley_target, CurStats, {.cur_atk = 1, .cur_hp = 2});
  ecs_set(world, alley_target, TapState, {.tapped = false, .cooldown = false});

  create_basic_weapon_card(world, defender_player, defender_zones.hand,
                           CARD_DEF_AZK01_094, CARD_ELEMENT_LIGHTNING,
                           "AlleyAttack_ResponseWeapon_Test");
  create_ikz_card(world, defender_player, defender_zones.ikz_area,
                  "AlleyAttack_ResponseIKZ_Test");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  *ac = (ActionContext){
      .user_action =
          {
              .player = attacker_player,
              .type = ACT_ATTACK,
              .subaction_1 = 0,
              .subaction_2 = GARDEN_SIZE + 1,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t main_system = ecs_lookup(world, "MainPhaseSystem");
  assert(main_system != 0);
  ecs_run(world, main_system, 0, NULL);

  const GameState *after_attack = ecs_singleton_get(world, GameState);
  assert(after_attack != NULL);
  assert(after_attack->phase == PHASE_RESPONSE_WINDOW);
  assert(after_attack->active_player_index == 0);
  assert(after_attack->combat_state.attacking_card == attacker);
  assert(after_attack->combat_state.defender_card == alley_target);

  *ac = (ActionContext){
      .user_action =
          {
              .player = defender_player,
              .type = ACT_NOOP,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t response_system = ecs_lookup(world, "ResponsePhaseSystem");
  assert(response_system != 0);
  ecs_run(world, response_system, 0, NULL);

  const GameState *after_pass = ecs_singleton_get(world, GameState);
  assert(after_pass != NULL);
  assert(after_pass->phase == PHASE_COMBAT_RESOLVE);
  assert(after_pass->active_player_index == 1);
  assert(after_pass->combat_state.attacking_card == attacker);
  assert(after_pass->combat_state.defender_card == alley_target);

  ecs_entity_t combat_system = ecs_lookup(world, "CombatResolvePhaseSystem");
  assert(combat_system != 0);
  ecs_run(world, combat_system, 0, NULL);

  const GameState *after_combat = ecs_singleton_get(world, GameState);
  assert(after_combat != NULL);
  assert(after_combat->phase == PHASE_MAIN);
  assert(after_combat->combat_state.attacking_card == 0);
  assert(after_combat->combat_state.defender_card == 0);
  assert(after_combat->last_combat.attacker == attacker);
  assert(after_combat->last_combat.defender == alley_target);
  assert(after_combat->last_combat.defender_destroyed);
  assert(after_combat->last_combat.damage_to_defender == 3);
  assert(after_combat->last_combat.damage_to_attacker == 1);

  const CurStats *attacker_stats = ecs_get(world, attacker, CurStats);
  assert(attacker_stats != NULL);
  assert(attacker_stats->cur_hp == 1);
  assert(ecs_get_target(world, alley_target, EcsChildOf, 0) ==
         defender_zones.discard);

  ecs_fini(world);
}

static void test_declare_defender_taps_the_defending_entity(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});
  init_response_phase_system(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");
  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P1_Test");
  ecs_entity_t attacker = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "Attacker_test", 0);
  ecs_entity_t original_target = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "OriginalTarget_test", 0);
  ecs_entity_t defender = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT02_006, CARD_ELEMENT_WATER,
      "Defender_test", 1);
  ecs_add(world, defender, Defender);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 0;
  gs->phase = PHASE_RESPONSE_WINDOW;
  gs->turn_number = 2;
  gs->combat_state.attacking_card = attacker;
  gs->combat_state.defender_card = original_target;
  gs->combat_state.defender_intercepted = false;
  ecs_singleton_modified(world, GameState);

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  *ac = (ActionContext){
      .user_action =
          {
              .player = player,
              .type = ACT_DECLARE_DEFENDER,
              .subaction_1 = 1,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t response_system = ecs_lookup(world, "ResponsePhaseSystem");
  assert(response_system != 0);
  ecs_run(world, response_system, 0, NULL);

  const GameState *after = ecs_singleton_get(world, GameState);
  assert(after != NULL);
  assert(after->combat_state.defender_card == defender);
  assert(after->combat_state.defender_intercepted);
  assert(is_card_tapped(world, defender));

  ecs_fini(world);
}

static void test_when_attacked_uses_final_defender_after_response_window(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});
  init_main_phase_system(world);
  init_response_phase_system(world);

  ecs_entity_t defender_player = 0;
  PlayerZones defender_zones = {0};
  setup_single_player_play_fixture(world, &defender_player, &defender_zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t attacker_player = gs_ro->players[1];
  PlayerZones attacker_zones = gs_ro->zones[1];

  create_basic_leader(world, defender_player, defender_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "Leader_P0_WhenAttacked_Test");
  create_basic_leader(world, attacker_player, attacker_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "Leader_P1_WhenAttacked_Test");
  ecs_entity_t attacker = create_basic_entity_card(
      world, attacker_player, attacker_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "WhenAttacked_Attacker_Test", 0);
  ecs_entity_t denmu = create_basic_entity_card(
      world, defender_player, defender_zones.garden, CARD_DEF_AZK01_036,
      CARD_ELEMENT_LIGHTNING, "WhenAttacked_Denmu_Test", 0);
  ecs_set(world, denmu, TapState, {.tapped = true, .cooldown = false});
  ecs_entity_t interceptor = create_basic_entity_card(
      world, defender_player, defender_zones.garden, CARD_DEF_STT02_006,
      CARD_ELEMENT_WATER, "WhenAttacked_Interceptor_Test", 1);
  ecs_add(world, interceptor, Defender);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  *ac = (ActionContext){
      .user_action =
          {
              .player = attacker_player,
              .type = ACT_ATTACK,
              .subaction_1 = 0,
              .subaction_2 = 0,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t main_system = ecs_lookup(world, "MainPhaseSystem");
  assert(main_system != 0);
  ecs_run(world, main_system, 0, NULL);

  const GameState *after_attack = ecs_singleton_get(world, GameState);
  assert(after_attack != NULL);
  assert(after_attack->phase == PHASE_RESPONSE_WINDOW);
  assert(after_attack->active_player_index == 0);
  assert(after_attack->combat_state.attacking_card == attacker);
  assert(after_attack->combat_state.defender_card == denmu);
  assert(!azk_has_queued_triggered_effects(world));
  assert(!ecs_has(world, attacker, Shocked));

  *ac = (ActionContext){
      .user_action =
          {
              .player = defender_player,
              .type = ACT_DECLARE_DEFENDER,
              .subaction_1 = 1,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t response_system = ecs_lookup(world, "ResponsePhaseSystem");
  assert(response_system != 0);
  ecs_run(world, response_system, 0, NULL);

  const GameState *after_defender = ecs_singleton_get(world, GameState);
  assert(after_defender != NULL);
  assert(after_defender->combat_state.defender_card == interceptor);

  *ac = (ActionContext){
      .user_action =
          {
              .player = defender_player,
              .type = ACT_NOOP,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_run(world, response_system, 0, NULL);

  const GameState *after_pass = ecs_singleton_get(world, GameState);
  assert(after_pass != NULL);
  assert(after_pass->phase == PHASE_COMBAT_RESOLVE);
  assert(after_pass->active_player_index == 1);
  assert(after_pass->combat_state.defender_card == interceptor);
  assert(!azk_has_queued_triggered_effects(world));
  assert(!ecs_has(world, attacker, Shocked));

  ecs_fini(world);
}

static void test_when_attacked_queues_at_combat_handoff_without_response(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});
  init_main_phase_system(world);

  ecs_entity_t defender_player = 0;
  PlayerZones defender_zones = {0};
  setup_single_player_play_fixture(world, &defender_player, &defender_zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t attacker_player = gs_ro->players[1];
  PlayerZones attacker_zones = gs_ro->zones[1];

  create_basic_leader(world, defender_player, defender_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "Leader_P0_WhenAttacked_NoResponse_Test");
  create_basic_leader(world, attacker_player, attacker_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "Leader_P1_WhenAttacked_NoResponse_Test");
  ecs_entity_t attacker = create_basic_entity_card(
      world, attacker_player, attacker_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "WhenAttacked_NoResponse_Attacker_Test", 0);
  ecs_entity_t denmu = create_basic_entity_card(
      world, defender_player, defender_zones.garden, CARD_DEF_AZK01_036,
      CARD_ELEMENT_LIGHTNING, "WhenAttacked_NoResponse_Denmu_Test", 0);
  ecs_set(world, denmu, TapState, {.tapped = true, .cooldown = false});

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  *ac = (ActionContext){
      .user_action =
          {
              .player = attacker_player,
              .type = ACT_ATTACK,
              .subaction_1 = 0,
              .subaction_2 = 0,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t main_system = ecs_lookup(world, "MainPhaseSystem");
  assert(main_system != 0);
  ecs_run(world, main_system, 0, NULL);

  const GameState *after_attack = ecs_singleton_get(world, GameState);
  assert(after_attack != NULL);
  assert(after_attack->phase == PHASE_COMBAT_RESOLVE);
  assert(after_attack->active_player_index == 1);
  assert(azk_has_queued_triggered_effects(world));
  assert(!ecs_has(world, attacker, Shocked));

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);
  assert(!azk_has_queued_triggered_effects(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_has(world, attacker, Shocked));

  const CardConditionCountdown *countdown =
      ecs_get(world, attacker, CardConditionCountdown);
  assert(countdown != NULL);
  assert(countdown->shocked_duration == 1);

  ecs_fini(world);
}

static void
test_when_attacked_target_selection_runs_before_combat_resolution(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});
  init_main_phase_system(world);

  ecs_entity_t defender_player = 0;
  PlayerZones defender_zones = {0};
  setup_single_player_play_fixture(world, &defender_player, &defender_zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t attacker_player = gs_ro->players[1];
  PlayerZones attacker_zones = gs_ro->zones[1];

  create_basic_leader(world, defender_player, defender_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "Leader_P0_WhenAttacked_Target_Test");
  create_basic_leader(world, attacker_player, attacker_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "Leader_P1_WhenAttacked_Target_Test");
  create_basic_entity_card(world, attacker_player, attacker_zones.garden,
                           CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
                           "WhenAttacked_Target_Attacker_Test", 0);
  ecs_entity_t vault_master = create_basic_entity_card(
      world, defender_player, defender_zones.garden, CARD_DEF_AZK01_040,
      CARD_ELEMENT_LIGHTNING, "WhenAttacked_Target_VaultMaster_Test", 0);
  ecs_set(world, vault_master, TapState, {.tapped = true, .cooldown = false});

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  *ac = (ActionContext){
      .user_action =
          {
              .player = attacker_player,
              .type = ACT_ATTACK,
              .subaction_1 = 0,
              .subaction_2 = 0,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t main_system = ecs_lookup(world, "MainPhaseSystem");
  assert(main_system != 0);
  ecs_run(world, main_system, 0, NULL);

  const GameState *after_attack = ecs_singleton_get(world, GameState);
  assert(after_attack != NULL);
  assert(after_attack->phase == PHASE_COMBAT_RESOLVE);
  assert(after_attack->active_player_index == 1);
  assert(azk_has_queued_triggered_effects(world));

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const GameState *during_ability = ecs_singleton_get(world, GameState);
  assert(during_ability != NULL);
  assert(during_ability->phase == PHASE_COMBAT_RESOLVE);
  assert(during_ability->active_player_index == 0);

  bool skipped = azk_process_effect_skip(world);
  assert(skipped);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(!azk_has_queued_triggered_effects(world));

  const GameState *after_skip = ecs_singleton_get(world, GameState);
  assert(after_skip != NULL);
  assert(after_skip->phase == PHASE_COMBAT_RESOLVE);
  assert(after_skip->active_player_index == 1);

  ecs_fini(world);
}

static void test_azk01_006_return_to_hand_fizzles_combat(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  init_combat_resolve_phase_system(world);

  ecs_entity_t defender_player = 0;
  PlayerZones defender_zones = {0};
  setup_single_player_play_fixture(world, &defender_player, &defender_zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t attacker_player = gs_ro->players[1];
  PlayerZones attacker_zones = gs_ro->zones[1];

  create_basic_leader(world, defender_player, defender_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "AZK01-006_Leader_Test");
  create_basic_leader(world, attacker_player, attacker_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "AZK01-006_Opponent_Leader_Test");
  ecs_entity_t attacker = create_basic_entity_card(
      world, attacker_player, attacker_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-006_Attacker_Test", 0);
  ecs_set(world, attacker, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, attacker, CurStats, {.cur_atk = 1, .cur_hp = 2});

  ecs_entity_t gus = create_basic_entity_card(
      world, defender_player, defender_zones.garden, CARD_DEF_AZK01_006,
      CARD_ELEMENT_NORMAL, "AZK01-006_Gus_Test", 0);
  ecs_set(world, gus, BaseStats, {.attack = 1, .health = 1});
  ecs_set(world, gus, CurStats, {.cur_atk = 1, .cur_hp = 1});
  ecs_set(world, gus, TapState, {.tapped = true, .cooldown = false});

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 1;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  AttackIntent attack_intent = {
      .attacking_player = attacker_player,
      .defending_player = defender_player,
      .attacking_card = attacker,
      .defending_card = gus,
      .attacker_index = 0,
      .defender_index = 0,
      .attacker_is_leader = false,
  };
  assert(attack(world, &attack_intent) == 0);
  assert(azk_transition_to_combat_resolve(world));

  const GameState *after_attack = ecs_singleton_get(world, GameState);
  assert(after_attack != NULL);
  assert(after_attack->phase == PHASE_COMBAT_RESOLVE);
  assert(after_attack->active_player_index == 1);
  assert(azk_has_queued_triggered_effects(world));

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);

  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_get_target(world, gus, EcsChildOf, 0) == defender_zones.hand);

  ecs_entity_t combat_system = ecs_lookup(world, "CombatResolvePhaseSystem");
  assert(combat_system != 0);
  ecs_run(world, combat_system, 0, NULL);

  const GameState *after_combat = ecs_singleton_get(world, GameState);
  assert(after_combat != NULL);
  assert(after_combat->phase == PHASE_MAIN);
  assert(after_combat->combat_state.attacking_card == 0);
  assert(after_combat->combat_state.defender_card == 0);
  assert(after_combat->last_combat.attacker == 0);
  assert(after_combat->last_combat.defender == 0);

  const CurStats *attacker_stats = ecs_get(world, attacker, CurStats);
  assert(attacker_stats != NULL);
  assert(attacker_stats->cur_hp == 2);

  ecs_fini(world);
}

static void test_stt03_006_destroyed_draw_then_discard_uses_hand_targets(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");

  const AbilityDef *def = azk_get_ability_def(CARD_DEF_STT03_006);
  assert(def != NULL);
  assert(def->effect_req.type == ABILITY_TARGET_FRIENDLY_HAND);

  ecs_entity_t hand_card =
      create_earth_entity_card(world, player, zones.hand, "HandCard");
  ecs_entity_t deck_card_keep =
      create_earth_entity_card(world, player, zones.deck, "DeckCardKeep");
  ecs_entity_t deck_card =
      create_earth_entity_card(world, player, zones.deck, "DeckCard");

  ecs_entity_t cactus_farmer = ecs_new(world);
  ecs_set_name(world, cactus_farmer, "STT03-006_test");
  ecs_set(world, cactus_farmer, CardId, {.id = CARD_DEF_STT03_006});
  ecs_set(world, cactus_farmer, Type, {.value = CARD_TYPE_ENTITY});
  ecs_set(world, cactus_farmer, Element, {.element = CARD_ELEMENT_EARTH});
  ecs_set(world, cactus_farmer, BaseStats, {.attack = 2, .health = 1});
  ecs_set(world, cactus_farmer, CurStats, {.cur_atk = 2, .cur_hp = 1});
  ecs_set(world, cactus_farmer, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, cactus_farmer, EcsChildOf, zones.garden);
  ecs_add_pair(world, cactus_farmer, Rel_OwnedBy, player);

  discard_card(world, cactus_farmer);
  assert(azk_has_queued_triggered_effects(world));

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  ecs_entities_t hand_cards = ecs_get_ordered_children(world, zones.hand);
  assert(hand_cards.count == 2);
  assert(hand_cards.ids[0] == hand_card);
  assert(hand_cards.ids[1] == deck_card);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_hand_target0 = false;
  bool found_hand_target1 = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }
    if (action->subaction_1 == 0) {
      found_hand_target0 = true;
    }
    if (action->subaction_1 == 1) {
      found_hand_target1 = true;
    }
  }

  assert(found_hand_target0);
  assert(found_hand_target1);

  bool selected = azk_process_effect_selection(world, 1);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  hand_cards = ecs_get_ordered_children(world, zones.hand);
  ecs_entities_t discard_cards = ecs_get_ordered_children(world, zones.discard);
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, zones.deck);
  assert(hand_cards.count == 1);
  assert(hand_cards.ids[0] == hand_card);
  assert(deck_cards.count == 1);
  assert(deck_cards.ids[0] == deck_card_keep);
  assert(discard_cards.count == 2);
  assert(discard_cards.ids[0] == cactus_farmer);
  assert(discard_cards.ids[1] == deck_card);

  ecs_fini(world);
}

static void test_stt04_001_effect_selection_accepts_garden_and_alley_targets(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t leader = create_basic_leader(
      world, player, zones.leader, CARD_DEF_STT04_001, CARD_ELEMENT_FIRE,
      "STT04-001_test");
  ecs_entity_t garden_target = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_FIRE,
      "GardenTarget", 0);
  ecs_entity_t alley_target = create_basic_entity_card(
      world, player, zones.alley, CARD_DEF_STT03_003, CARD_ELEMENT_FIRE,
      "AlleyTarget", 0);

  ecs_set(world, garden_target, BaseStats, {.attack = 2, .health = 2});
  ecs_set(world, garden_target, CurStats, {.cur_atk = 2, .cur_hp = 2});
  ecs_set(world, alley_target, BaseStats, {.attack = 2, .health = 2});
  ecs_set(world, alley_target, CurStats, {.cur_atk = 2, .cur_hp = 2});

  const AbilityDef *def = azk_get_ability_def(CARD_DEF_STT04_001);
  assert(def != NULL);
  assert(def->effect_req.type == ABILITY_TARGET_FRIENDLY_GARDEN_OR_ALLEY_ENTITY);

  bool entered_phase = azk_trigger_main_ability(world, leader, player, 0);
  assert(entered_phase);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const CurStats *leader_after_cost = ecs_get(world, leader, CurStats);
  assert(leader_after_cost != NULL);
  assert(leader_after_cost->cur_hp == 19);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_garden_target = false;
  bool found_alley_target = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }
    if (action->subaction_1 == 0) {
      found_garden_target = true;
    }
    if (action->subaction_1 == GARDEN_SIZE) {
      found_alley_target = true;
    }
  }

  assert(found_garden_target);
  assert(found_alley_target);

  ecs_entity_t resolved_garden = azk_resolve_ability_target_choice_entity(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, player, 0);
  ecs_entity_t resolved_alley = azk_resolve_ability_target_choice_entity(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, player, GARDEN_SIZE);
  assert(resolved_garden == garden_target);
  assert(resolved_alley == alley_target);

  bool selected = azk_process_effect_selection(world, GARDEN_SIZE);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *garden_stats = ecs_get(world, garden_target, CurStats);
  const CurStats *alley_stats = ecs_get(world, alley_target, CurStats);
  const CurStats *leader_stats = ecs_get(world, leader, CurStats);
  assert(garden_stats != NULL);
  assert(alley_stats != NULL);
  assert(leader_stats != NULL);
  assert(garden_stats->cur_hp == 2);
  assert(garden_stats->cur_atk == 2);
  assert(alley_stats->cur_hp == 1);
  assert(alley_stats->cur_atk == 3);
  assert(leader_stats->cur_hp == 19);

  ecs_fini(world);
}

static void test_stt04_017_cost_selection_allows_fifth_garden_sacrifice(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT04_001,
                      CARD_ELEMENT_FIRE, "STT04-017_Leader_P0");
  create_basic_leader(world, opponent, opponent_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "STT04-017_Leader_P1");

  ecs_entity_t spell = ecs_new(world);
  ecs_set_name(world, spell, "STT04-017_spell");
  ecs_set(world, spell, CardId, {.id = CARD_DEF_STT04_017, .code = "STT04-017"});
  ecs_set(world, spell, Type, {.value = CARD_TYPE_SPELL});
  ecs_set(world, spell, Element, {.element = CARD_ELEMENT_FIRE});
  ecs_set(world, spell, IKZCost, {.ikz_cost = 3});
  ecs_add_pair(world, spell, Rel_OwnedBy, player);
  ecs_add_pair(world, spell, EcsChildOf, zones.hand);
  initialize_test_card_runtime_components(world, spell);
  attach_ability_components(world, spell);

  for (uint8_t i = 0; i < GARDEN_SIZE; ++i) {
    char name[64];
    snprintf(name, sizeof(name), "STT04-017_Sacrifice_%u", (unsigned)i);
    create_basic_entity_card(world, player, zones.garden, CARD_DEF_STT03_003,
                             CARD_ELEMENT_FIRE, name, i);
  }

  ecs_entity_t enemy_target = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_004,
      CARD_ELEMENT_EARTH, "STT04-017_EnemyTarget", 0);
  ecs_set(world, enemy_target, BaseStats, {.attack = 2, .health = 6});
  ecs_set(world, enemy_target, CurStats, {.cur_atk = 2, .cur_hp = 6});

  const AbilityDef *def = azk_get_ability_def(CARD_DEF_STT04_017);
  AZK_TEST_ASSERT(def != NULL);
  AZK_TEST_ASSERT(def->cost_req.max == GARDEN_SIZE);

  bool triggered = azk_trigger_spell_ability(world, spell, player, 0);
  AZK_TEST_ASSERT(triggered);
  AZK_TEST_ASSERT(azk_get_ability_phase(world) == ABILITY_PHASE_COST_SELECTION);

  const int selection_order[] = {4, 3, 2, 1};
  for (size_t step = 0; step < sizeof(selection_order) / sizeof(selection_order[0]);
       ++step) {
    const int expected_target_index = selection_order[step];

    AzkActionMaskSet mask = {0};
    bool built = azk_build_action_mask_for_player(
        world, ecs_singleton_get(world, GameState), 0, &mask);
    AZK_TEST_ASSERT(built);

    bool found_noop = false;
    bool found_expected_target = false;
    int cost_target_count = 0;
    for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
      const UserAction *action = &mask.legal_actions[i];
      if (action->type == ACT_NOOP) {
        found_noop = true;
        continue;
      }
      if (action->type != ACT_SELECT_COST_TARGET) {
        continue;
      }

      cost_target_count++;
      if (action->subaction_1 == expected_target_index) {
        found_expected_target = true;
      }
    }

    AZK_TEST_ASSERT(found_expected_target);
    AZK_TEST_ASSERT(found_noop == (step > 0));
    AZK_TEST_ASSERT(cost_target_count == (int)(GARDEN_SIZE - step));

    bool selected = azk_process_cost_selection(world, expected_target_index);
    AZK_TEST_ASSERT(selected);
    AZK_TEST_ASSERT(azk_get_ability_phase(world) == ABILITY_PHASE_COST_SELECTION);

    const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
    AZK_TEST_ASSERT(ctx != NULL);
    AZK_TEST_ASSERT(ctx->cost.selected_count == (uint8_t)(step + 1));
  }

  AzkActionMaskSet final_cost_mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &final_cost_mask);
  AZK_TEST_ASSERT(built);

  bool found_noop = false;
  bool found_last_target = false;
  int remaining_cost_targets = 0;
  for (uint16_t i = 0; i < final_cost_mask.legal_action_count; ++i) {
    const UserAction *action = &final_cost_mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
      continue;
    }
    if (action->type != ACT_SELECT_COST_TARGET) {
      continue;
    }

    remaining_cost_targets++;
    if (action->subaction_1 == 0) {
      found_last_target = true;
    }
  }

  AZK_TEST_ASSERT(found_noop);
  AZK_TEST_ASSERT(found_last_target);
  AZK_TEST_ASSERT(remaining_cost_targets == 1);

  bool final_cost_selected = azk_process_cost_selection(world, 0);
  AZK_TEST_ASSERT(final_cost_selected);
  AZK_TEST_ASSERT(
      azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  AZK_TEST_ASSERT(ctx != NULL);
  AZK_TEST_ASSERT(ctx->cost.selected_count == GARDEN_SIZE);
  AZK_TEST_ASSERT(ecs_get_ordered_children(world, zones.garden).count == 0);
  AZK_TEST_ASSERT(ecs_get_ordered_children(world, zones.discard).count ==
                  GARDEN_SIZE);

  AbilityTargetChoice choices[AZK_MAX_ABILITY_TARGET_CHOICES] = {0};
  int choice_count = azk_collect_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, spell, player, choices,
      AZK_MAX_ABILITY_TARGET_CHOICES);
  AZK_TEST_ASSERT(choice_count > 0);

  int enemy_target_action_index = -1;
  for (int i = 0; i < choice_count; ++i) {
    if (choices[i].entity == enemy_target) {
      enemy_target_action_index = choices[i].action_index;
      break;
    }
  }
  AZK_TEST_ASSERT(enemy_target_action_index >= 0);

  AzkActionMaskSet effect_mask = {0};
  built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &effect_mask);
  AZK_TEST_ASSERT(built);

  bool found_effect_target = false;
  for (uint16_t i = 0; i < effect_mask.legal_action_count; ++i) {
    const UserAction *action = &effect_mask.legal_actions[i];
    if (action->type == ACT_SELECT_EFFECT_TARGET &&
        action->subaction_1 == enemy_target_action_index) {
      found_effect_target = true;
      break;
    }
  }
  AZK_TEST_ASSERT(found_effect_target);

  bool effect_selected =
      azk_process_effect_selection(world, enemy_target_action_index);
  AZK_TEST_ASSERT(effect_selected);
  AZK_TEST_ASSERT(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *enemy_stats = ecs_get(world, enemy_target, CurStats);
  AZK_TEST_ASSERT(enemy_stats != NULL);
  AZK_TEST_ASSERT(enemy_stats->cur_hp == 1);

  ecs_fini(world);
}

static void
test_azk01_041_main_ability_is_activatable_with_valid_discard_weapon(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t selection = create_zone(world, player, ZSelection, "Selection_P0");
  zones.selection = selection;
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  create_basic_leader(world, player, zones.leader, CARD_DEF_AZK01_119,
                      CARD_ELEMENT_LIGHTNING, "AZK01-041_Leader_P0");
  create_basic_leader(world, gs->players[1], gs->zones[1].leader,
                      CARD_DEF_STT04_001, CARD_ELEMENT_FIRE,
                      "AZK01-041_Leader_P1");
  gs->zones[0].selection = selection;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t wu_cha = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_041, CARD_ELEMENT_LIGHTNING,
      "AZK01-041_test", 0);
  create_basic_weapon_card(world, player, zones.discard, CARD_DEF_AZK01_018,
                           CARD_ELEMENT_NORMAL, "AZK01-041_discard_weapon");
  create_ikz_card(world, player, zones.ikz_area, "AZK01-041_IKZ_Test");

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_activate_action = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY &&
        action->subaction_1 == 0) {
      found_activate_action = true;
      break;
    }
  }

  assert(found_activate_action);

  bool triggered = azk_trigger_main_ability(world, wu_cha, player, 0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);
  assert(ecs_get_ordered_children(world, selection).count == 1);

  ecs_fini(world);
}

static void test_azk01_096_effect_selection_accepts_friendly_alley_targets(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t spell = ecs_new(world);
  ecs_set_name(world, spell, "AZK01-096_spell");
  ecs_set(world, spell, CardId, {.id = CARD_DEF_AZK01_096, .code = "AZK01-096"});
  ecs_set(world, spell, Type, {.value = CARD_TYPE_SPELL});
  ecs_set(world, spell, Element, {.element = CARD_ELEMENT_LIGHTNING});
  ecs_add_pair(world, spell, Rel_OwnedBy, player);
  initialize_test_card_runtime_components(world, spell);
  attach_ability_components(world, spell);

  ecs_entity_t garden_target = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-096_GardenTarget", 0);
  ecs_entity_t alley_target = create_basic_entity_card(
      world, player, zones.alley, CARD_DEF_STT03_004, CARD_ELEMENT_EARTH,
      "AZK01-096_AlleyTarget", 0);

  bool triggered = azk_trigger_spell_ability(world, spell, player, 0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_COST_SELECTION);

  bool cost_selected = azk_process_cost_selection(world, 0);
  assert(cost_selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityDef *def = azk_get_ability_def(CARD_DEF_AZK01_096);
  assert(def != NULL);
  assert(def->effect_req.type == ABILITY_TARGET_FRIENDLY_ALLEY_ENTITY);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_alley_target = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_SELECT_EFFECT_TARGET && action->subaction_1 == 0) {
      found_alley_target = true;
      break;
    }
  }

  assert(found_alley_target);

  ecs_entity_t resolved_alley = azk_resolve_ability_target_choice_entity(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, player, 0);
  assert(resolved_alley == alley_target);

  bool effect_selected = azk_process_effect_selection(world, 0);
  assert(effect_selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_get_target(world, garden_target, EcsChildOf, 0) == zones.alley);
  assert(ecs_get_target(world, alley_target, EcsChildOf, 0) == zones.garden);

  ecs_fini(world);
}

static void
test_azk01_097_on_play_discards_revealed_cards_when_no_weapons_found(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);
  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "AZK01-084_Leader_Test");
  create_basic_leader(world, opponent, opponent_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "AZK01-084_Opponent_Leader_Test");

  ecs_entity_t selection = create_zone(world, player, ZSelection, "Selection_P0");
  zones.selection = selection;
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->zones[0].selection = selection;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t pawnbroker = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_097, CARD_ELEMENT_LIGHTNING,
      "AZK01-097_NoWeapons", 0);
  create_basic_entity_card(world, player, zones.deck, CARD_DEF_STT03_003,
                           CARD_ELEMENT_EARTH, "AZK01-097_Mill_Entity_0", 0);
  create_basic_entity_card(world, player, zones.deck, CARD_DEF_STT03_004,
                           CARD_ELEMENT_EARTH, "AZK01-097_Mill_Entity_1", 1);
  create_basic_entity_card(world, player, zones.deck, CARD_DEF_STT03_005,
                           CARD_ELEMENT_EARTH, "AZK01-097_Mill_Entity_2", 2);

  bool queued = azk_trigger_on_play_ability(world, pawnbroker, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_get_ordered_children(world, selection).count == 0);
  assert(ecs_get_ordered_children(world, zones.deck).count == 0);
  assert(ecs_get_ordered_children(world, zones.discard).count == 3);

  ecs_fini(world);
}

static void
test_azk01_097_on_play_adds_selected_weapon_to_hand_and_discards_rest(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t selection = create_zone(world, player, ZSelection, "Selection_P0");
  zones.selection = selection;
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->zones[0].selection = selection;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t pawnbroker = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_097, CARD_ELEMENT_LIGHTNING,
      "AZK01-097_WithWeapon", 0);
  ecs_entity_t discard_a = create_basic_entity_card(
      world, player, zones.deck, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-097_Discard_A", 0);
  ecs_entity_t discard_b = create_basic_entity_card(
      world, player, zones.deck, CARD_DEF_STT03_004, CARD_ELEMENT_EARTH,
      "AZK01-097_Discard_B", 1);
  ecs_entity_t kept_weapon = create_basic_weapon_card(
      world, player, zones.deck, CARD_DEF_AZK01_018, CARD_ELEMENT_NORMAL,
      "AZK01-097_KeepWeapon");

  bool queued = azk_trigger_on_play_ability(world, pawnbroker, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_weapon_pick = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_SELECT_FROM_SELECTION && action->subaction_1 == 0) {
      found_weapon_pick = true;
      break;
    }
  }
  assert(found_weapon_pick);

  bool picked = azk_process_selection_pick(world, 0);
  assert(picked);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_get_target(world, kept_weapon, EcsChildOf, 0) == zones.hand);
  assert(ecs_get_target(world, discard_a, EcsChildOf, 0) == zones.discard);
  assert(ecs_get_target(world, discard_b, EcsChildOf, 0) == zones.discard);
  assert(ecs_get_ordered_children(world, selection).count == 0);
  assert(ecs_get_ordered_children(world, zones.deck).count == 0);

  ecs_fini(world);
}

static void
test_azk01_084_selection_includes_more_than_five_discard_targets(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);
  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "AZK01-084_Leader_Test");
  create_basic_leader(world, opponent, opponent_zones.leader,
                      CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
                      "AZK01-084_Opponent_Leader_Test");

  zones.gate = create_zone(world, player, ZGate, "AZK01-084_Gate_P0");
  zones.ikz_pile = create_zone(world, player, ZIKZPileTag, "AZK01-084_IKZPile_P0");
  opponent_zones.gate =
      create_zone(world, opponent, ZGate, "AZK01-084_Gate_P1");
  opponent_zones.ikz_pile =
      create_zone(world, opponent, ZIKZPileTag, "AZK01-084_IKZPile_P1");
  opponent_zones.ikz_area =
      create_zone(world, opponent, ZIKZAreaTag, "AZK01-084_IKZArea_P1");

  ecs_entity_t gate0 = ecs_new(world);
  ecs_set(world, gate0, CardId, {.id = CARD_DEF_STT01_002, .code = "STT01-002"});
  ecs_set(world, gate0, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate0, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate0, Rel_OwnedBy, player);
  ecs_add_pair(world, gate0, EcsChildOf, zones.gate);

  ecs_entity_t gate1 = ecs_new(world);
  ecs_set(world, gate1, CardId, {.id = CARD_DEF_STT01_002, .code = "STT01-002"});
  ecs_set(world, gate1, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate1, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate1, Rel_OwnedBy, opponent);
  ecs_add_pair(world, gate1, EcsChildOf, opponent_zones.gate);

  ecs_entity_t selection = create_zone(world, player, ZSelection, "Selection_P0");
  zones.selection = selection;
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->zones[0].gate = zones.gate;
  gs->zones[0].ikz_pile = zones.ikz_pile;
  gs->zones[1].gate = opponent_zones.gate;
  gs->zones[1].ikz_pile = opponent_zones.ikz_pile;
  gs->zones[1].ikz_area = opponent_zones.ikz_area;
  gs->zones[0].selection = selection;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t source = ecs_new(world);
  ecs_set_name(world, source, "AZK01-084_Test");
  ecs_set(world, source, CardId, {.id = CARD_DEF_AZK01_084});
  ecs_set(world, source, Type, {.value = CARD_TYPE_SPELL});
  ecs_set(world, source, IKZCost, {.ikz_cost = 2});
  ecs_set(world, source, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, source, Rel_OwnedBy, player);
  ecs_add_pair(world, source, EcsChildOf, zones.hand);

  for (uint8_t i = 0; i < 7; ++i) {
    char name[64];
    snprintf(name, sizeof(name), "AZK01-084_Target_%u", (unsigned)i);
    ecs_entity_t target = create_basic_entity_card(
        world, player, zones.discard, CARD_DEF_STT03_003, CARD_ELEMENT_NORMAL,
        name, i);
    ecs_set(world, target, IKZCost, {.ikz_cost = 2});
  }

  ecs_entity_t invalid_target = create_basic_entity_card(
      world, player, zones.discard, CARD_DEF_STT03_003, CARD_ELEMENT_NORMAL,
      "AZK01-084_Invalid_Target", 7);
  ecs_set(world, invalid_target, IKZCost, {.ikz_cost = 7});

  bool triggered = azk_trigger_main_ability(world, source, player, 0);
  assert(triggered);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_SELECTION_PICK);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  assert(ctx != NULL);
  assert(ctx->selection.count == 7);
  assert(ecs_get_ordered_children(world, selection).count == 7);

  ObservationData observation = create_observation_data(world, 0);
  assert(observation.my_observation_data.selection_count == 7);
  assert(observation.my_observation_data.selection[6].id.id ==
         CARD_DEF_STT03_003);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(world, gs, 0, &mask);
  assert(built);

  bool found_last_selection = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_SELECT_FROM_SELECTION && action->subaction_1 == 6) {
      found_last_selection = true;
      break;
    }
  }
  assert(found_last_selection);

  ecs_fini(world);
}

static void test_azk01_059_triggers_after_nonlethal_damage(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  ecs_entity_t spice = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_059, CARD_ELEMENT_FIRE,
      "AZK01-059_nonlethal_test", 0);
  ecs_entity_t ally = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-059_target_nonlethal_test", 1);
  ecs_entity_t source = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-059_source_nonlethal_test", 0);

  ecs_set(world, spice, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, spice, CurStats, {.cur_atk = 1, .cur_hp = 2});

  bool damaged = deal_effect_damage_from_source(world, source, spice, 1);
  assert(damaged);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const GameState *after_trigger = ecs_singleton_get(world, GameState);
  assert(after_trigger != NULL);
  assert(after_trigger->active_player_index == 0);

  AzkActionMaskSet mask = {0};
  bool built =
      azk_build_action_mask_for_player(world, after_trigger, 0, &mask);
  assert(built);

  bool found_target = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_SELECT_EFFECT_TARGET && action->subaction_1 == 1) {
      found_target = true;
      break;
    }
  }
  assert(found_target);

  bool selected = azk_process_effect_selection(world, 1);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *ally_stats = ecs_get(world, ally, CurStats);
  assert(ally_stats != NULL);
  assert(ally_stats->cur_atk == 2);
  assert(ecs_get_target(world, spice, EcsChildOf, 0) == zones.garden);

  ecs_fini(world);
}

static void test_azk01_059_triggers_after_lethal_damage(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  ecs_entity_t spice = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_059, CARD_ELEMENT_FIRE,
      "AZK01-059_lethal_test", 0);
  ecs_entity_t ally = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
      "AZK01-059_target_lethal_test", 1);
  ecs_entity_t source = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "AZK01-059_source_lethal_test", 0);

  ecs_set(world, spice, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, spice, CurStats, {.cur_atk = 1, .cur_hp = 2});

  bool damaged = deal_effect_damage_from_source(world, source, spice, 2);
  assert(damaged);
  assert(ecs_get_target(world, spice, EcsChildOf, 0) == zones.discard);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  bool selected = azk_process_effect_selection(world, 1);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *ally_stats = ecs_get(world, ally, CurStats);
  assert(ally_stats != NULL);
  assert(ally_stats->cur_atk == 2);

  ecs_fini(world);
}

static void test_azk01_062_auto_resolves_self_damage_when_no_redirect_targets(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];

  ecs_entity_t pekiro = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_062, CARD_ELEMENT_FIRE,
      "AZK01-062_no_redirect_target", 0);
  ecs_entity_t source = ecs_new(world);
  ecs_set_name(world, source, "AZK01-062_source_spell");
  ecs_set(world, source, CardId, {.id = CARD_DEF_AZK01_066, .code = "AZK01-066"});
  ecs_set(world, source, Type, {.value = CARD_TYPE_SPELL});
  ecs_set(world, source, Element, {.element = CARD_ELEMENT_FIRE});
  ecs_add_pair(world, source, Rel_OwnedBy, opponent);
  initialize_test_card_runtime_components(world, source);

  ecs_set(world, pekiro, BaseStats, {.attack = 2, .health = 2});
  ecs_set(world, pekiro, CurStats, {.cur_atk = 2, .cur_hp = 2});

  bool damaged = deal_effect_damage_from_source(world, source, pekiro, 2);
  assert(damaged);

  const CurStats *before_resolution = ecs_get(world, pekiro, CurStats);
  assert(before_resolution != NULL);
  assert(before_resolution->cur_hp == 2);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_get_target(world, pekiro, EcsChildOf, 0) == zones.discard);

  const PendingDamageRedirectQueue *redirect_queue =
      ecs_singleton_get(world, PendingDamageRedirectQueue);
  assert(redirect_queue != NULL);
  assert(redirect_queue->count == 0);

  ecs_fini(world);
}

static void test_azk01_059_lethal_damage_prompts_in_engine_action_flow(void) {
  const CardInfo player0_deck[] = {
      {.card_id = CARD_DEF_STT01_001, .card_count = 1},
      {.card_id = CARD_DEF_STT01_002, .card_count = 1},
      {.card_id = CARD_DEF_AZK01_059, .card_count = 50},
      {.card_id = CARD_DEF_IKZ_001, .card_count = 10},
  };
  const CardInfo player1_deck[] = {
      {.card_id = CARD_DEF_STT01_001, .card_count = 1},
      {.card_id = CARD_DEF_STT01_002, .card_count = 1},
      {.card_id = CARD_DEF_AZK01_114, .card_count = 50},
      {.card_id = CARD_DEF_IKZ_001, .card_count = 10},
  };

  AzkEngine *engine = azk_engine_create_with_decks(
      12345, player0_deck,
      sizeof(player0_deck) / sizeof(player0_deck[0]), player1_deck,
      sizeof(player1_deck) / sizeof(player1_deck[0]));
  assert(engine != NULL);

  GameState *gs = ecs_singleton_get_mut(engine, GameState);
  assert(gs != NULL);
  gs->phase = PHASE_MAIN;
  gs->active_player_index = 0;
  gs->turn_number = 1;
  ecs_singleton_modified(engine, GameState);

  ecs_entity_t player0 = gs->players[0];
  ecs_entity_t player1 = gs->players[1];

  grant_ikz_cards_to_player(engine, 0, 4);
  grant_ikz_cards_to_player(engine, 1, 3);

  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player0,
                  .type = ACT_PLAY_ENTITY_TO_GARDEN,
                  .subaction_1 = 0,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });
  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player0,
                  .type = ACT_PLAY_ENTITY_TO_GARDEN,
                  .subaction_1 = 0,
                  .subaction_2 = 1,
                  .subaction_3 = 0,
              });
  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player0,
                  .type = ACT_NOOP,
                  .subaction_1 = 0,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });

  const GameState *turn_two = ecs_singleton_get(engine, GameState);
  assert(turn_two != NULL);
  assert(turn_two->phase == PHASE_MAIN);
  assert(turn_two->active_player_index == 1);
  assert(turn_two->turn_number == 2);

  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player1,
                  .type = ACT_PLAY_ENTITY_TO_ALLEY,
                  .subaction_1 = 0,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });

  assert(azk_get_ability_phase(engine) == ABILITY_PHASE_EFFECT_SELECTION);

  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player1,
                  .type = ACT_SELECT_EFFECT_TARGET,
                  .subaction_1 = 0,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });

  const GameState *after_damage = ecs_singleton_get(engine, GameState);
  assert(after_damage != NULL);
  assert(after_damage->phase == PHASE_MAIN);
  assert(after_damage->active_player_index == 0);
  assert(azk_get_ability_phase(engine) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityContext *ctx = ecs_singleton_get(engine, AbilityContext);
  assert(ctx != NULL);
  const CardId *source_card_id = ecs_get(engine, ctx->runtime.source_card, CardId);
  assert(source_card_id != NULL);
  assert(source_card_id->id == CARD_DEF_AZK01_059);

  ObservationData observation = {0};
  bool observed = azk_engine_observe(engine, 0, &observation);
  assert(observed);

  bool found_surviving_spice_target = false;
  for (uint16_t i = 0; i < observation.action_mask.legal_action_count; ++i) {
    if (observation.action_mask.legal_primary[i] == ACT_SELECT_EFFECT_TARGET &&
        observation.action_mask.legal_sub1[i] == 1) {
      found_surviving_spice_target = true;
      break;
    }
  }
  assert(found_surviving_spice_target);

  ecs_entity_t surviving_spice =
      find_card_in_zone_index(engine, after_damage->zones[0].garden, 1);
  assert(surviving_spice != 0);
  ecs_entity_t destroyed_spice =
      find_card_in_zone_index(engine, after_damage->zones[0].garden, 0);
  assert(destroyed_spice == 0);

  submit_engine_action_and_advance(
      engine, &(UserAction){
                  .player = player0,
                  .type = ACT_SELECT_EFFECT_TARGET,
                  .subaction_1 = 1,
                  .subaction_2 = 0,
                  .subaction_3 = 0,
              });

  assert(azk_get_ability_phase(engine) == ABILITY_PHASE_NONE);
  const CurStats *surviving_spice_stats = ecs_get(engine, surviving_spice, CurStats);
  assert(surviving_spice_stats != NULL);
  assert(surviving_spice_stats->cur_atk == 2);

  azk_engine_destroy(engine);
}

static void test_azk01_057_start_of_turn_selects_self_when_enemy_garden_empty(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t saeko = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_057, CARD_ELEMENT_FIRE,
      "AZK01-057_self_only", 0);
  ecs_set(world, saeko, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, saeko, CurStats, {.cur_atk = 1, .cur_hp = 2});

  bool queued = azk_trigger_start_of_turn_abilities(world);
  assert(queued);
  assert(azk_has_queued_triggered_effects(world));

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  assert(ctx != NULL);
  assert(ctx->effect.min_required == 1);
  assert(ctx->effect.max_allowed == 1);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_self_target = false;
  bool found_enemy_target = false;
  bool found_noop = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
    }
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }
    if (action->subaction_1 == 0) {
      found_self_target = true;
    }
    if (action->subaction_1 >= GARDEN_SIZE) {
      found_enemy_target = true;
    }
  }

  assert(found_self_target);
  assert(!found_enemy_target);
  assert(!found_noop);

  bool selected = azk_process_effect_selection(world, 0);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *saeko_stats = ecs_get(world, saeko, CurStats);
  assert(saeko_stats != NULL);
  assert(saeko_stats->cur_hp == 1);

  ecs_fini(world);
}

static void test_azk01_057_start_of_turn_requires_enemy_second_target(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  ecs_entity_t saeko = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_057, CARD_ELEMENT_FIRE,
      "AZK01-057_enemy_second", 0);
  ecs_entity_t enemy = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "EnemyGardenTarget", 0);
  ecs_set(world, saeko, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, saeko, CurStats, {.cur_atk = 1, .cur_hp = 2});
  ecs_set(world, enemy, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, enemy, CurStats, {.cur_atk = 1, .cur_hp = 2});

  bool queued = azk_trigger_start_of_turn_abilities(world);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  assert(ctx != NULL);
  assert(ctx->effect.min_required == 2);
  assert(ctx->effect.max_allowed == 2);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_self_target = false;
  bool found_enemy_target = false;
  bool found_noop = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
    }
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }
    if (action->subaction_1 == 0) {
      found_self_target = true;
    }
    if (action->subaction_1 == GARDEN_SIZE) {
      found_enemy_target = true;
    }
  }

  assert(found_self_target);
  assert(!found_enemy_target);
  assert(!found_noop);

  bool selected = azk_process_effect_selection(world, 0);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_EFFECT_SELECTION);

  mask = (AzkActionMaskSet){0};
  built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  found_self_target = false;
  found_enemy_target = false;
  found_noop = false;
  for (uint16_t i = 0; i < mask.legal_action_count; i++) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_NOOP) {
      found_noop = true;
    }
    if (action->type != ACT_SELECT_EFFECT_TARGET) {
      continue;
    }
    if (action->subaction_1 == 0) {
      found_self_target = true;
    }
    if (action->subaction_1 == GARDEN_SIZE) {
      found_enemy_target = true;
    }
  }

  assert(!found_self_target);
  assert(found_enemy_target);
  assert(!found_noop);

  selected = azk_process_effect_selection(world, GARDEN_SIZE);
  assert(selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *saeko_stats = ecs_get(world, saeko, CurStats);
  const CurStats *enemy_stats = ecs_get(world, enemy, CurStats);
  assert(saeko_stats != NULL);
  assert(enemy_stats != NULL);
  assert(saeko_stats->cur_hp == 1);
  assert(enemy_stats->cur_hp == 1);

  ecs_fini(world);
}

static void test_stt03_010_heals_leader_when_both_combatants_are_destroyed(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  ecs_entity_t leader = create_basic_leader(
      world, player, zones.leader, CARD_DEF_STT01_001, CARD_ELEMENT_LIGHTNING,
      "Leader_P0_Test");
  ecs_set(world, leader, CurStats, {.cur_atk = 0, .cur_hp = 19});

  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P1_Test");

  ecs_entity_t shroommancer = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_010, CARD_ELEMENT_EARTH,
      "Shroommancer_test", 0);
  ecs_entity_t defender = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "Defender_test", 0);

  AttackIntent intent = {
      .attacking_player = player,
      .defending_player = opponent,
      .attacking_card = shroommancer,
      .defending_card = defender,
      .attacker_index = 0,
      .defender_index = 0,
      .attacker_is_leader = false,
  };

  int attack_result = attack(world, &intent);
  assert(attack_result == 0);

  resolve_combat(world);

  const CardId *attacker_id = ecs_get(world, shroommancer, CardId);
  assert(attacker_id != NULL);
  if (azk_has_ability_with_timing(attacker_id->id, ecs_id(AAfterAttacking))) {
    azk_queue_triggered_effect(
        world,
        get_first_timed_ability(world, shroommancer, ecs_id(AAfterAttacking)),
        player, TIMING_TAG_AFTER_ATTACKING);
  }

  const GameState *after_combat = ecs_singleton_get(world, GameState);
  assert(after_combat != NULL);
  assert(after_combat->last_combat.attacker == shroommancer);
  assert(after_combat->last_combat.defender == defender);
  assert(after_combat->last_combat.defender_destroyed);
  assert(after_combat->last_combat.attacker_destroyed);
  assert(ecs_get_target(world, shroommancer, EcsChildOf, 0) == zones.discard);
  assert(ecs_get_target(world, defender, EcsChildOf, 0) ==
         opponent_zones.discard);
  assert(azk_has_queued_triggered_effects(world));

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);

  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const CurStats *leader_stats = ecs_get(world, leader, CurStats);
  assert(leader_stats != NULL);
  assert(leader_stats->cur_hp == 20);

  ecs_fini(world);
}

static void test_stt04_008_after_attacking_only_untaps_once_per_turn(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  ecs_entity_t emberheart = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT04_008, CARD_ELEMENT_FIRE,
      "STT04-008_test", 0);
  ecs_entity_t defender = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "Defender_test", 0);

  ecs_set(world, emberheart, TapState, {.tapped = true, .cooldown = false});

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 0;
  gs->last_combat = (LastCombatResult){
      .attacker = emberheart,
      .defender = defender,
      .defender_was_garden_entity = true,
  };
  ecs_singleton_modified(world, GameState);

  bool queued = azk_queue_triggered_effect(
      world,
      get_first_timed_ability(world, emberheart, ecs_id(AAfterAttacking)),
      player, TIMING_TAG_AFTER_ATTACKING);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);

  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  const TapState *tap = ecs_get(world, emberheart, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(!tap->cooldown);

  ecs_entity_t emberheart_abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t emberheart_ability_count = azk_collect_card_abilities(
      world, emberheart, emberheart_abilities, AZK_MAX_CARD_ABILITIES);
  assert(emberheart_ability_count > 0);
  ecs_entity_t emberheart_ability = emberheart_abilities[0];
  const AbilityRepeatContext *repeat_ctx =
      ecs_get(world, emberheart_ability, AbilityRepeatContext);
  assert(repeat_ctx != NULL);
  assert(repeat_ctx->was_applied);

  ecs_set(world, emberheart, TapState, {.tapped = true, .cooldown = false});

  queued = azk_queue_triggered_effect(
      world,
      get_first_timed_ability(world, emberheart, ecs_id(AAfterAttacking)),
      player, TIMING_TAG_AFTER_ATTACKING);
  assert(!queued);

  tap = ecs_get(world, emberheart, TapState);
  assert(tap != NULL);
  assert(tap->tapped);

  ecs_fini(world);
}

static void test_stt04_008_after_attacking_does_not_prompt_if_destroyed(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  ecs_entity_t emberheart = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT04_008, CARD_ELEMENT_FIRE,
      "STT04-008_destroyed_test", 0);
  ecs_entity_t defender = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT03_003,
      CARD_ELEMENT_EARTH, "Defender_destroyed_test", 0);

  discard_card(world, emberheart);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 0;
  gs->last_combat = (LastCombatResult){
      .attacker = emberheart,
      .defender = defender,
      .defender_was_garden_entity = true,
      .attacker_destroyed = true,
  };
  ecs_singleton_modified(world, GameState);

  bool queued = azk_queue_triggered_effect(
      world,
      get_first_timed_ability(world, emberheart, ecs_id(AAfterAttacking)),
      player, TIMING_TAG_AFTER_ATTACKING);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);
  assert(!azk_has_queued_triggered_effects(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_get_target(world, emberheart, EcsChildOf, 0) == zones.discard);

  ecs_fini(world);
}

static void
test_stt04_013_destroy_observer_only_untaps_once_per_turn(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");
  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P1_Test");
  assert(find_leader_card_in_zone(world, zones.leader) != 0);
  assert(find_leader_card_in_zone(world, opponent_zones.leader) != 0);

  ecs_entity_t kurai = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT04_013, CARD_ELEMENT_FIRE,
      "STT04-013_test", 0);
  ecs_entity_t enemy_one = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT04_011,
      CARD_ELEMENT_NORMAL, "EnemyOne_test", 0);
  ecs_entity_t enemy_two = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT04_011,
      CARD_ELEMENT_NORMAL, "EnemyTwo_test", 1);

  ecs_set(world, kurai, TapState, {.tapped = true, .cooldown = false});

  discard_card(world, enemy_one);

  const TapState *tap = ecs_get(world, kurai, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(!tap->cooldown);

  const STT04KuraiState *state = ecs_get(world, kurai, STT04KuraiState);
  assert(state != NULL);
  assert(state->last_untap_turn == 1);

  ecs_set(world, kurai, TapState, {.tapped = true, .cooldown = false});

  discard_card(world, enemy_two);

  tap = ecs_get(world, kurai, TapState);
  assert(tap != NULL);
  assert(tap->tapped);
  assert(!tap->cooldown);

  state = ecs_get(world, kurai, STT04KuraiState);
  assert(state != NULL);
  assert(state->last_untap_turn == 1);

  ecs_fini(world);
}

static void
test_stt04_013_destroy_observer_does_not_clear_cooldown(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");
  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P1_Test");
  assert(find_leader_card_in_zone(world, zones.leader) != 0);
  assert(find_leader_card_in_zone(world, opponent_zones.leader) != 0);

  ecs_entity_t kurai = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT04_013, CARD_ELEMENT_FIRE,
      "STT04-013_cooldown_test", 0);
  ecs_entity_t enemy = create_basic_entity_card(
      world, opponent, opponent_zones.garden, CARD_DEF_STT04_011,
      CARD_ELEMENT_NORMAL, "EnemyCooldown_test", 0);

  ecs_set(world, kurai, TapState, {.tapped = false, .cooldown = true});

  discard_card(world, enemy);

  const TapState *tap = ecs_get(world, kurai, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(tap->cooldown);
  assert(!can_tap_card(world, kurai, false));

  const STT04KuraiState *state = ecs_get(world, kurai, STT04KuraiState);
  assert(state != NULL);
  assert(state->last_untap_turn == 1);

  ecs_fini(world);
}

static void
test_azk01_118_on_play_does_not_prompt_without_two_other_cards(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];

  ecs_entity_t ringleader = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_118, CARD_ELEMENT_FIRE,
      "AZK01-118_not_ready", 0);
  create_basic_entity_card(world, opponent, opponent_zones.garden,
                           CARD_DEF_STT03_003, CARD_ELEMENT_EARTH,
                           "AZK01-118_enemy_target", 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->cards_played_this_turn[0] = 2;
  ecs_singleton_modified(world, GameState);

  bool queued = azk_trigger_on_play_ability(world, ringleader, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  ecs_fini(world);
}

static void test_azk01_039_when_equipped_grants_charge_and_clears_cooldown(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_set(world, ecs_id(ActionContext), ActionContext, {0});
  init_main_phase_system(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t piko = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_039, CARD_ELEMENT_LIGHTNING,
      "AZK01-039_test", 0);
  ecs_set(world, piko, TapState, {.tapped = false, .cooldown = true});

  ecs_entity_t weapon = create_basic_weapon_card(
      world, player, zones.hand, CARD_DEF_AZK01_018, CARD_ELEMENT_NORMAL,
      "AZK01-039_Weapon_Test");
  create_ikz_card(world, player, zones.ikz_area, "AZK01-039_IKZ_Test");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 0;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 2;
  ecs_singleton_modified(world, GameState);

  ActionContext *ac = ecs_singleton_get_mut(world, ActionContext);
  assert(ac != NULL);
  *ac = (ActionContext){
      .user_action =
          {
              .player = player,
              .type = ACT_ATTACH_WEAPON_FROM_HAND,
              .subaction_1 = 0,
              .subaction_2 = 0,
          },
  };
  ecs_singleton_modified(world, ActionContext);

  ecs_entity_t main_system = ecs_lookup(world, "MainPhaseSystem");
  assert(main_system != 0);
  ecs_run(world, main_system, 0, NULL);

  assert(ecs_get_target(world, weapon, EcsChildOf, 0) == piko);
  assert(azk_has_queued_triggered_effects(world));
  assert(!ecs_has(world, piko, Charge));

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);
  assert(!azk_has_queued_triggered_effects(world));
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);
  assert(ecs_has(world, piko, Charge));

  const TapState *tap = ecs_get(world, piko, TapState);
  assert(tap != NULL);
  assert(!tap->cooldown);

  ecs_fini(world);
}

static void
test_azk01_112_on_play_from_alley_grants_in_play_charge_across_gate(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);
  const GameState *gs_ro = ecs_singleton_get(world, GameState);
  assert(gs_ro != NULL);
  ecs_entity_t opponent = gs_ro->players[1];
  PlayerZones opponent_zones = gs_ro->zones[1];
  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");
  create_basic_leader(world, opponent, opponent_zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P1_Test");

  ecs_entity_t tribute = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_FIRE,
      "AZK01-112_tribute", 0);
  ecs_entity_t enrai = create_basic_entity_card(
      world, player, zones.alley, CARD_DEF_AZK01_112, CARD_ELEMENT_FIRE,
      "AZK01-112_alley", 0);

  bool queued = azk_trigger_on_play_ability(world, enrai, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);

  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_COST_SELECTION);

  ecs_defer_begin(world);
  bool cost_selected = azk_process_cost_selection(world, 0);
  ecs_defer_end(world);
  assert(cost_selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  assert(ecs_get_target(world, tribute, EcsChildOf, 0) == zones.discard);
  assert(ecs_has(world, enrai, Charge));

  const TapState *tap = ecs_get(world, enrai, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(!tap->cooldown);

  tick_end_of_turn_effects_for_player(world, 0);

  assert(ecs_has(world, enrai, Charge));
  tap = ecs_get(world, enrai, TapState);
  assert(tap != NULL);
  assert(!tap->cooldown);

  ecs_entity_t gate_card = ecs_new(world);
  ecs_set_name(world, gate_card, "AZK01-122_gate_for_112");
  ecs_set(world, gate_card, CardId, {.id = CARD_DEF_AZK01_122});
  ecs_set(world, gate_card, Type, {.value = CARD_TYPE_GATE});
  ecs_set(world, gate_card, TapState, {.tapped = false, .cooldown = false});
  ecs_add_pair(world, gate_card, Rel_OwnedBy, player);

  ecs_defer_begin(world);
  int result = gate_card_into_garden(
      world,
      &(GatePortalIntent){
          .player = player,
          .gate_card = gate_card,
          .alley_card = enrai,
          .target_zone = zones.garden,
          .garden_index = 0,
      });
  ecs_defer_end(world);
  azk_finalize_pending_zone_move_logs(world);
  assert(result == 0);

  assert(ecs_get_target(world, enrai, EcsChildOf, 0) == zones.garden);
  const ZoneIndex *zone_index = ecs_get(world, enrai, ZoneIndex);
  assert(zone_index != NULL);
  assert(zone_index->index == 0);
  assert(ecs_has(world, enrai, Charge));

  tap = ecs_get(world, enrai, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(!tap->cooldown);

  AzkActionMaskSet mask = {0};
  bool built = azk_build_action_mask_for_player(
      world, ecs_singleton_get(world, GameState), 0, &mask);
  assert(built);

  bool found_attack = false;
  for (uint16_t i = 0; i < mask.legal_action_count; ++i) {
    const UserAction *action = &mask.legal_actions[i];
    if (action->type == ACT_ATTACK && action->subaction_1 == 0 &&
        action->subaction_2 == GARDEN_SIZE) {
      found_attack = true;
      break;
    }
  }
  assert(found_attack);

  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(world, &log_count);
  assert(logs != NULL);

  bool found_gate_move_log = false;
  for (uint8_t i = 0; i < log_count; ++i) {
    if (logs[i].type != GLOG_CARD_ZONE_MOVED ||
        logs[i].data.zone_moved.card.card_def_id != CARD_DEF_AZK01_112) {
      continue;
    }

    const GameLogZoneMoved *move = &logs[i].data.zone_moved;
    if (move->from_zone != GLOG_ZONE_ALLEY || move->to_zone != GLOG_ZONE_GARDEN) {
      continue;
    }

    found_gate_move_log = true;
    assert(move->to_index == 0);
    assert(move->metadata.has_charge);
    assert(!move->metadata.cooldown);
  }
  assert(found_gate_move_log);

  discard_card(world, enrai);
  assert(ecs_get_target(world, enrai, EcsChildOf, 0) == zones.discard);
  assert(!ecs_has(world, enrai, Charge));

  ecs_fini(world);
}

static void
test_azk01_112_on_play_in_garden_counts_self_when_checking_empty_garden(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);
  create_basic_leader(world, player, zones.leader, CARD_DEF_STT01_001,
                      CARD_ELEMENT_LIGHTNING, "Leader_P0_Test");

  ecs_entity_t enrai = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_112, CARD_ELEMENT_FIRE,
      "AZK01-112_garden", 0);
  ecs_entity_t tribute = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_STT03_003, CARD_ELEMENT_FIRE,
      "AZK01-112_other_garden_entity", 1);
  ecs_set(world, enrai, TapState, {.tapped = false, .cooldown = true});

  bool queued = azk_trigger_on_play_ability(world, enrai, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_CONFIRMATION);

  bool confirmed = azk_process_ability_confirmation(world);
  assert(confirmed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_COST_SELECTION);

  bool cost_selected = azk_process_cost_selection(world, 1);
  assert(cost_selected);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

  assert(ecs_get_target(world, tribute, EcsChildOf, 0) == zones.discard);
  assert(ecs_get_target(world, enrai, EcsChildOf, 0) == zones.garden);
  assert(!ecs_has(world, enrai, Charge));

  const TapState *tap = ecs_get(world, enrai, TapState);
  assert(tap != NULL);
  assert(!tap->tapped);
  assert(tap->cooldown);

  ecs_fini(world);
}

static void
test_azk01_118_on_play_does_not_prompt_without_enemy_garden_entity(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t player = 0;
  PlayerZones zones = {0};
  setup_single_player_play_fixture(world, &player, &zones);

  ecs_entity_t ringleader = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_118, CARD_ELEMENT_FIRE,
      "AZK01-118_no_enemy_target", 0);

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->cards_played_this_turn[0] = 3;
  ecs_singleton_modified(world, GameState);

  bool queued = azk_trigger_on_play_ability(world, ringleader, player);
  assert(queued);

  bool processed = azk_process_triggered_effect_queue(world);
  assert(!processed);
  assert(azk_get_ability_phase(world) == ABILITY_PHASE_NONE);

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

static void test_azk01_028_returns_all_other_garden_entities_without_skipping(
    void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);

  ecs_entity_t owner = 0;
  PlayerZones owner_zones = {0};
  setup_single_player_play_fixture(world, &owner, &owner_zones);

  const GameState *gs = ecs_singleton_get(world, GameState);
  assert(gs != NULL);
  ecs_entity_t opponent = gs->players[1];
  PlayerZones opponent_zones = gs->zones[1];

  ecs_entity_t source = create_basic_entity_card(
      world, owner, owner_zones.garden, CARD_DEF_AZK01_028, CARD_ELEMENT_WATER,
      "AZK01-028_Source", 0);
  ecs_entity_t owner_other_a = create_basic_entity_card(
      world, owner, owner_zones.garden, CARD_DEF_AZK01_001,
      CARD_ELEMENT_NORMAL, "AZK01-028_OwnerOtherA", 1);
  ecs_entity_t owner_other_b = create_basic_entity_card(
      world, owner, owner_zones.garden, CARD_DEF_AZK01_001,
      CARD_ELEMENT_NORMAL, "AZK01-028_OwnerOtherB", 2);

  ecs_entity_t opponent_cards[4] = {
      create_basic_entity_card(world, opponent, opponent_zones.garden,
                               CARD_DEF_AZK01_001, CARD_ELEMENT_NORMAL,
                               "AZK01-028_OppA", 0),
      create_basic_entity_card(world, opponent, opponent_zones.garden,
                               CARD_DEF_AZK01_001, CARD_ELEMENT_NORMAL,
                               "AZK01-028_OppB", 1),
      create_basic_entity_card(world, opponent, opponent_zones.garden,
                               CARD_DEF_AZK01_001, CARD_ELEMENT_NORMAL,
                               "AZK01-028_OppC", 2),
      create_basic_entity_card(world, opponent, opponent_zones.garden,
                               CARD_DEF_AZK01_001, CARD_ELEMENT_NORMAL,
                               "AZK01-028_OppD", 3),
  };

  ecs_entity_t hand_cards[4] = {0};
  for (uint8_t i = 0; i < 4; ++i) {
    char name[32];
    snprintf(name, sizeof(name), "AZK01-028_Hand_%u", (unsigned)i);
    hand_cards[i] = create_basic_entity_card(world, owner, owner_zones.hand,
                                             CARD_DEF_AZK01_001,
                                             CARD_ELEMENT_NORMAL, name, i);
    ecs_remove_id(world, hand_cards[i], ecs_id(ZoneIndex));
  }

  assert(ecs_get_ordered_children(world, owner_zones.hand).count == 4);
  assert(ecs_get_ordered_children(world, owner_zones.garden).count == 3);
  assert(ecs_get_ordered_children(world, opponent_zones.garden).count == 4);

  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);
  assert(ctx != NULL);
  ctx->runtime.owner = owner;
  ctx->runtime.source_card = source;
  ecs_singleton_modified(world, AbilityContext);

  azk01_028_apply_costs(world, ctx);
  azk01_028_apply_effects(world, ctx);

  for (uint8_t i = 0; i < 4; ++i) {
    assert(ecs_get_target(world, hand_cards[i], EcsChildOf, 0) ==
           owner_zones.discard);
    assert(ecs_get_target(world, opponent_cards[i], EcsChildOf, 0) ==
           opponent_zones.hand);
  }

  assert(ecs_get_target(world, source, EcsChildOf, 0) == owner_zones.garden);
  assert(ecs_get_target(world, owner_other_a, EcsChildOf, 0) ==
         owner_zones.hand);
  assert(ecs_get_target(world, owner_other_b, EcsChildOf, 0) ==
         owner_zones.hand);

  assert(ecs_get_ordered_children(world, owner_zones.garden).count == 1);
  assert(ecs_get_ordered_children(world, opponent_zones.garden).count == 0);
  assert(ecs_get_ordered_children(world, owner_zones.discard).count == 4);
  assert(ecs_get_ordered_children(world, owner_zones.hand).count == 2);
  assert(ecs_get_ordered_children(world, opponent_zones.hand).count == 4);

  ecs_fini(world);
}

static void test_observation_hides_action_mask_while_trigger_queue_pending(void) {
  ecs_world_t *world = azk_world_init(42);
  const GameState *initial = ecs_singleton_get(world, GameState);
  assert(initial != NULL);

  ecs_entity_t player = initial->players[0];
  PlayerZones zones = initial->zones[0];

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  assert(gs != NULL);
  gs->active_player_index = 0;
  gs->phase = PHASE_MAIN;
  gs->turn_number = 1;
  ecs_singleton_modified(world, GameState);

  ecs_entity_t saeko = create_basic_entity_card(
      world, player, zones.garden, CARD_DEF_AZK01_057, CARD_ELEMENT_FIRE,
      "AZK01-057_pending_queue", 0);
  ecs_set(world, saeko, BaseStats, {.attack = 1, .health = 2});
  ecs_set(world, saeko, CurStats, {.cur_atk = 1, .cur_hp = 2});
  ecs_set(world, saeko, IKZCost, {.ikz_cost = 1});

  bool queued = azk_trigger_start_of_turn_abilities(world);
  assert(queued);
  assert(azk_has_queued_triggered_effects(world));

  ObservationData observation = create_observation_data(world, 0);
  assert(observation.action_mask.legal_action_count == 0);
  for (size_t i = 0; i < AZK_ACTION_TYPE_COUNT; ++i) {
    assert(!observation.action_mask.primary_action_mask[i]);
  }

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

int main(int argc, char **argv) {
  if (argc > 1 &&
      strcmp(argv[1], "--run-stt04-017-regression") == 0) {
    test_stt04_017_cost_selection_allows_fifth_garden_sacrifice();
    return 0;
  }

  test_azk_world_init_sets_game_state();
  test_world_init_creates_player_zones();
  test_world_init_assigns_damage_trackers_to_cards();
  test_world_init_assigns_condition_countdowns_to_cards();
  test_apply_frozen_initializes_countdown_while_deferred();
  test_count_tappable_ikz_sources_ignores_zero_token_entity();
  test_get_tappable_ikz_cards_rejects_zero_token_entity();
  test_init_player_deck_raizen();
  test_azk01_001_card_def_and_instantiation();
  test_generated_card_def_stt03_015_has_correct_ikz_cost();
  test_generated_card_def_stt04_013_has_correct_gate_power();
  test_ability_registry_lookup();
  test_additional_card_abilities_get_sparse_action_indices();
  test_passive_observer_context_is_scoped_to_ability_entity();
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
  test_azk01_065_spell_damages_owner_leader_and_selected_target();
  test_azk01_087_spell_enters_effect_selection_and_revalidates_second_pick();
  test_azk01_089_main_ability_enters_effect_selection_and_allows_single_five_cost();
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
  test_azk01_122_gate_portal_selects_hand_entity_and_grants_charge();
  test_azk01_122_gate_portal_can_play_to_alley();
  test_gate_portal_effect_selection_stt03_002();
  test_gate_portal_no_valid_targets_auto_resolves_stt03_002();
  test_azk01_064_gate_portal_triggers_when_enters_garden();
  test_azk01_064_gate_portal_damages_all_enemy_garden_slots();
  test_main_phase_gate_portal_enters_effect_selection_stt03_002();
  test_main_phase_gate_portal_can_target_portaled_card_stt03_002();
  test_main_phase_gate_portal_can_target_damaged_portaled_card_stt04_002();
  test_main_phase_gate_portal_cannot_target_entity_damaged_last_turn_stt04_002();
  test_azk01_046_prefab_instances_inherit_garden_force_tapped_tag();
  test_summon_card_into_garden_taps_garden_force_tapped_cards();
  test_untap_all_cards_in_zone_keeps_garden_force_tapped_cards_tapped();
  test_summon_card_into_garden_logs_garden_force_tapped_metadata();
  test_gate_card_into_garden_logs_garden_force_tapped_metadata();
  test_engine_action_play_to_garden_logs_garden_force_tapped_metadata();
  test_azk01_103_cannot_sacrifice_itself_for_cost();
  test_prefab_instances_inherit_targeting_and_ikz_source_tags();
  test_can_target_leader_only_blocks_non_leader_attacks();
  test_stormglass_weapons_grant_and_remove_leader_alley_targeting();
  test_azk01_018_reduces_only_combat_damage_for_equipped_leader();
  test_azk01_018_does_not_reduce_combat_damage_for_nonleader_host();
  test_is_ikz_card_counts_as_source_only_in_garden();
  test_start_phase_skips_opening_draw_for_starting_player();
  test_start_phase_untaps_active_player_board_and_resources();
  test_start_phase_shocked_card_skips_next_owner_untap_after_manual_retap();
  test_end_phase_resets_alley_entity_health();
  test_observation_garden_slots_use_zone_index();
  test_leader_with_multiple_stt01_013_weapons_supports_attack_mask_and_combat();
  test_stt03_001_blocks_second_main_activation_same_turn();
  test_stt03_001_heals_only_once_for_first_destroy_or_sacrifice();
  test_stt03_012_heals_when_another_entity_is_destroyed_on_opponents_turn();
  test_stt03_012_heals_when_itself_is_destroyed_on_opponents_main_turn();
  test_stt03_012_heals_when_itself_is_destroyed_during_response_window();
  test_response_window_opens_for_hand_response_entity_azk01_035();
  test_response_window_opens_for_response_weapon_azk01_094();
  test_response_window_does_not_allow_playing_azk01_070_from_hand();
  test_alley_attack_opens_response_window_and_resolves_combat();
  test_declare_defender_taps_the_defending_entity();
  test_when_attacked_uses_final_defender_after_response_window();
  test_when_attacked_queues_at_combat_handoff_without_response();
  test_when_attacked_target_selection_runs_before_combat_resolution();
  test_azk01_006_return_to_hand_fizzles_combat();
  test_stt03_006_destroyed_draw_then_discard_uses_hand_targets();
  test_azk01_041_main_ability_is_activatable_with_valid_discard_weapon();
  test_azk01_096_effect_selection_accepts_friendly_alley_targets();
  test_azk01_097_on_play_discards_revealed_cards_when_no_weapons_found();
  test_azk01_097_on_play_adds_selected_weapon_to_hand_and_discards_rest();
  test_azk01_084_selection_includes_more_than_five_discard_targets();
  test_stt04_001_effect_selection_accepts_garden_and_alley_targets();
  test_stt04_017_cost_selection_allows_fifth_garden_sacrifice();
  test_azk01_059_triggers_after_nonlethal_damage();
  test_azk01_059_triggers_after_lethal_damage();
  test_azk01_062_auto_resolves_self_damage_when_no_redirect_targets();
  test_azk01_059_lethal_damage_prompts_in_engine_action_flow();
  test_azk01_057_start_of_turn_selects_self_when_enemy_garden_empty();
  test_azk01_057_start_of_turn_requires_enemy_second_target();
  test_stt03_010_heals_leader_when_both_combatants_are_destroyed();
  test_stt04_008_after_attacking_only_untaps_once_per_turn();
  test_stt04_008_after_attacking_does_not_prompt_if_destroyed();
  test_stt04_013_destroy_observer_only_untaps_once_per_turn();
  test_stt04_013_destroy_observer_does_not_clear_cooldown();
  test_azk01_039_when_equipped_grants_charge_and_clears_cooldown();
  test_azk01_112_on_play_from_alley_grants_in_play_charge_across_gate();
  test_azk01_112_on_play_in_garden_counts_self_when_checking_empty_garden();
  test_azk01_118_on_play_does_not_prompt_without_two_other_cards();
  test_azk01_118_on_play_does_not_prompt_without_enemy_garden_entity();
  test_azk01_028_returns_all_other_garden_entities_without_skipping();
  test_observation_hides_action_mask_while_trigger_queue_pending();

  // Game log tests
  printf("Running game log tests...\n");
  test_game_log_singleton_initialized();
  test_return_card_to_hand_uses_pending_hand_logs();
  test_move_selection_to_hand_uses_pending_hand_logs();
  test_deck_to_selection_to_hand_finalizes_each_committed_step();
  printf("Game log tests passed!\n");

  return 0;
}
