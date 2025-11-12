#include "utils/observation_util.h"
#include "utils/cli_rendering_util.h"

static WeaponObservationData get_weapon_observation(ecs_world_t *world, ecs_entity_t weapon_card) {
  WeaponObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, weapon_card, Type);
  observation_data.id = *ecs_get(world, weapon_card, CardId);
  const BaseStats *base_stats = ecs_get(world, weapon_card, BaseStats);
  if (base_stats != NULL) {
    observation_data.base_stats = *base_stats;
  }
  const IKZCost *ikz_cost = ecs_get(world, weapon_card, IKZCost);
  if (ikz_cost != NULL) {
    observation_data.ikz_cost = *ikz_cost;
  }
  return observation_data;
}

static uint8_t set_attached_weapon_observations(ecs_world_t *world, ecs_entity_t parent_card, WeaponObservationData *weapons) {
  uint8_t weapon_count = 0;
  bool logged_overflow = false;
  ecs_iter_t it = ecs_children(world, parent_card);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t child = it.entities[i];
      if (!ecs_has_id(world, child, TWeapon)) {
        continue;
      }
      if (weapon_count < MAX_ATTACHED_WEAPONS) {
        weapons[weapon_count++] = get_weapon_observation(world, child);
      } else if (!logged_overflow) {
        const CardId *parent_card_id = ecs_get(world, parent_card, CardId);
        const char *card_code = parent_card_id != NULL ? parent_card_id->code : "unknown";
        cli_render_logf(
          "[Observation] More than %d weapons attached to card %s (%d); truncating to %d",
          MAX_ATTACHED_WEAPONS,
          card_code,
          parent_card,
          MAX_ATTACHED_WEAPONS
        );
        logged_overflow = true;
      }
    }
  }
  return weapon_count;
}

static CardObservationData get_card_observation(ecs_world_t *world, ecs_entity_t card) {
  CardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, card, Type);
  observation_data.id = *ecs_get(world, card, CardId);
  observation_data.tap_state = *ecs_get(world, card, TapState);
  observation_data.ikz_cost = *ecs_get(world, card, IKZCost);
  const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
  if (zone_index != NULL) {
    observation_data.has_zone_index = true;
    observation_data.zone_index = zone_index->index;
  } else {
    observation_data.has_zone_index = false;
    observation_data.zone_index = 0;
  }

  const CurStats *cur_stats = ecs_get(world, card, CurStats);
  if (cur_stats != NULL) {
    observation_data.has_cur_stats = true;
    observation_data.cur_stats = *cur_stats;
  } else {
    observation_data.has_cur_stats = false;
  }

  const GatePoints *gate_points = ecs_get(world, card, GatePoints);
  if (gate_points != NULL) {
    observation_data.has_gate_points = true;
    observation_data.gate_points = *gate_points;
  } else {
    observation_data.has_gate_points = false;
  }

  observation_data.weapon_count = set_attached_weapon_observations(world, card, observation_data.weapons);

  return observation_data;
}

static void get_card_observation_array_for_zone(
  ecs_world_t *world,
  ecs_entity_t zone,
  CardObservationData *observation_data,
  size_t max_count
) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  size_t count = cards.count;
  if (count > max_count) {
    cli_render_logf(
      "[Observation] Zone %d has %zu cards; truncating to %zu for observation output",
      zone,
      count,
      max_count
    );
    count = max_count;
  }
  for (size_t i = 0; i < count; i++) {
    ecs_entity_t card = cards.ids[i];
    observation_data[i] = get_card_observation(world, card);
  }
}

static LeaderCardObservationData get_leader_card_observation(ecs_world_t *world, ecs_entity_t leader_zone) {
  ecs_entities_t leader_cards = ecs_get_ordered_children(world, leader_zone);
  ecs_assert(leader_cards.count == 1, ECS_INVALID_PARAMETER, "Leader zone must contain exactly 1 card, got %d", leader_cards.count);
  ecs_entity_t leader_card = leader_cards.ids[0];

  LeaderCardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, leader_card, Type);
  observation_data.id = *ecs_get(world, leader_card, CardId);
  observation_data.cur_stats = *ecs_get(world, leader_card, CurStats);
  observation_data.tap_state = *ecs_get(world, leader_card, TapState);
  observation_data.weapon_count = set_attached_weapon_observations(world, leader_card, observation_data.weapons);
  return observation_data;
}

static GateCardObservationData get_gate_card_observation(ecs_world_t *world, ecs_entity_t gate_zone) {
  ecs_entities_t gate_cards = ecs_get_ordered_children(world, gate_zone);
  ecs_assert(gate_cards.count == 1, ECS_INVALID_PARAMETER, "Gate zone must contain exactly 1 card, got %d", gate_cards.count);
  ecs_entity_t gate_card = gate_cards.ids[0];

  GateCardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, gate_card, Type);
  observation_data.id = *ecs_get(world, gate_card, CardId);
  observation_data.tap_state = *ecs_get(world, gate_card, TapState);
  return observation_data;
}
  
static IKZCardObservationData get_ikz_card_observation(ecs_world_t *world, ecs_entity_t card) {
  IKZCardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, card, Type);
  observation_data.id = *ecs_get(world, card, CardId);
  observation_data.tap_state = *ecs_get(world, card, TapState);
  return observation_data;
}

static void get_ikz_card_observations_for_zone(ecs_world_t *world, ecs_entity_t zone, IKZCardObservationData *observation_data) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  int32_t count = cards.count;
  for (int32_t i = 0; i < count; i++) {
    ecs_entity_t card = cards.ids[i];
    observation_data[i] = get_ikz_card_observation(world, card);
  }
}

static uint8_t get_zone_card_count(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  return cards.count;
}

static bool player_has_ready_ikz_token(ecs_world_t *world, ecs_entity_t player) {
  const IKZToken *ikz_token = ecs_get(world, player, IKZToken);
  if (ikz_token == NULL || ikz_token->ikz_token == 0) {
    return false;
  }

  const TapState *tap_state = ecs_get(world, ikz_token->ikz_token, TapState);
  if (tap_state == NULL) {
    return false;
  }

  return tap_state->tapped == 0;
}

ObservationData create_observation_data(ecs_world_t *world, int8_t player_index) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");
  ecs_assert(
    player_index >= 0 && player_index < MAX_PLAYERS_PER_MATCH,
    ECS_INVALID_PARAMETER,
    "Player index %d out of bounds",
    player_index
  );
  const int8_t opponent_player_index = (player_index + 1) % MAX_PLAYERS_PER_MATCH;

  const PlayerZones *my_zones = &gs->zones[player_index];
  const PlayerZones *opponent_zones = &gs->zones[opponent_player_index];

  const ecs_entity_t my_player = gs->players[player_index];
  const ecs_entity_t opponent_player = gs->players[opponent_player_index];

  MyObservationData my_observation_data = {0};
  my_observation_data.leader = get_leader_card_observation(world, my_zones->leader);
  my_observation_data.gate = get_gate_card_observation(world, my_zones->gate);
  // TODO: Need to handle weapon and spell cards also
  get_card_observation_array_for_zone(world, my_zones->hand, my_observation_data.hand, MAX_HAND_SIZE);
  get_card_observation_array_for_zone(world, my_zones->alley, my_observation_data.alley, ALLEY_SIZE);
  get_card_observation_array_for_zone(world, my_zones->garden, my_observation_data.garden, GARDEN_SIZE);
  get_card_observation_array_for_zone(world, my_zones->discard, my_observation_data.discard, MAX_DECK_SIZE);
  get_ikz_card_observations_for_zone(world, my_zones->ikz_area, my_observation_data.ikz_area);
  my_observation_data.deck_count = get_zone_card_count(world, my_zones->deck);
  my_observation_data.ikz_pile_count = get_zone_card_count(world, my_zones->ikz_pile);
  my_observation_data.has_ikz_token = player_has_ready_ikz_token(world, my_player);

  OpponentObservationData opponent_observation_data = {0};
  opponent_observation_data.leader = get_leader_card_observation(world, opponent_zones->leader);
  opponent_observation_data.gate = get_gate_card_observation(world, opponent_zones->gate);
  get_card_observation_array_for_zone(world, opponent_zones->alley, opponent_observation_data.alley, ALLEY_SIZE);
  get_card_observation_array_for_zone(world, opponent_zones->garden, opponent_observation_data.garden, GARDEN_SIZE);
  get_card_observation_array_for_zone(world, opponent_zones->discard, opponent_observation_data.discard, MAX_DECK_SIZE);
  get_ikz_card_observations_for_zone(world, opponent_zones->ikz_area, opponent_observation_data.ikz_area);
  opponent_observation_data.hand_count = get_zone_card_count(world, opponent_zones->hand);
  opponent_observation_data.deck_count = get_zone_card_count(world, opponent_zones->deck);
  opponent_observation_data.ikz_pile_count = get_zone_card_count(world, opponent_zones->ikz_pile);
  opponent_observation_data.has_ikz_token = player_has_ready_ikz_token(world, opponent_player);

  ObservationData observation_data = {0};
  observation_data.my_observation_data = my_observation_data;
  observation_data.opponent_observation_data = opponent_observation_data;
  observation_data.phase = gs->phase;
  return observation_data;
}

bool is_game_over(ecs_world_t *world) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");
  return gs->winner != -1;
}
