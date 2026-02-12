#include "utils/observation_util.h"

#include <stddef.h>

#include "components/abilities.h"
#include "utils/cli_rendering_util.h"
#include "utils/debug_log.h"
#include "validation/action_enumerator.h"

static void reset_legal_actions(ActionMaskObs *action_mask) {
  ecs_assert(action_mask != NULL, ECS_INVALID_PARAMETER, "Output mask is null");
  action_mask->legal_action_count = 0;
  for (size_t i = 0; i < AZK_ACTION_TYPE_COUNT; ++i) {
    action_mask->primary_action_mask[i] = false;
  }
}

static void populate_action_mask_from_set(const AzkActionMaskSet *mask_set,
                                          ActionMaskObs *out_mask) {
  ecs_assert(mask_set != NULL, ECS_INVALID_PARAMETER, "Mask set is null");
  ecs_assert(out_mask != NULL, ECS_INVALID_PARAMETER, "Output mask is null");

  size_t primary_action_count = AZK_ACTION_TYPE_COUNT;
  ecs_assert(primary_action_count == AZK_ACTION_HEAD0_SIZE,
             ECS_INVALID_PARAMETER,
             "Primary action count %zu out of bounds, expected %zu",
             primary_action_count, AZK_ACTION_TYPE_COUNT);

  for (size_t action_type = 0; action_type < primary_action_count;
       ++action_type) {
    out_mask->primary_action_mask[action_type] =
        mask_set->head0_mask[action_type] != 0;
  }

  uint16_t legal_action_count = mask_set->legal_action_count;
  if (legal_action_count > AZK_MAX_LEGAL_ACTIONS) {
    legal_action_count = AZK_MAX_LEGAL_ACTIONS;
  }

  for (uint16_t i = 0; i < legal_action_count; ++i) {
    const UserAction *action = &mask_set->legal_actions[i];
    ecs_assert(action->type >= 0 && action->type < AZK_ACTION_TYPE_COUNT,
               ECS_INVALID_PARAMETER, "Action type %d out of bounds",
               action->type);
    out_mask->legal_primary[i] = (uint8_t)action->type;
    out_mask->legal_sub1[i] = (uint8_t)action->subaction_1;
    out_mask->legal_sub2[i] = (uint8_t)action->subaction_2;
    out_mask->legal_sub3[i] = (uint8_t)action->subaction_3;
  }

  out_mask->legal_action_count = legal_action_count;

  if (!out_mask->primary_action_mask[ACT_NOOP]) {
    out_mask->primary_action_mask[ACT_NOOP] = true;
    if (out_mask->legal_action_count < AZK_MAX_LEGAL_ACTIONS) {
      uint16_t noop_idx = out_mask->legal_action_count++;
      out_mask->legal_primary[noop_idx] = (uint8_t)ACT_NOOP;
      out_mask->legal_sub1[noop_idx] = 0;
      out_mask->legal_sub2[noop_idx] = 0;
      out_mask->legal_sub3[noop_idx] = 0;
    }
  }
}

static ActionMaskObs build_observation_action_mask(ecs_world_t *world,
                                                   const GameState *gs,
                                                   int8_t player_index) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");

  ActionMaskObs action_mask = {0};
  reset_legal_actions(&action_mask);

  AzkActionMaskSet mask_set = {0};
  bool mask_result = azk_build_action_mask_for_player(world, gs, player_index, &mask_set);
  ecs_assert(mask_result, ECS_INVALID_PARAMETER, "Failed to build action mask");

  populate_action_mask_from_set(&mask_set, &action_mask);

  return action_mask;
}

static WeaponObservationData get_weapon_observation(ecs_world_t *world,
                                                    ecs_entity_t weapon_card) {
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

static uint8_t
set_attached_weapon_observations(ecs_world_t *world, ecs_entity_t parent_card,
                                 WeaponObservationData *weapons) {
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
        const char *card_code =
            parent_card_id != NULL ? parent_card_id->code : "unknown";
        cli_render_logf("[Observation] More than %d weapons attached to card "
                        "%s (%d); truncating to %d",
                        MAX_ATTACHED_WEAPONS, card_code, parent_card,
                        MAX_ATTACHED_WEAPONS);
        logged_overflow = true;
      }
    }
  }
  return weapon_count;
}

static CardObservationData get_card_observation(ecs_world_t *world,
                                                ecs_entity_t card,
                                                uint8_t fallback_zone_index) {
  CardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, card, Type);
  observation_data.id = *ecs_get(world, card, CardId);
  observation_data.tap_state = *ecs_get(world, card, TapState);
  observation_data.ikz_cost = *ecs_get(world, card, IKZCost);
  const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
  observation_data.zone_index =
      zone_index != NULL ? zone_index->index : fallback_zone_index;

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

  observation_data.weapon_count =
      set_attached_weapon_observations(world, card, observation_data.weapons);

  // Keyword tags
  observation_data.has_charge = ecs_has(world, card, Charge);
  observation_data.has_defender = ecs_has(world, card, Defender);
  observation_data.has_infiltrate = ecs_has(world, card, Infiltrate);

  // Check for status effects
  observation_data.is_frozen = ecs_has(world, card, Frozen);
  observation_data.is_shocked = ecs_has(world, card, Shocked);
  observation_data.is_effect_immune = ecs_has(world, card, EffectImmune);

  return observation_data;
}

static void
get_card_observation_array_for_zone(ecs_world_t *world, ecs_entity_t zone,
                                    CardObservationData *observation_data,
                                    size_t max_count, bool use_zone_index) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  size_t count = cards.count;
  if (count > max_count) {
    cli_render_logf("[Observation] Zone %d has %zu cards; truncating to %zu "
                    "for observation output",
                    zone, count, max_count);
    count = max_count;
  }

  if (!use_zone_index) {
    for (size_t i = 0; i < count; i++) {
      ecs_entity_t card = cards.ids[i];
      observation_data[i] = get_card_observation(world, card, (uint8_t)i);
    }
    return;
  }

  for (size_t i = 0; i < max_count; ++i) {
    observation_data[i] = (CardObservationData){0};
  }

  bool slot_has_card[max_count];
  for (size_t i = 0; i < max_count; ++i) {
    slot_has_card[i] = false;
  }

  size_t fallback_slot = 0;
  for (size_t i = 0; i < count; i++) {
    ecs_entity_t card = cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
    size_t target_index = max_count;

    if (zone_index != NULL && zone_index->index < max_count &&
        !slot_has_card[zone_index->index]) {
      target_index = zone_index->index;
    } else {
      if (zone_index == NULL) {
        AZK_DEBUG_WARN("[Observation] Card %d in zone %d missing ZoneIndex; "
                       "falling back to first empty slot",
                       (int)card, (int)zone);
      } else if (zone_index->index >= max_count) {
        AZK_DEBUG_WARN("[Observation] Card %d in zone %d has out-of-range "
                       "ZoneIndex %u; falling back to first empty slot",
                       (int)card, (int)zone, (unsigned)zone_index->index);
      } else {
        AZK_DEBUG_WARN("[Observation] Duplicate ZoneIndex %u in zone %d; "
                       "falling back to first empty slot",
                       (unsigned)zone_index->index, (int)zone);
      }

      while (fallback_slot < max_count && slot_has_card[fallback_slot]) {
        fallback_slot++;
      }
      if (fallback_slot < max_count) {
        target_index = fallback_slot;
        fallback_slot++;
      }
    }

    if (target_index >= max_count) {
      AZK_DEBUG_WARN("[Observation] Zone %d has more cards than slots; "
                     "dropping card %d from observation",
                     (int)zone, (int)card);
      continue;
    }

    observation_data[target_index] =
        get_card_observation(world, card, (uint8_t)target_index);
    slot_has_card[target_index] = true;
  }
}

static LeaderCardObservationData
get_leader_card_observation(ecs_world_t *world, ecs_entity_t leader_zone) {
  ecs_entities_t leader_cards = ecs_get_ordered_children(world, leader_zone);
  ecs_assert(leader_cards.count == 1, ECS_INVALID_PARAMETER,
             "Leader zone must contain exactly 1 card, got %d",
             leader_cards.count);
  ecs_entity_t leader_card = leader_cards.ids[0];

  LeaderCardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, leader_card, Type);
  observation_data.id = *ecs_get(world, leader_card, CardId);
  observation_data.cur_stats = *ecs_get(world, leader_card, CurStats);
  observation_data.tap_state = *ecs_get(world, leader_card, TapState);
  observation_data.has_charge = ecs_has(world, leader_card, Charge);
  observation_data.has_defender = ecs_has(world, leader_card, Defender);
  observation_data.has_infiltrate = ecs_has(world, leader_card, Infiltrate);
  observation_data.weapon_count = set_attached_weapon_observations(
      world, leader_card, observation_data.weapons);
  return observation_data;
}

static GateCardObservationData
get_gate_card_observation(ecs_world_t *world, ecs_entity_t gate_zone) {
  ecs_entities_t gate_cards = ecs_get_ordered_children(world, gate_zone);
  ecs_assert(gate_cards.count == 1, ECS_INVALID_PARAMETER,
             "Gate zone must contain exactly 1 card, got %d", gate_cards.count);
  ecs_entity_t gate_card = gate_cards.ids[0];

  GateCardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, gate_card, Type);
  observation_data.id = *ecs_get(world, gate_card, CardId);
  observation_data.tap_state = *ecs_get(world, gate_card, TapState);
  return observation_data;
}

static IKZCardObservationData
get_ikz_card_observation(ecs_world_t *world, ecs_entity_t card,
                         uint8_t fallback_zone_index) {
  IKZCardObservationData observation_data = {0};
  observation_data.type = *ecs_get(world, card, Type);
  observation_data.id = *ecs_get(world, card, CardId);
  observation_data.tap_state = *ecs_get(world, card, TapState);
  const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
  observation_data.zone_index =
      zone_index != NULL ? zone_index->index : fallback_zone_index;
  return observation_data;
}

static void
get_ikz_card_observations_for_zone(ecs_world_t *world, ecs_entity_t zone,
                                   IKZCardObservationData *observation_data) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  int32_t count = cards.count;
  for (int32_t i = 0; i < count; i++) {
    ecs_entity_t card = cards.ids[i];
    observation_data[i] = get_ikz_card_observation(world, card, (uint8_t)i);
  }
}

static uint8_t get_zone_card_count(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  return cards.count;
}

// Populate selection from AbilityContext during ability phases (preserves indices/gaps)
static void get_selection_from_ability_context(ecs_world_t *world,
                                               const AbilityContext *ctx,
                                               CardObservationData *observation_data,
                                               uint8_t *out_count) {
  for (int i = 0; i < ctx->selection_count && i < MAX_SELECTION_ZONE_SIZE; i++) {
    ecs_entity_t card = ctx->selection_cards[i];
    if (card != 0) {
      observation_data[i] = get_card_observation(world, card, (uint8_t)i);
      observation_data[i].zone_index = (uint8_t)i;  // Preserve original index
    } else {
      // Empty slot - leave as zero-initialized (id.code will be NULL)
      observation_data[i] = (CardObservationData){0};
      observation_data[i].zone_index = (uint8_t)i;
    }
  }
  // Preserve original selection_count so clients can align action indices
  *out_count = ctx->selection_count;
}

static bool player_has_ready_ikz_token(ecs_world_t *world,
                                       ecs_entity_t player) {
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

ObservationData create_observation_data(ecs_world_t *world,
                                        int8_t player_index) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");
  ecs_assert(player_index >= 0 && player_index < MAX_PLAYERS_PER_MATCH,
             ECS_INVALID_PARAMETER, "Player index %d out of bounds",
             player_index);
  const int8_t opponent_player_index =
      (player_index + 1) % MAX_PLAYERS_PER_MATCH;

  const PlayerZones *my_zones = &gs->zones[player_index];
  const PlayerZones *opponent_zones = &gs->zones[opponent_player_index];

  const ecs_entity_t my_player = gs->players[player_index];
  const ecs_entity_t opponent_player = gs->players[opponent_player_index];

  MyObservationData my_observation_data = {0};
  my_observation_data.leader =
      get_leader_card_observation(world, my_zones->leader);
  my_observation_data.gate = get_gate_card_observation(world, my_zones->gate);
  // TODO: Need to handle weapon and spell cards also
  get_card_observation_array_for_zone(world, my_zones->hand,
                                      my_observation_data.hand, MAX_HAND_SIZE,
                                      false);
  get_card_observation_array_for_zone(world, my_zones->alley,
                                      my_observation_data.alley, ALLEY_SIZE,
                                      true);
  get_card_observation_array_for_zone(world, my_zones->garden,
                                      my_observation_data.garden, GARDEN_SIZE,
                                      true);
  get_card_observation_array_for_zone(
      world, my_zones->discard, my_observation_data.discard, MAX_DECK_SIZE,
      false);

  // For selection zone, check if we're in an ability phase that uses selection
  // If so, read from AbilityContext to preserve original indices (with gaps for picked cards)
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  bool use_ability_context_selection =
      ctx && ctx->selection_count > 0 &&
      (ctx->phase == ABILITY_PHASE_SELECTION_PICK ||
       ctx->phase == ABILITY_PHASE_BOTTOM_DECK);

  if (use_ability_context_selection) {
    get_selection_from_ability_context(world, ctx, my_observation_data.selection,
                                       &my_observation_data.selection_count);
  } else {
    get_card_observation_array_for_zone(world, my_zones->selection,
                                        my_observation_data.selection,
                                        MAX_SELECTION_ZONE_SIZE, false);
    my_observation_data.selection_count =
        get_zone_card_count(world, my_zones->selection);
  }

  get_ikz_card_observations_for_zone(world, my_zones->ikz_area,
                                     my_observation_data.ikz_area);
  my_observation_data.deck_count = get_zone_card_count(world, my_zones->deck);
  my_observation_data.ikz_pile_count =
      get_zone_card_count(world, my_zones->ikz_pile);
  my_observation_data.has_ikz_token =
      player_has_ready_ikz_token(world, my_player);

  OpponentObservationData opponent_observation_data = {0};
  opponent_observation_data.leader =
      get_leader_card_observation(world, opponent_zones->leader);
  opponent_observation_data.gate =
      get_gate_card_observation(world, opponent_zones->gate);
  get_card_observation_array_for_zone(world, opponent_zones->alley,
                                      opponent_observation_data.alley,
                                      ALLEY_SIZE, true);
  get_card_observation_array_for_zone(world, opponent_zones->garden,
                                      opponent_observation_data.garden,
                                      GARDEN_SIZE, true);
  get_card_observation_array_for_zone(world, opponent_zones->discard,
                                      opponent_observation_data.discard,
                                      MAX_DECK_SIZE, false);
  get_ikz_card_observations_for_zone(world, opponent_zones->ikz_area,
                                     opponent_observation_data.ikz_area);
  opponent_observation_data.hand_count =
      get_zone_card_count(world, opponent_zones->hand);
  opponent_observation_data.deck_count =
      get_zone_card_count(world, opponent_zones->deck);
  opponent_observation_data.ikz_pile_count =
      get_zone_card_count(world, opponent_zones->ikz_pile);
  opponent_observation_data.has_ikz_token =
      player_has_ready_ikz_token(world, opponent_player);

  ObservationData observation_data = {0};
  observation_data.my_observation_data = my_observation_data;
  observation_data.opponent_observation_data = opponent_observation_data;
  observation_data.phase = gs->phase;
  observation_data.action_mask =
      build_observation_action_mask(world, gs, player_index);

  return observation_data;
}

bool is_game_over(ecs_world_t *world) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");
  return gs->winner != -1;
}
