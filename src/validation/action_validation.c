#include "validation/action_validation.h"

#include <string.h>

#include "abilities/ability.h"
#include "abilities/ability_runtime.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

#define VALIDATION_LOG(log_errors, ...) \
  do {                                  \
    if (log_errors) {                   \
      cli_render_logf(__VA_ARGS__);     \
    }                                   \
  } while (0)

static bool ensure_active_player(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  bool log_errors
) {
  uint8_t player_number = get_player_number(world, player);
  if (player_number != gs->active_player_index) {
    VALIDATION_LOG(log_errors, "Player %d is not active", player_number);
    return false;
  }
  return true;
}

static bool ability_selection_blocks_actions(
  ecs_world_t *world,
  ecs_entity_t player,
  ActionType type,
  bool log_errors
) {
  const AbilityContext *actx = ecs_singleton_get(world, AbilityContext);
  if (actx == NULL || !actx->has_pending) {
    return false;
  }
  if (actx->player != player) {
    return false;
  }
  switch (type) {
    case ACT_SELECT_COST_TARGET:
    case ACT_SELECT_EFFECT_TARGET:
    case ACT_NOOP:
    case ACT_ACCEPT_TRIGGERED_ABILITY:
      return false;
    default:
      VALIDATION_LOG(log_errors, "Ability selection pending; action %d not allowed", type);
      return true;
  }
}

static ecs_entity_t target_zone_for_placement(
  const GameState *gs,
  uint8_t player_number,
  ZonePlacementType placement_type
) {
  switch (placement_type) {
    case ZONE_GARDEN:
      return gs->zones[player_number].garden;
    case ZONE_ALLEY:
      return gs->zones[player_number].alley;
    default:
      return 0;
  }
}

static uint8_t zone_capacity_for_placement(ZonePlacementType placement_type) {
  (void)placement_type;
  return GARDEN_SIZE; // Garden and alley share the same capacity
}

static bool fetch_ikz_payment(
  ecs_world_t *world,
  ecs_entity_t ikz_zone,
  uint8_t ikz_cost,
  bool use_ikz_token,
  bool log_errors,
  ecs_entity_t out_cards[AZK_MAX_IKZ_PAYMENT],
  uint8_t *out_count
) {
  if (ikz_cost == 0) {
    *out_count = 0;
    return true;
  }

  ecs_assert(ikz_cost <= AZK_MAX_IKZ_PAYMENT, ECS_INVALID_PARAMETER, "IKZ cost %d exceeds buffer", ikz_cost);

  uint8_t count = 0;
  if (get_tappable_ikz_cards(
        world,
        ikz_zone,
        ikz_cost,
        &count,
        out_cards,
        use_ikz_token
      ) < 0) {
    return false;
  }

  if (count < ikz_cost) {
    VALIDATION_LOG(log_errors, "Not enough untapped IKZ cards");
    return false;
  }

  *out_count = ikz_cost;
  return true;
}

bool azk_validate_play_entity_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ZonePlacementType placement_type,
  const UserAction *action,
  bool log_errors,
  PlayEntityIntent *out_intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "Action is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  if (ability_selection_blocks_actions(world, player, action->type, log_errors)) {
    return false;
  }

  int hand_index = action->subaction_1;
  int zone_index = action->subaction_2;
  bool use_ikz_token = action->subaction_3 != 0;

  if (hand_index < 0 || hand_index >= MAX_HAND_SIZE) {
    VALIDATION_LOG(log_errors, "Hand index %d out of bounds", hand_index);
    return false;
  }

  if (zone_index < 0 || zone_index >= GARDEN_SIZE) {
    VALIDATION_LOG(log_errors, "Zone index %d out of bounds", zone_index);
    return false;
  }

  uint8_t player_number = get_player_number(world, player);
  ecs_entity_t hand_zone = gs->zones[player_number].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand_zone);
  if (hand_index >= hand_cards.count) {
    VALIDATION_LOG(log_errors, "Hand index %d exceeds card count %d", hand_index, hand_cards.count);
    return false;
  }

  ecs_entity_t card = hand_cards.ids[hand_index];
  if (card == 0) {
    VALIDATION_LOG(log_errors, "No card found at hand index %d", hand_index);
    return false;
  }

  if (!is_card_type(world, card, CARD_TYPE_ENTITY)) {
    VALIDATION_LOG(log_errors, "Card %d is not an entity", card);
    return false;
  }

  const IKZCost *ikz_cost = ecs_get(world, card, IKZCost);
  ecs_assert(ikz_cost != NULL, ECS_INVALID_PARAMETER, "IKZCost missing for card %d", card);

  ecs_entity_t target_zone = target_zone_for_placement(gs, player_number, placement_type);
  if (target_zone == 0) {
    VALIDATION_LOG(log_errors, "Target zone missing for placement %d", placement_type);
    return false;
  }

  ecs_entity_t displaced_card = find_card_in_zone_index(world, target_zone, zone_index);
  ecs_entities_t zone_cards = ecs_get_ordered_children(world, target_zone);
  uint8_t capacity = zone_capacity_for_placement(placement_type);
  bool zone_full = zone_cards.count >= capacity;
  if (displaced_card != 0 && !zone_full) {
    VALIDATION_LOG(log_errors, "Zone %d at index %d already occupied", target_zone, zone_index);
    return false;
  }

  ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT] = {0};
  uint8_t ikz_card_count = 0;
  ecs_entity_t ikz_zone = gs->zones[player_number].ikz_area;
  if (!fetch_ikz_payment(
        world,
        ikz_zone,
        ikz_cost->ikz_cost,
        use_ikz_token,
        log_errors,
        ikz_cards,
        &ikz_card_count
      )) {
    return false;
  }

  if (out_intent) {
    PlayEntityIntent intent = {
      .player = player,
      .card = card,
      .placement_type = placement_type,
      .target_zone = target_zone,
      .zone_index = zone_index,
      .use_ikz_token = use_ikz_token,
      .ikz_card_count = ikz_card_count,
      .displaced_card = displaced_card
    };
    memcpy(intent.ikz_cards, ikz_cards, sizeof(ecs_entity_t) * ikz_card_count);
    *out_intent = intent;
  }

  return true;
}

bool azk_validate_gate_portal_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  GatePortalIntent *out_intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "Action is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  if (ability_selection_blocks_actions(world, player, action->type, log_errors)) {
    return false;
  }

  int alley_index = action->subaction_1;
  int garden_index = action->subaction_2;

  if (alley_index < 0 || alley_index >= ALLEY_SIZE) {
    VALIDATION_LOG(log_errors, "Alley index %d out of bounds", alley_index);
    return false;
  }

  if (garden_index < 0 || garden_index >= GARDEN_SIZE) {
    VALIDATION_LOG(log_errors, "Garden index %d out of bounds", garden_index);
    return false;
  }

  uint8_t player_number = get_player_number(world, player);
  ecs_entity_t gate_zone = gs->zones[player_number].gate;
  ecs_entity_t gate_card = find_gate_card_in_zone(world, gate_zone);
  if (is_card_tapped(world, gate_card)) {
    VALIDATION_LOG(log_errors, "Gate card %d is tapped", gate_card);
    return false;
  }

  ecs_entity_t alley_zone = gs->zones[player_number].alley;
  ecs_entity_t alley_card = find_card_in_zone_index(world, alley_zone, alley_index);
  if (alley_card == 0) {
    VALIDATION_LOG(log_errors, "No card at alley index %d", alley_index);
    return false;
  }
  if (is_card_tapped(world, alley_card)) {
    VALIDATION_LOG(log_errors, "Alley card %d is tapped", alley_card);
    return false;
  }

  ecs_entity_t garden_zone = gs->zones[player_number].garden;
  ecs_entity_t displaced_card = find_card_in_zone_index(world, garden_zone, garden_index);
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden_zone);
  bool garden_full = garden_cards.count >= GARDEN_SIZE;
  if (displaced_card != 0 && !garden_full) {
    VALIDATION_LOG(log_errors, "Garden index %d already occupied", garden_index);
    return false;
  }

  if (out_intent) {
    GatePortalIntent intent = {
      .player = player,
      .alley_card = alley_card,
      .target_zone = garden_zone,
      .garden_index = garden_index,
      .displaced_card = displaced_card,
      .gate_card = gate_card
    };
    *out_intent = intent;
  }

  return true;
}

bool azk_validate_attack_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  AttackIntent *out_intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "Action is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  if (ability_selection_blocks_actions(world, player, action->type, log_errors)) {
    return false;
  }

  int attacker_index = action->subaction_1;
  int defender_index = action->subaction_2;

  if (attacker_index < 0 || attacker_index > GARDEN_SIZE) {
    VALIDATION_LOG(log_errors, "Attacker index %d out of bounds", attacker_index);
    return false;
  }
  if (defender_index < 0 || defender_index > GARDEN_SIZE) {
    VALIDATION_LOG(log_errors, "Defender index %d out of bounds", defender_index);
    return false;
  }

  uint8_t attacking_player_number = get_player_number(world, player);
  uint8_t defending_player_number = (attacking_player_number + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t defending_player = gs->players[defending_player_number];

  ecs_entity_t attacking_card = 0;
  bool attacker_is_leader = attacker_index == GARDEN_SIZE;
  if (attacker_is_leader) {
    attacking_card = find_leader_card_in_zone(world, gs->zones[attacking_player_number].leader);
    const CurStats *cur_stats = ecs_get(world, attacking_card, CurStats);
    ecs_assert(cur_stats != NULL, ECS_INVALID_PARAMETER, "Leader cur stats missing");
    if (cur_stats->cur_atk <= 0) {
      VALIDATION_LOG(log_errors, "Leader cannot attack without weapon");
      return false;
    }
  } else {
    attacking_card = find_card_in_zone_index(world, gs->zones[attacking_player_number].garden, attacker_index);
    if (attacking_card == 0) {
      VALIDATION_LOG(log_errors, "No attacker at garden index %d", attacker_index);
      return false;
    }
  }

  if (is_card_tapped(world, attacking_card) || is_card_cooldown(world, attacking_card)) {
    VALIDATION_LOG(log_errors, "Attacking card %d is tapped or on cooldown", attacking_card);
    return false;
  }

  ecs_entity_t defending_card = 0;
  bool defender_is_leader = defender_index == GARDEN_SIZE;
  if (defender_is_leader) {
    defending_card = find_leader_card_in_zone(world, gs->zones[defending_player_number].leader);
  } else {
    defending_card = find_card_in_zone_index(world, gs->zones[defending_player_number].garden, defender_index);
    if (defending_card == 0) {
      VALIDATION_LOG(log_errors, "No defender at garden index %d", defender_index);
      return false;
    }
    if (!is_card_tapped(world, defending_card)) {
      VALIDATION_LOG(log_errors, "Defender card %d is not tapped", defending_card);
      return false;
    }
  }

  if (out_intent) {
    AttackIntent intent = {
      .attacking_player = player,
      .defending_player = defending_player,
      .attacking_card = attacking_card,
      .defending_card = defending_card,
      .attacker_index = attacker_index,
      .defender_index = defender_index,
      .attacker_is_leader = attacker_is_leader
    };
    *out_intent = intent;
  }

  return true;
}

bool azk_validate_attach_weapon_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  AttachWeaponIntent *out_intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "Action is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  if (ability_selection_blocks_actions(world, player, action->type, log_errors)) {
    return false;
  }

  int hand_index = action->subaction_1;
  int entity_index = action->subaction_2;
  bool use_ikz_token = action->subaction_3 != 0;

  if (hand_index < 0 || hand_index >= MAX_HAND_SIZE) {
    VALIDATION_LOG(log_errors, "Hand index %d out of bounds", hand_index);
    return false;
  }
  if (entity_index < 0 || entity_index > GARDEN_SIZE) {
    VALIDATION_LOG(log_errors, "Entity index %d out of bounds", entity_index);
    return false;
  }

  uint8_t player_number = get_player_number(world, player);
  ecs_entity_t hand_zone = gs->zones[player_number].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand_zone);
  if (hand_index >= hand_cards.count) {
    VALIDATION_LOG(log_errors, "Hand index %d exceeds card count %d", hand_index, hand_cards.count);
    return false;
  }

  ecs_entity_t weapon_card = hand_cards.ids[hand_index];
  if (!is_card_type(world, weapon_card, CARD_TYPE_WEAPON)) {
    VALIDATION_LOG(log_errors, "Card %d is not a weapon", weapon_card);
    return false;
  }

  ecs_entity_t target_card = 0;
  bool target_is_leader = entity_index == GARDEN_SIZE;
  if (target_is_leader) {
    target_card = find_leader_card_in_zone(world, gs->zones[player_number].leader);
  } else {
    target_card = find_card_in_zone_index(world, gs->zones[player_number].garden, entity_index);
    if (target_card == 0) {
      VALIDATION_LOG(log_errors, "No entity at garden index %d", entity_index);
      return false;
    }
  }

  const IKZCost *ikz_cost = ecs_get(world, weapon_card, IKZCost);
  ecs_assert(ikz_cost != NULL, ECS_INVALID_PARAMETER, "IKZCost missing for weapon %d", weapon_card);

  ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT] = {0};
  uint8_t ikz_card_count = 0;
  ecs_entity_t ikz_zone = gs->zones[player_number].ikz_area;
  if (!fetch_ikz_payment(
        world,
        ikz_zone,
        ikz_cost->ikz_cost,
        use_ikz_token,
        log_errors,
        ikz_cards,
        &ikz_card_count
      )) {
    return false;
  }

  if (out_intent) {
    AttachWeaponIntent intent = {
      .player = player,
      .weapon_card = weapon_card,
      .target_card = target_card,
      .target_is_leader = target_is_leader,
      .use_ikz_token = use_ikz_token,
      .ikz_card_count = ikz_card_count
    };
    memcpy(intent.ikz_cards, ikz_cards, sizeof(ecs_entity_t) * ikz_card_count);
    *out_intent = intent;
  }

  return true;
}

bool azk_validate_simple_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ActionType type,
  bool log_errors
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  const AbilityContext *actx = ecs_singleton_get(world, AbilityContext);
  bool has_pending_for_player = actx != NULL && actx->has_pending && actx->player == player;

  switch (type) {
    case ACT_NOOP:
      if (has_pending_for_player) {
        return true;
      }
      if (gs->phase == PHASE_MAIN || gs->phase == PHASE_PREGAME_MULLIGAN || gs->phase == PHASE_RESPONSE_WINDOW) {
        return true;
      }
      VALIDATION_LOG(log_errors, "NOOP not allowed in phase %d", gs->phase);
      return false;
    case ACT_MULLIGAN_SHUFFLE:
      if (gs->phase != PHASE_PREGAME_MULLIGAN) {
        VALIDATION_LOG(log_errors, "Mulligan action not allowed in phase %d", gs->phase);
        return false;
      }
      return true;
    default:
      VALIDATION_LOG(log_errors, "No validator for action type %d", type);
      return false;
  }
}

static bool ability_timing_allows_phase(const AbilityDef *def, Phase phase) {
  switch (phase) {
    case PHASE_MAIN:
      return def->timing == ABILITY_TIMING_MAIN;
    case PHASE_RESPONSE_WINDOW:
      return def->timing == ABILITY_TIMING_RESPONSE;
    default:
      return false;
  }
}

static ecs_entity_t ability_source_card(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  AbilitySourceZone zone,
  int slot_index,
  bool log_errors
) {
  uint8_t player_number = get_player_number(world, player);
  switch (zone) {
    case ABILITY_SOURCE_GARDEN:
      if (slot_index < 0 || slot_index >= GARDEN_SIZE) {
        VALIDATION_LOG(log_errors, "Garden ability slot %d out of bounds", slot_index);
        return 0;
      }
      return find_card_in_zone_index(world, gs->zones[player_number].garden, slot_index);
    case ABILITY_SOURCE_ALLEY:
      if (slot_index < 0 || slot_index >= ALLEY_SIZE) {
        VALIDATION_LOG(log_errors, "Alley ability slot %d out of bounds", slot_index);
        return 0;
      }
      return find_card_in_zone_index(world, gs->zones[player_number].alley, slot_index);
    case ABILITY_SOURCE_LEADER:
      return find_leader_card_in_zone(world, gs->zones[player_number].leader);
    case ABILITY_SOURCE_GATE:
      return find_gate_card_in_zone(world, gs->zones[player_number].gate);
    case ABILITY_SOURCE_HAND: {
      if (slot_index < 0 || slot_index >= MAX_HAND_SIZE) {
        VALIDATION_LOG(log_errors, "Hand ability slot %d out of bounds", slot_index);
        return 0;
      }
      ecs_entities_t hand_cards = ecs_get_ordered_children(world, gs->zones[player_number].hand);
      if (slot_index >= hand_cards.count) {
        VALIDATION_LOG(log_errors, "Hand ability slot %d exceeds card count %d", slot_index, hand_cards.count);
        return 0;
      }
      return hand_cards.ids[slot_index];
    }
    default:
      VALIDATION_LOG(log_errors, "Unknown ability source zone %d", zone);
      return 0;
  }
}

bool azk_validate_activate_ability_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  ActivateAbilityIntent *out_intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "Action is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  if (ability_selection_blocks_actions(world, player, action->type, log_errors)) {
    return false;
  }

  AbilitySourceZone zone = (AbilitySourceZone)action->subaction_1;
  int slot_index = action->subaction_2;
  bool use_ikz_token = action->subaction_3 != 0;

  if (zone < ABILITY_SOURCE_GARDEN || zone > ABILITY_SOURCE_HAND) {
    VALIDATION_LOG(log_errors, "Ability source zone %d invalid", zone);
    return false;
  }

  ecs_entity_t source_card = ability_source_card(world, gs, player, zone, slot_index, log_errors);
  if (source_card == 0) {
    VALIDATION_LOG(log_errors, "No card found for ability activation");
    return false;
  }

  const CardId *card_id = ecs_get(world, source_card, CardId);
  ecs_assert(card_id != NULL, ECS_INVALID_PARAMETER, "CardId missing for card %d", source_card);

  const AbilityDef *def = azk_find_ability_def(card_id->id, 0);
  if (def == NULL) {
    VALIDATION_LOG(log_errors, "No ability definition for card id %d", card_id->id);
    return false;
  }

  if (def->kind != ABILITY_KIND_ACTIVATED) {
    VALIDATION_LOG(log_errors, "Ability is not activatable");
    return false;
  }

  if (!ability_timing_allows_phase(def, gs->phase)) {
    VALIDATION_LOG(log_errors, "Ability not allowed in current phase");
    return false;
  }

  if (def->source_zone != zone) {
    VALIDATION_LOG(log_errors, "Ability source zone mismatch");
    return false;
  }

  if (def->once_per_turn && azk_is_once_per_turn_used(world, source_card, def->ability_index)) {
    VALIDATION_LOG(log_errors, "Ability already used this turn");
    return false;
  }

  if (def->cost.tap_self && (is_card_tapped(world, source_card) || is_card_cooldown(world, source_card))) {
    VALIDATION_LOG(log_errors, "Card is tapped or on cooldown");
    return false;
  }

  if (def->cost.sacrifice_self && (ecs_has_id(world, source_card, TLeader) || ecs_has_id(world, source_card, TGate))) {
    VALIDATION_LOG(log_errors, "Cannot sacrifice leader or gate");
    return false;
  }

  if (azk_ability_has_target_requirement(&def->cost_targets) &&
      !azk_ability_targets_available(world, gs, player, source_card, def->cost_targets, ABILITY_TARGET_NONE)) {
    VALIDATION_LOG(log_errors, "No valid cost targets");
    return false;
  }

  if (azk_ability_has_target_requirement(&def->effect_targets) &&
      !azk_ability_targets_available(world, gs, player, source_card, def->effect_targets, ABILITY_TARGET_NONE)) {
    VALIDATION_LOG(log_errors, "No valid effect targets");
    return false;
  }

  ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT] = {0};
  uint8_t ikz_card_count = 0;
  ecs_entity_t ikz_zone = gs->zones[get_player_number(world, player)].ikz_area;
  if (!fetch_ikz_payment(world, ikz_zone, def->cost.ikz_cost, use_ikz_token, log_errors, ikz_cards, &ikz_card_count)) {
    return false;
  }

  if (out_intent) {
    ActivateAbilityIntent intent = {
      .player = player,
      .source_card = source_card,
      .ability = def,
      .use_ikz_token = use_ikz_token,
      .ikz_card_count = ikz_card_count
    };
    memcpy(intent.ikz_cards, ikz_cards, sizeof(ecs_entity_t) * ikz_card_count);
    *out_intent = intent;
  }

  return true;
}

static ecs_entity_t resolve_selection_target(
  ecs_world_t *world,
  const GameState *gs,
  int target_player,
  int zone_kind,
  int slot_index,
  bool log_errors
) {
  if (target_player < 0 || target_player >= MAX_PLAYERS_PER_MATCH) {
    VALIDATION_LOG(log_errors, "Selection target player %d out of bounds", target_player);
    return 0;
  }

  ecs_entity_t zone = 0;
  switch (zone_kind) {
    case 0: // Garden
      if (slot_index < 0 || slot_index >= GARDEN_SIZE) {
        VALIDATION_LOG(log_errors, "Garden target index %d out of bounds", slot_index);
        return 0;
      }
      zone = gs->zones[target_player].garden;
      return find_card_in_zone_index(world, zone, slot_index);
    case 1: // Alley
      if (slot_index < 0 || slot_index >= ALLEY_SIZE) {
        VALIDATION_LOG(log_errors, "Alley target index %d out of bounds", slot_index);
        return 0;
      }
      zone = gs->zones[target_player].alley;
      return find_card_in_zone_index(world, zone, slot_index);
    case 2: // Leader
      zone = gs->zones[target_player].leader;
      return find_leader_card_in_zone(world, zone);
    default:
      VALIDATION_LOG(log_errors, "Unknown target zone kind %d", zone_kind);
      return 0;
  }
}

bool azk_validate_select_ability_target_action(
  ecs_world_t *world,
  const GameState *gs,
  const AbilityContext *actx,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  AbilitySelectIntent *out_intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "Action is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  if (actx == NULL || !actx->has_pending) {
    VALIDATION_LOG(log_errors, "No pending ability selection");
    return false;
  }

  if (actx->player != player) {
    VALIDATION_LOG(log_errors, "Ability selection not owned by active player");
    return false;
  }

  bool is_cost = action->type == ACT_SELECT_COST_TARGET;
  bool is_effect = action->type == ACT_SELECT_EFFECT_TARGET;
  if (!is_cost && !is_effect) {
    VALIDATION_LOG(log_errors, "Invalid selection action type %d", action->type);
    return false;
  }

  if (is_cost && actx->phase != ABILITY_SELECTION_COST) {
    VALIDATION_LOG(log_errors, "Not in cost selection phase");
    return false;
  }
  if (is_effect && actx->phase != ABILITY_SELECTION_EFFECT) {
    VALIDATION_LOG(log_errors, "Not in effect selection phase");
    return false;
  }

  if (actx->awaiting_consent) {
    VALIDATION_LOG(log_errors, "Ability requires consent before selecting targets");
    return false;
  }

  if (is_cost && actx->cost_filled >= actx->cost_expected) {
    VALIDATION_LOG(log_errors, "Cost selections already satisfied");
    return false;
  }
  if (is_effect && actx->effect_filled >= actx->effect_expected) {
    VALIDATION_LOG(log_errors, "Effect selections already satisfied");
    return false;
  }

  uint8_t card_def_id = (uint8_t)(actx->ability_uid >> 8);
  uint8_t ability_index = (uint8_t)(actx->ability_uid & 0xFF);
  const AbilityDef *def = azk_find_ability_def(card_def_id, ability_index);
  if (def == NULL) {
    VALIDATION_LOG(log_errors, "Ability definition missing for selection");
    return false;
  }

  int target_player = action->subaction_1;
  int target_zone_kind = action->subaction_2;
  int target_slot = action->subaction_3;

  ecs_entity_t target = resolve_selection_target(world, gs, target_player, target_zone_kind, target_slot, log_errors);
  if (target == 0) {
    VALIDATION_LOG(log_errors, "No target found for selection");
    return false;
  }

  AbilityTargetRequirement req = is_cost ? def->cost_targets : def->effect_targets;
  if (!azk_validate_target_against_req(world, gs, player, actx->source_card, target, req, ABILITY_TARGET_NONE)) {
    VALIDATION_LOG(log_errors, "Selected target does not meet requirement");
    return false;
  }

  if (out_intent) {
    AbilitySelectIntent intent = {
      .player = player,
      .phase = actx->phase,
      .target = target
    };
    *out_intent = intent;
  }

  return true;
}

bool azk_validate_accept_triggered_ability_action(
  ecs_world_t *world,
  const GameState *gs,
  const AbilityContext *actx,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "Action is null");

  if (!ensure_active_player(world, gs, player, log_errors)) {
    return false;
  }

  if (ability_selection_blocks_actions(world, player, action->type, log_errors)) {
    return false;
  }

  if (actx == NULL || !actx->has_pending) {
    VALIDATION_LOG(log_errors, "No pending ability to accept");
    return false;
  }

  if (actx->player != player) {
    VALIDATION_LOG(log_errors, "Ability prompt belongs to another player");
    return false;
  }

  if (actx->phase != ABILITY_SELECTION_PROMPT || !actx->awaiting_consent || !actx->optional) {
    VALIDATION_LOG(log_errors, "No optional ability awaiting consent");
    return false;
  }

  return true;
}
