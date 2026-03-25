#include "utils/card_utils.h"

#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/entity_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"
#include <stdio.h>

static bool is_card_in_play_zone(ecs_world_t *world, ecs_entity_t card) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    if (parent == gs->zones[p].garden || parent == gs->zones[p].alley ||
        parent == gs->zones[p].leader) {
      return true;
    }
  }

  return false;
}

bool azk_card_has_godmode_in_play(ecs_world_t *world, ecs_entity_t card) {
  return ecs_has(world, card, Godmode) && is_card_in_play_zone(world, card);
}

bool azk_card_enters_garden_tapped(ecs_world_t *world, ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  return card_id != NULL && card_id->id == CARD_DEF_AZK01_046;
}

bool azk_card_cannot_be_untapped(ecs_world_t *world, ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  return card_id != NULL && card_id->id == CARD_DEF_AZK01_046;
}

bool azk_card_can_only_attack_leaders(ecs_world_t *world, ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  return card_id != NULL && card_id->id == CARD_DEF_AZK01_077;
}

bool azk_card_can_attack_opponent_alley(ecs_world_t *world, ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id != NULL &&
      (card_id->id == CARD_DEF_AZK01_037 || card_id->id == CARD_DEF_AZK01_038)) {
    return true;
  }

  if (!ecs_has(world, card, TLeader)) {
    return false;
  }

  ecs_iter_t child_it = ecs_children(world, card);
  while (ecs_children_next(&child_it)) {
    for (int i = 0; i < child_it.count; ++i) {
      const CardId *weapon_id = ecs_get(world, child_it.entities[i], CardId);
      if (weapon_id != NULL &&
          (weapon_id->id == CARD_DEF_AZK01_043 ||
           weapon_id->id == CARD_DEF_AZK01_095)) {
        return true;
      }
    }
  }

  return false;
}

bool azk_card_counts_as_ikz_source(ecs_world_t *world, ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id == NULL || card_id->id != CARD_DEF_STT03_007) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (owner == 0) {
    return false;
  }

  uint8_t player_num = get_player_number(world, owner);
  return ecs_get_target(world, card, EcsChildOf, 0) == gs->zones[player_num].garden;
}

static uint8_t count_defender_entities_in_garden(ecs_world_t *world,
                                                 ecs_entity_t player) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || player == 0) {
    return 0;
  }

  uint8_t player_num = get_player_number(world, player);
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[player_num].garden);
  uint8_t count = 0;
  for (int32_t i = 0; i < cards.count; ++i) {
    if (cards.ids[i] != 0 && ecs_has(world, cards.ids[i], Defender)) {
      ++count;
    }
  }

  return count;
}

int8_t azk_get_effective_card_play_cost(ecs_world_t *world, ecs_entity_t player,
                                        ecs_entity_t card) {
  const IKZCost *cost = ecs_get(world, card, IKZCost);
  if (cost == NULL) {
    return 0;
  }

  int16_t effective_cost = cost->ikz_cost;
  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id != NULL && card_id->id == CARD_DEF_AZK01_106) {
    effective_cost -= count_defender_entities_in_garden(world, player);
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs != NULL && player != 0) {
    uint8_t player_num = get_player_number(world, player);
    effective_cost -= gs->next_card_play_cost_reduction[player_num];
  }

  if (effective_cost < 0) {
    effective_cost = 0;
  }

  return (int8_t)effective_cost;
}

typedef enum {
  AZK_DISCARD_REASON_DESTROY = 0,
  AZK_DISCARD_REASON_SACRIFICE = 1,
  AZK_DISCARD_REASON_REPLACEMENT = 2,
} AzkDiscardReason;

static void maybe_queue_self_leave_play_trigger(ecs_world_t *world,
                                                ecs_entity_t card,
                                                ecs_entity_t owner,
                                                AzkDiscardReason reason) {
  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id == NULL || !azk_has_ability(card_id->id)) {
    return;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (def == NULL) {
    return;
  }

  if (reason == AZK_DISCARD_REASON_DESTROY &&
      azk_has_ability_with_timing(card_id->id, ecs_id(AWhenDestroyed))) {
    azk_queue_triggered_effect(world, card, owner, TIMING_TAG_WHEN_DESTROYED);
  } else if (reason == AZK_DISCARD_REASON_SACRIFICE &&
             azk_has_ability_with_timing(card_id->id, ecs_id(AWhenSacrificed))) {
    azk_queue_triggered_effect(world, card, owner, TIMING_TAG_WHEN_SACRIFICED);
  }
}

static void heal_leader_in_zone(ecs_world_t *world, ecs_entity_t leader_zone,
                                int8_t max_heal) {
  ecs_entity_t leader = find_leader_card_in_zone(world, leader_zone);
  if (leader == 0) {
    return;
  }

  const BaseStats *base = ecs_get(world, leader, BaseStats);
  CurStats *cur = ecs_get_mut(world, leader, CurStats);
  if (base == NULL || cur == NULL) {
    return;
  }

  const int16_t missing_hp = (int16_t)base->health - (int16_t)cur->cur_hp;
  const int8_t heal_amount =
      (int8_t)(missing_hp <= 0
                   ? 0
                   : (missing_hp < max_heal ? missing_hp : max_heal));
  if (heal_amount <= 0) {
    return;
  }

  cur->cur_hp += heal_amount;
  ecs_modified(world, leader, CurStats);
  azk_log_card_stat_change(world, leader, 0, heal_amount, cur->cur_atk,
                           cur->cur_hp);
}

static void maybe_trigger_bobu_state(ecs_world_t *world, const GameState *gs,
                                     ecs_entity_t destroyed_card,
                                     ecs_entity_t owner,
                                     ecs_entity_t from_zone_entity,
                                     AzkDiscardReason reason) {
  const Type *type = ecs_get(world, destroyed_card, Type);
  const Element *element = ecs_get(world, destroyed_card, Element);
  if (gs == NULL || owner == 0 || reason == AZK_DISCARD_REASON_REPLACEMENT ||
      type == NULL || type->value != CARD_TYPE_ENTITY || element == NULL ||
      element->element != CARD_ELEMENT_EARTH) {
    return;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  if (from_zone_entity != gs->zones[owner_num].garden &&
      from_zone_entity != gs->zones[owner_num].alley) {
    return;
  }

  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[owner_num].leader);
  const CardId *leader_id = leader != 0 ? ecs_get(world, leader, CardId) : NULL;
  if (leader_id == NULL || leader_id->id != CARD_DEF_STT03_001) {
    return;
  }

  STT03BobuState *state = ecs_get_mut(world, leader, STT03BobuState);
  if (state == NULL || state->expires_turn == 0 ||
      gs->turn_number >= state->expires_turn) {
    return;
  }

  state->expires_turn = 0;
  ecs_modified(world, leader, STT03BobuState);
  heal_leader_in_zone(world, gs->zones[owner_num].leader, 1);
}

static uint8_t current_turn_owner_index(const GameState *gs) {
  if (gs == NULL) {
    return 0;
  }

  if (gs->phase == PHASE_RESPONSE_WINDOW) {
    return (uint8_t)((gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH);
  }

  return (uint8_t)gs->active_player_index;
}

static void maybe_trigger_miharu_state(ecs_world_t *world, const GameState *gs,
                                       ecs_entity_t owner,
                                       ecs_entity_t from_zone_entity,
                                       AzkDiscardReason reason) {
  if (gs == NULL || owner == 0 || reason != AZK_DISCARD_REASON_DESTROY) {
    return;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  if (current_turn_owner_index(gs) == owner_num ||
      from_zone_entity != gs->zones[owner_num].garden) {
    return;
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    ecs_entity_t candidate = garden_cards.ids[i];
    const CardId *card_id = ecs_get(world, candidate, CardId);
    if (card_id == NULL || card_id->id != CARD_DEF_STT03_012) {
      continue;
    }

    STT03MiharuState *state = ecs_get_mut(world, candidate, STT03MiharuState);
    if (state != NULL && state->last_heal_turn == gs->turn_number) {
      continue;
    }

    if (state == NULL) {
      ecs_set(world, candidate, STT03MiharuState,
              {.last_heal_turn = gs->turn_number});
    } else {
      state->last_heal_turn = gs->turn_number;
      ecs_modified(world, candidate, STT03MiharuState);
    }

    heal_leader_in_zone(world, gs->zones[owner_num].leader, 2);
  }
}

static void maybe_trigger_kurai_state(ecs_world_t *world, const GameState *gs,
                                      ecs_entity_t owner,
                                      ecs_entity_t from_zone_entity,
                                      AzkDiscardReason reason) {
  if (gs == NULL || owner == 0 || reason != AZK_DISCARD_REASON_DESTROY) {
    return;
  }

  const uint8_t destroyed_owner_num = get_player_number(world, owner);
  if (from_zone_entity != gs->zones[destroyed_owner_num].garden) {
    return;
  }

  const uint8_t kurai_owner_num =
      (destroyed_owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[kurai_owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    ecs_entity_t candidate = garden_cards.ids[i];
    const CardId *card_id = ecs_get(world, candidate, CardId);
    if (card_id == NULL || card_id->id != CARD_DEF_STT04_013) {
      continue;
    }

    STT04KuraiState *state = ecs_get_mut(world, candidate, STT04KuraiState);
    if (state != NULL && state->last_untap_turn == gs->turn_number) {
      continue;
    }

    if (state == NULL) {
      ecs_set(world, candidate, STT04KuraiState,
              {.last_untap_turn = gs->turn_number});
    } else {
      state->last_untap_turn = gs->turn_number;
      ecs_modified(world, candidate, STT04KuraiState);
    }

    const TapState *tap = ecs_get(world, candidate, TapState);
    if (tap != NULL && tap->tapped) {
      ecs_set(world, candidate, TapState,
              {.tapped = false, .cooldown = tap->cooldown});
      azk_log_card_tap_state_changed(
          world, candidate,
          tap->cooldown ? GLOG_TAP_COOLDOWN : GLOG_TAP_UNTAPPED);
    }
  }
}

static void maybe_trigger_special_destroy_observers(ecs_world_t *world,
                                                    ecs_entity_t card,
                                                    ecs_entity_t owner,
                                                    ecs_entity_t from_zone_entity,
                                                    AzkDiscardReason reason) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || owner == 0 || reason == AZK_DISCARD_REASON_REPLACEMENT) {
    return;
  }

  maybe_trigger_bobu_state(world, gs, card, owner, from_zone_entity, reason);
  maybe_trigger_miharu_state(world, gs, owner, from_zone_entity, reason);
  maybe_trigger_kurai_state(world, gs, owner, from_zone_entity, reason);
}

static void discard_card_internal(ecs_world_t *world, ecs_entity_t card,
                                  AzkDiscardReason reason,
                                  bool ignore_godmode_protection) {
  if (!ignore_godmode_protection && azk_card_has_godmode_in_play(world, card)) {
    cli_render_logf("[CardUtils] Godmode prevented discard");
    return;
  }

  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER,
             "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t discard_zone = gs->zones[player_number->player_number].discard;

  ecs_entity_t from_zone_entity = ecs_get_target(world, card, EcsChildOf, 0);
  GameLogZone from_zone = azk_zone_entity_to_log_zone(world, from_zone_entity);
  int8_t from_index =
      azk_get_card_index_in_zone(world, card, from_zone_entity);

  if (from_zone == GLOG_ZONE_NONE && ecs_has_id(world, card, TWeapon)) {
    from_zone = GLOG_ZONE_EQUIPPED;
  }

  bool from_play = from_zone == GLOG_ZONE_GARDEN || from_zone == GLOG_ZONE_ALLEY ||
                   from_zone == GLOG_ZONE_LEADER || from_zone == GLOG_ZONE_EQUIPPED;
  bool from_hand = from_zone == GLOG_ZONE_HAND;
  if (from_play && reason != AZK_DISCARD_REASON_REPLACEMENT) {
    maybe_queue_self_leave_play_trigger(world, card, owner, reason);
    maybe_trigger_special_destroy_observers(world, card, owner, from_zone_entity,
                                           reason);
  }

  ecs_remove_id(world, card, ecs_id(ZoneIndex));
  ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
  const BaseStats *base = ecs_get(world, card, BaseStats);
  if (base && ecs_has(world, card, CurStats)) {
    ecs_set(world, card, CurStats,
            {.cur_atk = base->attack, .cur_hp = base->health});
  }
  clear_card_temporary_state(world, card);
  ecs_add_pair(world, card, EcsChildOf, discard_zone);

  azk_log_card_zone_moved(world, card, from_zone, from_index, GLOG_ZONE_DISCARD,
                          -1);

  if (from_hand) {
    GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
    if (gs_mut != NULL) {
      uint8_t player_num = get_player_number(world, owner);
      gs_mut->discarded_cards_this_turn[player_num]++;
      ecs_singleton_modified(world, GameState);
    }
  }
}

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type) {
  const Type *card_type = ecs_get(world, card, Type);
  ecs_assert(card_type != NULL, ECS_INVALID_PARAMETER,
             "Type component not found for card %d", card);
  return card_type->value == type;
}

void discard_card(ecs_world_t *world, ecs_entity_t card) {
  discard_card_internal(world, card, AZK_DISCARD_REASON_DESTROY, false);
}

void sacrifice_card(ecs_world_t *world, ecs_entity_t card) {
  discard_card_internal(world, card, AZK_DISCARD_REASON_SACRIFICE, false);
}

void discard_card_for_replacement(ecs_world_t *world, ecs_entity_t card) {
  discard_card_internal(world, card, AZK_DISCARD_REASON_REPLACEMENT, true);
}

void return_card_to_hand(ecs_world_t *world, ecs_entity_t card) {
  if (azk_card_has_godmode_in_play(world, card)) {
    cli_render_logf("[CardUtils] Godmode prevented return to hand");
    return;
  }

  // Check source zone BEFORE changing ChildOf
  ecs_entity_t source_parent = ecs_get_target(world, card, EcsChildOf, 0);
  GameLogZone from_zone = azk_zone_entity_to_log_zone(world, source_parent);
  int8_t from_index = azk_get_card_index_in_zone(world, card, source_parent);

  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER,
             "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = player_number->player_number;
  ecs_entity_t hand_zone = gs->zones[player_num].hand;

  // Determine if card is returning from play (garden/alley)
  bool from_play = false;
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
    if (source_parent == gs->zones[p].garden ||
        source_parent == gs->zones[p].alley) {
      from_play = true;
      break;
    }
  }

  // Discard any equipped weapons before returning to hand
  discard_equipped_weapon_cards(world, card);
  // Remove zone index if present (entity was in garden/alley)
  ecs_remove_id(world, card, ecs_id(ZoneIndex));
  // Reset tap state
  ecs_set(world, card, TapState, {.tapped = false, .cooldown = false});
  // Reset current stats to base stats
  const BaseStats *base = ecs_get(world, card, BaseStats);
  if (base) {
    ecs_set(world, card, CurStats,
            {.cur_atk = base->attack, .cur_hp = base->health});
  }
  clear_card_temporary_state(world, card);

  // Include already-logged hand moves in this action batch so repeated
  // returns under deferred ops still get monotonic append indices.
  int32_t hand_index =
      azk_get_effective_hand_count(world, hand_zone, player_num);

  // Move to hand
  ecs_add_pair(world, card, EcsChildOf, hand_zone);

  // Log zone movement to hand
  azk_log_card_zone_moved(world, card, from_zone, from_index, GLOG_ZONE_HAND,
                          (int8_t)hand_index);

  cli_render_logf("[CardUtils] Returned card to hand");

  // Trigger observers only if card came from play (garden/alley)
  if (from_play) {
    const CardId *card_id = ecs_get(world, card, CardId);
    if (card_id != NULL && azk_has_ability(card_id->id)) {
      const AbilityDef *def = azk_get_ability_def(card_id->id);
      if (def != NULL && def->timing_tag == ecs_id(AWhenReturnedToHand)) {
        azk_queue_triggered_effect(world, card, owner,
                                   TIMING_TAG_WHEN_RETURNED_TO_HAND);
      }
    }

    GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
    gs_mut->entities_returned_to_hand_this_turn[player_num]++;
    ecs_singleton_modified(world, GameState);
    azk_trigger_return_to_hand_observers(world, card);
  }
}

bool can_tap_card(ecs_world_t *world, ecs_entity_t card, bool ignore_cooldown) {
  const TapState *ts = ecs_get(world, card, TapState);
  ecs_assert(ts != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  if (ts->tapped)
    return false;
  if (ts->cooldown && !ignore_cooldown)
    return false;
  return true;
}

void tap_card(ecs_world_t *world, ecs_entity_t card) {
  const TapState *ts = ecs_get(world, card, TapState);
  ecs_assert(ts != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  ecs_set(world, card, TapState, {.tapped = true, .cooldown = ts->cooldown});
  azk_log_card_tap_state_changed(world, card, GLOG_TAP_TAPPED);
}

void set_card_to_cooldown(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  ecs_set(world, card, TapState,
          {.tapped = tap_state->tapped, .cooldown = true});
  azk_log_card_tap_state_changed(world, card, GLOG_TAP_COOLDOWN);
}

bool is_card_tapped(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  return tap_state->tapped;
}

bool is_card_cooldown(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER,
             "TapState component not found for card %d", card);
  return tap_state->cooldown;
}

bool is_weapon_card(ecs_world_t *world, ecs_entity_t card) {
  const Type *card_type = ecs_get(world, card, Type);
  if (!card_type) {
    return false;
  }
  return card_type->value == CARD_TYPE_WEAPON;
}

int count_weapons_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t children = ecs_get_ordered_children(world, zone);
  int count = 0;
  for (int32_t i = 0; i < children.count; i++) {
    if (is_weapon_card(world, children.ids[i])) {
      count++;
    }
  }
  return count;
}

bool has_subtype(ecs_world_t *world, ecs_entity_t card, ecs_id_t subtype_tag) {
  if (card == 0 || subtype_tag == 0) {
    return false;
  }

  if (ecs_has_id(world, card, subtype_tag)) {
    return true;
  }

  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id == NULL || card_id->id != CARD_DEF_AZK01_081) {
    return false;
  }

  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (owner == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return parent == gs->zones[player_num].deck ||
         parent == gs->zones[player_num].selection;
}

bool is_watercrafting_card(ecs_world_t *world, ecs_entity_t card) {
  return has_subtype(world, card, ecs_id(TSubtype_Watercrafting));
}

bool is_water_element_card(ecs_world_t *world, ecs_entity_t card) {
  const Element *elem = ecs_get(world, card, Element);
  if (!elem) {
    return false;
  }
  return elem->element == (uint8_t)CARD_ELEMENT_WATER;
}

bool is_normal_element_card(ecs_world_t *world, ecs_entity_t card) {
  const Element *elem = ecs_get(world, card, Element);
  if (!elem) {
    return false;
  }
  return elem->element == (uint8_t)CARD_ELEMENT_NORMAL;
}

CardElement get_card_element(ecs_world_t *world, ecs_entity_t card) {
  const Element *elem = ecs_get(world, card, Element);
  if (!elem) {
    return CARD_ELEMENT_NORMAL;
  }
  return (CardElement)elem->element;
}

int count_subtype_in_zone(ecs_world_t *world, ecs_entity_t zone,
                          ecs_id_t subtype_tag) {
  ecs_entities_t children = ecs_get_ordered_children(world, zone);
  int count = 0;
  for (int32_t i = 0; i < children.count; i++) {
    if (has_subtype(world, children.ids[i], subtype_tag)) {
      count++;
    }
  }
  return count;
}

bool has_equipped_weapon(ecs_world_t *world, ecs_entity_t entity) {
  ecs_iter_t it = ecs_children(world, entity);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      if (ecs_has_id(world, it.entities[i], TWeapon)) {
        return true;
      }
    }
  }
  return false;
}
