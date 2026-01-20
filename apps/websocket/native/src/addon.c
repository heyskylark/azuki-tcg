/**
 * Node.js native addon for Azuki game engine.
 * Pure C implementation using N-API.
 */

#include <node_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

#include "azuki/engine.h"
#include "components/game_log.h"
#include "constants/game.h"
#include "utils/observation_util.h"
#include "utils/game_log_util.h"
#include "utils/debug_log.h"

// Maximum number of concurrent game worlds
#define MAX_WORLDS 256

// World storage
typedef struct {
  char id[32];
  AzkEngine *engine;
  bool in_use;
} WorldSlot;

static WorldSlot world_slots[MAX_WORLDS];
static uint64_t world_counter = 0;

// Helper to throw JS error
static napi_value throw_error(napi_env env, const char *message) {
  napi_throw_error(env, NULL, message);
  return NULL;
}

// Generate unique world ID
static void generate_world_id(char *out_id) {
  snprintf(out_id, 32, "world_%llu", (unsigned long long)++world_counter);
}

// Find world by ID
static AzkEngine *find_world(const char *id) {
  for (int i = 0; i < MAX_WORLDS; i++) {
    if (world_slots[i].in_use && strcmp(world_slots[i].id, id) == 0) {
      return world_slots[i].engine;
    }
  }
  return NULL;
}

// Store world
static bool store_world(const char *id, AzkEngine *engine) {
  for (int i = 0; i < MAX_WORLDS; i++) {
    if (!world_slots[i].in_use) {
      strncpy(world_slots[i].id, id, 31);
      world_slots[i].id[31] = '\0';
      world_slots[i].engine = engine;
      world_slots[i].in_use = true;
      return true;
    }
  }
  return false;
}

// Remove world
static bool remove_world(const char *id) {
  for (int i = 0; i < MAX_WORLDS; i++) {
    if (world_slots[i].in_use && strcmp(world_slots[i].id, id) == 0) {
      world_slots[i].in_use = false;
      world_slots[i].engine = NULL;
      return true;
    }
  }
  return false;
}

// createWorld(seed: number) -> { worldId: string, success: boolean }
static napi_value CreateWorld(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  if (argc < 1) {
    return throw_error(env, "createWorld requires seed argument");
  }

  uint32_t seed;
  napi_get_value_uint32(env, args[0], &seed);

  AzkEngine *engine = azk_engine_create(seed);
  if (!engine) {
    napi_value result, success_val, error_val;
    napi_create_object(env, &result);
    napi_get_boolean(env, false, &success_val);
    napi_set_named_property(env, result, "success", success_val);

    // Include error message if available
    const char *last_error = azk_engine_get_last_error();
    if (last_error) {
      napi_create_string_utf8(env, last_error, NAPI_AUTO_LENGTH, &error_val);
      napi_set_named_property(env, result, "error", error_val);
      azk_engine_clear_last_error();
    }
    return result;
  }

  char world_id[32];
  generate_world_id(world_id);

  if (!store_world(world_id, engine)) {
    azk_engine_destroy(engine);
    return throw_error(env, "Too many active worlds");
  }

  napi_value result, world_id_val, success_val;
  napi_create_object(env, &result);
  napi_create_string_utf8(env, world_id, NAPI_AUTO_LENGTH, &world_id_val);
  napi_get_boolean(env, true, &success_val);
  napi_set_named_property(env, result, "worldId", world_id_val);
  napi_set_named_property(env, result, "success", success_val);

  return result;
}

// createWorldWithDecks(seed: number, player0Deck: DeckCardEntry[], player1Deck: DeckCardEntry[])
static napi_value CreateWorldWithDecks(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  if (argc < 3) {
    return throw_error(env, "createWorldWithDecks requires seed, player0Deck, player1Deck");
  }

  uint32_t seed;
  napi_get_value_uint32(env, args[0], &seed);

  // Parse player0 deck
  uint32_t p0_len, p1_len;
  napi_get_array_length(env, args[1], &p0_len);
  napi_get_array_length(env, args[2], &p1_len);

  CardInfo *p0_deck = calloc(p0_len, sizeof(CardInfo));
  CardInfo *p1_deck = calloc(p1_len, sizeof(CardInfo));

  if (!p0_deck || !p1_deck) {
    free(p0_deck);
    free(p1_deck);
    return throw_error(env, "Failed to allocate deck memory");
  }

  // Parse player0 deck entries
  for (uint32_t i = 0; i < p0_len; i++) {
    napi_value entry, card_id_val, count_val;
    napi_get_element(env, args[1], i, &entry);
    napi_get_named_property(env, entry, "cardId", &card_id_val);
    napi_get_named_property(env, entry, "count", &count_val);

    int32_t card_id, count;
    napi_get_value_int32(env, card_id_val, &card_id);
    napi_get_value_int32(env, count_val, &count);

    p0_deck[i].card_id = (CardDefId)card_id;
    p0_deck[i].card_count = count;
  }

  // Parse player1 deck entries
  for (uint32_t i = 0; i < p1_len; i++) {
    napi_value entry, card_id_val, count_val;
    napi_get_element(env, args[2], i, &entry);
    napi_get_named_property(env, entry, "cardId", &card_id_val);
    napi_get_named_property(env, entry, "count", &count_val);

    int32_t card_id, count;
    napi_get_value_int32(env, card_id_val, &card_id);
    napi_get_value_int32(env, count_val, &count);

    p1_deck[i].card_id = (CardDefId)card_id;
    p1_deck[i].card_count = count;
  }

  AzkEngine *engine = azk_engine_create_with_decks(seed, p0_deck, p0_len, p1_deck, p1_len);

  free(p0_deck);
  free(p1_deck);

  if (!engine) {
    napi_value result, success_val, error_val;
    napi_create_object(env, &result);
    napi_get_boolean(env, false, &success_val);
    napi_set_named_property(env, result, "success", success_val);

    // Include error message if available
    const char *last_error = azk_engine_get_last_error();
    if (last_error) {
      napi_create_string_utf8(env, last_error, NAPI_AUTO_LENGTH, &error_val);
      napi_set_named_property(env, result, "error", error_val);
      azk_engine_clear_last_error();
    }
    return result;
  }

  char world_id[32];
  generate_world_id(world_id);

  if (!store_world(world_id, engine)) {
    azk_engine_destroy(engine);
    return throw_error(env, "Too many active worlds");
  }

  napi_value result, world_id_val, success_val;
  napi_create_object(env, &result);
  napi_create_string_utf8(env, world_id, NAPI_AUTO_LENGTH, &world_id_val);
  napi_get_boolean(env, true, &success_val);
  napi_set_named_property(env, result, "worldId", world_id_val);
  napi_set_named_property(env, result, "success", success_val);

  return result;
}

// destroyWorld(worldId: string)
static napi_value DestroyWorld(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char world_id[32];
  size_t len;
  napi_get_value_string_utf8(env, args[0], world_id, sizeof(world_id), &len);

  AzkEngine *engine = find_world(world_id);
  if (engine) {
    azk_engine_destroy(engine);
    remove_world(world_id);
  }

  return NULL;
}

// requiresAction(worldId: string) -> boolean
static napi_value RequiresAction(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char world_id[32];
  size_t len;
  napi_get_value_string_utf8(env, args[0], world_id, sizeof(world_id), &len);

  AzkEngine *engine = find_world(world_id);
  if (!engine) {
    napi_value result;
    napi_get_boolean(env, false, &result);
    return result;
  }

  bool requires = azk_engine_requires_action(engine);
  napi_value result;
  napi_get_boolean(env, requires, &result);
  return result;
}

// isGameOver(worldId: string) -> boolean
static napi_value IsGameOver(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char world_id[32];
  size_t len;
  napi_get_value_string_utf8(env, args[0], world_id, sizeof(world_id), &len);

  AzkEngine *engine = find_world(world_id);
  if (!engine) {
    napi_value result;
    napi_get_boolean(env, true, &result);
    return result;
  }

  bool game_over = azk_engine_is_game_over(engine);
  napi_value result;
  napi_get_boolean(env, game_over, &result);
  return result;
}

// getActivePlayer(worldId: string) -> number
static napi_value GetActivePlayer(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char world_id[32];
  size_t len;
  napi_get_value_string_utf8(env, args[0], world_id, sizeof(world_id), &len);

  AzkEngine *engine = find_world(world_id);
  if (!engine) {
    napi_value result;
    napi_create_int32(env, -1, &result);
    return result;
  }

  const GameState *state = azk_engine_game_state(engine);
  napi_value result;
  napi_create_int32(env, state->active_player_index, &result);
  return result;
}

// ============================================================================
// Game log serialization helpers
// ============================================================================

// Serialize GameLogCardRef to JS object
static napi_value serialize_card_ref(napi_env env, const GameLogCardRef *ref) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value player_val, card_def_id_val, zone_val, zone_index_val;
  napi_create_int32(env, ref->player, &player_val);
  napi_create_int32(env, ref->card_def_id, &card_def_id_val);
  napi_create_string_utf8(env, azk_log_zone_to_string(ref->zone), NAPI_AUTO_LENGTH, &zone_val);
  napi_create_int32(env, ref->zone_index, &zone_index_val);

  napi_set_named_property(env, obj, "player", player_val);
  napi_set_named_property(env, obj, "cardDefId", card_def_id_val);
  napi_set_named_property(env, obj, "zone", zone_val);
  napi_set_named_property(env, obj, "zoneIndex", zone_index_val);

  return obj;
}

// Serialize GameLogCardMetadata to JS object
static napi_value serialize_card_metadata(napi_env env, const GameLogCardMetadata *meta) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value cur_atk, cur_hp, tapped, cooldown;
  napi_value has_charge, has_defender, has_infiltrate;
  napi_value is_frozen, is_effect_immune;

  napi_create_int32(env, meta->cur_atk, &cur_atk);
  napi_create_int32(env, meta->cur_hp, &cur_hp);
  napi_get_boolean(env, meta->tapped, &tapped);
  napi_get_boolean(env, meta->cooldown, &cooldown);
  napi_get_boolean(env, meta->has_charge, &has_charge);
  napi_get_boolean(env, meta->has_defender, &has_defender);
  napi_get_boolean(env, meta->has_infiltrate, &has_infiltrate);
  napi_get_boolean(env, meta->is_frozen, &is_frozen);
  napi_get_boolean(env, meta->is_effect_immune, &is_effect_immune);

  napi_set_named_property(env, obj, "curAtk", cur_atk);
  napi_set_named_property(env, obj, "curHp", cur_hp);
  napi_set_named_property(env, obj, "tapped", tapped);
  napi_set_named_property(env, obj, "cooldown", cooldown);
  napi_set_named_property(env, obj, "hasCharge", has_charge);
  napi_set_named_property(env, obj, "hasDefender", has_defender);
  napi_set_named_property(env, obj, "hasInfiltrate", has_infiltrate);
  napi_set_named_property(env, obj, "isFrozen", is_frozen);
  napi_set_named_property(env, obj, "isEffectImmune", is_effect_immune);

  return obj;
}

// Serialize a single GameStateLog to JS object
static napi_value serialize_game_log(napi_env env, const GameStateLog *log) {
  napi_value obj, type_val, data_val;
  napi_create_object(env, &obj);

  // Get type string
  const char *type_str = azk_log_type_to_string(log->type);
  napi_create_string_utf8(env, type_str, NAPI_AUTO_LENGTH, &type_val);
  napi_set_named_property(env, obj, "type", type_val);

  // Serialize data based on log type
  napi_create_object(env, &data_val);

  switch (log->type) {
    case GLOG_CARD_ZONE_MOVED: {
      const GameLogZoneMoved *d = &log->data.zone_moved;
      napi_value card, from_zone, from_idx, to_zone, to_idx, metadata;

      card = serialize_card_ref(env, &d->card);
      napi_create_string_utf8(env, azk_log_zone_to_string(d->from_zone), NAPI_AUTO_LENGTH, &from_zone);
      napi_create_int32(env, d->from_index, &from_idx);
      napi_create_string_utf8(env, azk_log_zone_to_string(d->to_zone), NAPI_AUTO_LENGTH, &to_zone);
      napi_create_int32(env, d->to_index, &to_idx);
      metadata = serialize_card_metadata(env, &d->metadata);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "fromZone", from_zone);
      napi_set_named_property(env, data_val, "fromIndex", from_idx);
      napi_set_named_property(env, data_val, "toZone", to_zone);
      napi_set_named_property(env, data_val, "toIndex", to_idx);
      napi_set_named_property(env, data_val, "metadata", metadata);
      break;
    }

    case GLOG_CARD_TAP_STATE_CHANGED: {
      const GameLogTapStateChanged *d = &log->data.tap_changed;
      napi_value card, new_state;

      card = serialize_card_ref(env, &d->card);
      const char *state_str = "UNTAPPED";
      if (d->new_state == GLOG_TAP_TAPPED) state_str = "TAPPED";
      else if (d->new_state == GLOG_TAP_COOLDOWN) state_str = "COOLDOWN";
      napi_create_string_utf8(env, state_str, NAPI_AUTO_LENGTH, &new_state);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "newState", new_state);
      break;
    }

    case GLOG_CARD_STAT_CHANGE: {
      const GameLogStatChange *d = &log->data.stat_change;
      napi_value card, atk_delta, hp_delta, new_atk, new_hp;

      card = serialize_card_ref(env, &d->card);
      napi_create_int32(env, d->atk_delta, &atk_delta);
      napi_create_int32(env, d->hp_delta, &hp_delta);
      napi_create_int32(env, d->new_atk, &new_atk);
      napi_create_int32(env, d->new_hp, &new_hp);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "atkDelta", atk_delta);
      napi_set_named_property(env, data_val, "hpDelta", hp_delta);
      napi_set_named_property(env, data_val, "newAtk", new_atk);
      napi_set_named_property(env, data_val, "newHp", new_hp);
      break;
    }

    case GLOG_TURN_STARTED: {
      const GameLogTurnStarted *d = &log->data.turn_started;
      napi_value player, turn;

      napi_create_int32(env, d->player, &player);
      napi_create_int32(env, d->turn_number, &turn);

      napi_set_named_property(env, data_val, "player", player);
      napi_set_named_property(env, data_val, "turnNumber", turn);
      break;
    }

    case GLOG_TURN_ENDED: {
      const GameLogTurnEnded *d = &log->data.turn_ended;
      napi_value player, turn;

      napi_create_int32(env, d->player, &player);
      napi_create_int32(env, d->turn_number, &turn);

      napi_set_named_property(env, data_val, "player", player);
      napi_set_named_property(env, data_val, "turnNumber", turn);
      break;
    }

    case GLOG_DECK_SHUFFLED: {
      const GameLogDeckShuffled *d = &log->data.deck_shuffled;
      napi_value player, reason;

      napi_create_int32(env, d->player, &player);
      const char *reason_str = "GAME_START";
      if (d->reason == GLOG_SHUFFLE_MULLIGAN) reason_str = "MULLIGAN";
      else if (d->reason == GLOG_SHUFFLE_EFFECT) reason_str = "EFFECT";
      napi_create_string_utf8(env, reason_str, NAPI_AUTO_LENGTH, &reason);

      napi_set_named_property(env, data_val, "player", player);
      napi_set_named_property(env, data_val, "reason", reason);
      break;
    }

    case GLOG_GAME_ENDED: {
      const GameLogGameEnded *d = &log->data.game_ended;
      napi_value winner, reason;

      napi_create_int32(env, d->winner, &winner);
      const char *reason_str = "LEADER_DEFEATED";
      if (d->reason == GLOG_END_DECK_OUT) reason_str = "DECK_OUT";
      else if (d->reason == GLOG_END_CONCEDE) reason_str = "CONCEDE";
      napi_create_string_utf8(env, reason_str, NAPI_AUTO_LENGTH, &reason);

      napi_set_named_property(env, data_val, "winner", winner);
      napi_set_named_property(env, data_val, "reason", reason);
      break;
    }

    case GLOG_COMBAT_DECLARED: {
      const GameLogCombatDeclared *d = &log->data.combat_declared;
      napi_value attacker, target;

      attacker = serialize_card_ref(env, &d->attacker);
      target = serialize_card_ref(env, &d->target);

      napi_set_named_property(env, data_val, "attacker", attacker);
      napi_set_named_property(env, data_val, "target", target);
      break;
    }

    case GLOG_DEFENDER_DECLARED: {
      const GameLogDefenderDeclared *d = &log->data.defender_declared;
      napi_value defender = serialize_card_ref(env, &d->defender);
      napi_set_named_property(env, data_val, "defender", defender);
      break;
    }

    case GLOG_COMBAT_DAMAGE: {
      const GameLogCombatDamage *d = &log->data.combat_damage;
      napi_value attacker, defender, atk_dmg_dealt, atk_dmg_taken, def_dmg_dealt, def_dmg_taken;

      attacker = serialize_card_ref(env, &d->attacker);
      defender = serialize_card_ref(env, &d->defender);
      napi_create_int32(env, d->attacker_damage_dealt, &atk_dmg_dealt);
      napi_create_int32(env, d->attacker_damage_taken, &atk_dmg_taken);
      napi_create_int32(env, d->defender_damage_dealt, &def_dmg_dealt);
      napi_create_int32(env, d->defender_damage_taken, &def_dmg_taken);

      napi_set_named_property(env, data_val, "attacker", attacker);
      napi_set_named_property(env, data_val, "defender", defender);
      napi_set_named_property(env, data_val, "attackerDamageDealt", atk_dmg_dealt);
      napi_set_named_property(env, data_val, "attackerDamageTaken", atk_dmg_taken);
      napi_set_named_property(env, data_val, "defenderDamageDealt", def_dmg_dealt);
      napi_set_named_property(env, data_val, "defenderDamageTaken", def_dmg_taken);
      break;
    }

    case GLOG_ENTITY_DIED: {
      const GameLogEntityDied *d = &log->data.entity_died;
      napi_value card, cause;

      card = serialize_card_ref(env, &d->card);
      const char *cause_str = "COMBAT";
      if (d->cause == GLOG_DEATH_ABILITY) cause_str = "ABILITY";
      else if (d->cause == GLOG_DEATH_EFFECT) cause_str = "EFFECT";
      napi_create_string_utf8(env, cause_str, NAPI_AUTO_LENGTH, &cause);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "cause", cause);
      break;
    }

    case GLOG_EFFECT_QUEUED: {
      const GameLogEffectQueued *d = &log->data.effect_queued;
      napi_value card, ability_idx, trigger;

      card = serialize_card_ref(env, &d->card);
      napi_create_int32(env, d->ability_index, &ability_idx);
      napi_create_int32(env, d->trigger_tag, &trigger);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "abilityIndex", ability_idx);
      napi_set_named_property(env, data_val, "triggerTag", trigger);
      break;
    }

    case GLOG_CARD_EFFECT_ENABLED: {
      const GameLogEffectEnabled *d = &log->data.effect_enabled;
      napi_value card, ability_idx;

      card = serialize_card_ref(env, &d->card);
      napi_create_int32(env, d->ability_index, &ability_idx);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "abilityIndex", ability_idx);
      break;
    }

    case GLOG_STATUS_EFFECT_APPLIED: {
      const GameLogStatusApplied *d = &log->data.status_applied;
      napi_value card, effect, duration;

      card = serialize_card_ref(env, &d->card);
      const char *eff_str = "FROZEN";
      if (d->effect == GLOG_STATUS_SHOCKED) eff_str = "SHOCKED";
      else if (d->effect == GLOG_STATUS_EFFECT_IMMUNE) eff_str = "EFFECT_IMMUNE";
      napi_create_string_utf8(env, eff_str, NAPI_AUTO_LENGTH, &effect);
      napi_create_int32(env, d->duration, &duration);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "effect", effect);
      napi_set_named_property(env, data_val, "duration", duration);
      break;
    }

    case GLOG_STATUS_EFFECT_EXPIRED: {
      const GameLogStatusExpired *d = &log->data.status_expired;
      napi_value card, effect;

      card = serialize_card_ref(env, &d->card);
      const char *eff_str = "FROZEN";
      if (d->effect == GLOG_STATUS_SHOCKED) eff_str = "SHOCKED";
      else if (d->effect == GLOG_STATUS_EFFECT_IMMUNE) eff_str = "EFFECT_IMMUNE";
      napi_create_string_utf8(env, eff_str, NAPI_AUTO_LENGTH, &effect);

      napi_set_named_property(env, data_val, "card", card);
      napi_set_named_property(env, data_val, "effect", effect);
      break;
    }

    default:
      // Unknown log type - leave data empty
      break;
  }

  napi_set_named_property(env, obj, "data", data_val);
  return obj;
}

// submitAction(worldId: string, playerIndex: number, action: [number, number, number, number])
static napi_value SubmitAction(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char world_id[32];
  size_t len;
  napi_get_value_string_utf8(env, args[0], world_id, sizeof(world_id), &len);

  int32_t player_index;
  napi_get_value_int32(env, args[1], &player_index);

  // Parse action array
  int action_values[4];
  for (int i = 0; i < 4; i++) {
    napi_value elem;
    napi_get_element(env, args[2], i, &elem);
    napi_get_value_int32(env, elem, &action_values[i]);
  }

  AzkEngine *engine = find_world(world_id);
  if (!engine) {
    napi_value result, success_val, error_val;
    napi_create_object(env, &result);
    napi_get_boolean(env, false, &success_val);
    napi_create_string_utf8(env, "World not found", NAPI_AUTO_LENGTH, &error_val);
    napi_set_named_property(env, result, "success", success_val);
    napi_set_named_property(env, result, "error", error_val);
    return result;
  }

  // Parse and validate action
  UserAction action;
  bool valid = azk_engine_parse_action_values(engine, action_values, &action);

  if (!valid) {
    napi_value result, success_val, invalid_val, error_val;
    napi_create_object(env, &result);
    napi_get_boolean(env, false, &success_val);
    napi_get_boolean(env, true, &invalid_val);
    napi_create_string_utf8(env, "Invalid action", NAPI_AUTO_LENGTH, &error_val);
    napi_set_named_property(env, result, "success", success_val);
    napi_set_named_property(env, result, "invalid", invalid_val);
    napi_set_named_property(env, result, "error", error_val);
    return result;
  }

  // Submit action
  bool submitted = azk_engine_submit_action(engine, &action);
  if (!submitted) {
    napi_value result, success_val, invalid_val, error_val;
    napi_create_object(env, &result);
    napi_get_boolean(env, false, &success_val);
    napi_get_boolean(env, true, &invalid_val);
    napi_create_string_utf8(env, "Action rejected", NAPI_AUTO_LENGTH, &error_val);
    napi_set_named_property(env, result, "success", success_val);
    napi_set_named_property(env, result, "invalid", invalid_val);
    napi_set_named_property(env, result, "error", error_val);
    return result;
  }

  // Clear logs before ticking (so we only capture logs from this action)
  azk_clear_game_logs(engine);

  // Process the submitted action (always tick once)
  azk_engine_tick(engine);

  // Continue ticking through phases that don't require user input
  while (!azk_engine_requires_action(engine) && !azk_engine_is_game_over(engine)) {
    azk_engine_tick(engine);
  }

  // Build result
  const GameState *state = azk_engine_game_state(engine);
  bool game_over = azk_engine_is_game_over(engine);

  napi_value result, success_val, invalid_val, game_over_val, winner_val;
  napi_value state_context, logs_arr;

  napi_create_object(env, &result);
  napi_get_boolean(env, true, &success_val);
  napi_get_boolean(env, false, &invalid_val);
  napi_get_boolean(env, game_over, &game_over_val);

  if (game_over && state->winner >= 0) {
    napi_create_int32(env, state->winner, &winner_val);
  } else {
    napi_get_null(env, &winner_val);
  }

  // Build state context
  napi_create_object(env, &state_context);
  napi_value phase_val, active_player_val, turn_val;

  const char *phase_str = "UNKNOWN";
  switch (state->phase) {
    case PHASE_PREGAME_MULLIGAN: phase_str = "PREGAME_MULLIGAN"; break;
    case PHASE_START_OF_TURN: phase_str = "START_OF_TURN"; break;
    case PHASE_MAIN: phase_str = "MAIN"; break;
    case PHASE_RESPONSE_WINDOW: phase_str = "RESPONSE_WINDOW"; break;
    case PHASE_COMBAT_RESOLVE: phase_str = "COMBAT_RESOLVE"; break;
    case PHASE_END_TURN_ACTION: phase_str = "END_TURN_ACTION"; break;
    case PHASE_END_TURN: phase_str = "END_TURN"; break;
    case PHASE_END_MATCH: phase_str = "END_MATCH"; break;
    default: break;
  }
  napi_create_string_utf8(env, phase_str, NAPI_AUTO_LENGTH, &phase_val);
  napi_create_int32(env, state->active_player_index, &active_player_val);
  napi_create_int32(env, state->turn_number, &turn_val);

  napi_set_named_property(env, state_context, "phase", phase_val);
  napi_set_named_property(env, state_context, "activePlayer", active_player_val);
  napi_set_named_property(env, state_context, "turnNumber", turn_val);

  // Extract game logs
  uint8_t log_count = 0;
  const GameStateLog *logs = azk_get_game_logs(engine, &log_count);
  napi_create_array_with_length(env, log_count, &logs_arr);
  for (uint8_t i = 0; i < log_count; i++) {
    napi_value log_obj = serialize_game_log(env, &logs[i]);
    napi_set_element(env, logs_arr, i, log_obj);
  }

  napi_set_named_property(env, result, "success", success_val);
  napi_set_named_property(env, result, "invalid", invalid_val);
  napi_set_named_property(env, result, "gameOver", game_over_val);
  napi_set_named_property(env, result, "winner", winner_val);
  napi_set_named_property(env, result, "stateContext", state_context);
  napi_set_named_property(env, result, "logs", logs_arr);

  return result;
}

// getGameState(worldId: string) -> StateContext
static napi_value GetGameState(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char world_id[32];
  size_t len;
  napi_get_value_string_utf8(env, args[0], world_id, sizeof(world_id), &len);

  AzkEngine *engine = find_world(world_id);
  if (!engine) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }

  const GameState *state = azk_engine_game_state(engine);

  napi_value result, phase_val, active_player_val, turn_val, winner_val;
  napi_create_object(env, &result);

  const char *phase_str = "UNKNOWN";
  switch (state->phase) {
    case PHASE_PREGAME_MULLIGAN: phase_str = "PREGAME_MULLIGAN"; break;
    case PHASE_START_OF_TURN: phase_str = "START_OF_TURN"; break;
    case PHASE_MAIN: phase_str = "MAIN"; break;
    case PHASE_RESPONSE_WINDOW: phase_str = "RESPONSE_WINDOW"; break;
    case PHASE_COMBAT_RESOLVE: phase_str = "COMBAT_RESOLVE"; break;
    case PHASE_END_TURN_ACTION: phase_str = "END_TURN_ACTION"; break;
    case PHASE_END_TURN: phase_str = "END_TURN"; break;
    case PHASE_END_MATCH: phase_str = "END_MATCH"; break;
    default: break;
  }

  napi_create_string_utf8(env, phase_str, NAPI_AUTO_LENGTH, &phase_val);
  napi_create_int32(env, state->active_player_index, &active_player_val);
  napi_create_int32(env, state->turn_number, &turn_val);

  if (state->winner >= 0) {
    napi_create_int32(env, state->winner, &winner_val);
  } else {
    napi_get_null(env, &winner_val);
  }

  napi_set_named_property(env, result, "phase", phase_val);
  napi_set_named_property(env, result, "activePlayer", active_player_val);
  napi_set_named_property(env, result, "turnNumber", turn_val);
  napi_set_named_property(env, result, "winner", winner_val);

  return result;
}

// ============================================================================
// Observation serialization helpers
// ============================================================================

static const char *card_type_to_string(CardType type) {
  switch (type) {
    case CARD_TYPE_LEADER: return "LEADER";
    case CARD_TYPE_GATE: return "GATE";
    case CARD_TYPE_ENTITY: return "ENTITY";
    case CARD_TYPE_WEAPON: return "WEAPON";
    case CARD_TYPE_SPELL: return "SPELL";
    case CARD_TYPE_IKZ: return "IKZ";
    case CARD_TYPE_EXTRA_IKZ: return "EXTRA_IKZ";
    default: return "UNKNOWN";
  }
}

static const char *phase_to_string(Phase phase) {
  switch (phase) {
    case PHASE_PREGAME_MULLIGAN: return "PREGAME_MULLIGAN";
    case PHASE_START_OF_TURN: return "START_OF_TURN";
    case PHASE_MAIN: return "MAIN";
    case PHASE_RESPONSE_WINDOW: return "RESPONSE_WINDOW";
    case PHASE_COMBAT_RESOLVE: return "COMBAT_RESOLVE";
    case PHASE_END_TURN_ACTION: return "END_TURN_ACTION";
    case PHASE_END_TURN: return "END_TURN";
    case PHASE_END_MATCH: return "END_MATCH";
    default: return "UNKNOWN";
  }
}

static napi_value serialize_weapon(napi_env env, const WeaponObservationData *weapon) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value card_code, card_def_id, cur_atk;

  if (weapon->id.code) {
    napi_create_string_utf8(env, weapon->id.code, NAPI_AUTO_LENGTH, &card_code);
  } else {
    napi_get_null(env, &card_code);
  }
  napi_set_named_property(env, obj, "cardCode", card_code);

  napi_create_int32(env, weapon->id.id, &card_def_id);
  napi_set_named_property(env, obj, "cardDefId", card_def_id);

  napi_create_int32(env, weapon->base_stats.attack, &cur_atk);
  napi_set_named_property(env, obj, "curAtk", cur_atk);

  return obj;
}

static napi_value serialize_leader(napi_env env, const LeaderCardObservationData *leader) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value val;

  // cardCode
  if (leader->id.code) {
    napi_create_string_utf8(env, leader->id.code, NAPI_AUTO_LENGTH, &val);
  } else {
    napi_get_null(env, &val);
  }
  napi_set_named_property(env, obj, "cardCode", val);

  // cardDefId
  napi_create_int32(env, leader->id.id, &val);
  napi_set_named_property(env, obj, "cardDefId", val);

  // type
  napi_create_string_utf8(env, card_type_to_string(leader->type.value), NAPI_AUTO_LENGTH, &val);
  napi_set_named_property(env, obj, "type", val);

  // tapped/cooldown
  napi_get_boolean(env, leader->tap_state.tapped != 0, &val);
  napi_set_named_property(env, obj, "tapped", val);
  napi_get_boolean(env, leader->tap_state.cooldown != 0, &val);
  napi_set_named_property(env, obj, "cooldown", val);

  // curAtk/curHp
  napi_create_int32(env, leader->cur_stats.cur_atk, &val);
  napi_set_named_property(env, obj, "curAtk", val);
  napi_create_int32(env, leader->cur_stats.cur_hp, &val);
  napi_set_named_property(env, obj, "curHp", val);

  return obj;
}

static napi_value serialize_gate(napi_env env, const GateCardObservationData *gate) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value val;

  // cardCode
  if (gate->id.code) {
    napi_create_string_utf8(env, gate->id.code, NAPI_AUTO_LENGTH, &val);
  } else {
    napi_get_null(env, &val);
  }
  napi_set_named_property(env, obj, "cardCode", val);

  // cardDefId
  napi_create_int32(env, gate->id.id, &val);
  napi_set_named_property(env, obj, "cardDefId", val);

  // type
  napi_create_string_utf8(env, card_type_to_string(gate->type.value), NAPI_AUTO_LENGTH, &val);
  napi_set_named_property(env, obj, "type", val);

  // tapped/cooldown
  napi_get_boolean(env, gate->tap_state.tapped != 0, &val);
  napi_set_named_property(env, obj, "tapped", val);
  napi_get_boolean(env, gate->tap_state.cooldown != 0, &val);
  napi_set_named_property(env, obj, "cooldown", val);

  return obj;
}

static napi_value serialize_card(napi_env env, const CardObservationData *card) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value val;

  // cardCode
  if (card->id.code) {
    napi_create_string_utf8(env, card->id.code, NAPI_AUTO_LENGTH, &val);
  } else {
    napi_get_null(env, &val);
  }
  napi_set_named_property(env, obj, "cardCode", val);

  // cardDefId
  napi_create_int32(env, card->id.id, &val);
  napi_set_named_property(env, obj, "cardDefId", val);

  // type
  napi_create_string_utf8(env, card_type_to_string(card->type.value), NAPI_AUTO_LENGTH, &val);
  napi_set_named_property(env, obj, "type", val);

  // zoneIndex
  napi_create_int32(env, card->zone_index, &val);
  napi_set_named_property(env, obj, "zoneIndex", val);

  // ikzCost
  napi_create_int32(env, card->ikz_cost.ikz_cost, &val);
  napi_set_named_property(env, obj, "ikzCost", val);

  // tapped/cooldown
  napi_get_boolean(env, card->tap_state.tapped != 0, &val);
  napi_set_named_property(env, obj, "tapped", val);
  napi_get_boolean(env, card->tap_state.cooldown != 0, &val);
  napi_set_named_property(env, obj, "cooldown", val);

  // curAtk/curHp (nullable)
  if (card->has_cur_stats) {
    napi_create_int32(env, card->cur_stats.cur_atk, &val);
    napi_set_named_property(env, obj, "curAtk", val);
    napi_create_int32(env, card->cur_stats.cur_hp, &val);
    napi_set_named_property(env, obj, "curHp", val);
  } else {
    napi_get_null(env, &val);
    napi_set_named_property(env, obj, "curAtk", val);
    napi_set_named_property(env, obj, "curHp", val);
  }

  // gatePoints (nullable)
  if (card->has_gate_points) {
    napi_create_int32(env, card->gate_points.gate_points, &val);
  } else {
    napi_get_null(env, &val);
  }
  napi_set_named_property(env, obj, "gatePoints", val);

  // Status effects
  napi_get_boolean(env, card->is_frozen, &val);
  napi_set_named_property(env, obj, "isFrozen", val);
  napi_get_boolean(env, card->is_shocked, &val);
  napi_set_named_property(env, obj, "isShocked", val);
  napi_get_boolean(env, card->is_effect_immune, &val);
  napi_set_named_property(env, obj, "isEffectImmune", val);

  // Weapons array
  napi_value weapons_arr;
  napi_create_array_with_length(env, card->weapon_count, &weapons_arr);
  for (uint8_t i = 0; i < card->weapon_count; i++) {
    napi_value weapon = serialize_weapon(env, &card->weapons[i]);
    napi_set_element(env, weapons_arr, i, weapon);
  }
  napi_set_named_property(env, obj, "weapons", weapons_arr);

  return obj;
}

static napi_value serialize_ikz(napi_env env, const IKZCardObservationData *ikz) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value val;

  // cardCode
  if (ikz->id.code) {
    napi_create_string_utf8(env, ikz->id.code, NAPI_AUTO_LENGTH, &val);
  } else {
    napi_get_null(env, &val);
  }
  napi_set_named_property(env, obj, "cardCode", val);

  // cardDefId
  napi_create_int32(env, ikz->id.id, &val);
  napi_set_named_property(env, obj, "cardDefId", val);

  // type
  napi_create_string_utf8(env, card_type_to_string(ikz->type.value), NAPI_AUTO_LENGTH, &val);
  napi_set_named_property(env, obj, "type", val);

  // tapped/cooldown
  napi_get_boolean(env, ikz->tap_state.tapped != 0, &val);
  napi_set_named_property(env, obj, "tapped", val);
  napi_get_boolean(env, ikz->tap_state.cooldown != 0, &val);
  napi_set_named_property(env, obj, "cooldown", val);

  return obj;
}

// Check if a card slot is valid (has a card in it)
static bool is_card_valid(const CardObservationData *card) {
  return card->id.code != NULL;
}

static bool is_ikz_valid(const IKZCardObservationData *ikz) {
  return ikz->id.code != NULL;
}

static napi_value serialize_action_mask(napi_env env, const ActionMaskObs *mask) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value val;

  // primaryActionMask - boolean array
  napi_value primary_mask;
  napi_create_array_with_length(env, AZK_ACTION_TYPE_COUNT, &primary_mask);
  for (size_t i = 0; i < AZK_ACTION_TYPE_COUNT; i++) {
    napi_get_boolean(env, mask->primary_action_mask[i], &val);
    napi_set_element(env, primary_mask, i, val);
  }
  napi_set_named_property(env, obj, "primaryActionMask", primary_mask);

  // legalActionCount
  napi_create_int32(env, mask->legal_action_count, &val);
  napi_set_named_property(env, obj, "legalActionCount", val);

  // Legal action arrays (only up to legal_action_count entries)
  uint16_t count = mask->legal_action_count;

  napi_value legal_primary, legal_sub1, legal_sub2, legal_sub3;
  napi_create_array_with_length(env, count, &legal_primary);
  napi_create_array_with_length(env, count, &legal_sub1);
  napi_create_array_with_length(env, count, &legal_sub2);
  napi_create_array_with_length(env, count, &legal_sub3);

  for (uint16_t i = 0; i < count; i++) {
    napi_create_int32(env, mask->legal_primary[i], &val);
    napi_set_element(env, legal_primary, i, val);
    napi_create_int32(env, mask->legal_sub1[i], &val);
    napi_set_element(env, legal_sub1, i, val);
    napi_create_int32(env, mask->legal_sub2[i], &val);
    napi_set_element(env, legal_sub2, i, val);
    napi_create_int32(env, mask->legal_sub3[i], &val);
    napi_set_element(env, legal_sub3, i, val);
  }

  napi_set_named_property(env, obj, "legalPrimary", legal_primary);
  napi_set_named_property(env, obj, "legalSub1", legal_sub1);
  napi_set_named_property(env, obj, "legalSub2", legal_sub2);
  napi_set_named_property(env, obj, "legalSub3", legal_sub3);

  return obj;
}

static napi_value serialize_my_observation(napi_env env, const MyObservationData *my_obs) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value val;

  // leader
  napi_set_named_property(env, obj, "leader", serialize_leader(env, &my_obs->leader));

  // gate
  napi_set_named_property(env, obj, "gate", serialize_gate(env, &my_obs->gate));

  // hand - dynamic array of valid cards
  size_t hand_count = 0;
  for (size_t i = 0; i < MAX_HAND_SIZE && is_card_valid(&my_obs->hand[i]); i++) {
    hand_count++;
  }
  napi_value hand_arr;
  napi_create_array_with_length(env, hand_count, &hand_arr);
  for (size_t i = 0; i < hand_count; i++) {
    napi_set_element(env, hand_arr, i, serialize_card(env, &my_obs->hand[i]));
  }
  napi_set_named_property(env, obj, "hand", hand_arr);

  // garden - fixed size array with nulls for empty slots
  napi_value garden_arr;
  napi_create_array_with_length(env, GARDEN_SIZE, &garden_arr);
  for (size_t i = 0; i < GARDEN_SIZE; i++) {
    if (is_card_valid(&my_obs->garden[i])) {
      napi_set_element(env, garden_arr, i, serialize_card(env, &my_obs->garden[i]));
    } else {
      napi_get_null(env, &val);
      napi_set_element(env, garden_arr, i, val);
    }
  }
  napi_set_named_property(env, obj, "garden", garden_arr);

  // alley - fixed size array with nulls for empty slots
  napi_value alley_arr;
  napi_create_array_with_length(env, ALLEY_SIZE, &alley_arr);
  for (size_t i = 0; i < ALLEY_SIZE; i++) {
    if (is_card_valid(&my_obs->alley[i])) {
      napi_set_element(env, alley_arr, i, serialize_card(env, &my_obs->alley[i]));
    } else {
      napi_get_null(env, &val);
      napi_set_element(env, alley_arr, i, val);
    }
  }
  napi_set_named_property(env, obj, "alley", alley_arr);

  // ikzArea - array of valid IKZ cards
  size_t ikz_count = 0;
  for (size_t i = 0; i < IKZ_AREA_SIZE && is_ikz_valid(&my_obs->ikz_area[i]); i++) {
    ikz_count++;
  }
  napi_value ikz_arr;
  napi_create_array_with_length(env, ikz_count, &ikz_arr);
  for (size_t i = 0; i < ikz_count; i++) {
    napi_set_element(env, ikz_arr, i, serialize_ikz(env, &my_obs->ikz_area[i]));
  }
  napi_set_named_property(env, obj, "ikzArea", ikz_arr);

  // discard - dynamic array of valid cards
  size_t discard_count = 0;
  for (size_t i = 0; i < MAX_DECK_SIZE && is_card_valid(&my_obs->discard[i]); i++) {
    discard_count++;
  }
  napi_value discard_arr;
  napi_create_array_with_length(env, discard_count, &discard_arr);
  for (size_t i = 0; i < discard_count; i++) {
    napi_set_element(env, discard_arr, i, serialize_card(env, &my_obs->discard[i]));
  }
  napi_set_named_property(env, obj, "discard", discard_arr);

  // selection - dynamic array up to selection_count
  napi_value selection_arr;
  napi_create_array_with_length(env, my_obs->selection_count, &selection_arr);
  for (size_t i = 0; i < my_obs->selection_count; i++) {
    if (is_card_valid(&my_obs->selection[i])) {
      napi_set_element(env, selection_arr, i, serialize_card(env, &my_obs->selection[i]));
    } else {
      napi_get_null(env, &val);
      napi_set_element(env, selection_arr, i, val);
    }
  }
  napi_set_named_property(env, obj, "selection", selection_arr);

  // Counts and flags
  napi_create_int32(env, my_obs->deck_count, &val);
  napi_set_named_property(env, obj, "deckCount", val);

  napi_create_int32(env, my_obs->ikz_pile_count, &val);
  napi_set_named_property(env, obj, "ikzPileCount", val);

  napi_create_int32(env, my_obs->selection_count, &val);
  napi_set_named_property(env, obj, "selectionCount", val);

  napi_get_boolean(env, my_obs->has_ikz_token, &val);
  napi_set_named_property(env, obj, "hasIkzToken", val);

  return obj;
}

static napi_value serialize_opponent_observation(napi_env env, const OpponentObservationData *opp_obs) {
  napi_value obj;
  napi_create_object(env, &obj);

  napi_value val;

  // leader
  napi_set_named_property(env, obj, "leader", serialize_leader(env, &opp_obs->leader));

  // gate
  napi_set_named_property(env, obj, "gate", serialize_gate(env, &opp_obs->gate));

  // garden - fixed size array with nulls for empty slots
  napi_value garden_arr;
  napi_create_array_with_length(env, GARDEN_SIZE, &garden_arr);
  for (size_t i = 0; i < GARDEN_SIZE; i++) {
    if (is_card_valid(&opp_obs->garden[i])) {
      napi_set_element(env, garden_arr, i, serialize_card(env, &opp_obs->garden[i]));
    } else {
      napi_get_null(env, &val);
      napi_set_element(env, garden_arr, i, val);
    }
  }
  napi_set_named_property(env, obj, "garden", garden_arr);

  // alley - fixed size array with nulls for empty slots
  napi_value alley_arr;
  napi_create_array_with_length(env, ALLEY_SIZE, &alley_arr);
  for (size_t i = 0; i < ALLEY_SIZE; i++) {
    if (is_card_valid(&opp_obs->alley[i])) {
      napi_set_element(env, alley_arr, i, serialize_card(env, &opp_obs->alley[i]));
    } else {
      napi_get_null(env, &val);
      napi_set_element(env, alley_arr, i, val);
    }
  }
  napi_set_named_property(env, obj, "alley", alley_arr);

  // ikzArea - array of valid IKZ cards
  size_t ikz_count = 0;
  for (size_t i = 0; i < IKZ_AREA_SIZE && is_ikz_valid(&opp_obs->ikz_area[i]); i++) {
    ikz_count++;
  }
  napi_value ikz_arr;
  napi_create_array_with_length(env, ikz_count, &ikz_arr);
  for (size_t i = 0; i < ikz_count; i++) {
    napi_set_element(env, ikz_arr, i, serialize_ikz(env, &opp_obs->ikz_area[i]));
  }
  napi_set_named_property(env, obj, "ikzArea", ikz_arr);

  // discard - dynamic array of valid cards
  size_t discard_count = 0;
  for (size_t i = 0; i < MAX_DECK_SIZE && is_card_valid(&opp_obs->discard[i]); i++) {
    discard_count++;
  }
  napi_value discard_arr;
  napi_create_array_with_length(env, discard_count, &discard_arr);
  for (size_t i = 0; i < discard_count; i++) {
    napi_set_element(env, discard_arr, i, serialize_card(env, &opp_obs->discard[i]));
  }
  napi_set_named_property(env, obj, "discard", discard_arr);

  // Counts and flags
  napi_create_int32(env, opp_obs->hand_count, &val);
  napi_set_named_property(env, obj, "handCount", val);

  napi_create_int32(env, opp_obs->deck_count, &val);
  napi_set_named_property(env, obj, "deckCount", val);

  napi_create_int32(env, opp_obs->ikz_pile_count, &val);
  napi_set_named_property(env, obj, "ikzPileCount", val);

  napi_get_boolean(env, opp_obs->has_ikz_token, &val);
  napi_set_named_property(env, obj, "hasIkzToken", val);

  return obj;
}

// getObservation(worldId: string, playerIndex: number) -> ObservationData
static napi_value GetObservation(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char world_id[32];
  size_t len;
  napi_get_value_string_utf8(env, args[0], world_id, sizeof(world_id), &len);

  int32_t player_index;
  napi_get_value_int32(env, args[1], &player_index);

  AzkEngine *engine = find_world(world_id);
  if (!engine) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }

  // Debug: Check game state BEFORE calling observe
  const GameState *gs = azk_engine_game_state(engine);
  if (gs) {
    fprintf(stderr, "[addon.c] BEFORE observe: phase=%d, active_player=%d, winner=-1, player_index=%d\n",
            gs->phase, gs->active_player_index, player_index);

    // DEBUG: Print struct sizes to check for mismatch
    fprintf(stderr, "[addon.c] STRUCT SIZES: ObservationData=%zu, ActionMaskObs=%zu, MyObservationData=%zu, OpponentObservationData=%zu\n",
            sizeof(ObservationData), sizeof(ActionMaskObs), sizeof(MyObservationData), sizeof(OpponentObservationData));
    fprintf(stderr, "[addon.c] ActionMaskObs offsets: primary_action_mask@0, legal_action_count@%zu, legal_primary@%zu\n",
            offsetof(ActionMaskObs, legal_action_count), offsetof(ActionMaskObs, legal_primary));
    fprintf(stderr, "[addon.c] ObservationData offsets: my@0, opponent@%zu, phase@%zu, action_mask@%zu\n",
            offsetof(ObservationData, opponent_observation_data), offsetof(ObservationData, phase), offsetof(ObservationData, action_mask));
    fflush(stderr);
  }

  ObservationData obs;
  bool success = azk_engine_observe(engine, player_index, &obs);

  // Debug: Print action mask info from C engine
  fprintf(stderr, "[addon.c] GetObservation: player_index=%d, success=%d\n", player_index, success);
  fprintf(stderr, "[addon.c] action_mask: legal_action_count=%d, primary_mask[0]=%d, primary_mask[23]=%d\n",
          obs.action_mask.legal_action_count,
          obs.action_mask.primary_action_mask[0],
          obs.action_mask.primary_action_mask[23]);
  if (obs.action_mask.legal_action_count > 0) {
    fprintf(stderr, "[addon.c] First legal action: [%d, %d, %d, %d]\n",
            obs.action_mask.legal_primary[0],
            obs.action_mask.legal_sub1[0],
            obs.action_mask.legal_sub2[0],
            obs.action_mask.legal_sub3[0]);
  }
  fflush(stderr);

  if (!success) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }

  // Build full observation object
  napi_value result;
  napi_create_object(env, &result);

  napi_value val;

  // myObservationData
  napi_set_named_property(env, result, "myObservationData",
    serialize_my_observation(env, &obs.my_observation_data));

  // opponentObservationData
  napi_set_named_property(env, result, "opponentObservationData",
    serialize_opponent_observation(env, &obs.opponent_observation_data));

  // phase
  napi_create_string_utf8(env, phase_to_string(obs.phase), NAPI_AUTO_LENGTH, &val);
  napi_set_named_property(env, result, "phase", val);

  // actionMask
  napi_set_named_property(env, result, "actionMask",
    serialize_action_mask(env, &obs.action_mask));

  return result;
}

// getGameLogs(worldId: string) -> GameLog[]
static napi_value GetGameLogs(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  // Return empty array for now (TODO: implement log extraction)
  napi_value result;
  napi_create_array(env, &result);
  return result;
}

// ============================================================================
// Debug logging functions
// ============================================================================

// setDebugLogging(enabled: boolean): void
static napi_value SetDebugLogging(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  if (argc < 1) {
    return throw_error(env, "setDebugLogging requires enabled argument");
  }

  bool enabled;
  napi_get_value_bool(env, args[0], &enabled);

  azk_debug_log_set_enabled(enabled);

  return NULL;
}

// getDebugLogs(): { level: string, message: string }[]
static napi_value GetDebugLogs(napi_env env, napi_callback_info info) {
  (void)info; // Unused

  uint16_t count = 0;
  const AzkDebugLogEntry *entries = azk_debug_log_get_entries(&count);

  napi_value result;
  napi_create_array_with_length(env, count, &result);

  for (uint16_t i = 0; i < count; i++) {
    const AzkDebugLogEntry *entry = &entries[i];

    napi_value obj, level_val, message_val;
    napi_create_object(env, &obj);

    // Convert level to string
    const char *level_str = "INFO";
    switch (entry->level) {
      case AZK_DEBUG_LEVEL_INFO: level_str = "INFO"; break;
      case AZK_DEBUG_LEVEL_WARN: level_str = "WARN"; break;
      case AZK_DEBUG_LEVEL_ERROR: level_str = "ERROR"; break;
    }

    napi_create_string_utf8(env, level_str, NAPI_AUTO_LENGTH, &level_val);
    napi_create_string_utf8(env, entry->message, NAPI_AUTO_LENGTH, &message_val);

    napi_set_named_property(env, obj, "level", level_val);
    napi_set_named_property(env, obj, "message", message_val);

    napi_set_element(env, result, i, obj);
  }

  return result;
}

// clearDebugLogs(): void
static napi_value ClearDebugLogs(napi_env env, napi_callback_info info) {
  (void)env; // Unused
  (void)info; // Unused

  azk_debug_log_clear();

  return NULL;
}

// Module initialization
static napi_value Init(napi_env env, napi_value exports) {
  // Initialize world slots
  memset(world_slots, 0, sizeof(world_slots));

  napi_status status;

  // Define functions individually for better error handling
  napi_value fn;

  status = napi_create_function(env, "createWorld", NAPI_AUTO_LENGTH, CreateWorld, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "createWorld", fn);

  status = napi_create_function(env, "createWorldWithDecks", NAPI_AUTO_LENGTH, CreateWorldWithDecks, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "createWorldWithDecks", fn);

  status = napi_create_function(env, "destroyWorld", NAPI_AUTO_LENGTH, DestroyWorld, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "destroyWorld", fn);

  status = napi_create_function(env, "submitAction", NAPI_AUTO_LENGTH, SubmitAction, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "submitAction", fn);

  status = napi_create_function(env, "getObservation", NAPI_AUTO_LENGTH, GetObservation, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "getObservation", fn);

  status = napi_create_function(env, "getGameState", NAPI_AUTO_LENGTH, GetGameState, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "getGameState", fn);

  status = napi_create_function(env, "getGameLogs", NAPI_AUTO_LENGTH, GetGameLogs, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "getGameLogs", fn);

  status = napi_create_function(env, "requiresAction", NAPI_AUTO_LENGTH, RequiresAction, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "requiresAction", fn);

  status = napi_create_function(env, "isGameOver", NAPI_AUTO_LENGTH, IsGameOver, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "isGameOver", fn);

  status = napi_create_function(env, "getActivePlayer", NAPI_AUTO_LENGTH, GetActivePlayer, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "getActivePlayer", fn);

  status = napi_create_function(env, "setDebugLogging", NAPI_AUTO_LENGTH, SetDebugLogging, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "setDebugLogging", fn);

  status = napi_create_function(env, "getDebugLogs", NAPI_AUTO_LENGTH, GetDebugLogs, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "getDebugLogs", fn);

  status = napi_create_function(env, "clearDebugLogs", NAPI_AUTO_LENGTH, ClearDebugLogs, NULL, &fn);
  if (status == napi_ok) napi_set_named_property(env, exports, "clearDebugLogs", fn);

  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
