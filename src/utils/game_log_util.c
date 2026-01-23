#include "utils/game_log_util.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"

/* Internal helper to add a log entry */
static GameStateLog *add_log_entry(ecs_world_t *world) {
  GameStateLogContext *ctx = ecs_singleton_get_mut(world, GameStateLogContext);
  if (!ctx) {
    return NULL;
  }
  if (ctx->count >= MAX_GAME_STATE_LOGS) {
    cli_render_logf("[GameLog] Warning: Log buffer full, dropping log");
    return NULL;
  }
  GameStateLog *log = &ctx->logs[ctx->count++];
  ecs_singleton_modified(world, GameStateLogContext);
  return log;
}

/* Get player number from card entity */
static uint8_t get_card_player(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (owner == 0) {
    return 0;
  }
  const PlayerNumber *pn = ecs_get(world, owner, PlayerNumber);
  return pn ? pn->player_number : 0;
}

/* Get CardDefId from card entity */
static CardDefId get_card_def_id(ecs_world_t *world, ecs_entity_t card) {
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id) {
    // Try prefab
    ecs_entity_t prefab = ecs_get_target(world, card, EcsIsA, 0);
    if (prefab != 0) {
      card_id = ecs_get(world, prefab, CardId);
    }
  }
  return card_id ? card_id->id : CARD_DEF_IKZ_001;
}

/* Get zone index from card entity (uses ZoneIndex component only) */
static int8_t get_card_zone_index(ecs_world_t *world, ecs_entity_t card) {
  const ZoneIndex *zi = ecs_get(world, card, ZoneIndex);
  return zi ? (int8_t)zi->index : -1;
}

int8_t azk_get_card_index_in_zone(ecs_world_t *world, ecs_entity_t card,
                                  ecs_entity_t zone) {
  if (card == 0 || zone == 0) {
    return -1;
  }

  // Fast path: check for ZoneIndex component (garden/alley cards)
  const ZoneIndex *zi = ecs_get(world, card, ZoneIndex);
  if (zi) {
    return (int8_t)zi->index;
  }

  // Slow path: search ordered children (hand/deck/other zones)
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    if (cards.ids[i] == card) {
      return (int8_t)i;
    }
  }

  return -1;
}

void azk_clear_game_logs(ecs_world_t *world) {
  GameStateLogContext *ctx = ecs_singleton_get_mut(world, GameStateLogContext);
  if (ctx) {
    ctx->count = 0;
    ecs_singleton_modified(world, GameStateLogContext);
  }
}

uint8_t azk_get_game_log_count(ecs_world_t *world) {
  const GameStateLogContext *ctx = ecs_singleton_get(world, GameStateLogContext);
  return ctx ? ctx->count : 0;
}

const GameStateLog *azk_get_game_logs(ecs_world_t *world, uint8_t *out_count) {
  const GameStateLogContext *ctx = ecs_singleton_get(world, GameStateLogContext);
  if (!ctx) {
    if (out_count) {
      *out_count = 0;
    }
    return NULL;
  }
  if (out_count) {
    *out_count = ctx->count;
  }
  return ctx->logs;
}

GameLogZone azk_zone_entity_to_log_zone(ecs_world_t *world,
                                        ecs_entity_t zone_entity) {
  if (zone_entity == 0) {
    return GLOG_ZONE_NONE;
  }
  if (ecs_has_id(world, zone_entity, ZDeck)) {
    return GLOG_ZONE_DECK;
  }
  if (ecs_has_id(world, zone_entity, ZHand)) {
    return GLOG_ZONE_HAND;
  }
  if (ecs_has_id(world, zone_entity, ZLeader)) {
    return GLOG_ZONE_LEADER;
  }
  if (ecs_has_id(world, zone_entity, ZGate)) {
    return GLOG_ZONE_GATE;
  }
  if (ecs_has_id(world, zone_entity, ZGarden)) {
    return GLOG_ZONE_GARDEN;
  }
  if (ecs_has_id(world, zone_entity, ZAlley)) {
    return GLOG_ZONE_ALLEY;
  }
  if (ecs_has_id(world, zone_entity, ZIKZPileTag)) {
    return GLOG_ZONE_IKZ_PILE;
  }
  if (ecs_has_id(world, zone_entity, ZIKZAreaTag)) {
    return GLOG_ZONE_IKZ_AREA;
  }
  if (ecs_has_id(world, zone_entity, ZDiscard)) {
    return GLOG_ZONE_DISCARD;
  }
  if (ecs_has_id(world, zone_entity, ZSelection)) {
    return GLOG_ZONE_SELECTION;
  }
  return GLOG_ZONE_NONE;
}

GameLogCardRef azk_make_card_ref(ecs_world_t *world, ecs_entity_t card) {
  GameLogCardRef ref = {0};
  if (card == 0) {
    return ref;
  }

  ref.player = get_card_player(world, card);
  ref.card_def_id = get_card_def_id(world, card);

  // Get zone from parent entity
  ecs_entity_t zone_entity = ecs_get_target(world, card, EcsChildOf, 0);
  ref.zone = azk_zone_entity_to_log_zone(world, zone_entity);
  // Use azk_get_card_index_in_zone for slow path search (handles IKZ cards)
  ref.zone_index = azk_get_card_index_in_zone(world, card, zone_entity);

  return ref;
}

GameLogCardMetadata azk_make_card_metadata(ecs_world_t *world,
                                           ecs_entity_t card) {
  GameLogCardMetadata meta = {0};
  if (card == 0) {
    return meta;
  }

  // Get current stats
  const CurStats *stats = ecs_get(world, card, CurStats);
  if (stats) {
    meta.cur_atk = stats->cur_atk;
    meta.cur_hp = stats->cur_hp;
  }

  // Get tap state
  const TapState *tap = ecs_get(world, card, TapState);
  if (tap) {
    meta.tapped = tap->tapped != 0;
    meta.cooldown = tap->cooldown != 0;
  }

  // Check keywords
  meta.has_charge = ecs_has(world, card, Charge);
  meta.has_defender = ecs_has(world, card, Defender);
  meta.has_infiltrate = ecs_has(world, card, Infiltrate);
  meta.is_frozen = ecs_has(world, card, Frozen);
  meta.is_effect_immune = ecs_has(world, card, EffectImmune);

  // Get attached weapons (children with TWeapon tag)
  meta.weapon_count = 0;
  ecs_iter_t it = ecs_children(world, card);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count && meta.weapon_count < MAX_ATTACHED_WEAPONS_LOG;
         i++) {
      if (ecs_has_id(world, it.entities[i], TWeapon)) {
        meta.attached_weapons[meta.weapon_count++] =
            get_card_def_id(world, it.entities[i]);
      }
    }
  }

  return meta;
}

/* ========== Zone Movement Logs ========== */

void azk_log_card_zone_moved(ecs_world_t *world, ecs_entity_t card,
                             GameLogZone from_zone, int8_t from_index,
                             GameLogZone to_zone, int8_t to_index) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_CARD_ZONE_MOVED;
  log->data.zone_moved.card = azk_make_card_ref(world, card);
  // Override zone with to_zone (card ref gets current zone which may be to_zone
  // already)
  log->data.zone_moved.card.zone = to_zone;
  log->data.zone_moved.card.zone_index = to_index;
  log->data.zone_moved.from_zone = from_zone;
  log->data.zone_moved.from_index = from_index;
  log->data.zone_moved.to_zone = to_zone;
  log->data.zone_moved.to_index = to_index;
  log->data.zone_moved.metadata = azk_make_card_metadata(world, card);
}

void azk_log_card_zone_moved_ex(ecs_world_t *world, uint8_t player,
                                CardDefId card_def_id, GameLogZone from_zone,
                                int8_t from_index, GameLogZone to_zone,
                                int8_t to_index,
                                const GameLogCardMetadata *metadata) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_CARD_ZONE_MOVED;
  log->data.zone_moved.card.player = player;
  log->data.zone_moved.card.card_def_id = card_def_id;
  log->data.zone_moved.card.zone = to_zone;
  log->data.zone_moved.card.zone_index = to_index;
  log->data.zone_moved.from_zone = from_zone;
  log->data.zone_moved.from_index = from_index;
  log->data.zone_moved.to_zone = to_zone;
  log->data.zone_moved.to_index = to_index;
  if (metadata) {
    log->data.zone_moved.metadata = *metadata;
  }
}

/* ========== Tap State Logs ========== */

void azk_log_card_tap_state_changed(ecs_world_t *world, ecs_entity_t card,
                                    GameLogTapState new_state) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_CARD_TAP_STATE_CHANGED;
  log->data.tap_changed.card = azk_make_card_ref(world, card);
  log->data.tap_changed.new_state = new_state;
}

void azk_log_card_tap_state_changed_ex(ecs_world_t *world, ecs_entity_t card,
                                       GameLogTapState new_state,
                                       GameLogZone zone, int8_t zone_index) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_CARD_TAP_STATE_CHANGED;
  log->data.tap_changed.card = azk_make_card_ref(world, card);
  // Override zone/index with explicit values (bypasses deferred zone lookup)
  log->data.tap_changed.card.zone = zone;
  log->data.tap_changed.card.zone_index = zone_index;
  log->data.tap_changed.new_state = new_state;
}

/* ========== Stat Change Logs ========== */

void azk_log_card_stat_change(ecs_world_t *world, ecs_entity_t card,
                              int8_t atk_delta, int8_t hp_delta, int8_t new_atk,
                              int8_t new_hp) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_CARD_STAT_CHANGE;
  log->data.stat_change.card = azk_make_card_ref(world, card);
  log->data.stat_change.atk_delta = atk_delta;
  log->data.stat_change.hp_delta = hp_delta;
  log->data.stat_change.new_atk = new_atk;
  log->data.stat_change.new_hp = new_hp;
}

/* ========== Status Effect Logs ========== */

void azk_log_status_effect_applied(ecs_world_t *world, ecs_entity_t card,
                                   GameLogStatusEffect effect, int8_t duration) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_STATUS_EFFECT_APPLIED;
  log->data.status_applied.card = azk_make_card_ref(world, card);
  log->data.status_applied.effect = effect;
  log->data.status_applied.duration = duration;
}

void azk_log_status_effect_expired(ecs_world_t *world, ecs_entity_t card,
                                   GameLogStatusEffect effect) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_STATUS_EFFECT_EXPIRED;
  log->data.status_expired.card = azk_make_card_ref(world, card);
  log->data.status_expired.effect = effect;
}

/* ========== Combat Logs ========== */

void azk_log_combat_declared(ecs_world_t *world, ecs_entity_t attacker,
                             ecs_entity_t target) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_COMBAT_DECLARED;
  log->data.combat_declared.attacker = azk_make_card_ref(world, attacker);
  log->data.combat_declared.target = azk_make_card_ref(world, target);
}

void azk_log_defender_declared(ecs_world_t *world, ecs_entity_t defender) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_DEFENDER_DECLARED;
  log->data.defender_declared.defender = azk_make_card_ref(world, defender);
}

void azk_log_combat_damage(ecs_world_t *world, ecs_entity_t attacker,
                           ecs_entity_t defender, int8_t attacker_dmg_dealt,
                           int8_t attacker_dmg_taken, int8_t defender_dmg_dealt,
                           int8_t defender_dmg_taken) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_COMBAT_DAMAGE;
  log->data.combat_damage.attacker = azk_make_card_ref(world, attacker);
  log->data.combat_damage.defender = azk_make_card_ref(world, defender);
  log->data.combat_damage.attacker_damage_dealt = attacker_dmg_dealt;
  log->data.combat_damage.attacker_damage_taken = attacker_dmg_taken;
  log->data.combat_damage.defender_damage_dealt = defender_dmg_dealt;
  log->data.combat_damage.defender_damage_taken = defender_dmg_taken;
}

void azk_log_entity_died(ecs_world_t *world, ecs_entity_t card,
                         GameLogDeathCause cause) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_ENTITY_DIED;
  log->data.entity_died.card = azk_make_card_ref(world, card);
  log->data.entity_died.cause = cause;
}

/* ========== Ability Logs ========== */

void azk_log_effect_queued(ecs_world_t *world, ecs_entity_t card,
                           uint8_t ability_index, uint8_t trigger_tag) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_EFFECT_QUEUED;
  log->data.effect_queued.card = azk_make_card_ref(world, card);
  log->data.effect_queued.ability_index = ability_index;
  log->data.effect_queued.trigger_tag = trigger_tag;
}

void azk_log_effect_enabled(ecs_world_t *world, ecs_entity_t card,
                            uint8_t ability_index) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_CARD_EFFECT_ENABLED;
  log->data.effect_enabled.card = azk_make_card_ref(world, card);
  log->data.effect_enabled.ability_index = ability_index;
}

/* ========== Game Flow Logs ========== */

void azk_log_deck_shuffled(ecs_world_t *world, uint8_t player,
                           GameLogShuffleReason reason) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_DECK_SHUFFLED;
  log->data.deck_shuffled.player = player;
  log->data.deck_shuffled.reason = reason;
}

void azk_log_turn_started(ecs_world_t *world, uint8_t player,
                          uint16_t turn_number) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  // Update turn number in context
  GameStateLogContext *ctx = ecs_singleton_get_mut(world, GameStateLogContext);
  if (ctx) {
    ctx->turn_number = turn_number;
    ecs_singleton_modified(world, GameStateLogContext);
  }

  log->type = GLOG_TURN_STARTED;
  log->data.turn_started.player = player;
  log->data.turn_started.turn_number = turn_number;
}

void azk_log_turn_ended(ecs_world_t *world, uint8_t player,
                        uint16_t turn_number) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_TURN_ENDED;
  log->data.turn_ended.player = player;
  log->data.turn_ended.turn_number = turn_number;
}

void azk_log_game_ended(ecs_world_t *world, int8_t winner,
                        GameLogEndReason reason) {
  GameStateLog *log = add_log_entry(world);
  if (!log) {
    return;
  }

  log->type = GLOG_GAME_ENDED;
  log->data.game_ended.winner = winner;
  log->data.game_ended.reason = reason;
}

/* ========== String Conversion Utilities ========== */

const char *azk_log_type_to_string(GameLogType type) {
  switch (type) {
  case GLOG_NONE:
    return "NONE";
  case GLOG_CARD_ZONE_MOVED:
    return "ZONE_MOVED";
  case GLOG_CARD_STAT_CHANGE:
    return "STAT_CHANGE";
  case GLOG_CARD_TAP_STATE_CHANGED:
    return "TAP_CHANGED";
  case GLOG_STATUS_EFFECT_APPLIED:
    return "STATUS_APPLIED";
  case GLOG_STATUS_EFFECT_EXPIRED:
    return "STATUS_EXPIRED";
  case GLOG_COMBAT_DECLARED:
    return "COMBAT_DECLARED";
  case GLOG_DEFENDER_DECLARED:
    return "DEFENDER_DECLARED";
  case GLOG_COMBAT_DAMAGE:
    return "COMBAT_DAMAGE";
  case GLOG_ENTITY_DIED:
    return "ENTITY_DIED";
  case GLOG_EFFECT_QUEUED:
    return "EFFECT_QUEUED";
  case GLOG_CARD_EFFECT_ENABLED:
    return "EFFECT_ENABLED";
  case GLOG_DECK_SHUFFLED:
    return "DECK_SHUFFLED";
  case GLOG_TURN_STARTED:
    return "TURN_STARTED";
  case GLOG_TURN_ENDED:
    return "TURN_ENDED";
  case GLOG_GAME_ENDED:
    return "GAME_ENDED";
  default:
    return "UNKNOWN";
  }
}

const char *azk_log_zone_to_string(GameLogZone zone) {
  switch (zone) {
  case GLOG_ZONE_NONE:
    return "NONE";
  case GLOG_ZONE_DECK:
    return "DECK";
  case GLOG_ZONE_HAND:
    return "HAND";
  case GLOG_ZONE_LEADER:
    return "LEADER";
  case GLOG_ZONE_GATE:
    return "GATE";
  case GLOG_ZONE_GARDEN:
    return "GARDEN";
  case GLOG_ZONE_ALLEY:
    return "ALLEY";
  case GLOG_ZONE_IKZ_PILE:
    return "IKZ_PILE";
  case GLOG_ZONE_IKZ_AREA:
    return "IKZ_AREA";
  case GLOG_ZONE_DISCARD:
    return "DISCARD";
  case GLOG_ZONE_SELECTION:
    return "SELECTION";
  case GLOG_ZONE_EQUIPPED:
    return "EQUIPPED";
  default:
    return "UNKNOWN";
  }
}
