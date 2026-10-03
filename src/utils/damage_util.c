#include "utils/damage_util.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/ability_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

#include <stdlib.h>

static bool reward_entity_damage_tracking_enabled(void) {
  static int enabled = -1;
  if (enabled < 0) {
    const char *raw = getenv("AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP");
    enabled = raw != NULL && raw[0] != '\0' && strtof(raw, NULL) > 0.0f;
  }
  return enabled != 0;
}

static DamageTracker *ensure_damage_tracker(ecs_world_t *world,
                                            ecs_entity_t entity) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(entity != 0, ECS_INVALID_PARAMETER,
             "Cannot access DamageTracker for null entity");
  bool entity_alive = ecs_is_alive(world, entity);
  ecs_assert(entity_alive, ECS_INVALID_PARAMETER,
             "Cannot access DamageTracker for non-live entity %llu",
             (unsigned long long)entity);

  if (!ecs_has(world, entity, DamageTracker)) {
    ecs_set(world, entity, DamageTracker, {0});
  }

  return ecs_get_mut(world, entity, DamageTracker);
}

static void reset_damage_tracker_for_turn(const GameState *gs,
                                          DamageTracker *tracker) {
  if (gs == NULL || tracker == NULL || tracker->turn_marker == gs->turn_number) {
    return;
  }

  tracker->turn_marker = gs->turn_number;
  tracker->took_damage_this_turn = false;
  tracker->dealt_damage_this_turn = false;
  tracker->last_taken_from_effect = false;
  tracker->last_dealt_from_effect = false;
  tracker->last_damage_taken = 0;
  tracker->last_damage_dealt = 0;
  tracker->last_damage_source = 0;
  tracker->last_damage_recipient = 0;
  tracker->tracked_source_count = 0;
  for (uint8_t i = 0; i < AZK_MAX_TRACKED_DAMAGE_SOURCES; ++i) {
    tracker->tracked_sources[i] = 0;
  }
}

bool azk_damage_tracker_is_current_turn(ecs_world_t *world,
                                        const DamageTracker *tracker) {
  if (world == NULL || tracker == NULL) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  return gs != NULL && tracker->turn_marker == gs->turn_number;
}

const DamageTracker *azk_get_current_turn_damage_tracker(ecs_world_t *world,
                                                         ecs_entity_t entity) {
  const DamageTracker *tracker = ecs_get(world, entity, DamageTracker);
  if (!azk_damage_tracker_is_current_turn(world, tracker)) {
    return NULL;
  }

  return tracker;
}

static void maybe_queue_damage_trigger(ecs_world_t *world, ecs_entity_t card,
                                       uint8_t timing_tag, ecs_id_t tag_id) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (owner == 0) {
    return;
  }

  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_timed_abilities(
      world, card, tag_id, abilities, AZK_MAX_CARD_ABILITIES);
  for (uint8_t i = 0; i < ability_count; ++i) {
    azk_queue_triggered_effect(world, abilities[i], owner, timing_tag);
  }
}

void azk_record_damage_event(ecs_world_t *world, ecs_entity_t source,
                             ecs_entity_t target, int8_t actual_damage,
                             bool from_effect) {
  if (world == NULL || target == 0 || actual_damage <= 0) {
    return;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);

  if (reward_entity_damage_tracking_enabled() &&
      ecs_has(world, target, TEntity) && !ecs_has(world, target, TLeader)) {
    const CurStats *target_stats = ecs_get(world, target, CurStats);
    const ecs_entity_t owner = ecs_get_target(world, target, Rel_OwnedBy, 0);
    const PlayerNumber *player_number =
        owner != 0 ? ecs_get(world, owner, PlayerNumber) : NULL;
    if (target_stats != NULL && player_number != NULL &&
        player_number->player_number < MAX_PLAYERS_PER_MATCH) {
      int effective_damage = (int)actual_damage;
      const int hp_before = (int)target_stats->cur_hp + effective_damage;
      if (effective_damage > hp_before) {
        effective_damage = hp_before;
      }
      if (effective_damage > 0) {
        GameState *reward_gs = ecs_singleton_get_mut(world, GameState);
        if (reward_gs != NULL) {
          reward_gs->entity_damage_taken[player_number->player_number] +=
              (uint32_t)effective_damage;
          ecs_singleton_modified(world, GameState);
          gs = reward_gs;
        }
      }
    }
  }

  DamageTracker *target_tracker = ensure_damage_tracker(world, target);
  ecs_assert(target_tracker != NULL, ECS_INVALID_OPERATION,
             "DamageTracker missing for target %llu",
             (unsigned long long)target);
  reset_damage_tracker_for_turn(gs, target_tracker);
  target_tracker->took_damage_this_turn = true;
  target_tracker->last_taken_from_effect = from_effect;
  target_tracker->last_damage_taken = actual_damage;
  target_tracker->last_damage_source = source;

  bool source_seen = false;
  for (uint8_t i = 0; i < target_tracker->tracked_source_count; ++i) {
    if (target_tracker->tracked_sources[i] == source) {
      source_seen = true;
      break;
    }
  }
  if (!source_seen &&
      target_tracker->tracked_source_count < AZK_MAX_TRACKED_DAMAGE_SOURCES) {
    target_tracker
        ->tracked_sources[target_tracker->tracked_source_count++] = source;
  }
  ecs_modified(world, target, DamageTracker);

  maybe_queue_damage_trigger(world, target, TIMING_TAG_WHEN_TAKES_DAMAGE,
                             ecs_id(AWhenTakesDamage));

  if (source != 0) {
    DamageTracker *source_tracker = ensure_damage_tracker(world, source);
    ecs_assert(source_tracker != NULL, ECS_INVALID_OPERATION,
               "DamageTracker missing for source %llu",
               (unsigned long long)source);
    reset_damage_tracker_for_turn(gs, source_tracker);
    source_tracker->dealt_damage_this_turn = true;
    source_tracker->last_dealt_from_effect = from_effect;
    source_tracker->last_damage_dealt = actual_damage;
    source_tracker->last_damage_recipient = target;
    ecs_modified(world, source, DamageTracker);

    maybe_queue_damage_trigger(world, source, TIMING_TAG_WHEN_DEALS_DAMAGE,
                               ecs_id(AWhenDealsDamage));
  }
}

static ecs_entity_t current_damage_source(ecs_world_t *world) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return ctx != NULL ? ctx->runtime.source_card : 0;
}

bool azk_enqueue_pending_damage_redirect(ecs_world_t *world, ecs_entity_t source,
                                         ecs_entity_t original_target,
                                         ecs_entity_t owner, int8_t damage) {
  PendingDamageRedirectQueue *queue =
      ecs_singleton_get_mut(world, PendingDamageRedirectQueue);
  if (queue == NULL || queue->count >= MAX_PENDING_DAMAGE_REDIRECTS) {
    return false;
  }

  queue->entries[queue->count++] = (PendingDamageRedirect){
      .source_card = source,
      .original_target = original_target,
      .owner = owner,
      .damage = damage,
  };
  ecs_singleton_modified(world, PendingDamageRedirectQueue);
  return true;
}

bool azk_has_pending_damage_redirect_for_target(ecs_world_t *world,
                                                ecs_entity_t target) {
  const PendingDamageRedirectQueue *queue =
      ecs_singleton_get(world, PendingDamageRedirectQueue);
  if (queue == NULL) {
    return false;
  }

  for (uint8_t i = 0; i < queue->count; ++i) {
    if (queue->entries[i].original_target == target) {
      return true;
    }
  }

  return false;
}

bool azk_consume_pending_damage_redirect(ecs_world_t *world,
                                         ecs_entity_t target,
                                         PendingDamageRedirect *out_redirect) {
  PendingDamageRedirectQueue *queue =
      ecs_singleton_get_mut(world, PendingDamageRedirectQueue);
  if (queue == NULL) {
    return false;
  }

  for (uint8_t i = 0; i < queue->count; ++i) {
    if (queue->entries[i].original_target != target) {
      continue;
    }

    if (out_redirect != NULL) {
      *out_redirect = queue->entries[i];
    }

    for (uint8_t j = i + 1; j < queue->count; ++j) {
      queue->entries[j - 1] = queue->entries[j];
    }
    queue->count--;
    ecs_singleton_modified(world, PendingDamageRedirectQueue);
    return true;
  }

  return false;
}

static bool maybe_queue_pekiro_redirect(ecs_world_t *world, ecs_entity_t source,
                                        ecs_entity_t target, int8_t damage) {
  const CardId *card_id = ecs_get(world, target, CardId);
  if (card_id == NULL || card_id->id != CARD_DEF_AZK01_062 || damage <= 0 ||
      azk_has_pending_damage_redirect_for_target(world, target)) {
    return false;
  }

  ecs_entity_t owner = ecs_get_target(world, target, Rel_OwnedBy, 0);
  if (owner == 0 ||
      !azk_enqueue_pending_damage_redirect(world, source, target, owner, damage)) {
    return false;
  }

  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_timed_abilities(
      world, target, ecs_id(AWhenTakesDamage), abilities,
      AZK_MAX_CARD_ABILITIES);
  for (uint8_t i = 0; i < ability_count; ++i) {
    if (azk_queue_triggered_effect(world, abilities[i], owner,
                                   TIMING_TAG_WHEN_TAKES_DAMAGE)) {
      return true;
    }
  }

  azk_consume_pending_damage_redirect(world, target, NULL);
  return false;
}

static bool deal_effect_damage_from_source_internal(ecs_world_t *world,
                                                    ecs_entity_t source,
                                                    ecs_entity_t target,
                                                    int8_t damage,
                                                    bool allow_redirect) {
  if (allow_redirect &&
      maybe_queue_pekiro_redirect(world, source, target, damage)) {
    return true;
  }

  // Check if target is effect immune
  if (is_effect_immune(world, target)) {
    cli_render_logf("[Damage] Effect damage blocked by EffectImmune");
    return false;
  }

  int16_t adjusted_damage =
      (int16_t)damage - (int16_t)get_total_carapace_value(world, target);
  if (adjusted_damage <= 0) {
    cli_render_logf("[Damage] Effect damage blocked by Carapace");
    return false;
  }

  // Apply damage to target's current HP
  CurStats *cur_stats = ecs_get_mut(world, target, CurStats);
  if (cur_stats == NULL) {
    cli_render_logf("[Damage] Target has no CurStats component");
    return false;
  }

  const bool target_has_godmode = azk_card_has_godmode_in_play(world, target);
  int8_t prev_hp = cur_stats->cur_hp;
  int16_t next_hp = (int16_t)prev_hp - adjusted_damage;
  if (target_has_godmode && next_hp < 0) {
    next_hp = 0;
  }
  cur_stats->cur_hp = (int8_t)next_hp;
  ecs_modified(world, target, CurStats);
  int8_t hp_delta = (int8_t)(cur_stats->cur_hp - prev_hp);
  int8_t actual_damage = (int8_t)(prev_hp - cur_stats->cur_hp);

  // Log the HP change for frontend state updates
  azk_log_card_stat_change(world, target, 0, hp_delta, cur_stats->cur_atk,
                           cur_stats->cur_hp);

  cli_render_logf("[Damage] Dealt %d effect damage (HP: %d)",
                  (int)actual_damage, cur_stats->cur_hp);

  if (actual_damage > 0) {
    azk_record_damage_event(world, source, target, actual_damage, true);
  }

  // Check for death (HP <= 0)
  if (cur_stats->cur_hp <= 0) {
    if (target_has_godmode) {
      cli_render_logf("[Damage] Godmode kept card in play at 0 HP");
    } else if (ecs_has(world, target, TLeader)) {
      // Leader defeated - determine winner based on target's owner
      GameState *gs = ecs_singleton_get_mut(world, GameState);
      ecs_entity_t target_parent = ecs_get_target(world, target, EcsChildOf, 0);

      // Find which player owns this leader
      for (int i = 0; i < MAX_PLAYERS_PER_MATCH; i++) {
        if (target_parent == gs->zones[i].leader) {
          // Owner of defeated leader loses, opponent wins
          gs->winner = (i + 1) % MAX_PLAYERS_PER_MATCH;
          ecs_singleton_modified(world, GameState);
          cli_render_logf("[Damage] Leader defeated - player %d wins",
                          gs->winner);
          break;
        }
      }
    } else {
      // Non-leader entity - discard it
      discard_card(world, target);
      cli_render_logf("[Damage] Entity defeated by effect damage - discarded");
    }
  }

  return true;
}

bool deal_effect_damage_from_source(ecs_world_t *world, ecs_entity_t source,
                                    ecs_entity_t target, int8_t damage) {
  return deal_effect_damage_from_source_internal(world, source, target, damage,
                                                 true);
}

bool deal_effect_damage_from_source_no_redirect(ecs_world_t *world,
                                                ecs_entity_t source,
                                                ecs_entity_t target,
                                                int8_t damage) {
  return deal_effect_damage_from_source_internal(world, source, target, damage,
                                                 false);
}

bool deal_effect_damage(ecs_world_t *world, ecs_entity_t target,
                        int8_t damage) {
  return deal_effect_damage_from_source(world, current_damage_source(world),
                                        target, damage);
}
