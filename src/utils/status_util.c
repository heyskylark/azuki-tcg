#include "utils/status_util.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"

static CardConditionCountdown
default_condition_countdown(ecs_world_t *world, ecs_entity_t entity) {
  ecs_entity_t prefab = ecs_get_target(world, entity, EcsIsA, 0);
  bool has_innate_effect_immune =
      prefab != 0 && ecs_has_id(world, prefab, ecs_id(EffectImmune));
  return (CardConditionCountdown){
      .frozen_duration = 0,
      .shocked_duration = 0,
      .effect_immune_duration = has_innate_effect_immune ? -1 : 0,
      .timed_tag_grant_count = 0,
  };
}

static CardConditionCountdown *ensure_condition_countdown(ecs_world_t *world,
                                                          ecs_entity_t entity) {
  const CardConditionCountdown *existing =
      ecs_get(world, entity, CardConditionCountdown);
  CardConditionCountdown *countdown =
      ecs_ensure(world, entity, CardConditionCountdown);
  if (existing == NULL) {
    *countdown = default_condition_countdown(world, entity);
  }
  return countdown;
}

static bool prefab_has_tag(ecs_world_t *world, ecs_entity_t entity, ecs_id_t tag) {
  ecs_entity_t prefab = ecs_get_target(world, entity, EcsIsA, 0);
  return prefab != 0 && ecs_has_id(world, prefab, tag);
}

void remove_all_combat_damage_modifiers(ecs_world_t *world,
                                        ecs_entity_t entity);

static bool is_keyword_tag(ecs_id_t tag) {
  return tag == ecs_id(Charge) || tag == ecs_id(Defender) ||
         tag == ecs_id(Infiltrate) || tag == ecs_id(Godmode);
}

static bool try_get_status_effect_for_tag(ecs_id_t tag,
                                          GameLogStatusEffect *out_effect) {
  if (tag == ecs_id(Rooted)) {
    if (out_effect != NULL) {
      *out_effect = GLOG_STATUS_ROOTED;
    }
    return true;
  }

  return false;
}

static void log_tag_change_if_keyword(ecs_world_t *world, ecs_entity_t entity,
                                      ecs_id_t tag, bool new_present) {
  if (is_keyword_tag(tag)) {
    azk_log_card_keywords_changed_override(world, entity, tag, new_present);
  }
}

static bool countdown_has_active_tag_grant(const CardConditionCountdown *countdown,
                                           ecs_id_t tag) {
  if (countdown == NULL) {
    return false;
  }

  for (uint8_t i = 0; i < countdown->timed_tag_grant_count; i++) {
    if (countdown->timed_tag_grants[i].tag == tag) {
      return true;
    }
  }

  return false;
}

static void maybe_remove_unbacked_tag(ecs_world_t *world, ecs_entity_t entity,
                                      ecs_id_t tag) {
  const CardConditionCountdown *countdown =
      ecs_get(world, entity, CardConditionCountdown);
  if (prefab_has_tag(world, entity, tag) ||
      countdown_has_active_tag_grant(countdown, tag) ||
      !ecs_has_id(world, entity, tag)) {
    return;
  }

  ecs_remove_id(world, entity, tag);
  GameLogStatusEffect effect;
  if (try_get_status_effect_for_tag(tag, &effect)) {
    azk_log_status_effect_expired(world, entity, effect);
  }
  log_tag_change_if_keyword(world, entity, tag, false);
}

static void remove_tag_grant_at(CardConditionCountdown *countdown, uint8_t index) {
  if (countdown == NULL || index >= countdown->timed_tag_grant_count) {
    return;
  }

  for (uint8_t i = index; i + 1 < countdown->timed_tag_grant_count; i++) {
    countdown->timed_tag_grants[i] = countdown->timed_tag_grants[i + 1];
  }

  countdown->timed_tag_grant_count--;
  countdown->timed_tag_grants[countdown->timed_tag_grant_count] =
      (TimedTagGrant){0};
}

void apply_frozen(ecs_world_t *world, ecs_entity_t entity, int8_t duration) {
  // Add the Frozen tag
  ecs_add(world, entity, Frozen);

  // Set or update CardConditionCountdown component
  CardConditionCountdown *countdown = ensure_condition_countdown(world, entity);
  countdown->frozen_duration = duration;
  ecs_modified(world, entity, CardConditionCountdown);

  // Log status effect applied
  azk_log_status_effect_applied(world, entity, GLOG_STATUS_FROZEN, duration);

  cli_render_logf("[Status] Applied Frozen (duration=%d) to entity", duration);
}

void remove_frozen(ecs_world_t *world, ecs_entity_t entity) {
  ecs_remove(world, entity, Frozen);

  CardConditionCountdown *countdown =
      ecs_get_mut(world, entity, CardConditionCountdown);
  if (countdown) {
    countdown->frozen_duration = 0;
    ecs_modified(world, entity, CardConditionCountdown);
  }

  // Log status effect expired
  azk_log_status_effect_expired(world, entity, GLOG_STATUS_FROZEN);

  cli_render_logf("[Status] Removed Frozen from entity");
}

bool is_frozen(ecs_world_t *world, ecs_entity_t entity) {
  return ecs_has(world, entity, Frozen);
}

void apply_shocked(ecs_world_t *world, ecs_entity_t entity, int8_t duration) {
  ecs_add(world, entity, Shocked);

  CardConditionCountdown *countdown = ensure_condition_countdown(world, entity);
  countdown->shocked_duration = duration;
  ecs_modified(world, entity, CardConditionCountdown);

  azk_log_status_effect_applied(world, entity, GLOG_STATUS_SHOCKED, duration);
  cli_render_logf("[Status] Applied Shocked (duration=%d) to entity",
                  duration);
}

void remove_shocked(ecs_world_t *world, ecs_entity_t entity) {
  ecs_remove(world, entity, Shocked);

  CardConditionCountdown *countdown =
      ecs_get_mut(world, entity, CardConditionCountdown);
  if (countdown) {
    countdown->shocked_duration = 0;
    ecs_modified(world, entity, CardConditionCountdown);
  }

  azk_log_status_effect_expired(world, entity, GLOG_STATUS_SHOCKED);
  cli_render_logf("[Status] Removed Shocked from entity");
}

bool is_shocked(ecs_world_t *world, ecs_entity_t entity) {
  return ecs_has(world, entity, Shocked);
}

void apply_effect_immune(ecs_world_t *world, ecs_entity_t entity,
                         int8_t duration) {
  // Add the EffectImmune tag
  ecs_add(world, entity, EffectImmune);

  // Set or update CardConditionCountdown component
  CardConditionCountdown *countdown = ensure_condition_countdown(world, entity);
  // Don't override permanent immunity (-1) with temporary
  if (countdown->effect_immune_duration != -1) {
    countdown->effect_immune_duration = duration;
  }
  ecs_modified(world, entity, CardConditionCountdown);

  // Log status effect applied
  azk_log_status_effect_applied(world, entity, GLOG_STATUS_EFFECT_IMMUNE,
                                duration);

  cli_render_logf("[Status] Applied EffectImmune (duration=%d) to entity",
                  duration);
}

void remove_effect_immune(ecs_world_t *world, ecs_entity_t entity) {
  ecs_remove(world, entity, EffectImmune);

  CardConditionCountdown *countdown =
      ecs_get_mut(world, entity, CardConditionCountdown);
  if (countdown) {
    countdown->effect_immune_duration = 0;
    ecs_modified(world, entity, CardConditionCountdown);
  }

  // Log status effect expired
  azk_log_status_effect_expired(world, entity, GLOG_STATUS_EFFECT_IMMUNE);

  cli_render_logf("[Status] Removed EffectImmune from entity");
}

bool is_effect_immune(ecs_world_t *world, ecs_entity_t entity) {
  return ecs_has(world, entity, EffectImmune);
}

int8_t get_total_carapace_value(ecs_world_t *world, ecs_entity_t entity) {
  const CarapaceValue *value = ecs_get(world, entity, CarapaceValue);
  int16_t total = value ? value->amount : 0;

  const ecs_type_t *type = ecs_get_type(world, entity);
  if (!type) {
    return (int8_t)total;
  }

  for (int i = 0; i < type->count; i++) {
    ecs_id_t id = type->array[i];
    if (!ECS_IS_PAIR(id)) {
      continue;
    }

    if (ecs_pair_first(world, id) != ecs_id(CarapaceBuff)) {
      continue;
    }

    const CarapaceBuff *buff =
        (const CarapaceBuff *)ecs_get_id(world, entity, id);
    if (buff) {
      total += buff->amount;
    }
  }

  if (total < 0) {
    total = 0;
  }

  return (int8_t)total;
}

void apply_carapace_modifier(ecs_world_t *world, ecs_entity_t entity,
                             ecs_entity_t source, int8_t amount,
                             bool expires_eot) {
  if (amount == 0) {
    return;
  }

  ecs_set_pair(world, entity, CarapaceBuff, source, {
    .amount = amount,
    .expires_eot = expires_eot,
  });

  cli_render_logf("[Status] Applied carapace modifier %+d from source "
                  "(expires_eot=%d)",
                  amount, expires_eot);
}

void remove_all_carapace_modifiers(ecs_world_t *world, ecs_entity_t entity) {
  ecs_entity_t sources[32];
  int source_count = 0;

  const ecs_type_t *type = ecs_get_type(world, entity);
  if (!type) {
    return;
  }

  for (int i = 0; i < type->count && source_count < 32; i++) {
    ecs_id_t id = type->array[i];
    if (!ECS_IS_PAIR(id)) {
      continue;
    }

    if (ecs_pair_first(world, id) == ecs_id(CarapaceBuff)) {
      sources[source_count++] = ecs_pair_second(world, id);
    }
  }

  for (int i = 0; i < source_count; i++) {
    ecs_remove_pair(world, entity, ecs_id(CarapaceBuff), sources[i]);
  }

  if (source_count > 0) {
    cli_render_logf("[Status] Removed all carapace modifiers (%d sources)",
                    source_count);
  }
}

bool apply_timed_tag_grant(ecs_world_t *world, ecs_entity_t entity, ecs_id_t tag,
                           TagGrantTickPhase tick_phase,
                           int8_t remaining_ticks) {
  if (entity == 0 || tag == 0) {
    return false;
  }

  bool valid_permanent =
      tick_phase == TAG_GRANT_TICK_NONE && remaining_ticks == -1;
  bool valid_timed =
      tick_phase != TAG_GRANT_TICK_NONE && remaining_ticks > 0;
  if (!valid_permanent && !valid_timed) {
    cli_render_logf("[Status] Invalid timed tag grant request");
    return false;
  }

  CardConditionCountdown *countdown = ensure_condition_countdown(world, entity);
  for (uint8_t i = 0; i < countdown->timed_tag_grant_count; i++) {
    TimedTagGrant *grant = &countdown->timed_tag_grants[i];
    if (grant->tag == tag && grant->tick_phase == tick_phase &&
        grant->remaining_ticks == remaining_ticks) {
      return true;
    }
  }

  if (countdown->timed_tag_grant_count >= MAX_TIMED_TAG_GRANTS) {
    cli_render_logf("[Status] Timed tag grant buffer full");
    return false;
  }

  bool already_had_tag = ecs_has_id(world, entity, tag);
  countdown->timed_tag_grants[countdown->timed_tag_grant_count++] =
      (TimedTagGrant){
          .tag = tag,
          .remaining_ticks = remaining_ticks,
          .tick_phase = (uint8_t)tick_phase,
      };
  ecs_modified(world, entity, CardConditionCountdown);

  if (!already_had_tag) {
    ecs_add_id(world, entity, tag);
    GameLogStatusEffect effect;
    if (try_get_status_effect_for_tag(tag, &effect)) {
      azk_log_status_effect_applied(world, entity, effect, remaining_ticks);
    }
    log_tag_change_if_keyword(world, entity, tag, true);
  }

  return true;
}

void apply_charge_grant(ecs_world_t *world, ecs_entity_t entity,
                        TagGrantTickPhase tick_phase, int8_t remaining_ticks) {
  if (entity == 0) {
    return;
  }

  bool has_charge = ecs_has(world, entity, Charge);
  bool grant_applied =
      apply_timed_tag_grant(world, entity, ecs_id(Charge), tick_phase,
                            remaining_ticks);
  if (!grant_applied && !has_charge) {
    return;
  }

  const TapState *tap = ecs_get(world, entity, TapState);
  if (tap && tap->cooldown) {
    ecs_set(world, entity, TapState,
            {.tapped = tap->tapped, .cooldown = false});
    azk_log_card_tap_state_changed(
        world, entity,
        tap->tapped ? GLOG_TAP_TAPPED : GLOG_TAP_UNTAPPED);
  }

  cli_render_logf("[Status] Applied Charge grant");
}

void clear_card_temporary_state(ecs_world_t *world, ecs_entity_t entity) {
  if (entity == 0) {
    return;
  }

  bool has_prefab_effect_immune =
      prefab_has_tag(world, entity, ecs_id(EffectImmune));
  if (ecs_has(world, entity, Frozen)) {
    ecs_remove(world, entity, Frozen);
  }
  if (ecs_has(world, entity, Shocked)) {
    ecs_remove(world, entity, Shocked);
  }
  if (!has_prefab_effect_immune && ecs_has(world, entity, EffectImmune)) {
    ecs_remove(world, entity, EffectImmune);
  }

  remove_all_combat_damage_modifiers(world, entity);
  remove_all_carapace_modifiers(world, entity);

  if (ecs_has(world, entity, CardConditionCountdown)) {
    CardConditionCountdown *countdown =
        ecs_get_mut(world, entity, CardConditionCountdown);
    ecs_id_t tags_to_recheck[MAX_TIMED_TAG_GRANTS] = {0};
    uint8_t tags_to_recheck_count = 0;

    for (uint8_t i = 0; i < countdown->timed_tag_grant_count; i++) {
      ecs_id_t tag = countdown->timed_tag_grants[i].tag;
      bool seen = false;
      for (uint8_t j = 0; j < tags_to_recheck_count; j++) {
        if (tags_to_recheck[j] == tag) {
          seen = true;
          break;
        }
      }
      if (!seen && tags_to_recheck_count < MAX_TIMED_TAG_GRANTS) {
        tags_to_recheck[tags_to_recheck_count++] = tag;
      }
    }

    countdown->frozen_duration = 0;
    countdown->shocked_duration = 0;
    countdown->effect_immune_duration = has_prefab_effect_immune ? -1 : 0;
    countdown->timed_tag_grant_count = 0;
    for (uint8_t i = 0; i < MAX_TIMED_TAG_GRANTS; i++) {
      countdown->timed_tag_grants[i] = (TimedTagGrant){0};
    }
    ecs_modified(world, entity, CardConditionCountdown);

    for (uint8_t i = 0; i < tags_to_recheck_count; i++) {
      maybe_remove_unbacked_tag(world, entity, tags_to_recheck[i]);
    }
  }
}

static void tick_timed_tag_grants(ecs_world_t *world, ecs_entity_t card,
                                  CardConditionCountdown *countdown,
                                  TagGrantTickPhase tick_phase) {
  for (uint8_t i = 0; i < countdown->timed_tag_grant_count;) {
    TimedTagGrant grant = countdown->timed_tag_grants[i];
    if (grant.tick_phase != (uint8_t)tick_phase || grant.remaining_ticks <= 0) {
      i++;
      continue;
    }

    countdown->timed_tag_grants[i].remaining_ticks--;
    if (countdown->timed_tag_grants[i].remaining_ticks > 0) {
      i++;
      continue;
    }

    remove_tag_grant_at(countdown, i);
    maybe_remove_unbacked_tag(world, card, grant.tag);
  }
}

// Helper to process a single zone's cards for status tick-down
static void tick_zone_status_effects(ecs_world_t *world, ecs_entity_t zone,
                                     TagGrantTickPhase tick_phase) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];

    if (!ecs_has(world, card, CardConditionCountdown)) {
      continue;
    }

    CardConditionCountdown *countdown =
        ecs_get_mut(world, card, CardConditionCountdown);

    if (tick_phase == TAG_GRANT_TICK_START_OF_TURN) {
      // Process Frozen duration
      if (countdown->frozen_duration > 0) {
        countdown->frozen_duration--;
        if (countdown->frozen_duration == 0) {
          ecs_remove(world, card, Frozen);
          azk_log_status_effect_expired(world, card, GLOG_STATUS_FROZEN);
          cli_render_logf("[Status] Frozen expired on entity");
        }
      }

      // Process EffectImmune duration
      if (countdown->effect_immune_duration > 0) {
        countdown->effect_immune_duration--;
        if (countdown->effect_immune_duration == 0) {
          ecs_remove(world, card, EffectImmune);
          azk_log_status_effect_expired(world, card, GLOG_STATUS_EFFECT_IMMUNE);
          cli_render_logf("[Status] EffectImmune expired on entity");
        }
      }
    }

    tick_timed_tag_grants(world, card, countdown, tick_phase);
    ecs_modified(world, card, CardConditionCountdown);
  }
}

void tick_status_effects_for_player(ecs_world_t *world, uint8_t player_index) {
  const GameState *gs = ecs_singleton_get(world, GameState);

  // Tick status effects on garden entities
  tick_zone_status_effects(world, gs->zones[player_index].garden,
                           TAG_GRANT_TICK_START_OF_TURN);

  // Tick status effects on alley entities
  tick_zone_status_effects(world, gs->zones[player_index].alley,
                           TAG_GRANT_TICK_START_OF_TURN);

  // Tick status effects on leader
  tick_zone_status_effects(world, gs->zones[player_index].leader,
                           TAG_GRANT_TICK_START_OF_TURN);
}

void tick_end_of_turn_effects_for_player(ecs_world_t *world,
                                         uint8_t player_index) {
  const GameState *gs = ecs_singleton_get(world, GameState);

  tick_zone_status_effects(world, gs->zones[player_index].garden,
                           TAG_GRANT_TICK_END_OF_TURN);
  tick_zone_status_effects(world, gs->zones[player_index].alley,
                           TAG_GRANT_TICK_END_OF_TURN);
  tick_zone_status_effects(world, gs->zones[player_index].leader,
                           TAG_GRANT_TICK_END_OF_TURN);
}

// Helper to iterate AttackBuff pairs on an entity and sum modifiers
static int16_t sum_attack_buff_modifiers(ecs_world_t *world, ecs_entity_t entity) {
  int16_t total = 0;
  const ecs_type_t *type = ecs_get_type(world, entity);
  if (!type) {
    return 0;
  }

  for (int i = 0; i < type->count; i++) {
    ecs_id_t id = type->array[i];
    if (ECS_IS_PAIR(id)) {
      ecs_entity_t first = ecs_pair_first(world, id);
      if (first == ecs_id(AttackBuff)) {
        // Use ecs_get_id to get the component data for this specific pair
        const AttackBuff *buff = (const AttackBuff *)ecs_get_id(world, entity, id);
        if (buff) {
          total += buff->modifier;
        }
      }
    }
  }
  return total;
}

void recalculate_attack_from_buffs(ecs_world_t *world, ecs_entity_t entity) {
  const BaseStats *base = ecs_get(world, entity, BaseStats);
  if (!base) {
    return;
  }

  int16_t total_attack = base->attack;

  // Add weapon attack from attached weapon children
  ecs_iter_t weapon_it = ecs_children(world, entity);
  while (ecs_children_next(&weapon_it)) {
    for (int i = 0; i < weapon_it.count; i++) {
      ecs_entity_t child = weapon_it.entities[i];
      if (ecs_has_id(world, child, TWeapon)) {
        const CurStats *weapon_stats = ecs_get(world, child, CurStats);
        if (weapon_stats) {
          total_attack += weapon_stats->cur_atk;
        }
      }
    }
  }

  // Sum all AttackBuff modifiers from relationship pairs
  total_attack += sum_attack_buff_modifiers(world, entity);

  // Clamp to minimum of 0
  if (total_attack < 0) {
    total_attack = 0;
  }

  // Update CurStats
  const CurStats *cur_stats = ecs_get(world, entity, CurStats);
  if (cur_stats) {
    ecs_set(world, entity, CurStats, {
      .cur_atk = (int8_t)total_attack,
      .cur_hp = cur_stats->cur_hp,
    });
  }
}

void apply_attack_modifier(ecs_world_t *world, ecs_entity_t entity,
                           ecs_entity_t source, int8_t modifier, bool expires_eot) {
  const CurStats *cur = ecs_get(world, entity, CurStats);
  if (!cur) {
    cli_render_logf("[Status] Warning: Cannot apply attack modifier - entity has no CurStats");
    return;
  }

  // Calculate the actual modifier after clamping to 0
  int16_t new_atk = cur->cur_atk + modifier;
  int8_t actual_modifier = modifier;
  if (new_atk < 0) {
    // Can only reduce by cur_atk amount to reach 0
    actual_modifier = -(cur->cur_atk);
    new_atk = 0;
  }

  // Store the actual applied modifier in the pair (for correct reversal later)
  ecs_set_pair(world, entity, AttackBuff, source, {
    .modifier = actual_modifier,
    .expires_eot = expires_eot,
  });

  // Apply to CurStats
  ecs_set(world, entity, CurStats, {
    .cur_atk = (int8_t)new_atk,
    .cur_hp = cur->cur_hp,
  });

  // Log stat change
  azk_log_card_stat_change(world, entity, actual_modifier, 0, (int8_t)new_atk, cur->cur_hp);

  cli_render_logf("[Status] Applied attack modifier %+d (requested %+d) from source (expires_eot=%d)",
                  actual_modifier, modifier, expires_eot);
}

void remove_attack_modifier(ecs_world_t *world, ecs_entity_t entity,
                            ecs_entity_t source) {
  // Get the modifier value before removing so we can adjust CurStats
  ecs_id_t pair_id = ecs_pair(ecs_id(AttackBuff), source);
  const AttackBuff *buff = (const AttackBuff *)ecs_get_id(world, entity, pair_id);
  if (!buff) {
    return;
  }
  int8_t modifier = buff->modifier;

  // Remove the (AttackBuff, source) pair
  ecs_remove_pair(world, entity, ecs_id(AttackBuff), source);

  // Directly remove modifier from CurStats since pair removal won't be visible until deferred flush
  const CurStats *cur = ecs_get(world, entity, CurStats);
  const BaseStats *base = ecs_get(world, entity, BaseStats);
  if (cur && base) {
    int16_t new_atk = cur->cur_atk - modifier;
    // Clamp to minimum of 0
    if (new_atk < 0) new_atk = 0;
    ecs_set(world, entity, CurStats, {
      .cur_atk = (int8_t)new_atk,
      .cur_hp = cur->cur_hp,
    });

    // Log stat change (negative since we're removing)
    azk_log_card_stat_change(world, entity, -modifier, 0, (int8_t)new_atk, cur->cur_hp);
  }

  cli_render_logf("[Status] Removed attack modifier from source");
}

void remove_all_attack_modifiers(ecs_world_t *world, ecs_entity_t entity) {
  // Collect all sources and their modifiers first to avoid iterator invalidation
  ecs_entity_t sources[32];
  int8_t modifiers[32];
  int source_count = 0;

  const ecs_type_t *type = ecs_get_type(world, entity);
  if (type) {
    for (int i = 0; i < type->count && source_count < 32; i++) {
      ecs_id_t id = type->array[i];
      if (ECS_IS_PAIR(id)) {
        ecs_entity_t first = ecs_pair_first(world, id);
        if (first == ecs_id(AttackBuff)) {
          const AttackBuff *buff = (const AttackBuff *)ecs_get_id(world, entity, id);
          if (buff) {
            sources[source_count] = ecs_pair_second(world, id);
            modifiers[source_count] = buff->modifier;
            source_count++;
          }
        }
      }
    }
  }

  if (source_count > 0) {
    // Calculate total modifier to remove
    int16_t total_modifier = 0;
    for (int i = 0; i < source_count; i++) {
      total_modifier += modifiers[i];
      ecs_remove_pair(world, entity, ecs_id(AttackBuff), sources[i]);
    }

    // Directly adjust CurStats
    const CurStats *cur = ecs_get(world, entity, CurStats);
    if (cur) {
      int16_t new_atk = cur->cur_atk - total_modifier;
      if (new_atk < 0) new_atk = 0;
      ecs_set(world, entity, CurStats, {
        .cur_atk = (int8_t)new_atk,
        .cur_hp = cur->cur_hp,
      });
    }

    cli_render_logf("[Status] Removed all attack modifiers (%d sources)", source_count);
  }
}

void expire_eot_attack_modifiers_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];

    // Collect EOT buff sources and modifiers to remove
    ecs_entity_t sources_to_remove[32];
    int8_t modifiers_to_remove[32];
    int remove_count = 0;

    const ecs_type_t *type = ecs_get_type(world, card);
    if (type) {
      for (int j = 0; j < type->count && remove_count < 32; j++) {
        ecs_id_t id = type->array[j];
        if (ECS_IS_PAIR(id)) {
          ecs_entity_t first = ecs_pair_first(world, id);
          if (first == ecs_id(AttackBuff)) {
            ecs_entity_t source = ecs_pair_second(world, id);
            const AttackBuff *buff = (const AttackBuff *)ecs_get_id(world, card, id);
            if (buff && buff->expires_eot) {
              sources_to_remove[remove_count] = source;
              modifiers_to_remove[remove_count] = buff->modifier;
              remove_count++;
              cli_render_logf("[Status] EOT: Expiring attack modifier %+d from entity",
                              buff->modifier);
            }
          }
        }
      }
    }

    if (remove_count > 0) {
      // Calculate total modifier to remove
      int16_t total_modifier = 0;
      for (int j = 0; j < remove_count; j++) {
        total_modifier += modifiers_to_remove[j];
        ecs_remove_pair(world, card, ecs_id(AttackBuff), sources_to_remove[j]);
      }

      // Directly adjust CurStats
      const CurStats *cur = ecs_get(world, card, CurStats);
      if (cur) {
        int16_t new_atk = cur->cur_atk - total_modifier;
        if (new_atk < 0) new_atk = 0;
        ecs_set(world, card, CurStats, {
          .cur_atk = (int8_t)new_atk,
          .cur_hp = cur->cur_hp,
        });
        // Log stat change so clients update attack after EOT modifier expiry
        int16_t atk_delta = -total_modifier;
        int8_t atk_delta8 = (int8_t)atk_delta;
        if (atk_delta > 127) {
          atk_delta8 = 127;
        } else if (atk_delta < -128) {
          atk_delta8 = -128;
        }
        azk_log_card_stat_change(world, card, atk_delta8, 0,
                                 (int8_t)new_atk, cur->cur_hp);
      }
    }
  }
}

// Helper to iterate HealthBuff pairs on an entity and sum modifiers
static int16_t sum_health_buff_modifiers(ecs_world_t *world, ecs_entity_t entity) {
  int16_t total = 0;
  const ecs_type_t *type = ecs_get_type(world, entity);
  if (!type) {
    return 0;
  }

  for (int i = 0; i < type->count; i++) {
    ecs_id_t id = type->array[i];
    if (ECS_IS_PAIR(id)) {
      ecs_entity_t first = ecs_pair_first(world, id);
      if (first == ecs_id(HealthBuff)) {
        const HealthBuff *buff = (const HealthBuff *)ecs_get_id(world, entity, id);
        if (buff) {
          total += buff->modifier;
        }
      }
    }
  }
  return total;
}

void recalculate_health_from_buffs(ecs_world_t *world, ecs_entity_t entity) {
  const BaseStats *base = ecs_get(world, entity, BaseStats);
  if (!base) {
    return;
  }

  int16_t total_health = base->health;

  // Sum all HealthBuff modifiers from relationship pairs
  total_health += sum_health_buff_modifiers(world, entity);

  // Note: Unlike attack, health can go to 0 (death)
  // but we don't clamp here - death is handled elsewhere

  // Update CurStats
  const CurStats *cur_stats = ecs_get(world, entity, CurStats);
  if (cur_stats) {
    ecs_set(world, entity, CurStats, {
      .cur_atk = cur_stats->cur_atk,
      .cur_hp = (int8_t)total_health,
    });
  }
}

void apply_health_modifier(ecs_world_t *world, ecs_entity_t entity,
                           ecs_entity_t source, int8_t modifier, bool expires_eot) {
  const CurStats *cur = ecs_get(world, entity, CurStats);
  if (!cur) {
    cli_render_logf("[Status] Warning: Cannot apply health modifier - entity has no CurStats");
    return;
  }

  // Calculate new health (no clamping - health buffs can increase above base)
  int16_t new_hp = cur->cur_hp + modifier;

  // Store the modifier in the pair
  ecs_set_pair(world, entity, HealthBuff, source, {
    .modifier = modifier,
    .expires_eot = expires_eot,
  });

  // Apply to CurStats
  ecs_set(world, entity, CurStats, {
    .cur_atk = cur->cur_atk,
    .cur_hp = (int8_t)new_hp,
  });

  // Log stat change
  azk_log_card_stat_change(world, entity, 0, modifier, cur->cur_atk, (int8_t)new_hp);

  cli_render_logf("[Status] Applied health modifier %+d from source (expires_eot=%d)",
                  modifier, expires_eot);
}

bool remove_health_modifier(ecs_world_t *world, ecs_entity_t entity,
                            ecs_entity_t source) {
  // Get the modifier value before removing so we can adjust CurStats
  ecs_id_t pair_id = ecs_pair(ecs_id(HealthBuff), source);
  const HealthBuff *buff = (const HealthBuff *)ecs_get_id(world, entity, pair_id);
  if (!buff) {
    return false;
  }
  int8_t modifier = buff->modifier;

  // Remove the (HealthBuff, source) pair
  ecs_remove_pair(world, entity, ecs_id(HealthBuff), source);

  // Directly remove modifier from CurStats
  const CurStats *cur = ecs_get(world, entity, CurStats);
  if (cur) {
    int16_t new_hp = cur->cur_hp - modifier;
    ecs_set(world, entity, CurStats, {
      .cur_atk = cur->cur_atk,
      .cur_hp = (int8_t)new_hp,
    });

    // Log stat change (negative since we're removing)
    azk_log_card_stat_change(world, entity, 0, -modifier, cur->cur_atk, (int8_t)new_hp);

    cli_render_logf("[Status] Removed health modifier %+d from source (new_hp=%d)",
                    modifier, (int)new_hp);

    // Return true if entity should die
    return new_hp <= 0;
  }

  return false;
}

bool remove_all_health_modifiers(ecs_world_t *world, ecs_entity_t entity) {
  // Collect all sources and their modifiers first to avoid iterator invalidation
  ecs_entity_t sources[32];
  int8_t modifiers[32];
  int source_count = 0;

  const ecs_type_t *type = ecs_get_type(world, entity);
  if (type) {
    for (int i = 0; i < type->count && source_count < 32; i++) {
      ecs_id_t id = type->array[i];
      if (ECS_IS_PAIR(id)) {
        ecs_entity_t first = ecs_pair_first(world, id);
        if (first == ecs_id(HealthBuff)) {
          const HealthBuff *buff = (const HealthBuff *)ecs_get_id(world, entity, id);
          if (buff) {
            sources[source_count] = ecs_pair_second(world, id);
            modifiers[source_count] = buff->modifier;
            source_count++;
          }
        }
      }
    }
  }

  if (source_count > 0) {
    // Calculate total modifier to remove
    int16_t total_modifier = 0;
    for (int i = 0; i < source_count; i++) {
      total_modifier += modifiers[i];
      ecs_remove_pair(world, entity, ecs_id(HealthBuff), sources[i]);
    }

    // Directly adjust CurStats
    const CurStats *cur = ecs_get(world, entity, CurStats);
    if (cur) {
      int16_t new_hp = cur->cur_hp - total_modifier;
      ecs_set(world, entity, CurStats, {
        .cur_atk = cur->cur_atk,
        .cur_hp = (int8_t)new_hp,
      });

      cli_render_logf("[Status] Removed all health modifiers (%d sources, new_hp=%d)",
                      source_count, (int)new_hp);

      // Return true if entity should die
      return new_hp <= 0;
    }
  }

  return false;
}

void expire_eot_health_modifiers_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];

    // Collect EOT buff sources and modifiers to remove
    ecs_entity_t sources_to_remove[32];
    int8_t modifiers_to_remove[32];
    int remove_count = 0;

    const ecs_type_t *type = ecs_get_type(world, card);
    if (type) {
      for (int j = 0; j < type->count && remove_count < 32; j++) {
        ecs_id_t id = type->array[j];
        if (ECS_IS_PAIR(id)) {
          ecs_entity_t first = ecs_pair_first(world, id);
          if (first == ecs_id(HealthBuff)) {
            ecs_entity_t source = ecs_pair_second(world, id);
            const HealthBuff *buff = (const HealthBuff *)ecs_get_id(world, card, id);
            if (buff && buff->expires_eot) {
              sources_to_remove[remove_count] = source;
              modifiers_to_remove[remove_count] = buff->modifier;
              remove_count++;
              cli_render_logf("[Status] EOT: Expiring health modifier %+d from entity",
                              buff->modifier);
            }
          }
        }
      }
    }

    if (remove_count > 0) {
      // Calculate total modifier to remove
      int16_t total_modifier = 0;
      for (int j = 0; j < remove_count; j++) {
        total_modifier += modifiers_to_remove[j];
        ecs_remove_pair(world, card, ecs_id(HealthBuff), sources_to_remove[j]);
      }

      // Directly adjust CurStats
      const CurStats *cur = ecs_get(world, card, CurStats);
      if (cur) {
        int16_t new_hp = cur->cur_hp - total_modifier;
        ecs_set(world, card, CurStats, {
          .cur_atk = cur->cur_atk,
          .cur_hp = (int8_t)new_hp,
        });
        azk_log_card_stat_change(world, card, 0, -total_modifier, cur->cur_atk,
                                 (int8_t)new_hp);

        if (new_hp <= 0) {
          if (ecs_has(world, card, TLeader)) {
            GameState *gs = ecs_singleton_get_mut(world, GameState);
            ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
            for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
              if (parent == gs->zones[p].leader) {
                gs->winner = (p + 1) % MAX_PLAYERS_PER_MATCH;
                ecs_singleton_modified(world, GameState);
                azk_log_entity_died(world, card, GLOG_DEATH_EFFECT);
                azk_log_game_ended(world, gs->winner, GLOG_END_LEADER_DEFEATED);
                break;
              }
            }
          } else {
            azk_log_entity_died(world, card, GLOG_DEATH_EFFECT);
            discard_card(world, card);
          }
        }
      }
    }
  }
}

void expire_eot_carapace_modifiers_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    ecs_entity_t sources_to_remove[32];
    int remove_count = 0;

    const ecs_type_t *type = ecs_get_type(world, card);
    if (!type) {
      continue;
    }

    for (int j = 0; j < type->count && remove_count < 32; j++) {
      ecs_id_t id = type->array[j];
      if (!ECS_IS_PAIR(id)) {
        continue;
      }

      if (ecs_pair_first(world, id) != ecs_id(CarapaceBuff)) {
        continue;
      }

      const CarapaceBuff *buff =
          (const CarapaceBuff *)ecs_get_id(world, card, id);
      if (buff && buff->expires_eot) {
        sources_to_remove[remove_count++] = ecs_pair_second(world, id);
      }
    }

    for (int j = 0; j < remove_count; j++) {
      ecs_remove_pair(world, card, ecs_id(CarapaceBuff), sources_to_remove[j]);
    }

    if (remove_count > 0) {
      cli_render_logf("[Status] EOT: Expired %d carapace modifiers",
                      remove_count);
    }
  }
}

static int16_t sum_incoming_combat_damage_modifiers(ecs_world_t *world,
                                                    ecs_entity_t entity) {
  int16_t total = 0;
  const ecs_type_t *type = ecs_get_type(world, entity);
  if (!type) {
    return 0;
  }

  for (int i = 0; i < type->count; i++) {
    ecs_id_t id = type->array[i];
    if (!ECS_IS_PAIR(id)) {
      continue;
    }

    ecs_entity_t first = ecs_pair_first(world, id);
    if (first != ecs_id(CombatDamageModifier)) {
      continue;
    }

    const CombatDamageModifier *modifier =
        (const CombatDamageModifier *)ecs_get_id(world, entity, id);
    if (modifier) {
      total += modifier->incoming_modifier;
    }
  }

  return total;
}

static int16_t sum_outgoing_combat_damage_modifiers(ecs_world_t *world,
                                                    ecs_entity_t entity) {
  int16_t total = 0;
  const ecs_type_t *type = ecs_get_type(world, entity);
  if (!type) {
    return 0;
  }

  for (int i = 0; i < type->count; i++) {
    ecs_id_t id = type->array[i];
    if (!ECS_IS_PAIR(id)) {
      continue;
    }

    ecs_entity_t first = ecs_pair_first(world, id);
    if (first != ecs_id(CombatDamageModifier)) {
      continue;
    }

    const CombatDamageModifier *modifier =
        (const CombatDamageModifier *)ecs_get_id(world, entity, id);
    if (modifier) {
      total += modifier->outgoing_modifier;
    }
  }

  return total;
}

int16_t get_total_incoming_combat_damage_modifier(ecs_world_t *world,
                                                  ecs_entity_t entity) {
  return sum_incoming_combat_damage_modifiers(world, entity);
}

int16_t get_total_outgoing_combat_damage_modifier(ecs_world_t *world,
                                                  ecs_entity_t entity) {
  return sum_outgoing_combat_damage_modifiers(world, entity);
}

void apply_combat_damage_modifier(ecs_world_t *world, ecs_entity_t entity,
                                  ecs_entity_t source,
                                  int8_t incoming_modifier,
                                  int8_t outgoing_modifier,
                                  bool expires_eot) {
  ecs_set_pair(world, entity, CombatDamageModifier, source, {
      .incoming_modifier = incoming_modifier,
      .outgoing_modifier = outgoing_modifier,
      .expires_eot = expires_eot,
  });

  cli_render_logf(
      "[Status] Applied combat damage modifier (in=%+d, out=%+d, expires_eot=%d)",
      incoming_modifier, outgoing_modifier, expires_eot);
}

bool remove_combat_damage_modifier(ecs_world_t *world, ecs_entity_t entity,
                                   ecs_entity_t source) {
  ecs_id_t pair_id = ecs_pair(ecs_id(CombatDamageModifier), source);
  const CombatDamageModifier *modifier =
      (const CombatDamageModifier *)ecs_get_id(world, entity, pair_id);
  if (!modifier) {
    return false;
  }

  ecs_remove_pair(world, entity, ecs_id(CombatDamageModifier), source);
  cli_render_logf("[Status] Removed combat damage modifier from source");
  return true;
}

void remove_all_combat_damage_modifiers(ecs_world_t *world,
                                        ecs_entity_t entity) {
  ecs_entity_t sources[32];
  int source_count = 0;

  const ecs_type_t *type = ecs_get_type(world, entity);
  if (!type) {
    return;
  }

  for (int i = 0; i < type->count && source_count < 32; i++) {
    ecs_id_t id = type->array[i];
    if (!ECS_IS_PAIR(id)) {
      continue;
    }

    ecs_entity_t first = ecs_pair_first(world, id);
    if (first == ecs_id(CombatDamageModifier)) {
      sources[source_count++] = ecs_pair_second(world, id);
    }
  }

  for (int i = 0; i < source_count; i++) {
    ecs_remove_pair(world, entity, ecs_id(CombatDamageModifier), sources[i]);
  }

  if (source_count > 0) {
    cli_render_logf("[Status] Removed all combat damage modifiers (%d sources)",
                    source_count);
  }
}

void expire_eot_combat_damage_modifiers_in_zone(ecs_world_t *world,
                                                ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    ecs_entity_t sources_to_remove[32];
    int remove_count = 0;

    const ecs_type_t *type = ecs_get_type(world, card);
    if (!type) {
      continue;
    }

    for (int j = 0; j < type->count && remove_count < 32; j++) {
      ecs_id_t id = type->array[j];
      if (!ECS_IS_PAIR(id)) {
        continue;
      }

      ecs_entity_t first = ecs_pair_first(world, id);
      if (first != ecs_id(CombatDamageModifier)) {
        continue;
      }

      const CombatDamageModifier *modifier =
          (const CombatDamageModifier *)ecs_get_id(world, card, id);
      if (modifier && modifier->expires_eot) {
        sources_to_remove[remove_count++] = ecs_pair_second(world, id);
      }
    }

    for (int j = 0; j < remove_count; j++) {
      ecs_remove_pair(world, card, ecs_id(CombatDamageModifier),
                      sources_to_remove[j]);
    }

    if (remove_count > 0) {
      cli_render_logf("[Status] EOT: Expired %d combat damage modifiers",
                      remove_count);
    }
  }
}

void azk_queue_passive_buff_update(ecs_world_t *world, ecs_entity_t entity,
                                   ecs_entity_t source, int8_t atk_modifier,
                                   int8_t hp_modifier, bool is_removal) {
  PassiveBuffQueue *queue = ecs_singleton_get_mut(world, PassiveBuffQueue);

  if (queue->count >= MAX_PASSIVE_BUFF_QUEUE) {
    cli_render_logf("[Status] Passive buff queue full, cannot queue");
    return;
  }

  queue->buffs[queue->count].entity = entity;
  queue->buffs[queue->count].source = source;
  queue->buffs[queue->count].atk_modifier = atk_modifier;
  queue->buffs[queue->count].hp_modifier = hp_modifier;
  queue->buffs[queue->count].is_removal = is_removal;
  queue->count++;

  cli_render_logf("[Status] Queued passive buff (atk=%+d, hp=%+d, removal=%d, count=%d)",
                  atk_modifier, hp_modifier, is_removal, queue->count);
  ecs_singleton_modified(world, PassiveBuffQueue);
}

bool azk_has_pending_passive_buffs(ecs_world_t *world) {
  const PassiveBuffQueue *queue = ecs_singleton_get(world, PassiveBuffQueue);
  return queue && queue->count > 0;
}

void azk_process_passive_buff_queue(ecs_world_t *world) {
  PassiveBuffQueue *queue = ecs_singleton_get_mut(world, PassiveBuffQueue);

  if (!queue || queue->count == 0) {
    return;
  }

  cli_render_logf("[Status] Processing %d passive buff updates", queue->count);

  for (uint8_t i = 0; i < queue->count; i++) {
    PendingPassiveBuff *buff = &queue->buffs[i];

    // Check if entity is still valid
    if (!ecs_is_valid(world, buff->entity)) {
      cli_render_logf("[Status] Skipping buff - entity no longer valid");
      continue;
    }

    // Track if we actually changed the attack buff (for weapon propagation)
    bool atk_buff_changed = false;
    int8_t actual_atk_modifier = 0;

    if (buff->is_removal) {
      // Remove attack buff if it exists
      if (ecs_has_pair(world, buff->entity, ecs_id(AttackBuff), buff->source)) {
        ecs_id_t pair_id = ecs_pair(ecs_id(AttackBuff), buff->source);
        const AttackBuff *existing = ecs_get_id(world, buff->entity, pair_id);
        if (existing) {
          actual_atk_modifier = existing->modifier;
        }
        remove_attack_modifier(world, buff->entity, buff->source);
        atk_buff_changed = true;
        cli_render_logf("[Status] Processed passive attack buff removal");
      }

      // Remove health buff if it exists
      if (ecs_has_pair(world, buff->entity, ecs_id(HealthBuff), buff->source)) {
        bool died = remove_health_modifier(world, buff->entity, buff->source);
        cli_render_logf("[Status] Processed passive health buff removal (died=%d)", died);

        // Handle death if HP dropped to 0 or below
        if (died && ecs_is_valid(world, buff->entity)) {
          if (ecs_has(world, buff->entity, TLeader)) {
            // Leader defeated - determine winner based on entity's owner
            GameState *gs = ecs_singleton_get_mut(world, GameState);
            ecs_entity_t parent = ecs_get_target(world, buff->entity, EcsChildOf, 0);
            for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
              if (parent == gs->zones[p].leader) {
                gs->winner = (p + 1) % MAX_PLAYERS_PER_MATCH;
                ecs_singleton_modified(world, GameState);
                cli_render_logf("[Status] Leader defeated by health buff removal - player %d wins", gs->winner);
                break;
              }
            }
          } else {
            // Non-leader entity - discard it
            discard_card(world, buff->entity);
            cli_render_logf("[Status] Entity defeated by health buff removal - discarded");
          }
        }
      }
    } else {
      // Apply attack buff if modifier is non-zero and buff doesn't already exist
      if (buff->atk_modifier != 0 &&
          !ecs_has_pair(world, buff->entity, ecs_id(AttackBuff), buff->source)) {
        apply_attack_modifier(world, buff->entity, buff->source, buff->atk_modifier,
                              false);
        actual_atk_modifier = buff->atk_modifier;
        atk_buff_changed = true;
        cli_render_logf("[Status] Processed passive attack buff apply (%+d)",
                        buff->atk_modifier);
      }

      // Apply health buff if modifier is non-zero and buff doesn't already exist
      if (buff->hp_modifier != 0 &&
          !ecs_has_pair(world, buff->entity, ecs_id(HealthBuff), buff->source)) {
        apply_health_modifier(world, buff->entity, buff->source, buff->hp_modifier,
                              false);
        cli_render_logf("[Status] Processed passive health buff apply (%+d)",
                        buff->hp_modifier);
      }
    }

    // If the buffed entity is a weapon and we actually changed its attack buff,
    // propagate the modifier delta to the parent entity's attack.
    // We can't use recalculate_attack_from_buffs here because ecs_children
    // may not see deferred children relationships yet.
    if (atk_buff_changed && ecs_has_id(world, buff->entity, TWeapon)) {
      ecs_entity_t parent = ecs_get_target(world, buff->entity, EcsChildOf, 0);
      if (parent && ecs_is_valid(world, parent)) {
        const CurStats *parent_cur = ecs_get(world, parent, CurStats);
        if (parent_cur) {
          // For apply: add modifier. For removal: subtract modifier.
          int16_t delta = buff->is_removal ? -actual_atk_modifier : actual_atk_modifier;
          int16_t new_atk = parent_cur->cur_atk + delta;
          if (new_atk < 0) new_atk = 0;
          ecs_set(world, parent, CurStats, {
            .cur_atk = (int8_t)new_atk,
            .cur_hp = parent_cur->cur_hp,
          });
          azk_log_card_stat_change(world, parent, (int8_t)delta, 0,
                                   (int8_t)new_atk, parent_cur->cur_hp);
          cli_render_logf("[Status] Propagated weapon buff (%+d) to parent entity attack",
                          delta);
        }
      }
    }
  }

  // Clear the queue
  queue->count = 0;
  ecs_singleton_modified(world, PassiveBuffQueue);
}
