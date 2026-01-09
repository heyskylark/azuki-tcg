#include "utils/status_util.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"

void apply_frozen(ecs_world_t *world, ecs_entity_t entity, int8_t duration) {
  // Add the Frozen tag
  ecs_add(world, entity, Frozen);

  // Set or update CardConditionCountdown component
  CardConditionCountdown *countdown =
      ecs_get_mut(world, entity, CardConditionCountdown);
  if (!countdown) {
    ecs_set(world, entity, CardConditionCountdown,
            {.frozen_duration = duration,
             .shocked_duration = 0,
             .effect_immune_duration = 0});
  } else {
    countdown->frozen_duration = duration;
    ecs_modified(world, entity, CardConditionCountdown);
  }

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

void apply_effect_immune(ecs_world_t *world, ecs_entity_t entity,
                         int8_t duration) {
  // Add the EffectImmune tag
  ecs_add(world, entity, EffectImmune);

  // Set or update CardConditionCountdown component
  CardConditionCountdown *countdown =
      ecs_get_mut(world, entity, CardConditionCountdown);
  if (!countdown) {
    ecs_set(world, entity, CardConditionCountdown,
            {.frozen_duration = 0,
             .shocked_duration = 0,
             .effect_immune_duration = duration});
  } else {
    // Don't override permanent immunity (-1) with temporary
    if (countdown->effect_immune_duration != -1) {
      countdown->effect_immune_duration = duration;
    }
    ecs_modified(world, entity, CardConditionCountdown);
  }

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

// Helper to process a single zone's cards for status tick-down
static void tick_zone_status_effects(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];

    if (!ecs_has(world, card, CardConditionCountdown)) {
      continue;
    }

    CardConditionCountdown *countdown =
        ecs_get_mut(world, card, CardConditionCountdown);

    // Process Frozen duration
    if (countdown->frozen_duration > 0) {
      countdown->frozen_duration--;
      if (countdown->frozen_duration == 0) {
        ecs_remove(world, card, Frozen);
        azk_log_status_effect_expired(world, card, GLOG_STATUS_FROZEN);
        cli_render_logf("[Status] Frozen expired on entity");
      }
    }

    // Process Shocked duration (for future use)
    if (countdown->shocked_duration > 0) {
      countdown->shocked_duration--;
      if (countdown->shocked_duration == 0) {
        ecs_remove(world, card, Shocked);
        azk_log_status_effect_expired(world, card, GLOG_STATUS_SHOCKED);
        cli_render_logf("[Status] Shocked expired on entity");
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

    ecs_modified(world, card, CardConditionCountdown);
  }
}

void tick_status_effects_for_player(ecs_world_t *world, uint8_t player_index) {
  const GameState *gs = ecs_singleton_get(world, GameState);

  // Tick status effects on garden entities
  tick_zone_status_effects(world, gs->zones[player_index].garden);

  // Tick status effects on alley entities
  tick_zone_status_effects(world, gs->zones[player_index].alley);

  // Tick status effects on leader
  tick_zone_status_effects(world, gs->zones[player_index].leader);
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
        // Note: death handling for EOT health loss should be handled separately
      }
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
