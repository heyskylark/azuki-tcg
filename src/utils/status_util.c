#include "utils/status_util.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"

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
        cli_render_logf("[Status] Frozen expired on entity");
      }
    }

    // Process Shocked duration (for future use)
    if (countdown->shocked_duration > 0) {
      countdown->shocked_duration--;
      if (countdown->shocked_duration == 0) {
        ecs_remove(world, card, Shocked);
        cli_render_logf("[Status] Shocked expired on entity");
      }
    }

    // Process EffectImmune duration
    if (countdown->effect_immune_duration > 0) {
      countdown->effect_immune_duration--;
      if (countdown->effect_immune_duration == 0) {
        ecs_remove(world, card, EffectImmune);
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

void azk_queue_passive_buff_update(ecs_world_t *world, ecs_entity_t entity,
                                   ecs_entity_t source, int8_t modifier,
                                   bool is_removal) {
  PassiveBuffQueue *queue = ecs_singleton_get_mut(world, PassiveBuffQueue);

  if (queue->count >= MAX_PASSIVE_BUFF_QUEUE) {
    cli_render_logf("[Status] Passive buff queue full, cannot queue");
    return;
  }

  queue->buffs[queue->count].entity = entity;
  queue->buffs[queue->count].source = source;
  queue->buffs[queue->count].modifier = modifier;
  queue->buffs[queue->count].is_removal = is_removal;
  queue->count++;

  cli_render_logf("[Status] Queued passive buff (removal=%d, count=%d)",
                  is_removal, queue->count);
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

    // Track if we actually changed the buff
    bool buff_changed = false;
    int8_t actual_modifier = 0;

    if (buff->is_removal) {
      // Check if buff exists before trying to remove
      if (ecs_has_pair(world, buff->entity, ecs_id(AttackBuff), buff->source)) {
        // Get the actual modifier before removal for parent update
        ecs_id_t pair_id = ecs_pair(ecs_id(AttackBuff), buff->source);
        const AttackBuff *existing = ecs_get_id(world, buff->entity, pair_id);
        if (existing) {
          actual_modifier = existing->modifier;
        }
        remove_attack_modifier(world, buff->entity, buff->source);
        buff_changed = true;
        cli_render_logf("[Status] Processed passive buff removal");
      }
    } else {
      // Check if buff already exists (avoid double-applying)
      if (!ecs_has_pair(world, buff->entity, ecs_id(AttackBuff), buff->source)) {
        apply_attack_modifier(world, buff->entity, buff->source, buff->modifier,
                              false);
        actual_modifier = buff->modifier;
        buff_changed = true;
        cli_render_logf("[Status] Processed passive buff apply (+%d)",
                        buff->modifier);
      }
    }

    // If the buffed entity is a weapon and we actually changed it,
    // propagate the modifier delta to the parent entity's attack.
    // We can't use recalculate_attack_from_buffs here because ecs_children
    // may not see deferred children relationships yet.
    if (buff_changed && ecs_has_id(world, buff->entity, TWeapon)) {
      ecs_entity_t parent = ecs_get_target(world, buff->entity, EcsChildOf, 0);
      if (parent && ecs_is_valid(world, parent)) {
        const CurStats *parent_cur = ecs_get(world, parent, CurStats);
        if (parent_cur) {
          // For apply: add modifier. For removal: subtract modifier.
          int16_t delta = buff->is_removal ? -actual_modifier : actual_modifier;
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
