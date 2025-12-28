#include "utils/status_util.h"

#include "components/abilities.h"
#include "components/components.h"
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
    countdown->effect_immune_duration = duration;
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
