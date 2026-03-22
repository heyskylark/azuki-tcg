#include "abilities/cards/azk01_017.h"

#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/zone_util.h"

static int count_hook_sword_target_classes(ecs_world_t *world) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  int max_targets = 0;

  for (int p = 0; p < MAX_PLAYERS_PER_MATCH && max_targets < 2; p++) {
    ecs_entities_t cards =
        ecs_get_ordered_children(world, gs->zones[p].garden);
    if (cards.count > 0) {
      max_targets++;
      break;
    }
  }

  ecs_entity_t leader0 = find_leader_card_in_zone(world, gs->zones[0].leader);
  ecs_entity_t leader1 = find_leader_card_in_zone(world, gs->zones[1].leader);
  if (leader0 != 0 || leader1 != 0) {
    max_targets++;
  }

  return max_targets;
}

bool azk01_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  (void)owner;
  return count_hook_sword_target_classes(world) > 0;
}

bool azk01_017_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  bool target_is_leader = ecs_has(world, target, TLeader);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  if (!target_is_leader) {
    const GameState *gs = ecs_singleton_get(world, GameState);
    bool in_garden = false;
    for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
      if (parent == gs->zones[p].garden) {
        in_garden = true;
        break;
      }
    }
    if (!in_garden) {
      return false;
    }
  }

  for (int i = 0; i < ctx->effect.selected_count; i++) {
    ecs_entity_t selected = ctx->effect.entities[i];
    if (selected == target) {
      return false;
    }
    if (selected == 0) {
      continue;
    }
    bool selected_is_leader = ecs_has(world, selected, TLeader);
    if (selected_is_leader == target_is_leader) {
      return false;
    }
  }

  return true;
}

void azk01_017_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  int max_targets = count_hook_sword_target_classes(world);
  if (max_targets <= 0) {
    return;
  }

  ctx->effect.min_required = 0;
  ctx->effect.max_allowed = (uint8_t)max_targets;
  ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
}

void azk01_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  for (int i = 0; i < ctx->effect.selected_count; i++) {
    ecs_entity_t target = ctx->effect.entities[i];
    if (target != 0) {
      deal_effect_damage(world, target, 1);
    }
  }
}
