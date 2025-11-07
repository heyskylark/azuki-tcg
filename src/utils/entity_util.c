#include "utils/entity_util.h"
#include "utils/card_utils.h"

void reset_entity_health(ecs_world_t *world, ecs_entity_t entity) {
  const BaseStats *base_stats = ecs_get(world, entity, BaseStats);
  ecs_assert(base_stats != NULL, ECS_INVALID_PARAMETER, "BaseStats component not found for entity %d", entity);

  const CurStats *cur_stats = ecs_get(world, entity, CurStats);
  ecs_assert(cur_stats != NULL, ECS_INVALID_PARAMETER, "CurStats component not found for entity %d", entity);

  ecs_set(world, entity, CurStats, { .cur_atk = cur_stats->cur_atk, .cur_hp = base_stats->health });
}

static void discard_weapon_card(ecs_world_t *world, ecs_entity_t entity, ecs_entity_t weapon_card) {
  const CurStats *entity_cur_stats = ecs_get(world, entity, CurStats);
  ecs_assert(entity_cur_stats != NULL, ECS_INVALID_PARAMETER, "CurStats component not found for entity %d", entity);

  const BaseStats *weapon_card_base_stats = ecs_get(world, weapon_card, BaseStats);
  ecs_assert(weapon_card_base_stats != NULL, ECS_INVALID_PARAMETER, "BaseStats component not found for weapon card %d", weapon_card);

  ecs_set(
    world,
    entity,
    CurStats,
    {
      .cur_atk = entity_cur_stats->cur_atk - weapon_card_base_stats->attack,
      .cur_hp = entity_cur_stats->cur_hp,
    }
  );

  discard_card(world, weapon_card);
}

void discard_equipped_weapon_cards(ecs_world_t *world, ecs_entity_t entity) {
  ecs_iter_t it = ecs_children(world, entity);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t child = it.entities[i];
      if (ecs_has_id(world, child, TWeapon)) {
        discard_weapon_card(world, entity, child);
      }
    }
  }
}