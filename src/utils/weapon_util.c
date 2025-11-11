#include "utils/weapon_util.h"
#include "utils/card_utils.h"
#include "generated/card_defs.h"

int attach_weapon_from_hand(
  ecs_world_t *world,
  const AttachWeaponIntent *intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER, "AttachWeaponIntent is null");

  const BaseStats *weapon_base_stats = ecs_get(world, intent->weapon_card, BaseStats);
  ecs_assert(weapon_base_stats != NULL, ECS_INVALID_PARAMETER, "BaseStats component not found for card %d", intent->weapon_card);

  const CurStats *entity_cur_stats = ecs_get(world, intent->target_card, CurStats);
  ecs_assert(entity_cur_stats != NULL, ECS_INVALID_PARAMETER, "CurStats component not found for card %d", intent->target_card);

  ecs_set(world, intent->target_card, CurStats, {
    .cur_atk = entity_cur_stats->cur_atk + weapon_base_stats->attack,
    .cur_hp = entity_cur_stats->cur_hp,
  });

  ecs_add_pair(world, intent->weapon_card, EcsChildOf, intent->target_card);

  for (uint8_t i = 0; i < intent->ikz_card_count; ++i) {
    set_card_to_tapped(world, intent->ikz_cards[i]);
  }

  return 0;
}
