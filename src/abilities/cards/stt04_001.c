#include "abilities/cards/stt04_001.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

static bool is_friendly_garden_or_alley_entity(ecs_world_t *world,
                                               ecs_entity_t owner,
                                               ecs_entity_t target) {
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  const ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[owner_num].garden || parent == gs->zones[owner_num].alley;
}

bool stt04_001_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_ordered_children(world, gs->zones[owner_num].garden).count > 0 ||
         ecs_get_ordered_children(world, gs->zones[owner_num].alley).count > 0;
}

bool stt04_001_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  return is_friendly_garden_or_alley_entity(world, owner, target);
}

void stt04_001_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  deal_effect_damage(world, ctx->runtime.source_card, 1);
}

void stt04_001_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  deal_effect_damage(world, target, 1);
  const CurStats *damaged = ecs_get(world, target, CurStats);
  if (damaged != NULL && damaged->cur_hp > 0 &&
      is_friendly_garden_or_alley_entity(world, ctx->runtime.owner, target)) {
    const CurStats *before = ecs_get(world, target, CurStats);
    const int8_t attack_before = before != NULL ? before->cur_atk : 0;
    apply_attack_modifier(world, target, ctx->runtime.source_card, 1, true);
    const CurStats *after = ecs_get(world, target, CurStats);
    if (after != NULL && after->cur_atk > attack_before) {
      azk_record_leader_ability_outcome(world, ctx->runtime.owner);
    }
  }
}
