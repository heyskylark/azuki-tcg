#include "abilities/cards/azk01_005.h"

#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_005_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);

  for (int32_t i = 0; i < cards.count; i++) {
    if (!is_effect_immune(world, cards.ids[i])) {
      return true;
    }
  }

  return false;
}

bool azk01_005_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || is_effect_immune(world, target)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == gs->zones[opponent_num].garden;
}

void azk01_005_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    cli_render_logf("[AZK01-005] Skipped damage");
    return;
  }

  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  if (deal_effect_damage(world, target, 1)) {
    cli_render_logf("[AZK01-005] Dealt 1 damage to target");
  } else {
    cli_render_logf("[AZK01-005] Damage blocked by EffectImmune");
  }
}
