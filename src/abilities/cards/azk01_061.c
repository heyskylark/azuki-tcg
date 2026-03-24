#include "abilities/cards/azk01_061.h"

#include "components/abilities.h"
#include "utils/damage_util.h"

bool azk01_061_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)owner;

  const DamageTracker *tracker = azk_get_current_turn_damage_tracker(world, card);
  return tracker != NULL && tracker->tracked_source_count >= 3;
}

void azk01_061_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  deal_effect_damage(world, ctx->effect.entities[0], 3);
}
