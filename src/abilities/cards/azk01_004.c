#include "abilities/cards/azk01_004.h"

#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/status_util.h"

// AZK01-004 "Alley Thug": [When Attacking] This card gets +1 attack until the
// end of the turn.

bool azk01_004_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)owner;

  return card != 0 && ecs_has(world, card, CurStats);
}

void azk01_004_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t card = ctx->runtime.source_card;

  if (card == 0) {
    cli_render_logf("[AZK01-004] No attacking card to buff");
    return;
  }

  apply_attack_modifier(world, card, card, 1, true);
  cli_render_logf("[AZK01-004] This card gets +1 attack until end of turn");
}
