#include "abilities/cards/stt04_003.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

bool stt04_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  const ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return parent == gs->zones[owner_num].garden || parent == gs->zones[owner_num].alley;
}

void stt04_003_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  deal_effect_damage(world, ctx->runtime.source_card, 1);
}
