#include "abilities/cards/stt04_007.h"

#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool stt04_007_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  const ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  const DamageTracker *tracker = azk_get_current_turn_damage_tracker(world, card);
  return tracker != NULL && tracker->took_damage_this_turn &&
         (parent == gs->zones[owner_num].garden || parent == gs->zones[owner_num].alley);
}

void stt04_007_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  apply_attack_modifier(world, ctx->runtime.source_card, ctx->runtime.source_card,
                        1, true);
}
