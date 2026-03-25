#include "abilities/cards/stt04_008.h"

#include "components/components.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

bool stt04_008_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || owner == 0) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  const ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return gs->active_player_index == (int8_t)owner_num &&
         parent == gs->zones[owner_num].garden &&
         gs->last_combat.attacker == card && gs->last_combat.defender_was_garden_entity;
}

void stt04_008_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const TapState *tap = ecs_get(world, ctx->runtime.source_card, TapState);
  if (tap != NULL && (tap->tapped || tap->cooldown)) {
    ecs_set(world, ctx->runtime.source_card, TapState,
            {.tapped = false, .cooldown = false});
    azk_log_card_tap_state_changed(world, ctx->runtime.source_card,
                                   GLOG_TAP_UNTAPPED);
  }
}
