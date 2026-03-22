#include "abilities/cards/stt03_001.h"

#include "components/components.h"

bool stt03_001_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void stt03_001_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  STT03BobuState *state =
      ecs_get_mut(world, ctx->runtime.source_card, STT03BobuState);
  if (gs == NULL || state == NULL) {
    return;
  }

  state->expires_turn = gs->turn_number + 2;
  ecs_modified(world, ctx->runtime.source_card, STT03BobuState);
}
