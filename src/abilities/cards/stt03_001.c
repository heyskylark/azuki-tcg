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
  const STT03BobuState *existing =
      ecs_get(world, ctx->runtime.source_card, STT03BobuState);
  STT03BobuState *state =
      ecs_ensure(world, ctx->runtime.source_card, STT03BobuState);
  if (gs == NULL || state == NULL) {
    return;
  }

  if (existing == NULL) {
    *state = (STT03BobuState){0};
  }

  state->expires_turn = gs->turn_number + 2;
  ecs_modified(world, ctx->runtime.source_card, STT03BobuState);
}
