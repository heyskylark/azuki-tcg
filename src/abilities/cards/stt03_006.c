#include "abilities/cards/stt03_006.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

bool stt03_006_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void stt03_006_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  (void)draw_cards_with_deckout_check(world, ctx->runtime.owner, 1, NULL);

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || gs->winner != -1) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
    return;
  }

  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  if (ecs_get_ordered_children(world, gs->zones[owner_num].hand).count == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
    return;
  }

  ctx->effect.min_required = 1;
  ctx->effect.max_allowed = 1;
  ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
}

bool stt03_006_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) == gs->zones[owner_num].hand;
}

void stt03_006_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    discard_card(world, ctx->effect.entities[0]);
  }
}
