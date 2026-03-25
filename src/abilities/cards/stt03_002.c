#include "abilities/cards/stt03_002.h"

#include "components/abilities.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

static int8_t current_gate_power(const AbilityContext *ctx, ecs_world_t *world) {
  if (ctx->scratch.kind != ABILITY_SCRATCH_GATE_PORTAL) {
    return 0;
  }

  const GatePoints *gp =
      ecs_get(world, ctx->scratch.data.gate_portal.portaled_card, GatePoints);
  return gp != NULL ? gp->gate_points : 0;
}

bool stt03_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

bool stt03_002_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (gs == NULL || ctx == NULL) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  const bool is_portaled_card =
      ctx->scratch.kind == ABILITY_SCRATCH_GATE_PORTAL &&
      target == ctx->scratch.data.gate_portal.portaled_card;
  if (ecs_get_target(world, target, EcsChildOf, 0) != gs->zones[owner_num].garden &&
      !is_portaled_card) {
    return false;
  }

  if (ecs_has(world, target, Defender)) {
    return false;
  }

  const BaseStats *base = ecs_get(world, target, BaseStats);
  return base != NULL && base->health <= current_gate_power(ctx, world);
}

void stt03_002_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  apply_timed_tag_grant(world, ctx->effect.entities[0], ecs_id(Defender),
                        TAG_GRANT_TICK_START_OF_TURN, 2);
}
