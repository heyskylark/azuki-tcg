#include "abilities/cards/stt04_002.h"

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

bool stt04_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

bool stt04_002_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, target, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }

  const DamageTracker *tracker = ecs_get(world, target, DamageTracker);
  return tracker != NULL && tracker->took_damage_this_turn;
}

void stt04_002_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  const int8_t gate_power = current_gate_power(ctx, world);
  if (gate_power > 0) {
    apply_attack_modifier(world, ctx->effect.entities[0], ctx->runtime.source_card,
                          gate_power, true);
  }
}
