#include "abilities/cards/azk01_124.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

static uint8_t gate_power_from_ctx(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx == NULL || ctx->scratch.kind != ABILITY_SCRATCH_GATE_PORTAL) {
    return 0;
  }

  const GatePoints *gp =
      ecs_get(world, ctx->scratch.data.gate_portal.portaled_card, GatePoints);
  return gp != NULL ? gp->gate_points : 0;
}

static bool is_valid_devotion_cost_target(ecs_world_t *world, ecs_entity_t owner,
                                          ecs_entity_t target,
                                          ecs_entity_t portaled_card,
                                          uint8_t max_cost) {
  if (target == 0 || target == portaled_card ||
      !is_card_type(world, target, CARD_TYPE_ENTITY) || is_card_tapped(world, target)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, target, IKZCost);
  if (cost == NULL || cost->ikz_cost > max_cost) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

bool azk01_124_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  if (max_cost == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (is_valid_devotion_cost_target(world, owner, garden_cards.ids[i],
                                      ctx->scratch.data.gate_portal.portaled_card,
                                      max_cost)) {
      return true;
    }
  }

  return false;
}

bool azk01_124_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return is_valid_devotion_cost_target(world, owner, target,
                                       ctx->scratch.data.gate_portal.portaled_card,
                                       gate_power_from_ctx(world, ctx));
}

bool azk01_124_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t enemy_num =
      (get_player_number(world, owner) + 1) % MAX_PLAYERS_PER_MATCH;
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[enemy_num].garden;
}

void azk01_124_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count == 0 || ctx->cost.entities[0] == 0) {
    return;
  }

  const CurStats *stats = ecs_get(world, ctx->cost.entities[0], CurStats);
  const int8_t damage = stats != NULL && stats->cur_hp > 0 ? stats->cur_hp : 0;

  AbilityContext *ctx_mut = ecs_singleton_get_mut(world, AbilityContext);
  ctx_mut->scratch = (AbilityScratchState){
      .kind = ABILITY_SCRATCH_SACRIFICE_VALUE,
      .data.sacrifice_value =
          {
              .damage = damage,
              .draw_after_effect = false,
          },
  };

  sacrifice_card(world, ctx->cost.entities[0]);
}

void azk01_124_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->scratch.kind != ABILITY_SCRATCH_SACRIFICE_VALUE ||
      ctx->scratch.data.sacrifice_value.damage <= 0 ||
      ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  deal_effect_damage(world, ctx->effect.entities[0],
                     ctx->scratch.data.sacrifice_value.damage);
}
