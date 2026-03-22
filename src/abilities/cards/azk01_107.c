#include "abilities/cards/azk01_107.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

bool azk01_107_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_ordered_children(world, gs->zones[owner_num].hand).count > 0;
}

bool azk01_107_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) == gs->zones[owner_num].hand;
}

void azk01_107_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count > 0 && ctx->cost.entities[0] != 0) {
    discard_card(world, ctx->cost.entities[0]);
  }
}

void azk01_107_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  IKZToken *ikz_token = ecs_get_mut(world, ctx->runtime.owner, IKZToken);
  if (ikz_token != NULL && ikz_token->ikz_token != 0) {
    return;
  }

  const ecs_entity_t prefab = azk_prefab_from_id(CARD_DEF_IKZ_002);
  if (prefab == 0) {
    return;
  }

  const ecs_entity_t token = ecs_new_w_pair(world, EcsIsA, prefab);
  ecs_set_name(world, token, "IKZTokenCard");
  ecs_set(world, token, DamageTracker, {0});
  if (!ecs_get(world, token, CardConditionCountdown)) {
    ecs_set(world, token, CardConditionCountdown,
            {.frozen_duration = 0,
             .shocked_duration = 0,
             .effect_immune_duration = 0,
             .timed_tag_grant_count = 0});
  }
  ecs_set(world, ctx->runtime.owner, IKZToken,
          {.ikz_token = token, .expires_eot = true});
}
