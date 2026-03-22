#include "abilities/cards/azk01_027.h"

#include "abilities/core/ability_context.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

bool azk01_027_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t hand_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].hand);
  return hand_cards.count >= 1;
}

bool azk01_027_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, target, EcsChildOf, 0) !=
      gs->zones[owner_num].hand) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (!ctx) {
    return false;
  }

  for (uint8_t i = 0; i < ctx->cost.selected_count; ++i) {
    if (ctx->cost.entities[i] == target) {
      return false;
    }
  }

  return true;
}

void azk01_027_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < ctx->cost.selected_count; ++i) {
    ecs_entity_t to_discard = ctx->cost.entities[i];
    if (to_discard != 0) {
      discard_card(world, to_discard);
    }
  }

  cli_render_logf("[AZK01-027] Discarded %u card(s) as cost",
                  (unsigned)ctx->cost.selected_count);
}

void azk01_027_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->cost.selected_count == 0) {
    return;
  }

  draw_cards_with_deckout_check(world, ctx->runtime.owner,
                                ctx->cost.selected_count, NULL);
  cli_render_logf("[AZK01-027] Drew %u card(s)",
                  (unsigned)ctx->cost.selected_count);
}
