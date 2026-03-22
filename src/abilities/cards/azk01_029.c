#include "abilities/cards/azk01_029.h"

#include "abilities/core/ability_context.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

static uint8_t count_other_hand_cards(ecs_world_t *world, ecs_entity_t owner,
                                      ecs_entity_t excluded_card) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t hand_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].hand);

  uint8_t count = 0;
  for (int32_t i = 0; i < hand_cards.count; ++i) {
    if (hand_cards.ids[i] != excluded_card) {
      ++count;
    }
  }

  return count;
}

static bool is_in_any_garden_or_leader_zone(ecs_world_t *world,
                                            ecs_entity_t target) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    if (parent == gs->zones[p].garden || parent == gs->zones[p].leader) {
      return true;
    }
  }

  return false;
}

bool azk01_029_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  return count_other_hand_cards(world, owner, card) >= 2;
}

bool azk01_029_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
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

bool azk01_029_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_in_any_garden_or_leader_zone(world, target);
}

void azk01_029_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < ctx->cost.selected_count; ++i) {
    ecs_entity_t to_discard = ctx->cost.entities[i];
    if (to_discard != 0) {
      discard_card(world, to_discard);
    }
  }

  cli_render_logf("[AZK01-029] Discarded %u card(s) as cost",
                  (unsigned)ctx->cost.selected_count);
}

void azk01_029_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  apply_attack_modifier(world, target, ctx->runtime.source_card, -3, true);
  cli_render_logf("[AZK01-029] Reduced target attack by 3 until end of turn");
}
