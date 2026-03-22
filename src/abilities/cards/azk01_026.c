#include "abilities/cards/azk01_026.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

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

bool azk01_026_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  if (parent != gs->zones[owner_num].garden &&
      parent != gs->zones[owner_num].alley) {
    return false;
  }

  ecs_entities_t hand_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].hand);
  return hand_cards.count >= 1;
}

bool azk01_026_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].hand;
}

bool azk01_026_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  return is_in_any_garden_or_leader_zone(world, target);
}

void azk01_026_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t to_discard = ctx->cost.entities[0];
  if (to_discard == 0) {
    return;
  }

  discard_card(world, to_discard);
  cli_render_logf("[AZK01-026] Discarded 1 card as cost");
}

void azk01_026_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect.entities[0];
  if (target == 0) {
    return;
  }

  apply_attack_modifier(world, target, ctx->runtime.source_card, -1, true);
  cli_render_logf("[AZK01-026] Reduced target attack by 1 until end of turn");
}
