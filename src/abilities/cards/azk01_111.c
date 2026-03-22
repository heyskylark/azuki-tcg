#include "abilities/cards/azk01_111.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"

static bool is_small_hand_entity(ecs_world_t *world, ecs_entity_t card,
                                 const void *user_ctx) {
  (void)user_ctx;

  if (!is_card_type(world, card, CARD_TYPE_ENTITY)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, card, IKZCost);
  return cost != NULL && cost->ikz_cost <= 2;
}

static bool owner_has_valid_hand_entity(ecs_world_t *world, ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t hand_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].hand);
  for (int32_t i = 0; i < hand_cards.count; ++i) {
    if (is_small_hand_entity(world, hand_cards.ids[i], NULL)) {
      return true;
    }
  }

  return false;
}

bool azk01_111_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].alley) {
    return false;
  }

  const uint8_t enemy_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  return ecs_get_ordered_children(world, gs->zones[enemy_num].garden).count > 0 ||
         owner_has_valid_hand_entity(world, owner);
}

bool azk01_111_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
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

void azk01_111_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  sacrifice_card(world, ctx->runtime.source_card);
}

void azk01_111_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t enemy_num =
      (get_player_number(world, ctx->runtime.owner) + 1) % MAX_PLAYERS_PER_MATCH;
  if (ecs_get_ordered_children(world, gs->zones[enemy_num].garden).count > 0) {
    ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
  } else {
    azk01_111_apply_effects(world, ctx);
  }
}

void azk01_111_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count > 0 && ctx->effect.entities[0] != 0) {
    deal_effect_damage(world, ctx->effect.entities[0], 2);
  }

  AbilityContext *ctx_mut = ecs_singleton_get_mut(world, AbilityContext);
  if (ctx_mut == NULL) {
    return;
  }

  const uint8_t selection_count = azk_move_matching_hand_cards_to_selection(
      world, ctx_mut, 1, is_small_hand_entity, NULL);
  if (selection_count == 0) {
    ctx_mut->runtime.phase = ABILITY_PHASE_NONE;
  }
}

bool azk01_111_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && is_small_hand_entity(world, target, NULL);
}

void azk01_111_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  azk_return_remaining_selection_cards_to_hand(world, ctx);
}
