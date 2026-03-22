#include "abilities/cards/azk01_016.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

bool azk01_016_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entities_t hand_cards =
      ecs_get_ordered_children(world, gs->zones[player_num].hand);
  ecs_entities_t deck_cards =
      ecs_get_ordered_children(world, gs->zones[player_num].deck);

  int hand_after_spell = hand_cards.count;
  for (int32_t i = 0; i < hand_cards.count; i++) {
    if (hand_cards.ids[i] == card) {
      hand_after_spell--;
      break;
    }
  }

  if (deck_cards.count < 1) {
    return false;
  }

  int available_after_draw = hand_after_spell + (deck_cards.count >= 2 ? 2 : 1);
  return available_after_draw >= 2;
}

bool azk01_016_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  if (parent != gs->zones[player_num].hand) {
    return false;
  }

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  for (int i = 0; i < ctx->effect.selected_count; i++) {
    if (ctx->effect.entities[i] == target) {
      return false;
    }
  }

  return true;
}

void azk01_016_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  (void)draw_cards_with_deckout_check(world, ctx->runtime.owner, 2, NULL);
}

void azk01_016_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs->winner != -1) {
    return;
  }

  ctx->effect.min_required = 2;
  ctx->effect.max_allowed = 2;
  ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
}

void azk01_016_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  for (int i = 0; i < ctx->effect.selected_count; i++) {
    if (ctx->effect.entities[i] != 0) {
      discard_card(world, ctx->effect.entities[i]);
    }
  }
}
