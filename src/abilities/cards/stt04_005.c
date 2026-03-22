#include "abilities/cards/stt04_005.h"

#include "abilities/cards/common/reveal_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "utils/card_utils.h"

static bool is_pyreskin_selection_card(ecs_world_t *world, ecs_entity_t card,
                                       const void *user_ctx) {
  (void)user_ctx;
  return has_subtype(world, card, ecs_id(TSubtype_Pyreskin));
}

bool stt04_005_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void stt04_005_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  azk_setup_reveal_top_cards_selection(world, ctx, 5, 1,
                                       is_pyreskin_selection_card, NULL);
}

bool stt04_005_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && is_pyreskin_selection_card(world, target, NULL);
}

void stt04_005_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  azk_move_picked_selection_cards_to_hand(world, ctx);
  azk_begin_bottom_deck_for_remaining_selection(ctx);
}
