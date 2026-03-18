#include "abilities/selection/ability_selection.h"

#include "abilities/core/ability_context.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"

bool azk_finish_selection_resolution(ecs_world_t *world, AbilityContext *ctx,
                                     const AbilityDef *def,
                                     AbilitySelectionCompletionMode mode) {
  if (!ctx) {
    return false;
  }

  if (def && def->on_selection_complete) {
    def->on_selection_complete(world, ctx);
    cli_render_logf("[Ability] Called on_selection_complete callback");
  }

  if (mode == AZK_SELECTION_COMPLETION_ALLOW_BOTTOM_DECK) {
    if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK &&
        ctx->runtime.phase != ABILITY_PHASE_NONE) {
      const uint8_t remaining =
          azk_begin_bottom_deck_for_remaining_selection(ctx);
      if (remaining == 0) {
        azk_clear_ability_context(world);
        return true;
      }
    }

    ecs_singleton_modified(world, AbilityContext);
    return true;
  }

  if (ctx->runtime.phase != ABILITY_PHASE_NONE) {
    azk_clear_ability_context(world);
    return true;
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_bottom_deck_selection_card(ecs_world_t *world, AbilityContext *ctx,
                                    int selection_index) {
  if (!ctx) {
    return false;
  }

  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid bottom deck index %d", selection_index);
    return false;
  }

  ecs_entity_t card = ctx->selection.cards[selection_index];
  if (card == 0) {
    cli_render_logf("[Ability] Selection slot %d already empty",
                    selection_index);
    return false;
  }

  move_selection_to_deck_bottom(world, ctx->runtime.owner, card);
  ctx->selection.cards[selection_index] = 0;

  cli_render_logf("[Ability] Bottom decked card from slot %d", selection_index);

  if (azk_count_remaining_selection_cards(ctx) == 0) {
    cli_render_logf("[Ability] All cards bottom decked, ability complete");
    azk_clear_ability_context(world);
    return true;
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_bottom_deck_all_selection_cards(ecs_world_t *world,
                                         AbilityContext *ctx) {
  if (!ctx) {
    return false;
  }

  for (uint8_t i = 0;
       i < ctx->selection.count && i < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_entity_t card = ctx->selection.cards[i];
    if (card == 0) {
      continue;
    }

    move_selection_to_deck_bottom(world, ctx->runtime.owner, card);
    ctx->selection.cards[i] = 0;
  }

  cli_render_logf(
      "[Ability] Bottom decked all remaining cards, ability complete");
  azk_clear_ability_context(world);
  return true;
}
