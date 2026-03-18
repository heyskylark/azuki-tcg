#include "abilities/cards/stt02_003.h"

#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// STT02-003: "[On Play] Look at the top 5 cards of your deck, reveal up to 1
// (Watercrafting) card and add it to your hand, then bottom deck the rest in
// any order"

// Validate if ability can be activated
// This ability has no cost, so it can always be activated
bool stt02_003_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;

  // No cost requirement, always valid to trigger
  return true;
}

// Called after the ability is accepted: move top 5 cards from deck to
// selection zone
void stt02_003_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  // Look at top 5 cards
  ecs_entity_t cards[MAX_SELECTION_ZONE_SIZE];
  int count = look_at_top_n_cards(world, ctx->runtime.owner, 5, cards);

  if (count == 0) {
    cli_render_logf("[STT02-003] No cards in deck to look at");
    // Skip directly to done - no cards to process
    return;
  }

  // Store cards in selection context
  ctx->selection.count = count;
  for (int i = 0; i < count; i++) {
    ctx->selection.cards[i] = cards[i];
  }

  // Set up selection pick phase - "up to 1" watercrafting
  ctx->selection.pick_max = 1;
  ctx->selection.picked_count = 0;

  // Count how many watercrafting cards are in the selection
  int watercrafting_count = 0;
  for (int i = 0; i < count; i++) {
    if (is_watercrafting_card(world, cards[i])) {
      watercrafting_count++;
    }
  }

  if (watercrafting_count > 0) {
    ctx->runtime.phase = ABILITY_PHASE_SELECTION_PICK;
    cli_render_logf(
        "[STT02-003] Looking at top %d cards, found %d Watercrafting card(s)",
        count, watercrafting_count);
  } else {
    // No watercrafting cards to pick - go directly to bottom deck phase
    ctx->runtime.phase = ABILITY_PHASE_BOTTOM_DECK;
    cli_render_logf("[STT02-003] Looking at top %d cards, no Watercrafting "
                    "cards found - bottom decking",
                    count);
  }

  ecs_singleton_modified(world, AbilityContext);
}

// Validate selection target - must be a watercrafting card
bool stt02_003_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  // Target must be a watercrafting card
  return is_watercrafting_card(world, target);
}

// Called after selection pick is complete: move picked watercrafting card to
// hand
void stt02_003_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  // Move any picked watercrafting cards to hand
  for (int i = 0;
       i < ctx->selection.picked_count && i < MAX_ABILITY_SELECTION; i++) {
    ecs_entity_t picked = ctx->selection.picked_cards[i];
    if (picked != 0) {
      move_selection_to_hand(world, picked);
      cli_render_logf("[STT02-003] Added Watercrafting card to hand");
    }
  }

  // Count remaining cards to bottom deck
  int remaining = 0;
  for (int i = 0; i < ctx->selection.count; i++) {
    if (ctx->selection.cards[i] != 0) {
      remaining++;
    }
  }

  if (remaining > 0) {
    ctx->runtime.phase = ABILITY_PHASE_BOTTOM_DECK;
    cli_render_logf("[STT02-003] %d cards remaining to bottom deck", remaining);
  } else {
    cli_render_logf("[STT02-003] Ability complete");
  }

  ecs_singleton_modified(world, AbilityContext);
}
