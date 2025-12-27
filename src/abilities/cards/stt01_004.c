#include "abilities/cards/stt01_004.h"

#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// STT01-004: "On Play; You may discard a weapon card: look at the top 5 cards
// of your deck, reveal up to 1 weapon card and add it to your hand, then
// bottom deck the rest in any order"

// Validate if ability can be activated
// Returns true if player has at least one weapon card in hand
bool stt01_004_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t hand = gs->zones[player_num].hand;

  // Check if player has at least one weapon in hand
  int weapon_count = count_weapons_in_zone(world, hand);
  return weapon_count > 0;
}

// Validate cost target - must be a weapon card in owner's hand
bool stt01_004_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t hand = gs->zones[player_num].hand;

  // Target must be in owner's hand
  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  if (parent != hand) {
    return false;
  }

  // Target must be a weapon card
  return is_weapon_card(world, target);
}

// Apply cost: discard the selected weapon card
void stt01_004_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->cost_targets[0];

  if (target == 0) {
    cli_render_logf("[STT01-004] No cost target to discard");
    return;
  }

  discard_card(world, target);
  cli_render_logf("[STT01-004] Discarded weapon card as cost");
}

// Called after cost is paid: move top 5 cards from deck to selection zone
void stt01_004_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);

  // Look at top 5 cards
  ecs_entity_t cards[MAX_SELECTION_ZONE_SIZE];
  int count = look_at_top_n_cards(world, ctx->owner, 5, cards);

  if (count == 0) {
    cli_render_logf("[STT01-004] No cards in deck to look at");
    // Skip directly to done - no cards to process
    return;
  }

  // Store cards in selection context
  ctx->selection_count = count;
  for (int i = 0; i < count; i++) {
    ctx->selection_cards[i] = cards[i];
  }

  // Set up selection pick phase - "up to 1" weapon
  ctx->selection_pick_max = 1;
  ctx->selection_picked = 0;

  // Count how many weapons are in the selection
  int weapon_count = 0;
  for (int i = 0; i < count; i++) {
    if (is_weapon_card(world, cards[i])) {
      weapon_count++;
    }
  }

  if (weapon_count > 0) {
    ctx->phase = ABILITY_PHASE_SELECTION_PICK;
    cli_render_logf("[STT01-004] Looking at top %d cards, found %d weapon(s)",
                    count, weapon_count);
  } else {
    // No weapons to pick - go directly to bottom deck phase
    ctx->phase = ABILITY_PHASE_BOTTOM_DECK;
    cli_render_logf("[STT01-004] Looking at top %d cards, no weapons found - "
                    "bottom decking",
                    count);
  }

  ecs_singleton_modified(world, AbilityContext);
}

// Validate selection target - must be a weapon card
bool stt01_004_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  if (target == 0) {
    return false;
  }

  // Target must be a weapon card
  return is_weapon_card(world, target);
}

// Called after selection pick is complete: move picked weapon to hand
void stt01_004_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  // Move any picked weapons to hand
  for (int i = 0; i < ctx->selection_picked && i < MAX_ABILITY_SELECTION; i++) {
    ecs_entity_t picked = ctx->effect_targets[i];
    if (picked != 0) {
      move_selection_to_hand(world, picked);
      cli_render_logf("[STT01-004] Added weapon card to hand");
    }
  }

  // Count remaining cards to bottom deck
  int remaining = 0;
  for (int i = 0; i < ctx->selection_count; i++) {
    if (ctx->selection_cards[i] != 0) {
      remaining++;
    }
  }

  if (remaining > 0) {
    ctx->phase = ABILITY_PHASE_BOTTOM_DECK;
    cli_render_logf("[STT01-004] %d cards remaining to bottom deck", remaining);
  } else {
    cli_render_logf("[STT01-004] Ability complete");
  }

  ecs_singleton_modified(world, AbilityContext);
}
