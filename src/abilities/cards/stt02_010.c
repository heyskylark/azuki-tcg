#include "abilities/cards/stt02_010.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// STT02-010: "Garden only; whenever an entity is returned to its owner's hand,
// you may tap this card, then draw 1. (this ability is not affected by
// cooldown)"

bool stt02_010_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);

  // Card must be in garden (garden only ability)
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  if (parent != gs->zones[player_num].garden) {
    return false;
  }

  // Card must not be tapped (cost is tapping this card)
  if (is_card_tapped(world, card)) {
    return false;
  }

  // Note: Cooldown does NOT prevent activation (AIgnoresCooldown)
  // No cooldown check!

  // Must have at least one card in deck to draw
  ecs_entity_t deck = gs->zones[player_num].deck;
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  if (deck_cards.count == 0) {
    return false;
  }

  return true;
}

void stt02_010_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  // Tap this card
  set_card_to_tapped(world, ctx->source_card);
  cli_render_logf("[STT02-010] Tapped card as cost");
}

void stt02_010_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  // Draw 1 card
  if (draw_cards_with_deckout_check(world, ctx->owner, 1, NULL)) {
    cli_render_logf("[STT02-010] Drew 1 card");
  } else {
    cli_render_logf("[STT02-010] Could not draw (deck empty - deckout)");
  }
}
