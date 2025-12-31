#include "abilities/cards/stt01_012.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

// STT01-012 "Lightning Shuriken": [When Attacking] Put the top card of your
// deck into your discard pile.

bool stt01_012_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  uint8_t player_num = get_player_number(world, owner);
  const GameState *gs = ecs_singleton_get(world, GameState);

  ecs_entity_t deck_zone = gs->zones[player_num].deck;
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck_zone);

  // Effect requires at least 1 card in deck to mill
  return deck_cards.count > 0;
}

void stt01_012_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  uint8_t player_num = get_player_number(world, ctx->owner);
  const GameState *gs = ecs_singleton_get(world, GameState);

  ecs_entity_t deck_zone = gs->zones[player_num].deck;
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck_zone);

  if (deck_cards.count == 0) {
    cli_render_logf("[STT01-012] No cards in deck to discard");
    return;
  }

  // Mill top card (top of deck = last element in array)
  ecs_entity_t top_card = deck_cards.ids[deck_cards.count - 1];
  discard_card(world, top_card);

  const CardId *discarded_id = ecs_get(world, top_card, CardId);
  if (discarded_id) {
    cli_render_logf("[STT01-012] Discarded %s from top of deck",
                    discarded_id->code);
  } else {
    cli_render_logf("[STT01-012] Discarded top card from deck");
  }
}
