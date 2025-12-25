#include "abilities/cards/stt01_005.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"
#include "utils/cli_rendering_util.h"

// STT01-005: "Main; Alley Only; You may sacrifice this card: Draw 3 cards and discard 2"

bool stt01_005_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);

    // Card must be in alley (checked by action validation, but double-check)
    ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
    if (parent != gs->zones[player_num].alley) {
        return false;
    }

    // Need at least 1 card in deck to attempt draw (deck-out check happens during draw)
    // Note: Player may choose to activate even knowing they'll deck-out
    ecs_entity_t deck = gs->zones[player_num].deck;
    ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
    if (deck_cards.count < 1) {
        return false;  // Can't draw anything
    }

    return true;
}

bool stt01_005_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target) {
    (void)card;  // unused

    if (target == 0) {
        return false;
    }

    // Target must be in owner's hand
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);
    ecs_entity_t hand = gs->zones[player_num].hand;

    ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
    return parent == hand;
}

void stt01_005_apply_costs(ecs_world_t* world, const AbilityContext* ctx) {
    // 1. Sacrifice this card (move to discard)
    discard_card(world, ctx->source_card);
    cli_render_logf("[STT01-005] Sacrificed card");

    // 2. Draw 3 cards (with deck-out check after each draw)
    // Note: If deck-out occurs, gs->winner is set and game transitions to END_MATCH
    bool success = draw_cards_with_deckout_check(world, ctx->owner, 3, NULL);
    if (success) {
        cli_render_logf("[STT01-005] Drew 3 cards");
    } else {
        cli_render_logf("[STT01-005] Deck-out occurred during draw");
    }
}

void stt01_005_apply_effects(ecs_world_t* world, const AbilityContext* ctx) {
    // Discard the 2 selected cards
    for (int i = 0; i < ctx->effect_filled; i++) {
        discard_card(world, ctx->effect_targets[i]);
    }
    cli_render_logf("[STT01-005] Discarded %d cards", ctx->effect_filled);
}
