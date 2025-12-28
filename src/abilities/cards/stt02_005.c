#include "abilities/cards/stt02_005.h"

#include "components/components.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// STT02-005: On Play; If you played 2 other entities this turn, draw 1

bool stt02_005_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    (void)card;

    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);

    // Total entities played this turn (including this card)
    uint8_t total = gs->entities_played_garden_this_turn[player_num] +
                    gs->entities_played_alley_this_turn[player_num];

    // Need 3+ total (this card + 2 others)
    return total >= 3;
}

void stt02_005_apply_effects(ecs_world_t* world, const AbilityContext* ctx) {
    // Draw 1 card (with deck-out check)
    draw_cards_with_deckout_check(world, ctx->owner, 1, NULL);
}
