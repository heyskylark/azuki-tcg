#include "abilities/cards/stt02_007.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// STT02-007 "Benzai the Merchant": On Play; Draw 1

bool stt02_007_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    (void)world;
    (void)card;
    (void)owner;

    // Always valid - no cost to pay, and if deck is empty the player has already lost
    return true;
}

void stt02_007_apply_effects(ecs_world_t* world, const AbilityContext* ctx) {
    // Draw 1 card (with deck-out check)
    draw_cards_with_deckout_check(world, ctx->owner, 1, NULL);
}
