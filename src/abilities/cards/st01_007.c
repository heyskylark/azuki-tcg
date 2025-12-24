#include "abilities/cards/st01_007.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"

// ST01-007 "Alley Guy": On Play; You may discard 1:Draw 1

bool st01_007_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    (void)card;  // unused

    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);

    // Need at least 1 card in hand to discard
    ecs_entity_t hand = gs->zones[player_num].hand;
    ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
    if (hand_cards.count < 1) {
        return false;
    }

    // Need at least 1 card in deck to draw
    ecs_entity_t deck = gs->zones[player_num].deck;
    ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
    if (deck_cards.count < 1) {
        return false;
    }

    return true;
}

bool st01_007_validate_cost_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target) {
    (void)card;  // unused

    if (target == 0) {
        return false;
    }

    // Target must be in the owner's hand
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);
    ecs_entity_t hand = gs->zones[player_num].hand;

    // Check if target is a child of the hand zone
    ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
    if (parent != hand) {
        return false;
    }

    return true;
}

void st01_007_apply_costs(ecs_world_t* world, const AbilityContext* ctx) {
    // Discard the selected card
    ecs_entity_t to_discard = ctx->cost_targets[0];
    discard_card(world, to_discard);
}

void st01_007_apply_effects(ecs_world_t* world, const AbilityContext* ctx) {
    // Draw 1 card
    ecs_entity_t owner = ctx->owner;
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);

    ecs_entity_t deck = gs->zones[player_num].deck;
    ecs_entity_t hand = gs->zones[player_num].hand;

    move_cards_to_zone(world, deck, hand, 1, NULL);
}
