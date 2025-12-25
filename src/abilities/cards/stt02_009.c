#include "abilities/cards/stt02_009.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

// STT02-009 "Aya": [On Play] You may return an entity with cost >= 2 in your Garden
// to your hand: Return up to 1 entity with cost <= 2 in your opponent's Garden to its owner's hand.

// Helper to check if an entity has cost >= 2 (valid cost target)
static bool is_valid_cost_target(ecs_world_t* world, ecs_entity_t entity) {
    // Must be an entity card
    if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
        return false;
    }

    // Must have IKZ cost >= 2
    const IKZCost* cost = ecs_get(world, entity, IKZCost);
    if (!cost) {
        return false;
    }

    return cost->ikz_cost >= 2;
}

// Helper to check if an entity has cost <= 2 (valid effect target)
static bool is_valid_effect_target(ecs_world_t* world, ecs_entity_t entity) {
    // Must be an entity card
    if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
        return false;
    }

    // Must have IKZ cost <= 2
    const IKZCost* cost = ecs_get(world, entity, IKZCost);
    if (!cost) {
        return false;
    }

    return cost->ikz_cost <= 2;
}

bool stt02_009_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);
    ecs_entity_t garden = gs->zones[player_num].garden;

    // First check if the just-played card itself is a valid cost target
    // (handles case where card was played to garden and is cost >= 2)
    // This is needed because ecs_get_ordered_children may not immediately
    // return the just-played card due to Flecs staging/deferred operations
    ecs_entity_t card_parent = ecs_get_target(world, card, EcsChildOf, 0);
    if (card_parent == garden && is_valid_cost_target(world, card)) {
        return true;
    }

    // Check other cards in the garden
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
    for (int i = 0; i < garden_cards.count; i++) {
        if (is_valid_cost_target(world, garden_cards.ids[i])) {
            return true;  // Found at least one valid cost target
        }
    }

    return false;  // No valid cost targets in player's garden
}

bool stt02_009_validate_cost_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target) {
    (void)card;

    if (target == 0) {
        return false;
    }

    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);

    // Target must be in owner's garden
    ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
    if (parent != gs->zones[player_num].garden) {
        return false;
    }

    // Target must have cost >= 2
    return is_valid_cost_target(world, target);
}

bool stt02_009_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target) {
    (void)card;

    if (target == 0) {
        return false;
    }

    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, owner);
    uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;

    // Target must be in opponent's garden
    ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
    if (parent != gs->zones[enemy_num].garden) {
        return false;
    }

    // Target must have cost <= 2
    return is_valid_effect_target(world, target);
}

void stt02_009_apply_costs(ecs_world_t* world, const AbilityContext* ctx) {
    // Return the cost target to owner's hand
    ecs_entity_t target = ctx->cost_targets[0];

    if (target == 0) {
        cli_render_logf("[STT02-009] No cost target to bounce");
        return;
    }

    return_card_to_hand(world, target);
    cli_render_logf("[STT02-009] Returned cost target to owner's hand");
}

void stt02_009_apply_effects(ecs_world_t* world, const AbilityContext* ctx) {
    // Return the effect target to its owner's hand (if any - this is an "up to 1" effect)
    ecs_entity_t target = ctx->effect_targets[0];

    if (target == 0) {
        cli_render_logf("[STT02-009] No effect target selected (skipped)");
        return;
    }

    return_card_to_hand(world, target);
    cli_render_logf("[STT02-009] Returned opponent's entity to owner's hand");
}
