#include "abilities/cards/stt02_015.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

// STT02-015 "Commune with Water": [Response] Return an entity with cost <= 3 in any Garden to its owner's hand

// Helper to check if an entity has cost <= 3
static bool is_valid_bounce_target(ecs_world_t* world, ecs_entity_t entity) {
    // Must be an entity card
    if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
        return false;
    }

    // Must have IKZ cost <= 3
    const IKZCost* cost = ecs_get(world, entity, IKZCost);
    if (!cost) {
        return false;
    }

    return cost->ikz_cost <= 3;
}

bool stt02_015_validate(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    (void)card;
    (void)owner;

    const GameState* gs = ecs_singleton_get(world, GameState);

    // Check both players' gardens for valid targets
    for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
        ecs_entity_t garden = gs->zones[p].garden;
        ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

        for (int i = 0; i < garden_cards.count; i++) {
            if (is_valid_bounce_target(world, garden_cards.ids[i])) {
                return true;  // Found at least one valid target
            }
        }
    }

    return false;  // No valid targets in any garden
}

bool stt02_015_validate_effect_target(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner, ecs_entity_t target) {
    (void)card;
    (void)owner;

    if (target == 0) {
        return false;
    }

    // Target must be in a garden zone
    const GameState* gs = ecs_singleton_get(world, GameState);
    ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

    bool in_garden = false;
    for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
        if (parent == gs->zones[p].garden) {
            in_garden = true;
            break;
        }
    }

    if (!in_garden) {
        return false;
    }

    // Target must have cost <= 3
    return is_valid_bounce_target(world, target);
}

void stt02_015_apply_effects(ecs_world_t* world, const AbilityContext* ctx) {
    // Return the target entity to its owner's hand
    ecs_entity_t target = ctx->effect_targets[0];

    if (target == 0) {
        cli_render_logf("[STT02-015] No target to bounce");
        return;
    }

    return_card_to_hand(world, target);
    cli_render_logf("[STT02-015] Returned entity to owner's hand");
}
